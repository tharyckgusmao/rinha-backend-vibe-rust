//! lb — FD-passing load balancer for Rinha de Backend 2026.
//!
//! Accepts TCP on :9999, passes each client fd to an API worker via SCM_RIGHTS
//! over Unix DGRAM (round-robin). Zero HTTP parsing, zero data copy.
//!
//! Environment:
//!   UPSTREAMS  comma-separated Unix socket paths (default: /sockets/api1.sock,/sockets/api2.sock)

fn main() {
    #[cfg(target_os = "linux")]
    linux_main();

    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("[lb] fd-passing LB only runs on Linux. Use HAProxy for local dev.");
        std::process::exit(1);
    }
}

#[cfg(target_os = "linux")]
fn linux_main() {
    use std::os::unix::io::RawFd;

    let upstreams = parse_upstreams();
    if upstreams.is_empty() {
        eprintln!("[lb] no upstreams configured");
        std::process::exit(1);
    }

    let listen_fd = tcp_listen(9999);
    let uds_fd = unix_dgram_socket();

    // Large send buffer for burst tolerance
    unsafe {
        libc::setsockopt(
            uds_fd,
            libc::SOL_SOCKET,
            libc::SO_SNDBUF,
            &(16 * 1024 * 1024i32) as *const i32 as *const libc::c_void,
            4,
        );
    }

    eprintln!("[lb] listening on :9999, upstreams={upstreams:?}");

    let n_up = upstreams.len();
    let mut rr: usize = 0;

    loop {
        let client_fd = unsafe {
            libc::accept4(
                listen_fd,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                libc::SOCK_CLOEXEC,
            )
        };
        if client_fd < 0 {
            continue;
        }

        // TCP_NODELAY
        unsafe {
            libc::setsockopt(
                client_fd,
                libc::IPPROTO_TCP,
                libc::TCP_NODELAY,
                &1i32 as *const i32 as *const libc::c_void,
                4,
            );
        }

        let idx = rr % n_up;
        rr = rr.wrapping_add(1);

        // Try both upstreams before giving up
        let mut sent = false;
        for attempt in 0..n_up {
            let target = (idx + attempt) % n_up;
            if send_fd(uds_fd, &upstreams[target], client_fd).is_ok() {
                sent = true;
                break;
            }
        }
        if !sent {
            send_503(client_fd);
        }

        unsafe { libc::close(client_fd) };
    }

    fn tcp_listen(port: u16) -> RawFd {
        unsafe {
            let fd = libc::socket(
                libc::AF_INET,
                libc::SOCK_STREAM | libc::SOCK_CLOEXEC,
                0,
            );
            assert!(fd >= 0, "[lb] socket failed");

            let one: i32 = 1;
            libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR, &one as *const _ as *const libc::c_void, 4);
            libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEPORT, &one as *const _ as *const libc::c_void, 4);

            let mut addr: libc::sockaddr_in = std::mem::zeroed();
            addr.sin_family = libc::AF_INET as u16;
            addr.sin_port = port.to_be();
            addr.sin_addr.s_addr = 0;

            let ret = libc::bind(fd, &addr as *const _ as *const libc::sockaddr, std::mem::size_of::<libc::sockaddr_in>() as u32);
            assert!(ret == 0, "[lb] bind :9999 failed");

            let ret = libc::listen(fd, 8192);
            assert!(ret == 0, "[lb] listen failed");

            // TCP_DEFER_ACCEPT: kernel only wakes us when data arrives, not on SYN
            let defer: i32 = 1;
            libc::setsockopt(fd, libc::IPPROTO_TCP, libc::TCP_DEFER_ACCEPT, &defer as *const _ as *const libc::c_void, 4);

            fd
        }
    }

    fn unix_dgram_socket() -> RawFd {
        unsafe {
            let fd = libc::socket(libc::AF_UNIX, libc::SOCK_DGRAM, 0);
            assert!(fd >= 0, "[lb] unix socket failed");
            fd
        }
    }

    fn send_fd(uds_fd: RawFd, path: &str, client_fd: RawFd) -> Result<(), ()> {
        unsafe {
            let mut addr: libc::sockaddr_un = std::mem::zeroed();
            addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
            let path_bytes = path.as_bytes();
            if path_bytes.len() >= addr.sun_path.len() {
                return Err(());
            }
            std::ptr::copy_nonoverlapping(
                path_bytes.as_ptr(),
                addr.sun_path.as_mut_ptr() as *mut u8,
                path_bytes.len(),
            );

            let cmsg_space = libc::CMSG_SPACE(4) as usize;
            let mut cmsg_buf = [0u8; 64]; // CMSG_SPACE(4) is always < 64

            let dummy: [u8; 1] = [1];
            let mut iov = libc::iovec {
                iov_base: dummy.as_ptr() as *mut libc::c_void,
                iov_len: 1,
            };

            let mut msg: libc::msghdr = std::mem::zeroed();
            msg.msg_name = &addr as *const _ as *mut libc::c_void;
            msg.msg_namelen = std::mem::size_of::<libc::sockaddr_un>() as u32;
            msg.msg_iov = &mut iov;
            msg.msg_iovlen = 1;
            msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
            msg.msg_controllen = cmsg_space;

            let cmsg = libc::CMSG_FIRSTHDR(&msg);
            (*cmsg).cmsg_level = libc::SOL_SOCKET;
            (*cmsg).cmsg_type = libc::SCM_RIGHTS;
            (*cmsg).cmsg_len = libc::CMSG_LEN(4) as _;
            std::ptr::copy_nonoverlapping(
                &client_fd as *const i32,
                libc::CMSG_DATA(cmsg) as *mut i32,
                1,
            );

            let ret = libc::sendmsg(uds_fd, &msg, libc::MSG_NOSIGNAL);
            if ret < 0 { Err(()) } else { Ok(()) }
        }
    }

    fn send_503(fd: RawFd) {
        const RESP: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        unsafe { libc::write(fd, RESP.as_ptr() as *const libc::c_void, RESP.len()) };
    }
}

fn parse_upstreams() -> Vec<String> {
    let env = std::env::var("UPSTREAMS")
        .unwrap_or_else(|_| "/sockets/api1.sock,/sockets/api2.sock".into());
    env.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}
