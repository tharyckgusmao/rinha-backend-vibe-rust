//! Receive TCP file descriptors via SCM_RIGHTS on a Unix DGRAM socket.

use std::io;
use std::os::unix::io::RawFd;
use std::path::Path;

/// Create and bind a Unix DGRAM socket at the given path.
pub fn bind_unix_dgram(path: &Path) -> io::Result<RawFd> {
    let _ = std::fs::remove_file(path);

    unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_DGRAM, 0);
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }

        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_RCVBUF,
            &(16 * 1024 * 1024i32) as *const i32 as *const libc::c_void,
            4,
        );

        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
        let path_bytes = path.as_os_str().as_encoded_bytes();
        if path_bytes.len() >= addr.sun_path.len() {
            libc::close(fd);
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "path too long"));
        }
        std::ptr::copy_nonoverlapping(
            path_bytes.as_ptr(),
            addr.sun_path.as_mut_ptr() as *mut u8,
            path_bytes.len(),
        );

        if libc::bind(
            fd,
            &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as u32,
        ) < 0
        {
            libc::close(fd);
            return Err(io::Error::last_os_error());
        }

        Ok(fd)
    }
}

/// Blocking receive of one fd from a Unix DGRAM socket via SCM_RIGHTS.
pub fn recv_fd(uds_fd: RawFd) -> io::Result<RawFd> {
    unsafe {
        let cmsg_space = libc::CMSG_SPACE(std::mem::size_of::<i32>() as u32) as usize;
        let mut cmsg_buf = vec![0u8; cmsg_space];

        let mut dummy = [0u8; 1];
        let mut iov = libc::iovec {
            iov_base: dummy.as_mut_ptr() as *mut libc::c_void,
            iov_len: 1,
        };

        let mut msg: libc::msghdr = std::mem::zeroed();
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
        msg.msg_controllen = cmsg_space as _;


        let ret = libc::recvmsg(uds_fd, &mut msg, 0);
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }

        let cmsg = libc::CMSG_FIRSTHDR(&msg);
        if cmsg.is_null()
            || (*cmsg).cmsg_level != libc::SOL_SOCKET
            || (*cmsg).cmsg_type != libc::SCM_RIGHTS
        {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "no SCM_RIGHTS"));
        }

        let mut fd: i32 = 0;
        std::ptr::copy_nonoverlapping(libc::CMSG_DATA(cmsg) as *const i32, &mut fd, 1);
        Ok(fd)
    }
}
