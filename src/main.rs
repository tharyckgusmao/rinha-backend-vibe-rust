mod app;
mod config;
mod dataset;
mod domain;
mod fdpass;
mod index;
mod ivf;
mod vector;

use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    os::unix::io::FromRawFd,
    path::PathBuf,
    sync::Arc,
    thread,
};

use crate::{
    config::Config, ivf::IvfIndex, vector::Vectorizer,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config::from_env();
    let vectorizer_config = config.load_vectorizer_config()?;
    let ivf = IvfIndex::load(&config.dataset_dir)
        .expect("[api] failed to load IVF index");
    eprintln!("[api] index loaded");
    let state = Arc::new(State {
        ivf,
        vectorizer: Vectorizer::new(vectorizer_config),
    });

    let fd_socket_path = std::env::var("FD_SOCKET_PATH").ok().map(PathBuf::from);

    if let Some(ref sock_path) = fd_socket_path {
        let ready_state = Arc::clone(&state);
        let port = config.port;
        thread::spawn(move || ready_listener(port, &ready_state));

        let uds_fd = fdpass::bind_unix_dgram(sock_path)?;
        eprintln!("[api] fd-pass mode on {:?}", sock_path);

        loop {
            match fdpass::recv_fd(uds_fd) {
                Ok(client_fd) => {
                    let stream = unsafe { TcpStream::from_raw_fd(client_fd) };
                    let st = Arc::clone(&state);
                    thread::Builder::new()
                        .stack_size(64 * 1024)
                        .spawn(move || { let _ = serve_connection(stream, &st); })
                        .ok();
                }
                Err(_) => continue,
            }
        }
    } else {
        let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
        let listener = TcpListener::bind(addr)?;
        eprintln!("[api] listening on {}", listener.local_addr()?);

        for stream in listener.incoming() {
            let stream = stream?;
            let st = Arc::clone(&state);
            thread::Builder::new()
                .stack_size(64 * 1024)
                .spawn(move || { let _ = serve_connection(stream, &st); })
                .ok();
        }
    }

    Ok(())
}

struct State {
    ivf: IvfIndex,
    vectorizer: Vectorizer,
}

fn ready_listener(port: u16, state: &Arc<State>) {
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).unwrap();
    eprintln!("[api] health-check on {}", listener.local_addr().unwrap());
    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            let ready = state.ivf.is_ready();
            let _ = write_ready_response(&mut stream, ready);
        }
    }
}

fn write_ready_response(stream: &mut TcpStream, ready: bool) -> std::io::Result<()> {
    if ready {
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
    } else {
        stream.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
    }
}

const MAX_BUF: usize = 2048;

fn serve_connection(mut stream: TcpStream, state: &State) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    stream.set_read_timeout(Some(std::time::Duration::from_secs(10)))?;
    let mut buf = [0u8; MAX_BUF];
    let mut filled = 0;

    loop {
        let (method, content_length, header_end) = loop {
            if filled >= MAX_BUF {
                return Ok(());
            }
            let n = stream.read(&mut buf[filled..])?;
            if n == 0 {
                return Ok(());
            }
            filled += n;

            if let Some(parsed) = parse_request_head(&buf[..filled]) {
                break parsed;
            }
        };

        let body_start = header_end + 4;
        let request_end = body_start + content_length;

        while filled < request_end {
            if request_end > MAX_BUF {
                return Ok(());
            }
            let n = stream.read(&mut buf[filled..])?;
            if n == 0 {
                return Ok(());
            }
            filled += n;
        }

        match method {
            Method::GetReady => {
                if state.ivf.is_ready() {
                    stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n")?;
                } else {
                    stream.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n")?;
                }
            }
            Method::PostFraudScore => {
                let body = &buf[body_start..request_end];
                let response = match state.vectorizer.vectorize_json_bytes(body) {
                    Ok(query) => {
                        // Inline quantize - avoid separate pass
                        let mut quantized = [0i16; 14];
                        for i in 0..14 {
                            quantized[i] = (query[i] * 10000.0).round().clamp(-32768.0, 32767.0) as i16;
                        }
                        let votes = state.ivf.fraud_votes(&quantized);
                        fraud_response(votes)
                    }
                    Err(_) => b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n" as &[u8],
                };
                stream.write_all(response)?;
            }
            Method::Unknown => {
                stream.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")?;
                return Ok(());
            }
        }

        if filled > request_end {
            buf.copy_within(request_end..filled, 0);
            filled -= request_end;
        } else {
            filled = 0;
        }
    }
}

#[derive(Clone, Copy)]
enum Method {
    GetReady,
    PostFraudScore,
    Unknown,
}

fn parse_request_head(buf: &[u8]) -> Option<(Method, usize, usize)> {
    let header_end = find_header_end(buf)?;
    let method = if buf.starts_with(b"GET /ready") {
        Method::GetReady
    } else if buf.starts_with(b"POST /fraud-score") {
        Method::PostFraudScore
    } else {
        Method::Unknown
    };

    let content_length = content_length_from_headers(&buf[..header_end]);
    Some((method, content_length, header_end))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    memchr_find(buf)
}

/// Fast \r\n\r\n search using manual scan with early byte check
#[inline(always)]
fn memchr_find(buf: &[u8]) -> Option<usize> {
    let len = buf.len();
    if len < 4 { return None; }
    let mut i = 0;
    while i + 3 < len {
        if buf[i] == b'\r' && buf[i+1] == b'\n' && buf[i+2] == b'\r' && buf[i+3] == b'\n' {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn content_length_from_headers(headers: &[u8]) -> usize {
    let len = headers.len();
    let needle = b"content-length:";
    let nlen = needle.len();

    let mut i = 0;
    while i + nlen <= len {
        if headers[i..i + nlen].eq_ignore_ascii_case(needle) {
            let start = i + nlen;
            let mut j = start;
            while j < len && headers[j] == b' ' {
                j += 1;
            }
            let mut val = 0usize;
            while j < len && headers[j].is_ascii_digit() {
                val = val * 10 + (headers[j] - b'0') as usize;
                j += 1;
            }
            return val;
        }
        i += 1;
    }
    0
}

fn fraud_response(votes: usize) -> &'static [u8] {
    match votes {
        0 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
        1 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
        2 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
        3 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
        4 => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
        _ => b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
    }
}
