# Rinha de Backend 2026 – Rust

Detecção de fraude por busca vetorial com k-NN (k=5) sobre 3M de vetores quantizados em i16.

## Stack

- **Rust** puro (zero frameworks, zero async runtime)
- **LB custom** com fd-passing via SCM_RIGHTS (zero-copy proxy)
- Busca vetorial com partição coarse + poda por lower-bound + mmap
- HTTP/1.1 parser manual com keep-alive

## Arquitetura

```
cliente → LB:9999 ──[SCM_RIGHTS]──→ api-1 (serve HTTP direto no fd)
                  ──[SCM_RIGHTS]──→ api-2 (serve HTTP direto no fd)
```

O LB não faz proxy HTTP — ele aceita a conexão TCP e passa o file descriptor
diretamente para o worker via Unix DGRAM socket. O worker recebe o fd e serve
HTTP/1.1 com keep-alive sem nenhum intermediário.

Recursos: 0.10 CPU + 32MB (LB) | 0.45 CPU + 159MB (cada API) = **1 CPU + 350MB total**

## Build e execução local

```bash
# Gerar dataset binário
cargo run --release --bin build-dataset -- references/references.json.gz data/index

# Rodar API (modo TCP direto, sem fd-passing)
DATASET_DIR=data/index cargo run --release

# Testar
curl http://127.0.0.1:9999/ready
curl -X POST http://127.0.0.1:9999/fraud-score -d '{"id":"tx-1",...}'
```

## Docker

```bash
docker build --platform linux/amd64 -t rinha-rust .
```

## Submissão

Branch `submission` contém apenas:
- `docker-compose.yml`
- `info.json`
