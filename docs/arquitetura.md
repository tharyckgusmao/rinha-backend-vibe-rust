# Arquitetura e Algoritmo — Rinha de Backend 2026

## Visão Geral

```
                         ┌─────────────────────────────────────────────┐
                         │            Docker Network (bridge)           │
                         │                                             │
  k6 ──TCP:9999──►  ┌───┴───┐   SCM_RIGHTS    ┌─────────────────┐    │
                     │  LB   │────────────────►│     API-1       │    │
                     │0.10CPU│   Unix DGRAM     │   0.45 CPU      │    │
                     │ 32MB  │──┐              │   159 MB        │    │
                     └───┬───┘  │              └─────────────────┘    │
                         │      │               ┌─────────────────┐    │
                         │      └──────────────►│     API-2       │    │
                         │         SCM_RIGHTS   │   0.45 CPU      │    │
                         │                      │   159 MB        │    │
                         │                      └─────────────────┘    │
                         └─────────────────────────────────────────────┘
                                    Total: 1 CPU + 350 MB
```

## Load Balancer — FD-Passing (zero-copy)

### O que é FD-Passing?

Em vez de fazer proxy HTTP (ler request → reenviar → ler response → reenviar),
o LB passa o **file descriptor do socket TCP** diretamente para o worker via
`sendmsg()` com `SCM_RIGHTS`. O kernel duplica o fd no processo destino.

```
┌──────────┐                    ┌──────────┐                    ┌──────────┐
│  Cliente │──TCP connect──────►│    LB    │                    │  Worker  │
│          │                    │          │                    │          │
│          │                    │ accept() │                    │          │
│          │                    │    │     │                    │          │
│          │                    │    ▼     │   sendmsg(fd)      │          │
│          │                    │ sendmsg()│───SCM_RIGHTS──────►│ recvmsg()│
│          │                    │    │     │   Unix DGRAM       │    │     │
│          │                    │ close(fd)│                    │    ▼     │
│          │                    │          │                    │ read(fd) │
│          │◄───────────────────┼──────────┼────────────────────│write(fd) │
│          │   HTTP response    │          │  (direto, sem LB)  │          │
└──────────┘                    └──────────┘                    └──────────┘
```

### Vantagens vs. Proxy HTTP tradicional

| Aspecto | nginx/HAProxy | FD-Passing |
|---------|---------------|------------|
| Cópias de dados | 4 (client→LB, LB→backend, backend→LB, LB→client) | 0 |
| Syscalls por request | ~8 (read, write × 4) | 2 (accept + sendmsg) |
| Parsing HTTP no LB | Sim | Não |
| Latência adicionada | 50-200μs | ~5μs |
| CPU no LB | Proporcional ao throughput | Quase zero |

### Código do LB (simplificado)

```rust
loop {
    let client_fd = accept4(listen_fd);     // aceita conexão TCP
    set_tcp_nodelay(client_fd);             // desabilita Nagle
    sendmsg(uds_fd, path, client_fd);       // passa fd via SCM_RIGHTS
    close(client_fd);                       // LB fecha sua cópia
}
```

O worker recebe o fd e serve HTTP/1.1 com keep-alive diretamente nele.

---

## Busca Vetorial — IVF (Inverted File Index)

### Problema

Dado um vetor query de 14 dimensões, encontrar os 5 vizinhos mais próximos
(k-NN, k=5) entre 3.000.000 de vetores de referência usando distância
euclidiana (L2).

### Brute-force vs. IVF

```
Brute-force:  query → compara com 3.000.000 vetores → top-5
              Complexidade: O(3M × 14) ≈ 42M operações
              Latência: ~50-80ms

IVF:          query → compara com 4.096 centróides → top-3 células
                    → compara com ~2.200 vetores → top-5
              Complexidade: O(4096×14 + 2200×14) ≈ 88K operações
              Latência: <1ms
```

**Redução: ~480x menos comparações.**

### Algoritmo IVF — Build (offline, no Docker build)

```
┌─────────────────────────────────────────────────────────────┐
│                    BUILD (offline)                            │
│                                                              │
│  1. Inicializar 4096 centróides (amostragem uniforme)        │
│                                                              │
│  2. K-Means (5 iterações):                                   │
│     ┌──────────────────────────────────────────┐             │
│     │  Para cada vetor v dos 3M:               │             │
│     │    assignment[v] = centróide mais próximo │             │
│     │                                          │             │
│     │  Para cada centróide c:                  │             │
│     │    c = média dos vetores atribuídos a c  │             │
│     └──────────────────────────────────────────┘             │
│                                                              │
│  3. Construir listas invertidas:                             │
│     cell[0] = [vec_12, vec_847, vec_2301, ...]              │
│     cell[1] = [vec_3, vec_99, vec_5042, ...]                │
│     ...                                                      │
│     cell[4095] = [vec_7, vec_444, ...]                       │
│                                                              │
│  4. Salvar em disco:                                         │
│     - ivf-centroids.bin (128KB)                              │
│     - ivf-vectors.bin (96MB, agrupados por célula)           │
│     - ivf-labels.bin (3MB)                                   │
│     - ivf-offsets.bin (16KB, prefix-sum das células)         │
└─────────────────────────────────────────────────────────────┘
```

### Algoritmo IVF — Query (runtime, <1ms)

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY (runtime)                            │
│                                                              │
│  Input: query vector q[14] (normalizado, quantizado i16)     │
│                                                              │
│  ┌─────────────────────────────────────────────┐             │
│  │ PASSO 1: Encontrar células mais próximas    │             │
│  │                                             │             │
│  │   Para cada centróide c[i] (i=0..4095):     │             │
│  │     dist[i] = L2(q, c[i])                  │             │
│  │                                             │             │
│  │   Selecionar top-3 (NPROBE=3)              │             │
│  │   → probe_cells = [cell_A, cell_B, cell_C] │             │
│  └─────────────────────────────────────────────┘             │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────┐             │
│  │ PASSO 2: Busca exata nas células selecionadas│            │
│  │                                             │             │
│  │   top_k = MinHeap(k=5)                     │             │
│  │                                             │             │
│  │   Para cada célula em probe_cells:          │             │
│  │     Para cada vetor v na célula:            │             │
│  │       d = L2(q, v)                         │             │
│  │       top_k.push(d, v.label)               │             │
│  └─────────────────────────────────────────────┘             │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────┐             │
│  │ PASSO 3: Decisão                            │             │
│  │                                             │             │
│  │   fraud_count = count(top_k where label=fraud)│           │
│  │   fraud_score = fraud_count / 5             │             │
│  │   approved = fraud_score < 0.6              │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  Output: { approved: bool, fraud_score: f64 }                │
└─────────────────────────────────────────────────────────────┘
```

### Distância L2 com SIMD

Os vetores são armazenados com **padding para 16 dimensões** (14 reais + 2 zeros).
Isso permite processar em dois blocos de 8×i16, alinhados para SSE2:

```
Vetor: [d0 d1 d2 d3 d4 d5 d6 d7 | d8 d9 d10 d11 d12 d13  0  0]
        ├────── 128 bits ────────┤  ├────── 128 bits ────────┤
              SSE2 op 1                    SSE2 op 2

L2²(a, b) = Σ(a[i] - b[i])²

Com SSE2 (8 × i16 por instrução):
  - 2 loads (a, b)
  - 1 subtract (a - b)
  - 1 multiply-add (diff² acumulado)
  × 2 blocos = 4 instruções no total
```

O compilador Rust auto-vectoriza o loop unrolled para SSE2/AVX2 com
`-C target-cpu=x86-64-v2`.

### Parâmetros do IVF

| Parâmetro | Valor | Efeito |
|-----------|-------|--------|
| NLIST | 4096 | Número de células (centróides) |
| NPROBE | 3 | Células buscadas por query |
| DIMS_PADDED | 16 | Dimensões com padding SIMD |
| KMEANS_ITERS | 5 | Iterações do k-means offline |
| Vetores/célula (avg) | ~732 | Tamanho médio de cada célula |
| Vetores buscados/query | ~2200 | NPROBE × avg_cell_size |

### Trade-off: Precisão vs. Velocidade

IVF é uma busca **aproximada** (ANN). Se o vizinho mais próximo real está numa
célula que não foi probeada, ele é perdido. Com NPROBE=3:

- A maioria das queries encontra os 5 vizinhos corretos
- Algumas podem ter 1-2 vizinhos diferentes do brute-force
- Isso pode causar FP/FN na detecção de fraude

O scoring da Rinha penaliza erros de detecção, mas o ganho de latência
(78ms → <1ms = +2000 pontos no score_p99) compensa amplamente os poucos
erros adicionais.

---

## HTTP Server — Parser Manual

Sem framework (axum, hyper, actix). Parser HTTP/1.1 manual otimizado:

```
┌─────────────────────────────────────────────────────────┐
│  serve_connection(fd)                                    │
│                                                          │
│  loop {                                                  │
│    read() até encontrar \r\n\r\n (fim dos headers)      │
│                                                          │
│    route = match buf[0..]:                               │
│      "GET /ready"        → respond 200                  │
│      "POST /fraud-score" → parse body, search, respond  │
│      _                   → respond 404, close           │
│                                                          │
│    content_length = scan headers (case-insensitive)      │
│    read() remaining body bytes                           │
│                                                          │
│    vectorize(body) → query[14]                          │
│    quantize(query) → query_i16[14]                      │
│    ivf.fraud_votes(query_i16) → 0..5                    │
│                                                          │
│    write() pre-computed response bytes                   │
│    shift buffer, continue (keep-alive)                   │
│  }                                                       │
└─────────────────────────────────────────────────────────┘
```

### Respostas pré-computadas

As 6 respostas possíveis (0-5 votos de fraude) são constantes `&'static [u8]`
com headers HTTP incluídos. Zero allocation no hot path.

---

## Pipeline Completo — Request Flow

```
 Cliente                LB              Worker
    │                   │                  │
    │──TCP SYN────────►│                  │
    │◄─TCP SYN+ACK────│                  │
    │──TCP ACK────────►│                  │
    │                   │                  │
    │                   │ accept4()        │
    │                   │ sendmsg(fd)─────►│ recvmsg()
    │                   │ close(fd)        │
    │                   │                  │
    │──HTTP POST───────────────────────────►│ read()
    │                                      │ parse headers
    │                                      │ read body
    │                                      │ vectorize (manual JSON parse)
    │                                      │ quantize f32→i16
    │                                      │ IVF: closest_cells (4096 L2)
    │                                      │ IVF: scan cells (~2200 L2)
    │                                      │ count fraud in top-5
    │◄─HTTP 200 + JSON─────────────────────│ write()
    │                                      │
    │──HTTP POST (keep-alive)──────────────►│ (loop)
    │◄─HTTP 200────────────────────────────│
    │                                      │
```

---

## Memória

| Componente | Tamanho | Descrição |
|------------|---------|-----------|
| ivf-vectors.bin | 96 MB | 3M × 16 dims × 2 bytes (mmap) |
| ivf-labels.bin | 3 MB | 3M × 1 byte (mmap) |
| ivf-centroids.bin | 128 KB | 4096 × 16 × 2 bytes |
| ivf-offsets.bin | 16 KB | 4097 × 4 bytes |
| Binário rinha | ~460 KB | Código compilado |
| Binário lb | ~300 KB | Código compilado |
| **Total por worker** | **~100 MB** | Cabe em 159 MB |

O `mmap` permite que o OS gerencie a memória — páginas não acessadas
não ocupam RAM física.

---

## Resultados

### Local (macOS, sem Docker, sem LB)

```
c=1:   p99 < 1ms,  14,890 req/s
c=10:  p99 < 1ms,  49,800 req/s
c=50:  p99 = 2ms,  53,527 req/s
```

---

## Referências

- [IVF (Inverted File Index)](https://www.pinecone.io/learn/series/faiss/inverted-file-index/)
- [SCM_RIGHTS fd-passing](https://man7.org/linux/man-pages/man7/unix.7.html)
- [SSE2 intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [k-NN search](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
