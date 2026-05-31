# Plano de Otimização - Busca Vetorial

## Status Atual

- p99 local: ~58ms (modo TCP direto, sem LB)
- Top 5 do ranking: 0.36-0.44ms (100x mais rápido)
- Gargalo: busca vetorial brute-force na partição

## Análise do Gap

Nosso p99 de 58ms vem da busca em partições grandes. Com 3M vetores e
particionamento coarse, uma partição pode ter centenas de milhares de vetores.
A busca faz L2 distance em cada um.

Os top 5 (p99 < 0.5ms) provavelmente usam:
1. **Índice ANN agressivo** (IVF com muitas células, ou HNSW)
2. **SIMD** para distância euclidiana (AVX2/SSE4)
3. **Layout cache-friendly** (vetores contíguos, alinhados)
4. **Pré-computação** de centróides/bounds no build

## Plano de Ataque (prioridade)

### 1. Mais partições (IVF com nlist grande)
- Atual: partição coarse simples
- Meta: IVF com 1000-4000 células → busca em ~5-20 células × ~750-3000 vetores
- Impacto esperado: 10-50x redução no número de comparações

### 2. SIMD para L2 distance
- 14 dimensões × i16 = 28 bytes → cabe em 1 registrador AVX2 (32 bytes)
- Usar `std::arch` com `_mm256_sub_epi16` + `_mm256_madd_epi16`
- Impacto esperado: 4-8x speedup na comparação individual

### 3. Quantização mais agressiva
- Atual: i16 (2 bytes/dim) → 28 bytes/vetor
- Alternativa: u8 (1 byte/dim) → 14 bytes/vetor (melhor cache)
- Trade-off: pode afetar precisão da detecção

### 4. Pré-sort por centróide
- No build, calcular centróides das células IVF
- No runtime, calcular distância query→centróide, buscar apenas top-N células
- Reduz drasticamente o espaço de busca

## Benchmark de Referência

Para atingir p99 < 1ms com 3M vetores:
- Precisa buscar em < 10.000 vetores por query
- Com SIMD: ~10.000 comparações i16×14 ≈ 0.1ms
- Com IVF(1000 células, probe=5): 5 × 3000 = 15.000 comparações → ~0.15ms

## Resultado com HAProxy (baseline)

```
ab -k -l -n 2000 -c 10: p99=78ms
```

## Resultado com Raw HTTP (sem LB)

```
ab -k -l -n 5000 -c 10: p99=58ms
ab -k -l -n 5000 -c 50: p99=82ms
```

## Próximo passo

Implementar IVF com nlist=1024 e nprobe=5-10. Isso deve reduzir o p99 de 58ms
para ~1-5ms. Depois SIMD para chegar em sub-ms.
