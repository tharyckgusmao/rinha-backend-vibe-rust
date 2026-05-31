#!/bin/bash
set -e
cd "$(dirname "$0")/.."

# Payload
cat > /tmp/rinha-payload.json << 'EOF'
{"id":"tx-1","transaction":{"amount":120.0,"installments":1,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":60.0,"tx_count_24h":2,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-002","mcc":"9999","avg_amount":300.0},"terminal":{"is_online":true,"card_present":false,"km_from_home":13.7},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8}}
EOF

echo "=== Building Docker image ==="
docker build --platform linux/amd64 -t rinha-local .

echo ""
echo "=== Starting containers ==="
docker compose -f docker/benchmark/docker-compose.yml down -v 2>/dev/null || true
docker compose -f docker/benchmark/docker-compose.yml up -d

echo ""
echo "=== Waiting for ready ==="
for i in $(seq 1 120); do
  CODE=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:9999/ready 2>/dev/null || echo "000")
  if [ "$CODE" = "200" ]; then
    echo "Ready in ${i}s"
    break
  fi
  if [ $i -eq 120 ]; then
    echo "TIMEOUT"
    docker compose -f docker/benchmark/docker-compose.yml logs
    exit 1
  fi
  sleep 1
done

# Smoke test
echo ""
echo "=== Smoke test ==="
curl -s http://127.0.0.1:9999/fraud-score -d @/tmp/rinha-payload.json
echo ""

# Warmup
echo ""
echo "=== Warmup ==="
ab -k -l -n 500 -c 5 -p /tmp/rinha-payload.json -T application/json http://127.0.0.1:9999/fraud-score 2>&1 | grep -E "Requests per second|Non-2xx|Failed"

# Benchmarks
echo ""
echo "=== c=1 n=1000 ==="
ab -k -l -n 1000 -c 1 -p /tmp/rinha-payload.json -T application/json http://127.0.0.1:9999/fraud-score 2>&1 | grep -E "Requests per second|50%|95%|99%|100%|Non-2xx|Failed"

echo ""
echo "=== c=10 n=5000 ==="
ab -k -l -n 5000 -c 10 -p /tmp/rinha-payload.json -T application/json http://127.0.0.1:9999/fraud-score 2>&1 | grep -E "Requests per second|50%|95%|99%|100%|Non-2xx|Failed"

echo ""
echo "=== c=50 n=5000 ==="
ab -k -l -n 5000 -c 50 -p /tmp/rinha-payload.json -T application/json http://127.0.0.1:9999/fraud-score 2>&1 | grep -E "Requests per second|50%|95%|99%|100%|Non-2xx|Failed"

# Cleanup
echo ""
echo "=== Stopping ==="
docker compose -f docker/benchmark/docker-compose.yml down -v
echo "=== Done ==="
