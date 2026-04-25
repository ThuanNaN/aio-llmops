# Monitoring Stack

A monitoring and logging stack for the routed LLM services using Prometheus, Grafana, Loki, and Promtail.

The monitoring stack now reads `VLLM_HOST`, `VLLM_PORT`, `TRTLLM_HOST`, `TRTLLM_PORT`, `BACKEND_HOST`, and `BACKEND_PORT` from `monitor/.env` before generating the Prometheus scrape config.

## Components

- **Prometheus**: Time-series database for metrics collection
- **Grafana**: Visualization dashboard for metrics and logs
- **Loki**: Log aggregation system
- **Promtail**: Docker log collector that forwards container logs to Loki

## Features

- **Metric Collection**: Collects metrics from all services
- **Dashboard Visualization**: Pre-configured dashboard for vLLM performance plus gateway metrics in Prometheus
- **Log Aggregation**: Centralized Docker logging with filtering and search
- **Alerting**: Configurable alerts for service health

## Architecture

The monitoring stack provides:

- Centralized metrics collection via Prometheus
- Log collection via Promtail and Loki
- Visualization via Grafana dashboards
- Cross-service observability

## Dashboards

- **vLLM Performance**: Comprehensive dashboard for vLLM metrics including:
  - Token throughput
  - Latency metrics (TTFT, E2E, per-token)
  - Cache utilization
  - Scheduler state
  - Request patterns

## Running the Stack

### Using Docker Compose

```bash
docker compose up -d
```

Services will be available at:

- Grafana: `http://localhost:3000` (default credentials: admin/admin)
- Prometheus: `http://localhost:9090`
- Loki: `http://localhost:3100`
- Promtail: `http://localhost:9080/targets`

## Configuration

### Prometheus

Prometheus is configured to scrape metrics from:

- Prometheus itself
- Loki
- vLLM API
- TensorRT-LLM API
- Backend API

If your topology changes, update `monitor/.env`. Prometheus renders its live scrape config from `prometheus/prometheus.yml` at container startup.

The template is stored in `prometheus/prometheus.yml`.

### Loki

Loki is configured for efficient log collection and storage with:

- Local filesystem storage
- 24-hour index periods
- Structured metadata support

The configuration is stored in `loki/config.yml`.

### Promtail

Promtail discovers Docker containers through the Docker socket and pushes parsed logs to Loki.

The configuration is stored in `promtail/config.yml`.

### Grafana

Grafana is pre-configured with:

- Prometheus data source
- Loki data source
- vLLM performance dashboard

## Using the Dashboards

1. Access Grafana at `http://localhost:3000`
2. Log in with default credentials (admin/admin)
3. Navigate to Dashboards -> Import dashboard -> Copy and paste the dashboard config (grafana.json)
4. Select the model from the dropdown to view its metrics
