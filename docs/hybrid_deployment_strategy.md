# Hybrid Deployment Strategy: Open Core + SaaS

Below is a detailed blueprint for evolving Constraint-Lattice into a hybrid offering in which the constraint-enforcement core stays free and open source, while premium capabilities are delivered through a paid, cloud-hosted service.

---

## Architectural Split

1. **Core package (open)**  
   • Contains the constraint-enforcement engine, YAML loader, token-stream monitor, and thin SDK.  
   • Published to PyPI (`pip install constraint-lattice`).  
   • Provides a CLI entry-point via `console_scripts` in `pyproject.toml`.  
   • Minimal dependency footprint and fully annotated public APIs.

2. **SaaS microservice layer (closed)**  
   • Wraps the core engine and serves REST / gRPC endpoints.  
   • Endpoints: model registration, authenticated constraint application (streaming), audit history, constraint templates, model health.  
   • Implemented with FastAPI or Quart (async) + WebSocket / SSE for real-time streams.

---

## Persistent Storage

* **PostgreSQL** – tenant/org metadata, API usage logs, constraint traces.
* **Object store (S3 / GCS)** – JSONL audit logs (token-level), partitioned by `tenant_id` and `session_id`.

---

## Authentication & Multi-Tenancy

* OAuth2 (JWT) with API keys; bootstrap via Auth0 or Clerk.dev.  
* Middleware injects and validates `tenant_id`; every request execution is scoped accordingly.

---

## Containerization & Deployment

* Docker images for **core** and **service** layers.  
* Helm chart for Kubernetes – env-specific configuration (DB URL, Redis, rate limits, key lifetimes).  
* Optional Terraform / Pulumi modules for one-click infra on AWS / GCP / Azure.

---

## Streaming & Performance

* WebSocket or HTTP/2 SSE built on asyncio-enabled server.  
* GPU acceleration via autoscaled vLLM workers; models pre-warmed per premium tier.  
* LRU constraint-cache layer keeps per-request latency flat as constraint lists grow.

---

## Billing & Usage Metering

* Stripe integration with tiered plans – limits on monthly requests, concurrent streams, token caps, audit retention.  
* Middleware metering increments counters in Postgres; nightly roll-ups to analytics.

---

## Paid Dashboard

* Built with Next.js or Streamlit, behind JWT-protected session.  
* Visualises raw vs moderated outputs, violation summaries, token-diff views.  
* Only accessible to paying tenants via Stripe Customer Portal SSO.

---

## API Gateway & QoS

* Rate-limit and back-pressure via Traefik / Kong / AWS API Gateway, backed by Redis counters.

---

## CI/CD & Repo Layout

* GitHub Actions runs lint, unit tests, regression suite, Docker builds; tags `dev`, `staging`, `prod`.  
* Monorepo structure:  
  ```text
  /core        # open-source engine
  /saas        # closed APIs + billing
  /dashboard   # frontend
  /infra       # IaC (Helm, Terraform)
  ```  
* Licensed modules imported via extras (`pip install constraint-lattice[saas]`).

---

## Versioning & Community

* Semantic Versioning with CHANGELOG and migration guides.  
* CONTRIBUTING.md, issue templates, good-first-issue labels.  
* Plugin registry for third-party constraints (e.g., `GDPRCompliance`).

---

## Observability

* Structured logging with correlation IDs.  
* OpenTelemetry tracing; Prometheus metrics per constraint, latency, throughput.  
* Grafana dashboards; alerts on SLA breaches, queue growth, DB IOPS.

---

## Putting It All Together

---

*Last updated: 2025-06-29*

By separating an open-core engine from a containerised, multi-tenant SaaS wrapper—with robust auth, billing, observability, and IaC—you unlock both community adoption and a sustainable revenue stream while guaranteeing secure, performant, and policy-compliant text generation at scale.
