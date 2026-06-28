# Technical Requirements Document (TRD)
# Indian Climate Digital Twin (ICDT) — Deep Research Platform
## Prepared for the Bharatiya Antariksh Hackathon (BAH) 2026 — Challenge 05

**Version:** 3.0  
**Date:** 2026-06-28  
**Status:** Hackathon Architecture Specification  
**Team Name:** Team AIDT-India  
**Sponsor:** Indian Space Research Organisation (ISRO) & Hack2skill  
**Target Event:** Bharatiya Antariksh Hackathon (BAH) 2026  
**Problem Statement:** Challenge 05 — AI-Powered Digital Twin of India's Climate

---

## 1. Technical Overview

The ICDT platform is a distributed, cloud-native system built on a **multi-agent deep research architecture** that combines high-performance computing (HPC) for physics-based climate simulation with AI/ML for predictive analytics and autonomous research generation. The system follows a **hybrid cloud-HPC** model: operational real-time services run on Kubernetes (public cloud), while computationally intensive numerical weather prediction (NWP) and climate modeling execute on dedicated HPC clusters (on-premises and national supercomputing facilities).

### 1.1 Architecture Principles

1. **Modularity:** Each subsystem (Data Fabric, Simulation Engine, Deep Research Agent, Agricultural DSS, Urban Dashboard) is independently deployable and scalable
2. **Open Standards:** OGC-compliant geospatial services, FAIR data principles, WIS 2.0 (WMO Information System)
3. **AI-First:** Physics-informed neural networks (PINNs) augment traditional NWP; deep reinforcement learning optimizes resource allocation
4. **Edge-Cloud Continuum:** Heavy computation in cloud/HPC; lightweight inference at edge nodes (state capitals, Krishi Vigyan Kendras)
5. **Zero-Trust Security:** Every service authenticated; every data access logged; every model output traceable

---

## 2. System Architecture

### 2.1 High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE LAYER                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │ Web Dashboard│  │ Mobile App   │  │ API Gateway  │  │ JupyterHub      │ │
│  │ (React/Next) │  │ (Flutter)    │  │ (Kong/AWS)   │  │ (Research Env)  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Deep Research Agent (DRA)                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   Master    │  │   Planner   │  │  Researcher │  │   Writer    │  │   │
│  │  │  (Orchestrator)│  │   (DAG Gen) │  │  (Parallel) │  │  (Synthesis)│  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │   │
│  │  │   Searcher  │  │   Refiner   │  │  Clarifier  │                   │   │
│  │  │ (Web + API) │  │ (Query Opt) │  │  (HITL)     │                   │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Workflow Engine (LangGraph / Temporal.io)               │   │
│  │         DAG execution, state persistence, checkpointing             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION SERVICES LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Climate Data │  │  Digital Twin│  │ Agricultural │  │   Urban      │      │
│  │   Fabric     │  │ Sim Engine   │  │     DSS      │  │  Resilience  │      │
│  │  (Ingestion) │  │  (Forecast)  │  │  (Advisory)  │  │  (Flood/Heat)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                        │
│  │ Ganges Basin │  │  Report Gen  │  │  Notification│                        │
│  │    Twin      │  │  (LaTeX/Docx)│  │  (SMS/Push)  │                        │
│  └──────────────┘  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            AI/ML PLATFORM LAYER                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Model Serving (KServe / Triton)                    │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │  PINN Weather│  │  RL Resource  │  │  NLP Research│              │    │
│  │  │    Model     │  │   Optimizer   │  │   Agent      │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │  Crop Yield  │  │  Flood Inund │  │  Pest Pred   │              │    │
│  │  │  Predictor   │  │  ation Model  │  │  iction      │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              Feature Store (Feast) + Vector DB (Milvus/Pinecone)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA PLATFORM LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Data Lake (S3 / MinIO)  │  Data Warehouse (Snowflake/BigQuery)      │    │
│  │  ├─ Raw Ingestion Zone    │  ├─ Climate Analytics                    │    │
│  │  ├─ Curated Zone          │  ├─ Agricultural BI                      │    │
│  │  ├─ Feature Store Zone    │  ├─ Urban Resilience BI                   │    │
│  │  └─ Archive Zone           │  └─ Policy Dashboard                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Time-Series DB (TimescaleDB / InfluxDB)  │  Graph DB (Neo4j)      │    │
│  │  ├─ Sensor telemetry (50M points/day)     │  ├─ Knowledge Graph    │    │
│  │  ├─ Forecast outputs                      │  ├─ Citation Network   │    │
│  │  └─ Historical climate records            │  └─ Policy Dependency    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Geospatial DB (PostGIS / GeoMesa)  │  Search (Elasticsearch)        │    │
│  │  ├─ Raster tiles (COG / Cloud Optimized)│  ├─ Full-text research   │    │
│  │  ├─ Vector layers (districts, watersheds) │  └─ Document indexing    │    │
│  │  └─ 3D terrain (Cesium / Mapbox)        │                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HPC / SIMULATION LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  HPC Cluster (Param Praveer / AWS ParallelCluster)                 │    │
│  │  ├─ WRF / MPAS (Atmospheric)    │  ├─ ROMS / HYCOM (Ocean)        │    │
│  │  ├─ Noah-MP / CLM (Land)      │  ├─ SWAT / MIKE (Hydrology)      │    │
│  │  ├─ PINN Surrogate Models     │  ├─ Ensemble Generation            │    │
│  │  └─ Data Assimilation (DART)  │  └─ Post-Processing (CDO/NCAR)    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  GPU Cluster (NVIDIA DGX / A100)                                     │    │
│  │  ├─ Climate Model Training    │  ├─ RL Agent Training              │    │
│  │  ├─ Inference Serving         │  └─ Generative AI (Report Writing) │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Kubernetes   │  │   Istio      │  │   Vault      │  │   Prometheus │      │
│  │  (EKS/GKE)   │  │  (Service Mesh)│  │ (Secrets)   │  │  + Grafana   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Terraform   │  │   ArgoCD     │  │   Kafka      │  │   Fluentd    │      │
│  │  (IaC)       │  │  (GitOps)    │  │ (Streaming)  │  │ (Logging)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Specifications

### 3.1 Climate Data Fabric (CDF)

#### 3.1.1 Data Ingestion Pipeline

- **Architecture:** Lambda + Kappa (hybrid)
- **Batch Sources:** ISRO Bhuvan (Landsat/Sentinel): Daily, ~500GB/day; CMIP6 projections: Monthly, ~2TB/month; Historical IMD gridded data: One-time, ~50TB
- **Streaming Sources:** IMD AWS network (6,000 stations): 15-min cadence, ~1M records/day; INCOIS ocean buoys: 10-min cadence, ~500K records/day; CWC hydrological stations: Hourly, ~200K records/day; ISRO INSAT-3D: 15-min imagery, ~100GB/day; Crowd-sourced weather (mobile): 5-min, ~2M records/day
- **Ingestion Framework:** Apache Kafka (event streaming backbone); Apache Flink (stream processing, anomaly detection); Apache Airflow (batch orchestration, DAG scheduling); NiFi (data flow management, schema transformation)
- **Schema Registry:** Confluent Schema Registry (Avro/Protobuf/JSON Schema); Versioned schemas with backward compatibility enforcement

#### 3.1.2 Data Harmonization & Quality

The Data Quality Engine implements:
- **Schema Validation:** Automatic validation against versioned schemas (Avro/JSON Schema)
- **Anomaly Detection:** Isolation Forest for outlier detection in sensor telemetry
- **Gap Filling:** Spatiotemporal Kriging for spatial interpolation of missing data
- **Quality Scoring:** Multi-dimensional scoring (completeness, timeliness, accuracy, consistency)
- **Physical Plausibility Checks:** Cross-variable consistency (e.g., temperature vs. humidity bounds)

#### 3.1.3 Storage Architecture

| Data Type | Technology | Retention | Format | Access Pattern |
|-----------|------------|-----------|--------|----------------|
| Raw sensor telemetry | TimescaleDB | 2 years | Compressed hypertables | Time-range queries |
| Satellite imagery | S3 + COG | 10 years | Cloud-Optimized GeoTIFF | Spatial tiling |
| Climate model outputs | Zarr on S3 | 30 years | Zarr (chunked arrays) | Multi-dimensional slicing |
| Vector geospatial | PostGIS | Permanent | GeoJSON/Shapefile | Spatial joins |
| Research documents | Elasticsearch | 7 years | JSON + embeddings | Full-text + semantic search |
| Knowledge graph | Neo4j | Permanent | Property graph | Relationship traversal |

### 3.2 Digital Twin Simulation Engine (DTSE)

#### 3.2.1 Model Architecture

The DTSE implements a **hybrid physics-ML** approach with four layers:

**Layer 1: Physics Core (Traditional NWP)**
- WRF-ARW v4.5 (Atmospheric dynamics, microphysics)
- ROMS v3.7 (Ocean circulation, SST)
- Noah-MP (Land surface processes, soil moisture)
- SWAT+ (Hydrology, river routing)
- Data Assimilation: DART / GSI (Ensemble Kalman Filter)

**Layer 2: Physics-Informed ML (PINNs)**
- Fourier Neural Operator (FNO) for PDE acceleration
- Neural Advection-Diffusion for tracer transport
- Graph Neural Network (GNN) for spatial downscaling
- Transformer-based temporal attention for long-range dependencies

**Layer 3: Surrogate & Emulator Models**
- Deep Kernel Learning (DKL) for ensemble generation
- Conditional Diffusion for scenario sampling
- Neural Radiance Fields (NeRF) for 3D visualization
- Model Compression: Knowledge distillation (teacher to student)

**Layer 4: AI-Based Correction & Bias Adjustment**
- Quantile Mapping Neural Network (QMNN)
- Deep Quantile Regression for uncertainty
- Reinforcement Learning for parameter tuning
- Federated Learning for multi-model ensemble weighting

#### 3.2.2 Computational Requirements

| Simulation Type | Resolution | Compute | Memory | Storage/Run | Frequency |
|-----------------|------------|---------|--------|-------------|-----------|
| Nowcast (0-6h) | 1km | 256 cores | 512 GB | 50 GB | Every 15 min |
| Short-range (0-72h) | 3km | 1024 cores | 2 TB | 200 GB | Every 6 hours |
| Medium-range (0-10 days) | 12km | 4096 cores | 8 TB | 1 TB | Every 12 hours |
| Seasonal (monsoon) | 25km | 2048 cores | 4 TB | 500 GB | Monthly |
| Climate scenario (30yr) | 50km | 8192 cores | 16 TB | 10 TB | Annual |
| Urban flood (event) | 100m | 512 cores + 8x A100 | 1 TB | 100 GB | On-demand |

#### 3.2.3 HPC Job Orchestration

Slurm job template for WRF ensemble:
- **Partition:** compute_gpu
- **Nodes:** 64, **Tasks per node:** 64, **GPUs per node:** 4
- **Memory:** 2TB, **Time:** 04:00:00
- **Preprocessing:** WPS ungrib + metgrid; GFS analysis + regional obs nudging; 6-hourly GFS forecasts
- **Execution:** Generate initial/boundary files; Run 50-member ensemble (stochastic physics); UPP for standard variables + custom ICDT outputs
- **Output:** Zarr + NetCDF4 format; CF-1.8 compliant, PROV-O provenance

### 3.3 Deep Research Agent (DRA)

#### 3.3.1 Multi-Agent Architecture

The DRA implements a **hierarchical multi-agent system** inspired by Anthropic's research architecture and OpenAI's Deep Research, adapted for climate science domains.

**Master Agent:** Orchestrator that delegates to Planner, dispatches parallel Researcher agents, synchronizes findings, and delegates to Writer.

**Planner Agent:** Creates DAG of sub-questions with exploratory search, context discovery, and HITL approval.

**Researcher Agents (Parallel):** Execute searches, synthesize findings, identify gaps, and self-evaluate with coverage-driven stopping.

**Writer Agent:** Performs thematic analysis, parallel section writing, citation management, and format export.

**Searcher Agent:** Query generation, web search, cross-encoder reranking, semantic chunking, and MMR diversification.

**Refiner Agent:** Coverage assessment, iteration control, and budget tracking.

**Clarifier Agent:** Human-in-the-loop for ambiguous queries.

**Formatter Agent:** LaTeX/PDF, HTML Dashboard, PowerPoint, and JSON API output generation.

#### 3.3.2 Agent Specifications

**Master Agent:**
- Model: gpt-4.1 or equivalent (Gemini 2.5 Pro, Claude 3.7 Sonnet)
- Temperature: 0.1, Max tokens: 4096
- Tools: planner_agent, researcher_agent, writer_agent, state_manager
- Workflow: Receive query -> Delegate to Planner -> Topological sort DAG -> Spawn parallel ResearcherAgents -> Synchronize -> Delegate to Writer -> Return artifact

**Planner Agent:**
- Model: gpt-4.1 (strong reasoning)
- Temperature: 0.2
- Human-in-the-loop: true (pause for user confirmation)
- Output: DAG (JSON) with topological ordering
- Max sub-questions: 15 (standard), 30 (deep research)

**Researcher Agent (Parallel):**
- Model: gpt-4.1-mini (cost-efficient)
- Temperature: 0.3, Max iterations: 5, Max pages per query: 10
- Search tools: Google Search API, Scholar API, Internal KB (Neo4j), Data Catalog (CKAN), Exa API
- Stopping criteria: Coverage threshold 0.85, Max iterations 5, Confidence threshold 0.80

**Writer Agent:**
- Model: gpt-4.1 (highest quality)
- Temperature: 0.1
- Output formats: Markdown, LaTeX, HTML, JSON, PowerPoint, Cabinet memo (bilingual)

#### 3.3.3 Search & Retrieval Architecture

The Search Orchestrator implements a 5-phase pipeline:
1. **Query Expansion:** Synonym expansion, entity linking, temporal context
2. **Parallel Source Search:** Web, Scholar, Internal KB executed concurrently
3. **Cross-Encoder Reranking:** Cross-encoder/ms-marco-MiniLM-L-6-v2 for relevance scoring
4. **Maximal Marginal Relevance:** Lambda=0.5 balancing relevance vs diversity, top_k=20
5. **Content Extraction & Chunking:** Semantic chunking (512 tokens, 128 overlap, hierarchical strategy)

### 3.4 Agricultural Decision Support System (ADSS)

#### 3.4.1 Farm Digital Twin Model

Per-farm digital twin with 30m resolution:
- **Soil Profile:** ISRO-NBSS&LUP data
- **Crop Model:** DSSAT / APSIM integration
- **State Updates:** Soil moisture estimation (Sentinel-1 + in-situ sensors), crop stage advancement, pest risk prediction
- **Advisory Generation:** AI-driven irrigation optimization, nutrient management, IPM recommendations
- **Output:** Multi-lingual advisory with confidence scores

#### 3.4.2 Mobile Architecture

**Mobile App (Flutter):**
- Offline-first architecture (SQLite + Hive)
- Background sync when connectivity available
- Voice-based interaction (Indic NLP: IndicWav2Vec + BERT)
- Push notifications for alerts (heatwave, pest outbreak)
- Low-bandwidth mode (<2MB/day data usage)

**Backend Services:**
- Farm API: FastAPI (Python) - CRUD for farm data
- Advisory Engine: Celery + Redis - Async task queue
- Notification Service: Firebase Cloud Messaging
- Voice Service: Indic TTS (AI4Bharat) + ASR

### 3.5 Urban Climate Resilience Dashboard (UCRD)

#### 3.5.1 Flood Simulation Pipeline

High-resolution urban flood modeling:
- **Input:** 0.5m LiDAR DEM, building footprints, drainage network, land use
- **Step 1:** Rainfall-runoff computation (SWMM or custom GPU solver)
- **Step 2:** 2D shallow water equations (GPU-accelerated, 1-second CFL-stable)
- **Step 3:** Impact assessment (affected buildings, population, economic loss)
- **Output:** FloodMap with depth grid, velocity grid, arrival time, impact assessment

#### 3.5.2 Real-Time Dashboard Stack

**Frontend:**
- React + Next.js + TypeScript
- MapLibre GL (open-source Mapbox alternative)
- CesiumJS (3D terrain + flood depth)
- Apache ECharts (time-series, ensemble spread)
- Zustand + React Query (state management)

**Backend:**
- GraphQL API: Hasura (real-time subscriptions via WebSocket)
- Tile Server: TiTiler (Cloud-Optimized GeoTIFF serving)
- Vector Tiles: Tippecanoe + Martin
- Real-time: WebSocket server (Socket.io) for live sensor feeds

---

## 4. Data Architecture

### 4.1 Data Flow

```
IMD AWS (Kafka) -> Ingestion Layer (Kafka + Flink) -> Raw Zone (S3/MinIO)
ISRO SAT (S3/MinIO) -> Streaming Processing (Flink) -> Curated (Spark/Delta Lake)
INCOIS OCEAN (FTP/API) -> Batch Pipeline (Airflow) -> Feature Store (Feast)
CWC HYDRO (API) -> Analytics Warehouse (Snowflake) -> ML Training Dataset (S3 + DVC)
                                                    -> Serving Layer (API + Cache)
                                                    -> Dashboard (React) / Mobile (Flutter) / Research API
```

### 4.2 Data Models

#### 4.2.1 Climate Observation Schema (Avro)

Fields: observation_id, station_id, timestamp (timestamp-millis), location (GeoPoint: latitude, longitude, elevation), variables (map of ClimateVariable: value, unit, quality_flag, sensor_type, uncertainty), metadata (map of strings), provenance (source_system, ingestion_timestamp, processing_version, data_lineage array)

#### 4.2.2 Digital Twin State Schema

Fields: twin_id, valid_time (timestamp-millis), forecast_hour, ensemble_member, spatial_extent (BoundingBox: min_lat, max_lat, min_lon, max_lon, resolution), atmospheric_fields (array of AtmosphericField: variable, levels, data_url, checksum), land_surface_fields, ocean_fields, uncertainty (ensemble_spread, confidence_interval, model_bias), model_metadata (model_name, model_version, physics_options, initialization_source, compute_cluster)

### 4.3 Knowledge Graph Schema (Neo4j)

**Nodes:** Concept (climate_phenomenon), Variable (climate variable), Dataset, Paper, Model, Location, Event

**Relationships:** INFLUENCED_BY (strength, mechanism), STUDIES (finding), USES_DATASET, SIMULATES (skill_score), OCCURRED_IN, CAUSED_BY (contribution), COVERS

---

## 5. AI/ML Architecture

### 5.1 Model Registry & MLOps

**ML Platform:** MLflow + Kubeflow

**Model Lifecycle:**
1. Experiment Tracking: MLflow (metrics, parameters, artifacts)
2. Feature Engineering: Feast (feature store with online/offline serving)
3. Training: Kubeflow Pipelines (distributed training on GPU cluster)
4. Validation: Custom test suite (physical plausibility, bias detection)
5. Deployment: KServe (canary, A/B testing, auto-scaling)
6. Monitoring: Evidently AI (data drift, concept drift, performance degradation)
7. Retraining: Automated trigger on drift detection or schedule

**Model Registry:** Staging, Production, Archived (7-year retention)

**Model Cards:** Google framework documenting intended use, limitations, training data, performance metrics, fairness analysis, environmental impact, ethical considerations

### 5.2 Key ML Models

#### 5.2.1 Physics-Informed Neural Network (PINN) for Weather

Fourier Neural Operator for atmospheric PDE approximation:
- Modes: 12, Width: 32
- In channels: 10 (u, v, T, q, p, etc.), Out channels: 10
- Physics loss: Navier-Stokes PDE residual
- Loss function: MSE + 0.1 * PDE residual + 0.01 * boundary condition loss

#### 5.2.2 Deep Reinforcement Learning for Resource Optimization

DDPG-based agent for adaptive grid/water resource management:
- Actor-Critic architecture with target networks
- State: [grid_voltage, renewable_generation, climate_forecast, demand]
- Training: Bellman target update, critic MSE loss, actor policy gradient

#### 5.2.3 Deep Research NLP Pipeline

SentenceTransformer (paraphrase-multilingual-mpnet-base-v2) for embeddings, BART-large-cnn for summarization, BERT-NER for entity extraction, custom CitationExtractor and FactChecker modules.

### 5.3 Training Infrastructure

**GPU Cluster:**
- 8x NVIDIA DGX A100 (64 GPUs total)
- 2x NVIDIA DGX H100 (16 GPUs total)
- InfiniBand HDR 200 Gbps interconnect
- NVMe SSD storage: 2 PB

**Training Frameworks:** PyTorch 2.0+, JAX + Flax, Ray Train, DeepSpeed

**Experiment Tracking:** Weights & Biases, MLflow, DVC

---

## 6. Security Architecture

### 6.1 Threat Model

| Threat | Vector | Impact | Mitigation |
|--------|--------|--------|------------|
| Data poisoning (sensor) | Compromised IoT devices | Incorrect forecasts, false alarms | Anomaly detection, multi-source validation, cryptographic sensor attestation |
| Model inversion attack | API probing | Reconstruction of sensitive training data | Differential privacy, output perturbation, rate limiting |
| Adversarial inputs | Manipulated satellite imagery | Wrong classification (e.g., flood missed) | Adversarial training, input sanitization, ensemble diversity |
| Insider threat | Authorized user misuse | Data exfiltration, model theft | Zero-trust access, DLP, behavioral analytics, least privilege |
| DDoS on forecast APIs | Botnet traffic | Service unavailability during disasters | CDN, WAF, auto-scaling, rate limiting, circuit breakers |
| Supply chain (open-source) | Compromised dependency | Backdoor in model serving | SBOM, dependency scanning, signed containers, reproducible builds |

### 6.2 Security Controls

**Identity & Access:**
- Identity Provider: Keycloak (OIDC/OAuth2)
- MFA: TOTP + hardware keys (YubiKey) for admins
- RBAC: 50+ predefined roles
- Service Mesh: Istio mTLS for all inter-service communication

**Data Protection:**
- Encryption: AES-256-GCM at rest; TLS 1.3 in transit
- Key Management: HashiCorp Vault (HSM-backed)
- Field-level encryption for PII
- Data Residency: All Indian citizen data stored in India (MeitY compliance)

**Network Security:**
- WAF: AWS WAF / Cloudflare (DDoS protection)
- API Gateway: Kong with OAuth2 introspection
- VPC segmentation: Microservices isolated in private subnets
- VPN: WireGuard for remote access to HPC cluster

**Monitoring & Response:**
- SIEM: Splunk / Wazuh
- IDS/IPS: Suricata
- Vulnerability scanning: Trivy (containers), Snyk (dependencies)
- Incident response: 24/7 SOC (Security Operations Center)

---

## 7. Infrastructure & Deployment

### 7.1 Cloud Architecture

**Primary Region:** Pune (AWS Mumbai / Azure Pune)
**DR Region:** Hyderabad (AWS Hyderabad / Azure Hyderabad)
**Edge Nodes:** Delhi, Chennai, Kolkata, Bengaluru, Guwahati, Jaipur, Lucknow, Patna, Bhubaneswar, Thiruvananthapuram

**Compute:**
- Kubernetes: EKS/AKS with 3+ node pools (General: c6i.2xlarge, Memory: r6i.4xlarge, GPU: p4d.24xlarge)
- HPC: On-premises + Param Praveer (C-DAC) + AWS ParallelCluster

**Storage:**
- Hot: EFS / Azure Files
- Warm: S3 Standard-IA / Azure Blob Hot tier
- Cold: S3 Glacier / Azure Blob Archive (7-year retention)
- Database: RDS PostgreSQL, ElastiCache Redis

**Networking:**
- CDN: CloudFront / Azure CDN
- Load Balancer: ALB + NLB
- DNS: Route 53 / Azure DNS (geo-routing)
- DDoS: AWS Shield Advanced / Azure DDoS Protection

### 7.2 Deployment Pipeline

**CI/CD:**
- Git: GitLab (self-hosted, air-gapped for classified code)
- CI: GitLab CI / GitHub Actions
- CD: ArgoCD (GitOps, declarative deployments)
- IaC: Terraform + Pulumi
- Container Registry: Harbor (self-hosted, vulnerability scanning)

**Deployment Strategy:**
- Development: Feature branches -> auto-deploy to dev namespace
- Staging: Merge to main -> auto-deploy to staging
- Production: Manual approval -> ArgoCD sync -> blue-green deployment
- HPC: Ansible playbooks for bare-metal provisioning

**Rollback:**
- Kubernetes: Rollback to previous ReplicaSet (<30s)
- Database: Point-in-time recovery (PITR) via RDS
- Models: MLflow model version rollback (<1 minute)

### 7.3 Disaster Recovery

| Scenario | RTO | RPO | Strategy |
|----------|-----|-----|----------|
| Single AZ failure | 5 min | 0 | Multi-AZ Kubernetes, auto-failover |
| Region failure | 4 hours | 15 min | Cross-region replication, DR site warm standby |
| HPC cluster failure | 8 hours | 1 hour | Backup cluster at C-DAC; queue migration |
| Data corruption | 2 hours | 15 min | Immutable backups (WORM), point-in-time recovery |
| Cyberattack (ransomware) | 24 hours | 1 hour | Air-gapped backups, immutable snapshots |

---

## 8. Integration Specifications

### 8.1 External System Interfaces

| System | Protocol | Data Format | Frequency | Authentication |
|--------|----------|-------------|-----------|----------------|
| IMD GFS | FTP/SFTP | GRIB2 | 6-hourly | IP whitelist + SSH key |
| ISRO Bhuvan | REST API | GeoTIFF/JSON | On-demand | API key (ISRO Auth) |
| INCOIS | HTTP/HTTPS | NetCDF/JSON | Hourly | OAuth2 |
| CWC | REST API | JSON/XML | Hourly | mTLS + API key |
| AgriStack | REST API | JSON | Real-time | OAuth2 + DPDP consent |
| PM-KISAN | REST API | JSON | Daily | Government gateway |
| NDMA ERSS | REST API | JSON | Real-time | VPN + API key |
| Smart City CC | MQTT/WebSocket | JSON | Real-time | X.509 certificates |
| WMO WIS 2.0 | WMO standard | BUFR/GRIB2 | Hourly | WMO credentials |
| Google Flood Hub | REST API | GeoTIFF/JSON | On-demand | API key |

### 8.2 API Specifications

#### 8.2.1 Core API (OpenAPI 3.0)

**Endpoints:**
- `GET /forecast/weather` - Get weather forecast for a location (lat, lon, lead_time, ensemble)
- `POST /research/deep-research` - Initiate deep research query (query, depth, output_format, language, human_approval)
- `POST /agriculture/advisory` - Get farm-level advisory (farm_id, crop, growth_stage, language)

**Schemas:**
- WeatherForecast: location, valid_time, lead_time_hours, variables, ensemble
- FarmAdvisory: farm_id, generated_at, recommendations (type, action, timing, confidence, rationale)

---

## 9. Monitoring & Observability

### 9.1 Observability Stack

**Metrics:** Prometheus (collection) + Grafana (visualization)
**Logs:** Fluentd (collection) + Elasticsearch (storage) + Kibana (visualization); Retention: Hot 7 days, Warm 90 days, Cold 7 years
**Traces:** Jaeger (distributed tracing) + OpenTelemetry (instrumentation); Trace sampling: 10% normal, 100% errors
**Alerting:** PagerDuty (incident management) + Alertmanager (Prometheus integration); Escalation: L1 -> L2 -> L3

**SLOs:**
- API availability: 99.95% (monthly)
- API latency p99: <500ms
- Forecast accuracy (48h rainfall): >0.50 CSI
- Deep research completion: <30 min (standard), <2 hours (comprehensive)
- Data freshness: <15 min for real-time sources

### 9.2 Model Monitoring

Continuous monitoring using EvidentlyAI for data drift detection, MLflowMetrics for performance tracking, and FairnessIndicators for bias auditing. Automated retraining triggers on drift detection or accuracy < 0.85.

---

## 10. Development Standards

### 10.1 Coding Standards

**Languages:** Python 3.11+ (primary), TypeScript (frontend), Go (high-performance), C++ (HPC), Fortran (legacy NWP)

**Code Quality:**
- Linting: Ruff (Python), ESLint (TS), golangci-lint (Go)
- Formatting: Black (Python), Prettier (TS), gofmt (Go)
- Type Checking: mypy (Python), strict TypeScript
- Testing: pytest, Jest, Go test
- Coverage: 80% minimum for critical paths, 60% for non-critical

**Documentation:**
- API: OpenAPI 3.0 with Swagger UI
- Code: Google-style docstrings, mkdocs
- Architecture: C4 model
- Runbooks: Markdown in /docs/runbooks/

### 10.2 Git Workflow

**Branching:** GitFlow + trunk-based development (main, develop, feature/*, hotfix/*, release/*)
**Commit Standards:** Conventional Commits (feat, fix, docs, style, refactor, perf, test, chore)
**PR Requirements:** 1+ approval (2 for critical), all CI passing, coverage maintained, security scan clean, documentation updated

---

## 11. Testing Strategy

### 11.1 Test Pyramid

**Unit Tests (70%):** pytest, 80% coverage, mocking with unittest.mock/moto, run every PR
**Integration Tests (20%):** pytest + TestContainers, service-to-service interactions, synthetic data, run on merge to develop
**E2E Tests (10%):** Playwright (frontend), pytest + requests (API), full user journeys, nightly/before release
**Performance Tests:** k6/Locust, 10K concurrent users, 1M API calls/day, HPC benchmark suite
**Security Tests:** SAST (SonarQube, Semgrep), DAST (OWASP ZAP), quarterly penetration testing, dependency scanning (Snyk)

### 11.2 Test Data Management

**Data Classification:** Public, Internal, Confidential, Restricted
**Test Data Strategy:** Synthetic (Faker + perturbation), Anonymized (K-anonymity k=5), Snapshot (daily anonymized production), Fixture (version-controlled JSON/CSV)

---

## 12. Appendices

### Appendix A: Technology Stack Summary

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Frontend | React + Next.js + TypeScript | 18+ | Web dashboard |
| Mobile | Flutter | 3.0+ | Cross-platform app |
| API Gateway | Kong | 3.0+ | Traffic management |
| API Framework | FastAPI | 0.100+ | Python microservices |
| Workflow | LangGraph / Temporal.io | Latest | DRA orchestration |
| Stream Processing | Apache Flink | 1.17+ | Real-time analytics |
| Batch Processing | Apache Spark | 3.4+ | Large-scale ETL |
| Message Queue | Apache Kafka | 3.5+ | Event streaming |
| Time-Series DB | TimescaleDB | 2.11+ | Sensor data |
| Graph DB | Neo4j | 5.0+ | Knowledge graph |
| Vector DB | Milvus | 2.3+ | Semantic search |
| Search | Elasticsearch | 8.0+ | Full-text search |
| Feature Store | Feast | 0.34+ | ML features |
| Model Serving | KServe | 0.11+ | ML inference |
| ML Framework | PyTorch | 2.0+ | Deep learning |
| Scientific | JAX + Flax | 0.4+ | PINNs, physics ML |
| Distributed | Ray | 2.6+ | Distributed training |
| Container | Docker | 24+ | Application packaging |
| Orchestration | Kubernetes | 1.28+ | Container orchestration |
| CI/CD | GitLab CI + ArgoCD | Latest | Build & deploy |
| IaC | Terraform + Pulumi | 1.5+ | Infrastructure |
| Monitoring | Prometheus + Grafana | Latest | Metrics |
| Logging | Fluentd + ELK | 8.0+ | Log aggregation |
| Tracing | Jaeger + OpenTelemetry | Latest | Distributed tracing |
| Security | Vault + Istio + Keycloak | Latest | Secrets, mesh, IAM |

### Appendix B: HPC Software Stack

| Component | Software | Version | Notes |
|-----------|----------|---------|-------|
| Compiler | Intel oneAPI | 2023.2 | icx, ifx, icpx |
| MPI | Intel MPI | 2021.10 | Optimized for InfiniBand |
| NWP | WRF-ARW | 4.5 | Custom Indian monsoon physics |
| Ocean | ROMS | 3.7 | Indian Ocean configuration |
| Land | Noah-MP | 5.0 | ISRO land use parameters |
| Hydrology | SWAT+ | 61.0 | Ganges basin calibration |
| Data Assimilation | DART | 10.0 | Ensemble Kalman Filter |
| Post-processing | CDO | 2.2.0 | Climate Data Operators |
| Visualization | NCL / Python (matplotlib, cartopy) | Latest | Publication-quality plots |
| Workflow | Cylc / Rocoto | Latest | HPC job scheduling |

### Appendix C: Glossary

- **CDF:** Climate Data Fabric
- **DTSE:** Digital Twin Simulation Engine
- **DRA:** Deep Research Agent
- **ADSS:** Agricultural Decision Support System
- **UCRD:** Urban Climate Resilience Dashboard
- **PINN:** Physics-Informed Neural Network
- **NWP:** Numerical Weather Prediction
- **DA:** Data Assimilation
- **HPC:** High-Performance Computing
- **RTO:** Recovery Time Objective
- **RPO:** Recovery Point Objective
- **SLO:** Service Level Objective
- **HITL:** Human-in-the-Loop
- **MMR:** Maximal Marginal Relevance
- **DAG:** Directed Acyclic Graph
- **FNO:** Fourier Neural Operator
- **GNN:** Graph Neural Network
- **DDPG:** Deep Deterministic Policy Gradient
- **MARL:** Multi-Agent Reinforcement Learning

---

*Document Owner: Technical Architecture Board, NCCC*  
*Review Cycle: Bi-weekly during development; Monthly in production*
