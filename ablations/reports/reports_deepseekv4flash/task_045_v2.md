# Comprehensive Research Report: MLOps Platforms for Ensemble Fraud Detection in Southeast Asian Fintech

## Executive Summary

This report provides a deeply researched, production-oriented comparison of four MLOps platforms—Seldon Core with Triton Inference Server, Ray Serve, KServe on Kubernetes, and BentoML—for deploying ensemble fraud detection models (XGBoost, LightGBM, neural networks) at a fintech startup processing 50 million daily payment transactions across India, Indonesia, and the Philippines. The analysis addresses six specific gaps identified in prior research: platform-specific operational mechanisms, concrete comparative performance metrics, detailed real-world company experiences, reconciliation of benchmark versus production performance, regional burst capacity and cost modeling, and regulatory and compliance implications. Each section draws on official documentation, published engineering blogs, academic papers, and cloud provider pricing pages.

---

## 1. Platform-Specific Operational Mechanisms

### 1.1 Seldon Core with Triton Inference Server

#### Request Routing Components and Full Request Path

**Seldon Core v1 Architecture:**
The Service Orchestrator is a sidecar container added to every inference graph. Its responsibilities include managing request/response paths, exposing Prometheus metrics, providing OpenTracing support, and adding CloudEvent-based payload logging. From Seldon Core v1.1, the orchestrator supports Seldon, TensorFlow, and V2 (Open Inference) protocols. For single-model deployments minimizing latency, the annotation `seldon.io/no-engine: "true"` removes the orchestrator entirely. By default, the orchestrator forwards request payloads without deserialization; this can be changed via the `SELDON_ENABLE_ROUTING_INJECTION` environment variable [1].

**Seldon Core v2 Architecture:**
Seldon Core 2 employs a microservice architecture with separate control and data planes:

- **Control Plane:** The Scheduler manages loading/unloading of models, pipelines, and experiments. The Agent manages model loading and acts as a reverse proxy for inference requests. The Controller Manager integrates with Kubernetes through CRD reconciliation [2].

- **Data Plane:** The Pipeline Gateway translates REST/gRPC to Kafka operations for pipeline requests. The Model Gateway manages the flow from models to inference requests via Kafka. The Dataflow Engine handles inter-component data flow using Kafka Streams. Envoy routes requests, balancing load across replicas using weighted least-request balancing [2].

**Full Request Path:**
1. Client sends inference request to Istio Gateway or Ambassador ingress
2. Ingress routes to Pipeline Gateway (v2) or Service Orchestrator (v1)
3. For v2: Pipeline Gateway converts HTTP/gRPC to Kafka messages → Model Gateway → Dataflow Engine → Envoy → backend Triton server
4. Triton receives request via HTTP/REST, gRPC, or C API → routes to per-model scheduler → dynamic batcher groups requests → backend executes inference → response returns through reverse path
5. For v1: Service Orchestrator routes through inference graph nodes (MODEL, TRANSFORMER, ROUTER, COMBINER, OUTLIER_DETECTOR) [1][2]

**Inference Graph / DAG Mechanism:**
The inference graph chains multiple model servers as a single entity. Graph nodes can be:
- **MODEL**: Individual model server
- **TRANSFORMER**: Pre- or post-processing
- **ROUTER**: Traffic splitting
- **COMBINER**: Ensemble methods combining results
- **OUTLIER_DETECTOR**: Monitoring for anomalies [1]

For ensemble fraud detection, a typical graph routes requests through a transformer (feature engineering), then fans out to XGBoost, LightGBM, and neural network model nodes in parallel, aggregating via a combiner, with an optional outlier detector for drift monitoring.

**Ingress Support:**
Seldon Core officially supports Ambassador and Istio. Seldon Enterprise Platform uses Istio for traffic splitting and leverages Knative for drift/outlier detection. Seldon Core 2 works with any service mesh or ingress controller but exposes the `seldon-mesh` service for external communication [3].

#### Autoscaling Triggers, Tunables, and Strategies

Seldon Core supports four autoscaling mechanisms:

**1. Kubernetes HPA with Prometheus Custom Metrics:**
The `seldon_model_infer_total` Prometheus metric computes `infer_rps` (inference requests per second) over a 2-minute sliding window. A Prometheus Adapter exposes this as a Kubernetes custom metric. The formula for target replicas is: `targetReplicas = infer_rps / averageValue` [4].

**2. Inference Lag-Based Scaling (Seldon Core 2):**
The difference between incoming and outgoing requests for a model drives scaling decisions. This is the simplest mechanism, suitable for multi-model serving [2].

**3. KEDA (Kubernetes Event-Driven Autoscaling):**
KEDA provides flexible scaling with support for custom metrics from Prometheus and specialized scalers (Kafka lag, RabbitMQ queue depth, etc.) [5].

**4. Combined Model and Server Autoscaling with HPA:**
Requires a 1:1 mapping of Models and Servers (no multi-model serving). HPA targets the same custom metric for both [2].

**Configurable HPA Parameters:**
- `minReplicas` / `maxReplicas`: Scale range
- `stabilizationWindowSeconds`: Prevents flapping (default varies; recommended 60-300s)
- `scaleDown` / `scaleUp` policies: `periodSeconds`, `type` (Pods or Percent), `value`, `selectPolicy`, `stabilizationWindowSeconds` [4]

**Scale-to-Zero:**
Knative integration enables scale-to-zero through the KnativePodAutoscaler (KPA). Seldon Enterprise Platform configures outlier, drift detectors, and metrics servers as Knative services. Key parameters:
- `enable-scale-to-zero: "true"` globally
- `scale-to-zero-grace-period`: Upper bound for internal setup before last replica removal (default 30s)
- `scale-to-zero-pod-retention-period`: Minimum time last pod remains active after scale-down decision (default 0s) [6]

**Triton's Internal Scheduling (Separate from K8s Autoscaling):**
Triton handles its own dynamic batching and scheduling independent of Kubernetes. Key parameters:
- `preferred_batch_size`: Preferred batch sizes (e.g., `[16]`)
- `max_queue_delay_microseconds`: Maximum delay to wait for batch formation (e.g., 100μs)
- `max_batch_size`: Maximum batch size the model supports
- `instance_group`: Configure multiple model instances (e.g., `instance_group { count: 2 kind: KIND_GPU }`) [7][8]

Triton does not have built-in autoscaling; it relies entirely on external tools like Kubernetes [9].

#### Natively Supported Frameworks and Version Constraints

Triton Inference Server supports the following backends per the official Backend-Platform Support Matrix (Ubuntu 22.04) [10]:

| Backend | x86 CPU | x86 GPU | ARM-SBSA CPU | ARM-SBSA GPU |
|---------|---------|---------|-------------|-------------|
| TensorRT | ❌ | ✅ | ❌ | ✅ |
| ONNX Runtime | ✅ | ✅ | ✅ | ✅ |
| TensorFlow | ✅ | ✅ | ✅ | ✅ |
| PyTorch | ✅ | ✅ | ✅ | ✅ |
| OpenVINO | ✅ | ❌ | ❌ | ❌ |
| Python | ✅ | ✅ | ✅ | ✅ |
| DALI | ✅ | ✅ | ✅ | ✅ (partial) |
| FIL (Forest Inference Library) | ✅ | ✅ | Unsupported | Unsupported |
| TensorRT-LLM | ❌ | ✅ | ❌ | ✅ |
| vLLM | ✅ | ✅ | Unsupported | Unsupported |

**Critical Version Notes:**
- The TensorFlow backend was dropped with Triton version 25.03 [11]
- The FIL backend supports XGBoost, LightGBM, Scikit-Learn random forest, and cuML random forest models [12]
- The Python backend is required for AWS Inferentia support [10]

**CPU vs GPU Constraints:**
The FIL backend supports both CPU and GPU for tree-based models. OpenVINO is CPU-only. TensorRT and TensorRT-LLM are GPU-only. Within the same Seldon deployment, different inference servers can run on different hardware by configuring separate SeldonDeployment resources with appropriate resource specifications [1][10].

---

### 1.2 Ray Serve

#### Request Routing Components and Full Request Path

**Architecture Overview:**
Ray Serve runs on Ray and utilizes four actor types: Controller, HTTP Proxy, gRPC Proxy, and Replicas [13].

**Full Request Path:**
1. Incoming HTTP request received via FastAPI/ASGI integration (Uvicorn server in HTTP Proxy)
2. HTTP Proxy applies routing algorithm to select replica
3. Default router uses "Power of Two Choices": randomly sample two replicas, route to the replica with fewer ongoing requests [14]
4. Request queued and distributed to replica for processing
5. Response returned through reverse path

**Advanced Routing Capabilities:**
- **PrefixCacheAffinityRouter (Ray 2.49+):** Routes requests with similar prefixes to same replicas, optimizing for workloads with shared prefixes. Maintains a character-level prefix tree and queries it with new request contents [14][15]

- **Custom Routing:** Users can extend `RequestRouter` base class with two patterns:
  - Pattern 1 (Centralized store): Singleton metric store with strong consistency, potential bottleneck
  - Pattern 2 (Metrics broadcast): Controller polls replicas, distributes metrics to all routers [14]

- **DeploymentHandles:** Wrap local routing to other replicas, enabling model composition and efficient inter-deployment calls [13]

**Deployment Graph API for DAG Composition:**
The Deployment Graph API (introduced in alpha) allows building scalable inference pipelines as directed acyclic graphs (DAGs) using Python-native syntax. Key features [16]:

- **Model Chaining:** Passing output through a sequence of models
- **Fanout and Ensemble:** Sending to multiple downstream models in parallel and combining outputs
- **Dynamic Selection and Dispatch:** Dynamically choosing path based on request metadata
- **Independent Scalability:** Each node scales independently
- **DAGDriver:** Ingress component that holds the graph as a DAG Python object

Example for fraud detection ensemble:
```python
with InputNode() as transaction:
    features = preprocessor.process.bind(transaction)  # CPU
    xgb_pred = xgboost_model.forward.bind(features)     # CPU
    lgb_pred = lightgbm_model.forward.bind(features)    # CPU
    nn_pred = neural_net.forward.bind(features)         # GPU
    ensemble = aggregator.bind(xgb_pred, lgb_pred, nn_pred)  # CPU
```

**gRPC Support:**
Ray Serve supports gRPC via a dedicated gRPC Proxy actor. Large requests over 100KiB use Ray's shared memory object store for zero-copy reads. Throughput optimizations using HAProxy ingress + gRPC communication showed 3.2x improvement on DLRM pipelines [17].

#### Autoscaling Triggers, Tunables, and Strategies

Ray Serve has an application-level autoscaler that sits on top of the Ray Autoscaler (node-level) [18].

**Complete Configurable Parameters for AutoscalingConfig [19][20]:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_replicas` | 1 | Minimum replicas. Set to 0 for scale-to-zero |
| `max_replicas` | 100 (with `num_replicas="auto"`) or 1 (manual) | Maximum replicas |
| `target_num_ongoing_requests_per_replica` | N/A | Average ongoing requests per replica the autoscaler targets |
| `max_concurrent_queries` | 5 (deployment default) | Maximum ongoing requests per replica (must be > target) |
| `metrics_export_interval` | N/A | How often to scrape metrics [DEPRECATED] |
| `look_back_period` | N/A | Time window for metrics aggregation |
| `smoothing_factor` | N/A | Multiplicative gain to limit scaling decisions |
| `upscale_delay_s` | N/A | Delay before upscaling |
| `downscale_delay_s` | N/A | Delay before downscaling |
| `initial_replicas` | 1 | Starting replica count |

**Critical Constraint:** If `target_num_ongoing_requests_per_replica` equals `max_concurrent_queries`, autoscaling never occurs. Workaround: set `max_concurrent_queries` to target + 1 [21].

**Ray Autoscaler (Node-Level) Parameters:**
- `max_workers`: Max worker nodes (excluding head)
- `min_workers`: Minimum worker nodes
- `upscaling_speed`: Rate of node additions (default 1.0 = 100% increase)
- `idle_timeout_minutes`: Time before idle node removal (default 5 minutes) [22]

**Scale-to-Zero Support:**
Setting `min_replicas` to 0 is supported, with the caveat that extra tail latency during upscale is expected. Future roadmap includes "Scale-to-zero with lazily invoked models" [16][18].

**Kubernetes Integration:**
Ray Service (KubeRay CRD) does not natively support the `/scale` subresource required by KEDA. HPA is available but scaling involves adjusting `num_replicas` within the `serveConfigV2` string, complicating integration. Custom metrics autoscaling is proposed in GitHub Issue #51632 [23][24].

#### Natively Supported Frameworks

Ray Serve's Python-native architecture means any Python model framework is supported. Explicitly documented support includes [25]:
- PyTorch (including Lightning)
- TensorFlow and Keras
- scikit-learn
- XGBoost
- LightGBM
- ONNX Runtime
- HuggingFace Transformers
- vLLM (with continuous batching and multi-GPU tensor parallelism)
- NVIDIA TensorRT-LLM
- JAX, DeepSpeed, Horovod
- MLflow models
- Custom Python functions

**CPU vs GPU Deployment:**
Each node in the deployment graph can specify fractional CPU or GPU resources via `ray_actor_options`. Ray supports fractional GPUs (e.g., `num_gpus=0.2` allows up to 5 replicas per GPU). Resources specified are "logical"—Ray does not enforce physical core isolation but looks for nodes with sufficient capacity [25][26].

**Model Multiplexing:**
Ray Serve supports model multiplexing with LRU eviction policy (`max_num_models_per_replica`), routing requests to replicas with loaded models [27].

---

### 1.3 KServe on Kubernetes

#### Request Routing Components and Full Request Path

**Deployment Modes [28][29]:**

**Standard Mode (RawDeployment):**
- Uses standard Kubernetes Deployments and Services
- Supports Gateway API (preferred) or Kubernetes Ingress (legacy)
- No native scale-to-zero
- Minimal overhead, no cold starts
- Recommended for production and LLM workloads

**Knative Mode (Serverless):**
- Leverages event-driven scaling with scale-to-zero, queuing, and traffic splitting
- Incurs overhead from queue proxy and cold start latency

**Full Request Path (Knative Mode) [30]:**
1. HTTP request arrives at Istio Ingress Gateway (HTTP Router)
2. Routing decisions recorded in internal headers; router tagged with selected Revision
3. **Low/Zero Traffic Path:** Router sends to Activator, which buffers requests and signals the autoscaler to increase capacity
4. **High Traffic Path:** When spare capacity exceeds `target-burst-capacity`, router bypasses Activator and goes directly to pod addresses
5. **Selectorless Kubernetes Service:** Dynamically switches between Activator endpoints and pod endpoints
6. **Queue-Proxy Sidecar:** Always in path—measures concurrency, enforces `containerConcurrency` limits, handles graceful shutdown, readiness checks, and reports metrics to autoscaler
7. **Inference Pod:** Model serving container processes request

**KServe InferenceGraph CRD [31][32]:**

Four routing node types:
- **Sequence Node:** Steps execute in order, passing request/response between steps
- **Switch Node:** Routes based on defined conditions, executes first matching step
- **Ensemble Node:** Combines multiple model scores using majority vote or averaging—runs all steps in parallel, combines responses keyed by step names
- **Splitter Node:** Distributes traffic among targets with weighted proportions (must sum to 100)

The InferenceGraph is deployed behind an HTTP endpoint and can autoscale according to request volume. Header propagation is configurable through `inferenceservice-config` ConfigMap [31].

**KServe Model Agent (ModelMesh) [33]:**
In multi-model serving, a sidecar dynamically loads/unloads models based on demand, routing inference requests across multiple models and replicas. The inference pool groups model servers together.

#### Autoscaling Triggers, Tunables, and Strategies

**KnativePodAutoscaler (KPA) [34][35][36]:**

KPA operates in two modes—stable and panic—based on concurrency or requests per second.

**Exact Configurable Parameters (from config-autoscaler ConfigMap) [35][37]:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `container-concurrency-target-percentage` | "70" | Target concurrency utilization in stable state |
| `container-concurrency-target-default` | "100" | Target concurrency when unlimited |
| `requests-per-second-target-default` | "200" | Target RPS for RPS-based scaling |
| `target-burst-capacity` | "211" | Burst size; if spare capacity < this, Activator stays in path. -1 = unlimited, 0 = activator only when scaled to 0 |
| `stable-window` | "60s" | Average concurrency over this window in stable mode |
| `panic-window-percentage` | "10.0" | Panic window = 10% of stable window (6s) |
| `panic-threshold-percentage` | "200.0" | Enter panic mode when concurrency reaches 200% of target |
| `max-scale-up-rate` | "1000.0" | Max ratio of desired to existing pods per evaluation cycle (2s) |
| `max-scale-down-rate` | "2.0" | Max ratio of existing to desired pods per evaluation cycle |
| `enable-scale-to-zero` | "true" | Allow scaling to zero replicas |
| `scale-to-zero-grace-period` | "30s" | Max time for internal network programming before last replica removal |
| `scale-to-zero-pod-retention-period` | "0s" | Minimum time last pod remains active |
| `initial-scale` | "1" | Default initial target scale after revision creation |
| `allow-zero-initial-scale` | "false" | Allow initial-scale of 0 |
| `min-scale` | "0" | Cluster-wide default for minimum replicas |
| `max-scale` | "0" | Cluster-wide default for maximum replicas (0 = unlimited) |
| `scale-down-delay` | "0s" | Time that must pass at reduced concurrency before scale-down |
| `activator-capacity` | "100.0" | Capacity of a single activator task |

**Panic Mode Behavior:**
When `desiredPanicPodCount / currentReadyPodsCount >= PanicThreshold` (default 200%), the autoscaler switches to the 6-second panic window. In panic mode, only scale-ups are allowed. Panic mode persists for the duration of the stable window (60s). The autoscaler uses exponentially weighted moving average, giving more weight to recent data [35][37].

**HPA Mode:**
Configured via annotation `autoscaling.knative.dev/class: "hpa.autoscaling.knative.dev"`. Uses CPU/memory metrics. Does NOT support scale-to-zero. KPA reacts faster than HPA for both low- and high-latency requests [36].

**KEDA Integration (KServe v0.15+):**
Native integration with KEDA for scaling based on token throughput, power consumption, and Prometheus metrics. Particularly important for LLM inference where token-level metrics matter. The vLLM Production Stack Helm chart (v0.1.9+) integrates KEDA with configurable min/max replicas, polling interval, and cooldown period. Scale-to-zero with KEDA is possible by setting `minReplicaCount: 0` with traffic-based keepalive triggers [38][39][40].

**Scale-to-Zero Strategies:**
- Set `autoscaling.knative.dev/minScale: "0"` per service to enable scale-to-zero
- For latency-critical applications (fraud detection), set `minScale: 1` or higher to avoid cold starts [36]
- The `scale-to-zero-pod-retention-period` retains pods briefly after scaling to handle residual traffic

#### Natively Supported Frameworks

KServe supports the following model serving runtimes [41]:

1. **TensorFlow Serving** - CPU + GPU
2. **TorchServe** - CPU + GPU (PyTorch)
3. **Triton Inference Server** - CPU + GPU (multi-framework)
4. **ONNX Runtime** - CPU + GPU
5. **MLServer** - CPU (Python-based, by Seldon)
6. **SKLearn** - CPU (scikit-learn)
7. **XGBoost** - CPU + GPU
8. **LightGBM** - CPU
9. **PMML** - CPU
10. **PaddlePaddle** - CPU + GPU
11. **HuggingFace ModelServer** - CPU + GPU
12. **HuggingFace VLLM** - GPU
13. **Custom Runtimes** - User-defined

**Storage URI Mechanism:**
Models are loaded from S3, GCS, Azure Blob, HTTP/HTTPS, PVC, or local filesystem via the `storageUri` field in InferenceService specs [28][33].

**Multi-Model Serving (ModelMesh) [33]:**
- Traditional deployment on 8 CPU, 64GB node: ~40 models
- ModelMesh on same hardware: up to 20,000 models
- Achieves ~1000 QPS with single-digit millisecond latency
- GPU becomes bottleneck depending on load; sidecar resource requirements unchanged

**Revision Management:**
Each update to a Knative Service creates a unique revision. Revisions are snapshots of application code and configuration. Traffic splitting between revisions enables canary and blue/green deployments [42].

---

### 1.4 BentoML

#### Request Routing Components and Full Request Path

**Architecture Overview:**
BentoML Services are class-based, defined with the `@bentoml.service` decorator. Methods with `@bentoml.api` become HTTP API endpoints. The API server is built on Uvicorn/Starlette/FastAPI, providing async request handling [43].

**Full Request Path:**
1. Client sends HTTP request to BentoML Service endpoint (default port 3000)
2. Uvicorn/Starlette handles routing to the appropriate `@bentoml.api` method
3. If adaptive batching is enabled, the dispatcher accumulates requests until batch size or batch window conditions are met
4. Request is processed by the model runner (or legacy runner architecture)
5. Response returned through reverse path

**gRPC Support (Beta):**
BentoML supports gRPC but it remains a beta feature. Performance testing showed gRPC achieved 1.308 seconds vs HTTP's 4.53 seconds for transferring a tensor of shape 1 × 4.19 million. Binary encoding in gRPC makes structured data payloads 30% smaller than REST using JSON [44][45].

**Adaptive Batching Mechanism [46]:**
The adaptive batching algorithm continuously learns and adjusts batching parameters based on recent trends in request patterns and processing time. Key configuration:
- Disabled by default; enabled via `@bentoml.api` decorator
- Batchable APIs accept only one parameter (plus `bentoml.Context`)
- If processing exceeds `max_latency_ms`, returns HTTP 503 Service Unavailable
- Default max latency is 10,000ms

**Distributed Services / Runner Architecture [47]:**
BentoML 1.2+ uses a Service-dependent graph where Services declare dependencies using `bentoml.depends()`. Services interact with remote dependencies using simple Python function calls. Each Service operates on its designated instance type and scales independently.

For ensemble fraud detection:
- Preprocessing Service (CPU) with feature store lookups
- XGBoost Runner (CPU)
- LightGBM Runner (CPU)
- Neural Network Runner (GPU with fractional allocation)
- Ensemble Service (CPU) aggregating outputs
- Services wired together via `bentoml.depends()`

The legacy "runner" architecture allowed models to be deployed independently; `bentoml.runner_service()` helps convert legacy runners to Services for migration from BentoML 1.1 to 1.2+ [43].

#### Autoscaling Triggers, Tunables, and Strategies

**BentoCloud Autoscaling [48]:**
- Users configure minimum and maximum replicas
- Enable scale-to-zero by setting minimum replicas to 0
- Concurrency must be set via `@bentoml.service` decorator to enable request-based scaling
- Autoscaler uses formula involving concurrency thresholds and replica settings
- External request queue can be enabled to buffer requests exceeding concurrency limits
- Stabilization windows configurable between 0 and 3600 seconds

**Scaling Metrics:**
Traditional CPU and GPU utilization metrics were found insufficient. Request-based metrics, especially concurrency (active requests queued or processing), are more effective for proactive scaling [49].

**Self-Managed Kubernetes:**
Uses standard Kubernetes HPA with periodic evaluation (default every 15 seconds). The `autoscaling/v2` API supports multiple metrics, custom metrics, and container-level resource metrics. Configurable scaling behaviors include:
- Scale-up and scale-down rate policies
- Stabilization windows
- Tolerance settings [50]

**Interaction Between Adaptive Batching and Autoscaling:**
When adaptive batching is enabled, batch size adjusts dynamically. The BentoCloud autoscaler uses concurrency as the scaling metric. If a service exceeds `max_latency_ms`, it returns HTTP 503, signaling the need to scale [46][48].

**Cold Start Mitigation [51][52]:**
BentoML optimized cold starts through:
- Replacing container registries with direct object storage downloads (parallel, multi-part downloads reaching ~2 GB/s, reducing download times from minutes to ~10 seconds)
- FUSE-based filesystems for on-demand, lazy loading
- Zero-copy stream-based model loading directly into GPU memory
- Result: 25x faster cold starts (from ~11 minutes to under 30 seconds for Llama 3.1 8B)

#### Natively Supported Frameworks

BentoML's Framework APIs reference lists the following supported frameworks [53]:

| Framework | Category |
|-----------|----------|
| PyTorch | Deep Learning |
| TensorFlow | Deep Learning |
| Keras | Deep Learning |
| ONNX | Interoperability Format |
| scikit-learn | Classical ML |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Gradient Boosting |
| Transformers (Hugging Face) | NLP / LLMs |
| Flax | Deep Learning (JAX-based) |
| Diffusers | Generative AI |
| MLflow | Model Registry |
| fast.ai | Deep Learning |
| Picklable Model | Custom Python models |
| Detectron | Computer Vision |
| EasyOCR | OCR |
| Ray | Distributed Computing |

**CPU vs GPU Constraints:**
The `@bentoml.service` decorator allows specification of `resources` including `gpu` count and `gpu_type` (e.g., `"nvidia-l4"`). Distributed Services architecture supports pipelining CPU and GPU processing, with individual Services assigned different resource types and scaling independently [47][54].

**LLM Inference Backend Support:**
BentoML serves as an orchestrator for vLLM, LMDeploy, MLC-LLM, TensorRT-LLM, and Hugging Face TGI, providing minimal overhead with consistent REST APIs [55].

---

## 2. Concrete, Comparative Performance Metrics

### 2.1 Cold-Start Latency

Cold-start latency varies dramatically based on model size, hardware, and whether scale-to-zero is enabled. Below are empirically measured times for each platform under comparable conditions.

**First-Ever Cold Start (Scale from Zero, No Caching):**

| Platform | Model Size | Environment | Cold Start Time | Source |
|----------|-----------|-------------|-----------------|--------|
| **Seldon + Triton** | 32B parameter LLM | A100 GPU | ~1.3 seconds | [56] |
| **Seldon + Triton** | Mixtral-141B | A100 GPU | ~3.7 seconds | [56] |
| **Seldon + Triton** | Small XGBoost model | General K8s | ~1 minute | [57] |
| **Ray Serve** | Qwen3-235B-A22B | Optimized setup | ~3.88x reduction vs baseline | [58] |
| **Ray Serve** | General large model | Anyscale Services | <60 seconds node startup | [59] |
| **Ray Serve** | Llama 3 8B (15 GB) | AWS g5.12xlarge, S3 Model Streamer | 23.18 seconds total readiness | [60] |
| **Ray Serve** | Llama 3 8B (15 GB) | AWS g5.12xlarge, GP3 SSD | 35.08 seconds total readiness | [60] |
| **KServe + Knative** | 32B parameter LLM | A100 GPU | ~1.3 seconds (model load only) | [56] |
| **KServe + Knative** | Mixtral-141B | A100 GPU | ~3.7 seconds (model load only) | [56] |
| **KServe + Knative** | Full GPU cold start (node provisioning + image pull + model load) | General K8s | 3-8 minutes | [61] |
| **KServe + Knative** | Small model (helloworld sample) | Image cached | >10 seconds observed | [62] |
| **BentoML** | Llama 3.1 8B (~20+ GB) | Baseline (no optimization) | ~11 minutes | [51] |
| **BentoML** | Llama 3.1 8B (~20+ GB) | After optimization (FUSE + zero-copy) | <30 seconds | [51][52] |
| **BentoML** | General model | BentoCloud platform | 71 seconds | [63] |
| **BentoML** | General model | Vertex AI (comparison) | 148 seconds | [63] |

**Key Distinctions:**
- **First-ever cold start** includes node provisioning (if cluster autoscaler needed), container image pull, model download, CUDA initialization, and weight transfer to GPU memory. This is the longest case.
- **Subsequent cold starts with model caching** (image cached on node, model cached on persistent volume) are faster but still include CUDA initialization and weight transfer: typically 5-15 seconds for large models.
- **Warm starts** (replicas already running): No cold start penalty. Platform overhead is minimal (KServe adds 2-3ms from queue-proxy + activator; Seldon adds ~1-2ms from graph execution; Ray Serve adds negligible overhead in warm state).

**Production Implication for Fraud Detection:**
For a 50M daily transaction workload, scale-to-zero is not recommended. Maintaining a warm pool of 2-3 replicas per model avoids cold starts entirely while providing baseline capacity for normal traffic. Setting `minScale: 1` or `min_replicas: 1` eliminates cold start risk for latency-critical transactions.

### 2.2 Feature Store Integration Latency

Feature store lookup latency is often the dominant contributor to inference latency at high percentiles, particularly for fraud detection where 20-50+ features may need retrieval per transaction.

**Empirically Measured Feature Store Latencies:**

| Configuration | P50 | P95 | P99 | P99.9 | Source |
|--------------|-----|-----|-----|-------|--------|
| **Feast + Redis** (Java gRPC server) | Not published | Not published | Not published | Not published | "Fastest configuration" per Feast benchmarks [64] |
| **Feast + Redis** (first call overhead) | 12-50ms (subsequent) | N/A | ~300ms (first call) | N/A | [65] |
| **Tecton** (no caching, 10K QPS) | 7ms | N/A | 29ms | N/A | [66] |
| **Tecton** (95% cache hit, 10K QPS) | ~1.5ms | N/A | ~8ms | N/A | [66] |
| **Redis** (ElastiCache 7, optimized) | Sub-millisecond | N/A | Up to 71% reduction vs previous | N/A | [67] |
| **Redis** (sub-millisecond tuning) | <500μs achievable | N/A | <500μs | N/A | [68] |
| **Redis Enterprise** (vs DynamoDB) | 3x faster than DynamoDB | N/A | N/A | N/A | [69] |
| **Feedzai** (cache/disk read) | 0.01ms | N/A | 0.01ms | 0.10ms | [70] |
| **Feedzai** (external profile fetch) | Not published | Not published | Not published | "Primary source of latency in high percentiles" | [70] |
| **Swiggy** (ElastiCache) | "Low latency" for 50M QPS | N/A | N/A | N/A | [67] |

**Feature Vector Sizes for Fraud Detection [70]:**
- Feedzai Dataset A: 15 raw categorical + 2 numerical + 2 time features (small)
- Feedzai Dataset B: 53 raw categorical + 4 numerical + 2 time features (medium)
- BRIGHT GNN-based system: 512-dimensional embeddings per entity
- Hazelcast/Feast tutorial: ~10 features per transaction

**Feature Store Overhead Breakdown (Feedzai Production Data) [70]:**
- Total prediction time (RNN system): mean = 4.06ms, P99 = 10.47ms, P99.9 = 42.82ms, P99.99 = 75.90ms, P99.999 = 126.66ms
- Read (cache or disk): mean = 0.01ms, P99 = 0.01ms, P99.9 = 0.10ms, P99.99 = 0.50ms, P99.999 = 3.13ms
- Write disk (async): mean = 0.05ms, P99 = 0.06ms, P99.9 = 0.37ms, P99.99 = 62.37ms, P99.999 = 398.70ms

**The key insight**: At tail percentiles (P99.999), feature retrieval adds 3-400ms depending on the source. External profile fetching is the primary source of high-percentile latency. Optimizations like Tecton's caching reduce P99 from 29ms to 8ms.

**Platform-Specific Integration Overhead:**

| Platform | Native Integration | Integration Approach | Expected Overhead |
|----------|-------------------|---------------------|-------------------|
| **Seldon Core + Triton** | Yes (Feast via Transformer) | Transformer component fetches features before model inference | Feature retrieval latency + graph overhead (~1-2ms) |
| **Ray Serve** | No native integration | Custom logic in deployment handler | Feature retrieval latency + deployment handle overhead (~sub-ms) |
| **KServe** | Yes (native Feast Transformer) | `preprocess()` method in custom Transformer class retrieves features | Feature retrieval latency + queue-proxy overhead (~2-3ms) |
| **BentoML** | No native integration | Custom code in Service methods | Feature retrieval latency + Service call overhead (~sub-ms) |

### 2.3 Burst Handling Metrics

**KServe KPA Panic Mode for 10x Traffic Spikes [35][37]:**

The KPA's two-phase autoscaling handles bursts:
- **Stable Mode (60s window):** Desired pods = observed concurrency / target concurrency per pod. Scales conservatively.
- **Panic Mode (6s window):** Triggered when observed concurrency reaches 200% of target within 6s window. Only scale-ups allowed. Max scale-up rate: 1000.0 (theoretically 1000x in one 2-second evaluation cycle).

**Time to Scale:**
- KPA evaluation period: 2 seconds
- Stable window: 60 seconds for full stabilization
- Panic mode: 6-second window for rapid reaction
- In practice, scale-up time = time for pods to become ready (30s-5min depending on image size, model size, node availability)

**Request Queuing Behavior:**
- Activator buffers requests while pods scale up, protecting system from overload
- Queue-proxy measures queue depth via `kn.serving.queue.depth` metric
- Activator provides smarter load balancing than random, improving p95/p99 latency
- When spare capacity exceeds `target-burst-capacity`, Activator stays in path to buffer bursts
- Activator thrashing (rapid swapping) can cause 503 errors due to Istio certificate sync issues [30]

**Ray Serve Burst Handling [18]:**
- Autoscaler monitors queue sizes via `target_num_ongoing_requests_per_replica`
- `upscale_delay_s` controls response speed (setting to very small values like 0.00001 enables faster scaling)
- Request queuing with async processing (asyncio if handlers are `async def`)
- `max_ongoing_requests` provides backpressure mechanism
- During redeployment with new config, Serve de-provisions and re-provisions replicas, dropping unfinished requests [71]

**Seldon Core Burst Handling [4][5]:**
- HPA with custom Prometheus metrics (infer_rps over 2-min window)
- KEDA for event-based scaling with any Prometheus metrics
- Triton internal batching absorbs spikes at instance level via `max_queue_delay_microseconds`
- Priority-based queues with timeouts for critical requests during bursts

**BentoML Burst Handling [48][49]:**
- Concurrency-based scaling (active requests) for proactive response
- External request queue buffers exceeding requests
- Standby instances matched to incremental scaling steps
- If processing exceeds `max_latency_ms`, HTTP 503 returned (signaling scale-up need)

**Comparative Burst Metrics Summary:**

| Metric | KServe (Knative) | Ray Serve | Seldon Core | BentoML |
|--------|------------------|-----------|-------------|---------|
| Scale-up reaction time | 2-6s (panic mode) | `upscale_delay_s` (configurable) | 2-min sliding window | Concurrency-based (immediate) |
| Full stabilization | ~60s | Depends on downscale delay | ~2 min | Depends on stabilization window |
| Request buffering | Activator (zero/low traffic) | In-memory queue | KEDA-backed queue | External request queue |
| Backpressure mechanism | Queue-proxy (`containerConcurrency`) | `max_ongoing_requests` | Triton queue delay | `max_latency_ms` → 503 |
| Dropped requests during burst | Activator protects → Smarter LB reduces drops | Brief spikes during scale-up | Coordinated omission risk (closed-loop testing) | 503 if latency budget exceeded |
| Drop recovery time | <2s (panic mode re-eval) | Depends on replica startup | Depends on HPA cooldown | Depends on min/max range |
| Scale-to-zero suitability | ❌ (3-8 min cold start) | ❌ (tail latency penalty) | ❌ (1-min cold start) | ❌ (best: 30s cold start) |

---

## 3. Detailed Real-World Company Experiences

### 3.1 Razorpay (India)

**Mitra Platform Architecture [72]:**

Razorpay's Mitra platform is built on a **Kappa+ architecture** with Apache Flink as the core streaming engine. Key technical details:
- **Core Stack:** Apache Flink + Kafka + RocksDB (in-memory state management) + Graph DB + ML model servers
- **Scale:** Processes millions of transactions daily, billions of events in real-time
- **Capability:** Over 100 Flink tasks leveraging complex event processing (CEP) and asynchronous IO
- **Latency:** Generates hundreds of features on the fly and predicts using ML models in milliseconds
- **Training-Serving Separation:** Razorpay explicitly separates training and serving servers for "better resource allocation, network load management, and scalability"
- **Model Support:** XGBoost and NLP models for fraud detection, smart routing, and forecasting
- **Future Plans:** Online learning capabilities at scale

**Bumblebee Multi-Agent Fraud Detection [73][74]:**

Razorpay's Bumblebee evolved through three phases:
1. n8n prototype (validated feasibility, faced scalability issues)
2. Python-based ReAct agent (improved control, encountered token limits)
3. Multi-agent architecture with Planner, Fetcher, and Analyzer agents

**Quantified Results:**
- Processes 12,000 merchant reviews monthly
- 60% reduction in token usage
- Latency reduction from 35 seconds to 8-12 seconds
- Over 99% success rate
- Fraud detection time reduced from hours to seconds

**Engineering Lessons Published:**
"Token budgets are real constraints... prune early, prune often, and never pass raw, unstructured data to LLMs when you can send structured summaries instead."
"Specialization beats generalization at scale."
"Observability is not optional... Investing in structured logging and traceability is critical for debugging and regulatory compliance."

### 3.2 Gojek/GoTo (Indonesia)

**Merlin ML Platform [75][76][77]:**

Gojek's Merlin is built on KServe (then KFServing) + Knative + Istio + MLflow + Kaniko. Key technical details:
- Jupyter notebook-first deployment: "under 10 minutes" from notebook to production
- Framework support: XGBoost, scikit-learn, TensorFlow, PyTorch (standard); custom user-defined models supported
- Deployment strategies: canary, A/B testing, shadow, blue-green via Istio traffic splitting
- Container building: Kaniko inside Kubernetes (no Docker daemon privilege required)
- Autoscaling: Knative serverless auto-scaling with scale-to-zero
- "With Merlin, we aim to do for ML models what Heroku did for Web applications"
- Open-sourced under CaraML (github.com/caraml-dev/merlin)

**Production Operational Metrics:**
- Hundreds of millions of orders per month across 20+ products in 4 countries
- Merlin received "very favorable responses" from data scientists
- Planned enhancements: stream-to-stream inference, gRPC support, improved log management

**Feast Feature Store [78][79][80]:**

Originally developed by Gojek for the Jaeger driver allocation system:
- Scale: Millions of daily customer-driver matches
- Architecture: Two storage layers (warehouse for batch training, low-latency store for online serving) with unified gRPC API
- Redis Cluster: 1TB capacity (explored BigTable as alternative due to Redis scaling limitations)
- Industry adoption: Robinhood, NVIDIA, Discord, Walmart, Shopify, Salesforce, Twitter, IBM, Capital One
- 293+ contributors, 12M downloads, 5,500 Slack community members

**GoSage GNN Fraud Detection [81]:**

GoTo developed GoSage for collusion fraud detection: "Fraud detection is a constant battle, especially in a connected world where bad actors work in coordinated ways to exploit platforms like ours."

**JARVIS Fraud Detection [82]:**
- Processes over 100 million transactions monthly for 20+ million monthly users
- Fraud detection reduced from 30+ minutes to seconds
- ML creates trip attributes for classifying suspicious trips and auto-banning fraudulent drivers

### 3.3 Ant Group (China) - Ray Deployment

**Ant Ray Serving Platform [83][84]:**

Scale metrics:
- 240,000 cores in Model Serving (2022), 3.5x increase year-over-year
- 1.37 million TPS peak throughput during Double 11 sales event
- 4,800 cores processing 325,000 TPS in EventBus (2022), up from 656 cores processing 168,000 TPS (2021)

**Architecture Details:**
- "Each model into an independent Ray service for deployment so that service discovery and traffic of each model will be naturally isolated"
- Two-layer autoscaling: Cougar optimization service for both service instances and Ray clusters
- Model ensembles migrated from independent Java applications to Ray Actors
- Java language support contributed to Ray Serve (enabling cross-language deployment)
- Future goals: 5,000 nodes and 50,000 tasks

### 3.4 PayPal - Fraud Detection Platform

PayPal operates one of the largest production AI/ML environments for fraud detection [85][86]:

- **Transaction Volume:** $451 billion annually
- **Scale:** 8 million operations/second, 60 billion queries daily, 100+ petabytes of risk data
- **Latency:** "Approximately 75% of these decision calls must complete in under 50ms" 
- **Cost Savings:** $500+ million annual profit saved; fraud losses at 17-18 cents per $100 of transaction volume
- **Infrastructure:** Aerospike (hybrid memory), $9 million saved vs pure in-memory, 3x infrastructure efficiency
- **Model Inference:** 250+ features, <0.4ms inference latency on TPU/GPU clusters, 10M transactions/hour
- **Model Management:** Champion-challenger framework with 7-day shadow mode; auto-rollback if precision drops below 99%
- **Online Learning:** Models updated every few hours with labeled feedback from investigations
- **Results:** False positive rates fell 50%, manual reviews dropped 50%, detection accuracy surged 10%

### 3.5 Swiggy (India) - Feature Store at Scale

Swiggy's ML platform uses ElastiCache for Redis as the feature store [67][87]:

- **Scale:** 50 million queries per second for ML feature serving
- **Capability:** "low latency, multiple data structure support, and a highly scalable system"
- **Optimization:** Achieved 2x improvement in latency in Data Science Platform by reducing I/O operations
- **Cost:** Migrated from Redis OSS to Valkey achieving 40% cost reduction in caching
- **Architecture:** Hive for persistent features, Redis for low-latency/online features; 1000s of feature jobs

### 3.6 Xendit (Indonesia) - XenShield

Xendit's XenShield fraud prevention system [88][89]:

- **Impact:** Reduces chargebacks by up to 45% for high-risk merchants
- **Capability:** Evaluates payment risk by analyzing amount, card type, security checks, origin, customer behavior, device, location
- **Risk Levels:** High (red, blocked), Medium (orange, approved), Normal (green, legitimate)
- **Business Impact:** Improves payment acceptance rate by 30%
- **Scale:** Processes $40+ billion, serves 6,000+ clients including Grab and Traveloka
- **Uptime:** 99.999% via infrastructure reliability engineering

### 3.7 Grab (Southeast Asia) - GrabDefence

Grab's fraud prevention platform [90]:
- Uses advanced device intelligence, ML, facial recognition, and real-time liveness detection
- **Fraud Rate:** Maintained at ~0.2%, far below industry average (0.5-1.5% typical)
- **Tech Stack:** Apache Kafka (Confluent Cloud) → Apache Flink/Spark → ML models
- Integration with Amazon Fraud Detector: up to 23% increase in detection performance

### 3.8 PayMongo (Philippines) and Philippine Fintech

PayMongo [91][92]:
- Founded 2019, 51-100 employees, $31M Series B
- PCI-DSS Level 1 provider, BSP Electronic Money Institution license
- Serves 10,000+ businesses
- **No published engineering blog content about ML infrastructure found**

**Philippine Fintech ML Landscape:**
- Maya (formerly PayMaya): 41 million+ users, named Neobank of the Year, Best Digital Fraud Protection Experience
- GCash: Leading digital wallet, no detailed published ML architecture
- BSP-regulated entities rely on existing payment processor fraud tools (Stripe Radar) rather than custom ML infrastructure
- Tookitaki's FinCense: AI-driven AML and fraud detection for Philippine banks

---

## 4. Reconciling Benchmark vs. Production Performance

### 4.1 Why Benchmarks Differ from Production

**1. Feature Retrieval Latency Changes the P95/P99 Picture**

Feature store lookups are the dominant contributor to high-percentile latency in production fraud detection systems. Feedzai's production data shows [70]:
- Mean prediction time: 4.06ms (looks clean in benchmarks)
- P99: 10.47ms (still acceptable)
- P99.9: 42.82ms (degradation visible)
- P99.99: 75.90ms (significant)
- P99.999: 126.66ms (may exceed SLA)

The feedzai paper explicitly states: "The latter [fetching profiles from external systems] is, in fact, the primary source of our product's latency in high percentiles."

**2. Network Overhead Between Microservices**

Serialization/deserialization costs add significant latency in microservice architectures. The Go ML Benchmarks project measured [93]:

| Configuration | Latency per Inference |
|--------------|---------------------|
| Go native (Leaves) | 491 ns |
| Python XGBoost (UDS raw bytes) | 243 μs |
| CGo + XGBoost | 244 μs |
| gRPC over UDS to C++ XGBoost | 367 μs |
| gRPC over UDS to Python XGBoost | 785 μs |
| gRPC over UDS to Python + sklearn | 21.7 ms |
| HTTP/JSON Flask + sklearn + XGBoost | 21.9 ms |

The gap between Go native (491ns) and Python HTTP/JSON (21.9ms) is **44,600x**—almost entirely from serialization, network, and framework overhead, not model computation.

**3. Kubernetes Proxy Overhead in KServe Production vs. Benchmarked "Sub-2ms" Claims**

KServe's "sub-2ms" or "2-3ms" benchmarks refer specifically to queue-proxy and activator software overhead, NOT total end-to-end latency [94]. In production:

**Software Overhead Only (warm state, all components active):**
- Istio sidecar Envoy: ~3ms P50, ~10ms P99 (at 1000 RPS, 16 connections) [95]
- Activator: 2-3ms (when in path) [94]
- Queue-proxy: ~sub-1ms [94]
- **Total software overhead (warm): ~5-6ms P50, ~11-14ms P99**

**Additional Production Costs:**
- Model inference time (variable, 1ms-100ms+)
- Network transit between pods (0.1-1ms per hop)
- Serialization/deserialization (17μs JSON encode, 7μs JSON decode in Python) [93]
- Logging/monitoring overhead (Prometheus scrape, payload logging)
- Feature store lookup (1-29ms depending on caching) [66]

**Total realistic production overhead: 15-50ms P50, 50-150ms P99**

**4. Garbage Collection and Paging Under Sustained Load**

Rippling Engineering documented a **60x gap** between P50 (50ms) and P99 (3 seconds) caused by Python garbage collection [96]:
- Third-generation GC collections paused for "multiple seconds"
- Fix: `gc.freeze()` excluded long-lived objects → average GC time dropped from "over 2 seconds to below 500ms, an 80%+ speedup"
- P99 latency reduced by 40% overall

For JVM-based inference, full GC pauses can freeze all threads for 500ms to several seconds. Java 21's Generational ZGC delivers sub-millisecond pauses, but with high allocation rates (~30 GB/sec), ZGC's CPU-intensive concurrent threads can cause allocation stalls degrading tail latency [97][98].

### 4.2 Production-Adjusted Performance Table

Below is a "production-adjusted" latency table accounting for realistic overheads (feature store calls, serialization, network hops, K8s proxy overhead, logging/monitoring).

**Assumptions:**
- Medium fraud detection model (ensemble: XGBoost + LightGBM + small NN)
- 30 features retrieved from feature store (Redis)
- 3 microservice hops (ingress → transformer → predictor → response)
- 1KB payload size
- Sustained load at 70% CPU utilization

| Platform | P50 Adjusted | P95 Adjusted | P99 Adjusted | P99.9 Adjusted | Model Inference Only (Benchmark) |
|----------|-------------|-------------|-------------|----------------|----------------------------------|
| **Seldon Core + Triton** | 8-15ms | 20-35ms | 35-60ms | 80-200ms | ~2-5ms (GPU) |
| **Ray Serve** | 10-20ms | 25-45ms | 40-75ms | 90-250ms | ~2-8ms (varies) |
| **KServe (Knative, warm)** | 12-25ms | 30-50ms | 50-90ms | 100-300ms | ~3-10ms (includes 2-3ms proxy) |
| **KServe (Standard)** | 10-20ms | 25-40ms | 40-70ms | 90-250ms | ~1-5ms (no proxy overhead) |
| **BentoML** | 10-22ms | 28-48ms | 45-85ms | 95-280ms | ~2-8ms (adaptive batching) |

**Notes on Production Adjustment Factors:**
- **Seldon Core + Triton**: Best raw performance due to GPU-accelerated tree models (FIL backend). Kafka streaming adds slight baseline overhead but provides robust audit trails. Graph execution model adds ~1-2ms.
- **Ray Serve**: Python-native architecture eliminates some serialization overhead but adds deployment handle routing overhead. Recent HAProxy + gRPC optimizations (March 2026) reduce latency by up to 88% [17].
- **KServe (Knative)**: Queue-proxy + activator + Istio sidecar add ~3-6ms P50, ~10-14ms P99 overhead. Additional cold-start risk if scale-to-zero enabled. Standard mode avoids this overhead.
- **KServe (Standard)**: No queue-proxy or activator. Only Istio sidecar overhead (~3ms P50, ~10ms P99). Better for latency-critical fraud detection.
- **BentoML**: Adaptive batching adds latency proportional to batch wait times. Under sustained load, batching improves throughput but may increase P99 due to batch formation delay.

**The "5-10x" Rule of Thumb:**
Based on the Whatnot experience (5.8x gap), Rippling's GC issues (60x gap at P99), and general microservice overhead, production P99 latency is typically **5-10x higher** than benchmark inference-only latencies for well-optimized systems, and **10-50x higher** for systems with identified bottlenecks (GC, serialization, feature store).

---

## 5. Regional Burst Capacity and Cost Modeling

### 5.1 Infrastructure Assumptions

**Cloud Pricing Data (On-Demand, Linux, USD/hr) [99][100][101][102][103][104][105]:**

| Instance Type | Specs | ap-south-1 (Mumbai) | ap-southeast-1 (Singapore) | ap-southeast-3 (Jakarta) |
|--------------|-------|---------------------|---------------------------|--------------------------|
| **g5.xlarge** | 4 vCPU, 16GB, 1×A10G | $1.208/hr | ~$1.30/hr (est.) | $1.408/hr |
| **g5.2xlarge** | 8 vCPU, 32GB, 1×A10G | ~$1.46/hr (est.) | ~$1.56/hr (est.) | ~$1.70/hr (est.) |
| **g4dn.xlarge** | 4 vCPU, 16GB, 1×T4 | $0.579/hr | ~$0.62/hr (est.) | ~$0.70/hr (est.) |
| **g4dn.4xlarge** | 16 vCPU, 64GB, 1×T4 | ~$1.32/hr (est.) | ~$1.42/hr (est.) | ~$1.56/hr (est.) |
| **c6i.4xlarge** | 16 vCPU, 32GB (CPU only) | $0.680/hr | $0.784/hr | ~$0.70/hr (est.) |
| **c6i.8xlarge** | 32 vCPU, 64GB (CPU only) | $1.360/hr | ~$1.46/hr (est.) | $1.333/hr |
| **EKS Control Plane** | Per cluster | $0.10/hr | $0.10/hr | $0.10/hr |

**GCP Alternative Pricing (N2 Standard) [106]:**
- n2-standard-8 (8 vCPU, 32GB): asia-south1 (Mumbai) $0.467/hr, asia-southeast1 (Singapore) $0.479/hr, asia-southeast2 (Jakarta) $0.522/hr

**Workload Profile for 50M Daily Transactions:**
- Average throughput: ~578 transactions/second (50M / 86,400s)
- Peak hour (3x average): ~1,735 transactions/second
- Inference time per transaction (ensemble): ~3-8ms (GPU-accelerated)
- Feature retrieval per transaction: 30 features × ~0.5ms each = ~15ms total
- Total latency budget: <100ms P99 (fraud detection requirement)

### 5.2 Scenario A: Normal Peak Hour (3x Average Load)

**Parameters:** ~5M transactions/hour = ~1,389 transactions/second peak

**Node Requirements:**

| Platform | CPU Nodes (tree models) | GPU Nodes (NN) | Total Cost/hr (Mumbai) |
|----------|------------------------|----------------|----------------------|
| **Seldon + Triton** | 2 × c6i.4xlarge | 1 × g5.xlarge | $2.568/hr |
| **Ray Serve** | 2 × c6i.4xlarge | 1 × g5.xlarge | $2.568/hr |
| **KServe** | 2 × c6i.4xlarge | 1 × g5.xlarge | $2.568/hr |
| **BentoML** | 2 × c6i.4xlarge | 1 × g5.xlarge | $2.568/hr |

**Time to Scale Up:** All platforms can scale within 2-5 minutes from warm pool (assuming cluster autoscaler has spare capacity). No cold start risk if min replicas ≥ 2.

**SLA Violation Risk:** Low (<1%). All platforms can handle this load with warm replicas.

**Monthly Cost (720 hours):** ~$1,850/month (single region) × 3 regions = ~$5,550/month total infrastructure

### 5.3 Scenario B: Major Sales Event (10x Normal Load)

**Parameters:** ~17M transactions/hour = ~4,722 transactions/second peak (e.g., Indian festival Diwali, Indonesian "Harbolnas," Philippine Christmas)

**Node Requirements:**

| Platform | CPU Nodes | GPU Nodes | Total Cost/hr |
|----------|-----------|-----------|---------------|
| **Seldon + Triton** | 6 × c6i.4xlarge | 3 × g5.xlarge | $7.704/hr (Mumbai: $4.08 + $3.624) |
| **Ray Serve** | 6 × c6i.4xlarge | 3 × g5.xlarge | $7.704/hr |
| **KServe** | 6 × c6i.4xlarge | 3 × g5.xlarge | $7.704/hr |
| **BentoML** | 6 × c6i.4xlarge | 3 × g5.xlarge | $7.704/hr |

**Cost Variation by Region:**
- Mumbai (ap-south-1): ~$7.70/hr
- Singapore (ap-southeast-1): ~$8.80/hr
- Jakarta (ap-southeast-3): ~$9.20/hr

**Time to Scale Up:**
- **KServe (Knative):** Fastest—panic mode triggers within 6 seconds, can scale from 2 to 20 replicas in ~2 evaluation cycles (4 seconds) + pod startup (30-60s) = ~35-65s
- **Ray Serve:** Upscale delay configurable to near-zero, replica scheduling adds ~5-10s overhead + pod startup = ~35-70s
- **Seldon Core:** HPA with 2-min sliding window → slower to react (60-120s to first scale event) + pod startup = ~90-180s
- **BentoML:** Concurrency-based scaling provides immediate signal, but Kubernetes HPA evaluation adds ~15s delay + pod startup = ~45-75s

**Request Queuing Behavior:**
- **KServe:** Activator buffers requests during scale-up. At 10x surge, if buffer capacity exceeded, requests may queue (queue depth monitored via `kn.serving.queue.depth`). Smarter load balancing via Activator reduces dropped requests.
- **Ray Serve:** In-memory queue with backpressure. May drop requests if `max_ongoing_requests` exceeded during scale-up.
- **Seldon Core:** Triton's internal batching absorbs surge at instance level. Priority queues protect critical transactions. KEDA-backed scaling prevents overload.
- **BentoML:** External request queue buffers exceeding concurrency. Returns 503 if `max_latency_ms` exceeded.

**Dropped Request Rates (Estimated):**
- Well-configured system with warm pool + proactive scaling: <0.1% dropped
- System starting from min replicas: 1-5% dropped during scale-up window
- System with scale-to-zero (idle → 10x): 5-15% dropped or severely delayed

**SLA Violation Risk:** Moderate (5-15%) for systems without proactive burst preparation. Low (<2%) with pre-warmed pool at 3-4x normal peak capacity.

**Recommended Strategy:** Pre-scale to 3x normal peak 30 minutes before the sales event starts. Monitor queue depth metrics and scale proactively.

**Cost for 8-hour Sales Event:**
- Pre-warming (2 hours): ~$15.40
- Peak (6 hours): ~$46.20
- Total per region: ~$61.60
- All three regions: ~$185

### 5.4 Scenario C: Regional Spike (One Country Surges)

**Parameters:** Indonesia (Jakarta) experiences 10x surge while India and Philippines remain at normal peak.

**Architecture:** Each region deploys independently per data sovereignty requirements. Inter-region traffic is minimal (only for reporting/analytics).

| Country | Load | Nodes Required | Cost/hr |
|---------|------|----------------|---------|
| **India (Mumbai)** | Normal peak (3x) | 2 × c6i.4xlarge + 1 × g5.xlarge | $2.57/hr |
| **Indonesia (Jakarta)** | Surge (10x) | 6 × c6i.4xlarge + 3 × g5.xlarge | $9.20/hr (Jakarta pricing) |
| **Philippines (Singapore)** | Normal peak (3x) | 2 × c6i.4xlarge + 1 × g5.xlarge | $2.97/hr (Singapore pricing) |
| **Total** | | | **$14.74/hr** |

**Time to Scale Up (Jakarta):**
- Jakarta's ap-southeast-3 region is newer—some instance types may have limited availability
- GCP asia-southeast2 (Jakarta) only supports T4 GPUs (no L4, A100, V100) [107]
- Azure Indonesia Central region launched May 2025—instance availability may be limited [108]
- Scale-up time may be 2-3x longer than Mumbai/Singapore due to regional capacity constraints

**Risk of SLA Violation:** Higher in Jakarta due to:
- Limited GPU instance availability
- Newer region with potentially less capacity headroom
- Higher base latency to reach Singapore-fallback infrastructure
- Regional cloud provider constraints (GCP only has T4 in Jakarta; no L4 or A100)

**Mitigation Strategy:**
- Maintain warm pool at 4x normal peak in Jakarta during known sale periods
- Configure cross-region failover to Singapore for overflow during extreme bursts
- Use AWS/GCP/Azure multi-cloud to hedge against single-provider capacity constraints in Jakarta

---

## 6. Regulatory and Compliance Implications for Platform Choice

### 6.1 India: RBI Regulations

**FREE-AI Framework (August 2025) [109][110]:**

The RBI's Framework for Responsible and Ethical Enablement of Artificial Intelligence (FREE-AI) lays down seven "Sutras" (principles):
1. Trust is the Foundation
2. People First (human oversight, override capability)
3. Innovation over Restraint
4. Fairness and Equity (bias testing, financial inclusion)
5. Accountability (entity deploying AI remains fully accountable)
6. Understandable by Design (explainability, disclosures)
7. Safety, Resilience, and Sustainability

**Operationalized through 26 recommendations including:**

**Board-Approved AI Policy:** Risk classification framework for use-cases (low/medium/high based on customer impact)
**Data Lifecycle Governance:** Compliance with DPDP Act throughout data lifecycle
**AI System Governance:** Full model lifecycle governance (design, development, deployment, decommissioning, documentation, validation, monitoring)
**AI Audit Framework:** Comprehensive audit covering data inputs, model/algorithm, decision outputs. Internal audit + third-party audits for high-risk AI + biennial review
**Incident Reporting:** Dedicated AI incident reporting framework for REs and FinTechs
**Explainability:** Use of interpretation tools (SHAP, LIME required—RBI survey found only 15% using them)
**Consumer Protection:** Transparency, fairness, accessible recourse
**Red Teaming:** Structured processes for entire AI lifecycle

**Data Residency:**
2018 RBI directive requires payment data storage in India. DPDP Act 2023 adopts "blacklist" approach—data can flow freely except to blacklisted countries. Significant Data Fiduciaries face additional restrictions [111][112].

**Platform Implications for India Operations:**

| Requirement | Seldon Core + Triton | Ray Serve | KServe | BentoML |
|-------------|---------------------|-----------|--------|---------|
| **Audit Trail / Inference Logging** | Native (Kafka streaming provides natural audit log) | Via integration (no native audit) | Via integration (no native audit) | Via integration (no native audit) |
| **Model Versioning** | Native (inference graphs) | Native (Deployment Graph) | Native (K8s revisions) | Native (model store) |
| **Explainability (SHAP/LIME)** | Native (Alibi) | Via integration | Native (Alibi, AIX360) | Via integration |
| **Drift Detection** | Native (alibi-detect) | Via integration | Via integration | Limited |
| **Canary / A-B Testing** | Native (experiments) | Native (YAML patterns) | Native (canary rollout) | Limited |
| **Rollback / Fallback** | Native (traffic routing) | Native (deployment config) | Native (K8s rollback) | Manual |
| **Data Residency (Multi-region)** | Yes (any K8s) | Yes (distributed) | Yes (any K8s) | Yes (manual) |
| **Compliance Readiness** | High | Medium | High | Low-Medium |

**Key Constraints:**
- **RBI's AI Audit Framework** requires "comprehensive AI audit covering data inputs, model and algorithm, and the decision outputs." Only Seldon Core and KServe provide native explainability and drift detection, which are essential for generating audit artifacts.
- **Incident reporting within 2-6 hours** (existing RBI framework) requires robust rollback capabilities. KServe's automatic revision tracking and instant traffic pinning to `PreviousRolledoutRevision` provides the fastest rollback.
- **FREE-AI's Red Teaming requirement** benefits from Seldon Core's inference graphs and A/B testing infrastructure, allowing safe shadow deployment of model variants.
- **Model governance** covering the full lifecycle is natively supported by KServe's revision system and Seldon Core's experiment framework.

### 6.2 Indonesia: Bank Indonesia Regulations

**PBI 10/2025 (Effective March 31, 2026) [113][114]:**

- **Domestic Data Processing:** Payment transactions must be processed domestically within Indonesia
- **PSP Classification:** Payment Service Providers classified as Main or Non-Main based on TIKMI criteria (Transaction volume/value, Interconnection complexity, Competence, Risk Management, IT Infrastructure)
- **Capital Requirements:** Minimum 10% capital adequacy ratio with surcharges of 1.5-5% depending on provider type
- **Ownership Restrictions:** Minimum domestic share ownership and domestic control; Single Ownership Policy
- **Supporting Providers:** Critical and Important providers face registration and audit obligations
- **Non-Compliance:** Fines, activity suspension, license revocation

**Data Localization (GR71/2019) [115]:**
- Electronic System Operators (ESOs) must register with MCIT
- Cross-border data transfers require written consent in Bahasa Indonesia
- Non-compliance: fines up to 2% of annual revenue
- Indonesia's increasing data restrictiveness (2013-2018) reduced trade output by 9.1%

**OJK (Financial Services Authority) [116]:**
- POJK 11/2022 regulates data placement for banks
- POJK 27/2024 introduces specific infrastructure requirements for digital financial asset trading

**Cloud Provider Availability in Jakarta [117]:**
- AWS ap-southeast-3 (Jakarta): Available, some instance types limited
- GCP asia-southeast2 (Jakarta): T4 GPUs only (no L4, A100, V100, H100)
- Azure Indonesia Central: Launched May 2025

**Platform Implications for Indonesia Operations:**
- **Domestic processing requirement** mandates deploying inference infrastructure within Indonesia. All four platforms can run on any Kubernetes cluster in Jakarta, but KServe and Seldon Core have the most mature multi-region deployment patterns.
- **GCP's limited GPU availability** in Jakarta (T4 only) constrains GPU-accelerated inference. For neural network components, consider AWS (g5 with A10G available) or Azure (NC-series with T4/A100).
- **Internal PI classification** (Main PSPs) carries stricter audit and risk management requirements. Seldon Core's data-centric architecture with Kafka-based audit trails and KServe's comprehensive monitoring ecosystem are best positioned to meet these requirements.

### 6.3 Philippines: BSP Regulations

**BSP AI Regulation (Expected Q1-Q2 2026) [118]:**

BSP Senior Director Melchor Plabasan: "AI regulation that you can expect from the BSP is generally more clarificatory and at the same time, covering risk—the 20% of the risk that are not yet covered particularly on management of bias, accuracy and ethical use of AI."

**Project SAPIENS (BSP Thematic Review) [119]:**
Key findings:
- 44% of responding institutions have deployed AI in production
- 60% explicitly included AI/ML in their roadmap
- Key use cases: fraud detection, e-KYC, predictive analytics, hyper-personalization, credit risk scoring
- Governance and ethical frameworks lag behind data, talent, and tools development
- "The lack of clear AI governance leads to ad hoc initiatives that could result in resource wastage and ethical issues"
- Specific AI security risks: adversarial attacks and data poisoning

**Manual of Regulations for Payment Systems (MORPS) [120][121]:**
- Circular No. 1191 (March 14, 2024): Comprehensive codification of payment system regulations
- Aligned with Principles for Financial Market Infrastructures (PFMI)
- Operators of Designated Payment Systems must secure certification
- Governance, licensing, settlement, operational risk, AML, consumer protection

**Data Privacy Act of 2012 (RA 10173) [118]:**
- Governs data protection; no explicit data localization mandate across all sectors
- BSP regulations for financial institutions may include local processing requirements

**Platform Implications for Philippines Operations:**
- **No local cloud region:** AWS, GCP, and Azure lack dedicated Philippine regions. Infrastructure must operate from Singapore (ap-southeast-1) with edge considerations.
- **Data Privacy Act compliance** requires adequate security measures. All four platforms support encryption, mTLS (Istio), and secure storage.
- **BSP's expectation of proportionate AI policy frameworks** favors platforms with built-in governance features. Seldon Core's Alibi explainability and KServe's AIX360 integration directly address bias management and accuracy requirements.
- **Adversarial attack prevention** (noted as a BSP concern) benefits from platforms supporting model monitoring and outlier detection. Seldon Core's outlier detectors in inference graphs natively address this.

### 6.4 Platform-Regulation Mapping Summary

| Regulatory Requirement | Best-Suited Platform(s) | Key Features |
|------------------------|------------------------|--------------|
| **Audit trail for inference** | Seldon Core (native) | Kafka-based data streaming provides natural audit log |
| **Model versioning and lineage** | KServe, Seldon Core | Native revision tracking (KServe); inference graph versioning (Seldon) |
| **Explainability (SHAP/LIME)** | KServe, Seldon Core | Native Alibi (both) + AIX360 (KServe) |
| **Bias and fairness testing** | KServe, Seldon Core | Native drift detection + explainability |
| **Canary / shadow deployment** | KServe, Seldon Core | Native traffic splitting + automatic rollback (KServe) |
| **Rapid rollback (2-6 hr notification)** | KServe | Instant traffic pinning to PreviousRolledoutRevision |
| **Multi-region data residency** | KServe, Seldon Core | Mature Kubernetes multi-cluster deployment patterns |
| **Domestic processing (Indonesia)** | All (with K8s) | Run on any cloud provider in Jakarta |
| **GPU constraints (Jakarta T4 only)** | Seldon + Triton (best GPU support for limited hardware) | FIL backend supports tree models on T4 GPUs efficiently |
| **BSP bias/accuracy regulation** | KServe, Seldon Core | Native explainability + drift detection |
| **Adversarial attack prevention** | Seldon Core | Outlier detectors in inference graphs |
| **Security certifications (PCI-DSS, ISO)** | All (with K8s + Istio) | Encryption, mTLS, RBAC supported by all |

---

## 7. Final Recommendations

### 7.1 Primary Recommendation: KServe on Kubernetes (Standard RawDeployment Mode)

**Rationale:**

KServe emerges as the strongest choice for this specific use case for the following reasons:

1. **Regional Production Validation:** Gojek/GoTo's Merlin platform (KServe + Knative + Istio) processes hundreds of millions of orders monthly across 20+ products in 4 countries, including Indonesia. This provides a proven template for Southeast Asian fintech operations at scale.

2. **Ensemble Capability via InferenceGraph:** The Sequence, Switch, Ensemble, and Splitter routing nodes natively support the exact architecture required for combining XGBoost/LightGBM on CPU with neural networks on GPU in a single orchestrated pipeline.

3. **Feature Store Integration:** KServe's native Feast Transformer integration provides the most well-documented, production-validated path for online feature retrieval. The IBM-Gojek demonstration of Feast + KServe + ModelMesh for multi-region driver ranking validates this architecture [122].

4. **Compliance and Governance:** Automatic revision tracking, canary rollouts with instant rollback to `PreviousRolledoutRevision`, and integration with Explainability tools (Alibi, AIX360) directly address RBI FREE-AI requirements (AI audit framework, explainability, model governance), Bank Indonesia's risk management requirements, and BSP's upcoming AI regulations.

5. **Performance Optimization:** For latency-critical fraud detection, **Standard RawDeployment mode** eliminates Knative queue-proxy/activator overhead (saving ~3-6ms P50, ~10-14ms P99) while retaining all MLOps capabilities. ModelMesh enables packing up to 20,000 models on modest hardware if needed.

6. **Multi-Cloud Flexibility:** Can be deployed on any Kubernetes cluster across AWS (Mumbai, Singapore, Jakarta), GCP (limited GPU in Jakarta), and Azure (new Indonesia Central region), satisfying data residency requirements.

**Deployment Mode Recommendation:**
- **India (Mumbai) and Philippines (Singapore):** Standard RawDeployment mode for lowest latency (no Knative overhead). Use Gateway API for ingress.
- **Indonesia (Jakarta):** Standard RawDeployment mode with AWS g5 instances for GPU workloads. Consider GCP asia-southeast2 only for T4-compatible tree models.

### 7.2 Secondary Recommendation: Seldon Core with Triton Inference Server

**When to Choose Seldon Core:**

Seldon Core should be the primary choice if:
- **Maximum GPU-accelerated tree model performance is critical:** The FIL backend achieves sub-2ms p99 latency with 20x CPU throughput improvement for XGBoost/LightGBM. This directly impacts fraud detection speed.
- **Comprehensive audit trails are a top regulatory requirement:** Kafka-based data streaming provides natural, persistent audit logs for all inference requests, directly satisfying RBI's audit framework and incident reporting requirements.
- **Complex inference graphs are needed:** Seldon's graph execution model (TRANSFORMERS, ROUTERS, COMBINERS, OUTLIER_DETECTORS) provides the most sophisticated multi-model orchestration of any platform.
- **Existing Kafka infrastructure exists:** Seldon Core 2 requires Kafka. If the startup already uses Kafka for event streaming, the incremental overhead is minimal.

**Trade-offs:**
- Higher operational complexity (Kafka management, sequential version upgrades)
- Longer setup time (3-6 months for full MLOps stack)
- Business Source License (not fully open source for commercial use)

### 7.3 When to Consider Ray Serve

Ray Serve is appropriate if:
- **Developer productivity is the top priority:** Python-native, deploy in under 20 lines of code, no external dependencies (no Kafka required)
- **Cost efficiency is critical:** 30-70% inference cost reduction reported by users; 50% reduction for Samsara
- **Fractional GPU utilization is important:** Fractional GPU allocation (e.g., `num_gpus=0.2`) maximizes hardware utilization for small neural networks
- **Ant Group-level scale is required:** 240,000 cores, 1.37M TPS validated at Ant Group

**Not recommended as primary platform for regulatory-heavy environments** due to:
- No native A/B testing or canary deployments (requires external tools)
- No native model versioning/rollback (requires external GitOps)
- KubeRay's complete cutover deployment strategy (no incremental canary)
- Higher integration effort for compliance features

### 7.4 When to Consider BentoML

BentoML is appropriate for:
- **Rapid prototyping and iteration:** Dev-to-production in minutes, 90% cost reduction reported by fintech case study
- **Startups without existing Kubernetes expertise:** Can start with `bentoml serve` locally, graduate to Yatai for K8s
- **Teams prioritizing Python-centric workflows:** Best developer experience for custom model serving

**Not recommended as primary platform for this use case** due to:
- Limited native autoscaling (depends on runtime)
- No native gRPC support in open-source version
- Manual CI/CD pipeline adjustments required
- Smaller community and less mature than KServe/Seldon

### 7.5 Implementation Roadmap

**Phase 1 (Months 1-3): India Primary Deployment**
- Deploy KServe (Standard RawDeployment) on AWS ap-south-1 (Mumbai)
- Configure Feast feature store with ElastiCache for Redis (online store) and S3 (offline store)
- Implement InferenceGraph for ensemble fraud detection (XGBoost/LightGBM on CPU, NN on GPU)
- Set up GitOps with Flux CD for consistent deployment and rollback
- Configure Prometheus + Grafana for monitoring, Evidently AI for drift detection
- Establish audit logging (structured logs to S3/Elasticsearch for RBI compliance)

**Phase 2 (Months 4-6): Indonesia and Philippines Expansion**
- Deploy Indonesia infrastructure on AWS ap-southeast-3 (Jakarta) with g5 instances for GPU workloads
- Deploy Philippines infrastructure on AWS ap-southeast-1 (Singapore) with edge caching
- Configure Feast multi-region deployment (local online stores per region, shared offline store)
- Implement cross-region failover for overflow during burst scenarios
- Establish regional audit trails and compliance reporting

**Phase 3 (Months 7-12): Advanced MLOps and Compliance**
- Implement automated retraining pipelines triggered by drift detection
- Establish champion-challenger framework with 7-day shadow mode (PayPal pattern)
- Implement RBAC-powered model governance board approvals
- Conduct AI audit framework implementation (RBI FREE-AI compliance)
- Set up BCP drills for fallback workflows
- Establish real-time (hours) model updates from labeled feedback

**Ongoing:**
- Maintain PCI-DSS Level 1 compliance across all regions
- Conduct biennial AI audit framework review (RBI requirement)
- Participate in regulatory sandbox programs (RBI AI Innovation Sandbox, BSP regulatory sandbox)
- Monitor cloud provider capacity in Jakarta for GPU availability and scale planning

---

## Sources

[1] Service Orchestrator - Seldon Core 1: https://docs.seldon.ai/seldon-core-1

[2] Seldon Core 2 Architecture: https://docs.seldon.ai/seldon-core-2

[3] Seldon Ingress Controller: https://docs.seldon.ai/seldon-core-1/ingress.html

[4] Seldon Core 2 Autoscaling: https://docs.seldon.ai/seldon-core-2/user-guide/scaling

[5] Using HPA for Autoscaling - Seldon Core 2: https://docs.seldon.ai/seldon-core-2/user-guide/scaling/hpa.html

[6] Knative Scale to Zero Configuration: https://knative.dev/docs/serving/autoscaling/scale-to-zero

[7] Batchers - NVIDIA Triton: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batchers.html

[8] AWS Blog - Hyperscale Performance with Triton on SageMaker: https://aws.amazon.com/blogs/machine-learning/hyperscale-performance-with-triton-on-sagemaker

[9] Triton Autoscaling Discussion: https://github.com/triton-inference-server/server/issues/5401

[10] Triton Backend-Platform Support Matrix: https://github.com/triton-inference-server/server

[11] TensorFlow Backend Dropped in Triton 25.03: https://github.com/triton-inference-server/server/discussions/8239

[12] FIL Backend README: https://github.com/triton-inference-server/fil_backend

[13] Ray Serve Architecture: https://docs.ray.io/en/latest/serve/architecture.html

[14] Ray Serve Request Routing: https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html

[15] Ray Serve Faster First Token with Custom Routing: https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing

[16] Multi-model Composition with Ray Serve Deployment Graphs: https://www.anyscale.com/blog/multi-model-composition-with-ray-serve-deployment-graphs

[17] Major Upgrades to Ray Serve (March 2026): https://www.anyscale.com/blog/ray-serve-inference-lower-latency-higher-throughput-haproxy

[18] Ray Serve Autoscaling Guide: https://docs.ray.io/en/latest/serve/autoscaling-guide.html

[19] AutoscalingConfig API Reference: https://docs.ray.io/en/latest/serve/api/doc/ray.serve.config.AutoscalingConfig.html

[20] Advanced Ray Serve Autoscaling: https://docs.ray.io/en/latest/serve/advanced-guides/advanced-autoscaling.html

[21] Ray Serve Autoscaling Bug: https://github.com/ray-project/ray/issues/24793

[22] Ray Autoscaler Configuration: https://docs.ray.io/en/latest/cluster/vms/user-guides/configuring-autoscaling.html

[23] Autoscaling Ray Service with KEDA: https://discuss.ray.io/t/autoscaling-ray-service-with-keda/13714

[24] Ray Serve Custom Metrics Autoscaling: https://github.com/ray-project/ray/issues/51632

[25] Production LLM Serving with Ray Serve: https://www.linkedin.com/pulse/challenge-production-llm-serving-ray-serve-vinay-jayanna-08syc

[26] Ray Serve Sequential Execution Issue: https://discuss.ray.io/t/ray-serve-is-executing-the-requests-sequentially/12477

[27] Ray Serve Model Multiplexing: https://docs.ray.io/en/latest/serve/model-multiplexing.html

[28] KServe Control Plane Architecture: https://kserve.github.io/website/docs/concepts/architecture/control-plane

[29] KServe Deployment Modes: https://kserve.github.io/website/latest/get_started

[30] Knative Request Flow: https://knative.dev/docs/serving/request-flow

[31] KServe InferenceGraph Documentation: https://kserve.github.io/website/docs/concepts/resources/inferencegraph

[32] KServe InferenceGraph Routing Types: https://kserve.github.io/website/docs/model-serving/inferencegraph/overview

[33] Serving ML Models at Scale Using KServe (Bloomberg): https://www.youtube.com/watch?v=sE_A54T2n6k

[34] Knative Autoscaling Overview: https://knative.dev/docs/serving/autoscaling

[35] KPA-Specific Autoscaling Configuration: https://knative.dev/docs/serving/autoscaling/kpa-specific

[36] KServe Autoscaling README: https://github.com/kserve/kserve/blob/master/docs/samples/autoscaling/README.md

[37] config-autoscaler ConfigMap Example: https://gist.github.com/devops-school/645a0f0659e781111eaebd26ed921485

[38] KServe v0.15 Release Notes: https://github.com/kserve/kserve/releases

[39] KServe KEDA Integration Issue: https://github.com/kserve/kserve/issues/3561

[40] vLLM Production Stack Autoscaling with KEDA: https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/autoscaling-keda.html

[41] KServe Predictors Documentation: https://kserve.github.io/website/latest/modelserving/v1beta1/predictors

[42] Knative Traffic Splitting: https://knative.dev/docs/getting-started/first-traffic-split

[43] BentoML Services Documentation: https://docs.bentoml.com/en/latest/build-with-bentoml/services.html

[44] BentoML gRPC Status Discussion: https://github.com/orgs/bentoml/discussions/3635

[45] BentoML Blog - 3 Reasons for gRPC: https://bentoml.com/blog/3-reasons-for-grpc

[46] BentoML Adaptive Batching: https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html

[47] BentoML Distributed Services: https://docs.bentoml.com/en/latest/build-with-bentoml/distributed-services.html

[48] BentoML Concurrency and Autoscaling: https://docs.bentoml.com/en/latest/scale-with-bentocloud/scaling/autoscaling.html

[49] BentoML - Scaling AI Models Like You Mean It: https://www.bentoml.com/blog/scaling-ai-model-deployment

[50] Kubernetes Horizontal Pod Autoscaling: https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale

[51] BentoML - 25x Faster Cold Starts for LLMs on Kubernetes: https://www.bentoml.com/blog/25x-faster-cold-starts-for-llms-on-kubernetes

[52] How BentoML cuts LLM cold starts to under 30 seconds: https://www.linkedin.com/posts/bentoml_llm-genai-fuse-activity-7317947355519954944-ZcgT

[53] BentoML Framework APIs Reference: https://docs.bentoml.com/en/latest/reference/bentoml/frameworks/index.html

[54] BentoML 1.2 Release Blog: https://www.bentoml.com/blog/introducing-bentoml-1-2

[55] BentoML - Benchmarking LLM Inference Backends: https://www.bentoml.com/blog/benchmarking-llm-inference-backends

[56] Reddit - Cold Start Latency for Large Models: https://www.reddit.com/r/MachineLearning/comments/1n01odu/d_cold_start_latency_for_large_models_new

[57] General Model Serving Cold Start: https://billtcheng2013.medium.com/machine-learning-model-serving-251925111503

[58] Ray 2.55.1 Benchmarks: https://docs.ray.io/en/latest/serve/benchmarks

[59] Ray Serve with Anyscale: https://www.anyscale.com/product/library/ray-serve

[60] Reducing Cold Start Latency with NVIDIA Run:ai Model Streamer: https://developer.nvidia.com/blog/reducing-cold-start-latency-for-llm-inference-with-nvidia-runai-model-streamer

[61] KServe Production Deployment Checklist: https://kserve.github.io/website/latest/admin/production

[62] Knative Cold Start Issue: https://github.com/knative/serving/issues/1297

[63] Comparing BentoML and Vertex AI: https://www.bentoml.com/blog/comparison-between-vertex-ai-and-bentoml

[64] Feast Benchmarks: https://feast.dev/blog/feast-benchmarks

[65] Feast First Online Feature Request Issue: https://github.com/feast-dev/feast/issues/2952

[66] Tecton Caching Feature Views: https://docs.tecton.ai/docs/reading-feature-data

[67] Build an Ultra-Low Latency Online Feature Store with ElastiCache: https://aws.amazon.com/blogs/database/build-an-ultra-low-latency-online-feature-store-for-real-time-inferencing-using-amazon-elasticache-for-redis

[68] Redis Sub-Millisecond Latency Design: https://oneuptime.com/blog/post/2026-03-31-redis-design-sub-millisecond-latency

[69] Tecton + Redis Enterprise Benchmark: https://redis.io/blog/fast-machine-learning-with-tecton-and-redis-enterprise-cloud

[70] Feedzai Interleaved Sequence RNNs for Fraud Detection: https://research.feedzai.com/wp-content/uploads/2022/08/Branco_Interleaved_RNNs_KDD2020.pdf

[71] Ray Serve Autoscaling Parameter Update Issue: https://github.com/ray-project/ray/issues/21017

[72] Razorpay - Data Science at Scale Using Apache Flink: https://razorpay.com/unfiltered/data-science-at-scale-using-apache-flink

[73] Bumblebee: Multi-Agent AI Fraud Detection at Razorpay: https://engineering.razorpay.com/meet-bumblebee-the-multi-agent-ai-architecture-that-changed-fraud-detection-at-razorpay-c2b6d5704f51

[74] Bumblebee: Agentic AI Flagging Risky Merchants: https://dev.to/razorpaytech/meet-bumblebee-agentic-ai-flagging-risky-merchants-in-under-90-seconds-2nlf

[75] Merlin: Making ML Model Deployments Magical: https://www.gojek.io/blog/merlin-making-ml-model-deployments-magical

[76] Gojek ML Platform - ZenML Database: https://www.zenml.io/mlops-database/gojek-gojeks-ml-platform-merlin-jupyter-first-ml-model-deployment-platform-on-kubernetes-with-kfserving-mlflow-canary-an

[77] CaraML Merlin GitHub: https://github.com/caraml-dev/merlin

[78] Feast: Bridging ML Models and Data: https://www.gojek.io/blog/feast-bridging-ml-models-and-data

[79] Gojek Feast on Medium: https://medium.com/gojekengineering/feast-bridging-ml-models-and-data

[80] Feast Official Site: https://feast.dev

[81] GoSage: Graph Neural Networks for Fraud Detection: https://www.linkedin.com/posts/gotogroup_gosage-how-we-detect-fraud-syndicates-at-activity-7271784112967835648-dvHX

[82] Gojek JARVIS Fraud Detection Case Study: https://afi.io/case_studies/gojek

[83] How Ant Group Uses Ray for Large-Scale Online Serverless Platform: https://www.anyscale.com/blog/how-ant-group-uses-ray-to-build-a-large-scale-online-serverless-platform

[84] Ant Group Ray on Large-Scale Applications (PDF): https://static.sched.com/hosted_files/ray2020/2d/Using%20Ray%20On%20Large-scale%20Applications%20at%20Ant%20Group%20-%20Jiaying%20Zhou%2C%20Ant%20Group.pdf

[85] PayPal Aerospike Customer Story: https://aerospike.com/resources/customer-stories/paypal-aerospike-customer-story

[86] PayPal's Deep Learning Fraud Shield: https://reruption.com/en/knowledge/industry-cases/paypals-deep-learning-fraud-shield-blocks-billions

[87] Swiggy - 2x Improvement in Latency in Data Science Platform: https://bytes.swiggy.com/2x-improvement-in-latency-in-swiggy-data-science-platform-6101b7607530

[88] Xendit - XenShield: https://docs.xendit.co/docs/manage-fraud-with-xenshield

[89] Xendit How It Works: https://businessmodelcanvastemplate.com/blogs/how-it-works/xendit-how-it-works

[90] GrabDefence with Amazon Fraud Detector: https://aws.amazon.com/blogs/machine-learning/detect-fraud-in-mobile-oriented-businesses-using-grabdefence-device-intelligence-and-amazon-fraud-detector

[91] PayMongo Security: https://paymongo.com/security

[92] PayMongo QR Ph API: https://www.paymongo.com/products/fintech-infrastructure/qrph

[93] Go ML Benchmarks GitHub: https://github.com/nikolaydubina/go-ml-benchmarks

[94] KServe Benchmark README: https://github.com/kserve/kserve/blob/master/test/benchmark/README.md

[95] Istio Performance and Scalability: https://istio.io/latest/docs/ops/deployment/performance-and-scalability

[96] Rippling Engineering - Python Garbage Collection P99 Latency: https://rippling.com/blog

[97] Java 21 Generational ZGC: https://bluepes.com

[98] ZGC Benchmarking: https://morling.dev

[99] AWS EC2 g5.xlarge Pricing: https://www.devzero.io/instances/aws/g5.xlarge

[100] AWS EC2 g4dn.xlarge Pricing: https://www.emma.ms/blog/aws-ec2-g4dn-xlarge

[101] AWS EC2 g4dn.4xlarge Pricing: https://www.devzero.io/instances/aws/g4dn.4xlarge

[102] AWS EC2 c6i.4xlarge Pricing: https://www.devzero.io/instances/aws/c6i.4xlarge

[103] AWS EC2 c6i.8xlarge Pricing: https://cloudprice.net/aws/ec2/instances/c6i.8xlarge

[104] EKS Pricing Guide: https://cloudzero.com/blog/eks-pricing

[105] AWS Region Price Comparison: https://calculator.holori.com/aws-regions

[106] GCP n2-standard-8 Pricing by Region: https://gcloud-compute.com/n2-standard-8.html

[107] GCP GPU Zones List: https://holori.com/list-of-gpu-available-in-the-different-gcp-zones

[108] Azure Indonesia Central Region: https://windowsforum.com/threads/microsoft-azure-expands-across-asia-with-new-malaysia-indonesia-regions-and-india-capacity.383789

[109] RBI FREE-AI Framework Analysis (KPMG): https://kpmg.com

[110] RBI FREE-AI Report Summary (Khaitan & Co): https://khaitanco.com

[111] India DPDP Act 2023 Overview: https://itif.org

[112] DPDP Act Cross-Border Data Transfers: https://dpohub.com

[113] Bank Indonesia PBI 10/2025 Analysis: https://veritask.co.id

[114] PBI 10/2025 Summary (Assegaf Hamzah & Partners): https://ahp.co.id

[115] Indonesia GR71 Data Localization Analysis: https://itif.org

[116] OJK Regulations Overview: https://captaincompliance.com

[117] Google Cloud Bank Indonesia Compliance: https://cloud.google.com

[118] BSP AI Regulation Plans: https://fintechnews.ph

[119] BSP Project SAPIENS Thematic Review: https://bsp.gov.ph

[120] BSP MORPS Circular 1191: https://bsp.gov.ph

[121] BSP Digital Payments Transformation Roadmap: https://theasianbanker.com

[122] KServe + Feast Integration (IBM/Gojek): https://static.sched.com/hosted_files/ossna2022/87/Integrate%20KServe%20Modelmesh%20with%20high%20performance%20Feature%20server.pdf