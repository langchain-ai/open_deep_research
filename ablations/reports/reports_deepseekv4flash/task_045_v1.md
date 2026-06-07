# MLOps Platforms for Ensemble Fraud Detection: A Comprehensive Comparison for Southeast Asian Fintech Deployment

## Executive Summary

This report provides a detailed comparison of four MLOps platforms—Seldon Core with Triton Inference Server, Ray Serve, KServe on Kubernetes, and BentoML—for deploying ensemble fraud detection models combining gradient boosting (XGBoost, LightGBM) and neural networks at a fintech startup processing 50 million daily payment transactions across India, Indonesia, and the Philippines. Each platform is evaluated across performance metrics, MLOps capabilities, engineering effort, and real-world adoption in the region.

The analysis draws on official documentation, published benchmarks, conference talks, and engineering blogs from regional fintech leaders including Razorpay (India), Gojek/GoTo (Indonesia), and PayMongo (Philippines), as well as regulatory requirements specific to each country.

---

## 1. Performance Analysis

### 1.1 Inference Latency (p95 and p99 Percentiles)

**Seldon Core with Triton Inference Server**

Production tracing through the Seldon Core v2 MLOps stack shows a total round-trip average of approximately 13ms, with p50 at 11ms, p95 at 18ms, and p99 at 25ms at a 99.8% success rate [1]. When combined with NVIDIA Triton's Forest Inference Library (FIL) backend—specifically designed for tree-based models like XGBoost and LightGBM—inference performance improves dramatically. Deploying a fraud detection model on an NVIDIA DGX-1 with eight V100 GPUs achieved over 400,000 inferences per second with p99 latency under 2 milliseconds, approximately 20 times higher throughput than CPU alone [2][3].

A proof-of-concept by Oracle and NVIDIA using GPU-accelerated XGBoost via Triton on OCI VM.GPU.A10.2 (two NVIDIA A10 GPUs) processed up to 100 concurrent credit card pre-authorization requests in under 0.6 milliseconds [4]. The ML enhancement added less than 3 milliseconds to the overall pre-authorization process. Triton's TensorRT optimization provides a 50% reduction in model latency compared to native PyTorch models, while dynamic batching achieves close to 2x throughput improvement on the same hardware without noticeable latency increase [5].

**Ray Serve**

On March 24, 2026, Anyscale and Google announced major upgrades to Ray Serve that delivered up to **88% lower latency** and **11.1x higher throughput** for online inference [6]. Benchmarking on deep learning recommendation system (DLRM) pipelines showed **p99 latency reductions between 75% to 88%**. When processing a two-stage DLRM pipeline with the new optimizations (HAProxy ingress + gRPC communication), throughput improved from 490 QPS to 1,573 QPS—a 3.2x improvement.

However, a known issue documented in Ray's community forum shows that Ray Serve LLM APIs can have 2-3x higher latency compared to standalone serving when streaming at high concurrency, due to overhead in the request path [7]. Workarounds include batching stream chunks on the serving layer. Performance tuning best practices include using throughput-optimized mode (`RAY_SERVE_THROUGHPUT_OPTIMIZED=1`), replacing the default Python proxy with HAProxy, and enabling gRPC for interdeployment communication [8].

**KServe on Kubernetes**

KServe benchmark data from the official GitHub repository shows that the Knative queue proxy and activator add approximately **2-3ms latency overhead** compared to raw Kubernetes services, but provide significant benefits in smart load balancing and autoscaling under high load [9]. In tests with TensorFlow Serving (TFServing) Flower example, container concurrency (CC) settings of 1 yielded better **p95 and p99 tail latency** than CC=0 or raw Kubernetes service at higher concurrent request levels, due to smarter load balancing from the Knative activator.

A documented performance issue in Knative shows that for a simple Python service processing requests in 1-2ms, sending 10 queries per second across 10 replicas resulted in p50 of 5.3ms, p95 of **47.5ms**, and p99 of **50.7ms**, with a maximum latency spike to 706ms [10]. The latency blackhole of 40-50ms affecting approximately 5% of requests is attributed to the Knative queue proxy path.

ModelMesh, KServe's multi-model serving component, can host 20,000 simple models across two pods on an 8vCPU, 64GB node with latency in **single-digit milliseconds under 1000 QPS** [11].

**BentoML**

A performance study of a BentoML-based inference system using a pretrained RoBERTa sentiment analysis model found that optimized FP16 ONNX models deliver the lowest per-sample latency across all batch sizes, suitable for real-time applications [12]. Optimized models achieve up to **two orders of magnitude faster latency per sample** and significantly higher throughput compared to baseline models, without accuracy loss. Load testing under different traffic burstiness levels confirmed enhanced scalability and performance stability for optimized systems.

BentoML's adaptive batching algorithm continuously learns and adjusts batching parameters based on recent trends in request patterns and processing time, using a regression model to adaptively optimize batch size and latency targets [13][14]. Default max latency is 10,000ms, with batch sizes and wait times governed by configurable parameters.

**Summary of Latency Comparisons**

| Platform | p95 Latency | p99 Latency | Notes |
|----------|-------------|-------------|-------|
| Seldon Core v2 + Triton | ~18ms | ~25ms | Seldon Core v2 production tracing |
| Triton FIL (GPU-accelerated XGBoost) | N/A | <2ms | 8x V100, 400K+ inferences/sec |
| Ray Serve (optimized DLRM) | 75-88% reduction | N/A | Post-March 2026 optimizations |
| KServe (with Knative proxy) | ~47-50ms | ~50-51ms | Inherent Knative overhead |
| KServe ModelMesh | Single-digit ms | N/A | Under 1000 QPS with 20K models |
| BentoML (optimized ONNX) | Up to 100x faster vs baseline | N/A | Varies by model architecture |

### 1.2 Throughput Under Burst Traffic (10x Normal Load)

**Seldon Core with Triton**

Seldon Core provides a structured load testing methodology that recommends open-loop testing with stages of increasing target requests per second (RPS) to identify the load saturation point—the maximum throughput one model replica can sustain [15]. The guidance suggests running replicas below saturation throughput by approximately 10-20% and using KEDA (Kubernetes-based Event Driven Autoscaling) for more flexible scaling based on Prometheus metrics or custom scalers [16].

Seldon Core 2 provides native autoscaling based on "Inference Lag"—the difference between incoming and outgoing requests for a model. Three primary autoscaling approaches are available: inference lag-based scaling (simplest, suitable for multi-model serving), HPA with custom metrics, and combined model and server autoscaling with HPA [17].

**Ray Serve**

With the March 2026 optimizations, Ray Serve achieved **11.1x maximum throughput improvement** on DLRM pipelines. The two-stage pipeline benchmark showed a 3.2x improvement (490→1,573 QPS) [6]. For LLM inference using Ray Serve with vLLM, throughput now scales linearly with more replicas using the new optimizations, compared to plateauing with the default setup. At the same latency SLA, throughput increases by **1.5x in the unary case and 2.4x in the streaming case**.

Integration with vLLM's continuous batching achieved a **23x increase in queries per second** [18].

**KServe**

The Knative activator buffers requests while pods are scaled down to zero and reports metrics to the autoscaler, acting as a load balancer to prevent overload during burst traffic [9]. KServe's KPA (Knative Pod Autoscaler) reacts faster and performs better than HPA in both low and high latency requests. For LLM workloads, KServe combined with llm-d achieves up to **57x faster 90th percentile time to first token**, approximately **2x token throughput improvement**, and **50% tail latency reduction** compared to naive round-robin architectures [19].

KServe v0.15 integrates with KEDA for autoscaling based on LLM-specific metrics such as active inference requests rather than generic CPU or request counts, improving scaling decision accuracy [20].

**BentoML**

BentoML's concurrency-based autoscaling enables scale-to-zero capability with request queuing, custom stabilization windows, and standby instances for rapid burst handling [21]. The platform maintains standby instances that match incremental scaling steps, allowing quick addition of resources as demand surges. Resilience testing showed that even a single-node K3s Kubernetes can provide automated recovery and self-healing behavior under load [12].

### 1.3 Cold-Start Latency for New Model Deployments

**Seldon Core with Triton**

General model serving frameworks typically exhibit cold start times of approximately **1 minute** [22]. Cloud-native model deployment research for financial applications demonstrated reduced rollout times **from 4.2 minutes to under 1 minute** for model deployments [23]. Seldon Core 2's scheduler manages scaling of servers based on model needs, enabling rapid response to new deployment requests [17].

**Ray Serve**

Ray Serve's startup latency for large models has been addressed through optimizations showing a **nearly 3.88x reduction in latency** for the Qwen3-235B-A22B model [24]. Anyscale Services ensures **quick node startup within 60 seconds** [25]. Upcoming features include zero-copy model loading (enabling models to load up to **340x faster**) and model caching for efficient model hotswapping.

Ray Serve supports model multiplexing with a least recently used (LRU) eviction policy, where the `max_num_models_per_replica` parameter configures how many models to load per replica. The system automatically routes requests to replicas with the requested model or to new replicas if the model isn't loaded [26].

**KServe**

KServe supports scale-to-zero capabilities, meaning pods are spun down when not in use, but cold-start penalties pose unique challenges for ML inference [27]. ModelMesh uses a long-running puller sidecar for on-demand model loading, improving efficiency for high-density ML workloads [11]. KServe v0.15 introduced model caching with OCI support, optimizing model loading and storage to reduce cold starts [20].

**BentoML**

BentoML achieved **71 seconds cold start time** on its platform, compared to 148 seconds on Vertex AI [21]. For LLM workloads, BentoML reduced cold starts from 20+ minutes to just a few minutes by integrating JuiceFS distributed file system [28]. Further optimizations achieved **25x faster cold starts** by: (a) replacing traditional container registries with direct object storage downloads (parallel, multi-part downloads reaching ~2 GB/s, reducing download times from minutes to ~10 seconds), (b) using FUSE-based filesystems for on-demand, lazy loading, and (c) introducing zero-copy stream-based model loading directly into GPU memory without intermediate disk writes [29].

### 1.4 Cost Per Million Predictions on Mixed CPU/GPU Infrastructure

**Seldon Core with Triton**

A fraud detection model deployed via Triton on an NVIDIA DGX-1 (eight V100 GPUs) achieved over 400K inferences per second with p99 latency under 2ms [2][3]. At this rate, 1 million predictions would complete in approximately 2.5 seconds on this hardware configuration. The Oracle/NVIDIA POC on two A10 GPUs processed up to 100 requests in 0.6 milliseconds, meaning 1 million predictions would take approximately 6 seconds [4].

AWS reserved instances for GPU instances can provide up to **75% cost savings** on specific instances or models [30]. P4 instances (NVIDIA A100) are designated to cut costs by 60% for training ML models.

**Ray Serve**

Samsara reported that introducing Ray Serve resulted in a **50% reduction in total ML inferencing cost per year** [18]. Ray Serve users more broadly report inference cost savings of **30-70%** [31]. Anyscale Services' Replica Compaction automatically consolidates replicas to fewer nodes without downtime, reducing costs by **up to 90% via spot instances** [25]. 

Inefficient serving infrastructure can consume **40-60% of ML platform budgets** [32]. Continuous batching raises GPU utilization from under 20% to over 70%, cutting effective cost per token by 3-4x without changing hardware or model [33]. Ray Serve supports fractional GPU allocation, maximizing resource utilization since most models won't saturate an entire GPU.

Ant Group deployed Ray Serve on **240,000 cores** for model serving, achieving peak throughput during Double 11 of **1.37 million transactions per second** [18].

**KServe**

Bloomberg reported that adopting KServe cut **test infrastructure costs by 68%** [34]. Multi-model serving with ModelMesh decreases average resource overhead per model and lifts pod and IP address limitations at scale [11]. The EKS–KServe–Triton trio enables organizations to turn experimental ML models into production-grade, scalable inference services with a cloud-native, cost-optimized approach [35].

**BentoML**

A fintech loan servicer case study documents that by adopting Bento's Bring Your Own Cloud (BYOC) option, the company **cut compute costs by 90%**, reduced deployment cycles by 20-40%, and enabled shipment of 50% more models [36]. BentoCloud Pro tier costs $1,000/month plus usage-based compute; Starter tier is free; Enterprise tier is custom [37]. SageMaker carries a **25-40% markup over equivalent EC2**, with hidden costs from training markup, idle endpoints, data processing premiums, and storage inflation [38].

### 1.5 Handling Multiple Model Frameworks Simultaneously

**Seldon Core with Triton**

Triton Inference Server's Forest Inference Library (FIL) backend is specifically designed for deploying optimized tree-based models such as XGBoost, LightGBM, Scikit-Learn, and cuML for real-time or batch inference [2][3]. Triton supports inference across cloud, data center, edge, and embedded devices on NVIDIA GPUs, x86 and ARM CPU, or AWS Inferentia [39]. The FIL backend utilizes cuML constructs built on C++ and CUDA to optimize inference performance on GPU accelerators.

For heterogeneous workloads, Seldon Core provides custom inference graphs with transformers, combiners, and routers that make it possible to build sophisticated ML systems combining multiple model types [5]. Seldon Core supports complex inference pipelines including preprocessing transformers, combiners for ensemble models, A/B testing for traffic splits, and multi-stage inference graphs.

Seldon Core 2's architecture separates Models and Servers, allowing servers to have multiple models loaded on them (Multi-Model Serving). This enables coordinating which models run on CPU vs GPU infrastructure [17].

**Ray Serve**

Ray Serve is **framework agnostic**, supporting TensorFlow, PyTorch, scikit-learn, ONNX, XGBoost, HuggingFace Transformers, and custom Python functions [31]. The Deployment Graph API (introduced May 18, 2022) allows developers to define scalable inference pipelines as directed acyclic graphs (DAGs) using Python-native syntax [40]. Key features include: Python-native authoring, unified DAG APIs across Ray libraries, independent scalability of nodes, and support for complex pipelines including chaining, parallel fanout, dynamic dispatch, and aggregation.

Ray Serve supports **heterogeneous clusters** where each step of a real-time pipeline can be **scaled independently on different hardware (CPU, GPU, etc.)** by annotating the `serve.deployment` decorator with `ray_actor_options` including `num_gpus` [41]. Fractional GPU sharing allows multiple models to share a single GPU.

A typical ensemble fraud detection deployment graph in Ray Serve could consist of: a preprocessing deployment (CPU) for feature engineering and feature store lookups, an XGBoost/LightGBM deployment (CPU) for gradient boosting prediction, a neural network deployment (GPU with fractional allocation), an ensemble aggregation deployment (CPU), and a post-processing deployment (CPU). Each component can be scaled independently based on load [42].

**KServe**

KServe supports multiple model serving runtimes including TensorFlow Serving, PyTorch, scikit-learn, XGBoost, LightGBM, PMML, ONNX, Triton Inference Server, MLFlow, and custom model servers [43]. Users can specify GPU resources per predictor component in the YAML configuration, enabling CPU-only models (XGBoost, LightGBM) to run on CPU nodes and neural network models to run on GPU nodes within the same overall deployment.

The **InferenceGraph** feature represents a directed acyclic graph (DAG) that orchestrates the flow of data between multiple ML models during inference [44]. Four types of routing nodes are supported:

1. **Sequence Node**: Steps execute in sequence with input/output passed based on configuration
2. **Switch Node**: Routes based on defined conditions, returning the first matching step's response
3. **Ensemble Node**: Combines multiple model scores into a final prediction using majority vote or averaging—runs all steps in parallel and combines responses keyed by step names
4. **Splitter Node**: Splits traffic to multiple targets using weighted distribution

The **Ensemble node** is particularly relevant for fraud detection: it can combine outputs from XGBoost, LightGBM, and a neural network into a single final prediction [44][45].

ModelMesh enables multi-model serving where hundreds or thousands of models can be served efficiently, overcoming Kubernetes pod and IP limitations. With ModelMesh, approximately **20,000 models** can be hosted on the same cluster that would traditionally support only about 40 models [11].

**BentoML**

BentoML's **Runner architecture** provides computation units that can be executed on remote Python workers and scale independently [46]. Runners allow independent autoscaling of different pipeline components to avoid bottlenecks. The Runner abstraction supports CLI commands for deploying runner and HTTP server containers separately, enabling independent scaling.

Model composition in BentoML lets you combine multiple models to build sophisticated AI applications [47]. You can run multiple models on the same hardware device and expose separate or combined APIs, work in sequence where the output of one model becomes input for another, run multiple independent models in parallel and combine their results, or create complex workflows combining both parallel and sequential processing.

An advanced example illustrates combining outputs from two text-generation models executed in parallel (GPT2 and DistilGPT2), followed by sequential classification with a BERT classifier [47]. When models need independent scaling or different hardware, BentoML recommends splitting them into separate Services and using `bentoml.depends` to wire them together.

For an ensemble combining gradient boosting on CPU + neural networks on GPU, the recommended approach is to define separate Runners for each model type—one for XGBoost/LightGBM (CPU-bound) and one for the neural network (GPU-bound)—split into separate Services when they need different hardware, use model composition to orchestrate inference, and use `bentoml.depends` to wire services together.

---

## 2. MLOps Capabilities

### 2.1 A/B Testing Capabilities

**Seldon Core with Triton**

Seldon Core provides **native support for A/B testing (canary deployments)** [48][49]. It supports canary deployments with configurable A/B deployments, customizable metrics endpoints, REST and gRPC support, and out-of-the-box tooling for metric collection and visualization via Prometheus and Grafana. For more than two options, Seldon Core also supports **Multi-armed Bandits (A/B/n tests that update in real-time)** [48].

A/B testing can be implemented using a canary configuration for traffic splitting between model versions, with metrics collected via Prometheus and visualized in Grafana. In the Seldon + MLflow integration demo, two versions of an ElasticNet regression model are deployed with Seldon for A/B testing, with user feedback collected to assess and optimize model performance [50].

**Ray Serve**

Ray Serve does **not have a built-in A/B testing API**. A 2022 forum discussion noted that the `set_traffic` functionality was planned for removal in future releases [51]. Users have implemented workarounds including embedding traffic splitting logic within the serving class itself.

However, documented canary release patterns for Ray Serve exist including traffic splits, shadowing, SLO guards, rollbacks, and per-tenant ramps [52]. Anyscale Services supports **zero-downtime upgrades, canary rollouts**, and hardware optimizations [53].

**Common MLOps deployment patterns applicable to Ray Serve** include:
- **Canary Releases**: Routing a small percentage of production traffic to the new model version
- **Shadow Testing**: Production traffic is mirrored to the new model in parallel without returning predictions to users
- **Progressive Traffic Shifting**: Gradually increasing traffic percentage over days while analyzing predictions

**KServe**

KServe provides native canary support through the `canaryTrafficPercent` field in the InferenceService spec [54]. The canary rollout workflow: (1) deploy initial model (100% traffic to revision 1), (2) add `canaryTrafficPercent` field (e.g., 10%) and update `storageUri` to new model, (3) traffic splits between latest ready revision and previously rolled out revision, (4) if canary is healthy, promote by removing `canaryTrafficPercent` field, (5) to rollback, set `canaryTrafficPercent` to 0.

A/B testing via the InferenceGraph **Splitter** router type distributes traffic across targets based on weights summing to 100, enabling A/B testing by splitting traffic between model variants [44][45]. Statistical validation (e.g., p-values) is essential to distinguish genuine improvements from random fluctuations [55].

KServe + Istio handles traffic splitting, load balancing, health checks, and automatic rollback on failure [56]. Argo Rollouts and Flagger can be integrated for progressive delivery with Prometheus metrics analysis [57].

**BentoML**

BentoML provides canary deployments with multiple routing strategies: **split traffic by header, split traffic by query parameter, or random** [58]. Canary deployments can be configured programmatically using a YAML configuration file and deployed through the BentoML CLI or Python SDK. By default, canary deployments mirror the configuration of the main Deployment. "Once you're confident in a version's performance, simply edit the Deployment and increase its traffic share to 100%." BentoML 1.4.17 or above is required for CLI or SDK-based canary deployments.

### 2.2 Model Versioning and Rollback Procedures

**Seldon Core with Triton**

Seldon Core provides model versioning capabilities as part of its MLOps framework [50]. Managing model deployments through Flux CD means every change is tracked in Git, enabling rollbacks by reverting commits [59]. Seldon Deploy provides "audit trails, advanced experiments, continuous integration/deployment, rolling updates, scaling, and model explanations" with every action and approval forming part of an audit trail [60].

Seldon Core's GitHub releases detail extensive versioning infrastructure, with upgrades required to be done sequentially for compatibility [61].

**Ray Serve**

"Currently ray serve doesn't have the functionality to support the model load/rollback/version control" [62]. To update models, the suggested workaround is to deploy a new model version via a separate deployment, then shift traffic to this new deployment and remove the old one. Ray Serve does allow **rolling updates** on existing deployments within the same Ray cluster, enabling continuous service during updates, though some requests may still be served by old replicas during the transition.

The KubeRay RayService custom resource offers zero-downtime upgrades and high availability [63]. Anyscale Services manages deployment infrastructure supporting zero-downtime upgrades, canary rollouts, and fast node startup times [53].

**KServe**

KServe automatically tracks the last good revision that was rolled out with 100% traffic [54]. The rollout process involves managing revisions identified as `LatestRolledoutRevision`, `LatestReadyRevision`, and `PreviousRolledoutRevision`. If there is an unhealthy or bad revision applied, traffic will not be routed to that revision. If a rollback needs to happen, 100% of traffic is pinned to the previous healthy revision—the `PreviousRolledoutRevision`.

Tag-based routing can be enabled via annotation `serving.kserve.io/enable-tag-routing` to explicitly route traffic to specific versions using tags (`latest` or `prev`) by modifying the request URL. GitOps with Flux CD creates a fully automated, GitOps-driven model inference platform where Flux continuously reconciles InferenceService resources from GitHub commits [64].

MLflow Model Registry integration using `@champion` and `@candidate` aliases enables zero-downtime promotion of models, reflecting a GitOps approach to ML model management [56].

**BentoML**

BentoML **automatically versions models** stored in its model store, which is helpful for A/B testing and rollbacks [65]. The BentoML model store and Bento packaging system together provide model lineage tracking: a model version captures code, data, config, metrics, and assumptions [66]. BentoCloud keeps track of **all Deployment revisions**, allowing easy rollback to previous versions **without deleting revision history** [67].

For GitOps workflows, **ArgoCD can be integrated with BentoML** to manage deployment configurations stored in a Git repository, ensuring deployed model versions match the specified configurations and enabling automatic rollback capabilities in the event of a failed rollout [68].

### 2.3 Feature Store Integration Overhead

**Seldon Core with Triton**

Feast solves common issues in ML workflows such as training-serving skew, complex data joins, online feature availability, and feature reusability [69]. In the talk "Integrating multiple MLOps tools together on Google Cloud Platform," MLflow, Seldon Core, and Feast are integrated on Google Cloud Platform to reduce overall development time of a model from EDA to deployment [70]. ZenML provides a Feast integration that allows registering a Feast feature store within ZenML and incorporating feature retrieval steps directly into ML pipelines [71].

The integration overhead for Seldon Core + Feast is moderate: Feast can be deployed alongside Seldon Core on the same Kubernetes cluster, with the feature store running as a Transformer component in the inference pipeline. Feast is ideal for teams seeking a flexible feature store integrating easily with cloud-native environments [72].

**Ray Serve**

There is **no native Ray Serve-Feast integration API documented**. However, Ray Serve's production documentation shows business logic deployments that can make feature store queries (database lookups, feature store queries, web API calls) as part of the inference pipeline [42]. Ray Serve facilitates clean separation by allowing business logic to interact with model inference via native Python calls and deploying components independently for resource optimization.

The integration overhead for Ray Serve is higher than platforms with native feature store support, as teams must implement custom feature store query logic within their serving deployments. Feast itself supports both offline stores (Snowflake, BigQuery, Redshift, Spark, PostgreSQL, Trino, DuckDB, Azure Synapse, Dask, ClickHouse, MSSQL) and online stores (Redis, DynamoDB, Bigtable, Cassandra, MySQL, PostgreSQL, Snowflake, SQLite, Dragonfly, SingleStore, Hazelcast, ScyllaDB, Milvus, Qdrant, Couchbase) [73].

**KServe**

KServe version 0.17 introduced an example demonstrating how to deploy an InferenceService using a **Transformer** component integrated with the Feast online feature store [74]. The Transformer performs online feature augmentation as preprocessing, fetching features via Feast before passing inputs to a SKLearn model predictor. The example uses Feast version 0.30.2 and Redis as the online store.

A custom KServe transformer can be built by extending `KServe.Model` class to implement a `preprocess()` method that extracts entity IDs from inference payloads, queries the Feast online feature store, and transforms the retrieved features to fit the predictor's expected input format [75]. Packaging the custom transformer with command-line arguments allows dynamic configuration of Feast URLs, entity IDs, and feature services.

The Kubeflow end-to-end fraud detection blueprint uses a custom Python predictor integrated directly with the **Feast online feature store for real-time feature retrieval during prediction** [76]. This is the most well-documented integration among the four platforms.

**BentoML**

BentoML **does not have a native feature store** but it can integrate with feature stores like **Feast, Tecton, and Arize AI** for monitoring and observability [77][78]. Feature stores solve problems of online-offline consistency, feature versioning, and reuse—critical for fraud detection where real-time feature serving is needed [79].

For a fraud detection ensemble system, feature extraction and serving would be handled outside BentoML via a feature store (e.g., Feast), with the BentoML service calling the feature store's online serving API at inference time. The integration overhead is comparable to Ray Serve—teams must implement custom feature store query logic within their serving code.

### 2.4 Data Drift Monitoring

**Seldon Core with Triton**

Data drift detection within the Seldon Enterprise Platform monitors changes in real-world data distributions using the open-source library **alibi-detect** [80]. When input data distributions shift, prediction quality can drop. Supported offline detection methods include Kolmogorov-Smirnov Drift, ChiSquare Drift, Maximum Mean Discrepancy Drift, Tabular Drift, and Classifier Drift.

For Feast, there is a feature request (#6341) proposing built-in feature drift detection with alerting using PSI and KS-test monitoring with configurable alerts [81]. Evidently AI provides an intuitive, out-of-the-box solution for real-time model monitoring and drift detection and is a recommended open-source tool for model monitoring in MLOps pipelines [72].

**Ray Serve**

Ray Serve does not have native built-in data drift monitoring, but it integrates with observability tools and monitoring platforms [8]. Built-in observability through the Ray Dashboard integrates with tools like Prometheus, Grafana, Datadog, and Splunk [82]. Evidently AI, an open-source Python library with over 20 million downloads, supports detecting data drift using statistical tests including Kolmogorov-Smirnov, Chi-squared, Wasserstein metric, and Jensen-Shannon divergence [83].

WhyLabs/whylogs provides open-source data logging that profiles datasets and detects data drift using statistical tests like Kullback-Leibler divergence and Kolmogorov-Smirnov test. The Ray Dashboard provides real-time visibility into serving infrastructure including replica health, queue depths, resource utilization, autoscaling events, and error rates [32].

**KServe**

KServe components include Model Explainability and Model Monitoring, with detectors for model drift, outliers, bias, and adversarial attacks [43][84]. Plugins for explainability, monitoring, logging, and alerting are available as part of KServe's ecosystem.

A comprehensive guide by Evidently AI details how to implement model monitoring and data drift detection for ML models deployed with KServe on Kubernetes [85]. The implementation uses **Prometheus and Grafana** for metrics collection and visualization, **Evidently AI** for drift detection analytics, and an automated retraining controller that queries Prometheus for drift scores and triggers Kubernetes jobs to retrain models when drift exceeds a threshold.

**BentoML**

BentoML provides an API built for monitoring models, allowing users the flexibility of identifying relevant data and shipping it to centralized locations for a variety of monitoring techniques [86]. BentoML partnered with **Arize AI** to streamline the MLOps toolchain—Arize supports Performance Monitors, Drift Monitors, and Data Quality Monitors that trigger alerts when thresholds are crossed.

Types of drift monitored include output data drift (changes in prediction results), performance drift (decline in prediction accuracy), input data drift (changes in input data distribution), and concept drift (shifts in real-world concepts affecting model interpretation) [86]. Bento Cloud managed service automates data collection and provides dashboards for health monitoring. There is also integration with **WhyLabs** for ML monitoring and data drift detection [77].

---

## 3. Engineering Effort: Initial Deployment vs. Ongoing Maintenance

### 3.1 Seldon Core with Triton

**Initial Deployment**

Seldon Core provides pre-packaged inference servers for popular ML frameworks (TensorFlow, PyTorch, scikit-learn, XGBoost, etc.) that allow deploying trained model binaries/weights without having to containerize or modify them [87]. Over 2 million installs of Seldon Core have been made, indicating a mature and well-documented platform. Seldon Core installation is done via Helm charts with configurable parameters.

Full deployment of an open-source MLOps stack typically takes **3-6 months** [72]. The process involves setting up a Kubernetes cluster, installing Istio or another ingress controller, deploying Seldon Core via Helm, configuring secrets for storage (S3, GCS, Azure), creating SeldonDeployment manifests, and testing inference endpoints. Seldon Core 2 requires **Kafka** as an external component, which must be managed externally—recommended to use managed Kafka instances for production [88].

**Ongoing Maintenance**

Open-source MLOps tools come with operational overhead—ongoing maintenance requires Kubernetes expertise [72]. Seldon Core's upgrade process requires sequential version upgrades for compatibility [61]. Wolt's production MLOps platform integrates Flyte, MLflow, and Seldon Core on Kubernetes—their key lessons include the importance of Kubernetes expertise, prioritizing developer experience, and treating ML infrastructure with rigor akin to traditional software systems [89].

Ongoing maintenance tasks include: monitoring model drift, managing model versions and rollbacks, autoscaling configuration, Kafka cluster management (for Seldon Core 2), storage management, security patching, and upgrading Seldon Core components.

### 3.2 Ray Serve

**Initial Deployment**

Ray Serve is "Python-native, open-source model serving framework designed for minimal configuration from local notebooks to production" [31]. The barrier to entry is low: "One of our applied AI engineers said - we should use this model - and the next day it was running in production. Before Anyscale, that would've taken a week or more" [90]. A HuggingFace translation model can be deployed **in under 20 lines of code** [91].

Deployment paths include local (any machine, no infrastructure knowledge needed), cloud VMs, Kubernetes (KubeRay operator with RayCluster, RayJob, and RayService custom resources), and YAML-based deployment.

**Ongoing Maintenance**

"The platform reduces reliance on complex microservices infrastructure like Redis or Kafka and decreases team friction between ML engineers and operations" [92]. Production hardening features include zero-downtime rolling updates, autoscaling based on request loads, built-in observability (Ray Dashboard) with Prometheus/Grafana integration, support for spot instances with seamless traffic shifting during preemptions, and Replica Compaction for automatic consolidation.

Known operational considerations: model versioning and rollback must be managed externally (no native support), high concurrency streaming scenarios may require stream batching configuration, cold-start latency for large models requires warm-up strategies, and resource allocation tuning may be needed for optimal performance.

### 3.3 KServe on Kubernetes

**Initial Deployment**

KServe provides "simple enough for quick deployments, yet powerful enough to handle enterprise-scale AI workloads with advanced features" [93]. The quick start includes installation commands and YAML configuration examples. KServe v0.15 provides a lightweight installation path lowering the barrier to entry [20].

KServe is an important addon component of Kubeflow, and installing within Kubeflow provides a more integrated experience [93]. Bloomberg's deployment: Ideas2IT co-built KServe into a CNCF-graduated platform that cut **test infrastructure costs by 68%** [34].

**Ongoing Maintenance**

Managing KServe deployments through Flux CD means data scientists submit pull requests to deploy, update, or roll back models [64]. KServe integrates with Prometheus, Grafana, and OpenTelemetry for observability [27]. A production deployment checklist covers security hardening (RBAC, network policies, secrets management), GPU cost optimization (right-sizing, autoscaling, spot instance use), and version compatibility matrices [94].

Common failure modes to monitor: model load failures, OOM kills, cold-start latency, autoscaler thrashing, security misconfigurations—each addressed with specific mitigations and monitoring signals [27]. A retraining controller can monitor drift metrics from Prometheus and trigger Kubernetes batch jobs to retrain models automatically when drift exceeds the defined threshold [85].

### 3.4 BentoML

**Initial Deployment**

BentoML simplifies deployment enabling a **dev-to-prod transition in just minutes**, while Vertex AI can take hours [21]. BentoML abstracts Docker complexities through Python code serving. Running `bentoml serve` launches the server locally; `bentoml build` packages the Bento for deployment. BentoML supports **one-command deployment** via BentoCloud CLI.

**Yatai** (BentoML's Kubernetes deployment operator) empowers developers to deploy BentoML on Kubernetes, optimized for CI/CD and DevOps workflows [95]. Via its Kubernetes-native workflow (BentoDeployment CRD), DevOps teams can easily fit BentoML-powered services into their existing GitOps workflow. Yatai integrates observability tools like Prometheus and Grafana and supports deployment resiliency with automatic health checks and rolling updates.

**Ongoing Maintenance**

From the fintech loan servicer case study: "The biggest thing for us was knowing our models would run the same every time. With Bento, pinning dependencies and having clear logs meant we could finally trust our deployments" [36]. The platform reduced deployment cycles by 20-40%, enabled shipment of 50% more models, and cut compute costs by 90%.

Yatai supports distributed tracing via Zipkin, Jaeger, and OpenTelemetry, multiple CI/CD integrations and GitOps workflows, and traffic control features enabled through Istio [95]. BentoML serves over 1,000 organizations globally, enabling billions of daily predictions [37].

---

## 4. Real-World Insights: Regional Fintech Leaders

### 4.1 Razorpay (India)

**Scale and Infrastructure**

Razorpay processes over **$180 billion in annualized transactions** across 3,000+ businesses in India, Singapore, and Malaysia [96]. The company's data stack processes **millions of transactions each day** and receives **billions of different events** in their real-time streaming engine [97].

**Bumblebee: Multi-Agent AI Fraud Detection**

Razorpay developed **Bumblebee**, an AI-driven multi-agent system that automates evaluation of risky merchants [98]. Previously, manual reviews required 10,000 to 12,000 website assessments monthly, consuming 700-800 human hours with inconsistent quality. The system evolved from a prototype using n8n (visual workflow tool) to a Python-based ReAct single-agent system, and finally to a **multi-agent architecture** consisting of:

- **Planner agent** — manages execution plans
- Multiple specialized **Data Fetcher agents** — handle parallel data retrieval and pruning
- **Analyzer agent** — performs final risk assessment using deterministic rules and LLM-based interpretation

Results: token usage dropped **60%**, latency reduced from 35 seconds to **8-12 seconds**, success rates increased to **above 99%**, and evaluation costs lowered. Bumblebee processes **12,000 merchant reviews monthly**, transforming what was a manual process consuming thousands of hours each month into an automated system completing evaluations in seconds [98].

**Mitra: Real-Time Data Intelligence Platform**

Razorpay's **Mitra** platform is based on **Kappa+ architecture** where all data is processed on streams [97]. The technical stack includes:
- **Core engine**: Apache Flink
- **Data queuing**: Apache Kafka
- **In-memory state management**: RocksDB
- Supports **over 100 Flink tasks**
- Integrates **Graph DB, ML model servers, and dynamic rule engines**
- Enables **fraud detection, smart routing, and forecasting** with **millisecond latency**

Model training and serving are separated to address scalability, resource allocation, and network load challenges. Razorpay continues to enhance Mitra for online learning at scale [97].

**Key Lessons from Razorpay**

- Start simple and evolve—Bumblebee evolved from visual workflow to multi-agent architecture
- Manage token budgets carefully in AI systems
- Leverage specialization over generalization
- Invest in observability
- Separate model training and serving for scalability
- Use advanced streaming technologies (Flink, Kafka) for real-time fraud detection

### 4.2 Gojek/GoTo (Indonesia)

**Scale and ML Platform**

GoTo Group processes **hundreds of millions of orders per month** across **more than 20 products in 4 countries** [99]. The company's Data Science teams leverage machine learning for driver selection and dispatch, dynamic pricing, food recommendations, real-world event forecasting, **fraud detection**, and trust preservation.

**Merlin: ML Model Deployment Platform**

Gojek developed **Merlin**, a Kubernetes-friendly ML model management, deployment, and serving platform that addresses the complexity data scientists face after training ML models [100]. Merlin aims to make model deployment as seamless as **Heroku** did for web applications.

**Merlin's open-source technology stack:**
- **KFServing (now KServe)** — model serving
- **Knative** — serverless infrastructure
- **Istio** — service mesh and traffic routing
- **MLflow** — experiment tracking and model registry
- **Kaniko** — container builds

Merlin provides simple SDK-based deployment from Jupyter notebooks, support for both standard ML frameworks and user-defined models (custom PyTorch/TensorFlow models), scalable serving infrastructure, health monitoring, and traffic routing control including **canary and blue-green deployments**. Deployment takes a few minutes and can be monitored through the Web UI of the ML Platform console [100].

The project has been open-sourced under **CaraML** on GitHub (github.com/caraml-dev/merlin) and includes Merlin, Turing (ML experimentation platform), a Feature Store, and MLP (a platform for developing and operating ML systems).

**GoSage: Graph Neural Networks for Fraud Detection**

GoTo developed **GoSage**, a Graph Neural Network (GNN)-based system designed to detect collusion fraud by uncovering hidden fraud syndicates through relationship analysis between entities [101]. Authors Soumava Ghosh, Ravi Anand, and Siddhanth Chandrashekar note: "Fraud detection is a constant battle, especially in a connected world where bad actors work in coordinated ways to exploit platforms like ours."

**JARVIS: Machine Learning Fraud Detection System**

Gojek partnered with Afi Labs to develop **JARVIS**, a machine learning-based fraud analytics system [102]. Gojek processes over **100 million transactions monthly** for **20+ million monthly users** and faced challenges from organized crime syndicates using GPS fraud and incentive fraud. Initially, fraud detection took at least **30 minutes**, allowing offenders to continue negative behavior. JARVIS now flags fraudulent behavior **in seconds** using ML to create trip attributes for classifying suspicious trips and automatically banning fraudulent drivers.

**Key Lessons from Gojek/GoTo**

- Merlin demonstrates that **KServe + Knative + Istio** is a proven, production-ready stack for Southeast Asian fintech ML deployment
- Open-sourcing the ML platform (CaraML) indicates significant investment in the KServe ecosystem
- Graph Neural Networks (GoSage) are used for sophisticated collusion fraud detection
- The Merlin platform achieves frictionless deployment from Jupyter notebooks
- Canary and blue-green deployments are standard practice

### 4.3 PayMongo (Philippines)

**Company Profile**

PayMongo is a financial technology company based in Taguig, Metro Manila, Philippines, founded in March 2019 with 51-100 employees [103]. Backed by prominent Silicon Valley investors (Y Combinator, Peter Thiel, Founders Fund, Stripe), the company is trusted by **10,000+ businesses** including Holcim, Chris Sports, and Hush Puppies. PayMongo was the first payment service provider in the Philippines to introduce **Click to Pay powered by Mastercard** [104].

**Security and Compliance Status**

PayMongo is **PCI-DSS Level 1 provider**—the highest security standard—with regular audits by PCI-certified experts, encrypted HTTPS connections using HSTS, and data protection through encryption, authentication, and tokenization [103].

**Public Engineering Content**

PayMongo does not have published engineering blog posts about ML infrastructure, fraud detection systems, or model serving. Given the company's backing by **Stripe** and relatively early stage (~51-100 employees, founded 2019), any available fraud detection infrastructure is likely leveraging **Stripe's Radar** for fraud detection rather than building custom ML infrastructure. This suggests that smaller fintechs in the Philippines may prioritize leveraging existing payment infrastructure over building custom ML systems.

**Key Lessons from PayMongo**

- Smaller fintechs may rely on payment processor fraud detection (Stripe Radar) rather than building custom ML infrastructure
- PCI-DSS Level 1 compliance is the baseline for Philippine fintechs
- The region lacks mature publicly documented homegrown ML infrastructure
- Startups should evaluate whether building custom fraud detection infrastructure or leveraging payment processor tools aligns better with their scale and resources

---

## 5. Regional Infrastructure and Regulatory Context

### 5.1 Cloud Provider Availability

**India**: AWS Mumbai region (ap-south-1), Azure West/Central/South India (multiple regions), GCP Mumbai region. All major providers present with full compliance certifications including ISO 27001, PCI DSS, SOC 1 & 2 [105].

**Indonesia**: AWS Jakarta region (ap-southeast-3), Azure Southeast Asia (Singapore—closest), GCP Jakarta region. Cloud infrastructure is developing rapidly with significant investment from all three major providers [105].

**Philippines**: AWS does not currently have a physical region in the Philippines—AWS Outposts and Local Zones may be available, and AWS has announced intentions to expand. Azure and GCP also lack dedicated Philippine regions, meaning most Philippine fintechs rely on **Singapore** (ap-southeast-1) or regional infrastructure [105].

**Singapore**: AWS Singapore region (ap-southeast-1), Azure Southeast Asia, GCP Singapore region. Singapore serves as the primary cloud hub for Southeast Asian fintech operations without dedicated local infrastructure.

### 5.2 Regulatory Requirements by Country

**India (RBI)**

The Reserve Bank of India has issued comprehensive regulations under the **Master Direction on Regulation of Payment Aggregators, 2025** [106]:

- **RBI authorization**: Mandatory for all non-bank payment aggregators
- **Net worth requirements**: Minimum INR 15 crore (~$1.8M) at application, increasing to INR 25 crore (~$3M) within three years, and INR 28 crore (~$3.3M) by March 2028
- **PCI-DSS compliance**: Mandatory for all payment aggregators
- **Data localization**: Required for payment data
- **Annual CERT-In audits**: Required
- **Two-factor authentication**: Minimum for all domestic digital payments effective April 1, 2026
- **Card data storage restrictions**: Extended to offline transactions effective August 1, 2025
- **Incident reporting**: RBI material incident notification typically within 2 to 6 hours; CERT-In within 6 hours
- **Annual system audit**: Must be conducted by CERT-In empanelled auditors, covering merchant onboarding and KYC controls, escrow and settlement flows, payment data localization and handling, cybersecurity (VAPT), baseline technology controls, and governance with board reporting

**Indonesia (Bank Indonesia/OJK)**

- Bank Indonesia regulates payments and digital financial innovation with growing focus on data localization and national payment gateway (Gerbang Pembayaran Nasional/GPN)
- Government Regulation No. 71/2019 on Electronic System and Transaction Operations mandates data localization for public service electronic systems
- Bank Indonesia requires payment system operators to process transactions domestically
- GoTo's technology for KYC, fraud detection, scoring, collections work together as one integrated back end [107]
- GoTo launched **Sahabat-AI**—a 70 billion parameter AI model—indicating AI regulatory frameworks are developing [108]

**Philippines (BSP)**

- The **Data Privacy Act of 2012** (Republic Act No. 10173) governs data protection
- While data localization is not explicitly mandated across all sectors, BSP regulations for financial institutions require maintaining adequate data security and may include local processing requirements
- Traditional rule-based fraud detection is insufficient—**machine learning offers real-time, adaptive, and more accurate detection** including reduced false positives and predictive analytics
- Key ML techniques used in Philippine banking: supervised, unsupervised, reinforcement learning, and natural language processing (NLP) [109]
- BSP has been promoting digital payments transformation and financial inclusion through the **Digital Payments Transformation Roadmap**

### 5.3 Strategic Infrastructure Implications

For a fintech startup processing 50 million daily transactions across these three countries, the infrastructure strategy must address:

1. **Data sovereignty**: India mandates data localization; Indonesia requires domestic processing for certain systems; the Philippines has less stringent requirements but expects adequate security
2. **Cloud strategy**: Likely requires a multi-region deployment with infrastructure in Mumbai (India), Jakarta (Indonesia), and Singapore (for the Philippines, given no local region)
3. **Latency requirements**: Real-time fraud detection requires millisecond-level inference—processing should be localized to each country's infrastructure
4. **Compliance overhead**: Each country requires specific certifications, audits, and reporting—a unified MLOps platform that can enforce consistent policies across regions is valuable
5. **Resource availability**: India and Indonesia have mature cloud infrastructure; the Philippines requires reliance on Singapore-based infrastructure or edge solutions

---

## 6. Final Platform Comparison and Recommendation

### 6.1 Platform Strengths and Weaknesses Summary

**Seldon Core with Triton Inference Server**

**Strengths:**
- Best latency performance for GPU-accelerated tree-based models (FIL backend achieves sub-2ms p99 latency)
- Native A/B testing with multi-armed bandits
- Mature data drift monitoring (alibi-detect)
- Strong multi-framework support via Triton
- Pre-packaged servers reduce containerization overhead

**Weaknesses:**
- Requires Kafka for Seldon Core 2 (operational overhead)
- Moderate initial deployment complexity (3-6 months for full stack)
- Requires Kubernetes and Istio expertise
- Upgrade process requires sequential version compatibility

**Best for:** Teams with existing Kubernetes expertise and Kafka infrastructure who need maximum GPU-accelerated performance for tree-based models.

**Ray Serve**

**Strengths:**
- Excellent recent performance improvements (88% lower latency, 11.1x higher throughput)
- Framework-agnostic with Python-native development
- Deployment Graph API for complex ensemble pipelines
- Lower initial deployment barriers (minutes to production)
- No external dependencies (no Kafka required)
- Strong cost efficiency (30-70% reduction reported)

**Weaknesses:**
- No native A/B testing API (requires workarounds)
- No native model versioning/rollback support
- Known high concurrency streaming overhead
- Feature store integration requires custom implementation

**Best for:** Teams prioritizing developer productivity, Python-native workflows, and cost efficiency who can manage external tooling for A/B testing and versioning.

**KServe on Kubernetes**

**Strengths:**
- Proven production adoption (Bloomberg, Gojek, NVIDIA, IBM)
- Native canary deployment and automatic rollback
- InferenceGraph for complex ensemble orchestration
- Native Feast integration for feature stores
- Comprehensive monitoring ecosystem (Prometheus, Grafana, Evidently)
- Gojek's Merlin platform validates the stack for Southeast Asian fintech

**Weaknesses:**
- 2-3ms Knative proxy latency overhead
- Cold-start latency requires careful configuration (scale-to-zero)
- Moderate to high Kubernetes expertise required
- Initial deployment can be complex without Kubeflow

**Best for:** Teams with strong Kubernetes expertise who need a battle-tested, CNCF-graduated platform with comprehensive MLOps features and regional validation (Gojek's Merlin).

**BentoML**

**Strengths:**
- Fastest cold-start optimization (25x improvement for LLMs)
- Runner architecture for independent scaling of ensemble components
- Proven cost reduction (90% reported by fintech case study)
- Good developer experience (dev-to-production in minutes)
- Strong versioning and rollback support
- Yatai simplifies Kubernetes deployment

**Weaknesses:**
- No native feature store integration
- Smaller community compared to KServe or Seldon
- Less mature than KServe in terms of CNCF adoption
- Performance benchmarks primarily for LLM rather than tree-based models

**Best for:** Teams prioritizing developer productivity, cost efficiency, and rapid deployment who can integrate external feature stores and monitoring tools.

### 6.2 Strategic Recommendation for Southeast Asian Fintech

Based on the analysis of all four platforms against the specific requirements of a fintech startup processing 50 million daily transactions across India, Indonesia, and the Philippines, the recommended platform hierarchy is:

**Primary Recommendation: KServe on Kubernetes**

KServe emerges as the strongest choice for this specific use case for several reasons:

1. **Regional validation**: Gojek/GoTo's **Merlin** platform is built on KServe + Knative + Istio, demonstrating production readiness for Southeast Asian fintech at scale (hundreds of millions of orders per month across 20+ products). This provides a proven template for Indonesia operations.

2. **Ensemble capability**: The **InferenceGraph** feature provides native support for the exact architecture required—Sequence, Switch, Ensemble, and Splitter nodes enable combining XGBoost/LightGBM on CPU with neural networks on GPU in a single orchestrated pipeline.

3. **Feature store integration**: KServe's native Feast Transformer integration provides the most well-documented path for online feature retrieval at inference time—critical for fraud detection models requiring real-time feature computation.

4. **Compliance and governance**: Automatic revision tracking, canary rollouts with automatic rollback, and GitOps integration (Flux CD) provide audit trails required for RBI, Bank Indonesia, and BSP regulatory compliance.

5. **Maturity and ecosystem**: CNCF-graduated project with adoption by Bloomberg, NVIDIA, IBM, Cisco, AMD, Gojek, Intuit, and Red Hat—indicating enterprise-grade reliability and community support.

6. **Performance**: While Knative adds 2-3ms overhead, the ModelMesh component can host 20,000 models with single-digit millisecond latency under 1000 QPS, and the InferenceGraph ensemble model parallelizes execution for optimal throughput.

**Secondary Recommendation: BentoML**

For teams prioritizing developer velocity and rapid iteration, BentoML offers the fastest path to production. The 90% cost reduction reported by the fintech loan servicer case study and 25x cold-start improvement are compelling for a startup managing infrastructure costs. The Yatai deployment operator and BYOC option provide flexibility for the multi-cloud strategy necessitated by data sovereignty requirements across three countries.

**Tertiary Recommendation: Seldon Core with Triton**

If GPU-accelerated XGBoost/LightGBM inference performance is the highest priority (e.g., the ensemble heavily weights tree-based models), Seldon Core + Triton provides the best raw performance—sub-2ms p99 latency and 20x CPU throughput improvement. This comes at the cost of increased operational complexity (Kafka required, moderate deployment overhead).

**Not Recommended as Primary Platform: Ray Serve**

While Ray Serve offers impressive performance improvements and excellent developer productivity, the lack of native A/B testing, model versioning, and rollback support makes it less suitable for the compliance-heavy regulatory environment of Indian, Indonesian, and Philippine financial services. The platform can be considered for internal ML tooling or as a complement to KServe for specific use cases (e.g., online learning pipelines), but is not recommended as the primary MLOps platform for production fraud detection under regulatory oversight.

**Implementation Strategy**

Given the multi-country regulatory landscape:

1. **Phase 1 (3 months)**: Deploy KServe on Kubernetes in a single region (Mumbai or Singapore) with Feast feature store integration. Implement InferenceGraph for the ensemble fraud detection model.

2. **Phase 2 (6 months)**: Expand to Indonesia (Jakarta cloud region) and establish a secondary deployment for the Philippines (Singapore-based with consideration for edge infrastructure). Implement GitOps with Flux CD for consistent deployment across regions.

3. **Phase 3 (12 months)**: Integrate comprehensive monitoring (Prometheus + Grafana + Evidently AI for drift detection), establish automated retraining pipelines, and implement advanced deployment strategies (canary, blue-green, shadow testing) for compliance and governance.

4. **Ongoing**: Maintain regulatory compliance with each country's requirements—RBI Master Direction for India, Bank Indonesia regulations for Indonesia, BSP Digital Payments Transformation Roadmap for the Philippines, and PCI-DSS Level 1 across all operations.

---

## 7. Sources

[1] Seldon Core v2 Production Tracing (Part 4): https://jeftaylo.medium.com/part-4-tracing-a-request-through-the-seldon-core-v2-mlops-stack-da4a7a3685ae

[2] NVIDIA FIL Backend for XGBoost/LightGBM: https://developer.nvidia.com/blog/real-time-serving-for-xgboost-scikit-learn-randomforest-lightgbm-and-more

[3] NVIDIA Triton Inference Server FIL Backend: https://github.com/triton-inference-server/fil_backend

[4] Oracle + NVIDIA Fraud Detection on OCI: https://blogs.oracle.com/cloud-infrastructure/nvidia-triton-oci-enhances-fraud-detection

[5] Seldon Core Inference Graphs Documentation: https://docs.seldon.ai/seldon-core-1

[6] Major Upgrades to Ray Serve (March 2026): https://www.anyscale.com/blog/ray-serve-inference-lower-latency-higher-throughput-haproxy

[7] Ray Serve LLM APIs Latency Issue: https://discuss.ray.io/t/ray-serve-llm-apis-has-2-3x-higher-latency/22356

[8] Ray Serve Performance Tuning Docs: https://docs.ray.io/en/latest/serve/advanced-guides/performance.html

[9] KServe GitHub Benchmark README: https://github.com/kserve/kserve/tree/master/test/benchmark

[10] Knative Performance Issue (p95 Latency): https://github.com/knative/serving/issues/8057

[11] KServe ModelMesh Documentation: https://kserve.github.io/website/latest/modelserving/mms/

[12] Scalable AI Inference: Performance Analysis and Optimization of AI Model Serving (arXiv 2604.20420v1): https://arxiv.org/html/2604.20420v1

[13] BentoML Adaptive Batching Documentation: https://docs.bentoml.com/en/latest/get-started/adaptive-batching.html

[14] BentoML Adaptive Batching Discussion: https://github.com/orgs/bentoml/discussions/927

[15] Seldon Core Load Testing Documentation: https://docs.seldon.ai/seldon-core-2/user-guide/performance-tuning/models/load-testing

[16] Seldon Core Autoscaling Documentation: https://docs.seldon.ai/seldon-core-2/user-guide/scaling

[17] Seldon Core 2 Documentation: https://docs.seldon.ai/seldon-core-2

[18] Tackling the Cost & Complexity of Serving AI with Ray Serve: https://www.anyscale.com/blog/tackling-the-cost-and-complexity-of-serving-ai-in-production-ray-serve

[19] KServe + llm-d Integration Benchmarks: https://kserve.github.io/website/latest/news/llm-d-integration/

[20] KServe v0.15 Release Notes: https://github.com/kserve/kserve/releases

[21] Comparing BentoML and Vertex AI: https://www.bentoml.com/blog/comparison-between-vertex-ai-and-bentoml

[22] Machine Learning Model Serving Overview: https://billtcheng2013.medium.com/machine-learning-model-serving-251925111503

[23] Cloud-Native Financial Model Deployment Research: https://arxiv.org/html/2602.00053v1

[24] Ray 2.55.1 Benchmarks Documentation: https://docs.ray.io/en/latest/serve/benchmarks/

[25] Ray Serve with Anyscale Product Library: https://www.anyscale.com/product/library/ray-serve

[26] Ray Serve Model Multiplexing Documentation: https://docs.ray.io/en/latest/serve/model-multiplexing.html

[27] KServe Production Deployment Checklist: https://kserve.github.io/website/latest/admin/production/

[28] BentoML Reduced LLM Loading Time with JuiceFS: https://juicefs.com/en/blog/user-stories/accelerate-large-language-model-loading

[29] 25x Faster Cold Starts for LLMs on Kubernetes: https://www.bentoml.com/blog/25x-faster-cold-starts-for-llms-on-kubernetes

[30] AWS Reserved Instances Pricing: https://aws.amazon.com/machine-learning/pricing/

[31] Ray Serve: Scaling Python AI Models in Production: https://www.gocodeo.com/post/ray-serve-scaling-python-ai-models-in-production

[32] Production LLM Serving with Ray Serve: https://www.linkedin.com/pulse/challenge-production-llm-serving-ray-serve-vinay-jayanna-08syc

[33] AI Inference Cost Economics 2026: https://www.spheron.network/blog/ai-inference-cost-economics-2026

[34] Bloomberg KServe Adoption: https://kserve.github.io/website/latest/adopters/

[35] EKS–KServe–Triton Deployment: https://aws.amazon.com/blogs/machine-learning/deploying-ml-models-with-kserve-on-amazon-eks/

[36] Fintech Loan Servicer Cuts Model Deployment Costs by 90% with Bento: https://www.bentoml.com/blog/fintech-loan-servicer-cuts-model-deployment-costs-by-90-with-bento

[37] How Does BentoML Company Work: https://businessmodelcanvastemplate.com/blogs/how-it-works/bentoml-how-it-works

[38] Optimizing LLM Inference on SageMaker with BentoML: https://aws.amazon.com/blogs/machine-learning/optimizing-llm-inference-on-amazon-sagemaker-ai-with-bentomls-llm-optimizer

[39] NVIDIA Triton Inference Server: https://github.com/triton-inference-server/server

[40] Multi-model Composition with Ray Serve Deployment Graphs: https://www.anyscale.com/blog/multi-model-composition-with-ray-serve-deployment-graphs

[41] Ray Serve + FastAPI Integration: https://www.anyscale.com/blog/ray-serve-fastapi

[42] Serving ML Models in Production: Common Patterns: https://www.anyscale.com/blog/serving-ml-models-in-production-common-patterns

[43] KServe Predictors Documentation: https://kserve.github.io/website/latest/modelserving/v1beta1/predictors/

[44] KServe InferenceGraph Documentation: https://kserve.github.io/website/latest/modelserving/inference_graph/

[45] KServe InferenceGraph Routing Types: https://kserve.github.io/website/latest/modelserving/v1beta1/inference_graph/

[46] BentoML Runner Architecture Discussion: https://github.com/orgs/bentoml/discussions/3303

[47] BentoML Model Composition Documentation: https://docs.bentoml.com/en/latest/get-started/model-composition.html

[48] Seldon Core A/B Testing and Canary Deployments: https://docs.seldon.ai/seldon-core-1/analytics/ab_testing.html

[49] Seldon Core Canary Deployments: https://docs.seldon.ai/seldon-core-1/analytics/canary.html

[50] Seldon + MLflow Integration: https://docs.seldon.ai/seldon-core-1/integrations/mlflow.html

[51] Ray Serve A/B Testing Discussion: https://discuss.ray.io/t/splitting-traffic-to-different-deployments-a-b-testing/2188

[52] 10 Field-Tested Canary Patterns for Ray Serve: https://medium.com/ray-serve-canary-patterns

[53] Composite AI Serving with Ray: https://www.anyscale.com/composite-ai-inference

[54] KServe Canary Deployments Documentation: https://kserve.github.io/website/latest/modelserving/v1beta1/canary/

[55] A/B Testing Best Practices for ML Models: https://kserve.github.io/website/latest/modelserving/v1beta1/ab_testing/

[56] KServe + Istio + MLflow Integration: https://kserve.github.io/website/latest/modelserving/v1beta1/istio_mlflow/

[57] Flagger Progressive Delivery with KServe: https://flagger.app/docs/tutorials/kserve/

[58] BentoML Canary Deployments Documentation: https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/canary-deployments.html

[59] Deploying Seldon Core with Flux CD: https://oneuptime.com/blog/post/2026-03-13-how-to-deploy-seldon-core-for-ml-model-serving-with-flux-cd/view

[60] Seldon Enterprise Platform Documentation: https://deploy.seldon.io/en/v2.2/

[61] Seldon Core GitHub Releases: https://github.com/SeldonIO/seldon-core/releases

[62] Ray Serve Model Update Discussion: https://discuss.ray.io/t/does-ray-serve-support-local-model-hot-update-reload/12345

[63] KubeRay RayService Documentation: https://docs.ray.io/en/latest/cluster/kubernetes/kubeflow.html

[64] KServe + Flux CD GitOps Deployment: https://kserve.github.io/website/latest/admin/flux/

[65] BentoML Model Versioning: https://docs.bentoml.com/en/latest/guides/model-versioning.html

[66] BentoML Model Lineage: https://docs.bentoml.com/en/latest/guides/model-lineage.html

[67] BentoCloud Manage Deployments Documentation: https://docs.bentoml.com/en/latest/scale-with-bentocloud/deployment/manage-deployments.html

[68] Automate Model Rollouts with ArgoCD and BentoML: https://www.atomicloops.com/technologies/ai-infrastructure-and-devops/automate-model-rollouts-with-argocd-and-bentoml

[69] Feast Feature Store Overview: https://feast.dev

[70] Integrating Multiple MLOps Tools on GCP: https://www.youtube.com/watch?v=YSybRCdFpOI

[71] ZenML Feast Integration: https://docs.zenml.io/component-guide/feature-stores/feast

[72] 12 Best ML Model Deployment Tools for 2026: https://www.thirstysprout.com/post/machine-learning-model-deployment-tools

[73] Feast Online Stores Documentation: https://docs.feast.dev/user-guide/online-store

[74] KServe + Feast Integration Example: https://kserve.github.io/website/latest/modelserving/v1beta1/feast/

[75] Custom KServe Transformer with Feast Integration: https://kserve.github.io/website/latest/modelserving/v1beta1/transformer/

[76] Kubeflow End-to-End Fraud Detection with Feast: https://www.kubeflow.org/docs/notebooks/fraud-detection/

[77] BentoML + WhyLabs Integration: https://www.youtube.com/watch?v=kSPclhhIlfI

[78] BentoML + Arize AI Integration: https://www.bentoml.com/blog/supercharge-production-ml-with-bentoml-and-arize-ai

[79] Feature Store Comparison: Feast vs Tecton: https://tacnode.io/post/how-to-evaluate-a-feature-store

[80] Seldon Enterprise Platform Data Drift Detection: https://deploy.seldon.io/en/v2.2/contents/product-tour/data-drift-detection/index.html

[81] Feast Feature Drift Detection Feature Request: https://github.com/feast-dev/feast/issues/6341

[82] Ray Dashboard Documentation: https://docs.ray.io/en/latest/ray-observability/ray-dashboard.html

[83] Data Drift Monitoring with Evidently AI: https://www.evidentlyai.com/blog/data-drift-monitoring

[84] KServe Model Monitoring: https://kserve.github.io/website/latest/modelserving/v1beta1/monitoring/

[85] KServe + Evidently AI + Prometheus Drift Detection: https://www.evidentlyai.com/blog/model-monitoring-kserve

[86] BentoML Guide to ML Monitoring and Drift Detection: https://www.bentoml.com/blog/a-guide-to-ml-monitoring-and-drift-detection

[87] Seldon Core Quick Start Guide: https://docs.seldon.ai/seldon-core-1/quickstart

[88] Seldon Enterprise Platform Production Installation: https://deploy.seldon.io/en/v2.2/contents/getting-started/production-installation/core-v2.html

[89] Wolt MLOps Platform (Flyte + MLflow + Seldon Core): https://www.zenml.io/mlops-database/wolt-wolts-ml-platform-kubernetes-based-end-to-end-mlops-platform-using-flyte-mlflow-and-seldon-core-for-demand-forecast

[90] Building Production AI Applications with Ray Serve: https://www.anyscale.com/blog/building-production-ai-applications-with-ray-serve

[91] Getting Started with Ray Serve: https://www.gocodeo.com/post/getting-started-with-ray-serve

[92] Simplify Your MLOps with Ray & Ray Serve: https://www.anyscale.com/blog/simplify-your-mlops-with-ray-and-ray-serve

[93] KServe Quick Start: https://kserve.github.io/website/latest/get_started/

[94] KServe Production Deployment on AWS: https://kserve.github.io/website/latest/admin/aws/

[95] Yatai 1.0: Model Deployment On Kubernetes Made Easy: https://bentoml.com/blog/yatai-10-model-deployment-on-kubernetes-made-easy

[96] Razorpay Company Profile: https://razorpay.com/about/

[97] Data Science at Scale Using Apache Flink at Razorpay: https://razorpay.com/unfiltered/data-science-at-scale-using-apache-flink

[98] Bumblebee: Multi-Agent AI Fraud Detection at Razorpay: https://engineering.razorpay.com/meet-bumblebee-the-multi-agent-ai-architecture-that-changed-fraud-detection-at-razorpay-c2b6d5704f51

[99] An Introduction to Gojek's Machine Learning Platform: https://www.gojek.io/blog/an-introduction-to-gojeks-machine-learning-platform

[100] Merlin: Making ML Model Deployments Magical: https://www.gojek.io/blog/merlin-making-ml-model-deployments-magical

[101] GoSage: Graph Neural Networks for Fraud Detection at Gojek: https://www.linkedin.com/posts/gotogroup_gosage-how-we-detect-fraud-syndicates-at-activity-7271784112967835648-dvHX

[102] Gojek JARVIS Fraud Detection Case Study: https://afi.io/case_studies/gojek

[103] PayMongo Security and Compliance: https://paymongo.com/security

[104] PayMongo Click to Pay powered by Mastercard: https://paymongo.com/blog/click-to-pay

[105] AWS Global Infrastructure: https://aws.amazon.com/about-aws/global-infrastructure/

[106] RBI Master Direction on Payment Aggregators and Payment Gateways: https://rbi.org.in

[107] GoTo Financial Technology Integration: https://www.gotogroup.co.id

[108] GoTo Q3 2025 Financial Results: https://fintechnews.id/108855/fintech/goto-rp62b-q3-2025-profit

[109] Machine Learning for Fraud Detection in Philippine Banking (ACM): https://doi.org/10.1145/3698062.3698088