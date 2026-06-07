# Comparative Analysis of MLOps Platforms for Real-Time Fraud Detection Ensembles in Large-Scale Southeast Asian Fintech Environments

This report presents a comprehensive evaluation of four leading MLOps platforms—**Seldon Core integrated with Triton Inference Server**, **Ray Serve**, **KServe on Kubernetes**, and **BentoML**—focused on deploying real-time fraud detection ensemble models combining gradient boosting methods (XGBoost, LightGBM) and neural networks. The setting involves processing 50 million daily payment transactions across India, Indonesia, and the Philippines, with stringent requirements for low-latency inference, burst traffic scalability, robust operational management, and cost efficiency on mixed CPU/GPU infrastructure.

---

## 1. Overview of Platforms and Architectural Foundations

### 1.1 Seldon Core with Triton Inference Server

Seldon Core, a Kubernetes-native platform, excels at deploying and orchestrating complex model graphs allowing multi-framework ensembles and advanced lifecycle management. When paired with NVIDIA's Triton Inference Server, it provides high-performance GPU-accelerated inference with dynamic batching and concurrent model execution.

- **Multi-framework support:** Robustly handles heterogeneous ensembles via Triton’s multi-backend support.
- **Scalability:** Supports GPU utilization with batching and concurrency for sustained high throughput.
- **Deployment model:** Kubernetes-based with custom YAML manifests.
- **Operational maturity:** Embedded A/B testing with traffic splitting (manual configuration), Alibi Detect integration for drift detection.
- **Challenges:** Requires Kubernetes and Triton domain expertise; higher integration effort.

### 1.2 Ray Serve

Ray Serve is a flexible, distributed Python-serving framework optimized for scalable, low-latency model serving. It supports custom ML workflows using Ray’s distributed computing capabilities.

- **Framework flexibility:** Highly Pythonic, easily integrating models from XGBoost, LightGBM, TensorFlow, and PyTorch.
- **Performance:** Incorporates HAProxy load balancing and gRPC between replicas, reducing p99 latency by ~75% and increasing throughput up to 11x.
- **A/B testing:** Lacks native A/B testing—this must be custom-built.
- **Ease of integration:** Python library support facilitates straightforward integration with feature stores via API calls.
- **Engineering demands:** Lower initial deployment overhead; ongoing maintenance requires custom development for traffic management and versioning.

### 1.3 KServe on Kubernetes

KServe (part of Kubeflow) is a Kubernetes and Knative-based serverless model serving system featuring autoscaling and advanced lifecycle management.

- **Autoscaling:** Supports scale-to-zero and scale-up on burst with Knative integration.
- **Traffic routing:** Native A/B testing, canary rollouts, and traffic splitting via Istio service mesh.
- **Feature store integration:** Tight integration with Feast (originated by Gojek), ensuring low-latency real-time feature serving.
- **Multi-framework & model mesh:** Equips multi-framework serving and optimizes multi-model deployments with ModelMesh.
- **Operational complexity:** Demands strong Kubernetes, Knative, and Istio user expertise; cold-start latency from scale-to-zero is a known operational challenge.

### 1.4 BentoML

BentoML is a Python-first model packaging and deployment toolkit, emphasizing developer experience and rapid iteration with containerized model “bentos”.

- **Simplicity & flexibility:** Eases packaging of heterogeneous models, supports both CPU and GPU inference.
- **Feature store integration:** Flexible but no native connectors—integration must be custom-coded.
- **A/B testing:** Minimal native support, requiring external orchestration.
- **Cost:** Low base platform footprint but relies heavily on external infrastructure for autoscaling and scaling.
- **Best fit:** Small-to-medium teams with rapid prototyping needs and low orchestration complexity.

---

## 2. Inference Latency and Throughput Under Normal and Burst Loads

### 2.1 Latency Metrics (p95 & p99)

- **Seldon Core + Triton:** Achieves highly optimized p95/p99 latencies, roughly 2–3 ms on GPU for core inference; including queuing and preprocessing, real-world p99 latencies hover around 12.5 ms. Dynamic batching and multi-instance grouping enhance throughput while maintaining stable latency at scale.
  
- **Ray Serve:** Latest upgrades with HAProxy and direct interreplica gRPC have delivered ~75% reduction in p99 latency, reaching average latencies as low as ~10 ms in tailored production environments.
  
- **KServe:** Provides low-latency inference (sub-50 ms typical) with autoscaling; however, cold starts and burst scaling can cause temporary latency spikes. Continuous tuning required to achieve SLAs.
  
- **BentoML:** Delivers p95 latencies under 50 ms on moderate hardware; cold start overhead can be significant relative to Kubernetes-based platforms due to container init time.

### 2.2 Throughput Handling During 10x Burst Traffic

- **Seldon Core + Triton:** Manages bursts by scaling GPU-powered inference replicas and employing dynamic batching. Multi-instance deployment ensures throughput sustenance at 10x normal load with minimal latency degradation.
  
- **Ray Serve:** Demonstrates near-linear scalability; benchmarks show 11.1x throughput improvements thanks to efficient load balancing and concurrency.
  
- **KServe:** Leverages Knative autoscaling to scale replicas at runtime, handling bursts well with some latency tradeoffs due to scale-up cold starts.
  
- **BentoML:** Throughput contingent on underlying orchestration environment; native scaling is minimal, necessitating integration with Kubernetes or cloud autoscaling to handle burst.

---

## 3. Native and Custom A/B Testing and Multi-Variant Deployment

- **KServe:** Industry leader in A/B testing and canary deployments via Istio traffic routing, enabling smooth, percentage-based traffic splits, experiments with multiple model versions, instant rollback capabilities, and integrated monitoring support—all declaratively configured.

- **Seldon Core:** Provides built-in A/B testing through multi-version deployments and traffic splitting, though often requiring manual setup and mesh tool assistance.

- **Ray Serve:** No native traffic management. A/B testing and variant routing must be custom-implemented via Python code or external services.

- **BentoML:** Minimal A/B testing functionality; requires external orchestration tools (CI/CD pipelines, service meshes) to perform traffic management.

---

## 4. Integration Overhead with Feature Stores

- **KServe:** Deeply integrated with Feast, an open-source feature store widely adopted in Southeast Asia fintech (notably Gojek). Feast supports centralized feature management and real-time low-latency online feature retrieval (p99 latency ~0.8–1.3 ms for 50+ features). KServe’s Feast transformers cleanly insert online features into serving workflows with minimal overhead.

- **Seldon Core:** Integration possible with many feature stores, but often entails custom pre/post-processing containers or sidecars to fetch features in inference pipelines, increasing deployment complexity.

- **Ray Serve:** Flexible Python integration allows easy feature store API calls inside the model serving function, simplifying online feature fetching but lacks standardized transformer components.

- **BentoML:** Supports feature store connections through custom Python code; no out-of-the-box standardized integration.

---

## 5. Total Cost per Million Predictions on Mixed CPU/GPU Infrastructure

- **GPU usage:** Accelerates inference dramatically but with 3–50x higher hourly cost than CPU instances; cost-efficiency hinges on workload batching and GPU utilization.

- **Seldon Core + Triton:** Achieves high throughput and low latency but entails higher GPU costs. Efficient batching and scaling minimize cost penalties. GPU-specific expertise and optimization increase operational costs.

- **Ray Serve:** Fractional GPU allocation and workload multiplexing lower infrastructure costs significantly. Elastic scaling and resource sharing reduce cost per million predictions by consolidating usage.

- **KServe:** Autoscaling from scale-to-zero reduces idle resource cost but introduces operational overhead. Kubernetes infrastructure costs and required human capital drive base platform cost upward.

- **BentoML:** Lightweight serving with minimal platform cost; actual cost depends on deployment environment and configuration. Scale-to-zero strategies and adaptive batching reduce compute costs but required orchestration complexity adds indirect costs.

- **Talent & Maintenance:** Skilled staff managing Kubernetes, GPUs, and complex orchestration form substantial indirect cost, often surpassing compute expenses.

---

## 6. Model Versioning, Rollback Handling, and Production Incident Management

- **KServe:** Strong native support for multi-version serving, canary rollouts, instant rollbacks via traffic management, integrated with Kubeflow Model Registry for auditability. Provides declarative APIs and automated policies for controlling model lifecycles.

- **Seldon Core:** Enterprise-grade versioning, supporting live updates and multi-model deployments managed via Kubernetes CRDs. Rollbacks managed through controlled deployment manifests, integrated with monitoring tools.

- **BentoML:** Contains a Model Store tracking versions and facilitating rollback through container redeployment but lacks complex rollout orchestration frameworks.

- **Ray Serve:** Zero-downtime deployments and flexible model swaps are possible but require custom scripts; lacks centralized registry-based version governance.

---

## 7. Data Drift Detection and Monitoring

- **Seldon Core:** Integrates Alibi Detect for runtime drift and outlier detection, with built-in monitoring dashboards and Prometheus compatibility, enabling automatic alerts for data shift scenarios.

- **KServe:** Provides integrated monitoring solutions compatible with Prometheus, Grafana, and third-party drift detection tools like Evidently AI. Supports automated retraining triggers based on drift detection.

- **BentoML:** Exposes system and application metrics through OpenTelemetry compatible exporters; drift detection requires external tooling integrations.

- **Ray Serve:** Offers Prometheus metric integration and observability hooks; drift detection is user-implemented, often via external pipelines and Ray ecosystem tools (e.g., Ray Tune).

---

## 8. Engineering Effort: Initial Deployment and Ongoing Maintenance

- **Seldon Core + Triton:** Moderate to high upfront engineering effort involving Kubernetes setup, Triton tuning, container orchestration, and pipeline integration. Maintenance includes monitoring model health, resource tuning, and batch size adjustments.

- **Ray Serve:** Lower initial engineering overhead with Python-native interfaces and simple cluster setup. However, advanced production features (A/B testing, versioning, autoscaling) require custom development and operational expertise.

- **KServe:** Highest learning curve and maintenance complexity due to Kubernetes, Knative, Istio, and Feast dependencies. Ideal for mature teams with dedicated platform engineers able to maintain the full ecosystem.

- **BentoML:** Simplest to deploy for small teams due to developer-friendly API and container packaging. Scaling and lifecycle management depend on external systems, increasing long-term maintenance complexity at scale.

---

## 9. Operational Complexity Managing Multiple ML Frameworks

- **Seldon Core:** Excels with multi-framework ensemble support through inference graphs and orchestration. Supports heterogeneous model types natively but requires careful resource and infrastructure management.

- **KServe:** Supports standard ML frameworks directly and custom frameworks using custom containers. Provides ModelMesh to orchestrate multi-model serving efficiently, but demands robust infrastructure and Kubernetes experience.

- **Ray Serve:** Framework-agnostic Python-native system, facilitating easy heterogeneous model serving within unified codebases. Good for rapid iteration, but orchestration and scaling multi-framework workloads need engineering effort.

- **BentoML:** Supports diverse ML frameworks within model packaging and serving containers, easing multi-framework use cases but lacks advanced orchestration features to coordinate complex ensemble pipelines.

---

## 10. Cold-Start Latency for New Model Deployments

- **KServe:** Scale-to-zero autoscaling reduces cost but causes cold start latency (from seconds to tens of seconds depending on model size and infrastructure). Mitigation strategies include pre-warmed pods or predictive scaling policies.

- **Seldon Core + Triton:** Does not inherently support scale-to-zero, but pre-warmed replicas reduce cold start impacts; GPU model loading times can cause startup delays.

- **Ray Serve:** Optimized for low cold-start latency with runtime improvements; pre-warming recommended for critical low-latency applications.

- **BentoML:** Cold start latency is higher due to container startup time, which may challenge real-time fraud detection SLAs.

---

## 11. Real-World Fintech Experiences and Best Practices

### Razorpay (India)

- Processes millions of transactions daily with strict latency (<200 ms) requirements.
- Implements custom fraud scoring models integrated with Flask and MLflow.
- Emphasizes reproducibility, robust serving infrastructure, and multi-framework support.
- Hybrid approach using multiple deployment technologies balancing speed and scale [27][28].

### Gojek (Southeast Asia)

- Pioneer of Feast feature store and KServe integration for real-time fraud detection across Indonesia and neighboring countries.
- Processes hundreds of millions daily transactions with target latencies around 30 ms.
- Adopt strict Kubernetes-native autoscaling and multi-model orchestration.
- Prioritizes feature consistency, low latency, and operational scalability [11][21][22].

### PayMongo (Philippines)

- Employs automated fraud workflows integrated with Sift for real-time scoring and transaction dispute reduction.
- Experiences emphasize cold-start latency and rapid model update effectiveness.
- Implements continuous data drift and monitoring pipelines to ensure stability in fast-growing transaction volumes [26].

---

## Conclusion and Practical Recommendations

For fintech startups processing 50 million daily transactions with complex ensemble models (XGBoost, LightGBM, neural nets), stringent latency, and burst traffic demands, the choice of MLOps platform considerably shapes capability and cost outcomes.

| Criteria                      | **Seldon Core + Triton**                   | **Ray Serve**                          | **KServe**                          | **BentoML**                        |
|-------------------------------|--------------------------------------------|--------------------------------------|------------------------------------|----------------------------------|
| Inference Latency (p95, p99)   | Low (~10-13 ms including queueing)         | Very low (single-digit ms, 75% p99 reduction) | Low with autoscaling; cold start spikes   | Moderate (~50 ms), higher cold start  |
| Throughput Bursts              | High GPU-driven throughput, batching       | 11x throughput gain with concurrency | Autoscaling handles bursts; scale-up lag | Depends on external infra          |
| Native A/B Testing             | Supported via manual config                  | No native support; custom implementation needed | Robust native canary + traffic splitting | Minimal; external tools required  |
| Feature Store Integration      | Moderate overhead via sidecars or transformers | Python API-based integration         | Deep native integration with Feast | Custom Python integration           |
| Cost per Million Predictions   | Efficient GPU utilization; higher platform ops cost | Lower cost with fractional GPU use   | Autoscaling reduces idle costs; higher Kubernetes cost | Low base cost; infrastructure-dependent |
| Model Versioning & Rollbacks  | Multi-version deployment; live rollback     | Zero downtime but custom version control | Native traffic routing and rollback | Model Store with manual rollback    |
| Data Drift Detection & Monitoring | Alibi Detect built-in; Prometheus support   | Custom tooling needed; Prometheus    | Integrated monitoring & drift detection | Needs external tooling              |
| Engineering Effort            | Moderate-high; Kubernetes + Triton complex  | Low-medium; Python-centered           | High; requires Kubernetes + Knative + Istio skills | Low for deployment; scaling complex |
| Cold-Start Latency            | Moderate; pre-warmed pods helpful             | Low with recent optimizations        | Can be high due to scale-to-zero   | Highest; container startup delays   |
| Multi-Framework Management    | Strong with inference graphs                   | Flexible Python-native                | Native multi-framework support     | Supports multiple frameworks but limited orchestration |

### Recommendations

- **For mature, scale-critical fraud detection on Kubernetes with strong ops teams:** *KServe* offers the most production-ready model lifecycle, autoscaling, and A/B testing capabilities, especially combined with Feast for feature store integration and model mesh for multi-model orchestration.
  
- **For high-throughput GPU-accelerated ensemble serving with moderate ops investment:** *Seldon Core plus Triton* provides excellent multi-framework support and latency optimizations suitable for strict SLAs.

- **For Python-centric startups emphasizing rapid iteration and flexible orchestration:** *Ray Serve* enables low-latency serving and throughput scaling, but requires custom extensions for deployment lifecycle and traffic management.

- **For small teams or early-stage startups seeking minimal operational overhead and rapid prototyping:** *BentoML* simplifies initial deployment but demands external systems for scale and sophisticated production use cases.

Fintech companies such as Razorpay, Gojek, and PayMongo successfully combine Kubernetes-native platforms (Seldon, KServe) with robust feature stores and monitoring frameworks, highlighting the importance of infrastructure maturity for operational success in Southeast Asia’s heterogeneous, high-volume payment landscape.

---

## Sources

[1] BentoML vs Seldon Core vs KServe: Model Serving Framework Comparison: https://reintech.io/blog/bentoml-vs-seldon-core-vs-kserve-model-serving-framework-comparison  
[2] BentoML vs Ray Serve vs Triton: Model Serving for AI Teams 2026: https://www.index.dev/skill-vs-skill/ai-bentoml-vs-ray-serve-vs-triton  
[3] Model Analyzer Metrics — NVIDIA Triton Inference Server: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/docs/metrics.html  
[6] Online Inference with 88% lower latency and 11.1x higher throughput (Ray Serve): https://www.anyscale.com/blog/ray-serve-inference-lower-latency-higher-throughput-haproxy  
[9] Performance Tuning - Ray Serve - Ray Docs: https://docs.ray.io/en/latest/serve/advanced-guides/performance.html  
[10] Enable High Throughput on Ray Serve with KubeRay — Ray 3.0.0.dev0: https://docs.ray.io/en/master/cluster/kubernetes/user-guides/kuberay-serve-high-throughput.html  
[11] Integrating High Performing Feast Stores with KServe Model Serving (PDF): https://static.sched.com/hosted_files/ossna2022/87/Integrate%20KServe%20Modelmesh%20with%20high%20performance%20Feature%20server.pdf  
[16] How to add CPU, GPU, and system metrics to the BentoML service metrics endpoint: https://softwaremill.com/how-to-add-cpu-gpu-and-system-metrics-to-the-bentoml-service/  
[17] BentoML vs Ray Serve vs Triton: Model Serving for AI Teams 2026: https://www.index.dev/skill-vs-skill/ai-bentoml-vs-ray-serve-vs-triton  
[18] How to Implement A/B Model Testing with KServe Traffic Routing on Kubernetes: https://oneuptime.com/blog/post/2026-02-09-kserve-ab-model-testing/view  
[19] Home - KServe Documentation Website: https://kserve.github.io/archive/0.14/  
[20] From Raw Data to Model Serving: A Blueprint for the AI/ML Lifecycle - Kubeflow: https://blog.kubeflow.org/fraud-detection-e2e/  
[21] An Introduction to Gojek’s Machine Learning Platform: https://www.gojek.io/blog/an-introduction-to-gojek-machine-learning-platform  
[22] The Multiverse of Mayhem: Gojek's Galactic-Scale Problems: https://www.linkedin.com/pulse/multiverse-mayhem-gojeks-galactic-scale-problems-wee-kiat-lau-hypgc  
[23] Design Considerations For Model Deployment Systems: https://www.bentoml.com/blog/ml-requirements  
[25] Model Drift vs Data Drift: How to Keep Your ML Models Accurate | KodeKloud LinkedIn: https://www.linkedin.com/posts/kodekloud_mlops-model-drifting-vs-data-drifting-activity-7395079197586755585-sirL  
[26] PayMongo Case Study | Sift: https://sift.com/resources/case-studies/paymongo-case-study/  
[27] Razorpay | Cybersource: https://www.cybersource.com/en/solutions/case-studies/razorpay-india.html  
[28] Fraud Detection in Financial Transactions: Razorpay Learning: https://razorpay.com/learn/fraud-detection-in-financial-transactions/  

---

This evaluation synthesizes up-to-date performance, operational functionality, integration overhead, cost considerations, and real fintech industry learnings to guide selecting the optimal MLOps platform tailored to demanding multi-framework, real-time fraud detection workloads in Southeast Asia.