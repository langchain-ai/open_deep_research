# Comprehensive Comparison of MLOps Platforms for Real-Time Fraud Detection at Fintech Scale

## Executive Summary

This report provides an in-depth, technical, and practical comparison of Seldon Core (with Triton Inference Server), Ray Serve, KServe, and BentoML, focusing on deploying real-time ensemble models (XGBoost, LightGBM, neural networks) at a fintech startup handling 50 million daily payment transactions in India, Indonesia, and the Philippines. The evaluation covers inference latency (p95/p99), throughput under burst traffic, A/B testing, feature store integration, operational costs, model versioning/rollback, monitoring and drift detection, engineering effort, cold-start latency, and multi-framework complexity. Industry best practices and real-world case studies from Razorpay, Gojek, and PayMongo are integrated throughout to inform recommendations.

---

## Platform Overviews

### Seldon Core (with Triton Inference Server)

Seldon Core is a Kubernetes-native open-source platform for deploying, scaling, and managing machine learning models. It supports complex inference graphs, multi-framework composition (including XGBoost, LightGBM, neural networks), traffic shaping, and advanced deployment strategies. When paired with NVIDIA Triton, it brings highly optimized serving for GPU/CPU and ensemble pipelines, but requires advanced Kubernetes skills and infrastructure to unlock its full potential. Canary and A/B deployments, declarative versioning, and integrations with Prometheus for monitoring are core strengths. Scale-to-zero is not natively supported and may require external tools (e.g., KEDA).

### Ray Serve

Ray Serve is a scalable, Python-native ML model serving library built atop Ray—a distributed execution framework. It excels in dynamically scaling workloads, running heterogeneous model ensembles (GBM, NN, etc.) in a single cluster, and supports advanced features like resource fractionalization, high-throughput streaming, multiplexed pipelines, and hot-swappable models. The latest updates (e.g., HAProxy and gRPC ingress) deliver major gains in tail latency and throughput. Although Ray focuses on Python, it is adaptable to Kubernetes (via KubeRay) and flexible in multi-framework settings.

### KServe

KServe, formerly KFServing, is a Kubernetes-based model serving framework, part of the Kubeflow ecosystem. It provides standardized APIs, unified handling for diverse frameworks (XGBoost, LightGBM, TensorFlow, PyTorch, etc.), and built-in integration with cloud-native autoscaling (via Knative). KServe features scale-to-zero to minimize idle costs, sophisticated routing for A/B or canary deployments, and declarative model management. It's highly customizable and tightly integrates with Kubernetes-native tooling (CI/CD, Prometheus, etc.), but requires significant Kubernetes expertise.

### BentoML

BentoML approaches serving as a developer-centric, Python-first problem, providing simple model packaging, versioning, and API serving. It's lightweight, quick to adopt (especially for small MLOps teams), and supports ensemble models by orchestrating Python code, though it lacks some advanced orchestration and auto-scaling features present in the other platforms. BentoML can be deployed in containers, on-prem, or as cloud/serverless functions, with dependencies defined per Bento, and is engineered for rapid iteration rather than heavy-duty orchestration.

---

## Comparison on Key Criteria

### Inference Latency (p95/p99) and Throughput (Including Burst Traffic)

- **Seldon Core (with Triton):**
  - Highly optimized when using Triton Inference Server for GPU/CPU batch inference.
  - Latency depends on orchestration and model size; performance can be affected by Kubernetes scheduling and sidecar overhead.
  - Throughput is robust for high-concurrency (tens-of-thousands QPS possible), but tuning is required.
- **Ray Serve:**
  - Major upgrades (gRPC ingress, HAProxy) reduced P99 latency by up to 88% and increased throughput by up to 11.1x in benchmarks with real-world models, scaling from ~490 QPS to over 1,500 QPS with maintained low tail latencies even under burst loads ([9][10]).
  - Supports both streaming and unary loads efficiently, with auto-scaling and resource-efficient scheduling.
- **KServe:**
  - Strong autoscaling via Knative; p95/p99 latency is competitive, though scale-to-zero introduces initial cold-start delays.
  - Benchmarks for typical workloads suggest stable low latency and burst resilience, though initial request after scale-up can add seconds of cold-start time ([1][2][4][7]).
- **BentoML:**
  - Achieves low latency especially in single-framework/small-team setups due to adaptive batching and minimal orchestration overhead.
  - Throughput generally limited by underlying hardware rather than framework overhead, but may not scale out as efficiently as Ray Serve or KServe for extreme loads.

**LightGBM has demonstrated lower p95/p99 inference latency than XGBoost when tested at production scale ([6]).**

---

### A/B Testing Capabilities

- **Seldon Core:** 
  - Full support for A/B testing via traffic splitting in deployment graphs, supporting canary, blue-green, and multi-armed bandit policies natively ([1][7]).
- **KServe:** 
  - Declarative traffic-splitting and canary via InferenceService API for seamless model version comparison ([1][7]).
- **Ray Serve:** 
  - Traffic splitting achievable through Python orchestration; advanced experiment frameworks require additional logic but are possible via API ([3][9]).
- **BentoML:** 
  - Orchestration-dependent and not packaged natively, but can implement traffic routing in deployment code or CI/CD.

---

### Integration Overhead with Feature Stores

- **Seldon Core/KServe:** 
  - Integrate cleanly with Kubeflow’s ecosystem; access to feature stores via Kubernetes services is standard practice, but may require template customization and security configuration ([1][7]).
- **Ray Serve:** 
  - Native Python code allows access to feature stores through SDK clients but may need custom engineering for optimal access and production readiness.
- **BentoML:** 
  - Flexible, but integration is manual and must be handled in service code or deployment pipelines, increasing engineering effort compared to other platforms ([1][7]).

---

### Cost Per Million Predictions (Mixed CPU/GPU Infra)

- **Seldon Core/KServe:** 
  - Operate most cost-effectively at scale due to autoscaling and resource control; KServe's scale-to-zero reduces idle costs ([2][4]).
  - Infrastructure/cost overhead can be significant for small teams, but unit cost per prediction drops as concurrency increases.
- **Ray Serve:** 
  - New performance improvements translate into lower hardware utilization per prediction, making it cost-effective on both CPU and GPU clusters; cost is highly sensitive to cluster management strategy ([9][10]).
- **BentoML:** 
  - Minimal base cost for low- to mid-scale; as demand grows, cost scales linearly with hardware use, but may require manual horizontal scaling for very high loads.

---

### Model Versioning and Rollback During Production Incidents

- **Seldon Core:** 
  - Supports versioned deployments; facilitates rapid rollback via declarative YAML updates in case of failure.
- **KServe:** 
  - Rollback and versioning built into InferenceService definitions; can switch traffic instantly between versions ([1][7]).
- **Ray Serve:** 
  - Python-level API enables redeployment of classes, supporting hot swapping; explicit versioning/rollback requires disciplined operational routines.
- **BentoML:** 
  - Versioning via Bento archives is straightforward, but production rollback relies on deployment pipeline discipline.

---

### Monitoring and Data Drift Detection

- **All Platforms:** 
  - Support Prometheus and custom metrics exports.
  - Seldon Core provides strong plugin support for custom monitoring and drift detection ([7]).
  - KServe relies on third-party/Prometheus with example integrations for drift detection.
  - BentoML and Ray Serve both support logging and integration with external monitoring systems, but detailed drift detection typically requires supplementary tools or cloud-managed options like Arize AI ([7][9]) or BentoCloud.

---

### Engineering Effort (Initial Deployment and Ongoing Maintenance)

- **BentoML:** 
  - Easiest for small teams; minimal setup, no Kubernetes required, rapid iterability. Ongoing maintenance is light unless advanced ops are needed ([1][7]).
- **Seldon Core/KServe:** 
  - High initial learning curve due to Kubernetes and CRD orchestration; scalable and resilient once operational, but require continuous DevOps and MLOps maintenance.
- **Ray Serve:** 
  - Local development is straightforward, but productionizing for large-scale distributed inference (especially on K8s) requires investment in Ray infrastructure and possible integration with KubeRay ([3][7][9]).

---

### Cold-Start Latency for New Models

- **KServe:** 
  - Scale-to-zero means pods spin up only on demand, introducing cold start (seconds per model spin-up) but reducing costs ([1][7]).
- **Seldon Core:** 
  - No native scale-to-zero, so cold-start equals Kubernetes pod/container and model load time.
- **Ray Serve:** 
  - New HAProxy/gRPC optimizations cut cold-start times, but still limited by underlying containerization and model load duration.
- **BentoML:** 
  - Minimal orchestration means cold-start equals server/process and model load time only.

---

### Operational Complexity with Multiple Model Frameworks

- **BentoML:** 
  - Supports mixed frameworks within Python services. Complexity scales with number of frameworks, but integration is straightforward for Python ecosystems ([1][8]).
- **KServe/Seldon Core:** 
  - Designed for operationalizing ensembles from multiple ML frameworks. Complexity of orchestration increases with variety, but platform is built for this use case ([1][7][8]).
- **Ray Serve:** 
  - Natively handles heterogeneous model workloads by managing resource pools intelligently, with efficient scheduling and high flexibility ([3][9]).

---

## Industry Case Studies and Real-World Insights

### Razorpay

- Razorpay (Thirdwatch) uses AI-powered real-time fraud detection integrated into their payment stack, leveraging extensive parameters and pooled data for fraudster fingerprinting. Their deployment is API-first and modular, facilitating rapid scaling ([11][14][15]).
- Partnership with Cybersource (using Decision Manager) highlights ML-powered fraud detection workflow, resulting in significant reductions in fraud-related chargebacks and false positives ([13]).
- Real-time fraud detection is critical, informed by robust ML infrastructure, but Razorpay has not publicly endorsed a single serving platform among those compared.

### Gojek

- Gojek deployed its JARVIS real-time fraud analytics tool to reduce detection lag from 30 minutes to seconds for >100M monthly transactions. They migrated from batch to real-time ML, use streaming data, and combine OLAP engines (ClickHouse over Kafka streams) for rapid rule-based and ML-based scoring ([16][20]).
- The core ML/AI ops stack includes Google Cloud, TensorFlow, BigQuery, and custom orchestration for model serving, achieving >10,000 req/sec with sub-30ms latencies ([17][18][19]).
- Highlights the importance of modular, scalable, and robust orchestration for fraud ML pipelines at fintech scale.

### PayMongo

- PayMongo Protect is a real-time ML risk scoring layer for payment transactions, integrating explainability dashboards, segmentation, and automated rules for fraud review ([24]).
- Details on serving stack are limited, but emphasis on real-time response and explainable ML matches the needs for low-latency, ensemble inference at scale.

---

## Practical Recommendations

- **For fastest onboarding with minimal DevOps:** Start with BentoML for rapid iteration and local workflow. As operational requirements and traffic scale, plan for migration to KServe or Ray Serve for larger-scale orchestration and multi-framework management.
- **For robust Kubernetes-based orchestration, advanced scaling, and traffic management:** KServe offers the best balance of autoscaling, cost control (scale-to-zero), framework coverage, and traffic management, making it ideal for technically mature teams with strong Kubernetes skills.
- **For high-throughput, mixed workload, multi-model environments:** Ray Serve's recent performance and throughput optimizations make it a top contender for Python-centric teams, especially if you need to orchestrate multiple GBM and neural network models at scale.
- **Seldon Core excels** in complex ensemble orchestration and traffic shaping, but operational overhead and licensing needs should be carefully considered for rapid fintech startup environments.

Industry leaders such as Razorpay and Gojek employ fraud ML stacks that prioritize real-time performance, modular orchestration, and explainability. While their production case studies do not endorse a single MLOps platform, their practices dovetail with workflows enabled by KServe and Ray Serve.

---

## Sources

1. [BentoML vs Seldon Core vs KServe: Model Serving Framework Comparison](https://reintech.io/blog/bentoml-vs-seldon-core-vs-kserve-model-serving-framework-comparison)
2. [Best MLOps Platforms To Scale ML Models](https://www.axelmendoza.com/posts/best-platforms-to-scale-ml-models/)
3. [Seldon Core VS Ray Serve, Ray Forum Discussion](https://discuss.ray.io/t/seldon-core-vs-ray-serve/9053)
4. [Top 10 AI Model Serving Frameworks Tools in 2026: Features, Pros, Cons & Comparison](https://www.devopsschool.com/blog/top-10-ai-model-serving-frameworks-tools-in-2025-features-pros-cons-comparison/)
5. [Machine learning serving using either kserve, seldon, or BentoML - Stack Overflow](https://stackoverflow.com/questions/74232893/machine-learning-serving-using-either-kserve-seldon-or-bentoml)
6. [CatBoost, XGBoost and LightGBM - Real-Time Evaluation](https://assets-eu.researchsquare.com/files/rs-7539803/v1/0abe3e89-71c1-4e25-9775-e5c996f8d552.pdf)
7. [ML Model Serving Tools Im Vergleich: KServe Vs Seldon Vs BentoML](https://xebia.com/blog/machine-learning-model-serving-tools-comparison-kserve-seldon-core-bentoml/)
8. [BentoML vs. KServe vs. Seldon Comparison - SourceForge](https://sourceforge.net/software/compare/BentoML-vs-KServe-vs-Seldon/)
9. [Major upgrades to Ray Serve: Online Inference with 88% lower latency and 11.1x higher throughput | Anyscale](https://www.anyscale.com/blog/ray-serve-inference-lower-latency-higher-throughput-haproxy)
10. [Ray Serve Upgrade Delivers 88% Lower Latency for AI Inference at Scale | MEXC News](https://www.mexc.co/news/978588)
11. [A Summary of the Razorpay Thirdwatch Webinar - Razorpay Blog](https://razorpay.com/blog/thirdwatch-ecommerce-fraud-webinar-summary/)
12. [Fraud Prevention Solution Thirdwatch Trending on Shopify! - Razorpay](https://razorpay.com/blog/shopify-thirdwatch-integration-activation/)
13. [Razorpay | Cybersource](https://www.cybersource.com/en-us/solutions/case-studies/razorpay-india.html)
14. [Razorpay Case Study: Building India's Most Complete FinTech Ecosystem for Business](https://www.bibs.co.in/blog/razorpay-case-study-building-indias-most-complete-fintech-ecosystem-for-business)
15. [Using Machine Learning to Detect Fraud: Introduction - Razorpay Tech](https://razorpay.com/blog/detect-fraud-using-ml-ai-thirdwatch/)
16. [Gojek Case Study | JARVIS | Detecting and decreasing fraudulent activity](https://afi.io/case_studies/gojek)
17. [Gojek's AI-Driven Transformation: A Case Study from Indonesia](https://www.linkedin.com/pulse/gojeks-ai-driven-transformation-case-study-from-indonesia-lau-pix4c)
18. [Case Study GO-JEK | Google Cloud and Google Maps Platform](https://terralogiq.com/case-study/cloud-platform/gojek/)
19. [Fraud Detection - Gojek Product + Tech - Medium](https://medium.com/gojekengineering/tagged/fraud-detection)
20. [Detecting Fraudsters In Near Real-Time With ClickHouse](https://www.gojek.io/blog/detecting-fraudsters-in-near-real-time-with-clickhouse)
24. [PayMongo Protect - PayMongo Docs](https://developers.paymongo.com/docs/paymongo-protect)