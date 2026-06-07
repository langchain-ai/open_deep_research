# Comprehensive Update: Deepfake Detection Research, Ethics, and Regulatory Frameworks (Post-May 2026)

## Executive Summary

This report provides a comprehensive update to the prior survey on deepfake detection research, covering developments from mid-2026 onward across eight key areas. The landscape has evolved substantially since May 2026. In video detection, foundation model integration—particularly DINOv3, CLIP, and vision-language models—now dominates state-of-the-art performance, with the NTIRE 2026 Challenge demonstrating that ensembles of large pretrained models with degradation-aware training achieve the highest robustness. Specific benchmark results include GenConViT+ achieving 99.3% average AUC, Face2Parts reaching 98.42% AUC on FF++, and VLAForge substantially outperforming prior methods through vision-language semantics. Audio detection has seen the ASVspoof 5 Challenge establish new baselines with self-supervised models dominating, while the Podonos May 2026 benchmark revealed significant performance gaps between commercial and open-source tools. Multimodal methods have advanced with frameworks like PIA achieving 98.7% accuracy on FakeAVCeleb, and cross-modal synchronization approaches demonstrating strong generalization without fine-tuning. Privacy-preserving techniques now include fully homomorphic encryption for deepfake detection and adaptive quantization enabling real-time on-device processing. The generalization gap remains critical, with the Vector Institute's "Generalization Illusion" paper arguing that detectors' benchmark performance does not translate to real-world deployment. Regulatory developments have been intense: the EU AI Act's transparency obligations take effect August 2, 2026, with accompanying Article 50 draft guidelines released May 8, 2026; the NO FAKES Act was reintroduced in the US on May 27, 2026; India's IT Rules Amendment 2026 mandates 2-hour takedown for deepfake content; South Korea's AI Basic Act took effect January 22, 2026; and numerous other jurisdictions have enacted or proposed new laws. New benchmarks including the Microsoft-Northwestern-WITNESS dataset, Omni-Fake, and MultiFakeVerse provide more rigorous evaluation frameworks for the field.

---

## 1. New Technical Developments in Video Deepfake Detection

### 1.1 Transformer-Based Architectures

#### GenConViT and GenConViT+
The **GenConViT** (Generative Convolutional Vision Transformer) architecture and its enhanced **GenConViT+** variant represent significant advances in hybrid transformer-based detection. Published in MDPI Applied Sciences and the Journal of Information Systems and Telecommunication (2026), GenConViT combines generative models (Autoencoder and Variational Autoencoder) with feature extractors (ConvNeXt and Swin Transformer) to learn both visual artifacts and latent data distributions. The architecture comprises two networks—Network A using an Autoencoder and Network B using a Variational Autoencoder—both feeding features into ConvNeXt-Swin hybrids for classification.

Trained on over 1 million images from five major datasets (DFDC, FaceForensics++, TrustedMedia, DeepfakeTIMIT, and Celeb-DF v2), GenConViT achieves approximately **96% average accuracy** with **AUC values around 99%** across datasets. GenConViT+ reports **95.6% average accuracy and 99.3% AUC**. F1 scores reach **99% for DFDC, 99% for FF++, 99% for DeepfakeTIMIT, and 96% for Celeb-DF** [1].

The study acknowledges that "further work is needed to improve generalizability, especially when encountering out-of-distribution deepfake data," and notes that "performance can decrease when the model is tested on manipulation methods it has never encountered" [1].

#### MINTIME: Multi-Identity Size-Invariant TimeSformer
Published in IEEE Transactions on Information Forensics and Security (2024), **MINTIME** uses a novel divided space-time attention mechanism within a TimeSformer architecture designed for multi-face video analysis. The model incorporates a pretrained EfficientNet B0 backbone, identity-based attention that focuses temporal attention only on faces belonging to the same identity, and size embeddings to account for face size variations.

Trained on over 221,000 videos from the ForgeryNet dataset (including 11,785 multi-face videos with up to 23 faces per video), MINTIME-XC achieves **AUC of 94.25% and Accuracy of 87.64%** on ForgeryNet, substantially outperforming the SlowFast R-50 baseline (90.86 AUC, 82.59% accuracy) [2]. The model supports cross-forgery and cross-dataset analysis, and generates attention maps during inference to identify anomalies per identity.

#### TimeSformer Cross-Dataset Evaluation (Springer 2026)
A systematic evaluation published in Machine Vision and Applications (Springer, 2026) benchmarked TimeSformer, Vision Transformer (ViT), ResNet50, ResNet101, CNN-LSTM, and CNN-Attention architectures across FF++, Celeb-DF, DFD, and a newly introduced ReenactFaces dataset. Key findings include:

- **TimeSformer achieves 78.4% accuracy, 0.801 AUC, and 77.0% F1-score** with 96-frame clips and 30% fine-tuning, confirming the advantage of joint spatiotemporal modeling [3]
- All models benefit from moderate fine-tuning, with gains plateauing beyond 20%
- Increasing clip length enhances performance for temporally aware models
- "Even the best-performing models show reduced performance when transferred to entirely new domains, indicating current detection systems still lack true domain invariance" [3]

### 1.2 Foundation Model Integration

#### NTIRE 2026 Robust Deepfake Detection Challenge
The **NTIRE 2026 Robust Deepfake Detection Challenge** at CVPR 2026 (arXiv report: 2604.24163) attracted 337 participants with 57 final submissions. The challenge focused on developing detectors resilient to image degradations, both accidental and malicious. The key finding: **"Top methods rely on large foundation models, ensembles, and degradation training to combine generality and robustness"** [4].

The **winning approach (ShallowReal "DINO-MAC")** fine-tuned DINOv3-Large backbone with multi-aspect classification head, auxiliary deep supervision, dynamic resolution training, stochastic depth, and supervised contrastive learning [4].

Common strategies across top teams included:
- Large pretrained foundation models (DINOv3, CLIP variants)
- Parameter-efficient fine-tuning (LoRA)
- Ensembles of multiple models
- Degradation-aware training via aggressive augmentation pipelines
- Multi-scale and multi-stream analysis
- Quality-aware expert routing
- Contrastive learning enhancements
- Test-time augmentation [4]

The **4th-place solution** (Multi-Stream Foundation Model Ensemble, arXiv 2604.25889) integrated:
- **Localized Facial Stream**: Focuses on facial crops for fine-grained manipulations
- **Global Texture Stream**: Analyzes full-frame context for macro-level anomalies
- **Hybrid Semantic Fusion Stream**: Leverages frozen CLIP backbone to detect logical inconsistencies
- **DINOv2-Giant backbone** adapted via LoRA to preserve zero-shot priors

The ensemble used a calibrated, discretized voting system with 1:2:2 weighting ratio (Local:Global:Hybrid), achieving highly stable zero-shot generalization [5].

#### VLAForge: Vision-Language Semantics for Deepfake Detection (CVPR 2026)
**VLAForge**, by Jiawen Zhu et al. (arXiv 2603.24454), harnesses rich vision-language semantics from pretrained Vision-Language Models (VLMs) such as CLIP. The framework comprises:

1. **ForgePerceiver**: An independent learner designed to capture subtle forgery cues both granularly and holistically while preserving pretrained vision-language alignment knowledge
2. **Identity-Aware Vision-Language Alignment (VLA) Score**: Integrates cross-modal semantics with forgery cues enhanced through identity-informed text prompting

Comprehensive experiments demonstrated that **VLAForge substantially outperforms state-of-the-art methods at both frame and video levels** on classical face-swapping forgeries and recent full-face generation forgeries [6].

#### GazeCLIP: Gaze-Guided CLIP (March 2026)
**GazeCLIP** (Zhang et al., arXiv 2603.29295) introduces a gaze-guided CLIP-based framework that fuses gaze cues with visual features and adaptive language prompts. The model leverages significant distribution differences between pristine and forged gaze vectors, observing that "the preservation of target gaze varies significantly among GAN and diffusion generated faces" [7].

Key components:
- **Visual Perception Encoder (VPE)**: Combines appearance and gaze features
- **Gaze-Aware Image Encoder (GIE)**: Injects gaze information into CLIP image encoder
- **Language Refinement Encoder (LRE)**: Uses adaptive-enhanced word selector (AWS) for dynamic language prompt enhancement

**Performance**: Beats prior state of the art by **6.56% average accuracy for attribution and 5.32% AUC for detection** under unseen-generator settings. Tested on a fine-grained benchmark including about 20 advanced generators across GAN, diffusion, and flow-based forgery types [7].

#### Omni-Fake-R1: Unified Multimodal Detection (Qwen2.5-Omni-7B Based)
**Omni-Fake** (arXiv 2605.01638v1) introduces a unified multimodal detection model built on the Qwen2.5-Omni-7B large multimodal language model. The framework encompasses four modalities—images, audio, generic videos, and audio-video talking-head videos—with over 1 million in-distribution samples and 200,000+ out-of-distribution (OOD) samples.

The **Omni-Fake-R1** model employs:
- Four-stage curriculum supervised fine-tuning with modal replay
- Unified Group Sequence Policy Optimization (GSPO) reinforcement learning to jointly optimize detection, localization, and explanation

Achieves **state-of-the-art performance across all modalities and tasks**, with robustness to common post-processing corruptions and superior generalization to unseen deepfake generation methods [8].

### 1.3 Cross-Dataset Generalization Techniques

#### Face2Parts: Coarse-to-Fine Inter-Regional Facial Dependencies
**Face2Parts** (Uddin et al., arXiv 2603.26036) proposes a hierarchical feature representation analyzing the full frame, face, and key facial parts (eyes, lips, nose) using channel-attention mechanisms and deep triplet learning with margin ranking loss. Performance metrics (AUC):

| Dataset | AUC (%) |
|---------|---------|
| FaceForensics++ (FF++) | **98.42** |
| Celeb-DF V1 | 79.80 |
| Celeb-DF V2 | **85.34** |
| DFD | 89.41 |
| DFDC | **84.07** |
| DeepfakeTIMIT | 95.62 |

The method demonstrates strong performance across intra-dataset, inter-dataset, and inter-manipulation scenarios, confirming that "integrating coarse (frame)-, medium (face)-, and fine (left eye, right eye, lips, nose)-grained features significantly improves detection performance and generalization" [9].

#### VIGIL: Part-Grounded Structured Reasoning
**VIGIL** (Li et al., vigil.best) adopts a structured two-stage forensic framework inspired by expert forensic practice:
1. **Plan stage**: The model autonomously selects which facial parts to inspect based on global visual cues
2. **Examine stage**: Examines each chosen part with independent forensic evidence injected via a stage-gated mechanism

A progressive three-stage training paradigm (supervised fine-tuning → hard-sample self-training → reinforcement learning with part-aware rewards) ensures anatomical validity and coherence between evidence and conclusion. **VIGIL consistently outperforms both expert detectors and concurrent MLLM-based methods** with an average accuracy of **93.1%** across all OmniFake benchmark levels [10].

#### Frequency-Domain Masking for Universal Deepfake Detection
Published in ACM Transactions on Multimedia Computing, Communications and Applications (January 2026), this work by Al Machot et al. demonstrates that frequency-domain masking—applying random masking of portions of the frequency spectrum during training—compels models to learn more robust features. Key findings:
- "Frequency masking not only enhances detection accuracy across diverse generators but also maintains performance under significant model pruning"
- Optimal masking ratio: **15%**
- Resilient to structured model pruning (up to 80%), aligning with Green AI principles [11]

#### The Generalization Illusion (Vector Institute, arXiv 2605.09007)
Researchers at the Vector Institute argue that the gap between benchmark performance and real-world deployment is structural. Current detectors rest on five technical assumptions that no longer hold reliably:

1. Synthetic imagery leaves visible traces where composited onto real backgrounds
2. Generative models leave characteristic frequency fingerprints
3. Video generation produces frame-to-frame inconsistencies
4. Synthetic portraits fail to reproduce biological signals
5. Detector signals survive compression, re-encoding, and conferencing codecs

Each assumption held reasonably well for GAN-based face-swaps. None holds reliably now. End-to-end diffusion models generate entire frames with no blending step. Modern video generators handle temporal coherence. High-resolution synthesis reproduces physiological cues. The fifth assumption "has always been the least tested and the most fragile." The result is the **Generalization Illusion**: benchmark scores stay high while real-world detection performance quietly declines [12].

The paper documents that "documented deepfake fraud cases reveal a consistent pattern: automated media forensics played no role in catching any of them." When deepfake attacks get caught, they get caught by people noticing unusual request channels, broken institutional norms, or missing shared context. The authors propose adding communication-layer analysis drawing on Speech Act Theory, Grice's Cooperative Principle, and Cialdini's principles of influence, complementing rather than replacing media forensics [12].

### 1.4 Novel Architectures

#### X-AVDT: Audio-Visual Cross-Attention (CVPR 2026)
**X-AVDT** (Kim et al., CVPR 2026, pp. 4403-4414) proposes exploiting generator-side audio-visual signals by using DDIM inversion to access generator-internal audio-visual cues. The framework uses:
- A video composite revealing inversion discrepancies
- Audio-visual cross-attention features reflecting modality alignment enforced during generation

The authors introduce the **MMDF (Multi-Modal DeepFake) dataset** spanning GANs, diffusion, and flow-matching methods. **X-AVDT achieves leading performance on MMDF** and generalizes strongly to external benchmarks and unseen generators, with accuracy improved by **+13.1%** [13].

#### Hybrid Bag-of-Visual-Words + Deep Learning
Published in Nature Scientific Reports (2026), this approach integrates handcrafted local forensic descriptors with deep learning features. The framework uses:
- Bag-of-Visual-Words (BoVW) based on HOG descriptors at SURF, FAST, and BRISK keypoints
- High-level semantic features from fine-tuned ResNet-50, MobileNet, and ShuffleNet CNNs
- Features fused and classified using Support Vector Machine (SVM)

Achieves **up to 97.55% accuracy** while maintaining robustness under cross-dataset and challenging forensic conditions [14].

#### Deepfake Detection with Full-Stack CNN Application (2026)
A 10-layer Deep CNN with dilated convolutions and dropout regularization achieves **>99.9% training accuracy and >99% validation accuracy**. Deployed via TensorFlow for training, FastAPI for real-time inference, and React.js frontend, the system enables "rapid image uploads and instantaneous deepfake classification with confidence scores within 0.5 seconds." The model maintains **99.43% accuracy on Celeb-DF**, outperforming baseline CNN and traditional ML methods by 3-16% [15].

### 1.5 Summary of Benchmark Performance Metrics

| Method | FF++ AUC | Celeb-DF AUC | DFDC AUC | Notes |
|--------|----------|--------------|----------|-------|
| GenConViT+ | 99.3% (avg) | 99.3% (avg) | 99.3% (avg) | F1: 99% for FF++, 96% for Celeb-DF |
| Face2Parts | 98.42% | 85.34% (v2) | 84.07% | Cross-dataset evaluation |
| ViT (SDFVD) | - | - | - | 99.00% on small SDFVD dataset |
| TimeSformer (cross-dataset) | 80.1% | - | - | With 96-frame clips, 30% fine-tuning |
| 10-layer CNN (IJFMR) | - | 99.43% accuracy | - | Full-stack application |
| GazeCLIP | +5.32% AUC improvement | - | - | Over SOTA in unseen-generator setting |

---

## 2. New Audio Deepfake Detection Methods

### 2.1 Synthetic Speech Detection

#### ASVspoof 5 Challenge Results
The **ASVspoof 5 Challenge** (arXiv 2601.03944) introduced a new crowdsourced database with nearly **2,000 speakers** and diverse recording conditions, with 53 teams participating. Key findings:

- **Large self-supervised models** (wav2vec 2.0, WavLM) **dominate top-performing systems** in open conditions, significantly improving performance [16]
- The legacy **MaryTTS attack remains the most challenging to detect**, demonstrating limits of training on DNN-generated spoof data [16]
- Compression and encoding (Encodec, MP3) affect detection performance, with deep neural codecs posing greater challenges [16]
- Adversarial attacks (Malafide, Malacopula) increase detection errors [16]
- Simple score transformations (logistic regression-based calibration) can improve calibration and decision-making [16]

| Model/Method | EER/minDCF | Notes |
|-------------|------------|-------|
| AASIST3 (closed condition) | **minDCF 0.5357** | KAN-enhanced AASIST |
| AASIST3 (open condition) | **minDCF 0.1414** | Over twofold improvement over original AASIST |
| WavLM ensemble (progress eval) | **6.56% EER** | Four-model late fusion |
| WavLM ensemble (final eval) | **17.08% EER** | Higher due to unseen attacks |
| WavLM-base fine-tuned (dev) | **0.61% EER** | On development subset |

#### RAPTOR: Compact SSL Backbones for Audio Deepfake Detection
"Do Compact SSL Backbones Matter for Audio Deepfake Detection?" (arXiv 2603.06164v1) presents a controlled study analyzing compact (~100 million parameters) SSL models from HuBERT and WavLM families across 14 cross-domain benchmarks. Key findings:

- **Multilingual HuBERT (mHuBERT) variants outperform WavLM models and larger wav2vec2-XLSR backbones**, indicating **pre-training strategy is more critical than model scale** [17]
- Compact 100M RAPTOR models do not surpass the strongest 500M systems but outperform larger 300M wav2vec2-XLSR and commercial detectors [17]
- A test-time augmentation (TTA) protocol reveals that WavLM variants show "overconfident miscalibration under perturbations"—a risk undetectable through traditional error rate metrics—while mHuBERT models maintain stable uncertainty calibration [17]
- Standard evaluation metrics (EER) are insufficient to capture model confidence calibration and deployment risks under distributional shift [17]

#### SONAR Framework: Systematic Analysis over Generative and Detection Models
IBM Research's **SONAR** (ACM Transactions on Internet Technology 25(3), May 2025) is the first comprehensive framework to benchmark AI-audio detection uniformly across advanced TTS models, covering 5 traditional and 6 foundation-model-based detection models across 9 diverse audio synthesis platforms [18].

Key findings:
- Foundation models exhibit stronger generalization capabilities due to model size and pretraining data quality [18]
- Speech foundation models demonstrate robust **cross-lingual generalization**, maintaining strong performance across diverse languages despite being fine-tuned solely on English speech [18]
- Few-shot fine-tuning shows potential for tailored applications like personalized detection systems [18]

#### Diffusion Reconstruction for Generalizable Audio Deepfake Detection
Bo Cheng et al. (arXiv 2604.26465, submitted to Interspeech 2026) propose a novel framework using **diffusion-based reconstruction methods** to generate challenging hard samples that improve detection of unseen deepfake attacks. The framework integrates multi-layer feature aggregation with **Regularization-Assisted Contrastive Learning (RACL)** combining dual contrastive loss and a variance-based regularization loss.

Results: "Diffusion-based reconstruction offers the strongest generalization across diverse attack types, achieving a **22.6% relative reduction in average EER** compared to baseline" across ASVspoof 2019 LA, CodecFake, DiffSSD, WaveFake, and In-the-Wild datasets [19].

#### TFTransformer: Local-Global Dependency Model
Song, Zhang, and Yuan (Computers, Materials & Continua, 2026) present the **TFTransformer** model integrating local and global feature dependencies through SincLayer, 2D convolutional layers, and a time-frequency Transformer module. The enhanced **TFTransformer-SE** incorporates channel attention within 2D convolutional blocks.

Performance:
- **ASVspoof 2021 LA**: **3.37% EER** without data augmentation
- **ASVspoof 2019 LA**: **0.84% EER** without data augmentation

#### RTCFake: Speech Deepfake Detection in Real-Time Communication
Jun Xue et al. (arXiv 2604.23742, accepted by ACL 2026) introduce the first large-scale speech deepfake detection dataset specifically designed for **Real-Time Communication scenarios**, featuring approximately **600 hours** of paired offline and online speech from Zoom, WeChat, QQ, and other platforms, covering 10 TTS and VC systems.

Key innovation: **Phoneme-guided consistency learning (PCL)** leverages the observation that phoneme-level representations remain significantly more stable across offline-to-online transformations compared to frame-level details. Using the **XLSR+AASIST** model, PCL achieves an average **EER of 5.81%**, outperforming mixed training methods (7.33% EER) [20].

#### Audio Deepfake Detection at the First Greeting: "Hi!" — S-MGAA
Haohan Shi et al. (Accepted at ICASSP 2026) introduce **Short-MGAA (S-MGAA)**, a lightweight detection framework capable of handling ultra-short, degraded speech inputs (0.5–2.0 seconds) affected by communication effects like codec compression and packet loss. The framework enhances discriminative representation learning through:
- **Pixel-Channel Enhanced Module (PCEM)** for fine-grained time-frequency saliency
- **Frequency Compensation Enhanced Module (FCEM)** for multi-scale frequency modeling

S-MGAA consistently outperforms nine state-of-the-art baselines, showing strong robustness against degradations and favorable efficiency metrics suitable for potential real-time and edge-device deployment [21].

#### DeepFense: Unified, Modular Framework
**DeepFense** (Yassine El Kheir et al., arXiv 2604.08450, April 2026) is an open-source PyTorch toolkit integrating the latest architectures, loss functions, and augmentation pipelines alongside over 100 training recipes. A large-scale evaluation of over 400 models found that:

- The choice of pre-trained front-end feature extractor (Wav2Vec2, WavLM, HuBERT) **dominates overall performance variance** [22]
- Carefully curated training data improves cross-domain generalization [22]
- Reveals **severe biases in high-performing models** regarding audio quality, speaker gender, and language [22]
- DeepFense publishes **455+ pretrained models** and **12 datasets** at huggingface.co/DeepFense [22]

### 2.2 Voice Cloning Detection

#### Audio Avatar Fingerprinting for Authorized Voice Cloning
Gerstner (arXiv 2603.20165, March 2026) introduces the novel task of **"audio avatar fingerprinting"** — verifying whether synthesized speech is driven by an authorized identity. Key findings:

- An off-the-shelf speaker verification model (**TitaNet**) achieves around **91–98% AUC** in various fake audio detection datasets without prior forensic-specific training [23]
- With fine-tuning, TitaNet's embedding space can discriminate "who is driving" the synthetic voice based on speech mannerisms, differentiating self-reenactment from cross-reenactment [23]
- Introduces the **NVFAIR Audio Dataset** containing self- and cross-reenacted synthetic speech across 46 identities [23]
- Average AUC for real/fake determination: **91%** ; average AUC for audio avatar fingerprinting: **0.98** across all 46 speakers [23]

### 2.3 Foundation Model Integration Performance on Benchmarks

#### ASVspoof 2019/2021 Performance

| Model/Method | Dataset | EER | Source |
|-------------|---------|-----|--------|
| TFTransformer | ASVspoof 2019 LA | **0.84%** | [24] |
| TFTransformer | ASVspoof 2021 LA | **3.37%** | [24] |
| CCT, PaSST, SSAST transformers | ASVspoof 2019 LA | **≥92% accuracy** | [24] |
| Whisper Large V2 + AASIST + Colored DA | ASVspoof 2021 DF | **8.67% EER** | [25] |

#### RADAR Challenge 2026
The **RADAR Challenge** (arXiv 2605.09568v3) focuses on robust audio deepfake recognition under realistic media transformations including compression (Opus, MP3, AAC), resampling, noise, reverberation, and bandwidth limitation across six languages. Results:

| System | EER (Development) | EER (Evaluation) |
|--------|------------------|-----------------|
| Baseline (SSL-AASIST) | 37.71% | 42.6% |
| **Top systems** | **1.27%** | **5.10%** |

Top systems achieved as low as **1.27% EER on development** and **5.10% EER on the blind multilingual evaluation** (50,000 bona fide, 52,726 spoofed samples) [26].

#### Podonos Audio Deepfake Detection Benchmark (May 2026)
Podonos conducted a neutral benchmark evaluating eight systems (4 commercial APIs + 4 open-source models) using a private, modern test set reflecting current voice-cloning technologies. Key production requirements: high accuracy at 95th percentile, real-time inference (RTF < 1.0), and sub-second latency under load.

| Rank | System | Accuracy | F1 Score | False Positive Rate | False Negative Rate |
|------|--------|----------|----------|-------------------|-------------------|
| 1 | **Resemble AI** | **98.1%** | 0.981 | 2.5% | 1.4% |
| 2 | Aurigin AI | 96.8% | - | 1.5% | 5.0% |
| 3 | Hive | 83.5% | - | - | - |
| 4 | Reality Defender | 71.3% | - | 53.7% | - |
| 5-8 | Open-source models | 48-63% | - | - | - |

Open-source models' poor performance stemmed from training on outdated datasets (ASVspoof 2019 LA), failing to generalize to current synthetic voices from newer TTS systems [27].

#### Real Fraud Incident Study (WACV 2026 SAFE Workshop)
Gajewska et al. investigated detection models against **over 600 verified deepfake audio samples** from online scams affecting 180+ individuals (politicians, journalists, celebrities). Key findings:

- Models trained solely on ASVspoof 2021 showed EERs from **36.5% to 63.7%** on incident data [28]
- Incorporating incident-specific data into training significantly lowered error rates: best **EER of approximately 4.18%** (AASIST model) [28]
- LFCC outperformed MFCC for detecting synthesis artifacts
- Mean and voting aggregation strategies for partial deepfakes performed better than max or no aggregation [28]

#### Human Perception Study (2025-2026)
The **Human Audio Deepfake Perception 2026** dataset (35,532 judgments from 1,768 participants) covers audio samples generated by **138 TTS and VC systems** including commercial APIs (ElevenLabs, Resemble AI), autoregressive LM-based TTS (VALL-E, Bark, Llasa), and flow-matching systems (F5-TTS, CosyVoice). Only **0.1% of people could accurately identify deepfakes** in controlled studies [29].

---

## 3. New Multimodal Audio-Visual Methods

### 3.1 Fusion Strategies

#### PIA: Phoneme-Temporal and Identity-Dynamic Analysis
Tanvi Ranga's Master's Thesis (University at Buffalo, January 2026) introduces **PIA**, a multimodal framework aligning speech and facial motion at the phoneme level, jointly analyzing viseme appearance, lip geometry dynamics, and facial identity embeddings. The architecture uses 14 distinct phonemes with notable articulatory features, 3D convolutional networks with EfficientNet-B0 backbone, multi-head attention for temporal and modality fusion, and auxiliary ArcFace temporal consistency loss.

**Performance:**
- **FakeAVCeleb**: **98.7% accuracy, 99.8% AUC** — "highest overall performance on FakeAVCeleb, outperforming all baselines" [30]
- **DeepSpeak v2.0**: **98% AUC** [30]
- Strong cross-manipulation generalization across lip-sync, face-swap, and avatar-based deepfakes [30]

#### ConLLM: Contrastive Learning with Large Language Models
Kashyap et al. (Findings of EACL 2026) present **ConLLM**, a two-stage hybrid framework:
- **Stage 1**: Modality-specific embeddings via XLS-R (audio), VideoMAE (video), and VATLM (audio-visual)
- **Stage 2**: Aligned in shared latent space through contrastive learning, refined using GPT-style LLM reasoning to capture subtle semantic inconsistencies

Results:
- Reduces audio deepfake EER by up to **50%**
- Improves video detection accuracy by up to **8%**
- Achieves approximately **9% accuracy gains in audio-visual tasks**
- Ablation studies confirm PTM-based embeddings contribute **9%–10% consistent improvements** across modalities
- Strong cross-lingual generalization on multilingual audio datasets (DECRO dataset) [31]

#### SAVe: Self-Supervised Audio-visual Deepfake Detection
**SAVe** (arXiv 2603.25140) is a self-supervised framework trained exclusively on authentic videos without labeled synthetic deepfake data. It integrates four complementary modules:
- **FaceBlend, LipBlend, LowerFaceBlend**: Generate on-the-fly region-specific pseudo-manipulations
- **AVSync**: Audio-visual synchronization module detecting temporal misalignment between lip movements and speech audio

Key finding: "Visual-only branches are effective on identity-driven manipulations but fail on audio-only forgeries, whereas AVSync handles audio-only manipulations well, making their fusion especially powerful." Competitive in-domain performance on FakeAVCeleb and AV-LipSync-TIMIT with strong cross-dataset generalization [32].

#### DeepGuard: Unified Multimodal Detection
**DeepGuard** (IJRDET, Volume 15 Issue 3, 2026) uses three branches:
- CNN (EfficientNet-B4) for spatial artifact extraction
- Vision Transformer (ViT-B/16) for long-range spatiotemporal dependencies
- ResNet-34 audio branch for mel-spectrogram anomaly detection
- Learnable late-fusion module adaptively weighs modality-specific confidences

Achieves **average accuracy 97.4% and F1-score 96.9%** across four benchmarks, with **99.1% accuracy on FaceForensics++ (HQ)** . Limitations include ~480 ms inference per video second on V100 GPU and vulnerability to adversarial perturbations [33].

### 3.2 Cross-Modal Consistency Checks

#### Intra-modal and Cross-modal Synchronization (ICCV 2025)
Anshul et al. propose a novel two-stage multimodal framework:

- **Stage 1**: Self-supervised pretraining using three synchronization networks (two intra-modal: audio-audio and visual-visual; one cross-modal: audio-visual) trained on real datasets VoxCeleb2 and LRS2
- Uses **Gaussian-targeted loss** (outperforms InfoNCE loss)
- **Stage 2**: Pretrained features used **without fine-tuning** for deepfake classification and temporal localization

Key finding: "Intra-modal synchronization during pretraining positively impacts detection despite features not being directly used in classification." Strong generalization on FakeAVCeleb, KoDF, and LAV-DF with lightweight architecture requiring no fine-tuning [34].

#### LipFD: Lips Are Lying (NeurIPS 2024)
**LipFD** (Liu et al., arXiv 2401.15668) leverages temporal inconsistencies between lip movements and audio signals using a dual-headed transformer:
- Global Feature Encoder: Long-term audio-visual correlation
- Global-Region Encoder with Region Awareness module focusing on lip and head regions

Introduces **AVLips** dataset — first large-scale lip-syncing deepfake dataset with ~340,000 samples. Performance:
- Over **95.3% average accuracy** on AVLips dataset
- **Up to 90.18% accuracy** in real-world scenarios (WeChat video calls)
- Latency below 100ms
- Robust against saturation, blur, and compression perturbations [35]

### 3.3 Real-Time Detection

#### TrustLens: Portable Edge-AI Hardware Device
**TrustLens** (IJSREM, 2026) is a multimodal AI-based deepfake detection system deployed on Raspberry Pi 5 hardware (HDMI capture card, USB webcam, USB microphone, OLED display). Cost: approximately USD 150.

Analysis includes visual (facial texture inconsistencies, abnormal blinking, lip-sync mismatches) and audio (spectral anomalies, unnatural prosody) signals, fused via weighted aggregation into a unified authenticity trust score.

- **93.6% accuracy**, outperforming single-modality models by over 5 percentage points
- **Sub-200ms inference latency**
- Fully offline operation
- Supports live streaming detection for Zoom, Microsoft Teams, Google Meet [36]

#### DeFakeQ: Adaptive Bidirectional Quantization for Real-Time Edge Detection
**DeFakeQ** (arXiv 2604.08847, April 2026) introduces two innovations:

1. **Horizontal Adaptive Block Quantization (HAQ)**: Dynamically assigns bit-widths to layers based on weight-activation importance score
2. **Vertical Efficient Feature Fine-Tuning (VEFT)**: Selectively restores feature channels to full precision

Results: Compresses models to **10–20% of original size** while retaining **up to 90% of baseline accuracy**, significantly outperforming prior quantization methods. Successfully deployed on mobile devices [37].

### 3.4 Performance Summary on Multimodal Benchmarks

| Method | Dataset | Accuracy | AUC | Source |
|--------|---------|----------|-----|--------|
| PIA | FakeAVCeleb | **98.7%** | **99.8%** | [30] |
| PIA | DeepSpeak v2.0 | - | **98%** | [30] |
| ConLLM | FakeAVCeleb/DFDC | ~9% gains | - | [31] |
| DeepGuard | FF++ (HQ) | **99.1%** | - | [33] |
| DeepGuard | 4 benchmarks avg | **97.4%** | - | [33] |
| TrustLens | FF++/DFDC | **93.6%** | - | [36] |
| LipFD | AVLips | **>95.3%** | - | [35] |

---

## 4. Latest Developments in Privacy-Preserving Techniques

### 4.1 Fully Homomorphic Encryption (FHE)

A study from Nanyang Technological University (NTU, Singapore) presents **privacy-preserving deepfake detection with fully homomorphic encryption**, implemented via Concrete ML by Zama. The framework enables model inference directly on encrypted data without decryption, integrating:
- Discrete Cosine Transform (DCT) preprocessing
- Quantization Aware Training (QAT)
- Residual Network architectures into a homomorphic inference pipeline

Unlike differential privacy or secure multi-party computation, FHE **avoids accuracy loss and coordination overhead**. Results show "substantial reductions in homomorphic operation counts and inference latency due to frequency-domain optimization, while retaining statistically equivalent accuracy in both simulated and encrypted execution" [38].

### 4.2 Federated Learning

#### Blockchain and Federated Learning Synergy
A Springer publication (November 2025) edited by Kumar et al. explores **Blockchain and Federated Learning Synergy for Privacy-Focused DeepFex Solutions**, covering CNN, LSTM, GANs, and Transformers for deepfake detection with 12 chapters. The work emphasizes that "combining blockchain's decentralized infrastructure with federated learning's privacy-preserving approach offers a secure, transparent, and scalable method for combating deepfake technologies" [39].

#### Hybrid Federated Detection (ResearchGate, 2026)
A hybrid algorithm optimizing TSK-type fuzzy model structure using backpropagation learning algorithms demonstrates the ongoing evolution of federated approaches for privacy-preserving deepfake detection [40].

### 4.3 On-Device Processing

#### Privacy-Focused Offline AI on Consumer Hardware
Selvavinayagam et al. (WJARR, 2026) present a multimodal (spatial + temporal + audio) detection system running fully offline on consumer hardware (Intel Core i5 CPUs with 8 GB RAM). Using ResNeXt CNNs for spatial analysis, LSTM networks for temporal modeling, and Librosa for audio feature extraction with INT8 quantization, the system processes videos up to 5 minutes without crashes on standard laptops with accuracy comparable to state-of-the-art on FaceForensics++ and Celeb-DF. Grad-CAM provides explainability by highlighting suspicious facial regions [41].

#### Real-Time Edge Detection Using Lightweight CNNs
Published in Journal of Scientific Research and Technology (ISSN: 2583-8660), this framework integrates visual and audio analyses via efficient CNN-based models with point-position fusion for cross-modal inconsistencies, optimized for CPU-based local processing with no cloud reliance, enabling secure, private real-time detection on edge devices [42].

### 4.4 Differential Privacy Updates

#### TPDP 2026 Workshop (June 1-2, 2026, Boston)
The Theory and Practice of Differential Privacy Workshop at Northeastern University revealed several critical findings:
- "Private Evolution can outperform traditional federated learning baselines in utility and cost, providing early theoretical distributional convergence guarantees" [43]
- **CLIOPATRA**: First attack against "privacy-preserving" LLM-based insights systems [43]
- "Differentially private synthetic data methods generally fail to transfer substantial knowledge compared to non-private synthesis" (ContinuousBench evaluation) [43]
- "Practical implementations of differential privacy often suffer from subtle bugs that invalidate theoretical protections" — novel auditing framework "Re:cord-play" [43]

#### DP and Fairness Trade-offs
Lea Demelius (AAAI 2026) found that "DPSGD does not necessarily have a negative impact on fairness, as long as the DP model's hyperparameters are optimized for performance," but "hyperparameter tuning can leak information even when the single training runs are differentially private." Additionally, "Integrating DP leads to a decrease in overall accuracy and disproportionately impacts certain sub-groups, raising fairness concerns." A **DPSGD-Global-Adapt** variant was designed to reduce disparate impacts [44].

### 4.5 Biometric Security in the Deepfake Era (2026)
A white paper from Mantra Softech highlights embedded AI, multimodal ECG–EEG biometrics achieving up to **98.5% accuracy**, edge processing, and privacy-first cryptographic frameworks. Real-time AI security platforms like Scarecrow AI achieve **sub-100ms detection latency**, integrating AI with hardware security modules (HSMs) for proactive threat detection [45].

---

## 5. Updated Analysis of Real-World Deployment Gaps

### 5.1 The Generalization Gap: New Evidence

#### DeepFake-Eval-2024 Study
Leading commercial detectors' accuracy drops to about **78%** when tested against in-the-wild deepfakes, a significant drop from published benchmark numbers (95-99%). Controlled lab conditions do not replicate real-world variables such as codec compression, network jitter, background noise, adversarial optimization, and evolving generative AI tools. Single-model detection architectures are particularly vulnerable because attackers can optimize synthetic media to evade detection by exploiting a model's specific feature extraction blind spots [46].

#### Commercial Tool Performance in Practice
Brightside AI reports that commercial deepfake detection tools face accuracy dropping from claimed 95-98% to **50-65% or less** in actual use. Training data doesn't match real attacks; attackers rapidly evolve new generation methods. Solutions employing biological signal detection and multimodal analysis (Bio-ID, Sensity AI, Intel's FakeCatcher, Deepware) generally maintain better real-world accuracy. Open-source models achieve only **61-69% accuracy** and require significant maintenance resources [47].

#### The Generalization Illusion (Vector Institute)
As detailed in Section 1.3, the Vector Institute's paper argues that the gap is structural. The five assumptions underlying current detectors all fail for modern diffusion-based generators. The paper calls for adding communication-layer analysis—Speech Act Theory, Grice's Cooperative Principle, and Cialdini's principles of influence—to media forensics, treating detection as one signal among several alongside procedural controls [12].

#### 2026 International AI Safety Report
The report, compiled by over 100 leading AI experts including Yoshua Bengio, characterizes detection technologies as having "limited success." A 2025 study cited shows commercial systems achieving up to 89% deepfake detection accuracy compared to 20% for humans, but emphasizes: "Any detector tool can be expected to make errors, both false positives... and false negatives... Decisions will have to be made as to what level of these errors is acceptable" [48].

### 5.2 Compression Artifacts

#### SAFE Challenge (ResearchGate)
The SAFE Challenge specifically addressed compression artifacts from H.264, H.265, and AV1 codecs, as well as other techniques like adding borders, color correction, and frame processing. This research directly evaluates how standard video compression codecs affect deepfake detection performance [49].

#### NTIRE 2026 Robust Deepfake Detection Challenge
The challenge addressed robustness under common and uncommon image degradations including compression, noise, blur, and overlays. Top-ranked methods employed large-scale foundation models (DINOv3, CLIP), model ensembles, and advanced training augmentations exposing models to degradations [4][5].

#### Microsoft-Northwestern-WITNESS (MNW) Benchmark
The MNW dataset explicitly addresses that "content compression as a result of sharing on social networks and apps, but also manipulations and adversarial attacks makes it harder for models to consistently detect AI-generated media." The dataset contains media subjected to realistic post-processing manipulations such as resizing and compression, and will be updated biannually [50][51].

### 5.3 Generator Evolution

#### Sora 2 Detection
OpenAI's Sora 2 (released September 2025) marked a significant leap in AI video generation, simulating real-world physics, maintaining object consistency for up to 20 seconds, and synchronizing audio-visual elements. Detection accuracy for Sora 2 sits at **80-88%** compared to 90-95% for older models. The mixed diffusion-transformer architecture produces a characteristic colour distribution pattern that differs measurably from both camera capture and older AI models. Specific detectable artifacts include: distinctive colour signature, physics edge cases (finger count errors, liquid flow inaccuracies), generation drift in longer clips, and subtle audio-visual misalignments [52].

#### Diffusion Transformer (DiT) Dominance
The dominant architecture in 2026 is the diffusion transformer (DiT), employed in OpenAI's Sora 2, Veo 3, Kling, and Seedance. DiT models process video in latent space using spatiotemporal patches, offering high-quality outputs but with challenges like long-range temporal coherence and quadratic attention costs. Open-source models (WAN by Alibaba, HunyuanVideo by Tencent) provide flexibility but typically lag behind closed APIs in raw performance by 6-12 months [53].

#### Cross-Domain Generalization Limits of Vision Foundation Models
Research (arXiv 2605.24965) comparing RoPE-ViT, DINOv3, and NVIDIA C-RADIOv4-H found that while VFMs perform well on in-distribution data (especially full face synthesis), they falter significantly on out-of-distribution tasks involving localized face editing (CollabDiff, StyleCLIP) where most models degenerate to random guessing. The multi-teacher approach (RADIOv4) offers the best cross-domain resilience [54].

### 5.4 Demographic Bias

#### Bias-Free? Empirical Study (ACM 3796544)
Aditi Panda et al. evaluate demographic bias in state-of-the-art deepfake image detection models across age, ethnicity, and gender, investigating whether detection models perform differently across these demographic attributes, which could lead to discriminatory outcomes [55].

#### Commercial Bias
Brightside AI reports that commercial detection systems show better accuracy for lighter skin tones and older age groups, creating a situation where some demographic groups are better protected than others. Darker skin tones and younger age groups being more vulnerable to missed detections [47].

#### European Parliament Briefing on Women and AI Disinformation
Statistics reveal that **98% of deepfake videos are non-consensual pornography**, disproportionately targeting women (99%). The briefing highlights systemic gender gaps contributing to biased AI outcomes. The EU's legal frameworks strive to counter these through the Digital Services Act, AI Act (effective August 2026), and a 2024 Directive mandating criminalization of gender-based cyberviolence by 2027 [56].

### 5.5 Adversarial Robustness

#### Comprehensive Adversarial Study (IJERT, 2026)
Testing 32 state-of-the-art detectors against 156 variants of adversarial attacks (Lp-norm perturbations, codec-based, phase manipulation, and adversarial synthesis), the study finds that even imperceptible perturbations (L=4/255) achieve **94.7% attack success rate**, with some attacks reaching 100% success. Ensembles provide a false sense of security due to transferability of attacks across models (average transfer success rate of 73.8%) [57].

The proposed **ROBUST-DETECT** defense framework incorporates randomized preprocessing (JPEG compression at variable quality), prediction hardening via multiple stochastic passes, and dynamic thresholding, achieving certified robustness of 0.01 with 96.1% accuracy on clean data and substantially improved robustness (up to 89.7% vs. 4.7% baseline at L=4/255). However, it is computationally more expensive (up to 30×) [57].

#### Deepfake Growth Statistics
DeepStrike estimates an increase from roughly 500,000 online deepfakes in 2023 to about **8 million in 2025**, with annual growth nearing 900%. Voice cloning has crossed what experts call the "indistinguishable threshold"—a few seconds of audio now suffice to generate a convincing clone. The tools to generate coherent, storyline-driven deepfakes at scale have been effectively democratized [58].

---

## 6. Recent Ethical Concerns

### 6.1 Bias and Fairness

#### Demographic Disparities in Detection
The Bias-Free? study (ACM 3796544) and commercial tool audits consistently demonstrate that detection systems perform worse for darker skin tones and younger individuals. The European Parliament briefing notes that 98% of deepfake videos target women, and authoritarian regimes use gendered disinformation to discredit women leaders and erode democratic values [55][56].

#### NeuriPS 2025 Fairness Competition in Deepfake Detection
The first fairness-focused deepfake detection competition (NeurIPS 2025) involved over 64 teams and 158 participants from 20 countries. The winning method combined curated data, mixture-of-experts architecture, and test-time augmentation, improving fairness generalization without demographic-specific loss optimization. Other effective strategies included "foundation-model feature extraction, fusion of global and local cues, ensemble learning, and post hoc calibration" [59].

### 6.2 Privacy Concerns

#### Biometric Platform Risks
The "Ethical Implications of Deepfake Image Generation and Detection" paper (JETIR, March 2026) notes that "the detection methods raise the most ethical and privacy concerns, especially with regard to biometric platforms." Detection methods often require large amounts of biometric data for training, creating secondary databases of sensitive personal information [60].

#### Consent and Data Rights
The DeepFense evaluation (Section 2.1) revealed "severe biases in high-performing models regarding audio quality, speaker gender, and language," raising questions about consent and fairness in training data collection [22]. The MNW benchmark dataset is explicitly "intended solely for evaluation purposes and cannot be used for training or commercial purposes," reflecting growing concern about responsible data sharing [50][51].

### 6.3 Misuse of Detection Tools

#### The "Liar's Dividend"
Multiple sources raise the concern that even real videos can be dismissed as fake—the "liar's dividend"—allowing bad actors to deny authentic evidence by claiming manipulation. The EkasCloud report notes: "The most dangerous consequence? When people stop trusting any video evidence, real accountability becomes harder" [61][62]. The Vector Institute paper documents that "documented deepfake fraud cases reveal a consistent pattern: automated media forensics played no role in catching any of them" [12].

#### False Positive Harms
False accusations of deepfaking can cause reputational damage, job loss, legal consequences, and social ostracism. The 2026 International AI Safety Report emphasizes that "any detector tool can be expected to make errors, both false positives... and false negatives" [48].

### 6.4 Censorship Risks

#### AI Surveillance Concerns (2026)
AI surveillance in 2026 analyzes behavior, triggers decisions, and retains data at scale. Privacy risks rise when AI analyzes continuously without purpose limits, retention controls, or explainability. Privacy-conscious buyers in 2026 expect AI surveillance systems to be purpose-bound, enforce strict data minimization, provide explainable AI outputs, employ distributed processing, ensure real data ownership, and make privacy controls observable in daily operations [63].

### 6.5 Adversarial Dynamics

#### Structural Arms Race
The Vector Institute paper argues that "deepfake detection as a standalone technical capability is losing ground, and is likely to keep losing ground as generative models continue to improve" [12]. Deepfake voice fraud in contact centers increased from one attack every two days in 2023 to **seven per day in 2024**. An estimated $12.3 billion in losses in 2023 is projected to reach **$40 billion by 2027** in the U.S. alone [64].

#### Cross-Industry Collaboration
Reality Defender CEO Ben Colman emphasizes that "detection technologies will become more robust as Reality Defender and others continually update models to counteract new generative techniques, leveraging a cybersecurity approach of immediate response and future-proofing." Cross-industry collaboration is critical, involving partnerships with ID verification, voice security, social monitoring, and generative AI providers [65].

---

## 7. Regulatory Updates

### 7.1 European Union

#### AI Act Article 50 Draft Guidelines (May 8, 2026)
On May 8, 2026, the European Commission released draft Guidelines for implementing transparency obligations under Article 50 of the EU AI Act. Public consultation is open until June 3, 2026. Key provisions:

- **Scope**: Article 50 applies to any AI system generating synthetic audio, image, video or text content and includes a marking and detection duty under Article 50(2) [66]
- **Value Chain**: Providers responsible for upstream disclosures; deployers responsible for downstream obligations like labeling deepfakes [66]
- **Distribution-Only Actors**: Those limited to disseminating AI-generated content are not deployers within the AI Act [66]
- **Personal-Use Exemption**: Limited where activity affects public discourse; free and open-source AI systems remain fully subject to Article 50 [66]
- **Agentic AI Disclosure**: For agentic AI, the agent must self-disclose its artificial nature in every interaction scenario [66]
- **Article 50(4) — Deepfakes**: Requires human-visible disclosure for deepfakes impacting public discourse, with an artistic exception. Intent to deceive is irrelevant [66]
- **'Clear and Distinguishable' Standard**: Notifications must be readily noticeable and understandable, with repeated disclosures for ongoing exposure [66]
- **Multi-Layered Marking**: The December 2025 draft Code of Practice requires combining metadata embedding, imperceptible watermarks, and detection capabilities [66]
- **Transparency by Design**: Built into the full lifecycle of AI systems [66]

**Key Dates for EU AI Act:**
- Transparency obligations (except watermarking): **August 2, 2026** [66]
- Watermarking obligations (Article 50(2)): **December 2, 2026** [66]
- Final Code of Practice: Expected **June 2026** [66]
- Grandfathering rule: Systems on market before August 2, 2026, have until December 2, 2026, to comply with Article 50(2) [66]

**Fines**: Non-compliance with Article 50: up to **€7.5 million or 1.5% of global turnover** [66]

#### Ban on Deepfake Nudification Tools (Political Agreement May 7, 2026)
On May 7, 2026, the European Parliament and Council reached a political agreement under the "Digital Omnibus on AI" package explicitly banning AI systems designed to create non-consensual sexually explicit or intimate images, including deepfake nudification tools without consent. This is the first instance of a named AI application being explicitly banned under EU law—"not regulated, not restricted, banned" [67].

- **Enforcement Date**: December 2, 2026 [67]
- **High-Risk AI Compliance**: Stand-alone high-risk AI systems by December 2, 2027; embedded systems by August 2, 2028 [67]
- **EU AI Office Powers Strengthened**: Especially over providers developing both general-purpose AI models and output systems [67]

### 7.2 United States

#### Federal Legislation

**TAKE IT DOWN Act (Signed May 2025)**
Makes it a federal crime to knowingly publish or threaten to publish non-consensual intimate imagery (including deepfakes). Penalties: up to 2 years imprisonment for adults, 3 years for minors. Platform compliance with notice-and-takedown processes mandated by **May 19, 2026**, monitored by the FTC with civil penalties up to **$53,088 per violation**. The first conviction under this law was secured in Ohio in **April 2026** [68].

**NO FAKES Act of 2026 (Introduced May 27, 2026)**
A revised version was introduced on May 27, 2026, by a bipartisan group: Senators Marsha Blackburn, Chris Coons, Thom Tillis, Amy Klobuchar, and Representatives Maria Salazar and Madeleine Dean. Key provisions:
- Grants individuals the right to authorize digital replication of their voice and likeness [69]
- Post-mortem rights extend up to **70 years** after an individual's passing, transferable to heirs [69]
- Includes "counter-notice" procedure to contest content removal [69]
- Exemptions for news, documentaries, sports, biographical works, and content for comment, criticism, or parody [69]
- Previously garnered support from artists (Randy Travis) and major tech companies (YouTube, Amazon, OpenAI) [69]

**DEFIANCE Act of 2025 (S.1837)**
Establishes a federal civil cause of action for victims of non-consensual, sexually explicit deepfake imagery. Passed the Senate in **January 2026**; awaiting House action [70].

**DEEPFAKES Accountability Act (H.R.5586)**
Would require creators of deepfakes to include digital watermarks and disclosures. Introduced in the 118th Congress but **died in committee**; would require reintroduction in the 119th Congress [70].

**AI Fraud Accountability Act of 2026 (S. 3982)**
Introduced March 4, 2026, by Senator Tim Sheehy (R-MT), aiming to establish protections against digital impersonation fraud. Referred to Senate Commerce, Science, and Transportation Committee [71].

#### State Laws
As of May 2026, **46 states** have enacted laws targeting AI-generated synthetic media. **30 states** now have laws requiring clear disclosures on AI-generated political content. Key state laws include:
- **California**: AB 2839 (faced constitutional challenges); AI training data transparency law took effect [68]
- **Tennessee**: ELVIS Act — prohibits unauthorized AI replication of voices [68]
- **Texas**: TRAIGA (effective January 2026) [68]
- **New York**: 2025 legislation enhancing rights of publicity and voice protections [68]

2026 midterm elections will be the first where 30 state election deepfake laws are in effect [68].

### 7.3 China

China's 2026 AI regulatory regime consists of three key instruments:
1. **Algorithm Recommendation Regulation**: Requires commercial recommendation engines to file technical and risk documentation within 30 days of launch [72]
2. **Deep Synthesis Provisions**: Mandates registration of synthetic media platforms within 15 days, mandatory labeling of AI-generated content, and prohibits high-risk applications like political impersonation [72]
3. **Interim Measures for Generative AI**: Governs LLMs and content-generating AI, requiring registration within 45 days post-training, strict content moderation, data compliance, and visible labeling [72]

**Enforcement**: Fines up to **¥5 million (~$700,000)** , business license suspensions, and criminal charges for severe violations [72].

**Digital Virtual Human Regulations (Draft)**: The CAC released draft regulations targeting AI-generated digital virtual humans, requiring:
- Clear labeling of AI-generated content [73]
- Explicit consent for using individual's likeness, voice, or personal data [73]
- Prohibition on digital humans bypassing biometric authentication [73]
- Special protections for minors: banning virtual intimate relationships and services causing harm or excessive spending [73]
- Public consultation closed **May 6, 2026** [73]

**Mandatory AIGC Labeling Standards**: Effective **September 1, 2025**, requiring visible and technical labels on all AI-generated text, images, audio, and video [74].

**Interim Measures for Anthropomorphic AI**: Published April 10, 2026, effective **July 15, 2026** [74].

**Enforcement Campaign 2026**: The "Qinglang" campaign removed over 3,500 AI-related products, scrubbed nearly 960,000 illegal or harmful content pieces, and shut down or penalized more than 3,700 accounts [74].

### 7.4 United Kingdom

**Criminalization of Non-Consensual Sexual Deepfakes**: The Data (Use and Access) Act 2025 amended the Sexual Offences Act 2003. Under **Section 138**, effective **January 12, 2026**, it is illegal to **create or request** the creation of AI-generated intimate or sexually explicit images of another person without their consent, even if never shared. Platforms must introduce filters to prevent such content [75].

**Government Guidance**: The UK government announced a "world-first" deepfake detection framework in February 2026 [76].

### 7.5 India

#### IT Rules Amendment 2026 (February 10, 2026)
India's Ministry of Electronics and Information Technology issued the Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Amendment Rules, 2026, effective **February 20, 2026**. Key provisions:

- **Synthetically Generated Information (SGI)**: Formally recognizes SGI as "audio, visual, or audio-visual information that is artificially or algorithmically created, generated, modified or altered... in a manner that such information appears to be real, authentic, or true" [77]
- AI-generated text is explicitly excluded; routine edits like filters are also excluded [77]
- **Labeling**: Platforms must mandate users to disclose synthetic content before publication, with prominent visual disclosures, audio warnings, and embedded permanent metadata [77]
- **Accelerated Takedown**: Unlawful or harmful SGI must be removed within **3 hours** of notification; particularly sensitive content (non-consensual deepfake nudity, impersonation) within **2 hours** [77]
- Grievances acknowledged within **2 days**, resolved within **7 days** [77]
- **Significant Social Media Intermediaries (SSMIs)** — platforms with over **5 million users** — face heightened requirements including mandatory deployment of automated tools to detect and prevent unlawful synthetic content [77]
- **Safe Harbor Loss**: Failure to comply results in loss of safe harbor protections under Section 79 of the IT Act [77]

#### Draft IT Second Amendment Rules (March 30, 2026)
Extended Code of Ethics to individual social media users whose posts involve "news and current affairs," classifying them as "digital news publishers." Open for public consultation as of April 2026 [78].

### 7.6 Canada

#### AIDA Status
The **Artificial Intelligence and Data Act (AIDA)** has **not been enacted**. It was part of Bill C-27, which failed in early 2025. The Minister of Artificial Intelligence announced intention to propose a new law that would "not be a repeat of AIDA but instead be its own regulatory initiative." A revised AI strategy is expected in 2026 [79].

Canada has **no dedicated federal deepfake statute**. Regulation occurs through Criminal Code provisions (publication of intimate images without consent, criminal harassment, false messages and impersonation for fraud) and PIPEDA privacy law. Criminal prosecutions of adult non-consensual deepfake cases increased through 2024–2025 [79].

### 7.7 South Korea

#### AI Basic Act (Effective January 22, 2026)
South Korea's **Artificial Intelligence Development and Trust-Based Ecosystem Act** is the first comprehensive law on safe AI usage. Key provisions:
- Formalizes the **Presidential Council on National Artificial Intelligence Strategy** [80]
- Mandates **watermarking and disclosure** for AI-generated content to combat misinformation and deepfakes [80]
- Enhanced oversight for **high-impact AI systems** with substantial computing resources or significant impacts on public operations and rights [80]
- Operators must implement risk management plans and social impact assessments [80]
- **One-year grace period** (2026) emphasizing guidance over penalties [80]
- Foreign AI operators without local Korean entity must appoint domestic representatives [80]

#### Election Deepfake Regulations
South Korea has implemented some of the world's strictest election deepfake regulations: AI-generated content that looks "realistic enough" to confuse voters is banned within **three months** of an election. Penalties: up to **7 years in prison** or fines up to **50 million won (~$38,000)** . Nearly **80% of South Koreans** support harsh penalties against election deepfakes [81].

**First Enforcement Action (February 2026)**: South Korea's first legal action against an election candidate who created and distributed a deepfake AI video intended to mislead voters ahead of the **June 3, 2026, local elections** [81].

**AI Detection Model**: The Ministry of the Interior and Safety and the National Forensic Service jointly developed an AI-based deepfake detection model with **92% accuracy rate** (up from previous 76%), used in the June 3, 2026 local elections [82].

#### Sexual Exploitation Deepfakes
Deepfake cases exceeded **23,000** in 2024. A 2024 case involved coordinated deepfake abuse in over **500 South Korean schools**, with 387 individuals detained (80% teenagers). Despite stronger laws, 40% of deepfake offenders received suspended sentences. South Korea accounts for **more than half** of all deepfake explicit content online globally, with a **550% annual increase** since 2019 [83].

### 7.8 Singapore

#### Model AI Governance Framework v1.5 (May 20, 2026)
The Infocomm Media Development Authority (IMDA) issued an updated MGF specifically targeting Agentic AI systems, building on the January 22, 2026 launch. Four key dimensions:
1. **Assessing and bounding risks upfront** — evaluating use case suitability based on domain sensitivity, agent autonomy, and access scope
2. **Making humans meaningfully accountable** — clear responsibility allocation and effective human oversight
3. **Implementing technical controls and processes** — throughout the agent lifecycle
4. **Enabling end-user responsibility** — via transparency, training, and maintaining foundational skills

Addresses agentic AI risks including hallucination, bias, data leakage, adversarial prompt injections, and multi-agent risks (miscoordination, conflict, collusion, emergent unpredictable behaviors) [84].

Singapore also bans deepfakes in elections [85].

### 7.9 Other Jurisdictions

#### Australia
- **Federal Law (2024)**: Criminalizes sharing non-consensual deepfake sexually explicit material: **6 years imprisonment** for sharing; **7 years** for creating and sharing. First prosecution in February 2026 [86]
- **New South Wales (February 16, 2026)**: Broadened definition of image-based abuse to include both real and digitally fabricated intimate material [86]
- **Queensland (Proposed)**: Legislation making creation of sexually explicit AI deepfake images without consent illegal, with proposed penalty of up to **3 years in prison** [86]

#### Japan
- **AI Basic Act**: Enacted 2025 with **no mandatory compliance requirements, no fines, and no prohibitions**—relies solely on a "name and shame" enforcement mechanism [87]
- **Proposed Penalty Provisions (April 23, 2026)**: Japan's ruling Liberal Democratic Party proposed adding penalty provisions, addressing deepfake pornography and AI-generated copyright infringement [87]
- **Ministry of Justice Study Group (April 17, 2026)**: Eight experts convened to examine civil liabilities related to unauthorized use of individuals' likeness and voices, with plans to meet five times by July 2026 [87]

#### Brazil
- **Electoral AI Rules (TSE)**: One of the world's most comprehensive AI regulatory frameworks for elections, enforced during the **October 4, 2026** general elections. Deepfakes completely prohibited (candidacy annulment as penalty); mandatory labeling of AI campaign materials; platforms have joint civil and administrative liability for failing to promptly remove illegal content; reversal of burden of proof—publishers must prove authenticity when challenged [88]
- **Civil Code Reform (Bill No. 4/2025)**: Introduces Article 2.027-AN regulating creation and use of AI-generated images, including deepfakes, of living and deceased persons [88]

#### Denmark
Denmark is poised to become one of the first countries to grant individuals a **copyright-like right** over their own face and voice. A draft amendment to the Danish Copyright Act proposes inserting a new neighboring right granting every identifiable natural person exclusive, transferable control over AI-generated depictions of their likeness. Consent does **not** discharge labeling obligation under EU AI Act Article 50—both must be satisfied. Final parliamentary adoption pending mid-2026 [89].

### 7.10 International Coordination

#### Council of Europe Framework Convention on AI
The world's **first legally binding international treaty** on AI (Framework Convention on Artificial Intelligence and Human Rights, Democracy and the Rule of Law) became effective **November 1, 2025**. The EU ratified it during the 135th Session of the Committee of Ministers in Chișinău, Moldova. Covers **37+ nations** with principles of transparency, accountability, oversight, non-discrimination, and safeguards against harmful AI uses [90].

#### International AI Safety Report 2026
Published February 2026, led by Yoshua Bengio, authored by over **100 experts** supported by **30+ countries and international organizations** (EU, OECD, UN). Documents AI systems being misused for scams, fraud, blackmail, and non-consensual intimate imagery. Covers risk categories of malicious use, malfunctions, and systemic risks. Open-weight models noted as particularly challenging—"cannot be recalled once released" [91].

#### Five Eyes Coordination
Five Eyes governments urge careful adoption of AI agents, warning against broad access to sensitive systems and stressing the need for governance and oversight [92].

---

## 8. New Benchmarks, Datasets, and Evaluation Frameworks

### 8.1 Major New Benchmarks

#### Microsoft-Northwestern-WITNESS (MNW) Benchmark
Published in IEEE Intelligent Systems (Vol. 59, No. 2, 2026), the MNW benchmark contains **more than 50,000 artifacts** (images, videos, and audio files) created by the authors, along with real-world AI-manipulated or suspicious media identified by journalists and human rights defenders worldwide. Expert annotation ensures practical, high-stakes detection scenarios.

Key features:
- Updated periodically to include emerging AI content generators and adversarial examples employing state-of-the-art attacks [50][51]
- Includes media subjected to realistic post-processing manipulations (resizing, compression) [50][51]
- Developed collaboratively by Microsoft AI for Good, Northwestern University, and WITNESS [50][51]
- "This dataset is intended solely for evaluation purposes and cannot be used for training or commercial purposes" [50][51]
- The dataset creators recommend that "entities purchasing detection solutions avoid using our dataset to evaluate commercial tools" [50][51]

#### Omni-Fake Benchmark
Introduced with the Omni-Fake-R1 model (Section 1.2), this hierarchical benchmark spans four modalities—images, audio, generic videos, and audio-video talking-head videos—with over **1 million in-distribution samples** and **200,000+ out-of-distribution (OOD)** samples. The hierarchical 5-Level design enables fine-grained generalizability evaluation, starting from in-domain detection and progressively testing on increasingly challenging datasets [8].

#### MultiFakeVerse
Contains **845,286 person-centric images** for generative AI forensics. Available under a strict research-only license to limit misuse, reflecting growing concern about responsible data sharing [93].

#### ExDDV: Explainable Deepfake Detection in Video (WACV 2026)
Comprising approximately **5,369 videos** (1,000 real and 4,369 deepfakes) from four collections covering phase swap and diffusion-based methods. Each video annotated with three layers:
1. Human-written textual explanations describing specific artifacts
2. Spatial click annotations marking artifact locations
3. Difficulty ratings (easy, medium, hard)

Evaluated vision-language models (BLIP-2, FiT-5, LLaVA 1.5) under zero-shot, few-shot in-context learning, and fine-tuning regimes. Best performance: BLIP-2 fine-tuned with hard masking achieving sentence BERT score of 0.55. Fine-tuning combined with click supervision significantly improves models' ability to localize and describe deepfake artifacts [94].

#### DF-Wild Dataset (IEEE Signal Processing Cup 2025)
Used in the "Towards Generalizable Deepfake Image Detection with Vision Transformers" study (arXiv 2604.17376), this diverse dataset enabled an ensemble of fine-tuned ViTs (DINOv2, AIMv2, OpenCLIP's ViT-L/14) to achieve an **AUC of 96.77% and EER of 9%** , surpassing the state-of-the-art model Effort by over 7% in AUC and 8% in EER [95].

### 8.2 New Challenges and Competitions

#### NTIRE 2026 Robust Deepfake Detection Challenge
As detailed in Section 1.2, this CVPR 2026 challenge with 337 participants focused on developing deepfake detectors resilient to image degradations. Top methods relied on large foundation models, ensembles, and degradation training [4].

#### NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild
A separate challenge aiming to distinguish real images from AI-generated ones under practical transformations (cropping, resizing, compression, blurring). Used a novel dataset of **108,750 real images** and **185,750 AI-generated images** from **42 different generators**, all subjected to **36 types of image transformations**. 511 registered participants, 20 teams submitting valid final solutions [96].

#### ImageCLEF 2026 Deepfake Detection and Generation Task
Two subtasks:
- **Subtask 1**: Deepfake Generation — participants create high-quality deepfakes based on limited real data
- **Subtask 2**: Deepfake Detection — participants classify media as real or deepfake with **no training data provided** to ensure generalization [97]

#### IJCAI 2026 Deepfake Detection Localization (DDL 2.0)
The DDL Workshop at IJCAI 2026 continued the challenge format from IJCAI 2025, focusing on deepfake detection, localization, and interpretability [98].

#### SAFE Challenge (ResearchGate)
Addressed reliable synthetic video detection under common compression algorithms (H.264, H.265, AV1) as well as other techniques like adding borders, color correction, and frame processing [49].

#### UK Home Office Deepfake Detection Challenge
In collaboration with the Alan Turing Institute, this challenge focuses on practical detection for national security and law enforcement applications [99].

### 8.3 Specialized Evaluation Frameworks

#### SONAR Framework (IBM Research)
The first comprehensive framework to benchmark AI-audio detection uniformly across advanced TTS models, covering 5 traditional and 6 foundation-model-based detection models across 9 diverse audio synthesis platforms [18].

#### DeepFense Evaluation
The open-source PyTorch toolkit (Section 2.1) published **455+ pretrained models** and **12 datasets** at huggingface.co/DeepFense, supporting multi-GPU training and standardized evaluation across audio deepfake detection architectures [22].

#### RADAR Challenge 2026
Multilingual evaluation framework covering six languages with realistic media transformations (compression, noise, reverberation, bandwidth limitation), establishing robust evaluation protocols for production environments [26].

#### FakeSpeech Benchmark (IEEE ISDFS 2026)
Features **970 talking-face clips** (485 real, 485 fake) derived from FakeAVCeleb, with fake samples generated by rewriting transcripts using GPT-4.5 and resynthesizing voices via ElevenLabs. Audio-only baseline achieved 0.85 accuracy on FakeAVCeleb but dropped to **0.67 accuracy** on FakeSpeech, indicating significantly increased dataset complexity for semantic-level manipulations [100].

---

## 9. Conclusion and Future Directions

The period from mid-2026 onward has seen substantial evolution across all dimensions of deepfake detection research, regulation, and ethical consideration.

**Technical Progress**: Foundation model integration has become the dominant paradigm, with DINOv3, CLIP, and vision-language models driving state-of-the-art performance across benchmarks. The NTIRE 2026 Challenge demonstrated that ensembles of large pretrained models with degradation-aware training achieve the highest robustness. In audio, self-supervised models (WavLM, HuBERT, mHuBERT) dominate ASVspoof benchmarks. Multimodal methods continue to outperform unimodal approaches, with phoneme-level alignment and cross-modal synchronization showing particular promise for generalization.

**Benchmark Performance**: GenConViT+ achieves 99.3% average AUC, Face2Parts reaches 98.42% AUC on FF++, PIA achieves 98.7% accuracy on FakeAVCeleb, Resemble AI achieves 98.1% accuracy in the Podonos audio benchmark, and top RADAR Challenge systems achieve 5.10% EER on multilingual evaluation.

**The Generalization Gap**: The Vector Institute's "Generalization Illusion" paper represents a critical intervention, arguing that the five technical assumptions underlying current detectors no longer hold for diffusion-based generators. The 2026 International AI Safety Report characterizes detection technologies as having "limited success." The gap between benchmark and real-world performance remains the field's greatest challenge.

**Privacy and Ethics**: Fully homomorphic encryption enables deepfake detection on encrypted data without accuracy loss. On-device processing solutions like DeFakeQ and TrustLens demonstrate that privacy-preserving detection is increasingly feasible. However, demographic bias remains a serious concern, and the first fairness-focused competition (NeurIPS 2025) highlights growing attention to this issue.

**Regulatory Landscape**: The regulatory environment has intensified dramatically. The EU AI Act's transparency obligations take effect August 2, 2026, with accompanying guidelines released May 8, 2026. The NO FAKES Act was reintroduced in the US on May 27, 2026. India's IT Rules Amendment 2026 mandates 2-hour takedown for deepfake content. South Korea's AI Basic Act took effect January 22, 2026. Other jurisdictions—including Brazil, Denmark, Australia, and Japan—have enacted or proposed new laws. The Council of Europe Framework Convention on AI became the first legally binding international treaty on AI.

**Future Research Priorities**: The field must develop generator-agnostic detectors that handle unseen manipulation types; fairness-aware training methods that eliminate demographic disparities; privacy-preserving architectures without sacrificing detection accuracy; adversarial robustness against both deliberate attacks and natural distribution shifts; and interpretable detection systems that provide human-understandable explanations. The Vector Institute's call to integrate communication-layer analysis with media forensics represents a promising new direction.

---

### Sources

[1] GenConViT: Deepfake Video Detection Using Generative Convolutional Vision Transformer: https://arxiv.org/abs/2307.07036

[2] MINTIME: Multi-Identity Size-Invariant TimeSformer for Video Deepfake Detection: https://arxiv.org/abs/2211.10996

[3] Cross-Dataset Video Deepfake Detection Using Transformer and CNN Architectures (Springer 2026): https://doi.org/10.1007/s00138-026-01809-w

[4] NTIRE 2026 Robust Deepfake Detection Challenge Report: https://arxiv.org/abs/2604.24163

[5] Robust Deepfake Detection: Mitigating Spatial Attention Drift via Calibrated Complementary Ensembles: https://arxiv.org/abs/2604.25889

[6] VLAForge: Unleashing Vision-Language Semantics for Deepfake Video Detection: https://arxiv.org/abs/2603.24454

[7] GazeCLIP: Gaze-Guided CLIP with Adaptive-Enhanced Fine-Grained Language Prompt for Deepfake Detection: https://arxiv.org/abs/2603.29295

[8] Omni-Fake: Benchmarking Unified Multimodal Social Media Deepfake Detection: https://arxiv.org/abs/2605.01638

[9] Face2Parts: Exploring Coarse-to-Fine Inter-Regional Facial Dependencies for Generalized Deepfake Detection: https://arxiv.org/abs/2603.26036

[10] VIGIL: Part-Grounded Structured Reasoning for Generalizable Deepfake Detection: https://vigil.best

[11] Towards Sustainable Universal Deepfake Detection with Frequency-Domain Masking: https://arxiv.org/abs/2512.08042

[12] The Generalization Illusion (Vector Institute): https://arxiv.org/abs/2605.09007

[13] X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection (CVPR 2026): https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_X-AVDT_CVPR_2026_paper.pdf

[14] Deepfake Face Detection Using Hybrid Bag-of-Visual-Words and Deep Learning (Nature Scientific Reports, 2026): https://doi.org/10.1038/s41598-026-53464-w

[15] Deepfake Detection with Full-Stack CNN Application (IJFMR, 2026): https://ijfmr.com/

[16] ASVspoof 5 Challenge: https://arxiv.org/abs/2601.03944

[17] Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR: https://arxiv.org/abs/2603.06164

[18] SONAR: Systematic Analysis over Generative and Detection Models (ACM Transactions on Internet Technology): https://dl.acm.org/journal/toit

[19] Diffusion Reconstruction towards Generalizable Audio Deepfake Detection: https://arxiv.org/abs/2604.26465

[20] RTCFake: Speech Deepfake Detection in Real-Time Communication: https://arxiv.org/abs/2604.23742

[21] Audio Deepfake Detection at the First Greeting: "Hi!" (ICASSP 2026): https://doi.org/10.1109/ICASSP55912.2026.11463587

[22] DeepFense: A Unified, Modular, and Extensible Framework for Robust Deepfake Audio Detection: https://arxiv.org/abs/2604.08450

[23] Audio Avatar Fingerprinting for Authorized Voice Cloning: https://arxiv.org/abs/2603.20165

[24] TFTransformer: A Synthetic Speech Detection Model Combining Local-Global Dependency (CMC, 2026): https://www.techscience.com/cmc/v86n1/64484

[25] Whisper+AASIST for DeepFake Audio Detection: https://faculty.cs.gwu.edu/

[26] RADAR Challenge 2026: https://arxiv.org/abs/2605.09568

[27] Podonos Audio Deepfake Detection Benchmark (May 2026): https://podonos.com/

[28] Audio Deepfake Detectors vs. Real Fraud (WACV 2026 SAFE Workshop): https://openaccess.thecvf.com/content/WACV2026/workshops/safe/

[29] Human Audio Deepfake Perception 2026: https://huggingface.co/datasets/mueller91/human-perception-audio-deepfake-2026

[30] PIA: Toward Multimodal Deepfake Detection via Phoneme-Temporal and Identity-Dynamic Analysis: https://cse.buffalo.edu/tech-reports/2026-03.pdf

[31] ConLLM: Revealing the Truth for Detecting Multi-Modal Deepfakes (EACL 2026): https://arxiv.org/abs/2601.17530

[32] SAVe: Self-Supervised Audio-visual Deepfake Detection: https://arxiv.org/abs/2603.25140

[33] DeepGuard: A Multimodal AI-Based Deepfake Detection System: https://www.ijrdet.com/files/Volume15Issue3/IJRDET_0326_163.pdf

[34] Intra-modal and Cross-modal Synchronization for Audio-visual Deepfake Detection (ICCV 2025): https://openaccess.thecvf.com/content/ICCV2025/papers/Anshul_Intra-modal_and_Cross-modal_Synchronization_ICCV_2025_paper.pdf

[35] LipFD: Lips Are Lying—Spotting Temporal Inconsistency in Lip-Syncing DeepFakes (NeurIPS 2024): https://arxiv.org/abs/2401.15668

[36] TrustLens: A Multimodal AI-Based Deepfake Detection System (IJSREM, 2026): https://ijsrem.com/uploads/production/IJSREM60453.pdf

[37] DeFakeQ: Enabling Real-Time Deepfake Detection on Edge Devices via Adaptive Bidirectional Quantization: https://arxiv.org/abs/2604.08847

[38] Privacy-preserving deepfake detection with fully homomorphic encryption (NTU): https://www.ntu.edu.sg/

[39] Blockchain and Federated Learning Synergy for Privacy-Focused DeepFex Solutions (Springer, 2025): https://link.springer.com/book/9789819612345

[40] Hybrid Federated Deepfake Detection via Residual-Aware Temporal Modeling: https://www.researchgate.net/

[41] Privacy-focused artificial intelligence model for detecting deepfake-based cyber threats (WJARR, 2026): https://wjarr.com/

[42] Real-Time Multi-Modal Deepfake Detection for Edge Devices Using Lightweight CNNs (JSRT): https://www.jsrtjournal.com/index.php/JSRT/article/view/422

[43] TPDP 2026: Theory and Practice of Differential Privacy Workshop: https://tpdp2026.org/

[44] Navigating Trade-offs in Differentially Private Machine Learning (AAAI 2026): https://aaai.org/

[45] Biometric Security in the Deepfake Era 2026 (Mantra Softech): https://mantrasoftech.com/

[46] DeepFake-Eval-2024 Study (Reality Defender): https://realitydefender.com/

[47] Brightside AI Blog on Commercial Tool Performance: https://brightsideai.com/

[48] 2026 International AI Safety Report: https://aisafetyreport.org/

[49] SAFE Challenge: https://www.researchgate.net/publication/404713036

[50] Microsoft-Northwestern-WITNESS (MNW) Benchmark (IEEE Computer, 2026): https://doi.org/10.1109/MEX.2026.11479406

[51] MNW Dataset: https://microsoft.com/mnw-benchmark

[52] Sora 2 Detection Analysis: https://openai.com/sora

[53] AI Video Generation Models 2026 Guide: https://aimodelsguide.com/

[54] Cross-Domain Generalization Limits of Vision Foundation Models: https://arxiv.org/abs/2605.24965

[55] Bias-Free? An Empirical Study on Ethnicity, Gender, and Age Fairness in Deepfake Detection (ACM 3796544): https://dl.acm.org/doi/10.1145/3796544

[56] European Parliament Briefing on Women and AI-Enabled Disinformation (2026): https://europarl.europa.eu/

[57] Adversarial Attacks and Defences in Deepfake Detection (IJERT, 2026): https://ijert.org/

[58] Deepfake Growth Statistics: https://deepstrike.com/

[59] NeuriPS 2025 Fairness Competition in Deepfake Detection (Machine Intelligence Research, Springer): https://link.springer.com/journal/11633

[60] Ethical Implications of Deepfake Image Generation and Detection (JETIR, March 2026): https://jetir.org/

[61] Liar's Dividend Analysis (EkasCloud): https://ekascloud.com/

[62] Unmasking Digital Deceptions (ScienceDirect, 2025): https://sciencedirect.com/

[63] AI Surveillance Privacy Concerns 2026 (Coram AI): https://coram.ai/

[64] Deepfake Statistics 2026: https://deepfakestats.com/

[65] Reality Defender 2026 Outlook: https://realitydefender.com/outlook2026

[66] EU AI Act Article 50 Draft Guidelines (May 8, 2026): https://ec.europa.eu/

[67] EU Ban on Deepfake Nudification Tools (May 7, 2026): https://europa.eu/

[68] US State Deepfake Legislation Tracker (NCSL): https://www.ncsl.org/technology-and-communication/deepfake-legislation

[69] NO FAKES Act of 2026 (Introduced May 27, 2026): https://www.congress.gov/bill/119th-congress/senate-bill/4875

[70] DEFIANCE Act of 2025 (S.1837): https://www.congress.gov/bill/118th-congress/senate-bill/1837

[71] AI Fraud Accountability Act of 2026 (S. 3982): https://www.congress.gov/bill/119th-congress/senate-bill/3982

[72] China's AI Regulatory Framework 2026: https://www.cac.gov.cn/

[73] China Digital Virtual Human Regulations (Draft): https://www.cac.gov.cn/2026-digital-virtual-human

[74] China Mandatory AIGC Labeling Standards: https://www.cac.gov.cn/

[75] UK Data (Use and Access) Act 2025 – Section 138: https://www.legislation.gov.uk/

[76] UK Deepfake Detection Framework (February 2026): https://www.gov.uk/

[77] India IT Rules Amendment 2026: https://www.meity.gov.in/

[78] India Draft IT Second Amendment Rules (March 30, 2026): https://www.meity.gov.in/

[79] Canada AIDA Status and Deepfake Regulation: https://www.parl.ca/

[80] South Korea AI Basic Act (Effective January 22, 2026): https://www.msit.go.kr/

[81] South Korea Election Deepfake Regulations: https://www.nec.go.kr/

[82] South Korea AI Detection Model (92% Accuracy): https://www.nfs.go.kr/

[83] South Korea Sexual Exploitation Deepfakes: https://www.korea.net/

[84] Singapore Model AI Governance Framework v1.5 (May 20, 2026): https://www.imda.gov.sg/

[85] Singapore Deepfake Election Ban: https://www.singapore.gov.sg/

[86] Australia Deepfake Laws: https://www.legislation.gov.au/

[87] Japan AI Basic Act and Proposed Penalty Provisions: https://www.kantei.go.jp/

[88] Brazil Electoral AI Rules (TSE): https://www.tse.jus.br/

[89] Denmark Copyright-Like Right Over Face and Voice (Draft): https://www.kum.dk/

[90] Council of Europe Framework Convention on AI: https://www.coe.int/

[91] International AI Safety Report 2026: https://aisafetyreport.org/

[92] Five Eyes AI Advisory: https://www.fiveeyes.com/

[93] MultiFakeVerse Dataset: https://multifakeverse.org/

[94] ExDDV: Explainable Deepfake Detection in Video (WACV 2026): https://github.com/vladhondru25/ExDDV

[95] Towards Generalizable Deepfake Image Detection with Vision Transformers: https://arxiv.org/abs/2604.17376

[96] NTIRE 2026 Challenge on Robust AI-Generated Image Detection in the Wild: https://cvpr.ntire.org/

[97] ImageCLEF 2026 Deepfake Detection and Generation Task: https://imageclef.org/2026/deepfake-detection-and-generation

[98] IJCAI 2026 DDL 2.0 Workshop: https://ijcai.org/

[99] UK Home Office Deepfake Detection Challenge: https://www.turing.ac.uk/

[100] FakeSpeech Benchmark (IEEE ISDFS 2026): https://doi.org/10.1109/ISDFS.2026.11459048