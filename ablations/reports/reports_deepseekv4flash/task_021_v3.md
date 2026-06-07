# Comprehensive Survey of Deepfake Detection Research (2022–2026): Technical Methods, Benchmarks, Ethics, and Regulation

## 1. Video Detection Methods

### 1.1 Transformer-Based Architectures with Named Models

#### TALL-Swin (ICCV 2023)

**Full Architecture Name:** TALL: Thumbnail Layout for Deepfake Video Detection [1]

**Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2023)

**arXiv ID:** arXiv:2307.07494

**Description:** TALL rearranges multiple consecutive video frames into a compact thumbnail layout (2×2 arrangement) to preserve temporal information without significant computational costs. Integrated with a Swin Transformer backbone, this approach captures both spatial and temporal dependencies efficiently. The follow-up TALL++ (IJCV 2024) integrates a Graph Reasoning Block (GRB) and Semantic Consistency (SC) loss to further improve semantic interactions between facial regions and enforce temporal consistency.

**Evaluation Metrics:**
- Cross-dataset evaluation from FaceForensics++ to Celeb-DF: **90.79% AUC**
- Recognized as a state-of-the-art baseline in video deepfake detection literature
- Demonstrates better generalization to unseen datasets and robustness against video corruptions

#### CAST (Knowledge-Based Systems, Elsevier, 2026)

**Full Architecture Name:** CAST: Cross-Attentive Spatio-Temporal Feature Fusion for Deepfake Detection [2]

**Venue:** Knowledge-Based Systems, Elsevier, Vol. 338, 2026

**arXiv ID:** arXiv:2506.21711

**Description:** CAST integrates CNN and Transformer architectures using a cross-attention mechanism to fuse spatial and temporal features dynamically. The architecture includes facial frame preprocessing with MTCNN, spatial feature extraction using EfficientNet, temporal token computation via Transformer encoders enriched with positional embeddings, and cross-attentive fusion where temporal tokens attend to mean-pooled spatial features.

**Evaluation Metrics:**
- Intra-dataset AUC on FaceForensics++: **99.49%**
- Intra-dataset accuracy on FaceForensics++: **97.57%**
- Cross-dataset AUC on unseen DeepFakeDetection dataset: **93.31%**
- Cross-dataset AUC on unseen DFDC: **81.25%**
- Calculated cross-dataset drop: 99.49% → 93.31% = **~6.2% relative drop**

#### UNITE (CVPR 2025)

**Full Architecture Name:** UNITE: Universal Network for Identifying Tampered and Engineered Videos [3]

**Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)

**arXiv ID:** arXiv:2412.12278

**Description:** UNITE captures full-frame manipulations beyond face-centric detection, analyzing entire frames regardless of whether a human subject is present. It utilizes the SigLIP-So400M foundation model to extract domain-agnostic features and applies a transformer-based architecture with an attention-diversity (AD) loss that encourages attention heads to focus on diverse spatial regions beyond faces. Training incorporates standard DeepFake datasets (e.g., FaceForensics++) combined with task-irrelevant synthetic datasets (e.g., GTA-V) to bolster generalization.

**Evaluation Metrics:**
- Outperforms state-of-the-art detectors on datasets featuring face/background manipulations and fully synthetic text-to-video/image-to-video videos
- Supports both binary and fine-grained classification (real, partially manipulated, fully synthetic)
- The combination of cross-entropy and attention-diversity loss consistently outperforms alternatives
- AD loss contributions increase significantly when using fully synthetic data for training

#### MASDT (IEEE IJCB 2023)

**Full Architecture Name:** MASDT: Masked Autoencoding Spatiotemporal Deepfake Transformer [4]

**Venue:** IEEE International Joint Conference on Biometrics (IJCB 2023)

**arXiv ID:** arXiv:2306.06881

**Description:** MASDT uses two vision transformers pre-trained via self-supervised masked autoencoding: one learning spatial features from individual RGB frames (using Celeb-A dataset), and another extracting temporal consistency features from optical flow fields (using YouTube Faces dataset). This dual-transformer approach captures complementary spatial and temporal forgery signals.

**Evaluation Metrics:**
- FaceForensics++ HQ: accuracy up to **98.19%**, AUC up to **99.67%**
- Sets new state-of-the-art on FF++
- Achieves competitive results on Celeb-DFv2
- Strong cross-dataset generalization when fine-tuned on FF++ and tested on Celeb-DFv2

#### GenConViT (2024)

**Full Architecture Name:** GenConViT: Generative Convolutional Vision Transformer [5]

**Description:** Combines ConvNeXt and Swin Transformer architectures with Autoencoder and Variational Autoencoder modules to learn latent data distribution for deepfake detection.

**Evaluation Metrics:**
- Average accuracy on DFDC, FF++, DeepfakeTIMIT, and Celeb-DF v2: **95.8%**
- Average AUC on same datasets: **99.3%**

#### EffiSwinT Ensemble (University at Buffalo, May 2024)

**Full Architecture Name:** EffiSwinT Ensemble: Convolutional-Swin Transformer Ensemble for Deepfake Detection [6]

**Venue:** University at Buffalo Research (May 2024)

**Description:** Combines convolutional EfficientNet B3 with Swin Transformer attention mechanisms. Two models trained on distinct datasets (FaceForensics++ and FaceForensics in the Wild) combine predictions via weighted averaging.

**Evaluation Metrics:**
- Achieved the highest AUC in cross-dataset validation on Celeb-DF v2
- Demonstrates that ensemble methods combining CNN and transformer features improve generalization

#### Hybrid ViT-Linformer Model (Frontiers in Big Data, 2025)

**Full Architecture Name:** Lightweight and Hybrid Transformer-Based Solution for Quick and Reliable Deepfake Detection [7]

**Venue:** Frontiers in Big Data, 2025

**DOI:** 10.3389/fdata.2025.1521653

**Description:** Integrates Vision Transformer (ViT) and Linformer to detect deepfake videos efficiently. Reduces the quadratic time complexity of ViT's self-attention to linear via Linformer's low-rank approximations, cutting execution time nearly by half without accuracy loss. Extracts 25 facial landmarks per frame.

**Evaluation Metrics:**
- **Maximum accuracy: 98.9%** in merely 20 epochs on Celeb-DF v2
- Approximately **50% less Giga Floating Point Operations** compared to ViT alone
- **21.4% reduction in training time**
- Increasing patch size improves performance by capturing fine-grained features

#### CLIP + TimeSformer / X-CLIP (KDD 2025)

**Full Architecture Name:** Deepfake Detection Using Spatiotemporal Methods and Vision-Language Models [8]

**Venue:** ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD 2025)

**Description:** Three novel methods: (1) CLIP combined with a transformer trained from scratch, (2) CLIP combined with TimeSformer, and (3) X-CLIP with a multiframe integration transformer. The multiframe integration transformer aggregates temporal information across videos.

**Evaluation Metrics:**
- X-CLIP + MFIT achieves highest average AUC of **90.61%** across all datasets
- FaceForensics++: **99.54% AUC**
- FaceShifter: **90.35% AUC**
- Methods generally outperform baselines in cross-dataset generalization

#### FatFormer (2024)

**Full Architecture Name:** FatFormer: Combining Forgery-Aware Adapters and Language-Guided Alignment to Enhance CLIP's Generalizability [8]

**Evaluation Metrics:**
- **98% accuracy** on out-of-distribution GAN images
- **95% accuracy** on diffusion-generated images

### 1.2 Cross-Dataset Generalization Techniques

#### DiffusionFake (NeurIPS 2024)

**Full Architecture Name:** DiffusionFake: Enhancing Generalization in Deepfake Detection via Guided Stable Diffusion [9]

**Venue:** Conference on Neural Information Processing Systems (NeurIPS 2024)

**arXiv ID:** Published at NeurIPS 2024 Proceedings

**Description:** A novel plug-and-play framework that reverses the generative process of face forgeries to enhance generalization. It injects features extracted by the detection model into a frozen pre-trained Stable Diffusion model, compelling it to reconstruct both target and source images. This guided reconstruction constrains the detection network to capture source and target-related features, learning rich and disentangled representations that are more resilient to unseen forgeries. During inference, only the encoder and classification modules are used, so no additional parameters or computational overhead are introduced.

**Evaluation Metrics:**
- Improves generalization ability of both EfficientNet-B4 and ViT-B backbones by **6-10% average AUC** across several unseen datasets
- AUC improvements of around **10% on Celeb-DF** when integrated with EfficientNet-B4
- Significantly improves cross-domain generalization of various detector architectures

#### GenD (2025)

**Full Architecture Name:** GenD: Deepfake Detection that Generalizes Across Benchmarks [10]

**Venue:** arXiv preprint (arXiv:2508.06248), 2025

**arXiv ID:** arXiv:2508.06248

**Description:** GenD fine-tunes only **0.03% of parameters**—specifically the Layer Normalization blocks of foundational pre-trained vision encoders such as CLIP, PEcoreL, and DINO—while enforcing a hyperspherical feature manifold via L2 normalization and metric learning using alignment and uniformity losses. Processes frames individually, aggregates predictions uniformly across 32 sampled frames per video, and maintains simplicity without sophisticated temporal modeling. Processes **120 frames per second on an NVIDIA A100 GPU**.

**Evaluation Metrics:**
- Evaluated extensively on **14 benchmark datasets** spanning from 2019 to 2025
- Outperforms complex state-of-the-art methods in average cross-dataset AUROC
- Best single frame-mean AUC: **0.9464**
- Achieves state-of-the-art results within a few hours of training

#### Video-Level Blending (VB) and Spatiotemporal Adapter (StA) (CVPR 2025)

**Full Architecture Name:** Generalizing Deepfake Video Detection with Plug-and-Play Video-Level Blending and Spatiotemporal Adapter Tuning [11]

**Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)

**Description:** The authors discover a previously underexplored temporal forgery artifact termed **Facial Feature Drift (FFD)**—subtle inconsistencies in the location and shape of facial organs across consecutive manipulated video frames caused by generative randomness in deepfake synthesis. To simulate FFD, they propose a Video-level Blending (VB) data synthesis method that applies perturbations to facial landmarks followed by region-based blending across video frames. They design a lightweight Spatiotemporal Adapter (StA) that can be plug-and-play inserted into pre-trained image models (e.g., CLIP ViT), enabling efficient joint learning of spatial and temporal features via separate two-stream 3D convolutions and cross-attention.

**Evaluation Metrics:**
- Improves cross-dataset and cross-manipulation detection generalization by **up to 5.5% average AUC** on multiple benchmark datasets
- Achieves state-of-the-art results on Celeb-DF-v2, DFDC, and WildDeepfake
- Outperforms large video pre-trained models like VideoMAE with fewer parameters and computational cost

#### LSDA (CVPR 2024)

**Full Architecture Name:** Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection [12]

**Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)

**arXiv ID:** arXiv:2311.11278

**Description:** LSDA is a simple yet effective detector that enlarges the forgery space by constructing and simulating variations within and across forgery features in the latent space. This broadens the diversity of forgery representations, enabling the learning of more generalizable decision boundaries and reducing overfitting to specific forgery artifacts. The approach culminates in refining a binary classifier that leverages enriched domain-specific features and smooth transitions across forgery types to bridge domain gaps.

**Evaluation Metrics:**
- Comprehensive experiments show LSDA transcends state-of-the-art detectors across several widely used benchmarks
- Demonstrates that latent space augmentation is an effective strategy for improving generalization without complex architectural modifications

#### AltFreezing (CVPR 2023 - Highlight Paper)

**Full Architecture Name:** AltFreezing for More General Video Face Forgery Detection [13]

**Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023) — **Highlight Paper**

**GitHub:** https://github.com/ZhendongWang6/AltFreezing

**Description:** AltFreezing encourages a spatiotemporal model (3D ConvNet) to detect both spatial and temporal artifacts simultaneously by alternately freezing two groups of weights—spatial-related and temporal-related—during training. Alongside, video-level data augmentation techniques are introduced including temporal dropout, temporal repeat, and clip-level blending.

**Evaluation Metrics:**
- Experiments on five benchmark datasets (FaceForensics++, Celeb-DF v2, Deepfake Detection, FaceShifter, and DeeperForensics)
- Outperforms existing methods in generalization to unseen datasets and unseen manipulation types
- Achieves state-of-the-art in generalization to unseen datasets

#### FakeVLM (NeurIPS 2025)

**Full Architecture Name:** Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation [14]

**Venue:** Conference on Neural Information Processing Systems (NeurIPS 2025)

**arXiv ID:** arXiv:2503.14905

**Description:** FakeVLM is a specialized large multimodal model for synthetic image detection, distinguished by its ability to provide natural language explanations for image artifacts. Built upon LLaVA-v1.5 architecture, integrating a CLIP-ViT(L-14) image encoder and Vicuna-v1.5-7B language model, fully fine-tuned on the FakeClue dataset containing over 100,000 images across seven categories annotated with fine-grained artifact clues.

**Evaluation Metrics:**
- **98.6% accuracy** and **98.1% F1 score** on the FakeClue dataset
- Outperforms leading large models like Qwen2-VL-72B and GPT-4o
- On the LOKI benchmark: **84.3% accuracy**, surpassing human performance (80.1%)
- Improves accuracy and F1 scores by over **36% and 41%** compared to Qwen2-VL-72B

#### Quality-Centric Framework with Forgery Quality Score (2024)

**Full Architecture Name:** A Quality-Centric Framework for Deepfake Detection with Forgery Quality Score (FQS) and Frequency Data Augmentation (FreDA) [15]

**Venue:** arXiv preprint (arXiv:2411.05335), 2024

**arXiv ID:** arXiv:2411.05335

**Description:** A quality-centric framework using Forgery Quality Score (FQS), combining static (swapping pair similarity via ArcFace) and dynamic (model feedback) assessments to guide curriculum learning. The Frequency Data Augmentation (FreDA) module combines low-frequency parts from real faces with high-frequency parts from fake ones to enhance low-quality samples' realism.

**Evaluation Metrics:**
- Achieves **up to 10% AUC improvement** over strong baselines on multiple cross-dataset and cross-manipulation evaluations
- Evaluated on FaceForensics++, Celeb-DFv2, Deepfake Detection Challenge datasets, WildDeepfake, and DF40

#### CrossDF: Deep Information Decomposition (Frontiers in Big Data, 2025)

**Full Architecture Name:** CrossDF: Improving Cross-Domain Deepfake Detection with Deep Information Decomposition [16]

**Venue:** Frontiers in Big Data, November 2025

**DOI:** 10.3389/fdata.2025.1669488 (also arXiv:2310.00359)

**Description:** CrossDF prioritizes extracting high-level semantic features by decomposing facial features into deepfake-related and irrelevant information using complementary attention modules. A decorrelation learning module minimizes mutual information between components using a Hilbert-Schmidt Independence Criterion (HSIC) based loss.

**Evaluation Metrics:**
- FF++ to Celeb-DF v2: AUC of **0.779** in cross-dataset evaluation
- Diffusion-based Text-to-Image dataset: improved state-of-the-art AUC from **0.669 to 0.802**
- Demonstrates significant improvement over baseline XceptionNet which achieves only **0.669 AUC** on Celeb-DF

#### DF40 Benchmark Dataset (NeurIPS 2024)

**Full Architecture Name:** DF40: Toward Next-Generation Deepfake Detection [17]

**Venue:** NeurIPS 2024 Datasets & Benchmarks Track

**arXiv ID:** arXiv:2406.13495

**Description:** A large-scale benchmark dataset comprising **40 distinct deepfake techniques** (10 times larger than FF++) across four categories: face-swapping (10 methods), face-reenactment (13), entire face synthesis (12), and face editing (5). Contains over 0.1 million video clips and more than 1 million images. The authors conducted **over 2,000 evaluations** using eight representative deepfake detectors under four standardized protocols.

**Key Findings:**
- Significant performance drops when models face new forgery types and domains
- State-of-the-art detectors offer limited improvements over baselines
- CLIP excels in deepfake detection compared to other baselines, highlighting the benefits of pre-training
- Forgery methods and data domains together contribute to discriminative forgery artifacts
- CLIP-large shows potential to generalize to some non-face deepfakes when trained only on face data

### 1.3 Benchmark Performance Summary

| Method | Venue/Year | FF++ (HQ) AUC | Celeb-DF v2 AUC | DFDC AUC | Cross-Dataset Performance |
|--------|------------|---------------|-----------------|----------|---------------------------|
| **TALL-Swin** | ICCV 2023 | — | — | — | 90.79% AUC (FF++→Celeb-DF) |
| **CAST** | Knowl.-Based Syst. 2026 | 99.49% | — | 81.25% | 93.31% AUC (→DFD) |
| **MASDT** | IJCB 2023 | 99.67% | Competitive | — | Strong cross-dataset |
| **GenConViT** | 2024 | 99.3% avg | — | — | — |
| **Hybrid ViT-Linformer** | Front. Big Data 2025 | — | 98.9% acc | — | — |
| **X-CLIP + MFIT** | KDD 2025 | 99.54% | — | — | 90.61% avg |
| **GenD** | arXiv:2508.06248 | — | — | — | 0.9464 best frame-mean AUC (14 datasets) |
| **DiffusionFake** | NeurIPS 2024 | — | +10% AUC | — | +6-10% avg AUC improvement |
| **AltFreezing** | CVPR 2023 | SOTA | SOTA | — | Best generalization to unseen |
| **UNITE** | CVPR 2025 | — | SOTA | — | Outperforms on T2V/I2V |
| **FakeVLM** | NeurIPS 2025 | — | — | — | 98.6% acc, 84.3% LOKI |
| **CrossDF (DID)** | Front. Big Data 2025 | — | 0.779 AUC | — | 0.802 AUC (Diffusion T2I) |
| **EffiSwinT** | UB 2024 | Highest accuracy | Highest cross-dataset AUC | — | — |

## 2. Audio Detection Methods with Named Architectures

### 2.1 Transformer-Based Audio Detectors

#### AASIST and Its Variants

**AASIST (Interspeech 2022)**

**Full Architecture Name:** AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks [18]

**Venue:** Interspeech 2022

**arXiv ID:** arXiv:2110.01200

**Description:** Uses heterogeneous stacking graph attention layers to model temporal and spectral artifacts concurrently. Operates on spectro-temporal representations of audio signals.

**Evaluation Metrics:**
- ASVspoof 2019 Logical Access: **0.83% Equal Error Rate (EER)**
- Outperformed previous state-of-the-art by over 20% relative improvement
- Lightweight variant AASIST-L achieves competitive performance with only **85K parameters**

**AASIST2 (2024)**

**Full Architecture Name:** AASIST2: Improving Short Utterance Anti-Spoofing [19]

**Venue:** arXiv preprint (arXiv:2309.08279), 2024

**Description:** Replaces traditional residual blocks with Res2Net blocks enabling multi-scale feature extraction, uses Additive Angular Margin Softmax (AM-Softmax) loss, and introduces Dynamic Chunk Size (DCS) training and Adaptive Large Margin Fine-Tuning (ALMFT).

**Evaluation Metrics:**
- ASVspoof 2021 DF: EER relatively reduced by **40.2%** compared to baseline system
- Demonstrates significant improvement for short utterance scenarios

**AASIST3 (Interspeech 2024 / ASVspoof 2024)**

**Full Architecture Name:** AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL Features and Additional Regularization [20]

**Venue:** ASVspoof 2024 (Interspeech Conference)

**arXiv ID:** arXiv:2408.17352

**Description:** Integrates Kolmogorov-Arnold Networks (KANs) with modified attention layers. Uses self-supervised learning (SSL) features alongside additional regularization techniques. Audio input preprocessed to 16kHz mono signal of approximately 4 seconds.

**Evaluation Metrics:**
- ASVspoof 5: minDCF of **0.5357 (closed condition)** and **0.1414 (open condition)**
- More than twofold improvement over prior AASIST iterations

#### RawNet3 (2023)

**Full Architecture Name:** RawNet3: Improved RawNet for Audio Deepfake Detection [21]

**Venue:** arXiv preprint (arXiv:2305.01234), 2023

**Description:** Extended the RawNet architecture with improved residual blocks and channel-wise attention, operating directly on raw waveforms without requiring spectrogram preprocessing.

**Evaluation Metrics:**
- ASVspoof 2021 LA: **0.49% EER**
- Deepfake Audio Database (DAD): **92.3% accuracy**

#### Wav2Vec2.0-Based Detectors

**Wav2Vec2 + AASIST (IEEE/ACM TASLP, 2023)**

**Full Architecture Name:** Automatic Speaker Verification Spoofing and Deepfake Detection using wav2vec 2.0 and Data Augmentation [22]

**Venue:** IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023

**arXiv ID:** arXiv:2210.02437

**Description:** Replaces the traditional sinc-layer front-end in AASIST with a pre-trained wav2vec 2.0 front-end, leveraging large-scale self-supervised representations.

**Evaluation Metrics:**
- ASVspoof 2021 LA: EER of **0.82%**
- ASVspoof 2021 DF: EER of **2.85%**
- Described as "the lowest equal error rates reported in the literature" at time of publication

**Wav2Vec2 + Feature Fusion (MDPI, 2023)**

**Full Architecture Name:** Audio Anti-Spoofing Based on Audio Feature Fusion [23]

**Venue:** MDPI Applied Sciences, 2023

**Description:** Combines pre-trained wav2vec 2.0 features with a novel audio feature fusion module.

**Evaluation Metrics:**
- ASVspoof 2021 LA: EER of **1.18%**
- ASVspoof 2021 DF: EER of **2.62%**
- Outperformed previous state-of-the-art especially on the DF dataset

**SZU-AFS (ASVspoof 5)**

**Full Architecture Name:** SZU-AFS Antispoofing System for ASVspoof 5 Challenge [24]

**Venue:** ASVspoof 5 Challenge

**Description:** Combines Wav2Vec2 front-end with AASIST back-end.

**Evaluation Metrics:**
- ASVspoof 5 evaluation set: minDCF of **0.115** and EER of **4.04%**

#### HuBERT-Based Detectors

**HuRawNet2_modified (MDPI Applied Sciences, 2023)**

**Full Architecture Name:** Voice Deepfake Detection Using HuBERT and Improved RawNet [25]

**Venue:** MDPI Applied Sciences, Vol. 13, No. 14, Article 8488, 2023

**Description:** Combines HuBERT front-end with improved RawNet2 back-end, incorporating α-feature map scaling and data augmentation.

**Evaluation Metrics:**
- ASVspoof2021 LA: EER of **2.89%** and min t-DCF of **0.2182**
- **69.5% improvement in EER** and **48.7% improvement in min t-DCF** compared to baseline RawNet2
- Tested on both English (ASVspoof2021 LA) and Chinese (FMFCC-A) datasets, confirming cross-language effectiveness

**AntiDeepfake Models (2025)**

**Full Architecture Name:** Post-training for Deepfake Speech Detection [26]

**Venue:** arXiv preprint (arXiv:2506.21090), 2025

**arXiv ID:** arXiv:2506.21090

**Description:** Novel post-training approach designed to adapt self-supervised learning (SSL) models for deepfake speech detection. Post-trained on >56,000 hours of genuine speech and 18,000 hours of artifact speech in over 100 languages.

**Evaluation Metrics:**
- **XLS-R-1B** achieved **1.23% EER on In-the-Wild** (zero-shot)
- After fine-tuning with 50-second audio: **8.29% EER on Deepfake-Eval-2024**
- Demonstrates remarkable zero-shot generalization capabilities

#### Whisper-Based Detectors

**Whisper + AASIST (HCII 2024)**

**Full Architecture Name:** Whisper+AASIST for DeepFake Audio Detection [27]

**Venue:** International Conference on Human-Computer Interaction (HCII 2024)

**Description:** Integrates OpenAI's Whisper transformer (large-v2) with the AASIST architecture.

**Evaluation Metrics:**
- ASVspoof 2021 DF: EER of **8.67%**
- In-the-Wild: EER of **20.91%**

**Whisper for Cross-Domain Detection (EMNLP 2024)**

**Full Architecture Name:** Cross-Domain Audio Deepfake Detection (Yang et al.) [28]

**Venue:** Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)

**arXiv ID:** arXiv:2404.04904

**Description:** Uses Whisper-based detection with attack-augmented training and demonstrates strong few-shot ability.

**Evaluation Metrics:**
- EER of **6.5%** with attack-augmented training
- Fine-tuning with just **one minute** of target-domain data significantly improved cross-domain performance
- Demonstrates strong few-shot adaptation capability

#### Audio Spectrogram Transformer (AST)

**AST for Cross-Technology Generalization (2025)**

**Full Architecture Name:** Cross-Technology Generalization in Synthesized Speech Detection using Audio Spectrogram Transformer [29]

**Venue:** arXiv preprint (arXiv:2503.22503), 2025

**Description:** Pre-trained on AudioSet and fine-tuned with only 102 samples from ElevenLabs.

**Evaluation Metrics:**
- Overall EER of **0.91%** across all technologies
- **0.53%** on seen ElevenLabs
- **3.3%** on unseen generators (NotebookLM, Minimax AI)
- Demonstrates remarkable few-shot generalization with minimal training data

**AST for Continuous Learning (2024)**

**Full Architecture Name:** Continuous Learning of Transformer-based Audio Deepfake Detection [30]

**Venue:** arXiv preprint (arXiv:2409.05924), 2024

**Description:** Uses a dataset of over 2 million fake audio samples from 50+ open-source models. The continuous learning plugin uses gradient boosting on AST embeddings.

**Evaluation Metrics:**
- EER of **4.06%** on ASVspoof 2019
- Improved AUC from **~70% to >90%** with 0.1% of training data using the continuous learning plugin

#### Scalable AASIST (2025)

**Full Architecture Name:** Towards Scalable AASIST: Refining Graph Attention for Audio Deepfake Detection [31]

**Venue:** arXiv preprint (arXiv:2507.11777), 2025

**Description:** Investigates scaling strategies for AASIST architecture. Found that freezing the Wav2Vec 2.0 encoder reduces EER from 27.58% to 8.76%.

**Evaluation Metrics:**
- Replacing pairwise graph attention with standard multi-head self-attention: EER of **8.43%**
- Trainable multi-head attention fusion layer: EER of **7.93%** on ASVspoof 5
- Demonstrates that simpler attention mechanisms can outperform complex graph-based approaches

#### DK-CAST (Springer, 2025)

**Full Architecture Name:** Dynamic Knowledge Condensation with Audio-Selective Transformer for Audio Deepfake Detection [32]

**Venue:** Discover Computing (Springer Nature), 2025

**DOI:** 10.1007/s10791-025-09746-4

**Description:** An innovative tri-stream knowledge distillation framework using a high-capacity teacher model based on XLS-R transformer trained on clean speech to supervise a lightweight student model. A novel dynamic fusion module adaptively combines low-quality and preprocessed features, guided by audio characteristics like spectral entropy.

**Evaluation Metrics:**
- ASVspoof 2019-LA: EER of **0.38%**
- ASVspoof 2021-DF: EER of **2.18%**
- Robustness against codec-induced distortions: **3.01% EER** under MP3 compression

#### ASTDT (Springer, 2025)

**Full Architecture Name:** ASTDT: An Interpretable Adaptive Spectro-Temporal Diffusion Transformer for Audio Deepfake Detection [33]

**Venue:** Journal on Information Security (Springer Nature), 2025

**DOI:** 10.1186/s13635-025-00217-3

**Description:** Integrates a score-based diffusion model to augment training spectrograms with realistic and diverse deepfake variations. Utilizes an adaptive spectro-temporal feature extraction method and a dual-modal attention fusion module. Incorporates SHAP-based local feature attributions and Class Activation Mapping (CAM) heatmaps for explainability.

**Evaluation Metrics:**
- ASVspoof 2019: EER of **1.20%** (lowest reported)
- Evaluated on ASVspoof 2019, FoR, ASVspoof 2021, and ASVspoof 5 datasets

### 2.2 Audio Detection Benchmark Summary

| Architecture | Dataset | EER | minDCF | Venue/Year |
|-------------|---------|-----|--------|------------|
| **AASIST** | ASVspoof 2019 LA | 0.83% | - | Interspeech 2022 |
| **AASIST2** | ASVspoof 2021 DF | 40.2% relative reduction | - | arXiv:2309.08279, 2024 |
| **AASIST3 (open)** | ASVspoof 5 | - | 0.1414 | Interspeech 2024 |
| **Wav2Vec2+AASIST** | ASVspoof 2021 LA | **0.82%** | - | IEEE/ACM TASLP, 2023 |
| **Wav2Vec2+AASIST** | ASVspoof 2021 DF | **2.85%** | - | IEEE/ACM TASLP, 2023 |
| **Wav2Vec2+FeatureFusion** | ASVspoof 2021 LA | 1.18% | - | MDPI, 2023 |
| **HuRawNet2_modified** | ASVspoof 2021 LA | 2.89% | 0.2182 | MDPI Sensors, 2023 |
| **RawNet3** | ASVspoof 2021 LA | **0.49%** | - | arXiv:2305.01234, 2023 |
| **XLS-R-1B (zero-shot)** | In-the-Wild | 1.23% | - | arXiv:2506.21090, 2025 |
| **XLS-R-1B (fine-tuned)** | Deepfake-Eval-2024 | 8.29% | - | arXiv:2506.21090, 2025 |
| **Whisper+AASIST** | ASVspoof 2021 DF | 8.67% | - | HCII 2024 |
| **Whisper (Cross-Domain)** | ASVspoof 2021 DF | 6.5% | - | EMNLP 2024 |
| **AST (ElevenLabs)** | Unseen generators | 3.3% | - | arXiv:2503.22503, 2025 |
| **DK-CAST** | ASVspoof 2019-LA | **0.38%** | - | Springer, 2025 |
| **ASTDT** | ASVspoof 2019 | **1.20%** | - | Springer, 2025 |
| **Scalable AASIST** | ASVspoof 5 | 7.93% | - | arXiv:2507.11777, 2025 |

## 3. Multimodal Audio-Visual Analysis Frameworks

### 3.1 Named Architectures with Exact Metrics

#### CAD: Cross-Modal Alignment and Distillation (2025)

**Full Architecture Name:** CAD: A General Multimodal Framework for Video Deepfake Detection via Cross-Modal Alignment and Distillation [34]

**Venue:** arXiv preprint (arXiv:2505.15233), May 2025

**arXiv ID:** arXiv:2505.15233

**Description:** CAD comprises two core components: (1) Cross-modal alignment that identifies inconsistencies in high-level semantic synchronization (e.g., lip-speech mismatches); (2) Cross-modal distillation that mitigates feature conflicts during fusion while preserving modality-specific forensic traces (e.g., spectral distortions in synthetic audio). Uses frozen pre-trained encoders—CLIP ViT-Base-16 for visual and Whisper-Small for audio—with cross-attention mechanisms and knowledge distillation. Leverages mutual information theory to maximize both shared and unique information in audio-visual representations.

**Evaluation Metrics:**
- IDForge dataset: **99.96% AUC** and **99.63% AP**
- State-of-the-art performance on FakeAVCeleb, IDForge-v2, FaceShifter, and Celeb-DF
- Outperforms baselines on unseen forgery types in cross-manipulation evaluations

#### SpeechForensics (NeurIPS 2024)

**Full Architecture Name:** SpeechForensics: Audio-Visual Speech Representation Learning for Face Forgery Detection [35]

**Venue:** Conference on Neural Information Processing Systems (NeurIPS 2024)

**Description:** A novel method that capitalizes on the correlation between audio signals and lip movements, learning semantically rich speech features in a self-supervised manner on real videos through masked prediction tasks. Encodes both local (phonetic) and global (linguistic) temporal information to detect discrepancies between visual and audio speech representations in fake videos **without using any fake video data during training**.

**Evaluation Metrics:**
- FakeAVCeleb: AUC of **99.0%**
- KoDF: AUC of **91.7%**
- Outperforms state-of-the-art methods in cross-dataset generalization and robustness
- Generalizes well across languages and is robust even against sophisticated manipulations involving lip synchronization (e.g., Wav2Lip)
- Performance improves with longer video lengths

#### AVFF: Audio-Visual Feature Fusion (CVPR 2024)

**Full Architecture Name:** AVFF: Audio-Visual Feature Fusion for Video Deepfake Detection [36]

**Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)

**Description:** A two-stage cross-modal learning method that explicitly captures the correspondence between audio and visual modalities. The first stage pursues representation learning via self-supervision on real videos using contrastive learning and autoencoding objectives with a novel audio-visual complementary masking and feature fusion strategy. The second stage fine-tunes these representations via supervised learning for deepfake classification.

**Evaluation Metrics:**
- FakeAVCeleb: **98.6% accuracy** and **99.1% AUC**
- Outperformed previous audio-visual state-of-the-art by **14.9% and 9.9%** respectively

#### AMDD: Attribution-Guided Multimodal Deepfake Detection (2025)

**Full Architecture Name:** AMDD: Attribution-Guided Multimodal Deepfake Detection [37]

**Venue:** arXiv preprint (arXiv:2604.26453), 2025

**Description:** Learns generator-specific forensic fingerprints across both audio and visual data using a Cross-Modal Forensic Fingerprint Consistency (CMFFC) loss. This enables both detection of deepfakes and attribution to specific generation methods.

**Evaluation Metrics:**
- FakeAVCeleb: **99.7% balanced accuracy** and **95.9% attribution accuracy**
- Removing attribution loss causes attribution accuracy to collapse from **95.9% to 11.0%**, confirming attribution supervision is essential
- Cross-dataset evaluation confirms fake detection on unseen generators remains an open challenge

#### DigiShield (with DigiFakeAV Benchmark, 2025)

**Full Architecture Name:** DigiShield: Beyond Face Swapping—DigiFakeAV Benchmark for Multimodal Deepfake Detection [38]

**Venue:** arXiv preprint (arXiv:2505.16512), 2025

**Description:** Uses 3D convolutions and cross-modal attention mechanisms. The accompanying DigiFakeAV dataset is the first large-scale multimodal digital human forgery dataset based on diffusion models, comprising 60,000 videos using five generation methods (Sonic, Hallo, Echomimic, V-Express) and voice cloning (CosyVoice 2).

**Evaluation Metrics:**
- DigiFakeAV: AUC of **80.1%**
- Human evaluators misidentify **68%** of synthetic videos as real
- Current detection methods suffer a **43.5% drop in AUC** on DigiFakeAV compared to face-swapping benchmarks
- SOTA deepfake detection models show AUC declines of **over 40%** on this benchmark

#### Multi-Task Audio-Visual Prompt Learning (AAAI 2025)

**Full Architecture Name:** Multi-modal Deepfake Detection via Multi-task Audio-Visual Prompt Learning [39]

**Venue:** AAAI Conference on Artificial Intelligence (AAAI 2025)

**Description:** Exploits frozen foundation models (CLIP for visual, Whisper for audio) with sequential visual prompts and short-time audio prompts. Introduces frame-level cross-modal feature matching (CMFM) loss.

**Evaluation Metrics:**
- State-of-the-art on FakeAVCeleb with only **4.4M trainable parameters**
- Cross-modal feature matching loss brings **17.0% gains in accuracy** and **4.4% in AUC** for cross-dataset generalization
- Demonstrates parameter efficiency through foundation model integration

#### Diffusion-Integrated Multimodal Detection (2025)

**Full Architecture Name:** Enhancing Multimodal Deepfake Detection with Local–Global Feature Integration and Diffusion Models [40]

**Venue:** Signal, Image and Video Processing (Springer Nature), 2025

**DOI:** 10.1007/s11760-025-03970-7

**Description:** Integrates CNNs for local features (lip, eye movements, facial regions) and Vision Transformers for capturing global contextual relationships. Denoising Diffusion Probabilistic Models (DDPMs) are incorporated as preprocessing steps. Multimodal fusion aligns audio and video features temporally using Dynamic Time Warping (DTW).

**Evaluation Metrics:**
- FakeAVCeleb: accuracy of **0.9987**
- AV-Deepfake1M: accuracy of **0.9825**
- TVIL: accuracy of **0.9915**
- LAV-DF: accuracy of **0.9812**
- Surpasses previous methods such as AVSFF

#### X-AVDT (2025)

**Full Architecture Name:** X-AVDT: Audio-Visual Cross-Attention for Robust Deepfake Detection [41]

**Venue:** arXiv preprint (arXiv:2603.08483), 2025

**Description:** Leverages internal audio-visual cross-attention signals from generative diffusion models. By accessing internal features via DDIM inversion, extracts two complementary signals: (i) a video composite capturing inversion-induced discrepancies, and (ii) audio-visual cross-attention features reflecting modality alignment enforced during generation. Also introduces MMDF, a new multi-modal deepfake dataset spanning diverse manipulation types.

**Evaluation Metrics:**
- Average accuracy improved by **+13.1%** across MMDF, FakeAVCeleb, and FaceForensics++
- Generalizes strongly to external benchmarks and unseen generators

#### ERF-BA-TFD+ (2025)

**Full Architecture Name:** ERF-BA-TFD+ Multimodal Deepfake Detection Model [41]

**Description:** Integrates audio and video data using an enhanced receptive field to capture long-range dependencies within the data.

**Evaluation Metrics:**
- Achieved state-of-the-art results on DDL-AV and LAV-DF datasets
- Won a competition focused on deepfake detection

#### LipFD (NeurIPS 2024)

**Full Architecture Name:** LipFD: Lips Are Lying—Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes [42]

**Venue:** Conference on Neural Information Processing Systems (NeurIPS 2024)

**Description:** Exploits temporal inconsistency between lip movements and audio signals for deepfake detection.

**Evaluation Metrics:**
- **Over 95.3% average accuracy** across AVLips, FaceForensics++, and DFDC
- **Up to 90.2% accuracy** in real-world scenarios like WeChat video calls

### 3.2 Multimodal Benchmark Summary

| Method | FakeAVCeleb | DFDC | AV-Deepfake1M | Other Datasets | Venue/Year |
|--------|-------------|------|---------------|----------------|------------|
| **AVFF** | 98.6% acc, 99.1% AUC | — | — | — | CVPR 2024 |
| **SpeechForensics** | 99.0% AUC | — | — | 91.7% AUC (KoDF) | NeurIPS 2024 |
| **LipFD** | — | 95.3% avg acc | — | 90.2% real-world | NeurIPS 2024 |
| **AMDD** | 99.7% bal. acc | — | — | 95.9% attribution | arXiv:2604.26453 |
| **CAD** | SOTA | — | — | 99.96% AUC (IDForge) | arXiv:2505.15233 |
| **Diffusion-Integrated** | 0.9987 acc | — | 0.9825 acc | 0.9915 TVIL | Springer 2025 |
| **DigiShield** | — | — | — | 80.1% AUC (DigiFakeAV) | arXiv:2505.16512 |
| **Multi-Task Prompt** | SOTA (4.4M params) | — | — | +17.0% acc gain | AAAI 2025 |
| **X-AVDT** | +13.1% avg acc | — | — | MMDF SOTA | arXiv:2603.08483 |
| **ERF-BA-TFD+** | — | — | — | SOTA (DDL-AV, LAV-DF) | 2025 |
| **Transformer LLM** | 96.55% acc | — | — | — | AJET 2025 |

## 4. Precise Quantitative Human-vs-Machine Detection Comparisons

### 4.1 Authoritative Meta-Analyses

#### Diel et al. (2024) – Largest Meta-Analysis of Human Deepfake Detection

**Study Title:** "Human Performance in Detecting Deepfakes: A Systematic Review and Meta-Analysis of 56 Papers" [43]

**Journal:** Computers in Human Behavior Reports, Vol. 16, 2024, Article 100538

**DOI:** 10.1016/j.chbr.2024.100538

**Sample Size:** 56 empirical studies, **86,155 participants**, 137 effects pooled across accuracy, odds ratio, and sensitivity index (d')

**Key Exact Figures:**
- **Overall human detection accuracy across all modalities: 55.54%** (95% CI [48.87, 62.10])
- By modality:
  - Audio deepfakes: **62.08%** (95% CI [38.23, 83.18])
  - Image deepfakes: **53.16%** (95% CI [42.12, 64.64])
  - Video deepfakes: **57.31%**
  - Text deepfakes: **52.00%**
- **Odds ratio: OR = 0.64** — participants have only a **39% chance** of successfully identifying deepfakes, which is worse than random guessing
- With interventions (feedback, AI support, caricaturization): detection increases to **up to 65.14%**
- Overall accuracy of 55.54% described as "not significantly better than chance"

#### Somoray, Miller & Holmes (2025) – Systematic Review

**Study Title:** "Human Performance in Deepfake Detection: A Systematic Review" [44]

**Journal:** Human Behavior and Emerging Technologies (Wiley), 2025

**DOI:** 10.1155/hbe2/1833228

**Sample Size:** 40 studies from 30 records

**Key Exact Figures:**
- Human detection accuracy ranged from **57.6% to 75.43%** in forced-choice tasks
- Humans and AI models focus on **different aspects**: humans attend to holistic facial cues (eyes, nose), while AI models analyze subtle pixel-level details
- Audio deepfakes are generally **harder for humans to detect** due to lack of visual cues
- "Truth bias" leads humans to misclassify deepfakes as real
- Providing predictions of AI detection programs **increases** human detection performance

### 4.2 Direct Human-Machine Comparison Studies

#### Pehlivanoglu et al. (2026) — "Is This Real? Susceptibility to Deepfakes in Machines and Humans"

**Full Citation:** Pehlivanoglu et al., "Is This Real? Susceptibility to Deepfakes in Machines and Humans" [45]

**Journal:** Cognitive Research: Principles and Implications (Springer Nature), published January 7, 2026. Open access.

**Study 1: Static Image Deepfakes**
- **Human sample:** Over **2,200 participants**
- **Machine models:** CNN achieved **97% accuracy** with no detection bias; FDA achieved **79% accuracy**
- **Human accuracy: ~49-53%** (near chance, described as "chance level")
- Humans showed a **truth bias** — mistaking deepfakes as real
- None of the cognitive, emotional, or internet-related psychological measures predicted human performance for static images

**Study 2: Dynamic Video Deepfakes**
- **Human sample:** Nearly **1,900 participants**
- **Machine models:** FaceForensics (Xception-based) achieved **49% accuracy**; RNN achieved **39% accuracy**; both showed a "lie bias" (misclassifying real videos as fake)
- **Human accuracy on videos: ~63%** — "correctly identified real and fake videos about two-thirds of the time"
- **Humans outperformed machines on video deepfakes**
- Human detection positively correlated with: higher analytical thinking, lower positive affect, greater internet/smart technology skills

**Overall Conclusion:** "Machines excel at static image deepfake detection but struggle with videos, whereas humans generally struggle with static images but perform better on video deepfakes."

#### Groh et al. (2022) — "Deepfake Detection by Human Crowds, Machines, and Machine-Informed Crowds"

**Full Citation:** Groh, M., Epstein, Z., Firestone, C., & Picard, R.W. [46]

**Journal:** Proceedings of the National Academy of Sciences (PNAS), January 4, 2022, Vol. 119, Issue 1, Article e2110013119

**DOI:** 10.1073/pnas.2110013119

**Sample Size:** 15,016 participants across two large online experiments

**Key Exact Figures:**
- Ordinary human observers and the leading computer vision deepfake detection model were found to be **similarly accurate** — "humans perform in the range of the leading machine learning model"
- However, humans and machines made **different kinds of mistakes**
- A system **integrating human and model predictions** was **more accurate than either alone**
- **Inaccurate model predictions often led humans to incorrectly update their responses**, reducing detection performance
- Manipulations disrupting visual processing of faces **hindered human participants** while **mostly not affecting the model**, suggesting specialized cognitive capacities (face processing) play a role in human detection

#### Korshunov & Marcel (2020) — "Deepfake Detection: Humans vs. Machines"

**Full Citation:** Korshunov, P. & Marcel, S. [47]

**Venue:** Idiap Research Institute, Research Report Idiap-RR-36-2020

**Sample Size:** ~60 naive human subjects (approximately 19 per video), evaluating 120 videos

**Datasets:** Facebook deepfake database (from Kaggle's DFDC 2020); pre-trained on FaceForensics++ and Celeb-DF

**Key Exact Figures:**
- Humans were **confused by good quality deepfakes in 75.5% of cases** — only **24.5% of high-quality deepfakes were correctly perceived as fakes**
- Humans were generally confident in their judgments but were fooled by high-quality deepfakes
- Algorithms (Xception, EfficientNet-B4) achieved **near 100% AUC** on training datasets but **struggled to generalize** to unseen Facebook deepfake videos
- **Algorithms had a totally different perception of deepfakes compared to human subjects** — algorithms struggled to detect deepfakes that looked obviously fake to humans, and succeeded on ones humans found difficult

### 4.3 Deepfake-Eval-2024 Benchmark (Chandra, Murtfeldt et al., 2025)

**Full Citation:** Chandra, S. et al., "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024" [48]

**Venue:** arXiv:2503.02857 [cs.CV]

**Dataset:** 897 deepfake + 897 real videos (45 hours), 834 deepfake + 834 real audio clips (56.5 hours), 864 deepfake + 864 real images (1,975 total), sourced from 88 websites across 52 languages

**Human vs. Machine Comparison:**
- **Human expert forensic analysts estimated at approximately 90% accuracy** — this serves as the estimated accuracy upper bound
- **No commercial models evaluated reached 90% accuracy or above**
- Commercial systems outperformed open-source models by **20-47%**
- Audio detection achieved highest accuracy at **89%** ; video at **78%**
- After fine-tuning on Deepfake-Eval-2024, SOTA models only reached **up to 0.63 AUC (63%)**
- The benchmark authors posit detection models should achieve at least **90% accuracy** based on inter-labeler agreement

### 4.4 Summary Table of Human vs. Machine Detection

| Study | Modality | Human Accuracy | Machine Accuracy | Machine Model | Sample Size |
|-------|----------|---------------|-----------------|---------------|-------------|
| **Diel et al. 2024 (Meta-analysis)** | All modalities | **55.54%** | — | — | 86,155 participants |
| **Diel et al. 2024** | Image deepfakes | **53.16%** | — | — | 18 studies |
| **Diel et al. 2024** | Audio deepfakes | **62.08%** | — | — | 8 studies |
| **Diel et al. 2024** | Video deepfakes | **57.31%** | — | — | Multiple |
| **Pehlivanoglu et al. 2026** | Static images | **~49-53%** | **97%** | CNN | 2,200+ |
| **Pehlivanoglu et al. 2026** | Dynamic videos | **~63%** | **49% (Xception), 39% (RNN)** | FaceForensics, RNN | ~1,900 |
| **Korshunov & Marcel 2020** | High-quality videos | **24.5% correct** | ~100% (in-domain) | Xception, EfficientNet | ~60 |
| **Groh et al. 2022** | Minimal context videos | Similar to ML model | Similar to humans | Leading model | 15,016 |
| **Somoray et al. 2025** | Various | **57.6-75.43%** | — | — | 40 studies |
| **Deepfake-Eval-2024** | Real-world multimodal | **~90% (expert)** | **63% AUC (best after finetuning)** | SOTA + commercial | 88 websites |

## 5. Specific, Verifiable Regulatory Details with Formal Legal Identifiers

### 5.1 European Union: EU AI Act

**Exact Official Legal Identifier:** Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 Laying Down Harmonised Rules on Artificial Intelligence (Artificial Intelligence Act) [49]

**Exact Dates:**
- **Adopted:** June 13, 2024 (European Parliament and Council)
- **Published in Official Journal:** July 12, 2024
- **Entry into Force:** August 1, 2024 (twentieth day after publication)
- **Phased Enforcement per Article 113:**
  - **February 2, 2025:** Prohibitions on unacceptable risk AI practices (Article 5), AI literacy requirements
  - **August 2, 2025:** Rules on General-Purpose AI models, governance, penalties
  - **August 2, 2026:** Transparency obligations (Article 50), most high-risk AI provisions
  - **August 2, 2027:** High-risk AI system obligations (Article 6(1))
  - **August 2, 2027:** GPAI model compliance deadline

**Official URL:** https://eur-lex.europa.eu/eli/reg/2024/1689/oj

**Article 50 Transparency Obligations for Deepfakes:**
- AI-generated synthetic audio, image, video, or text must be marked in machine-readable format as artificially generated
- Deployers of deepfake content must disclose the content has been artificially generated or manipulated
- AI-generated text for public interest matters must be disclosed

**Non-compliance Penalties:** Fines up to €35 million or 7% of worldwide annual revenue, whichever is higher

### 5.2 United States: TAKE IT DOWN Act

**Exact Bill Number:** S.146 — 119th Congress (2025–2026) [50]

**Title:** A Bill to Require Covered Platforms to Remove Nonconsensual Intimate Visual Depictions, and for Other Purposes

**Exact Legislative Status:** **Enacted — Signed by the President on May 19, 2025. Became Public Law No: 119-12**

**Timeline:**
- Introduced: January 16, 2025
- Enacted: May 19, 2025
- Length: 8 pages

**Congress.gov URL:** https://www.congress.gov/bill/119th-congress/senate-bill/146

**Key Provisions:**
- Criminalizes knowing publication of non-consensual intimate imagery (including AI-generated content)
- Penalties up to 2 years (adults) or 3 years (minors) imprisonment
- Platforms must comply with notice-and-takedown procedures within **48 hours**
- FTC enforcement active as of May 19, 2026, with civil penalty authority up to **$53,088 per violation**
- First conviction occurred in April 2026

### 5.3 United States: NO FAKES Act

**House Version:** H.R.2794 — 119th Congress (2025–2026) [51]

**Title:** NO FAKES Act of 2025 — "Nurture Originals, Foster Art, and Keep Entertainment Safe Act of 2025"

**Exact Legislative Status:** **Introduced** on April 9, 2025, by Representative Maria Elvira Salazar (R-FL-27). Referred to House Committee on the Judiciary.

**Congress.gov URL:** https://www.congress.gov/bill/119th-congress/house-bill/2794

**Cosponsors:** 10 bipartisan cosponsors (6 Democrats, 4 Republicans)

**Senate Version:** S.1367 — 119th Congress (2025–2026) [52]

**Congress.gov URL:** https://www.congress.gov/bill/119th-congress/senate-bill/1367

**Key Provisions:** Establishes liability for distributing or hosting unauthorized digital replicas (voice and visual likeness)
**Supported by:** RIAA, Sony Music, Universal Music Group, SAG-AFTRA, YouTube, TikTok, OpenAI, Disney
**Status as of May 2026:** Not yet passed

### 5.4 United States: DEEPFAKES Accountability Act

**Exact Bill Number:** H.R. 5586 — 118th Congress (2023–2024) [53]

**Congress.gov URL:** https://www.congress.gov/bill/118th-congress/house-bill/5586

**Key Provisions:** Mandates digital watermarking of deepfake content, criminalizes failure to identify malicious deepfakes, requires social media platforms to implement content credentialing technology
**Status:** Not enacted

### 5.5 China: Deep Synthesis Provisions

**Official Document Identification:** The "Provisions on the Administration of Deep Synthesis of Internet-based Information Service" (互联网信息服务深度合成管理规定) [54]

**Issuing Authorities:** Jointly issued by the Cyberspace Administration of China (CAC), the Ministry of Industry and Information Technology (MIIT), and the Ministry of Public Security (MPS)

**Document Number:** The **12th Decree** of the Cyberspace Administration of China

**Exact Dates:**
- **Adopted:** November 3, 2022 (by CAC)
- **Promulgated:** November 25, 2022
- **Effective:** January 10, 2023

**Official URL:** https://www.cac.gov.cn/2022-12/11/c_1672221949354811.htm

**Number of Articles:** 25
**Key Provisions:** Mandates real identity authentication, safety assessments for biometric data tools, conspicuous labeling on AI-generated content, and content log retention

### 5.6 China: 2025 Labeling Measures

**Official Document Identification:** "Measures for Labeling of AI-Generated Synthetic Content" (人工智能生成合成内容标识办法) [55]

**Issuing Authorities:** Jointly promulgated by four agencies: CAC, MIIT, MPS, and the State Administration of Radio and Television

**Exact Dates:**
- **Promulgation Date:** March 7, 2025 (per official Chinese text)
- **Effective Date:** September 1, 2025

**Official URL:** https://www.chinalawtranslate.com/en/ai-labeling

**Key Requirements:**
- Requires both **explicit labels** (text, audio cues, icons) and **implicit labels** (metadata embedding, digital watermarks)
- Platforms must detect labels and notify users when content is AI-generated
- **Users publishing AI-generated content must actively label it**
- Malicious alteration or deletion of labels prohibited
- Providers must keep logs for at least **six months**

### 5.7 United Kingdom: Online Safety Act 2023

**Exact Official Legal Identifier:** **Online Safety Act 2023** — **2023 Chapter 50** (ukpga/2023/50) [56]

**Exact Enactment Date:** **Royal Assent on October 26, 2023**

**Commencement:** Criminal offences (cyberflashing, threatening communications) came into effect on **January 31, 2024**

**Official URL:** https://www.legislation.gov.uk/ukpga/2023/50

**Key Provisions:**
- Criminalizes creation and dissemination of sexually explicit deepfakes without consent
- Penalties up to 2 years imprisonment
- Ofcom empowered to enforce compliance with penalties up to **£18 million or 10% of worldwide turnover**
- **March 17, 2025:** Illegal harms duties came into force
- **July 25, 2025:** Children's safety duties came into force

### 5.8 India: IT Rules Amendments (2026)

**Exact Statutory Instrument Citation:** Information Technology (Intermediary Guidelines and Digital Media Ethics Code) Amendment Rules, 2026 [57]

**Issuing Authority:** Ministry of Electronics and Information Technology (MeitY), Government of India

**Exact Dates:**
- **Notification/Adoption:** February 10, 2026
- **Enforcement:** February 20, 2026

**Official Gazette URL:** https://www.meity.gov.in/static/uploads/2025/10/065b6deb585441b5ccdf8be42502a49c.pdf

**Key Provisions:**
- Introduces India's first explicit compliance framework for "Synthetically Generated Information" (SGI)
- **Takedown timelines:** 3 hours for law enforcement directions (down from 36 hours); 2 hours for non-consensual intimate imagery
- Mandatory labeling of SGI with permanent metadata or unique identifiers
- Failure to comply risks loss of safe harbour protection under **Section 79 of the IT Act, 2000**

### 5.9 South Korea: Deepfake-Related Legislation

**Legislation 1:** Amendments to the **Act on Special Cases Concerning the Punishment, Etc. of Sexual Crimes** [58]

**Exact Date of Enactment:** **September 26, 2024** (National Assembly passed 70 bipartisan bills)

**Key Provisions:**
- Penalizes knowing possession, purchase, storage, or viewing of deepfake pornography: up to **3 years imprisonment** or fines up to **30 million won** (~$22,500)
- Maximum sentences raised to **7 years** for production and distribution
- Mandatory minimum sentences: 1 year for blackmail, 3 years for production for dissemination

**Legislation 2:** Amendments to the **Personal Information Protection Act (PIPA)** [59]

**Exact Date of Enactment:** **February 12, 2026**

**Effective Date:** Six months after enactment (approximately **September 11, 2026**)

**Key Provisions:**
- Authorizes administrative fines up to **10% of a company's total revenue** for high-severity data breach cases
- Designates the business owner as the "ultimate responsible person" for data protection
- Expanded reporting obligations covering forgery, alteration, and damage of personal data

**Official PIPC URL:** https://pipc.go.kr/eng/user/ltn/new/noticeDetail.do?bbsId=BBSMSTR_000000000001&nttId=2331

### 5.10 Other International Regulations

**Canada: Bill C-27 - Artificial Intelligence and Data Act (AIDA)**
- **Status:** Prorogation of Parliament on January 5, 2025 terminated all pending bills including Bill C-27
- Future AI regulation legislation considered inevitable

**Japan: AI Promotion Act**
- **Enacted:** May 28, 2025
- Takes a distinctive layered, flexible, non-binding approach prioritizing innovation

**Singapore:**
- No dedicated AI legislation
- Online Criminal Harms Act (OCHA): July 2023
- $20 million initiative (January 2024) through the Centre for Advanced Technologies in Online Safety (CATOS)

**Australia: Online Safety and Other Legislation Amendment (My Face, My Rights) Bill 2025**
- Introduced: November 24, 2025
- Australian AI Safety Institute launched in early 2026

**Global Context:** At least **72 countries** have proposed over **1,000 AI-related policy initiatives** as of early 2026. The G7's International Guiding Principles on AI (October 2023) call for reliable content authentication and provenance mechanisms.

## 6. Cross-Dataset Generalization with Exact Percentage Gaps Tied to Named Methods

### 6.1 Deepfake-Eval-2024 (Chandra, Murtfeldt et al., 2025)

**Full Citation:** Chandra, S., Murtfeldt, K., et al., "Deepfake-Eval-2024: A Multi-Modal In-the-Wild Benchmark of Deepfakes Circulated in 2024" [48]

**arXiv ID:** 2503.02857

**Exact Reported Generalization Gaps (Lab-to-Field):**
- **Video models: AUC decreasing by 50%** compared to previous academic benchmarks
- **Audio models: AUC decreasing by 48%** compared to previous academic benchmarks
- **Image models: AUC decreasing by 45%** compared to previous academic benchmarks
- After fine-tuning on Deepfake-Eval-2024: SOTA models only reached **up to 0.63 AUC (63%)**
- Fine-tuning improvements: average increases of **57.6% for video, 80.6% for audio, 15.6% for images**
- **Diffusion-generated content, non-facial and partial manipulations, non-English audio, background noise, and text overlays** are major challenges
- The addition of background music **significantly reduces accuracy** in audio deepfake detection

### 6.2 MDPI Comprehensive Review (2026)

**Full Citation:** "A Comprehensive Review of Deepfake Detection Techniques" [60]

**Venue:** AI Journal, MDPI, Volume 7(2), 2026

**Exact Reported Generalization Gaps:**
- **Transformer-based architectures: 11.33% average performance decline** when tested cross-dataset
- **CNN-based methods: More than 15% average performance decline** when tested cross-dataset
- **All method classes combined: 10-15% average deterioration** across all methodological classes
- **Computational cost:** Transformers require **3-5× more computation** than CNNs

**Critical Finding:** "Current detection systems are, to a high degree, learning dataset-specific compression artifacts, rather than deepfake characteristics that are generalizable."

### 6.3 Nadimpalli & Rattani (CVPRW 2022)

**Full Citation:** "On Improving Cross-Dataset Generalization of Deepfake Detectors" [61]

**Venue:** IEEE/CVF CVPR Workshops (CVPRW), 2022

**Exact AUC Values and Calculated Drops:**
- Intra-dataset (trained and tested on FF++): EfficientNet V2-L with PPO-based RL agent achieved AUC of **0.994**
- Cross-dataset on Celeb-DF: AUC of **0.669** (one source) or as low as **0.482** (another source)
- Calculated relative drops: **32.7%** (0.994→0.669) to **51.3%** (0.994→0.482)
- Cross-dataset on DeeperForensics-1.0: AUC of **0.952** (~4.2% relative drop)

### 6.4 CrossDF: Deep Information Decomposition (Frontiers in Big Data, 2025)

**Full Citation:** "CrossDF: Improving Cross-Domain Deepfake Detection with Deep Information Decomposition" [16]

**Exact AUC Values:**
- FF++ to Celeb-DF v2: AUC of **0.779** in cross-dataset evaluation
- Diffusion-based Text-to-Image dataset: improved state-of-the-art AUC from **0.669 to 0.802**
- Baseline XceptionNet achieves only **0.669 AUC** on Celeb-DF (from 0.98 intra-dataset)

### 6.5 CAST (arXiv:2506.21711, 2025)

**Exact AUC Values:**
- Intra-dataset on FF++: AUC of **99.49%**
- Cross-dataset on unseen DFD (DeepFakeDetection): AUC of **93.31%**
- Calculated relative drop: **~6.2%**

### 6.6 DeepAction Dataset (Timlrx, 2024)

**Full Citation:** "The Generalisability Gap - Evaluating Deepfake Detectors Across Domains" [62]

**Dataset:** DeepAction — 3,100 AI-generated full-body human action videos

**Exact AUC Values:**
- Pre-trained models (without fine-tuning) on DeepAction: AUC scores dropped to **67-71%**
- Fine-tuned models on DeepAction: **over 95% AUC scores**
- CLIP: AUC improved from **0.71 to 0.94** after retraining classification head

### 6.7 TimeSformer (Springer, 2026)

**Full Citation:** "Cross-dataset video deepfake detection using Transformer and CNN architectures" [63]

**Venue:** Machine Vision and Applications, Springer Nature Link (2026)

**DOI:** 10.1007/s00138-026-01809-w

**Exact Performance:**
- TimeSformer (best model with 96-frame clips and 30% fine-tuning): **78.4% accuracy, 0.801 AUC, 77.0% F1-score**
- All models benefit from moderate fine-tuning, with gains plateauing beyond 20%
- Even best-performing models show reduced performance when transferred to entirely new domains

### 6.8 Quality-Centric Framework (arXiv:2411.05335, 2024)

**Exact Improvement:**
- Achieves up to **10% AUC improvement** over strong baselines on multiple cross-dataset evaluations
- Evaluated on FF++, Celeb-DFv2, DFDC, WildDeepfake, and DF40

### 6.9 Seppälä (2025) — University of Vaasa Review

**Full Citation:** "Review of deepfake detection methods" [64]

**Exact Findings:**
- Most detection methods score **above 95% accuracy on standard benchmarks**
- Detection accuracy suffers greatly with compression: **over 20 percentage points lost** between uncompressed and heavily compressed video
- Approaches using synthetic training data showed better cross-dataset generalization

### 6.10 Xception-Based Enhancement (BRAC University)

**Exact Improvement:**
- Improved cross-dataset detection AUC by roughly **13-15%** (FF++ to Celeb-DF or DFDC) over baseline Xception

### 6.11 Summary Table of Generalization Gaps

| Method/Study | Training Data | Testing Data | Intra-Dataset | Cross-Dataset | Drop | Venue |
|-------------|--------------|--------------|---------------|---------------|------|-------|
| **Deepfake-Eval-2024** | Academic benchmarks | Real-world 2024 | ~95% AUC | **~45-50% drop** | 45-50% | arXiv:2503.02857 |
| **MDPI Review - Transformers** | Various | Cross-dataset | — | **11.33% avg decline** | 11.33% | MDPI AI 2026 |
| **MDPI Review - CNNs** | Various | Cross-dataset | — | **>15% avg decline** | >15% | MDPI AI 2026 |
| **MDPI Review - All methods** | Various | Cross-dataset | — | **10-15% avg decline** | 10-15% | MDPI AI 2026 |
| **Nadimpalli & Rattani** | FF++ | Celeb-DF | 0.994 AUC | **0.669-0.482 AUC** | 32.7-51.3% | CVPRW 2022 |
| **CrossDF (DID)** | FF++ | CDF2 | ~0.98 | **0.779 AUC** | ~20.5% | Front. Big Data 2025 |
| **CAST** | FF++ | DFD | 99.49% AUC | **93.31% AUC** | ~6.2% | arXiv:2506.21711 |
| **DeepAction (pre-trained)** | Facial data | Non-facial videos | >95% | **67-71% AUC** | >25% | Timlrx 2024 |
| **Seppälä 2025** | Various | Various | >95% | **>20 ppt loss** | >20 ppt | Vaasa thesis |

## 7. Privacy-Preserving Techniques with Named Frameworks and Exact Performance Trade-offs

### 7.1 Federated Learning Frameworks

#### FedForgery (IEEE TIFS, 2023)

**Full Architecture Name:** FedForgery: Generalized Face Forgery Detection with Residual Federated Learning [65]

**Venue:** IEEE Transactions on Information Forensics and Security, 2023

**arXiv ID:** arXiv:2210.09563

**GitHub:** https://github.com/GANG370/FedForgery

**Description:** Described as "the first exploration to introduce federated learning and explore generalization ability in the face forgery detection field." Combines a variational autoencoder to learn robust discriminative residual feature maps with a federated learning strategy enabling collaborative training across multiple decentralized data centers without sharing raw data.

**Exact Performance Metrics:**
- Deeperforensics-1.0 standard tests: **99.75% accuracy**
- Deeperforensics-1.0 perturbed data: **95.21% accuracy**
- Outperforms competitors by **up to 6.83%** on distorted data
- Achieved **3-4% higher accuracy than vanilla FL** on challenging datasets
- Shows strong generalization capability on unknown artifact types

#### FedPR (2024)

**Full Architecture Name:** Federated Face Forgery Detection Learning with Personalized Representation [66]

**Venue:** arXiv preprint (arXiv:2406.11145), June 2024

**arXiv ID:** arXiv:2406.11145

**GitHub:** https://github.com/GANG370/PFR-Forgery

**Description:** Addresses limitations of traditional centralized and federated learning by disentangling shared and personalized features, allowing individual client models to learn customized representations based on their own data while aggregating shared features centrally.

**Exact Performance Metrics:**
- Accuracy improvements of approximately **3% to 6%** over non-personalized federated learning
- Reaches **up to 89.90% accuracy on CelebDF-v2**
- Outperforms state-of-the-art methods like CADDM, RFM, GFF in most scenarios
- Outperforms across FaceForensics++, WildDeepfake, CelebDF-v2, Deeperforensics-1.0

### 7.2 Differential Privacy

#### DP-DeepDetect Framework (Systematic Analysis, 2025)

**Full Citation:** Systematic analysis of privacy-utility trade-offs for deepfake detection with differential privacy [67]

**Exact Privacy-Utility Trade-offs:**
- At **ε=4** (strong privacy guarantee): **AUC dropped by 8-12%** across benchmarks
- At **ε=10** (moderate privacy guarantee): **AUC dropped by 3-5%**
- Moderate privacy guarantees (ε=8-10) are achievable without severely compromising detection performance

**Attack-Aware Noise Calibration (NeurIPS 2024):**
- Demonstrates that standard practice of calibrating noise to satisfy a given privacy budget ε "leads to overly conservative risk assessments and unnecessarily low utility"
- Proposed method directly calibrates noise to a desired attack risk level, substantially improving model accuracy for the same risk level

### 7.3 On-Device Processing

#### DeFakeQ (2026)

**Full Architecture Name:** DeFakeQ: Enabling Real-Time Deepfake Detection on Edge Devices via Adaptive Bidirectional Quantization [68]

**Venue:** arXiv preprint (arXiv:2604.08847), April 2026

**arXiv ID:** arXiv:2604.08847

**Description:** "The first quantization framework specifically designed to enable real-time deepfake detection on resource-constrained edge devices such as mobile phones." Employs Horizontal Adaptive Quantization (HAQ) and Vertical Efficient Feature Fine-Tuning (VEFT) with progressive contrastive learning.

**Exact Performance Metrics:**
- Reduces model size to **10-20% of original** while retaining **up to 90% of baseline detection accuracy**
- Tested across **five benchmark deepfake datasets** and **eleven state-of-the-art backbone detectors**
- Outperforms existing quantization baselines such as BRECQ, Adalog, and FIMA-Q
- Successfully deployed on mobile devices, confirming practical real-time edge deployment capability

#### Mobile-FSBI (2024)

**Full Architecture Name:** Mobile-FSBI: Lightweight Deepfake Image Detection Framework [69]

**Description:** Uses MobileNetV3-Small backbone (~2.5 million parameters), combining Self-Blended Image (SBI) generation and frequency-domain enhancement via one-level Discrete Wavelet Transform (DWT).

**Exact Performance Metrics:**
- In-domain (DF40-derived): Accuracy **~88%**, F1-score **~0.88**, ROC-AUC **~0.954**
- Cross-dataset (Celeb-DF subset): ROC-AUC **~0.67** (highlighting domain shift challenges)

#### Attention-Enhanced MobileNet + FFT (2025)

**Full Architecture Name:** Lightweight Deepfake Detection on Mobile Devices Using Attention-Enhanced MobileNet and Frequency Domain Analysis [70]

**Venue:** Journal of Technology Informatics and Engineering, April 2025

**Exact Performance Metrics:**
- **Accuracy: 94.2%**
- **F1-score: 93.8%**
- **Computational efficiency improvement of 27.5%** over conventional CNN-based approaches

#### Lightweight MobileNetV2 with Grad-CAM (2024)

**Full Architecture Name:** Lightweight Deepfake Detection using MobileNetV2 with Grad-CAM [71]

**Exact Performance Metrics:**
- **Classification accuracy: 91.4%**
- **F1-score: 0.903**
- **AUC-ROC: 0.964**
- **Only 3.4 million parameters**
- **Inference time of 72 milliseconds per image on CPU**

#### Lightweight Vision Transformers (ASEE Peer, 2024)

**Full Citation:** "Are Lightweight Vision Transformers Enough for Deepfake Face Detection?" [72]

**Venue:** ASEE Peer, 2024

**Exact Performance Metrics:**
- **MobileViTv2 achieves 98.15% accuracy** on binary classification
- Matches ViT-Base while **reducing parameters by 19× and latency by 38%**
- Lightweight models showed limitations in cross-dataset generalization and multi-class classification

### 7.4 Secure Multi-Party Computation and Homomorphic Encryption

#### SecDFDNet (Digital Signal Processing, 2023)

**Full Architecture Name:** Privacy-Preserving DeepFake Face Image Detection (SecDFDNet) [73]

**Venue:** ScienceDirect / Digital Signal Processing, 2023

**DOI:** 10.1016/j.dsp.2023.104266

**Description:** "The first to address DeepFake face detection under privacy-preserving requirements." Uses additive secret sharing (ASS) and four novel secure interaction protocols—SecReLU, SecSigm, SecSpatial, and SecChannel—theoretically proven secure under the semi-honest adversary model with the universal composable (UC) security framework.

**Exact Performance Metrics:**
- **Achieves identical detection accuracy as the plaintext DFDNet**
- Low communication and space complexity
- Supports extension to multiple servers and other architectures (Xception, EfficientNet-B0)

#### FHE-Based Privacy-Preserving Detection (NTU Singapore)

**Full Architecture Name:** Privacy-Preserving Deepfake Detection with Fully Homomorphic Encryption (NTU Singapore) [74]

**Description:** Explores FHE implemented via Zama's Concrete ML, enabling inference on encrypted data without decryption. Incorporates DCT preprocessing, Quantization Aware Training (QAT), and Residual Network architectures.

**Exact Performance Metrics:**
- **Statistically equivalent accuracy** under simulated and encrypted execution modes compared to standard unencrypted methods
- Significant reductions in computational overhead through frequency-domain optimization

### 7.5 Combined Privacy Frameworks

#### SafeEar (ACM CCS 2024) — Privacy-Preserving Audio Deepfake Detection

**Full Architecture Name:** SafeEar: Content Privacy-Preserving Audio Deepfake Detection [75]

**Venue:** ACM Conference on Computer and Communications Security (ACM CCS 2024)

**arXiv ID:** arXiv:2409.09272

**Description:** The first framework for audio deepfake detection that preserves the privacy of speech content. Uses a neural audio codec decoupling model that separates semantic and acoustic information, using only acoustic information (prosody and timbre) for detection. Implements adversarial training with discriminators and real-world codec augmentations.

**Exact Performance Metrics:**
- **ASVspoof 2019, 2021, and CVoiceFake (multilingual): EER down to 2.02%**
- **Word Error Rates (WERs) above 93.93%** across five languages (English, Chinese, German, French, Italian)
- Effectively blocks content recovery by machine and human analysis

#### LaED (Scientific Reports, Nature, 2026)

**Full Architecture Name:** LaED: A Novel Lightweight, Edge-Aware and Explainable Deep Learning Model for Privacy-Preserving Facial Attendance Tracking [76]

**Venue:** Scientific Reports (Nature), 2026

**Description:** Integrates federated learning **with differential privacy** to ensure biometric data remains locally stored while enabling accountable model updates. Incorporates multimodal spoof detection combining physiological (remote photoplethysmography) and temporal facial cues.

**Exact Performance Metrics:**
- **Over 97.8% recognition accuracy**
- Attack Presentation Classification Error Rate (APCER) and Bona Fide Presentation Classification Error Rate (BPCER) **below 2%**
- Demographic fairness gaps **under 2%**
- Inference latency **below 150 milliseconds** on edge hardware (NVIDIA Jetson Nano and Raspberry Pi 4)

#### FMM-MMF (Discover Computing, Springer, 2026)

**Full Architecture Name:** A Data Analytics Based Federated Biometrics for Deepfake Fraud Detection in Financial Multimedia Systems [77]

**Venue:** Discover Computing (Springer Nature Link), 2026

**DOI:** 10.1007/s10791-026-10073-5

**Description:** Operates within a federated learning architecture, integrating facial micro-expression analysis through a lightweight µ-BERT encoder, audio cues, and behavioral and device metadata fused via cross-modal attention mechanisms.

**Exact Performance Metrics:**
- **96.74% accuracy**
- **F1-score of 0.987**
- Improved minority class detection precision by **11.6%** over state-of-the-art models
- Robust under challenging non-IID client distributions and adversarial perturbations

### 7.6 Privacy-Utility Trade-off Summary

| Framework | Privacy Mechanism | Utility Metric | Privacy Level | Trade-off | Venue/Year |
|-----------|------------------|----------------|---------------|-----------|------------|
| **DP-DeepDetect** | DP-SGD | AUC | ε=4 | 8-12% drop | Systematic Analysis 2025 |
| **DP-DeepDetect** | DP-SGD | AUC | ε=10 | 3-5% drop | Systematic Analysis 2025 |
| **DeFakeQ** | On-device quantization | Detection accuracy | Model 10-20% original size | **90% retained** | arXiv:2604.08847, 2026 |
| **FedForgery** | Federated learning | Accuracy | No raw data shared | **99.75% (standard), 95.21% (perturbed)** | IEEE TIFS 2023 |
| **FedPR** | Personalized FL | Accuracy | No raw data shared | **89.90% CelebDF-v2** | arXiv:2406.11145 |
| **SafeEar** | Semantic/acoustic decoupling | EER | **93.93% WER** blocking content | **2.02% EER** | ACM CCS 2024 |
| **SecDFDNet** | Additive secret sharing | Accuracy | Encrypted computation | **Identical to plaintext** | Digital Signal Process. 2023 |
| **LaED** | FL + DP | Accuracy | Local storage + DP | **97.8% accuracy, <150ms latency** | Nature Sci. Rep. 2026 |
| **FMM-MMF** | Federated learning | Accuracy | No raw data shared | **96.74% accuracy** | Springer 2026 |

## 8. Ethical Concerns

### 8.1 Demographic Bias and Intersectional Bias

**Key Study:** "Examination of Fairness of AI Models for Deepfake Detection" (IJCAI 2021) [78]

**Exact Figures:**
- **Up to 10.7% difference in error rate** depending on racial group
- Facial profiles of female Asian or female African are **1.5 to 3 times more likely** to be mistakenly labeled as fake than profiles of male Caucasian
- FaceForensics++ contains over **58% (mostly white) women** compared with 41.7% men, with less than **5% featuring Black or Indian individuals**

**University at Buffalo Bias Reduction Study (WACV 2024, DARPA-funded):**
- Created demographic-aware and demographic-agnostic methods that increased overall detection accuracy from **91.49% to as high as 94.17%** while reducing false positive disparities

**Fairness-Aware Deepfake Detection (arXiv:2511.10150):**
- Presents a dual-mechanism collaborative optimization framework integrating Structural Fairness Decoupling (SFD) and Global Distribution Alignment (GDA)
- Demonstrates superior intra-group and inter-group fairness metrics (Equal False Positive Rate disparity, Demographic Parity, es-AUC)

**Age-Diverse Deepfake Dataset (arXiv:2508.06552v1):**
- Common datasets (Celeb-DF, FaceForensics++) have skewed age distributions (heavily favoring 19-35 age group)
- Models trained on age-diverse dataset achieved AUC scores greater than **0.997 and lower EERs across all age groups**

### 8.2 Non-Consensual Deepfakes: Gender-Based Targeting

**Exact Figures:**
- **96% of deepfakes are nonconsensual sexual deepfakes**
- **99% of sexual deepfakes target women** (NY State OPDV)
- Analysis of 14,678 deepfake videos: "100% of content on major deepfake pornography sites depict women"
- Globally, **57% of women report experiencing image-based abuse**
- **1 in 3 people identify as victims of image-based abuse**

### 8.3 The Liar's Dividend

**Definition:** Coined by law professors Robert Chesney and Danielle Citron, this term describes "the benefit dishonest actors gain from the mere existence of synthetic content."

**Key Examples:**
- President Trump's dismissal of authentic videos as AI fabrications
- Venezuelan officials' denial of US military strike footage
- Elon Musk's legal defense branding his own statements as potential deepfakes

**Scale:** Deepfake incidents tracked globally surged from approximately **500,000 cases in 2023 to over 8 million in 2025**, a 900% increase in two years.

### 8.4 Complementary Strategies: Cryptographic Provenance (C2PA)

**Coalition for Content Provenance and Authenticity (C2PA):**
- Created in February 2021 by Adobe, Arm, BBC, Intel, Microsoft, and Truepic
- Open technical standard for embedding cryptographically signed provenance data inside digital media files
- Content Credentials function like "a nutrition label for digital content"

**Key Developments:**
- **Leica M11-P:** "The world's first camera with Content Credentials built in"
- **Qualcomm Snapdragon 8 Gen3** supports chip-level Content Credentials
- C2PA specification evolved from v1.0 (2022) to v2.2 (May 2025)
- **91% of creators** said they want a reliable way to attach attribution to their work

## 9. Conclusion and Future Directions

Deepfake detection research from 2022 to 2026 has achieved remarkable technical progress across all subdomains:

**Video Detection:** AUC on standard benchmarks has risen from approximately 0.92 (2022) to over 0.99 (2026) using foundation model-integrated architectures like CAST, MASDT, and GenConViT. Transformer-based approaches (TALL-Swin, UNITE, Hybrid ViT-Linformer) have demonstrated superior cross-dataset generalization (11.33% average decline) compared to CNNs (>15% decline), though at 3-5× computational cost.

**Audio Detection:** EER on ASVspoof has fallen from 2.1% to below 0.5% for the best systems, with DK-CAST achieving 0.38% EER on ASVspoof 2019-LA and AASIST3 showing more than twofold improvement over prior iterations. Self-supervised models (XLS-R-1B) have demonstrated remarkable zero-shot generalization (1.23% EER on In-the-Wild).

**Multimodal Detection:** Models integrating audio and visual streams (CAD, SpeechForensics, AVFF) consistently achieve over 99% AUC on standard benchmarks, with cross-modal attention and contrastive learning providing particularly strong results.

However, three critical challenges remain:

**First, the generalization gap** between benchmark and real-world performance persists as the field's greatest challenge. Deepfake-Eval-2024 demonstrated **45-50% AUC drops** for academic models tested on contemporary forgeries. Even after fine-tuning, SOTA models only reached 63% AUC. The rapid evolution of generation technology, domain shift, and dataset-specific artifacts all contribute to this gap.

**Second, demographic bias** remains a fundamental ethical concern. Disparities of up to **10.7% in error rates** across demographic groups, with marginalized groups **1.5-3 times more likely** to be falsely flagged, undermine trust in detection technology.

**Third, the adversarial co-evolution** of generation and detection creates an ongoing arms race. Diffusion-based forgeries evade detectors trained on GAN-based artifacts, and current detection methods suffer a **43.5% drop in AUC** on diffusion-based digital human forgeries.

**Privacy-preserving techniques** continue to advance significantly. Federated learning approaches (FedForgery achieving 99.75% accuracy, FedPR reaching 89.90% on CelebDF-v2) enable collaborative training without raw data sharing. Differential privacy (at ε=10, only 3-5% AUC drop) provides mathematically rigorous privacy guarantees. On-device processing (DeFakeQ retaining 90% accuracy with 10-20% of original model size) enables real-time detection on edge devices. Cryptographically secure methods (SecDFDNet achieving identical accuracy to plaintext through additive secret sharing) demonstrate that privacy need not come at the cost of detection performance.

**The regulatory landscape** is evolving rapidly: the EU AI Act (Regulation 2024/1689) establishes the most comprehensive framework with deepfake transparency obligations enforceable from August 2026; the US has enacted the TAKE IT DOWN Act (Public Law 119-12, May 2025) while the NO FAKES Act (H.R.2794/S.1367) remains pending; China enforces strict labeling requirements (new Measures effective September 1, 2025); and over 72 countries have proposed AI-related policy initiatives. The UK Online Safety Act (2023 Chapter 50) and India's IT Rules Amendments (February 20, 2026) add to the growing global regulatory framework.

**Future research priorities** identified in the literature include: (1) generator-agnostic detectors capable of handling unseen manipulation types; (2) fairness-aware training methods that eliminate demographic disparities; (3) privacy-preserving architectures that minimize accuracy trade-offs; (4) adversarial robustness against both deliberate attacks and natural distribution shifts; and (5) interpretable detection systems that provide human-understandable explanations for their decisions.

---

### Sources

[1] TALL: Thumbnail Layout for Deepfake Video Detection (ICCV 2023): https://arxiv.org/abs/2307.07494

[2] CAST: Cross-Attentive Spatio-Temporal Feature Fusion (Knowledge-Based Systems, 2026): https://arxiv.org/abs/2506.21711

[3] UNITE: Universal Network for Identifying Tampered and Engineered Videos (CVPR 2025): https://arxiv.org/abs/2412.12278

[4] MASDT: Masked Autoencoding Spatiotemporal Deepfake Transformer (IEEE IJCB 2023): https://arxiv.org/abs/2306.06881

[5] GenConViT: Generative Convolutional Vision Transformer: https://github.com/erprogs/GenConViT

[6] EffiSwinT Ensemble (University at Buffalo, 2024): https://www.buffalo.edu/ubnow/stories/2024/05/lyu-deepfake-detection.html

[7] Lightweight Hybrid Transformer (Frontiers in Big Data, 2025): https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1521653/full

[8] Deepfake Detection Using Spatiotemporal Methods and Vision-Language Models (KDD 2025): https://kdd2025.kdd.org/wp-content/uploads/2025/07/CameraReady-33.pdf

[9] DiffusionFake (NeurIPS 2024): https://papers.nips.cc/paper_files/paper/2024/file/b7d9b1d4a9464d5d1ece82198e351349-Paper-Conference.pdf

[10] GenD: Deepfake Detection that Generalizes Across Benchmarks (arXiv:2508.06248): https://arxiv.org/html/2508.06248v2

[11] Generalizing Deepfake Video Detection with Plug-and-Play Video-Level Blending and Spatiotemporal Adapter Tuning (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/papers/Yan_Generalizing_Deepfake_Video_Detection_with_Plug-and-Play_Video-Level_Blending_and_Spatiotemporal_CVPR_2025_paper.pdf

[12] LSDA: Transcending Forgery Specificity with Latent Space Augmentation (CVPR 2024): https://arxiv.org/abs/2311.11278

[13] AltFreezing for More General Video Face Forgery Detection (CVPR 2023): https://github.com/ZhendongWang6/AltFreezing

[14] FakeVLM: Spot the Fake (NeurIPS 2025): https://arxiv.org/abs/2503.14905

[15] Quality-Centric Framework with Forgery Quality Score (arXiv:2411.05335, 2024): https://arxiv.org/abs/2411.05335

[16] CrossDF: Improving Cross-Domain Deepfake Detection (Frontiers in Big Data, 2025): https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2025.1669488/full

[17] DF40: Toward Next-Generation Deepfake Detection (NeurIPS 2024): https://arxiv.org/abs/2406.13495

[18] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks (Interspeech 2022): https://arxiv.org/abs/2110.01200

[19] AASIST2 (arXiv:2309.08279, 2024): https://arxiv.org/abs/2309.08279

[20] AASIST3 (Interspeech 2024): https://arxiv.org/abs/2408.17352

[21] RawNet3 (2023): https://arxiv.org/abs/2305.01234

[22] Wav2Vec2.0 + AASIST (IEEE/ACM TASLP, 2023): https://arxiv.org/abs/2210.02437

[23] Audio Anti-Spoofing Based on Audio Feature Fusion (MDPI, 2023): https://www.mdpi.com/journal/applsci

[24] SZU-AFS (ASVspoof 5): https://arxiv.org/abs/2408.09933

[25] HuRawNet2_modified (MDPI Applied Sciences, 2023): https://www.mdpi.com/2076-3417/13/14/8488

[26] AntiDeepfake: Post-training for Speech Detection (arXiv:2506.21090, 2025): https://arxiv.org/abs/2506.21090

[27] Whisper+AASIST (HCII 2024): https://link.springer.com/chapter/10.1007/978-3-031-60875-9

[28] Cross-Domain Audio Deepfake Detection (EMNLP 2024): https://arxiv.org/abs/2404.04904

[29] AST for Cross-Technology Generalization (arXiv:2503.22503, 2025): https://arxiv.org/abs/2503.22503

[30] Continuous Learning of Transformer-based Audio Deepfake Detection (arXiv:2409.05924, 2024): https://arxiv.org/abs/2409.05924

[31] Towards Scalable AASIST (arXiv:2507.11777, 2025): https://arxiv.org/abs/2507.11777

[32] DK-CAST (Springer, 2025): https://link.springer.com/article/10.1007/s10791-025-09746-4

[33] ASTDT (Springer, 2025): https://link.springer.com/article/10.1186/s13635-025-00217-3

[34] CAD: Cross-Modal Alignment and Distillation (arXiv:2505.15233, 2025): https://arxiv.org/abs/2505.15233

[35] SpeechForensics (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024

[36] AVFF: Audio-Visual Feature Fusion (CVPR 2024): https://openaccess.thecvf.com/content/CVPR2024

[37] AMDD: Attribution-Guided Multimodal Deepfake Detection (arXiv:2604.26453, 2025): https://arxiv.org/abs/2604.26453

[38] DigiShield / DigiFakeAV (arXiv:2505.16512, 2025): https://arxiv.org/abs/2505.16512

[39] Multi-modal Deepfake Detection via Multi-task Audio-Visual Prompt Learning (AAAI 2025): https://ojs.aaai.org/index.php/AAAI/article/view/32042

[40] Enhancing Multimodal Deepfake Detection with Diffusion Models (Springer, 2025): https://link.springer.com/article/10.1007/s11760-025-03970-7

[41] X-AVDT (arXiv:2603.08483, 2025): https://arxiv.org/abs/2603.08483

[42] LipFD: Lips Are Lying (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/a5a5b0ff87c59172a13342d428b1e033-Paper-Conference.pdf

[43] Diel et al. (2024) Meta-Analysis: https://www.sciencedirect.com/science/article/pii/S2451958824001714

[44] Somoray et al. (2025) Systematic Review: https://onlinelibrary.wiley.com/doi/full/10.1155/hbe2/1833228

[45] Pehlivanoglu et al. (2026) Human vs Machine: https://link.springer.com/article/10.1186/s41235-025-00700-y

[46] Groh et al. (2022) PNAS: https://www.pnas.org/doi/10.1073/pnas.2110013119

[47] Korshunov & Marcel (2020): https://publications.idiap.ch/downloads/reports/2020/Korshunov_Idiap-RR-36-2020.pdf

[48] Deepfake-Eval-2024 (arXiv:2503.02857, 2025): https://arxiv.org/abs/2503.02857

[49] EU AI Act (Regulation 2024/1689): https://eur-lex.europa.eu/eli/reg/2024/1689/oj

[50] TAKE IT DOWN Act (S.146): https://www.congress.gov/bill/119th-congress/senate-bill/146

[51] NO FAKES Act House (H.R.2794): https://www.congress.gov/bill/119th-congress/house-bill/2794

[52] NO FAKES Act Senate (S.1367): https://www.congress.gov/bill/119th-congress/senate-bill/1367

[53] DEEPFAKES Accountability Act (H.R. 5586): https://www.congress.gov/bill/118th-congress/house-bill/5586

[54] China Deep Synthesis Provisions: https://www.cac.gov.cn/2022-12/11/c_1672221949354811.htm

[55] China 2025 Labeling Measures: https://www.chinalawtranslate.com/en/ai-labeling

[56] UK Online Safety Act 2023: https://www.legislation.gov.uk/ukpga/2023/50

[57] India IT Rules 2026 Amendments: https://www.meity.gov.in/static/uploads/2025/10/065b6deb585441b5ccdf8be42502a49c.pdf

[58] South Korea Sexual Crimes Act (September 2024): https://www.koreaherald.com/article/3480484

[59] South Korea PIPA Amendments (February 2026): https://pipc.go.kr/eng/user/ltn/new/noticeDetail.do?bbsId=BBSMSTR_000000000001&nttId=2331

[60] MDPI Comprehensive Review (2026): https://www.mdpi.com/2673-2688/7/2/68

[61] Nadimpalli & Rattani (CVPRW 2022): https://openaccess.thecvf.com/content/CVPR2022W/WMF/papers/Nadimpalli_On_Improving_Cross-Dataset_Generalization_of_Deepfake_Detectors_CVPRW_2022_paper.pdf

[62] DeepAction Generalisability Gap: https://www.timlrx.com/blog/how-generalisable-are-deepfake-detectors

[63] TimeSformer Cross-Dataset (Springer, 2026): https://link.springer.com/article/10.1007/s00138-026-01809-w

[64] Seppälä (2025) Review: https://osuva.uwasa.fi/bitstreams/cec124dc-7eae-4acf-8a91-9b96f2719342/download

[65] FedForgery (IEEE TIFS, 2023): https://arxiv.org/abs/2210.09563

[66] FedPR (arXiv:2406.11145, 2024): https://arxiv.org/abs/2406.11145

[67] DP-DeepDetect Systematic Analysis (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12855931

[68] DeFakeQ (arXiv:2604.08847, 2026): https://arxiv.org/abs/2604.08847

[69] Mobile-FSBI: https://github.com/AfsanaSharmin/Lightweight-FSBI-Deepfake-Image-Detection-

[70] Attention-Enhanced MobileNet (JTIE, 2025): https://jtie.stekom.ac.id/index.php/jtie/article/view/275

[71] Lightweight MobileNetV2 with Grad-CAM (2024): https://ijrpr.com/uploads/V7ISSUE5/IJRPR64440.pdf

[72] Lightweight ViTs for Deepfake (ASEE Peer, 2024): https://peer.asee.org/are-lightweight-vision-transformers-enough-for-deepfake-face-detection-performance-scalability-and-explainability-study.pdf

[73] SecDFDNet (Digital Signal Processing, 2023): https://www.sciencedirect.com/science/article/abs/pii/S1051200423003287

[74] FHE-Based Privacy-Preserving Detection (NTU Singapore): https://dr.ntu.edu.sg/entities/publication/7c9cbb1a-ffaa-487c-aee6-c3630f979169

[75] SafeEar (ACM CCS 2024): https://arxiv.org/abs/2409.09272

[76] LaED (Scientific Reports, Nature, 2026): https://www.nature.com/articles/s41598-026-42051-8

[77] FMM-MMF (Springer, 2026): https://link.springer.com/article/10.1007/s10791-026-10073-5

[78] Fairness of AI Models for Deepfake Detection (IJCAI 2021): https://arxiv.org/abs/2105.00558