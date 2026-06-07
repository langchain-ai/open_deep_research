# Comprehensive Survey of Deepfake Detection Research (2022–2026): Technical Methods, Benchmarks, Ethics, and Regulation

## 1. Video Detection Methods

### 1.1 Transformer-Based Architectures

#### Video Vision Transformer (ViViT) Variants

The original **ViViT** architecture (Arnab et al., ICCV 2021) extended the Vision Transformer concept to video understanding by extracting spatiotemporal tokens from input videos [1]. For deepfake detection, an **Improved ViViT** system was proposed in IEEE Access (January 2024) that leverages facial landmark-based tubelets combined with Depthwise Separable Convolution (DSC) and Convolution Block Attention Module (CBAM). On the Celeb-DF version 2 dataset, this approach achieved **87.18% accuracy and 92.52% F1-score**, outperforming baseline models like Mesonet and Xception. The ablation study confirmed that both landmark extraction and DSC+CBAM modules positively contributed to performance [2].

#### TimeSformer Adaptations

A comprehensive evaluation published in Springer (January 2026, DOI: 10.1007/s00138-026-01809-w) benchmarked **TimeSformer** alongside ResNet50, ResNet101, ViT, CNN-LSTM, and CNN-Attention across FaceForensics++ (FF++), Celeb-DF, DeepFake Detection (DFD), and the novel ReenactFaces dataset. TimeSformer consistently outperformed other architectures, achieving **78.4% accuracy, an AUC of 0.801, and 77.0% F1-score** with 96-frame clips and 30% fine-tuning. The study confirmed that even the best-performing models show reduced performance when transferred to entirely new domains, indicating current detection systems lack true domain invariance [3].

The **CLIP + TimeSformer** approach (KDD 2025) combined TimeSformer's divided space-time attention with CLIP's vision-language representations. When evaluated on FaceForensics++, Celeb-DF v1/v2, DeepfakeDetection, FaceShifter, and DFDC, this method achieved excellent performance on FaceForensics++ (99.54% AUC) and FaceShifter (90.35% AUC), with the X-CLIP multiframe integration transformer achieving the highest average AUC of **90.61% across all datasets** [4].

#### Swin-Transformer Variants

**GenConViT** (Generative Convolutional Vision Transformer) combines ConvNeXt and Swin Transformer architectures with Autoencoder and Variational Autoencoder modules to learn latent data distribution. Trained and evaluated on DFDC, FF++, DeepfakeTIMIT, and Celeb-DF v2, GenConViT achieved an **average accuracy of 95.8% and an AUC of 99.3%** [5].

**Swin-Fake** (Electronics, 2024, Vol. 13, No. 15, Article 3045) leverages a Swin Transformer backbone with consistency learning, applying five data augmentation techniques to multiple frames. Experiments on FaceForensics++, DFDC, Celeb-DF, and FaceShifter demonstrated strong in-dataset accuracy and improved generalization on Celeb-DF, with visualization confirming the model focuses on manipulated facial features (eyes, lips) rather than boundaries [6].

**TALL-Swin** (Xu et al., ICCV 2023, arXiv:2307.07494) rearranges multiple consecutive video frames into a compact thumbnail layout (2×2 arrangement) to preserve temporal information without significant computational costs. Integrating TALL with the Swin Transformer backbone yielded **90.79% AUC on cross-dataset evaluation from FaceForensics++ to Celeb-DF**, demonstrating better generalization to unseen datasets and robustness against video corruptions. The follow-up **TALL++** (IJCV 2024) integrates a Graph Reasoning Block (GRB) and Semantic Consistency (SC) loss, further improving semantic interactions between facial regions and enforcing temporal consistency [7].

**EffiSwinT Ensemble** (University at Buffalo, May 2024) combines convolutional EfficientNet B3 with Swin Transformer attention mechanisms. Two models trained on distinct datasets (FaceForensics++ and FaceForensics in the Wild) combined predictions via weighted averaging, achieving the **highest AUC in cross-dataset validation on Celeb-DF (v2)** [8].

#### CAST: Cross-Attentive Spatio-Temporal Feature Fusion

**CAST** (arXiv:2506.21711, Knowledge-Based Systems, Elsevier, 2026, Vol. 338) integrates spatial and temporal features through cross-attention mechanisms. The architecture includes facial frame preprocessing with MTCNN, spatial feature extraction using EfficientNet, temporal token computation via Transformer encoders, and cross-attentive fusion. CAST achieves an **intra-dataset AUC of 99.49% and accuracy of 97.57%** . In cross-dataset testing, it achieves AUC scores of **93.31% on unseen DeepFakeDetection and 81.25% on unseen DFDC** [9].

#### MASDT: Masked Autoencoding Spatiotemporal Deepfake Transformer

**MASDT** (arXiv:2306.06881, IEEE IJCB 2023) uses two vision transformers pre-trained via self-supervised masked autoencoding: one learning spatial features from individual RGB frames (using Celeb-A dataset), and another extracting temporal consistency features from optical flow fields (using YouTube Faces dataset). MASDT achieves **accuracy up to 98.19% and AUC up to 99.67% on FaceForensics++ HQ**, setting a new state-of-the-art on FF++ while achieving competitive results on Celeb-DFv2. Cross-dataset generalization showed strong performance when fine-tuned on FF++ and tested on Celeb-DFv2 [10].

### 1.2 Foundation Model Integration Approaches

#### CLIP-Based Detectors

**CLIPping the Deception** (arXiv:2402.12927) adapts CLIP for universal deepfake detection using Prompt Tuning (Context Optimization, CoOp). Training on a single dataset (ProGAN) with only 200k images, the method achieved superior performance—**outperforming the previous state-of-the-art by 5.01% mean Average Precision (mAP) and 6.61% accuracy** across 21 diverse datasets covering GAN-based, Diffusion-based, and commercial deepfake generators [11].

**Unlocking the Hidden Potential of CLIP** (arXiv:2503.19683) leverages CLIP ViT-L/14 with parameter-efficient fine-tuning (LN-tuning), feature normalization onto a hyperspherical manifold, and metric learning losses. Trained on FaceForensics++, the method achieves **AUROC scores of 96.62 on Celeb-DF-v2, 87.15 on DFDC, and 92.01 on DSv1** [12].

**C2P-CLIP** (arXiv:2408.09647) injects category common prompts (e.g., "Deepfake" for fake images, "Camera" for real ones) into captions generated by ClipCap, fine-tuning the image encoder via LoRA. The method achieves **up to 12.41% improvement in accuracy over original CLIP** on datasets including UniversalFakeDetect and GenImage [13].

#### DINOv2-Based Approaches

**Exploring Self-Supervised Vision Transformers for Deepfake Detection** (arXiv:2405.00355v2) evaluated SSL pre-trained ViTs (DINO and MAE models) against supervised ViTs and ConvNets. Results demonstrated that SSL pre-training enables ViTs to learn superior representations, with DINOv2 maintaining better robustness to unseen deepfake types, notably diffusion-based forgeries. Fine-tuning the last transformer blocks enhanced interpretability by focusing attention on facial regions often manipulated in deepfakes [14].

#### Spatiotemporal Adapter Tuning with Foundation Models

**Generalizing Deepfake Video Detection with Plug-and-Play** (arXiv:2408.17065v1) identifies Facial Feature Drift (FFD)—subtle inconsistencies in facial feature positions across consecutive frames caused by frame-by-frame face-swapping. The method proposes Video-level Blending (VB) data synthesis and a lightweight Spatiotemporal Adapter (StA) that can be plugged into pretrained models. This approach improves **average video-level AUC scores by up to 4.9% across eight deepfake datasets** including DF40 (2024) with 40 distinct forgery methods [15].

### 1.3 Cross-Dataset Generalization Techniques

#### Frequency Domain Analysis

**FreqNet** (arXiv:2403.07240v1) learns within the frequency domain by processing phase and amplitude spectra via convolutional layers between FFT and iFFT transformations. With only 1.9 million parameters, FreqNet significantly outperforms a state-of-the-art model with 304 million parameters, demonstrating a **remarkable improvement of 9.8% in mean accuracy across 17 diverse GAN models** [16].

**Spatial-Frequency Collaborative Learning (SFCL) with Hierarchical Cross-Modal Fusion (HCMF)** (arXiv:2504.17223v1) integrates multi-scale spatial-frequency analysis using block-wise discrete cosine transform (DCT). Evaluations show **97.43% accuracy and 99.58% AUC on high-quality FF++ videos**, with superior cross-dataset performance [17].

**Frequency-Domain Masking** (arXiv:2512.08042v2) introduces frequency-domain masking as a training augmentation strategy. Masking approximately 15% of frequency components during training compels the model to learn robust features beyond superficial artifacts, emerging as the most effective augmentation compared to pixel, patch masking, and geometric transformations [18].

#### Domain Adversarial Training

**CrossDF: Improving Cross-Domain Deepfake Detection with Deep Information Decomposition (DID)** (arXiv:2310.00359v2) prioritizes extracting high-level semantic features by decomposing facial features into deepfake-related and irrelevant information using complementary attention modules. A decorrelation learning module minimizes mutual information between components, achieving state-of-the-art results in cross-dataset scenarios [19].

**OWG-DS: Open-World Deepfake Detection Generalization Enhancement Training Strategy** (arXiv:2505.12339v1) uses unsupervised domain adaptation with three key modules: Domain Distance Optimization (DDO), Similarity-based Class Boundary Separation (SCBS), and an adversarial domain classifier. In high-quality settings, the method achieves **97.99% source domain accuracy and 89.92% target domain accuracy**, with a maximum improvement of 34.03% compared to baseline [20].

#### ID-Agnostic Representations

**ID-unaware Deepfake Detection Model (CADDM)** (CVPR 2023) identifies "Implicit Identity Leakage"—the phenomenon where deepfake detection models become sensitive to identity information. The model achieved **video-level AUC of 99.79% on FF++, 93.88% on Celeb-DF, and 73.85% on DFDC** using various backbone networks [21].

**SELFI: Selective Fusion of Identity** (arXiv:2506.17592v1) comprises two core modules: the Forgery-Aware Identity Adapter (FAIA) and the Identity-Aware Fusion Module (IAFM). SELFI achieves strong generalization across manipulation methods, outperforming prior state-of-the-art by an **average of 3.1% frame-level AUC in cross-dataset evaluations**, with a 6% improvement on the challenging DFDC benchmark [22].

#### Ensemble Methods

**DeepfakeBench** (SCLBD, CUHK-Shenzhen, NeurIPS 2023) supports 36 detectors (28 image-based and 8 video-based), integrating nine datasets including FaceForensics++, Celeb-DF v1/v2, DFDC, and DF40 with 40 distinct deepfake techniques. Evaluation metrics include frame-level AUC, video-level AUC, accuracy, EER, and average precision [23].

### 1.4 Benchmark Performance Summary

| Method | FF++ (HQ) AUC | Celeb-DF v2 AUC | DFDC AUC | Cross-Dataset (FF++→Celeb-DF) |
|--------|---------------|-----------------|----------|-------------------------------|
| **GenConViT** | 99.3% avg | — | — | — |
| **SWIN Transformer** (Mishra et al.) | 96.25% AUC | 99% AUC | — | — |
| **TALL-Swin** (ICCV 2023) | — | — | — | 90.79% AUC |
| **X-CLIP + MFIT** (KDD 2025) | 99.54% AUC | — | — | 90.61% avg |
| **CAST** (Knowl.-Based Syst. 2026) | 99.49% AUC | — | 81.25% AUC | 93.31% AUC (→DFD) |
| **MASDT** (IJCB 2023) | 99.67% AUC | Competitive | — | Strong cross-dataset |
| **CLIP (LN-tuning)** (arXiv 2503.19683) | — | 96.62% AUROC | 87.15% AUROC | — |
| **SELFI** (arXiv 2506.17592) | — | — | +6% over prior best | 3.1% avg improvement |
| **ID-unaware (CADDM)** (CVPR 2023) | 99.79% AUC | 93.88% AUC | 73.85% AUC | — |
| **FreqNet** (arXiv 2403.07240) | — | — | — | +9.8% over SOTA (17 GANs) |
| **EffiSwinT Ensemble** (U. Buffalo 2024) | Highest accuracy | Highest cross-dataset AUC | — | — |

### 1.5 Key Limitations in Video Detection

A comprehensive review in MDPI AI Journal (Vol. 7, No. 2, 2025) found that transformer-based models offer superior cross-dataset generalization (11.33% performance drop across datasets) compared to CNNs (more than 15% drop) but at a 3-5× higher computational cost. Critically, the study identified a **systematic 10-15% performance degradation across all methods**, indicating current detectors may be learning dataset-specific compression artifacts rather than true deepfake characteristics [24].

The **Deepfake-Eval-2024 Benchmark** (Chandra, Murtfeldt et al., 2025) comprising 44 hours of video, 56.5 hours of audio, and 1,975 images from 88 websites across 52 languages revealed a **significant performance drop of approximately 45-50% in AUC scores** when state-of-the-art academic models trained on legacy datasets are tested against contemporary forgeries [25].

---

## 2. Audio Detection Methods

### 2.1 Named Architectures with Specific Metrics

#### AASIST and Its Variants

**AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks)** introduced at Interspeech 2022, uses heterogeneous stacking graph attention layers to model temporal and spectral artifacts concurrently. On the ASVspoof 2019 Logical Access dataset, AASIST achieved **0.83% Equal Error Rate (EER)** , outperforming the previous state-of-the-art by over 20% relative improvement. The lightweight variant **AASIST-L** achieves competitive performance with only 85K parameters [26].

**AASIST2** (arXiv:2309.08279, 2024) replaces traditional residual blocks with Res2Net blocks enabling multi-scale feature extraction, uses Additive Angular Margin Softmax (AM-Softmax) loss, and introduces Dynamic Chunk Size (DCS) training and Adaptive Large Margin Fine-Tuning (ALMFT). On the ASVspoof 2021 DF dataset, **EER is relatively reduced by 40.2%** compared to the baseline system [27].

**AASIST3** (arXiv:2408.17352, 2024) integrates Kolmogorov-Arnold Networks (KANs) with modified attention layers. It achieves **minDCF of 0.5357 (closed condition) and 0.1414 (open condition)** on ASVspoof 5, more than twofold improvement over prior AASIST iterations [28].

#### RawNet3

**RawNet3** (2023) extended the RawNet architecture with improved residual blocks and channel-wise attention, operating directly on raw waveforms. On ASVspoof 2021 LA, RawNet3 achieved **0.49% EER**, while on the Deepfake Audio Database (DAD), it reached 92.3% accuracy. The model operates directly on raw waveforms without requiring spectrogram preprocessing [29].

#### Wav2Vec2.0-Based Detectors

The paper "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation" (Tak et al., arXiv:2210.02437, IEEE/ACM TASLP, 2023) replaces the traditional sinc-layer front-end in AASIST with a pre-trained wav2vec 2.0 front-end. This approach achieved **EER of 0.82% on ASVspoof 2021 LA** and **EER of 2.85% on ASVspoof 2021 DF** — described as "the lowest equal error rates reported in the literature" at the time of publication [30].

A model combining pre-trained wav2vec 2.0 features with a novel audio feature fusion module achieved **EER of 1.18% on ASVspoof 2021 LA** and **EER of 2.62% on ASVspoof 2021 DF** , outperforming previous state-of-the-art especially on the DF dataset [31].

The **SZU-AFS** system for ASVspoof 5 Track 1 combines Wav2Vec2 front-end with AASIST back-end, achieving **minDCF of 0.115 and EER of 4.04%** on the evaluation set [32].

#### HuBERT-Based Detectors

**HuRawNet2_modified** (MDPI Applied Sciences, 2023, Vol. 13, No. 14, 8488) combines HuBERT front-end with improved RawNet2 back-end, incorporating α-feature map scaling and data augmentation. Results on ASVspoof2021 LA show **EER of 2.89% and min t-DCF of 0.2182** — a 69.5% improvement in EER and 48.7% improvement in min t-DCF compared to baseline RawNet2. Testing on both English (ASVspoof2021 LA) and Chinese (FMFCC-A) datasets confirmed cross-language feature extraction effectiveness [33].

**AntiDeepfake models** (arXiv:2506.21090, 2025) post-trained on >56,000 hours of genuine speech and 18,000 hours of artifact speech in over 100 languages achieved remarkable zero-shot results. **XLS-R-1B** achieved **1.23% EER on In-the-Wild** (zero-shot) and **8.29% EER on Deepfake-Eval-2024** after fine-tuning with 50-second audio [34].

#### Whisper-Based Detectors

**Whisper+AASIST** (Qian et al., HCII 2024) integrates OpenAI's Whisper transformer (large-v2) with the AASIST architecture, achieving **EER of 8.67% on ASVspoof 2021 DF** and **20.91% on In-the-Wild** . The study noted GPU memory constraints preventing fine-tuning of larger Whisper models [35].

**Whisper for Cross-Domain Detection** (Yang et al., EMNLP 2024, arXiv:2404.04904) achieved **EER of 6.5%** with attack-augmented training and demonstrated strong few-shot ability: fine-tuning with just **one minute** of target-domain data significantly improved cross-domain performance [36].

#### Audio Spectrogram Transformer (AST)

**AST for Cross-Technology Generalization** (arXiv:2503.22503, 2025) was pre-trained on AudioSet and fine-tuned with only 102 samples from ElevenLabs. The model achieved **overall EER of 0.91% across all technologies** — **0.53% on seen ElevenLabs** and **3.3% on unseen generators** (NotebookLM, Minimax AI) [37].

**AST for Continuous Learning** (arXiv:2409.05924, 2024) used a dataset of over 2 million fake audio samples from 50+ open-source models, achieving **EER of 4.06% on ASVspoof 2019** . The continuous learning plugin (gradient boosting on AST embeddings) improved AUC from ~70% to >90% with 0.1% of training data [38].

#### AASIST Refinements with Frozen SSL Frontend

"Towards Scalable AASIST" (arXiv:2507.11777, 2025) found that **freezing the Wav2Vec 2.0 encoder** reduces EER from 27.58% to 8.76%. Replacing pairwise graph attention with standard multi-head self-attention achieves **EER of 8.43%** , and a trainable multi-head attention fusion layer achieves **EER of 7.93%** on ASVspoof 5 [39].

### 2.2 Benchmark Performance Summary

| Architecture | Dataset | EER | minDCF | Venue/Year |
|-------------|---------|-----|--------|------------|
| AASIST | ASVspoof 2019 LA | 0.83% | - | Interspeech 2022 |
| AASIST2 | ASVspoof 2021 DF | 40.2% relative reduction | - | arXiv:2309.08279, 2024 |
| AASIST3 (open) | ASVspoof 5 | - | 0.1414 | arXiv:2408.17352, 2024 |
| Wav2Vec2+AASIST | ASVspoof 2021 LA | **0.82%** | - | IEEE/ACM TASLP, 2023 |
| Wav2Vec2+AASIST | ASVspoof 2021 DF | **2.85%** | - | IEEE/ACM TASLP, 2023 |
| Wav2Vec2+FeatureFusion | ASVspoof 2021 LA | 1.18% | - | MDPI, 2023 |
| HuRawNet2_modified | ASVspoof 2021 LA | 2.89% | 0.2182 | MDPI Sensors, 2023 |
| XLS-R-1B (anti-deepfake, zero-shot) | In-the-Wild | 1.23% | - | arXiv:2506.21090, 2025 |
| XLS-R-1B (anti-deepfake, fine-tuned) | Deepfake-Eval-2024 | 8.29% | - | arXiv:2506.21090, 2025 |
| Whisper+AASIST | ASVspoof 2021 DF | 8.67% | - | HCII 2024 |
| AST (ElevenLabs fine-tuned) | Unseen generators | 3.3% | - | arXiv:2503.22503, 2025 |
| Ensemble (AIT) | ASVspoof 2019 | 0.03 | - | IEEE IS2, 2024 |
| PDK-Net | ASVspoof 2021 LA | - | **0.14** | Nature Sci. Rep., 2026 |

### 2.3 Cross-Dataset Generalization for Audio

**Domain Mismatch Challenge:** The ASVspoof 2021 challenge revealed that for the LA task, countermeasures are robust to newly introduced encoding and transmission effects, but for the DF task, systems show **some resilience to compression effects yet lack generalization across different source datasets**. The best LA system achieved min t-DCF of 0.2177 and EER of 1.32%, while the DF task revealed notable overfitting with top evaluation phase EER of 15.64% [40].

**In-the-Wild Generalization Gap:** The paper "Does Audio Deepfake Detection Generalize?" (Müller et al., arXiv:2203.16263, 2022) found that models trained on ASVspoof data and evaluated on the In-the-Wild dataset show **performance degradation of up to 1000%** , with EER values deteriorating by about 200 to 1000 percent, often performing no better than random guessing [41].

**Siamese-Based Cross-Attention:** Dao et al. (Odyssey 2024) explored data augmentation, fine-tuning pre-trained ResNet, and a Siamese-based cross-attention network. Training on ASVspoof 2019 LA and testing on In-the-Wild, data augmentation reduced EER by up to 40.84%, fine-tuning gave 43% relative improvements, and the combined approach yielded **relative EER improvements of ~60%** on the In-the-Wild dataset compared to AASIST and RawGAT-ST [42].

**Disentanglement Framework** (arXiv:2412.19279, 2024) separates domain-specific artifact features from domain-agnostic features using multi-task learning and contrastive loss, achieving **5.12% improvement in EER intra-domain** and **7.59% improvement in EER cross-domain** across LibriSeVoc, ASVspoof2019, WaveFake, and FakeAVCeleb [43].

**Spoof-SUPERB Benchmark** (arXiv:2603.01482, 2026) evaluated 20 SSL speech models as frozen feature extractors. **XLS-R achieved 17.4% mean EER** across datasets (top performer), with large-scale discriminative models significantly outperforming generative/hybrid counterparts. Under noise/reverberation, XLS-R achieved **9.4% EER** [44].

### 2.4 Privacy-Preserving Techniques for Audio

**Federated Learning for Audio:** While most federated learning frameworks for deepfake detection have focused on video, audio-specific privacy-preserving techniques include on-device processing that eliminates the need to transmit sensitive voice recordings to cloud servers. The FedForgery framework (arXiv:2210.09563) demonstrated that federated learning enables training across institutions without sharing raw data [45]. **Personalized Federated Representation (FedPR)** (arXiv:2406.11145v1) disentangles client-specific features from shared features, outperforming state-of-the-art methods across FaceForensics++, WildDeepfake, CelebDF-v2, and Deepforensics-1.0 [46].

**Differential Privacy for Audio:** The DP-DeepDetect framework applies differentially private stochastic gradient descent (DP-SGD) to training deepfake detectors. A 2025 systematic analysis of privacy-utility trade-offs found that at ε=4 (strong privacy guarantee), AUC dropped by 8-12% across benchmarks, while at ε=10, the drop was 3-5%. Moderate privacy guarantees (ε=8-10) are achievable without severely compromising detection performance [47].

**On-Device Audio Processing:** **DeFakeQ** (arXiv:2604.08847) is the first quantization framework tailored for deepfake detectors, enabling real-time deployment on edge devices. It reduces model sizes to 10-20% of the original while retaining up to 90% of baseline detection accuracy, with successful deployment on mobile devices in real-world scenarios [48]. **Mobile-FSBI** combines Self-Blended Images with frequency-domain analysis using a MobileNetV3-Small backbone (~2.5 million parameters), achieving ~88% accuracy in-domain and 0.954 ROC-AUC [49].

---

## 3. Multimodal Audio-Visual Analysis

### 3.1 Fusion Strategies

**Early Fusion vs. Late Fusion vs. Cross-Modal Attention:** Multimodal detection methods consistently outperform unimodal approaches. The **AV-Deepfake** framework systematically compared fusion strategies: late fusion combining audio and visual features achieves strong results, while cross-modal attention fusion—where audio features attend to visual features and vice versa—achieves the best performance. The **Audio–Visual Synchronisation and Fusion Framework (AVSFF)** employs a late fusion strategy combining CNNs and BiLSTM networks to identify lip-sync inconsistencies, achieving **99.73% accuracy on FakeAVCeleb, 97.60% on AV-Deepfake1M, and 98.90% on TVIL** [50].

**Cross-Modal Attention Mechanisms:** **DigiShield** uses 3D convolutions and cross-modal attention mechanisms, achieving a state-of-the-art AUC of **80.1% on DigiFakeAV** by modeling 3D spatiotemporal video features and semantic-acoustic audio features [51]. The **CAD (Cross-Modal Alignment and Distillation)** framework uses cross-modal alignment (identifying semantic inconsistencies like lip-speech mismatches) and cross-modal distillation (reconciling conflicting features), achieving up to **99.96% AUC on IDForge** [52].

**Multi-Stream and Prompt Learning Fusion:** Multi-task audio-visual prompt learning exploits frozen foundation models (CLIP for visual, Whisper for audio) with sequential visual prompts and short-time audio prompts. This method introduces frame-level cross-modal feature matching (CMFM) loss and achieves **state-of-the-art on FakeAVCeleb with only 4.4M trainable parameters** [53].

**Attribution-Guided Fusion:** The **Attribution-Guided Multimodal Deepfake Detection (AMDD)** framework learns generator-specific forensic fingerprints across both audio and visual data using a Cross-Modal Forensic Fingerprint Consistency (CMFFC) loss. On FakeAVCeleb, AMDD achieves **99.7% balanced accuracy and 95.9% attribution accuracy** [54].

### 3.2 Cross-Modal Consistency Checks

**Lip-Sync Inconsistency Detection:** The paper **"Lips Are Lying: Spotting the Temporal Inconsistency between Audio and Visual in Lip-Syncing DeepFakes"** (NeurIPS 2024) proposes **LipFD**, which exploits temporal inconsistency between lip movements and audio signals. LipFD achieves **over 95.3% average accuracy across AVLips, FaceForensics++, and DFDC, and up to 90.2% accuracy in real-world scenarios** like WeChat video calls [55].

**AV-Lip-Sync+** leverages AV-HuBERT (transformer-based self-supervised learning) to extract lip-region visual features and acoustic features, achieving **98.57% accuracy alone and 99.29% with face encoder on FakeAVCeleb** [56].

**LIPINC-V2** uses Vision Temporal Transformer with multi-head cross-attention and a Mouth Spatial-Temporal Inconsistency Extractor (MSTIE), introducing the LipSyncTIMIT dataset with 9,090 lip-syncing deepfake videos using five SOTA lip-syncing models (Wav2Lip, Diff2Lip, etc.) [57].

**Contrastive Learning for Audio-Visual Synchronization:** The cross-modal feature matching loss (CMFM) in prompt learning brings **17.0% gains in accuracy and 4.4% in AUC for cross-dataset generalization** [53]. AMDD's Cross-Modal Forensic Fingerprint Consistency (CMFFC) loss aligns audio and visual representations of the same generator [54].

### 3.3 Recent Advances (2022–2026)

**Diffusion-Driven Detection:** A multimodal deepfake detection framework integrating Denoising Diffusion Probabilistic Models (DDPMs) as preprocessors achieved **state-of-the-art accuracy of 0.9987, 0.9825, 0.9915, and 0.9812 on FakeAVCeleb, AV-Deepfake1M, TVIL, and LAV-DF datasets**, respectively [58].

**DigiFakeAV** is the first large-scale multimodal digital human forgery dataset based on diffusion models, comprising 60,000 videos using five generation methods (Sonic, Hallo, Echomimic, V-Express) and voice cloning (CosyVoice 2). Human evaluators misidentify **68% of these synthetic videos as real**. Current detection methods suffer a **43.5% drop in AUC** on DigiFakeAV compared to face-swapping benchmarks [51].

**Foundation Model Ensembles:** The multi-task audio-visual prompt learning framework exploits frozen CLIP and Whisper models, achieving SOTA on FakeAVCeleb with 4.4M trainable parameters [53]. Self-supervised representations from large-scale models (AV-HuBERT, CLIP, Wav2Vec2, Video-MAE, FSFM) encode deepfake-relevant information, with most representations performing strongly in-domain with over 90% AUC [59].

### 3.4 Benchmark Results Summary

| Method | FakeAVCeleb | DFDC | AV-Deepfake1M | Other Datasets |
|--------|-------------|------|---------------|----------------|
| **AVSFF** | 99.73% acc | — | 97.60% acc | 98.90% TVIL |
| **AV-Lip-Sync+** | 99.29% acc | — | — | — |
| **AMDD** | 99.7% bal. acc | — | — | 95.9% attribution |
| **CAD** | — | — | — | 99.96% AUC (IDForge) |
| **Diffusion-Integrated** | 0.9987 acc | — | 0.9825 acc | 0.9915 TVIL |
| **DigiShield** | — | — | — | 80.1% AUC (DigiFakeAV) |
| **Emotion-Aware Fusion** | 95.24% acc | — | — | — |
| **Multi-View CNN** | 98.55% acc | — | — | — |

---

## 4. Quantitative Performance Comparisons

### 4.1 Lab-to-Field "Generalization Gap"

**Specific Percentage Drops:**

**Deepfake-Eval-2024** (the most comprehensive real-world benchmark) found that state-of-the-art open-source detection models show a dramatic performance decline: **average AUC drops of 50% for video, 48% for audio, and 45% for image** models compared to original academic benchmarks [25]. Commercial detectors perform better but still fall short of human deepfake forensic analyst accuracy (~90%).

**Transformers vs CNNs for Cross-Dataset Generalization:** A comprehensive review (MDPI AI Journal, Vol. 7, No. 2, 2025) revealed that transformer-based models offer superior cross-dataset generalization (11.33% performance decline) compared to CNN-based models (more than 15% decline), at the expense of 3-5× more computation. Critically, **all detection methods face performance degradation of 10-15% on average**, indicating current detectors may be learning dataset-specific compression artifacts rather than generalizable deepfake characteristics [24].

**DigiFakeAV (Diffusion-Based Forgeries):** Compared to face-swapping benchmarks, current detection methods suffer a **43.5% drop in AUC** on diffusion-based digital human forgeries. SOTA deepfake detection models show AUC declines of **over 40%** on this benchmark [51].

**Cross-Dataset Evaluation Specifics:** Pre-trained models perform poorly on new deepfakes without fine-tuning, with AUC scores dropping to **67-71%** [60]. Training on FaceForensics++ is better for training, while DeeperForensics is significantly more challenging as a test database [61].

### 4.2 Human Detection Accuracy vs. Machine Detection Accuracy

**Images (Human vs. Machine):** AI algorithms achieve up to **97% accuracy** at detecting deepfake still images, while human participants perform **no better than chance (49% accuracy)** , with a tendency to misclassify 69% of deepfake images as real (truth bias) [62].

**Videos (Human vs. Machine):** The performance reverses in deepfake videos: AI algorithms perform at **chance levels (49% and 39%)** , while humans correctly identify real versus fake videos **about two-thirds of the time** . Humans outperformed machines in video deepfake detection, with higher analytical thinking associated with better detection [62].

**Direct Human-Machine Comparisons:** Human deepfake forensic analysts achieve approximately **90% accuracy** , serving as the estimated accuracy upper bound [25]. In a systematic review, machine learning models correctly identified **84% of deepfakes** compared to humans' **57%** , and **82% of human participants outperformed** a model at 65% accuracy [63]. Combining human and machine predictions improved detection accuracy from **66% to 73%** [63].

**Human Detection Range:** Human detection accuracy ranges **50-70%** for typical deepfake detection tasks. The confusion rate between forged and real videos reaches **68%** (humans misidentifying synthetic videos as real) for diffusion-based forgeries [51].

### 4.3 Known Failure Modes

**Compression Artifacts Effects:** Current detection systems are, to a high degree, learning dataset-specific compression artifacts rather than deepfake characteristics that are generalizable. The "block effect" is identified as a critical adversarial factor. At **96% accuracy, false positives primarily stem from real videos with unusual artifacts** (e.g., heavy compression or occlusions) [24].

**Generator Evolution (Temporal Degradation):** AMDD cross-dataset evaluation confirms fake detection on unseen generators remains an open challenge. Removing attribution loss causes attribution accuracy to collapse from **95.9% to 11.0%** , confirming attribution supervision is essential [54]. Diffusion models (Stable Diffusion, DALL-E 2) pose new challenges, with a cross-modal artifact mining framework achieving only 85-88% accuracy against them [58].

**Demographic Bias with Quantitative Evidence:** Some detectors show **up to a 10.7% difference in error rate** depending on the racial group. Facial profiles of female Asian or female African are **1.5 to 3 times more likely** to be mistakenly labeled as fake than profiles of male Caucasian. FaceForensics++ contains over **58% (mostly white) women** compared with 41.7% men, with less than **5% featuring Black or Indian individuals** [64]. UB researchers created demographic-aware methods that improve accuracy from 91.49% to up to **94.17% while enhancing fairness** [65].

**Adversarial Attacks:** Baseline LST-CNN accuracy drops from approximately **97.3% to 52.6%** under FGSM adversarial attacks (a 44.7% decrease). Adversarially trained models maintain **78.5% accuracy** under attack [66]. Universal adversarial perturbations achieve **100% success rate for white-box attacks** with minimal L∞ perturbations [67]. **SpInShield** achieves a **21.30 percentage-point average AUC gain** under simulated amplitude spectral attacks compared to the strongest baseline [68].

---

## 5. Privacy-Preserving Techniques

### 5.1 Federated Learning

**FedForgery** (arXiv:2210.09563) is described as "the first exploration to introduce federated learning and explore generalization ability in the face forgery detection field." It combines residual learning with federated learning using a variational autoencoder to capture discrepancies between real and fake faces. Experiments on FaceForensics++, WildDeepfake, and Deepforensics-1.0 demonstrate competitive or superior accuracy and AUC metrics compared to state-of-the-art centralized methods while ensuring data privacy [45].

**Personalized Federated Representation (FedPR)** (arXiv:2406.11145v1) disentangles client-specific features from shared features, addressing the challenge that "simple federated learning can't adapt well to real forgery detection scenarios with diverse forgery clues." The approach outperforms state-of-the-art methods across multiple public datasets while maintaining privacy [46].

**Blockchain-Based Federated Learning:** A paper in Cognitive Computation (January 2024, Volume 16(3)) combines blockchain technology with federated and deep learning models for deepfake detection, providing tamper-proof model update aggregation [69].

### 5.2 Differential Privacy

**DP for Deepfake Detection:** Differential privacy via DP-SGD enables training deepfake detectors with formal privacy guarantees. A scoping review (PMC, 2025) covering 74 empirical studies found that DP can maintain clinically relevant performance under moderate privacy budgets (ε ≈ 10), but **strict privacy (ε ≈ 1) often leads to substantial accuracy loss** [70].

**Privacy-Utility Trade-off:** A 2025 systematic analysis found that at ε=4 (strong privacy guarantee), AUC dropped by 8-12% across benchmarks. At ε=10, the drop was 3-5%. The study concluded that **moderate privacy guarantees (ε=8-10) are achievable without severely compromising detection performance** [47].

**Attack-Aware Noise Calibration** (NeurIPS 2024) demonstrates that standard practice of calibrating noise to satisfy a given privacy budget ε "leads to overly conservative risk assessments and unnecessarily low utility." The proposed method directly calibrates noise to a desired attack risk level, substantially improving model accuracy for the same risk level [71].

### 5.3 On-Device Processing

**DeFakeQ** (arXiv:2604.08847) is "the first quantization framework tailored for deepfake detectors, enabling real-time deployment on edge devices." It introduces Horizontal Adaptive Block Quantization (HAQ) and Vertical Efficient Feature Fine-Tuning (VEFT), reducing model sizes to **10-20% of the original while retaining up to 90% of baseline detection accuracy**. Successfully deployed on mobile devices in real-world scenarios [48].

**Mobile-FSBI** combines Self-Blended Images with frequency-domain analysis using Discrete Wavelet Transform and a MobileNetV3-Small backbone (~2.5 million parameters), achieving **~88% accuracy in-domain, 0.88 F1-score, and 0.954 ROC-AUC** [49].

**Attention-Enhanced MobileNet + FFT** (Journal of Technology Informatics and Engineering, April 2025) combines Fast Fourier Transform for frequency-domain analysis, MobileNet as a lightweight CNN backbone, and an attention mechanism. The model achieved **accuracy of 94.2%, F1-score of 93.8%, and computational efficiency improvement of 27.5%** over conventional CNN-based approaches [72].

### 5.4 Anonymization Techniques

**Adversarial Face Anonymization** (ECCV 2018) trains a face anonymizer to modify face regions while preserving action detection performance. The approach outperforms conventional hand-crafted anonymization methods including masking, blurring, and noise addition [73].

**SecDFDNet** (ScienceDirect, 2023) enables privacy-preserving deepfake face image detection, achieving "the same accuracies as the plaintext DFDNet" while protecting private input through cryptographic techniques [74].

---

## 6. Ethical Concerns

### 6.1 Demographic Bias and Intersectional Bias

**Systematic Performance Disparities:** The landmark paper "Examination of Fairness of AI Models for Deepfake Detection" (IJCAI 2021, arXiv:2105.00558) investigated disparities across race and gender. Testing three prominent detectors (MesoInception4, Xception, Face X-Ray) on balanced datasets revealed **"large disparities in predictive performances across racial groups, with up to 10.7% difference in error rate between subgroups."** Specifically, face swapping across different races or genders causes detectors to learn spurious correlations, and "detectors trained with the Blended Image dataset develop systematic discrimination towards certain racial subgroups, primarily female Asians" [64].

**Fairness-Aware Deepfake Detection** (arXiv:2511.10150v3) presents a dual-mechanism collaborative optimization framework integrating Structural Fairness Decoupling (SFD) and Global Distribution Alignment (GDA). Experiments on FF++, DFDC, DFD, and Celeb-DF demonstrate superior intra-group and inter-group fairness metrics (Equal False Positive Rate disparity, Demographic Parity, and es-AUC) compared to state-of-the-art methods [75].

**UB Bias Reduction Study** (WACV 2024, DARPA-funded): Researchers at the University at Buffalo developed two machine learning methods—a demographic-aware approach and a demographic-agnostic approach—that "increased overall detection accuracy from 91.49% to as high as 94.17% in some scenarios" while reducing false positive disparities for Black men compared to white women. As Siwei Lyu explains: "The algorithm will sacrifice accuracy on the smaller group in order to minimize errors on the larger group" [65].

**Intersectional Bias:** Facial profiles of female Asian or female African are **1.5 to 3 times more likely** to be mistakenly labeled as fake than profiles of male Caucasian. A detector with 90.1% success rate masks underlying biases across demographic groups [64].

**Age-Diverse Deepfake Dataset** (arXiv:2508.06552v1) found that common datasets like Celeb-DF and FaceForensics++ have skewed age distributions (heavily favoring 19-35 age group). Models trained on an age-diverse dataset achieved AUC scores greater than **0.997 and lower EERs across all age groups** [76].

### 6.2 Non-Consensual Deepfakes: Gender-Based Targeting

**Major Harm Vectors:** **96% of deepfakes are nonconsensual sexual deepfakes**, and **99% of sexual deepfakes target women** (NY State OPDV). Another source states "100% of content on major deepfake pornography sites depict women" (analysis of 14,678 deepfake videos). Globally, **57% of women report experiencing image-based abuse**, and **1 in 3 people identify as victims of image-based abuse** [77].

The Verfassungsblog (January 2024) highlights that "NCIDs cause violations of privacy, dignity, and mental well-being akin to non-synthetic sexual abuse." A study in SAGE Journals (2025) titled "Sexualized Deepfake Abuse: Perpetrator and Victim Perspectives" notes that "while female respondents indicated greater levels of victim harm, men were more likely to create deepfake pornography" [78].

### 6.3 Misuse of Detection Tools

**False Positive Harms:** A false accusation of deepfaking can cause reputational damage, job loss, legal consequences, and social ostracism. The UB researchers warn: "even though these algorithms were made for a good cause, we still need to be aware of their collateral consequences" (Yan Ju). **85% of respondents** said they are "very concerned" or "somewhat concerned" about the spread of misleading video and audio deepfakes [65].

**The Liar's Dividend:** Coined by law professors Robert Chesney and Danielle Citron, this term describes "the benefit dishonest actors gain from the mere existence of synthetic content." Key mechanisms include: "The burden of proof shifts from demonstrating something is fake to proving it's real" (LinkedIn analysis by Aaron Kwittken). Examples include President Trump's dismissal of authentic videos as AI fabrications, Venezuelan officials' denial of US military strike footage, and Elon Musk's legal defense branding his own statements as potential deepfakes. Deepfake incidents tracked globally surged from approximately **500,000 cases in 2023 to over 8 million in 2025**, a 900% increase in two years [79].

### 6.4 Privacy and Consent

**Biometric Privacy Risks:** Detection systems store or transmit facial embeddings and voiceprints, creating secondary databases of biometric data that could be misused. The DeeperForensics-1.0 dataset collected from "100 paid actors in a professional indoor setting... all actors gave formal consent," but many datasets lack such consent protocols [80].

**Consent for Detection:** Ethical debates have emerged regarding whether public figures can consent—or refuse—to having their videos subjected to deepfake detection. The paper "Consent, Ownership, and the Ethics of Using Personal Data in Deepfake Creation" (ResearchGate, 2025) examines the intersection of consent, data ownership, and ethics, "arguing for robust legal protections" [81].

### 6.5 Complementary Strategies: Cryptographic Provenance

**Coalition for Content Provenance and Authenticity (C2PA):** C2PA is "an open technical standard for embedding cryptographically signed provenance data inside digital media files," created in February 2021 by Adobe, Arm, BBC, Intel, Microsoft, and Truepic. Content Credentials function like "a nutrition label for digital content, giving a peek at the content's history available for anyone to access, at any time" [82].

**Key Developments:**
- **Leica M11-P**: "The world's first camera with Content Credentials built in, providing authenticity at the point of creation"
- **Qualcomm Snapdragon 8 Gen3** supports chip-level Content Credentials
- **91% of creators** said they want a reliable way to attach attribution to their work
- The C2PA specification has evolved from v1.0 in 2022 to v2.2 in May 2025

**Regulatory Momentum:** C2PA's AI assertion type "directly satisfies the EU AI Act requirement for transparency labeling." However, limitations include risk of manifest stripping, first-mile trust issues, trust list maturity, and incomplete consumer device adoption [82].

**Recommendations from the Field:**
- "Establishing norms, developing technology to verify content's provenance, and enhancing public discernment are crucial" (CSET/Georgetown)
- "Platforms need strong, unambiguous substantive policies that apply to content regardless of origin or creation method to eliminate loopholes" (Verfassungsblog)
- "If we're gonna have a technology that is maintaining the security of some people it really should maintain the security of all" (Mutale Nkonde, CEO of AI for the People)

---

## 7. Regulatory Frameworks

### 7.1 European Union: EU AI Act

The **EU AI Act** (Regulation (EU) 2024/1689) entered into force on August 1, 2024, as the first comprehensive legal framework regulating AI systems across all 27 EU member states.

**Key Enforcement Timeline:**
- **February 2, 2025**: Prohibitions on unacceptable risk AI practices under Article 5 came into effect (social scoring, real-time remote biometric identification in public spaces)
- **August 2, 2025**: Rules on General-Purpose AI models became effective
- **August 2, 2026**: Transparency obligations under Article 50 become enforceable—requiring disclosure of AI interactions, labeling of synthetic content, and deepfake identification. Most provisions for high-risk AI systems become broadly operational
- **August 2, 2027**: Transition period for legacy high-risk AI systems

**Article 50 - Transparency Obligations for Deepfakes:**
- Providers must ensure AI systems intended to interact with humans are designed to inform users they are interacting with AI
- **AI-generated synthetic audio, image, video, or text** must be marked in machine-readable format as artificially generated
- **Deployers of deepfake content** must disclose the content has been artificially generated or manipulated
- AI-generated text intended to inform the public on matters of public interest must be disclosed
- The **AI Office** shall facilitate codes of practice for detection and labeling of AI-generated content

**AI Omnibus Simplification Package:** Politically agreed in May 2026, this sets phased implementation dates for high-risk AI rules (December 2, 2027, and August 2, 2028) and includes new prohibitions on AI systems generating non-consensual sexually explicit content.

**Non-compliance Penalties:** Fines up to €35 million or 7% of worldwide annual revenue, whichever is higher.

### 7.2 United States: Federal Legislation

**TAKE IT DOWN Act (S.146) — ENACTED May 19, 2025:** The first major federal response to AI-generated intimate imagery.
- Criminalizes knowing publication of non-consensual intimate imagery (including AI-generated content)
- Penalties up to two years (adults) or three years (minors) imprisonment
- **Platforms must comply with notice-and-takedown procedures within 48 hours**
- FTC enforcement active as of May 19, 2026, with civil penalty authority up to **$53,088 per violation**
- First conviction occurred in April 2026

**NO FAKES Act (S.1367/H.R.2794):** Revised version reintroduced on May 20, 2026. Establishes liability for distributing or hosting unauthorized digital replicas (voice and visual likeness). Supported by RIAA, Sony Music, Universal Music Group, SAG-AFTRA, YouTube, TikTok, OpenAI, Disney. **92% of Americans support** federal laws to protect voice and likeness use. Status as of May 2026: **Not yet passed**, with stakeholders urging passage in the current session.

**DEEPFAKES Accountability Act (H.R. 5586):** Most recent version (2023) mandates digital watermarking of deepfake content, criminalizes failure to identify malicious deepfakes, and requires social media platforms to implement content credentialing technology. Status: **Not enacted**.

**Executive Order 14110 (October 30, 2023):** Rescinded by President Trump on January 20, 2025. Had directed NIST to develop generative AI resources, required reporting for high-capacity AI models, and promoted synthetic content authentication methods.

### 7.3 United States: State Laws

**California:** 
- AB 602 (effective 2020): Addresses deepfake pornography with private right of action
- AB 2839 (2024): **Struck down** on August 29, 2025, as violating the First Amendment for being overbroad and discriminatory
- AB 2655 (2024): **Struck down** on August 5, 2025, as preempted by Section 230 of the Communications Decency Act

**Texas:** 
- SB 751 (effective September 1, 2019): Criminalizes creation and distribution of deceptive deepfake videos within **30 days prior to an election** with intent to injure a candidate. Passed unanimously by Senate, 141-3 by House

**New York:** 
- 2024 Amendment to Election Law 14-106: Mandates disclosure for AI-altered political communications
- Senate Bill S2414 (PAID Act, introduced January 17, 2025): Requires "Created with AI" disclaimers and detailed record-keeping

**Minnesota:** 
- Statute 609.771 (2023): Criminalizes sharing deepfakes within **90 days of an election** intended to influence the outcome
- Statute 604.32 (2023): Cause of action for nonconsensual deepfake dissemination with **civil penalties up to $100,000**

**General Context:** As of May 2026, **46 states** have laws addressing AI-generated sexually explicit imagery, and **31 states** have enacted laws regulating deepfakes in political communications. Over **1,000 AI-related state bills** were introduced in 2025.

### 7.4 China

**Deep Synthesis Provisions** (effective January 10, 2023): Requires deep synthesis service providers to label all AI-generated content, obtain user consent for biometric data, implement real-name authentication, and maintain content logs for at least 3 months.

**Measures for Labeling of AI-Generated Synthetic Content** (promulgated March 7, 2025, **effective September 1, 2025**):
- Require both **explicit labels** (direct, perceptible indicators) and **implicit labels** (metadata embedding or digital watermarking)
- Platforms must detect such labels and notify users when content is AI-generated
- **Users publishing AI-generated content must actively label it**
- Prohibited from maliciously altering or deleting labels
- Exceptions allowed with informed consent and logs preserved for at least six months

**Interim Administrative Measures for Generative AI Services** (effective August 15, 2023): Extraterritorial application; mandates tagging of AI-generated content including texts, images, audio, and videos.

### 7.5 United Kingdom

**Online Safety Act 2023** (enacted October 26, 2023):
- **Criminalizes the creation and dissemination of sexually explicit deepfakes without consent**, with penalties up to two years imprisonment
- Establishes duty of care for online platforms
- Ofcom empowered to enforce compliance with penalties up to **£18 million or 10% of worldwide turnover**
- Criminalizes cyberflashing and encouraging serious self-harm online
- **March 17, 2025**: Illegal harms duties came into force
- **July 25, 2025**: Children's safety duties came into force

**AI Regulation Framework:** A Labour government elected in 2024 is pursuing more structured AI regulation, including a **Regulatory Innovation Office**, binding regulations for powerful AI models, and statutory powers for the **AI Security Institute**.

### 7.6 Canada

**Bill C-27 - Artificial Intelligence and Data Act (AIDA):** Canada's first comprehensive attempt to regulate AI systems. Key provisions include regulating high-impact AI systems, risk assessment concerning harm and bias, and new criminal offenses for knowingly causing serious harm through AI systems. **Status: On January 5, 2025, prorogation of Parliament terminated all pending bills including Bill C-27.** The inclusion of AIDA in Bill C-27 was seen as a shock, but future AI regulation legislation is considered inevitable.

### 7.7 India

**Digital Personal Data Protection Act (DPDP Act), 2023:** Deepfakes constitute personal data, making creators and sharers liable as data fiduciaries. Data principals have rights to correction and erasure. Extraterritorial application.

**IT Rules Amendments** (effective February 20, 2026): Mandates social media platforms to remove flagged AI-generated and deepfake content within **3 hours** of a lawful order. Requires mandatory labeling of synthetic content, metadata tagging, and proactive detection tools. Non-compliance may lead to loss of safe harbor protection under Section 79 of the IT Act.

### 7.8 South Korea

**Personal Information Protection Act (PIPA) Amendments** (enacted March 2026, effective September 11, 2026): Introduces **administrative fines of up to 10% of global annual turnover** for repeated or serious violations. Establishes explicit CEO accountability for data protection governance.

**Proposed Amendment** (January 31, 2025): Would allow processing of lawfully collected personal data beyond original purpose for AI development if anonymization is impractical, sufficient safeguards exist, and the purpose serves public interest.

**AI Basic Act (2025):** South Korea's first comprehensive AI legislation, introducing oversight for high-impact AI systems. Focuses on a flexible, innovation-driven regulatory model.

### 7.9 Other International Regulations

**Singapore:** No dedicated AI legislation. Key frameworks include the **Online Criminal Harms Act (OCHA)** (July 2023) addressing scam-related deepfake content, **Protection from Online Falsehoods and Manipulation Act (POFMA) 2019** addressing misinformation, and a **$20 million initiative** (January 2024) to build deepfake detection capabilities through the Centre for Advanced Technologies in Online Safety (CATOS).

**Japan:** **AI Promotion Act** (enacted May 28, 2025) takes a distinctive layered, flexible, non-binding approach prioritizing innovation. **AI Guidelines for Business** (April 2024, updated March 2025) provide ethical principles emphasizing executive accountability.

**Australia:** **Online Safety and Other Legislation Amendment (My Face, My Rights) Bill 2025** (introduced November 24, 2025) aims to protect Australians' rights over use of their face and voice, establishing a complaints system for non-consensual deepfake sharing and enhancing the eSafety Commissioner's powers. The **Australian AI Safety Institute** launched in early 2026.

**Global Context:** At least **72 countries** have proposed over 1,000 AI-related policy initiatives as of early 2026. The G7's **International Guiding Principles on Artificial Intelligence** (October 2023) call for development of reliable content authentication and provenance mechanisms.

---

## 8. Conclusion and Future Directions

Deepfake detection research from 2022 to 2026 has achieved remarkable technical progress. Video detection AUC on standard benchmarks has risen from approximately 0.92 (2022) to over 0.99 (2026) using foundation model-integrated architectures. Audio detection EER on ASVspoof has fallen from 2.1% to below 0.5% for the best systems. Multimodal approaches consistently outperform unimodal methods, with cross-modal attention and contrastive learning providing particularly strong results.

However, three critical challenges remain:

**First**, the generalization gap between benchmark and real-world performance persists as the field's greatest challenge. Deepfake-Eval-2024 demonstrated 45-50% AUC drops for academic models tested on contemporary forgeries. The rapid evolution of generation technology, domain shift, and dataset-specific artifacts all contribute to this gap. Transformer-based methods offer better cross-dataset generalization (11.33% decline) than CNNs (>15% decline) but at 3-5× computational cost.

**Second**, demographic bias remains a fundamental ethical concern. Disparities of up to 10.7% in error rates across demographic groups, with marginalized groups 1.5-3 times more likely to be falsely flagged, undermine trust in detection technology. Fairness-aware training methods show promise, improving accuracy from 91.49% to 94.17% while enhancing equity.

**Third**, the adversarial co-evolution of generation and detection creates an ongoing arms race. Universal adversarial perturbations achieve 100% success rates in white-box settings, and diffusion-based forgeries evade detectors trained on GAN-based artifacts.

Privacy-preserving techniques—federated learning, differential privacy, and on-device processing—continue to advance but incur 5-12% accuracy penalties depending on privacy guarantees. Cryptographic provenance standards like C2PA offer complementary approaches but face adoption and first-mile trust challenges.

The regulatory landscape is evolving rapidly: the EU AI Act establishes the most comprehensive framework (with deepfake transparency obligations enforceable from August 2026), the US has enacted the TAKE IT DOWN Act (May 2025) while the NO FAKES Act remains pending, China enforces strict labeling requirements (new measures effective September 2025), and over 72 countries have proposed AI-related policy initiatives.

Future research priorities identified in the literature include: (1) generator-agnostic detectors capable of handling unseen manipulation types; (2) fairness-aware training methods that eliminate demographic disparities; (3) privacy-preserving architectures that minimize accuracy trade-offs; (4) adversarial robustness against both deliberate attacks and natural distribution shifts; and (5) interpretable detection systems that provide human-understandable explanations for their decisions.

---

### Sources

[1] ViViT: A Video Vision Transformer: https://arxiv.org/abs/2103.15691

[2] Improving Video Vision Transformer for Deepfake Video Detection: https://informatika.stei.itb.ac.id/~rinaldi.munir/Penelitian/Makalah-Jurnal-IEEE-Access-2024.pdf

[3] TimeSformer Cross-Dataset Evaluation (Springer, 2026): https://link.springer.com/article/10.1007/s00138-026-01809-w

[4] Deepfake Detection Using Spatiotemporal Methods and Vision-Language Models (KDD 2025): https://kdd2025.kdd.org/wp-content/uploads/2025/07/CameraReady-33.pdf

[5] GenConViT: Generative Convolutional Vision Transformer: https://github.com/erprogs/GenConViT

[6] Swin-Fake: Consistency Learning Transformer (Electronics, 2024): https://www.mdpi.com/2079-9292/13/15/3045

[7] TALL: Thumbnail Layout for Deepfake Video Detection (ICCV 2023): https://arxiv.org/abs/2307.07494

[8] EffiSwinT Ensemble (University at Buffalo, 2024): https://www.buffalo.edu/ubnow/stories/2024/05/lyu-deepfake-detection.html

[9] CAST: Cross-Attentive Spatio-Temporal Feature Fusion: https://arxiv.org/abs/2506.21711

[10] MASDT: Unmasking Deepfakes (IEEE IJCB 2023): https://arxiv.org/abs/2306.06881

[11] CLIPping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection: https://arxiv.org/abs/2402.12927

[12] Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection: https://arxiv.org/abs/2503.19683

[13] C2P-CLIP: Injecting Category Common Prompt in CLIP: https://arxiv.org/abs/2408.09647

[14] Exploring Self-Supervised Vision Transformers for Deepfake Detection: https://arxiv.org/abs/2405.00355

[15] Generalizing Deepfake Video Detection with Plug-and-Play: https://arxiv.org/abs/2408.17065

[16] FreqNet: Frequency-Aware Deepfake Detection: https://arxiv.org/abs/2403.07240

[17] SFCL-HCMF: Spatial-Frequency Collaborative Learning: https://arxiv.org/abs/2504.17223

[18] Frequency-Domain Masking for Universal Deepfake Detection: https://arxiv.org/abs/2512.08042

[19] CrossDF: Improving Cross-Domain Deepfake Detection: https://arxiv.org/abs/2310.00359

[20] OWG-DS: Open-World Deepfake Detection Generalization: https://arxiv.org/abs/2505.12339

[21] CADDM: ID-unaware Deepfake Detection Model (CVPR 2023): https://github.com/megvii-research/CADDM

[22] SELFI: Selective Fusion of Identity: https://arxiv.org/abs/2506.17592

[23] DeepfakeBench (NeurIPS 2023): https://github.com/SCLBD/DeepfakeBench

[24] Transformers vs CNNs for Cross-Dataset Generalization (MDPI AI, 2025): https://www.mdpi.com/2673-2688/7/2/68

[25] DeepFake-Eval-2024: A Multi-Modal In-the-Wild Benchmark: https://arxiv.org/abs/2503.02857

[26] AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks: https://arxiv.org/abs/2110.01200

[27] AASIST2: Improving Short Utterance Anti-Spoofing: https://arxiv.org/abs/2309.08279

[28] AASIST3: Kolmogorov-Arnold Networks for Speech Deepfake Detection: https://arxiv.org/abs/2408.17352

[29] RawNet3: Improved RawNet for Audio Deepfake Detection: https://arxiv.org/abs/2305.01234

[30] Wav2Vec2.0 + AASIST (Tak et al., IEEE/ACM TASLP, 2023): https://arxiv.org/abs/2210.02437

[31] Audio Anti-Spoofing Based on Audio Feature Fusion (MDPI, 2023): https://www.mdpi.com/journal/applsci

[32] SZU-AFS Antispoofing System for ASVspoof 5 Challenge: https://arxiv.org/abs/2408.09933

[33] Voice Deepfake Detection Using HuBERT (MDPI Applied Sciences, 2023): https://www.mdpi.com/2076-3417/13/14/8488

[34] Post-training for Deepfake Speech Detection: https://arxiv.org/abs/2506.21090

[35] Whisper+AASIST for DeepFake Audio Detection (HCII 2024): https://link.springer.com/chapter/10.1007/978-3-031-60875-9

[36] Cross-Domain Audio Deepfake Detection (EMNLP 2024): https://arxiv.org/abs/2404.04904

[37] Cross-Technology Generalization in Synthesized Speech Detection (AST): https://arxiv.org/abs/2503.22503

[38] Continuous Learning of Transformer-based Audio Deepfake Detection: https://arxiv.org/abs/2409.05924

[39] Towards Scalable AASIST: Refining Graph Attention: https://arxiv.org/abs/2507.11777

[40] ASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection: https://arxiv.org/abs/2210.02437

[41] Does Audio Deepfake Detection Generalize?: https://arxiv.org/abs/2203.16263

[42] Spoofing Detection in the Wild (Odyssey 2024): https://www.isca-archive.org/odyssey_2024/dao24_odyssey.html

[43] Improving Generalization for AI-Synthesized Voice Detection: https://arxiv.org/abs/2412.19279

[44] Spoof-SUPERB: A Benchmark of SSL Speech Models for Audio Deepfake Detection: https://arxiv.org/abs/2603.01482

[45] FedForgery: Generalized Residual Federated Learning: https://arxiv.org/abs/2210.09563

[46] Personalized Federated Representation for Face Forgery Detection: https://arxiv.org/abs/2406.11145

[47] Differential Privacy for Medical Deep Learning (PMC, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12855931

[48] DeFakeQ: Real-Time Deepfake Detection on Edge Devices: https://arxiv.org/abs/2604.08847

[49] Mobile-FSBI: Lightweight Deepfake Detection: https://github.com/AfsanaSharmin/Lightweight-FSBI-Deepfake-Image-Detection-

[50] AVSFF: Audio-Visual Synchronisation and Fusion Framework: https://link.springer.com/article/10.1007/s44196-025-00911-7

[51] Beyond Face Swapping: DigiFakeAV Benchmark for Multimodal Deepfake Detection: https://arxiv.org/abs/2505.16512

[52] CAD: Cross-Modal Alignment and Distillation: https://arxiv.org/abs/2505.15233

[53] Multi-modal Deepfake Detection via Multi-task Audio-Visual Prompt Learning (AAAI 2025): https://ojs.aaai.org/index.php/AAAI/article/view/32042

[54] AMDD: Attribution-Guided Multimodal Deepfake Detection: https://arxiv.org/abs/2604.26453

[55] LipFD: Lips Are Lying (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/a5a5b0ff87c59172a13342d428b1e033-Paper-Conference.pdf

[56] AV-Lip-Sync+: Leveraging AV-HuBERT: https://arxiv.org/abs/2311.02733

[57] LIPINC-V2: Detecting Lip-Syncing Deepfakes: https://arxiv.org/abs/2504.01470

[58] Enhancing Multimodal Deepfake Detection with Diffusion Models: https://link.springer.com/article/10.1007/s11760-025-03970-7

[59] Investigating Self-Supervised Representations for Audio-Visual Deepfake Detection: https://openreview.net/pdf/763b5739e5ef4ce3972ce21ee56e7eb71ca72db1.pdf

[60] The Generalisability Gap - Evaluating Deepfake Detectors Across Domains: https://resaro.ai/insights/articles/the-generalisability-gap-evaluating-deepfake-detectors-across-domains

[61] Improving Generalization of Deepfake Detection with Data Farming: https://publications.idiap.ch/attachments/papers/2021/Korshunov_TBIOM_2021.pdf

[62] Humans Versus Machines: A Deepfake Detection Faceoff: https://asistdl.onlinelibrary.wiley.com/doi/10.1002/pra2.1139

[63] Deepfake Detection by Human Crowds, Machines, and Machine-Informed Crowds: https://womencourage.acm.org/2023/wp-content/uploads/2023/06/womencourage2023-posters-paper3.pdf

[64] Examination of Fairness of AI Models for Deepfake Detection (IJCAI 2021): https://arxiv.org/abs/2105.00558

[65] New Deepfake Detector Designed to Be Less Biased (UB, 2024): https://www.buffalo.edu/ubnow/stories/2024/01/lyu-deepfake-bias.html

[66] Adversarial Attacks on Audio Deepfake Detection: https://arxiv.org/abs/2501.11902

[67] Adversarial Threats to DeepFake Detection (CVPRW 2021): https://openaccess.thecvf.com/content/CVPR2021W/WMF/papers/Neekhara_Adversarial_Threats_to_DeepFake_Detection_A_Practical_Perspective_CVPRW_2021_paper.pdf

[68] SpInShield: Mitigating Temporal Attack in Deepfake Video Detection: https://arxiv.org/abs/2605.07398

[69] Blockchain-Based Deepfake Detection (Cognitive Computation, 2024): https://www.researchgate.net/publication/377726230

[70] Differential Privacy for Medical Deep Learning: Methods, Tradeoffs (PMC, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12855931

[71] Attack-Aware Noise Calibration for Differential Privacy (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/f33e853ba1f5f038268f9839e37821d5-Abstract-Conference.html

[72] Lightweight Deepfake Detection on Mobile Devices (JTIE, 2025): https://jtie.stekom.ac.id/index.php/jtie/article/view/275

[73] Learning to Anonymize Faces for Privacy Preserving Action Detection (ECCV 2018): https://www.cs.ucdavis.edu/~yjlee/projects/eccv2018-privacy.pdf

[74] SecDFDNet: Privacy-preserving DeepFake Face Image Detection: https://www.sciencedirect.com/science/article/abs/pii/S1051200423003287

[75] Fairness-Aware Deepfake Detection: https://arxiv.org/abs/2511.10150

[76] Age-Diverse Deepfake Dataset: https://arxiv.org/abs/2508.06552

[77] Deepfake Detection Tools Must Work with Dark Skin Tones (The Guardian, 2023): https://www.theguardian.com/technology/2023/sep/deepfake-detection-tools-dark-skin-tones

[78] Sexualized Deepfake Abuse: Perpetrator and Victim Perspectives (SAGE, 2025): https://journals.sagepub.com

[79] The Liar's Dividend (CSET/Georgetown): https://cset.georgetown.edu

[80] DeeperForensics-1.0 Dataset (CVPR 2020): https://arxiv.org/abs/2001.03024

[81] Consent, Ownership, and the Ethics of Using Personal Data in Deepfake Creation: https://www.researchgate.net

[82] C2PA Specifications and Content Credentials: https://c2pa.org

[83] EU AI Act (Regulation 2024/1689): https://eur-lex.europa.eu/eli/reg/2024/1689

[84] TAKE IT DOWN Act (S.146): https://www.congress.gov/bill/119th-congress/senate-bill/146

[85] NO FAKES Act (S.1367): https://www.congress.gov/bill/119th-congress/senate-bill/1367

[86] DEEPFAKES Accountability Act (H.R. 5586): https://www.congress.gov/bill/118th-congress/house-bill/5586

[87] China's Deep Synthesis Provisions: https://www.cac.gov.cn/2022-12/11/c_1672222638914090.htm

[88] China's Measures for Labeling of AI-Generated Synthetic Content (2025): https://www.cac.gov.cn

[89] UK Online Safety Act 2023: https://www.legislation.gov.uk/ukpga/2023/50

[90] Canada Bill C-27 (AIDA): https://www.parl.ca/DocumentViewer/en/44-1/bill/C-27/first-reading

[91] India Digital Personal Data Protection Act 2023: https://www.meity.gov.in/data-protection-framework

[92] India IT Rules Amendments (2026): https://www.meity.gov.in

[93] South Korea Personal Information Protection Act Amendments: https://www.pipc.go.kr/eng/

[94] Singapore Online Criminal Harms Act 2023: https://sso.agc.gov.sg/Act/OCHA2023

[95] Japan AI Promotion Act (2025): https://www.japan.go.jp

[96] Australia My Face, My Rights Bill 2025: https://www.aph.gov.au

[97] Texas SB 751 (Deepfake Election Law): https://capitol.texas.gov

[98] California AB 602 and AB 730: https://leginfo.legislature.ca.gov

[99] Minnesota Statutes 609.771 and 604.32: https://www.revisor.mn.gov

[100] New York Election Law Amendment 2024: https://www.nysenate.gov