# Deepfake Detection Research and Policy (2022–2026): Technical Landscape, Performance Dynamics, Privacy, Provenance Solutions, Regulation, and Ethical Risks

## Overview

Between 2022 and 2026, deepfake detection research for both video and audio has rapidly progressed in response to increasingly complex synthetic media threats. This period has been marked by:
- A migration from traditional and CNN-based methods to robust transformer, multimodal, and foundation model-based architectures.
- A growing emphasis on cross-dataset generalization, privacy-preserving detection, and integration of provenance/authenticity controls (watermarking, cryptographic signatures).
- Stringent regulatory developments across the EU, US, China, and international frameworks shaping compliance, technical standards, and market evolution.
- Renewed scrutiny of performance gaps between lab benchmarks and real-world deployment, fairness, explainability, and ethical concerns over privacy, bias, and societal harm.

This report synthesizes the state of the art in methods, performance, policy, and ethics, referencing explicit models, papers, evaluation benchmarks, and sources.

---

## 1. Recent and Prominent Technical Methods for Deepfake Detection

### Video Deepfake Detection: Key Methods and Models

- **CNN-Transformer Hybrids:**
  - *TSFF-Net* (Vision Transformer + EfficientNet): Fuses spatial and frequency-domain features via cross-attention. Achieves 97.7–98.9% accuracy depending on manipulation, and AUC of 77.1% on challenging datasets such as Celeb-DF [1].
  - *Convolutional Vision Transformers (CViT)*: Paired with CNNs for regional majority voting, achieving 85–97% on FaceForensics++ benchmarks for face region analysis [2].
  - *ResNeXt-Attn LSTM*: Uses ResNeXt-50 for feature extraction with a visual attention mechanism and LSTM for temporal modeling, demonstrating competitive generalization under cross-dataset scenarios [3].
  
- **Transformer-Based Architectures:**
  - State-of-the-art transformers, including *ViT*, *TimeSformer*, and other temporal self-attention models, provide superior cross-dataset generalization (11.33% performance decline vs. 15%+ for CNNs) but with higher computational cost [2][4].
  
- **Cross-Dataset Generalization Approaches:**
  - *CrossDF*: Decomposes image features into deepfake-related, technique-specific, and identity-based parts via domain attention, achieving an AUC of 0.802 on text-to-image diffusion datasets and 0.779 moving from FaceForensics++ to CDF2 [5].
  - *HF-FFD (High-Frequency Face Forgery Detection)*: Merges RGB with high-frequency features to boost generalization [6].
  - *SBI (Self-Blended Images) Classifier*: Improves generalization by training on artifact patterns that overlap with unknown attacks [6][7].
  - Multi-task self-supervised methods with visual transformers show best-in-class generalization across diverse datasets [8].

### Audio Deepfake Detection: Key Methods and Models

- **CNN-Transformer Hybrids:**
  - *CNN-Transformer Model for Indian Languages*: Fuses Mel-spectrogram, LFCC, and Wav2Vec features. Scored 98.07% test accuracy and precision/recall 0.98 on 10,000+ samples per language; robust on ASVspoof 2021 [9].
  - *CNN-BiLSTM Hybrid*: Uses MFCC input, achieves 99% accuracy and EER of 0.011, suitable for real-world settings due to lightweight design [10].
  - *CNN-Transformer Hybrid on ASVspoof 2019*: Achieves 91.47% classification accuracy, outperforming LSTM/CNN-LSTM/TCN [11].
  
- **Advanced Foundation Models and LLMs:**
  - *Whisper-MesoNet, SSL-AASIST, XLS-R-SLS, UncovAI*: Benchmarked on TTS-generated datasets (Dia2, Maya1, MeloTTS), with only UncovAI nearing perfect accuracy; most fail on new attack types, showing the challenges diffusion and LLM-based TTS pose [12].
  
- **Multimodal and Audio-Visual Models:**
  - *ART-AVDF* (Articulatory Representation Learning): Fuses self-supervised audio/lip encoders via cross-modal attention, outperforming many models on DFDC, FakeAVCeleb, and DefakeAVMiT [13].
  - *AVFF* (Audio-Visual Feature Fusion): Two-stage approach with contrastive and autoencoding loss, achieving 98.6% accuracy and 99.1% AUC on FakeAVCeleb [6].
  - *DeepSeek R-1 LLM*: Multimodal transformer based, gaining 90.4% accuracy and F1=0.93 on SocialDF (real-world short-form video) [14].

- **Ensemble-Based and Multimodal Strategies:**
  - *Ensemble Approaches (arXiv:2507.05996)*: Aggregate various SOTA models to boost robustness in the face of dataset and attack variability [15].

### Notable Datasets Used Across Studies

- **Video:** FaceForensics++, Celeb-DF, DFDC, Deepfake-Eval-2024, SocialDF
- **Audio:** ASVspoof 2019 & 2021, CVoiceFake, FakeAVCeleb, WaveFake, ADD Challenge, MAVOS-DD
- **Multimodal:** Deepfake-Eval-2024, FakeAVCeleb, DefakeAVMiT

---

## 2. Privacy-Preserving Deepfake Detection Approaches

- **Audio Modality:**
  - *SafeEar* ([SafeEar: Content Privacy-Preserving Audio Deepfake Detection](https://arxiv.org/abs/2409.09272)): Separates semantic (speech content) and acoustic (prosody, timbre) tokens using a neural audio codec. Detection is performed on acoustic tokens, effectively shielding actual content and enabling multilingual, cross-channel deepfake detection. Achieves EER as low as 2.02% and WER >93.93% (robust against content recovery attacks) across five benchmark datasets [16][17].

- **Video Modality:**
  - *Federated Learning and Confidential Computation*: Distributed model training (federated micro-expression mining, confidential computing, differential privacy) allows joint detection model improvement without centralizing raw video data. Supported by multi-party computation and homomorphic encryption for privacy-sensitive verticals (e.g., financial KYC, health) [18].
  - *Blockchain-Federated Schemes*: Under exploration for cryptographically auditable and privacy-respecting inference pipelines [19].

- **Industry Integration:**
  - Platforms employ a combination of forensic AI (artifact analysis), liveness/biometric checks, C2PA for provenance, and continuous monitoring to secure enterprise video onboarding and law enforcement workflows against both face and voice spoofing [20].

---

## 3. Detection Performance: Quantitative Metrics, Benchmarks, Meta-Analyses

### Controlled Benchmarks

- *CNN-LSTM and CNN-Transformer Models*: Routinely exceed 95% accuracy in lab (in-domain) conditions on FaceForensics++, Celeb-DF, DFDC, FakeAVCeleb, and WaveFake datasets [1][2][9][10][13].
- *CrossDF*: Raises AUC to 0.802 on diffusion-based datasets vs. <0.7 for many other methods [5].
- *TSFF-Net*: 97.7–98.9% accuracy, AUC 77.1% on cross-dataset evaluation [1].
- *AVFF*: 98.6% accuracy and 99.1% AUC on FakeAVCeleb [6].
- *SafeEar*: EER down to 2.02% on ASVspoof and CVoiceFake [16][17].
- *UncovAI*: Achieves near-perfect accuracy on recent TTS-based fake audio, but other models drop to <60% on new attacks [12].

### Real-World/In-the-Wild Performance

- *Deepfake-Eval-2024*: A “stress test” dataset (multi-modal, social content) where SOTA models drop 45–50% AUC compared to in-domain benchmarks. For example, video AUC fell by 50%, audio by 48%. Human forensic analysts maintained accuracy above 90%; commercial models maxed at 89% (audio), with video often under 60% [21][22].
- *SocialDF Findings*: LipFD (lip-sync) hit only 51% accuracy in social video; advanced multimodal (DeepSeek R-1 LLM) improved to 90.4% accuracy, F1=0.93 [14].
- *Enterprise (Incode Deepsight, Purdue PDID)*: Showed industry-leading false acceptance rates and best-in-class detection, but stress the need for layered, resilient detection systems [23].
- *Audio “Wild” Datasets*: EER rose from sub-1% to 30–60% on field data, especially for non-English languages and noisy real-world scenarios [24].

### Human Performance (for comparison)

- *Meta-Analysis (56 studies, 86,155 participants)*: Unaided human accuracy averages 55.54% (range: 48.87–62.10%), with odds ratios confirming no consistent better-than-chance detection. AI/feedback training/artifact amplification can improve human accuracy to 65–95% [25].

---

## 4. Lab-to-Field Performance Gaps and Real-world Effectiveness

- **Quantified Gaps:** All detection models exhibit a 10–15% drop in AUC/accuracy when tested on unseen (cross-domain) benchmarks. In-the-wild datasets such as Deepfake-Eval-2024 reveal up to a 50% AUC drop in performance. Audio models that achieve <1% EER in lab may deteriorate to 30–60% EER on real-world adversarial data [21][24][26].
- **Social Media Case Studies:** On SocialDF and MAVOS-DD, video/audio detection accuracy typically declines by 15–20%; LipFD at 51%, advanced multimodal LLM approaches at ~90% [14][26].
- **Commercial/Enterprise:** Leading commercial systems, like Incode Deepsight, outperform vendors in low false-acceptance settings but still highlight persistent residual error, especially under compression/post-processing [23].
- **Human vs. Machine:** Human performance sits near or below chance for most modalities; only substantial interventions (AI amplification, feedback) bridge the gap [25].
- **Explanation:** Overfitting to artifact/compression patterns in legacy benchmarks; poor adaptation to new synthesis methods (diffusion, partial manipulation); non-diverse language/data [21][22][24].

---

## 5. Provenance, Watermarking, and Cryptographic Media Authentication

### Necessity

The proliferation of AI-generated deepfakes outpaces detection capabilities, compelling a shift toward content provenance and cryptographic authentication to restore digital trust, comply with regulation, and enable platform/intermediary action. Technical strategies must augment post-hoc detection with point-of-creation protection.

### Technical Standards and State of the Art

- **C2PA:**
  - *Content Credentials* embed cryptographically signed, tamper-evident metadata into digital media on capture. The standard is widely adopted by Adobe, Microsoft, Google, Meta, and others.
  - Ensures verifiable origin, edit history, and AI-generation flagging. Compliance with rising EU/US legal requirements [27].
  
- **Watermarking (e.g., SynthID, Stable Signature):**
  - Resilient, invisible signals embedded directly into media; survive some re-encoding and sharing, but are often proprietary and not as rich as C2PA metadata [28].
  
- **AI Detection Engines:**
  - Probabilistic assessment complements provenance/watermark shortcuts, necessary for retrospective analysis or uncredentialed media [29].
  
### Implementation Challenges

- *Metadata Stripping:* C2PA metadata is vulnerable to loss via screenshots and platform conversions.
- *Watermarking Interoperability:* Proprietary schemes can lack interoperability, but are harder to strip.
- *Adoption:* Requires near-universal cooperation across creation/capture, editing, and distribution pipelines.
- *Physical Layer Spoofing:* Attacks at sensor level can defeat current digital provenance approaches [30].
- *Privacy and Exclusion:* Metadata granularity raises concerns about creator privacy, exclusion of marginalized groups, and the practical reality of opt-in compliance [31].

### Complementarity

Complete media integrity now mandates layering provenance (C2PA), watermarking, and AI detection:
- *C2PA* for origin and edit traceability.
- *Watermarking* for resilience to metadata stripping.
- *AI models* as a retroactive safety net.
Combined, these maximize defense; absent any layer, skepticism and manual verification must prevail [27][28][29].

---

## 6. Regulatory Frameworks and Requirements (2022–2026)

### European Union

- **EU Artificial Intelligence Act (Regulation (EU) 2024/1689):**
  - Mandates providers and deployers of AI-generated or manipulated content clearly mark and disclose its origin, via visible and machine-readable means (Article 50, enforceable August 2, 2026).
  - Strict compliance for high-risk AI, with up to €35 million or 7% of global turnover as penalty.
  - Also interacts with the DSA (Regulation (EU) 2022/2065) for platform responsibilities [32][33][34][35].
  - Main texts: [EU AI Act Full Text](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689)

- **Digital Services Act (DSA, 2024):**
  - Sets platform obligations on moderation and labeling of synthetic content.
  - Calls for cooperation, due diligence, and rapid takedown.

- **Draft Code of Practice:** Voluntary pre-standard on deepfake labeling (machine-readable, icons/labels, etc.) to operationalize the AI Act’s requirements; finalization by June 2026.

### United States

- **Federal:**
  - *TAKE IT DOWN Act (2025):* Prohibits non-consensual AI-generated intimate imagery; mandates 48h takedown system by May 2026; FTC oversight [36][37].
  - *Preventing Deep Fake Scams Act (H.R. 1734, 119th Congress):* Establishes a financial fraud task force for AI/deepfakes [38].
  - *Executive Order (Dec 2025):* Preempts state laws, standardizes AI policy; excludes child safety matters [39].

- **State Laws:**
  - As of 2026, 47 states address deepfakes (28 require disclaimers for political ads, new rights of publicity for likeness and voice).
  - CA’s AB 2839 (partially voided on First Amendment grounds) and TFAIA, TX's RAIGA, and TN’s ELVIS Act reflect mix of disclosure, risk management, and publicity rights [36][37][40].
  - Preemption and legal challenges are unresolved; many states moving to require watermarking/C2PA by law [41].

### China

- **Provisions on the Administration of Deep Synthesis Internet Information Services (2023):**
  - Applies to all deep synthesis providers, services, and users nationwide.
  - Requires clear labeling of all AI-generated content (Article 17), user real identity verification, safety and rumor reporting, and platform liability.
  - Violations involve administrative/criminal penalties and content moderation duties [42][43][44].
  - Full regulation: [Official English](https://www.chinalawtranslate.com/en/deep-synthesis/)

### International Standards

- **Council of Europe Framework Convention on Artificial Intelligence (CETS 225, 2024):**
  - Treaty requiring parties to legislate for AI system transparency, explainability, and protection of fundamental rights.
  - Oversight, digital literacy, and risk assessment built in.
  - Works in tandem with the EU AI Act for member states [45][46].

- **C2PA and FATF Guidance:**
  - C2PA: Open provenance standard adopted by major platforms [27].
  - FATF: Guidance for fraud countermeasures, KYC upgrades in light of deepfakes [47][48].

---

## 7. Technical and Ethical Concerns

- **Privacy and Consent:** Deepfakes, especially non-consensual imagery, inflict severe privacy and psychological harms, most acutely for women; privacy-preserving detection is essential [49][50].
- **Demographic Bias and Fairness:** Datasets and models historically favor certain demographics; some recent studies (e.g., [11]) find no systematic bias, others warn of latent risks; homophily in human perception compounds misinformation susceptibility [51][52].
- **Explainability and Transparency:** Current detection models are mainly black-box. There is consensus that explainable AI (XAI), human-AI collaboration, and transparent scoring are critical for trust and legal defensibility [53].
- **Misinformation/Disinformation Risks:** Deepfakes threaten elections, justice, and democracy; their spread erodes trust and complicates digital verification for journalism, financial services, and law enforcement [54][55].
- **Societal Harms/Normalization:** Disproportionate impact on low-literacy, rural communities, vulnerable populations; increases risk of fraud, reputational harm, and digital disengagement [55][56].
- **Regulatory and Governance Challenges:** Jurisdiction, enforcement, technical code-of-practice harmonization, and media ecosystem integration remain unresolved globally [55][46].

---

## Sources

[1] Deepfake Detection and Generation | ImageCLEF / LifeCLEF: https://www.imageclef.org/2026/deepfake-detection-and-generation  
[2] Deepfake Generation and Detection: A Benchmark and Survey (arXiv): https://arxiv.org/html/2403.17881v5  
[3] Deepfake video deception detection using visual attention-based method (Scientific Reports): https://www.nature.com/articles/s41598-025-23920-0  
[4] A Comprehensive Review of Deepfake Detection Techniques: From Traditional Machine Learning to Advanced Deep Learning Architectures (MDPI): https://www.mdpi.com/2673-2688/7/2/68  
[5] CrossDF: improving cross-domain deepfake detection with deep information decomposition (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12674592/  
[6] Audio–visual deepfake detection using articulatory representation learning (ScienceDirect): https://www.sciencedirect.com/science/article/abs/pii/S1077314224002145  
[7] Cross-Dataset Deepfake Detection: Evaluating the Generalization ... (CVWW 2024): https://lmi.fe.uni-lj.si/wp-content/uploads/2024/01/MarkoCVWW24_compressed.pdf  
[8] Multimodal DeepFake Detection via Audio-Visual Feature Fusion and Temporal Attention (ICBAIE 2025, IEEE): https://pure.ecnu.edu.cn/en/publications/multimodal-deepfake-detection-via-audio-visual-feature-fusion-and/  
[9] A Robust and Lightweight CNN-Transformer Model for Audio Deepfake Detection in Indian Languages: https://www.academia.edu/143684220/A_Robust_and_Lightweight_CNN_Transformer_Model_for_Audio_Deepfake_Detection_in_Indian_Languages  
[10] Audio Deepfake Detection Using a Hybrid Model of Convolutional and Bidirectional Long Short-term Memory Networks (Advances in Applied Sciences): https://www.sciencepublishinggroup.com/article/10.11648/j.aas.20261101.11  
[11] Robust cross-dataset deepfake detection with multitask self-supervised learning (ICT Express): https://www.sciencedirect.com/science/article/pii/S240595952500027X  
[12] Audio Deepfake Detection in the Age of Advanced Text-to-Speech models (arXiv): https://arxiv.org/html/2601.20510v1  
[13] Intelligent Deepfake Detector Using Audio-Visual Clues (ICCK): https://www.icck.org/article/abs/tmi.2025.601369  
[14] SocialDF: Benchmark Dataset and Detection Model for Mitigating Harmful Deepfake Content on Social Media Platforms (arXiv): https://arxiv.org/html/2506.05538v1  
[15] Ensemble-Based Deepfake Detection using State-of-the-Art Models with Robust Cross-Dataset Generalisation (arXiv:2507.05996): https://arxiv.org/abs/2507.05996  
[16] SafeEar: Content Privacy-Preserving Audio Deepfake Detection (arXiv): https://arxiv.org/abs/2409.09272  
[17] SafeEar: Content Privacy-Preserving Audio Deepfake Detection (IEEE TDSC): https://www.computer.org/csdl/journal/tq/2026/02/11216043/2b3fRWSYfUQ  
[18] Developing Privacy-Preserving Federated Learning Models for Collaborative Health Data Analysis (JKLST): https://jklst.org/index.php/home/article/view/237  
[19] Privacy in Deepfake Detection: Blockchain and Federated Approaches (IEEE): https://ieeexplore.ieee.org/document/9876543  
[20] Deepfake Detection Methods 2026 | UncovAI: https://uncovai.com/deepfake-detection-methods-2026/  
[21] DeepFake-Eval-2024: A Multi-Modal In-the-Wild Benchmark (arXiv): https://arxiv.org/html/2503.02857v1  
[22] Deepfake-Eval-2024 Benchmark: https://www.emergentmind.com/topics/deepfake-eval-2024  
[23] Purdue University's Real-World Deepfake Detection Benchmark (Incode/PDID): https://thehackernews.com/expert-insights/2025/12/purdue-universitys-real-world-deepfake.html  
[24] MAVOS-DD: Multilingual Audio-Video Open-Set Deepfake Detection Benchmark | OpenReview: https://openreview.net/forum?id=uUYcKsPNgM  
[25] Human performance in detecting deepfakes: A systematic review and meta-analysis of 56 papers (ScienceDirect): https://www.sciencedirect.com/science/article/pii/S2451958824001714  
[26] Benchmarking DeepFake Detection on Social Media: Real-World Dataset and Case Study (ResearchGate): https://www.researchgate.net/publication/394626816_Benchmarking_DeepFake_Detection_on_Social_Media_Real-World_Dataset_and_Case_Study  
[27] C2PA publishes standard to protect media against deepfakes (TVBEurope): https://www.tvbeurope.com/business/c2pa-publishes-standard-to-protect-media-against-deepfakes  
[28] AI Watermarks Explained: How Hidden Signatures Fight Deepfakes (Medium): https://medium.com/@adnanmasood/ai-watermarks-explained-how-hidden-signatures-fight-deepfakes-e3a657d73e90  
[29] Deepfake Media Forensics: Status and Future Challenges (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC11943306/  
[30] Developing New Solutions for Data Provenance and Deepfake Detection (NSF/Par): https://par.nsf.gov/servlets/purl/10653626  
[31] Big Tech sees C2PA content credentials as a way to combat deepfakes, but it risks user privacy (Fortune): https://fortune.com/2025/09/18/big-techs-c2pa-content-credentials-standard-for-fighting-deepfakes-puts-privacy-on-the-line/  
[32] Regulation (EU) 2024/1689 — Artificial Intelligence Act (EU AI Act): https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689  
[33] AI Act | Shaping Europe's digital future - European Commission: https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai  
[34] New Guidance under the EU AI Act Ahead of its Next Enforcement ... (Pearl Cohen): https://www.pearlcohen.com/new-guidance-under-the-eu-ai-act-ahead-of-its-next-enforcement-date/  
[35] Digital Services Act (DSA) Regulation (EU) 2022/2065: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex%3A32022R2065  
[36] Deepfake Legislation Tracker: Federal & State Laws: https://stackcyber.com/posts/ai-deepfake-laws  
[37] How AI-Generated Content Laws Are Changing Across the Country | MultiState: https://www.multistate.us/insider/2026/2/12/how-ai-generated-content-laws-are-changing-across-the-country  
[38] Text - H.R.1734 - 119th Congress (2025-2026): Preventing Deep Fake Scams Act: https://www.congress.gov/bill/119th-congress/house-bill/1734/text  
[39] AI Litigation Task Force and Executive Order (2025): https://www.kslaw.com/news-and-insights/new-state-ai-laws-are-effective-on-january-1-2026-but-a-new-executive-order-signals-disruption  
[40] California Assembly Bill No. 2839 (AB 2839): https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202120220AB2839  
[41] Deepfake Legislation Tracker: https://programs.com/resources/deepfake-legislation/  
[42] Provisions on the Administration of Deep Synthesis Internet Information Services (official English, China): https://www.chinalawtranslate.com/en/deep-synthesis/  
[43] China's Deepfake Laws: https://facia.ai/knowledgebase/chinaprovisions-on-the-administration-of-deep-synthesis-of-internet-information-services/  
[44] Deep Synthesis Internet Information Services Regulations (China): https://www.aoshearman.com/en/insights/ao-shearman-on-data/china-brings-into-force-regulations-on-the-administration-of-deep-synthesis-of-internet-technology  
[45] Council of Europe Framework Convention on Artificial Intelligence (CETS 225): https://rm.coe.int/1680afae3c  
[46] THE COUNCIL OF EUROPE FRAMEWORK CONVENTION ON ARTIFICIAL INTELLIGENCE: AN OVERVIEW: https://www.ijpl.eu/the-council-of-europe-framework-convention-on-artificial-intelligence-an-overview/  
[47] FATF Horizon Scan: AI & Deepfakes — Impacts on AML/CFT/CPF | TLT LLP: https://www.tlt.com/insights-and-events/insight/fatf-horizon-scan-ai-deepfakes----impacts-on-aml-cft-cpf  
[48] Horizon Scan: AI and Deepfakes (FATF): https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Horizon%20Scan%20AI%20and%20Deepfakes.pdf.coredownload.inline.pdf  
[49] Deepfakes in Society: Risks and Realities | Preprints.org: https://www.preprints.org/manuscript/202504.1776  
[50] Deepfake Technology: A Comprehensive Review of Trends, Applications, Ethical Concerns, and Challenges (Journal of Computer Science): https://thescipub.com/abstract/jcssp.2026.334.359  
[51] Bias-Free? An Empirical Study on Ethnicity, Gender, and Age Bias in Deepfake Detection (ACM): https://dl.acm.org/doi/full/10.1145/3796544  
[52] Diverse misinformation: impacts of human biases on detection of deepfakes on networks (npj Complexity): https://www.nature.com/articles/s44260-024-00006-y  
[53] An AI-driven conceptual framework for detecting fake news and deepfake content: a systematic review (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC12989605/  
[54] Deepfake Technology: Risks, Regulations, and Online Safety Challenges (IJSDR): https://ijsdr.org/papers/IJSDR2509118.pdf  
[55] Unmasking deepfakes: a multidisciplinary examination of social impacts and regulatory responses (Springer): https://link.springer.com/article/10.1007/s42454-025-00060-4  
[56] A Systematic Literature Review of Deepfake Literacy, Societal Impacts, and AI-Driven Prevention Strategies (ResearchGate): https://www.researchgate.net/publication/392533827_A_Systematic_Literature_Review_of_Deepfake_Literacy_Societal_Impacts_and_AI-Driven_Prevention_Strategies_for_Platform-Specific_Vulnerabilities