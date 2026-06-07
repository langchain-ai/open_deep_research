# AI-Powered Code Completion Interfaces for Enterprise Software Teams: A Comprehensive Research Synthesis

## Executive Summary

This report synthesizes findings from foundational HCI timing heuristics, academic research on interruption costs and developer cognition, vendor-specific latency architectures, empirical studies of developer experience levels, trust calibration research, and proactive vs. on-demand suggestion timing studies. The analysis provides actionable, granular design targets for enterprise IDE implementation, anchored in the three classic HCI response time thresholds—0.1 seconds (100ms) for perceived instantaneous response, 1.0 seconds (1000ms) for maintaining user flow without interruption, and 10 seconds for keeping user attention on a single task.

Key findings include: GitHub Copilot's ~200ms average response time (reduced to ~130ms with 2025 model) places it near the instantaneous regime for single-line autocomplete; Tabnine's local inference achieves 15-80ms TTFT in the true instantaneous regime; Amazon Q Developer's latency-optimized inference via Bedrock achieves 42-51% reductions in time-to-first-token; JetBrains FLCC offers 50-75ms local inference while Mellum cloud operates at 1-2 seconds. Interruption research reveals that even 250ms delays impose measurable resumption costs of 974ms, and full context recovery after interruptions requires 10-45 minutes. The experience paradox shows junior developers exhibit 78% trust in AI specificity versus 39% for seniors, yet seniors ship 2.5 times more AI-generated code to production. Systematic timing research demonstrates 52% engagement for proactive suggestions at workflow boundaries versus 62% dismissal mid-task. Trust calibration research shows confidence scores help calibration while explanations can hurt it, especially for novices.

---

## Section 1: Granular Latency Budgets per Suggestion Modality

### 1.1 Foundational HCI Response Time Thresholds

The three classic HCI response time thresholds—0.1 seconds (100ms), 1.0 seconds (1000ms), and 10 seconds—represent foundational principles that have remained remarkably stable over five decades of human-computer interaction research. As Jakob Nielsen explains: "Why have response time guidelines been the same for 55 years? Because they are derived from neuropsychology: the speed at which signals travel in the human body and how the human brain has evolved to cope with these biologically determined speed limits." [1]

**The Model Human Processor (Card, Moran, & Newell, 1983):** The MHP describes human information processing through three subsystems: the Perceptual Processor with a cycle time of approximately 100ms, the Cognitive Processor with a cycle time of approximately 70ms, and the Motor Processor with a cycle time of approximately 70ms. [2] These cycle times explain why the 100ms threshold is so fundamental: it corresponds to the time constant of the perceptual processor. Any system response faster than this cycle time will be perceived as occurring within the same perceptual "frame," creating the illusion of direct causality.

**Miller's Three Thresholds (1968):** Robert B. Miller's seminal paper identified that "a response time of one tenth of a second is viewed as instantaneous" for keypress confirmation, while "a response within 1 second is fast enough for users to feel they are interacting freely" and "about the limit for the user's flow of thought to stay uninterrupted." Miller also identified that "response delays of more than two seconds should follow only a condition of task closure as perceived by the human," and "response times must stay below 10 seconds to keep the user's attention focused on the dialogue." [3]

**Nielsen's Three Limits (1993):** Nielsen's "Usability Engineering" defined three key time thresholds: 0.1 second for the user to feel the system is reacting instantaneously, 1.0 second for the user's flow of thought to stay uninterrupted, and 10 seconds for keeping the user's attention focused on the dialogue. [4]

**Shneiderman's Response Time Research (1984):** Shneiderman found that "frequent users prefer response times of less than a second for most tasks, and productivity does increase as response time decreases; however, error rates increase with too short or too long a response time." He also noted that "users working with a new system first appreciate short response times, but can adapt their pace and strategies when delays increase." [5]

### 1.2 GitHub Copilot Latency Measurements

**Sub-200ms Engineering Target:** GitHub Copilot serves hundreds of millions of requests daily with an average response time of under 200 milliseconds for inline single-line autocomplete. David Cheney, Lead of the Copilot Proxy team at GitHub, revealed the key engineering decisions enabling this latency: HTTP/2 with long-lived multiplexed connections enabling efficient cancellation of individual streams, global deployment across multiple Azure regions with smart routing via octoDNS, and a proxy layer (copilot-proxy) enabling dynamic request mutation, traffic splitting, and immediate fixes. [6][7]

**2025 Custom Model - 35% Latency Reduction:** GitHub's new custom model delivers "20% more accepted and retained characters, 12% higher acceptance rate, 3x higher token-per-second throughput, and a 35% reduction in latency." [8] Assuming the prior baseline was approximately 200ms, a 35% reduction brings the average to approximately 130ms for inline single-line completions.

**March 2026 Agent Update - 50% Reduction:** A March 19, 2026 update introduced a "50% reduction in median completion latency, lowering it from 1.4 seconds to 0.7 seconds" for Copilot's agent/coding agent completion mode. This was achieved through speculative decoding and model distillation techniques, where a smaller draft model predicts multiple tokens in parallel that the larger model validates in batch. [9]

**GPT-4o Copilot Model:** Microsoft announced the public preview release of GPT-4o Copilot in Visual Studio 17.14 Preview 2 on March 20, 2025. This model, based on GPT-4o mini, was additionally trained on over 275,000 high-quality public repositories spanning more than 30 popular programming languages. [10]

### 1.3 Tabnine Latency Measurements

**Hybrid Architecture:** Tabnine introduced its Hybrid model on June 1, 2022, combining cloud-based and local inference. The system uses a "Team of Models" approach: lightweight local models operate offline on the user's machine for instantaneous suggestions, while larger cloud models handle more accurate and longer predictions. [11]

| Component | Latency | HCI Regime |
|-----------|---------|------------|
| **Local inference** | 15-80ms TTFT | **Instantaneous** (<100ms) |
| **Cloud inference** | 180-600ms TTFT | **Flow-preserving** (100ms-1s) |
| **CatBoost filter model** | 1-2ms | Negligible |

**Source:** [12][13]

**Local Inference (15-80ms TTFT):** The SitePoint 2026 analysis confirms: "Time to first token is the single most consequential metric for coding assistant responsiveness. It determines whether a completion feels instantaneous or introduces a perceptible pause in the developer's flow." Local TTFT of 15-80ms means local completions feel "instantaneous; no perceptible pause." For typical coding autocomplete tasks producing 20-80 tokens, local inference is 4x to 13x faster than cloud. [12]

**Cloud Inference (180-600ms TTFT):** Cloud APIs have TTFT of 180-600ms, dominated by network overhead. [12] Cloud providers reach higher sustained throughput (80-150 tokens/second) for longer outputs (200-300+ tokens), making cloud preferable for longer completions.

**CatBoost Filter (1-2ms):** Tabnine uses a CatBoost-based filtering system to rank and filter completion candidates. This lightweight filter adds minimal overhead to the overall inference pipeline, operating well below the threshold of human perception. [13]

**Enterprise Adoption Metrics:** Tabnine reports a 90% acceptance rate for single-line coding suggestions and an 11% productivity increase across projects. [13]

### 1.4 Amazon Q Developer / CodeWhisperer Latency Measurements

**Inline Completion Latency:** Amazon Q Developer's single-line completion feature provides inline code suggestions automatically enabled when installing the Amazon Q extension in IDEs. Industry comparisons suggest Amazon Q Developer achieves approximately 200ms latency for inline code completions, with top-1 correctness of approximately 78%. [14]

**Amazon Bedrock Latency-Optimized Inference:** At AWS re:Invent 2024, Amazon announced latency-optimized inference for foundation models in Amazon Bedrock. The AWS blog provides benchmarking data showing up to 51.65% reduction in TTFT for optimized models. [15]

| Model | Metric | Standard | Optimized | Improvement |
|-------|--------|----------|-----------|-------------|
| Claude 3.5 Haiku | TTFT P50 | 1.1 seconds | 0.6 seconds | **-42.20%** |
| Claude 3.5 Haiku | TTFT P90 | 2.9 seconds | 1.4 seconds | **-51.70%** |
| Claude 3.5 Haiku | OTPS P50 | 48.4 | 85.9 | **+77.34%** |
| Llama 3.1 70B | TTFT P50 | 0.9 seconds | 0.4 seconds | **-51.65%** |
| Llama 3.1 70B | TTFT P90 | 42.8 seconds | 1.2 seconds | **-97.10%** |
| Llama 3.1 70B | OTPS P50 | 30.2 | 137.0 | **+353.84%** |

**Source:** [15][16]

**Mapping to HCI Thresholds:** Standard Claude 3.5 Haiku (1.1s TTFT P50) falls above the 1.0-second threshold for flow preservation. The 2.9s P90 extends into the "user perceives themselves at the mercy of the computer" regime. Optimized Claude 3.5 Haiku (0.6s TTFT P50) falls within the flow-preserving regime (100ms-1s). Standard Llama 3.1 70B (0.9s TTFT P50) is borderline at the 1.0-second threshold. The 42.8s P90 is catastrophic, extending well beyond the 10-second attention limit. Optimized Llama 3.1 70B (0.4s TTFT P50) falls within the flow-preserving regime with a much more acceptable 1.2s P90.

**The P90 Variance Problem:** The key insight from Amazon's data is the P90 variance problem. Standard Llama 3.1 70B has a P90 of 42.8 seconds, risking complete abandonment of the task. For enterprise design, latency variance matters as much as average latency because unpredictable delays are more disruptive to flow than predictable ones.

### 1.5 JetBrains FLCC (Full Line Code Completion) Latency

**Local Inference Latency: 50-75ms:** JetBrains FLCC is a multi-token code completion feature integrated into the IntelliJ Platform, operating fully locally on the user's machine. The FLCC model is a 100 million-parameter LLaMA-like Transformer optimized via byte-pair encoding and INT4 quantization, maintaining a small memory footprint (~100 MB). [17]

The arXiv paper states: "We formulated and fulfilled the following restrictions and design principles: local-based operation, feasible memory footprint, speed for seamless typing, and proper integration into existing code completion pipelines." The reported local inference latency is between 50-75ms for single-line completions, leveraging a local C++ inference server connected via gRPC. [17]

INT4 quantization with the llama.cpp inference engine "reduced model size from almost 400 MB to slightly over 100 MB and speedup inference by 1.4 to 2.4 times depending on hardware." [17]

Offline and robust online A/B evaluations demonstrate that FLCC "increases Python code completion usage by 1.3 times without disrupting existing developer workflows" and "1.5 times the ratio of code completed by the AI versus standard single-token completion." [17]

### 1.6 JetBrains Mellum (Cloud Completion) Latency

**Cloud Inference Latency: 1-2 seconds:** Mellum is a 4-billion-parameter model based on a Llama-style architecture, trained on 4 trillion tokens of permissively licensed, multi-language code. Training includes a three-stage process: pre-training, context-aware fine-tuning, and reinforcement learning with AI feedback (RLAIF) using direct preference optimization (DPO). [18]

The Mellum arXiv paper states Mellum models are "designed specifically for in-editor completion rather than general dialog, which implies their industrial constraints: low latency for real-time suggestions, reasonable model size, and leveraging widely adopted architectures." [19]

**On-Premises Performance Specifications:** JetBrains IDE Services Documentation provides specific performance metrics: a single Nvidia L4 GPU supports 1 request per second for testing, while Nvidia H100/H200 GPUs can handle up to 10-11 requests per second with "p90 latency under 750ms" for 750-1500 seats. [20]

Mellum achieves a Ratio of Completed Code (RoCC) of up to 46% for Java and acceptance rates (AR) around 30-44% for various languages. [18][19]

### 1.7 JetBrains NES (Next Edit Suggestions) Latency

**Latency: <200ms:** JetBrains announced that Next Edit Suggestions (NES) are now generally available. NES provides AI-powered suggestions that can modify existing code or suggest additions beyond the immediate cursor area. The blog post states: "NES relies on a custom cloud-based small language model fine-tuned for this task, providing latency under 200 ms for most requests." [21]

NES operates silently in the background, offering in-editor diffs or popup suggestions depending on the change size. It uses recent code change history rather than just current file context, and integrates with IDE actions like Rename refactoring, updating usages across multiple files seamlessly. NES is enabled by default for JetBrains AI Pro, AI Ultimate, and AI Enterprise subscribers and does not consume AI quota. [21]

### 1.8 Cross-Vendor Latency Comparison Mapped to HCI Thresholds

| Vendor | Feature | Latency | HCI Regime | User Experience |
|--------|---------|---------|------------|-----------------|
| **Copilot** | Single-line autocomplete | ~200ms average | **Flow-preserving** (100ms-1s) | Noticeable but seamless |
| **Copilot (2025 model)** | Improved completions | ~130ms (35% reduction) | **Flow-preserving** | Near-instant |
| **Copilot (March 2026)** | Agent completion mode | 0.7s median (50% reduction) | **Flow-preserving** | Noticeable but acceptable |
| **Tabnine** | Local inference | 15-80ms TTFT | **Instantaneous** (<100ms) | Feels like own typing |
| **Tabnine** | Cloud inference | 180-600ms TTFT | **Flow-preserving** | Noticeable but acceptable |
| **Tabnine** | CatBoost filter | 1-2ms | Negligible | Below perception |
| **Amazon Q** | Inline completion | ~200ms | **Flow-preserving** | Noticeable but seamless |
| **Amazon Q (Bedrock optimized)** | Claude 3.5 Haiku | 600ms TTFT P50 | **Flow-preserving** | Noticeable |
| **Amazon Q (Bedrock optimized)** | Claude 3.5 Haiku | 1.4s TTFT P90 | **Flow-disruptive** (>1s) | Requires feedback |
| **Amazon Q (Bedrock optimized)** | Llama 3.1 70B | 400ms TTFT P50 | **Flow-preserving** | Noticeable |
| **Amazon Q (standard)** | Llama 3.1 70B | 42.8s TTFT P90 | **Attention-breaking** (>10s) | Potential abandonment |
| **JetBrains FLCC** | Local inference | 50-75ms | **Instantaneous** (<100ms) | Feels like own typing |
| **JetBrains Mellum** | Cloud completion | 1-2 seconds | **Flow-disruptive** (>1s) | Requires feedback indicator |
| **JetBrains Mellum (optimized)** | H100/H200 GPU | p90 <750ms (10-11 req/s) | **Flow-preserving** | Acceptable for scale |
| **JetBrains NES** | Next edit suggestions | <200ms | **Flow-preserving** | Seamless |

**Sources:** [6][7][8][9][11][12][13][14][15][17][18][19][20][21]

### 1.9 Implications by Suggestion Type

**Single-Line Autocomplete (Requires <200ms for Flow):** Tools operating at <100ms (Tabnine local, JetBrains FLCC) provide the best user experience, feeling like native IDE features rather than AI interventions. Tools at 100-200ms (Copilot, Amazon Q) still preserve flow but may be barely noticeable. Tools above 200ms begin to create perceptible pauses that can disrupt typing rhythm.

**Multi-Line Predictions (200-500ms Acceptable):** Users expect slightly longer response times for multi-line suggestions since these involve more complex generation. JetBrains Mellum cloud at 1-2 seconds exceeds the acceptable range, while Copilot Agent at 0.7s median is at the upper boundary. JetBrains NES at <200ms is well within range.

**Comment-to-Code Generation (1-3s Acceptable):** Since the user explicitly initiates this action by writing a comment, they have a task-related expectation of waiting. Delays of 1-3s feel reasonable, matching Miller's concept of "task closure"—the user completes writing a comment, perceives closure, and tolerates a longer wait for the result. [3]

**Explanation Availability (500ms-2s Acceptable):** Explanation requests are user-initiated and task-switching events. The user has paused their typing to seek understanding. Delays of 500ms-2s are appropriate, with the system aiming for <1s to feel responsive.

---

## Section 2: Task-Specific Workflow Metrics

### 2.1 Edit-Run Cycle Times by Task Type

**GitHub Copilot (ZoomInfo Enterprise Study):** The ZoomInfo engineering blog published a comprehensive evaluation of GitHub Copilot across 400+ developers. Key findings include an average acceptance rate of 33% for suggestions and 20% for lines of code, developer satisfaction reaching 72%, and developers reporting approximately 20% time savings, particularly benefiting boilerplate code and unit test generation. 90% of respondents stated that GitHub Copilot reduces the amount of time to complete their tasks. [22]

**DX Research (Cross-Company Aggregate Data):** Research across 38,880 developers at 184 companies shows mature rollouts achieve 40-50% daily AI tool usage and average time savings of 3 hours and 45 minutes per week. Real-world productivity boosts of 5-15% are observed, not the 50-100% claimed in vendor headlines. Heavy users (daily) of AI tools have nearly 5x more pull requests per week than non-users. [23]

**JetBrains AI Assistant Survey:** JetBrains surveyed 640 users, primarily developers from the US and Germany. 75% of users are satisfied with the AI Assistant, 91% reported time savings, with 37% saving 1-3 hours weekly and 22% saving 3-5 hours weekly. Less experienced developers (under 2 years) benefit the most, saving 3-5 hours weekly. 78% spend less time searching, 71% complete tasks faster. Top features: AI Chat, refactoring suggestions, find problems, explain code, generate documentation. [24]

**METR 2025 Study (The AI Productivity Paradox):** A 2025 study by METR found that experienced open-source developers were 19% slower when using AI coding assistants (like Copilot, Cursor, and ChatGPT) yet believed they were 24% faster—a 43 percentage point gap. Developers spend 45% of their time debugging AI-generated code. Review bottlenecks emerge as the primary constraint: 98% more PRs with 154% larger size and 91% longer review times. [25]

**State of Engineering Excellence 2026 Report:** A survey of 700 engineers and leaders found that 31% of a developer's day now goes to AI-related work that no metric tracks. 53% of that time is reviewing AI code for accuracy, 52% fixing subtle bugs in AI output, 48% explaining AI code to teammates, and 45% context switching between tools. The "AI velocity paradox" emerged: coding got faster, everything after coding got heavier. 89% of engineering leaders say their coding productivity metrics have improved with AI, but 81% say after-code time has gone up. [26]

### 2.2 Acceptance Rates Stratified by Task Type

**Primary Academic Source: Task-Stratified Analysis (arXiv 2602.08915):** This is the most directly relevant study, presenting an empirical analysis comparing five AI coding agents—OpenAI Codex, GitHub Copilot, Devin, Cursor, and Claude Code—using 7,156 pull requests from the AIDev dataset. [27]

| Task Type | Overall Acceptance | Highest Tool |
|-----------|-------------------|--------------|
| Documentation | 82.1% | Claude Code (92.3%) |
| Feature additions | 66.1% | Claude Code (72.6%) |
| Fix tasks (bug fixes) | Varies by tool | Cursor (80.4%) |
| All categories | 59.6-88.6% | OpenAI Codex |

**Source:** [27]

The study found that "PR task type is a dominant factor influencing acceptance rates: documentation tasks achieve 82.1% acceptance compared to 66.1% for new features—a 16 percentage point gap that exceeds typical inter-agent variance for most tasks." This means task type has a stronger influence on acceptance rates than which AI tool is being used.

**Implication for Latency Design:** Documentation and boilerplate tasks have high acceptance rates and tolerate moderate latency because the cognitive cost of evaluating the suggestion is low. Complex bug fixes and architectural decisions have lower acceptance rates and are more sensitive to latency because the evaluation cost is high.

**ZoomInfo Enterprise Copilot Data:** The ZoomInfo study (400+ developers) found an average acceptance rate of 33% for suggestions and 20% for lines of code. Acceptance rates for TypeScript, Java, Python, and JavaScript were sustained at about 30%, while HTML, CSS, JSON, and SQL had lower rates. [22]

**Anthropic Study on AI Interaction Patterns:** A study with 52 mostly junior engineers found that AI users showed negligible speed improvement but significantly lower comprehension scores (50% with AI vs. 67% without). However, six distinct AI interaction patterns were identified, with comprehension scores ranging from 24% to 86%. The key takeaway: "How you use AI matters more than whether you use it." [28]

### 2.3 Interruption Resumption Costs

**Monk et al. (2004) - Very Brief Interruptions:** The original CogSci 2004 paper by Monk, Boehm-Davis, and Trafton investigated the resumption cost associated with very brief interruptions during task performance. The experiment involved twelve undergraduate participants performing a simulated VCR programming task interrupted at intervals of 1/4 second, 1 second, and 5 seconds. [29]

Key findings:
- Even very brief interruptions (1/4 second) caused significant resumption costs: mean **974 ms** for the 250ms condition
- The 5-second interruption showed the longest lag: mean **1115 ms**
- The uninterrupted condition baseline was **706 ms**
- "Even for the briefest interruptions, there is a penalty to be paid when resuming the primary task"

**Monk et al. (2008) - Interruption Duration and Demand:** A follow-up paper extended these findings. Resumption times increase linearly between 3 and 13 seconds of interruption. For durations up to almost a minute, resumption times follow a logarithmic pattern, rising rapidly at shorter durations then asymptoting between 13 and 23 seconds. Higher interruption task demand inhibits goal rehearsal, leading to steeper decay and longer resumption lags. [30]

**Parnin & Rugaber (2009/2011) - Interruption Costs in Programming:** Chris Parnin and Spencer Rugaber analyzed over 10,000 recorded programming sessions from 86 developers. Key quantitative findings: [31][32]

- "Only 10% of the sessions have programming activity resume in less than 1 minute after an interruption"
- "Only 7% involve no navigation prior to editing"
- 83% of sessions navigated to new locations before editing
- Only 7.5% of sessions resumed editing in the last method edited without navigation
- Typically, programmers navigate to 2 to 12 code locations before editing
- "Interruptions often cost developers more than 15 minutes to regain focus"
- Tasks resumed after interruption take twice as long

The study identified suspension strategies (rehearsal, serialization, cue priming) and resumption strategies (global restoration, goal restoration, plan restoration, context restoration).

**Mark et al. (2008) - The Cost of Interrupted Work:** Gloria Mark and colleagues conducted a controlled experiment simulating an office email task. Key findings: [33]

- Interruptions led participants to complete tasks faster but with increased stress, frustration, time pressure, and effort
- "After only 20 minutes of interrupted performance people reported significantly higher stress, frustration, workload, effort, and pressure"
- Office workers are interrupted roughly every three minutes
- It takes an average of **23 minutes and 15 seconds** to return to the original task after an interruption
- About 82% of interrupted work is resumed on the same day

**Context Recovery Time for Developers:** Multiple sources converge on the finding that full context recovery for developers takes 30-45 minutes after interruption. The StackBlitz article on flow state notes: "It takes an average of 23 minutes to fully regain focused attention after any interruption." But for complex programming work, the full context recovery is longer: "After an interruption, it takes a programmer 30-45 minutes to rebuild the full mental context of their work." [34]

### 2.4 Csikszentmihalyi's Flow State Research

Mihaly Csikszentmihalyi's foundational work "Flow: The Psychology of Optimal Experience" (1990) defined flow as a state of complete absorption in challenging, creative activities. Characteristics include intense focus, merging of action and awareness, clear goals, immediate feedback, balance between challenge and skills, exclusion of distractions, loss of self-consciousness, and distortion of time perception. [35]

Applied to programming: Flow state can boost productivity by up to 500%, but requires 15-25 minutes of uninterrupted focus to enter. The median developer spends only 52 minutes per day writing or editing code. Deep work is also an activity that generates a sense of meaning and fulfillment in professional life.

### 2.5 How Latency During Peak Cognitive Load Multiplies Disruption Costs

**The Multiplier Effect of High-Cognitive-Load Tasks:** Research consistently shows that interruptions during high-cognitive-load tasks (debugging, complex reasoning) are significantly more disruptive than during low-cognitive-load tasks (simple edits, documentation).

**Monk et al. (2008) on Demand and Duration:** Experiment 3 of the Monk et al. 2008 study directly tested this by incorporating task demand levels—no-task (free rehearsal), medium-demand tracking, and high-demand verbal n-back task. Higher interruption task demand inhibits goal rehearsal, leading to steeper decay and longer resumption lags. [30] During cognitively demanding primary tasks, interruptions cause disproportionate disruption because the working memory resources needed for rehearsal are already fully occupied.

**NSF Interruption and Recovery Study (2023):** The NSF-funded study on interruption recovery found that "algorithmic tasks rely on maintaining complex logical relationships in working memory, which are disrupted by an interruption." Task type directly influenced recovery success: algorithmic tasks posed the greatest challenge, followed by spatial and user interface tasks. [36]

**The Disruption Multiplier: Debugging vs. Documentation:**

| Task Type | Intrinsic Cognitive Load | Interruption Cost | Recommended AI Behavior |
|-----------|------------------------|-------------------|----------------------|
| **Documentation tasks** | Low | Low (974ms resumption lag) | Proactive suggestions acceptable; high acceptance (82.1%) |
| **New feature development** | Medium | Moderate | Proactive at boundaries, on-demand during flow |
| **Debugging/fix tasks** | High | High (15+ minutes recovery) | On-demand only; interruptions double task time |
| **Complex architecture decisions** | Very high | Very high (30-45 min recovery) | On-demand only; AI can hinder experienced developers |

**The Perception Gap Magnifies the Cost:** The 43 percentage point perception gap—developers feel 24% faster but are actually 19% slower—is most pronounced during complex tasks. Junior developers (<2 years) gain 39% speed on simple tasks but this advantage disappears or reverses on complex, architectural work. [25]

### 2.6 The "Productive Struggle" Phase

The clearing-ai.com article explains a critical mechanism: "AI tools undermine flow by instantaneously resolving uncertainties, causing users to avoid this necessary cognitive effort"—the "productive struggle" phase. Key to entering flow is this initial discomfort that primes deep engagement. "AI tools make this phase feel unnecessary, leading engineers to exit the productive struggle phase prematurely." [37]

Each AI suggestion demands a decision: accept, reject, or modify, creating a cognitive "orchestration overhead" that shifts work from implementation to supervision. This is particularly problematic during high-cognitive-load tasks because the developer's working memory is saturated. Proactive suggestions during debugging create additional extraneous load on top of the already high intrinsic load.

---

## Section 3: Cohort-Specific Quantitative Acceptance Rates by Experience Level

### 3.1 The Peng/Demirer MIT Study (2023)

**Original Publication:** "The Impact of AI on Developer Productivity: Evidence from GitHub Copilot" (arXiv 2302.06590) by Sida Peng (Microsoft Research), Mert Demirer (MIT Sloan), and co-authors from Princeton, MIT, University of Pennsylvania, and Microsoft. [38][39]

**Methodology:** Three large-scale randomized controlled trials (RCTs) conducted at Microsoft, Accenture, and an anonymous Fortune 100 electronics manufacturing company between 2022 and 2024. Combined analysis across 4,867 developers who were randomly given access to GitHub Copilot.

**Exact Productivity Gain Percentages:**

| Finding | Junior/Less Experienced | Senior/More Experienced |
|---------|------------------------|------------------------|
| Productivity gain (pull requests) | **27% to 40%** | **7% to 16%** |
| Secondary: commits | +13.55% increase | Smaller increase |
| Secondary: builds | +38.38% increase | Smaller increase |

**Source:** [38][39]

The MIT Sloan article states: "The most significant gains (27% to 39%) observed among newer hires and less experienced developers." The IT Revolution article states: "Junior-level developers saw productivity boosts of 21% to 40%." Senior developers experienced smaller gains of 8% to 13% (MIT Sloan version) or 7% to 16% (IT Revolution version). [40]

**Adoption Rates:** Adoption rates plateaued at 60-75% across all three experiments despite low barriers to access. "30-40% of engineers never even tried the product." Younger developers adopted the tool more readily than seniors. [38]

**Code Quality Findings:** At Microsoft, pull request approval rates increased by approximately 10% after Copilot adoption, suggesting maintained or improved code quality. The study states: "We do not find any evidence that the quality of code at Microsoft decreases after the adoption of GitHub Copilot." However, at Accenture, there was a decrease in build success rate. [38]

The study concludes: "We find that generative AI yields greater productivity gains for lower-ability workers, even when performing tasks by tenure or seniority."

### 3.2 Jellyfish Copilot Dashboard Data (2025)

**Original Source:** Jellyfish's engineering efficiency analysis of 146,000 Jira tickets from 6,500 engineers using GitHub Copilot. [41]

**Key Findings:**

- **Senior engineers: 22% faster coding** with Copilot
- **Junior engineers: only 4% faster** with Copilot
- This is a **5.5x difference** in speed gains
- After adopting Copilot, engineers resolved 15% more Jira tickets per week (from 1.95 to 2.3 tickets per week)
- Power users (using Copilot 80%+ of the time) saw nearly 0.5 more resolved tickets per week versus lower adoption users who saw about 0.25 more

**Source:** [41][42]

**Discrepancy Note:** The Jellyfish finding that seniors gain more (22% faster) than juniors (4% faster) contradicts the MIT finding that juniors gain more. The discrepancy may be due to different metrics: MIT measured task completion (pull requests), while Jellyfish measured code writing speed. Seniors' greater ability to write effective prompts and evaluate AI output likely explains their larger speed gains in code writing speed specifically.

### 3.3 Fastly 2025 Developer Survey

**Original Source:** A July 2025 survey by Fastly involving 791 professional developers examining usage of AI coding assistants. [43]

**Key Cohort-Specific Quantitative Findings:**

| Metric | Senior (10+ years) | Junior (0-2 years) |
|--------|-------------------|-------------------|
| Over half of shipped code is AI-generated | **32%** | **13%** |
| AI tools help ship faster overall | **59%** | **49%** |
| AI makes them "a lot faster" | **26%** | ~13% |
| Fixing AI code offsets time savings | **30%** | **17%** |

**Source:** [43][44]

Key quote from Austin Spires, Fastly senior director of developer engagement: "Senior developers probably have a better understanding of how to build requirements, how to write prompts, how to work with those tools. Junior developers, though, I think it's actually a good thing that they're writing a little bit less AI code. You can't outsource taste to an AI. You need to have good professional sensibilities to know what good looks like from the AI." [43]

### 3.4 Barke et al. (2023): Acceleration vs. Exploration Modes

**Original Publication:** "Grounded Copilot: How Programmers Interact with Code-Generating Models" (OOPSLA 2023 Distinguished Paper) by Shraddha Barke, Michael B. James, and Nadia Polikarpova (UC San Diego). [45]

**Methodology:** First grounded theory analysis of how programmers interact with AI code-generating assistants. Based on observing 20 participants with varying experience levels across four programming languages (Python, Rust, Haskell, Java).

**Core Finding: Bimodal Interaction Model**

**Acceleration Mode:**
- The programmer knows what to do next and uses Copilot to get there faster
- Characterized by quick acceptance of short, precise suggestions without interrupting flow
- Participant P13: "I think of Copilot as an intelligent autocomplete... I already have the line of code in mind and I just want to see if it can do it, type it out faster than I can."
- **Critical finding:** Longer multi-line suggestions in acceleration mode tend to distract programmers and break their flow
- Typically used by experienced developers with clear mental models

**Exploration Mode:**
- The programmer is unsure how to proceed and uses Copilot to explore their options
- Characterized by deliberate prompting, use of multi-suggestion panes, and willingness to consider multiple alternatives
- Participant: "Copilot feels useful for doing novel tasks that I don't necessarily know how to do. It is easier to jump in and get started with the task."
- Typically more common for junior developers or unfamiliar tasks

**Design Recommendation:** Tailor AI assistant behavior to user mode: provide concise suggestions during acceleration, and richer affordances for comparison and validation during exploration.

### 3.5 Trust Calibration Differences by Experience Level

**Stack Overflow 2025 Developer Survey:** 84% of respondents are using or planning to use AI tools. However, positive sentiment toward AI tools declined to 60% in 2025 from over 70% in prior years. Only 29% expressed trust in AI tools (down from 40% in 2023). 46% of developers distrust the accuracy of AI tools, with experienced developers showing the highest caution. Only 3% "highly trust" AI tool outputs. [46]

**Anthropic/SoftwareSeni Trust Data:** "78% of junior developers trust AI specificity compared to just 39% for seniors." [47] This is the most direct cohort-specific trust calibration data point found: juniors trusting AI suggestions at double the rate of seniors.

### 3.6 Other Studies with Cohort-Specific Quantitative Data

**Google's Internal AI Developer Productivity Study:** A randomized controlled trial at Google evaluating the combined effect of three AI tools found that developers using AI tools completed tasks 21% faster. "Notably, senior developers experienced the largest productivity gains, challenging the belief that AI mainly aids junior developers." [48] This is a critical counterpoint to the MIT/Demirer findings—Google found seniors benefit most from AI in complex coding environments.

**METR Study (July 2025):** 16 experienced open-source developers worked on 246 tasks using Cursor Pro with Claude 3.5/3.7 and Sonnet 3.7. Experienced developers were on average 19% slower when using AI tools. Despite being objectively slower, developers believed AI had made them 20% faster—a 39-percentage-point perception gap. Developers accepted less than 44% of AI-generated code suggestions. [25]

**Harvard Study on AI and Junior Hiring:** A Harvard study tracking 62 million workers found that companies that adopt generative AI cut junior developer hiring by 9-10% within six quarters. [49]

---

## Section 4: Systematic Cross-Vendor Comparison

### 4.1 Latency Regimes Comparison

| Vendor | Single-Line Autocomplete | Multi-Line Predictions | Comment-to-Code | Filter/Quality Check | Explanation/Rationale |
|--------|-------------------------|----------------------|-----------------|---------------------|---------------------|
| **GitHub Copilot** | ~130-200ms | ~700ms (agent mode) | ~1-2s | N/A | Asynchronous |
| **Tabnine** | 15-80ms (local), 180-600ms (cloud) | 180-600ms (cloud) | 180-600ms | 1-2ms CatBoost | Asynchronous |
| **Amazon Q Developer** | ~200ms inline | 600ms-1.4s (optimized) | 600ms-1.4s | N/A | Asynchronous |
| **JetBrains FLCC** | 50-75ms (local) | N/A | N/A | 1-2ms filter | N/A |
| **JetBrains Mellum** | 1-2s (cloud), <750ms p90 (H100) | 1-2s (cloud) | 1-2s | N/A | Asynchronous |
| **JetBrains NES** | <200ms | <200ms | N/A | N/A | In-editor diff |

### 4.2 Acceptance Rates Comparison

| Vendor | Reported Acceptance Rate | Context |
|--------|------------------------|---------|
| **GitHub Copilot** (general) | 27-33% | Industry average across multi-line and single-line |
| **GitHub Copilot** (ZoomInfo) | 33% suggestions, 20% lines | 400+ developers enterprise deployment |
| **GitHub Copilot** (2025 model) | 12% higher than previous | New custom model |
| **Tabnine** | 30-40% code automation (90% single-line) | Across 1,000,000+ developers |
| **Tabnine** (Enterprise Context Engine) | 82% increase vs out-the-box | Enterprise with custom context |
| **Amazon Q Developer** (BT Group) | 37% | 1,200 developers |
| **Amazon Q Developer** (NAB) | 50% | 2,800-3,000 developers planned |
| **JetBrains FLCC** | 1.3x-1.5x usage increase | Local single-line completion |
| **JetBrains Mellum** | 30-44% (AR), 18-46% (RoCC) | Various languages |
| **OpenAI Codex** | 59.6-88.6% | Across nine task categories |

**Sources:** [8][13][14][17][18][19][22][27]

### 4.3 Task-Specific Performance Comparison

| Task Type | Best Tool | Acceptance Rate | Latency Sensitivity |
|-----------|-----------|----------------|-------------------|
| Documentation | Claude Code (92.3%) | 82.1% overall | Low (1-3s acceptable) |
| Feature development | Claude Code (72.6%) | 66.1% overall | Medium (200-600ms) |
| Bug fixes | Cursor (80.4%) | Varies | High (<200ms required) |
| Single-line completion | Tabnine local / FLCC | 15-80ms | Critical (<100ms ideal) |
| Multi-line suggestions | Copilot Agent / NES | <200ms-700ms | Medium (200-500ms acceptable) |
| Comment-to-code | Amazon Q / Copilot | 37-50% | Low (1-3s acceptable) |

### 4.4 Explanation/Trust Calibration Features Comparison

| Vendor | Confidence Scores | Explanation Availability | Quality Indicators |
|--------|------------------|------------------------|-------------------|
| **GitHub Copilot** | No explicit scores | No explanations by default | Ghost text only |
| **Tabnine** | No explicit scores | Basic code explanations | Acceptance rate history |
| **Amazon Q Developer** | Security scanning built-in | Chat interface for explanations | Code reference tracking |
| **JetBrains AI** | No explicit scores | AI Chat for explanations | Integration with PSI |

### 4.5 Junior vs. Senior Developer Implications

| Vendor | Strengths for Juniors | Strengths for Seniors | Concerns |
|--------|----------------------|---------------------|----------|
| **GitHub Copilot** | Fast suggestions, easy to accept | Acceleration mode support, 22% faster (Jellyfish) | Overreliance risk for juniors |
| **Tabnine** | Local inference eliminates network lag | Hybrid architecture provides flexibility | Local model may have lower quality |
| **Amazon Q Developer** | Security scanning protects juniors | Customizable models (Claude/Llama) | Higher latency variance (P90 problem) |
| **JetBrains FLCC** | Instantaneous local completion | Seamless IDE integration | Single-line only |
| **JetBrains Mellum** | Deep project context understanding | Superior cross-file understanding | 1-2s latency may disrupt flow |

---

## Section 5: Suggestion Timing: Proactive vs. On-Demand by Developer State

### 5.1 The ProAIDE Study: Timing Matters Most

**Source Paper:** "Developer Interaction Patterns with Proactive AI: A Five-Day Field Study" (arXiv:2601.10253, 2026) by Nadine Kuo, Agnia Sergeyuk, Valerie Chen, and Maliheh Izadi, conducted in JetBrains' Fleet IDE. [50]

**Methodology:** Five-day in-the-wild field study with 15 professional developers interacting with 229 AI interventions over 5,732 interaction points. Mixed-methods approach combining telemetry data with structured surveys.

**Key Quantitative Findings:**

| Timing Context | Engagement | Interpretation |
|---------------|------------|----------------|
| Workflow boundaries (post-commit, file save) | **52% engagement** | Highest engagement context |
| Mid-task interventions | **62% dismissal** | Active coding flow is disrupted |
| Well-timed proactive suggestions | 45.4s interpretation time | Enhanced cognitive alignment |
| Reactive suggestions | 101.4s interpretation time | Higher cognitive cost |

**Source:** [50]

The study found that "carefully timed proactive suggestions, particularly at natural workflow boundaries, significantly reduce cognitive load and increase engagement compared to reactive prompts." System Usability Scale (SUS) score of 72.8 out of 100.

### 5.2 The Codellaborator Study: The Fundamental Trade-Off

**Source Paper:** "Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support" (arXiv:2502.18658, 2025) by Kevin Pu and colleagues. [51]

**Methodology:** Within-subject study with 18 upper-level computer science students comparing three interface variants: PromptOnly (user-initiated), CodeGhost (proactive without AI presence indicators), and Codellaborator (full proactive with AI presence and interaction scopes).

**Key Findings:**

- "Proactive agents increase efficiency compared to prompt-only paradigm, but also incur workflow disruptions"
- "The standout finding is the trade-off between efficiency and workflow disruptions"
- "Presence indicators and interaction context supported alleviated disruptions and improved users' awareness of AI processes"
- "Participants expressed mixed feelings about AI proactivity, appreciating reduced effort but concerned about loss of understanding and ownership"
- "Future systems should prioritize fostering code understanding rather than focusing solely on efficiency"

### 5.3 Barke et al. Modes and Timing Implications

The bimodal interaction model from Barke et al. has direct implications for suggestion timing:

**Acceleration Mode (typically seniors with clear mental models):**
- Proactive suggestions should be short and precise (<200ms)
- Multi-line suggestions can break flow
- On-demand invocation preferred for complex completions

**Exploration Mode (typically juniors or unfamiliar tasks):**
- Proactive suggestions can be longer (200-1000ms)
- Multiple suggestions and explanations are valuable
- On-demand invocation for specific queries

### 5.4 Parnin & Rugaber Interruption Cost Integration

The Parnin finding that "only 10% of sessions resume programming activity in less than 1 minute after interruption" [31] combined with the ProAIDE finding of 62% dismissal mid-task [50] provides a clear imperative: **avoid proactive suggestions during active coding**.

The worst time to interrupt a programmer is "during peak memory load, such as while editing or comprehending complex code." [31] This means that proactive suggestions during debugging or complex reasoning are particularly costly.

### 5.5 Design Guidance: When to Show Suggestions Proactively

**Evidence-Based Proactive Timing:**

1. **At workflow boundaries (post-commit, file save, file open):** 52% engagement [50]. This is the highest-engagement context for proactive suggestions.

2. **During boilerplate generation:** High acceptance rates (82.1% for documentation tasks) [27]. Low cognitive load makes this an ideal time.

3. **During documentation generation:** 82.1% acceptance rate [27]. Low evaluation cost makes moderate latency acceptable.

4. **During idle periods >3 seconds:** Low disruption risk. The developer has paused and may welcome assistance. [51]

5. **After completing a function or method:** Natural task boundary. The developer is in a "closure" state per Miller's research. [3]

### 5.6 Design Guidance: When to Require On-Demand Invocation

**Evidence-Based On-Demand Requirements:**

1. **Mid-debugging sessions:** 62% dismissal rate [50]. Interruptions during debugging double task time per Parnin [31].

2. **During complex architecture decisions:** AI tools can hinder experienced developers (METR study: 19% slower) [25].

3. **During code review:** Developers explicitly separate writing from reviewing. One developer in ProAIDE stated: "I separate writing code and optimizing it... Reviewing should happen after I've written something and feel it's ready for review." [50]

4. **After a suggestion has been rejected:** Repeated interruptions after rejection have near-zero acceptance and compound disruption.

5. **During peak cognitive load (complex reasoning, algorithmic tasks):** The NSF study found "algorithmic tasks posed the greatest challenge" for recovery from interruptions. [36]

### 5.7 Latency Budgets by Timing Scenario

| Timing Scenario | Max Acceptable Latency | Recommended Approach | Rationale |
|----------------|----------------------|---------------------|-----------|
| Inline autocomplete (acceleration mode) | **<200ms** | Local model (15-80ms TTFT) | Flow state preservation; sub-200ms is non-negotiable |
| Proactive suggestion at workflow boundary | **<500ms** | Local or optimized cloud | Lower cognitive load at boundary; slight delay tolerable |
| Proactive documentation/explanation | **<1s** | Cloud (for longer outputs) | Non-edit suggestion; task context supports slightly higher latency |
| On-demand complex generation (exploration) | **1-3s** | Cloud (80-150 tok/s sustained) | User initiated; expects substantive output |
| Any interactive suggestion | **<3s max** | Optimized pipeline | >3s creates micro-interruptions and broken flow; P95/P99 must be controlled |

**Source:** Synthesized from [3][4][5][6][12][50][51]

---

## Section 6: Explanation Availability and Trust Calibration

### 6.1 Microsoft Research FAccT 2024 (Wang et al.)

**Source Paper:** "Investigating and Designing for Trust in AI-powered Code Generation Tools" presented at ACM FAccT 2024 by Wang, Cheng, Ford, and Zimmermann (Microsoft Research). [52]

**Three Dimensions of Trust:**

1. **Ability:** The AI tool's perceived practical benefits and utility. Developers assess AI ability based on practical benefits like time saved and code contributed.

2. **Integrity:** Understanding and agreeing with the AI's operation. Developers trust AI more when they understand its mechanisms and security/privacy aspects.

3. **Benevolence:** Alignment of the AI's goals with developers' goals. Developers worry that AI tools could hinder their learning or replace jobs.

The study found that trust is **situational**, varying according to context of usage, specifically task stakes and complexity.

**Three Key Challenges:**

1. **Lack of reliable data on AI ability:** Current AI tools lack affordances for efficient trust evaluation, resulting in reliance on personal intuition, which leads to inefficient and biased trust assessments.

2. **Time-consuming evaluation of AI output:** Developers must spend significant time evaluating whether AI suggestions are trustworthy.

3. **Lack of mechanisms to communicate user intentions to AI:** There is no effective way for developers to convey their goals, preferences, and intentions.

**Three Key Design Concepts:**

1. **Usage statistics dashboard:** A personalized and contextual performance data dashboard to reflect on AI capabilities.

2. **Quality indicators at solution, token, and file levels:** Showing confidence and familiarity levels at solution, token, and file context to help efficient in-situ evaluation.

3. **Control mechanisms:** Allowing users to explicitly convey intentions, goals, and preferences to align AI behavior with their needs.

### 6.2 UC Berkeley Trust Calibration Study

**Source:** "Calibrating Trust in AI-Assisted Decision Making" (UC Berkeley School of Information). [53]

**Key Findings:**

- **Confidence scores help calibrate trust in an AI model**, whereas local explanations had no effect on improving trust calibration and accuracy.
- **Explanations do not always aid trust calibration, and can actually hurt it, especially for novice users who have low self-competence.**
- Data availability explanations may have created perceptions that the system is more capable than it is, resulting in **overtrust or misuse**.
- Lowering self-competence appears to stimulate people's willingness to lean on AI recommendations, at least when seeing certain types of explanations.

**Methodology:** Two empirical studies. Study 1 tested how two types of local explanations—AI confidence scores and data availability—affect participants' trust and accuracy in a fictional plant classification task with AI assistance. Study 2 replicated this but manipulated participants' self-competence to be low.

**Quantitative Findings:**
- Confidence explanations did not significantly improve accuracy or trust compared to controls.
- Data availability explanations tended to increase participants' switching to AI recommendations, even when it lowered accuracy, particularly when self-competence was low, suggesting potential overtrust and misuse.

### 6.3 Amazon Science Trust Dynamics Research

**Source:** Sabouri et al., "Trust Dynamics in AI-Assisted Development: Definitions, Factors, and Implications," ICSE 2025, published by Amazon Science. [54]

**Key Quantitative Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Initial acceptance rate | **82%** | Most suggestions initially accepted |
| Retention rate after review | **52%** | Frequent alteration of trust decisions |
| Mistaken trust instances | **48%** | Nearly half involved mistakenly trusting code |

**Source:** [54]

Developers prioritize **correctness and comprehensibility** when defining trustworthy code, but rely even more heavily on **comprehensibility** when assessing AI code in practice. Developers maintain a **positive bias** toward AI assistance, tending to forgive errors.

**Four Proposed Guidelines:**

1. **Double-Sided Clarification:** Using structured prompts and precise language to define desired functionalities.
2. **Prioritize Code Quality Preferences:** Defining preferences like comprehensibility and style before using AI assistants.
3. **Evaluate Thoroughly:** Avoid blind acceptance to mitigate automation bias.
4. **Value Simplicity:** Valuing simplicity in code to aid understandability and trust.

### 6.4 Overreliance Mitigation Literature

**Cognitive Forcing Functions (Bucinca et al., CHI 2021):** Zana Buçinca and colleagues found that people supported by AI-powered decision support tools frequently overrely on the AI. Adding explanations to AI decisions does not appear to reduce overreliance and some studies suggest it might even increase it. [55]

Three cognitive forcing interventions were designed and tested: **On Demand**, **Update**, and **Wait**. Cognitive forcing functions significantly reduced overreliance compared to simple explainable AI approaches. However, there was a trade-off: cognitive forcing functions led to better performance but received lower subjective ratings in terms of user preference, trust, and perceived system complexity.

**The "Human Goes First" Pattern:** Requires users to complete a task or diagnosis before viewing AI suggestions, then compare results to foster learning and maintain skill. This process encourages reflection and microlearning based on comparative outcomes, and helps prevent user deskilling and overreliance on AI. [56]

**The "Generation-Then-Comprehension" Pattern (Anthropic Study):** A randomized controlled trial with 52 mostly junior engineers found that AI-assisted participants scored 17% lower on comprehension tests than those who coded manually (50% vs 67%). However, participants who used the Generation-Then-Comprehension pattern scored **86%** on comprehension, significantly better than the control group's 67%. [28]

This pattern involves: "Generate code first, then immediately ask 'Explain this line by line. What could go wrong?'" According to cognitive load theory analysis, this pattern minimized intrinsic and extraneous load by leveraging AI to generate code, freeing cognitive resources for active comprehension.

### 6.5 Design Recommendations for Enterprise IDEs

**Confidence Scores at Solution/Token/File Levels:**

From Wang et al. [52], the design concept of quality indicators involves showing confidence and familiarity levels at:
- **Solution level** – overall confidence in the suggested solution
- **Token level** – per-token confidence for fine-grained assessment
- **File context level** – confidence relative to the existing codebase

**Progressive Disclosure:**

Principles from the progressive disclosure literature for enterprise AI: [57][58]
- "Start simple. Give users clear, actionable summaries or defaults first."
- "In AI-powered SaaS, where users often interact with predictions, automations, or decision-support tools, progressive disclosure becomes a powerful ally in building trust and clarity."
- "Trust is fragile. If users don't feel in control, they'll avoid the feature, or worse, the product."

Best practices include starting with simple summaries, offering user control to explore complexity, leveraging familiar UI patterns, providing contextual education, designing for transparency, and using role-based access to AI functionalities.

**Seniority-Adapted Explanation Depth:**

From the UC Berkeley study [53]: Novice users (low self-competence) were more likely to overtrust AI when given data availability explanations. Explanations can hurt calibration for novices, while confidence scores help. This suggests that explanation depth should be adapted to developer seniority.

From the Fly.io trust calibration article [59]: "Adaptive calibration—when a system actively monitors user behavior and adjusts its communication accordingly—is orders of magnitude more effective than static calibration." The Transparency Paradox is noted: "While explanations can improve user understanding, they can also create information overload that reduces users' ability to detect and correct trash output."

### 6.6 Trust Calibration Mechanisms Summary

| Mechanism | Evidence | Recommendation |
|-----------|----------|----------------|
| **Confidence scores** | Berkeley: helps calibration [53]; Amazon Science: 82% initial acceptance [54] | Show at solution, token, and file levels |
| **Natural language explanations** | Berkeley: can hurt calibration for novices [53]; Bucinca: may increase overreliance [55] | Use sparingly; adapt depth to seniority |
| **Cognitive forcing functions** | Bucinca: reduces overreliance but lower user satisfaction [55] | Implement selectively for high-risk suggestions |
| **Generation-then-comprehension** | Anthropic: 86% comprehension score [28] | Recommend as usage pattern for training |
| **Progressive disclosure** | Semelin: builds trust through clarity [57] | Start simple, allow drilling down |
| **Double-sided clarification** | Amazon Science: structured prompts [54] | Implement preference setting interfaces |

---

## Section 7: Practical Recommendations for Enterprise IDE Design

### 7.1 Latency Targets by Suggestion Type

Based on the accumulated evidence, anchored in HCI foundational thresholds:

| Suggestion Type | Target Latency | HCI Regime | Rationale |
|----------------|---------------|------------|-----------|
| Single-line autocomplete | **<100ms** | **Instantaneous** | Must feel like own typing; Tabnine/FLCC demonstrate feasibility (15-80ms) |
| Multi-line suggestions | **200-600ms** | **Flow-preserving** | Acceptable cloud TTFT range; Copilot's proven target |
| Complex generation (comments→code) | **<2500ms** | **Flow-disruptive but tolerable** | Requires visual feedback; user-initiated action |
| Filter model (quality check) | **<10ms** | **Negligible** | Below perception; CatBoost model proves feasibility (1-2ms) |
| Explanations/rationales | **<1000ms** | **Asynchronous** | Can load after suggestion; should not delay primary suggestion |

### 7.2 Proactive vs. On-Demand by Task Type and Developer State

| State | Delivery Mode | Rationale |
|-------|---------------|-----------|
| Typing in flow (acceleration mode) | On-demand or proactive <100ms | Interruption cost is high; any delay breaks flow |
| At workflow boundary (post-commit) | Proactive, 200-1000ms | 52% engagement; user is at natural pause |
| Debugging (peak cognitive load) | On-demand only | Interruption doubles task time per Parnin |
| Boilerplate generation | Proactive, <200ms | High acceptance (82.1%); low evaluation cost |
| Exploration mode (junior/unfamiliar) | Proactive with explanations | Longer suggestions OK; trust calibration needed |
| After declined suggestion | Suppressed for 10-15 seconds | Repeated rejection has near-zero acceptance |

### 7.3 Seniority-Adapted Interface Design

**For Junior Developers (2-5 years):**

- **More proactive suggestions** with confidence indicators to calibrate trust appropriately (78% trust vs. 39% for seniors) [47]
- **Explanation availability** is critical—they exhibit high trust and are susceptible to over-trust
- **Structured training** on AI limitations (Microsoft recommends onboarding showing both correct and incorrect suggestions)
- **Shorter suggestions** to prevent comprehension gaps and anchoring bias
- **Cognitive forcing functions** requiring verification of high-risk suggestions before acceptance
- **Generation-then-comprehension pattern** recommended as a usage workflow to preserve learning [28]

**For Senior Developers (10+ years):**

- **More on-demand invocation**—they have clear mental models and benefit from acceleration mode (22% faster coding per Jellyfish) [41]
- **Less frequent proactive suggestions** to avoid disrupting flow
- **Richer explanation options** (code rationale, source attribution) for the 39% who demonstrate selective trust
- **Control mechanisms** (role settings, context sliders) to set AI behavior preferences
- **Partial acceptance support** (line-by-line, token-by-token) for precise integration
- **P95/P99 latency control**—seniors are more sensitive to unpredictable delays during flow state

### 7.4 Trust Calibration Mechanisms

| Strategy | Implementation | Evidence |
|----------|----------------|----------|
| Confidence scores | Show at solution/token/file levels | Berkeley: helps calibration [53]; Wang et al.: design concept [52] |
| Progressive disclosure | Start simple, allow drill-down | Semelin: builds trust through clarity [57] |
| Cognitive forcing functions | Require verification for security-critical suggestions | Bucinca: reduces overreliance [55] |
| Generation-then-comprehension | Recommended as training pattern | Anthropic: 86% comprehension score [28] |
| Double-sided clarification | Structured prompts for preferences | Amazon Science: reduces ambiguity [54] |
| Adaptive calibration | Monitor behavior, adjust communication | Fly.io: orders of magnitude more effective [59] |

### 7.5 Enterprise Deployment Architecture Recommendations

**Hybrid Local/Cloud Architecture** (Tabnine model): The optimal latency balance based on the evidence is a hybrid approach:

- **Local inference** for simple, repetitive completions (15-80ms TTFT, instantaneous regime)
- **Cloud inference** for complex, multi-line completions (180-600ms TTFT, flow-preserving regime)
- **Filter model** running locally in <10ms to pre-filter low-quality suggestions
- **Caching** of hidden states to reduce repeated context processing (JetBrains approach)
- **Global deployment** with smart routing and auto-failover (Copilot model)

**Metrics to Track Beyond Acceptance Rate:**

- Time-to-task-completion (distinguish from perceived speed)
- Code churn rate (lines reverted or updated within two weeks)
- Comprehension assessments (for junior developer growth tracking)
- Flow state disruption frequency (context switching events)
- Suggestion latency P50, P90, and P99 (not just averages)
- Perception-Actuality gap (subjective vs. objective productivity)
- Suggestion retention rate (Amazon Science: 52% retention after review) [54]

---

## 7.6 Research Gaps and Future Directions

1. **No controlled experiments directly comparing latency regimes and flow state impact:** While Copilot's 200ms target exists, no published research directly compares 50ms, 200ms, 500ms, and 2000ms latency conditions in a controlled experiment measuring flow state disruption.

2. **No published research on experience-level specific latency tolerance:** While indirect evidence suggests seniors may be more sensitive to latency during acceleration mode, no studies directly measure latency tolerance by experience level in controlled conditions.

3. **Limited longitudinal data on AI's impact on junior developer skill development:** The 17-point comprehension gap is documented but no long-term studies (>12 months) track how AI-dependent junior developers develop architectural judgment and debugging skills.

4. **No large-scale A/B tests comparing explanation modalities in production environments:** Trust calibration research is primarily from lab studies or design probes; production IDE experiments are needed.

5. **Cross-tool comparison studies are notably absent:** There are no independent, head-to-head comparisons of Copilot, Tabnine, and Amazon Q Developer on the same tasks with the same developer populations.

6. **The perception-actuality gap requires more research:** The METR finding that developers believed they were 20% faster while being 19% slower suggests subjective productivity assessments are unreliable.

7. **Task-specific latency thresholds are undefined:** We know that documentation generation accepts higher latency than inline completion, but precise latency thresholds by task type remain unquantified.

---

## Sources

[1] Nielsen, Jakob. "The Need for Speed in AI": https://jakobnielsenphd.substack.com/p/the-need-for-speed-in-ai

[2] Card, Stuart K.; Moran, Thomas P.; Newell, Allen (1983). "The Psychology of Human-Computer Interaction": https://en.wikipedia.org/wiki/Human_processor_model

[3] Miller, Robert B. (1968). "Response time in man-computer conversational transactions": https://www.semanticscholar.org/paper/Response-time-in-man-computer-conversational-Miller/1abb835694f93afe6335aa7a5fd6effe075b99d5

[4] Nielsen, Jakob (1993/2014). "Response Times: The 3 Important Limits": https://www.nngroup.com/articles/response-times-3-important-limits

[5] Shneiderman, Ben (1984). "Response Time and Display Rate in Human Performance with Computers": https://www.cs.umd.edu/~ben/papers/Shneiderman1984Response.pdf

[6] Cheney, David. "How GitHub Copilot Serves 400 Million Completion Requests a Day" (QCon SF 2024): https://www.youtube.com/watch?v=zemBW3diXIs

[7] "GitHub Copilot's Latency Secrets": https://www.classcentral.com/course/youtube-github-copilot-s-latency-secrets-how-they-built-sub-200ms-autocomplete-443997

[8] "Building a faster, smarter GitHub Copilot with a new custom model" (GitHub Blog): https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model

[9] "Copilot Coding Agent 50% Faster: March 19 Update": https://www.digitalapplied.com/blog/copilot-coding-agent-50-faster-march-19-performance

[10] "New GitHub Copilot code completion model: GPT-4o Copilot": https://www.youtube.com/watch?v=xrgK8cl_B7U

[11] "Tabnine goes hybrid, serving AI models on both cloud and local": https://www.tabnine.com/blog/tabnine-goes-hybrid

[12] "Local vs Cloud AI Coding: Latency, Privacy & Performance Guide" (SitePoint, 2026): https://www.sitepoint.com/local-vs-cloud-ai-coding-performance-analysis-2026

[13] Tabnine AI Code Assistant: https://www.tabnine.com

[14] "The Battle of AI Code Editors: Copilot vs Cursor vs CodeWhisperer": https://medium.com/@mbodhija80/the-battle-of-ai-code-editors-copilot-vs-cursor-vs-codewhisperer-1bc0f824b44d

[15] "Optimizing AI responsiveness: A practical guide to Amazon Bedrock latency-optimized inference": https://aws.amazon.com/blogs/machine-learning/optimizing-ai-responsiveness-a-practical-guide-to-amazon-bedrock-latency-optimized-inference

[16] "Introducing latency-optimized inference for foundation models in Amazon Bedrock": https://aws.amazon.com/about-aws/whats-new/2024/12/latency-optimized-inference-foundation-models-amazon-bedrock

[17] "Full Line Code Completion: Bringing AI to Desktop" (arXiv:2405.08704): https://arxiv.org/html/2405.08704v3

[18] "Mellum: How We Trained a Model to Excel in Code Completion" (JetBrains AI Blog): https://blog.jetbrains.com/ai/2025/04/mellum-how-we-trained-a-model-to-excel-in-code-completion

[19] "Mellum: Production-Grade in-IDE Contextual Code Completion with Multi-File Project Understanding" (arXiv:2510.05788): https://arxiv.org/html/2510.05788v1

[20] "JetBrains Mellum | IDE Services Documentation": https://www.jetbrains.com/help/ide-services/jetbrains-mellum.html

[21] "Next Edit Suggestions: Now Generally Available" (JetBrains AI Blog): https://blog.jetbrains.com/ai/2025/12/next-edit-suggestions-now-generally-available

[22] "Experience with GitHub Copilot for Developer Productivity at Zoominfo": https://arxiv.org/html/2501.13282v1

[23] "How to measure AI's impact on developer productivity" (DX): https://getdx.com/blog/ai-measurement-hub

[24] "Developers save up to 8 hours per week with JetBrains AI Assistant": https://blog.jetbrains.com/ai/2024/04/developers-save-up-to-8-hours-per-week-with-jetbrains-ai-assistant

[25] METR Study (July 2025): https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study

[26] "State of Engineering Excellence 2026 Report": https://www.okoone.com/spark/industry-insights/ai-coding-tools-are-slow-down-experienced-developers

[27] "Comparing AI Coding Agents: A Task-Stratified Analysis of PR Acceptance" (arXiv 2602.08915): https://arxiv.org/html/2602.08915v2

[28] Shen and Tamkin. "How AI Impacts Skill Formation" (arXiv:2601.20245): https://arxiv.org/abs/2601.20245

[29] Monk, Boehm-Davis, & Trafton (2004). "Very brief interruptions result in resumption cost": https://interruptions.net/literature/Monk-CogSci04.pdf

[30] Monk, Boehm-Davis, & Trafton (2008). "The Effect of Interruption Duration and Demand on Resuming Suspended Goals": https://interruptions.net/literature/Monk-JEPA08.pdf

[31] Parnin & Rugaber (2011). "Resumption strategies for interrupted programming tasks": http://chrisparnin.me/pdf/parnin-sqj11.pdf

[32] Parnin & Rugaber (2009). "Resumption Strategies for Interrupted Programming Tasks" (ICPC): http://chrisparnin.me/pdf/parnin-icpc09.pdf

[33] Mark, Gudith, & Klocke (2008). "The Cost of Interrupted Work: More Speed and Stress": https://ics.uci.edu/~gmark/chi08-mark.pdf

[34] "Flow state: Why fragmented thinking is worse than any interruption" (StackBlitz): https://blog.stackblitz.com/posts/flow-state

[35] Csikszentmihalyi, Mihaly (1990). "Flow: The Psychology of Optimal Experience"

[36] "Interruptions and Recovery: Leveraging Dynamic Code History in Development" (NSF-funded, 2023): https://par.nsf.gov/servlets/purl/10660110

[37] "Flow State and AI: How It Kills Deep Work for Engineers": https://clearing-ai.com/flow-state.html

[38] Peng, Demirer et al. "The Effects of Generative AI on High-Skilled Work" (MIT Economics): https://economics.mit.edu/sites/default/files/inline-files/draft_copilot_experiments.pdf

[39] "The Impact of AI on Developer Productivity: Evidence from GitHub Copilot" (arXiv 2302.06590): https://arxiv.org/abs/2302.06590

[40] "New Research Reveals AI Coding Assistants Boost Developer Productivity by 26%" (IT Revolution): https://itrevolution.com/articles/new-research-reveals-ai-coding-assistants-boost-developer-productivity-by-26-what-it-leaders-need-to-know

[41] Jellyfish LinkedIn Analysis: "What is GitHub Copilot really changing in engineering?": https://www.linkedin.com/posts/jellyfish-co_what-is-github-copilot-really-changing-in-activity-7284947633729609728-zNWL

[42] "With Copilot, Study Finds Engineers Get 15% More Capacity": https://jellyfish.co/blog/with-copilot-engineers-get-15-more-capacity-without-additional-headcount

[43] "Fastly: Senior Devs Ship 2.5x More AI Code Than Juniors" (The New Stack): https://thenewstack.io/fastly-senior-devs-ship-2-5x-more-ai-code-than-juniors

[44] "Senior Devs Ship 2.5x More AI Code Than Juniors" (Slashdot): https://developers.slashdot.org/story/25/09/07/0615217/32-of-senior-developers-say-half-their-shipped-code-is-ai-generated

[45] Barke, James, Polikarpova. "Grounded Copilot: How Programmers Interact with Code-Generating Models" (OOPSLA 2023): https://shraddhabarke.github.io/raw/copilot.pdf

[46] Stack Overflow 2025 Developer Survey: https://survey.stackoverflow.co/2025/ai

[47] "Junior Developers in the Age of AI" (SoftwareSeni): https://www.softwareseni.com/junior-developers-in-the-age-of-ai-who-trains-the-next-generation-of-engineers

[48] "Gen AI Research: Software Development Productivity At Google" (LinearB): https://linearb.io/blog/gen-AI-research-software-development-productivity-at-google

[49] "The AI coding productivity data is in and it's not what anyone expected" (Reddit): https://www.reddit.com/r/ExperiencedDevs/comments/1rnkv2t/the_ai_coding_productivity_data_is_in_and_its_not/

[50] "Developer Interaction Patterns with Proactive AI: A Five-Day Field Study" (arXiv:2601.10253): https://ui.adsabs.harvard.edu/abs/2026arXiv260110253K/abstract

[51] "Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support" (arXiv:2502.18658): https://arxiv.org/abs/2502.18658

[52] Wang et al. "Investigating and Designing for Trust in AI-powered Code Generation Tools" (FAccT 2024): https://dl.acm.org/doi/10.1145/3630106.3658984

[53] "Calibrating Trust in AI-Assisted Decision Making" (UC Berkeley): https://www.ischool.berkeley.edu/sites/default/files/sproject_attachments/humanai_capstonereport-final.pdf

[54] Sabouri et al. "Trust Dynamics in AI-Assisted Development" (ICSE 2025, Amazon Science): https://assets.amazon.science/99/78/f02aeaa049b4ba514d7f2790ade7/trust-dynamics-in-ai-assisted-development-definitions-factors-and-implications.pdf

[55] Buçinca et al. "To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI" (CHI 2021): https://arxiv.org/abs/2102.09692

[56] Christopher Noessel. "Mitigating AI Overreliance with Cognitive Forcing Functions": https://www.linkedin.com/posts/chrisnoessel_ai-overreliance-deskilling-activity-7462523063667015680-TuHs

[57] Lucas Semelin. "Progressive Disclosure in AI-Powered SaaS": https://www.linkedin.com/pulse/progressive-disclosure-ai-powered-saas-designing-clarity-semelin-95z2f

[58] Steve Smith (Ardalis). "Optimizing AI Agents with Progressive Disclosure": https://ardalis.com/optimizing-ai-agents-with-progressive-disclosure

[59] "Trust Calibration for AI Software Builders" (Fly.io): https://fly.io/blog/trust-calibration-for-ai-software-builders