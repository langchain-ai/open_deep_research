# AI-Powered Code Completion Interfaces: A Comprehensive Research Report for Enterprise IDE Design

## Executive Summary

This revised and expanded report synthesizes findings from over 80 primary sources—including Microsoft Research studies, academic publications on interruption costs, documented metrics from JetBrains deployments, security assessments from BSI/ANSSI, and official publications from GitHub Copilot, Tabnine, and Amazon Q Developer—to provide actionable guidance on designing AI-powered code completion interfaces for enterprise software teams. The evidence reveals seven critical dimensions that must be addressed holistically: latency thresholds and performance engineering, acceptance rates and interruption timing by task type, adaptation to developer experience levels, explanation availability and trust calibration, code quality outcomes, practical design recommendations, and identification of remaining research gaps.

Key findings include: GitHub Copilot's documented sub-200ms average response time represents the industry's only explicit latency target, enabled by HTTP/2 multiplexing, cancellation mechanisms, global routing via octoDNS, and a proxy layer; proactive suggestions at workflow boundaries achieve 52% engagement versus 62% dismissal rates mid-task; junior developers show 27-40% productivity gains compared to 7-16% for seniors, yet also exhibit 78% trust in AI specificity versus 39% for experienced engineers; explanation availability must be carefully calibrated to avoid both over-trust and under-trust, with confidence scores proving more effective than data availability explanations for trust calibration; and code quality concerns are substantial, with AI-generated code containing 1.7x more issues and 2.74x more security vulnerabilities according to CodeRabbit analysis.

---

## 1. Latency Thresholds & Performance Engineering

### 1.1 The Critical Role of Time-to-First-Token (TTFT)

The foundational metric for AI code completion responsiveness is Time-to-First-Token (TTFT). Analysis comparing local versus cloud AI coding infrastructure demonstrates that TTFT "is the single most consequential metric for coding assistant responsiveness" [18]. Local inference delivers a TTFT of **15–80ms** compared to **180–600ms** for cloud-based inference, making local inference superior for short completions (autocomplete scenarios). Cloud models have throughput advantages of 80–150 tokens per second versus 35–65 tokens/sec on local hardware, making cloud faster for longer generations exceeding approximately 200–300 output tokens [18][20].

The crossover point "is the single most important number for deciding how to split workloads in a hybrid setup" [18]. For autocomplete-heavy workflows, which describe most coding patterns, local inference should be prioritized for its latency characteristics alone.

### 1.2 GitHub Copilot's Sub-200ms Engineering Target

GitHub Copilot's latency architecture is the most concretely documented in the industry. David Cheney, Lead of Copilot Proxy at GitHub, revealed in his QCon San Francisco 2024 presentation that Copilot serves **"hundreds of millions of requests a day with an average response time of under 200 milliseconds"** , peaking at approximately **8,000 requests per second** during Europe-US time overlap [4][5]. This sub-200ms target was explicitly engineered to compete with local IDE autocomplete performance while operating as a cloud-hosted service.

The engineering decisions that enable this latency include:

**HTTP/2 Multiplexing:** A critical innovation was using end-to-end HTTP/2 to multiplex multiple requests over a single connection, enabling request cancellation and avoiding the expense of TCP/TLS connection setup on retries. Cheney stated: "HTTP/2 is critical to the Copilot latency story. Without cancellation, we'd make twice as many requests and waste half of them" [5]. GitHub's custom load balancer and proxy filled the gap, as "it's surprisingly difficult to do HTTP/2 end-to-end with off-the-shelf tools" [5].

**Cancellation Mechanisms:** GitHub's proxy handles cancellation signals so that when a user continues typing, the in-flight request is cancelled rather than allowing the model to complete it pointlessly. Without this, half the model time would be wasted on requests the client had already abandoned [4][5].

**Global Routing via octoDNS:** GitHub's proxy routes requests geographically using octoDNS across Azure regions hosting models worldwide, and automatically redirects traffic away from unhealthy regions for resilience. Cheney noted: "By having multiple models around the globe, that turns SEV1 incidents into just SEV2 alerts. If a region is down or overloaded, traffic just flows somewhere else" [5].

**Proxy Layer:** GitHub developed an authenticating proxy (copilot-proxy) to securely mediate IDE requests via OAuth and short-lived tokens. Cheney explained: "For every request we get, we don't have to call out to an external authentication service. The short-lived token is the authentication" [5]. The proxy enabled request mutation on the fly, immediate fixes, A/B testing, and management of heterogeneous client populations without requiring immediate client updates.

Cheney summarized: "If you want low latency, you have to bring your application closer to your users" [5].

### 1.3 Copilot SDK Profiling: Client Lifecycle Overhead

Kevin Tan's performance profiling of the Copilot SDK revealed that **client lifecycle overhead (start/stop) added ~2.5 seconds per request**, with most latency not in model inference but in avoidable client lifecycle overhead [6]. Token generation was approximately **1% of total latency**, while time to first token dominated at **66% of total latency**. Reusing clients and sessions instead of starting and stopping them per request cut approximately 1.4 seconds of overhead each time [6].

Key findings include:
- "Client lifecycle management alone accounts for ~2.5 seconds of overhead per request" [6]
- "Lifecycle management often matters more than request-level optimization" [6]
- "Model class names are marketing abstractions, not performance guarantees" [6]
- "Perceived latency is a first-order UX metric. Treat it as such" [6]
- "Picking 'gpt-4.1' over 'gpt-5' saves ~40% on latency" [6]

Tan's research emphasizes that architecture and lifecycle management often impact performance more than the choice of model, and recommends implementing session pooling and reusing clients rather than creating sessions on-demand.

### 1.4 Tabnine's Hybrid Architecture

Tabnine employs a **hybrid architecture** combining local and cloud models, introduced June 1, 2022 [8][12]. The system uses a **"Team of Models"** approach: lightweight local models operate offline on the user's machine for instantaneous suggestions, while larger cloud models augmented by a vector database handle more accurate and longer predictions [8]. This architecture is designed to optimize between speed (local) and accuracy (cloud).

Key specification: Tabnine's Hybrid Model "combines the benefits of both cloud and local inference" [8]. New installations have the hybrid model enabled by default, though users retain full control to switch between cloud-only, local-only, or hybrid modes via the Tabnine Hub [8]. The company emphasizes that "Tabnine continues to place the highest value on your privacy, never storing nor sharing any of your code" [8].

A case study with CI&T (a global IT company with over 7,000 employees) reported that developers accept **90% of single-line coding suggestions**, resulting in an **11% productivity increase** across projects [12]. Tabnine operates on Google Cloud's scalable AI infrastructure, employing high-performance GPUs and Google Kubernetes Engine to deliver low-latency services [12].

### 1.5 Amazon Q Developer / CodeWhisperer Latency Characteristics

As of April 30, 2024, Amazon CodeWhisperer evolved into **Amazon Q Developer** [39][40][42]. Amazon does not publish specific latency SLAs or millisecond targets for code suggestions. However, the underlying **Amazon Bedrock** infrastructure supports **latency-optimized inference** featuring significant improvements [13][14][16].

At re:Invent 2024, AWS launched latency-optimized inference for foundation models in Amazon Bedrock. Benchmark results show:
- **Llama 3.1 70B optimized model:** Up to **97.10% reduction in TTFT P90** and up to **529.33% improvement in OTPS P90**, enabling up to 5x faster token generation [16]
- **Claude 3.5 Haiku optimized model:** Up to **42% reduction in TTFT P50** and up to **125.50% improvement in output tokens per second** [16]
- Under realistic cross-region testing conditions, up to **51.7% reduction in TTFT** [16]

Users can set the "latency" parameter to "optimized" while calling the Amazon Bedrock runtime API, though once usage quotas are exceeded, requests revert to standard latency [14]. The service is available in select AWS regions: US East (Ohio) and US West (Oregon) for Claude and Llama models.

The blog recommends a holistic, iterative approach combining multiple strategies: latency-optimized inference, prompt engineering, streaming responses, system architecture design considering geographic data flow, asynchronous processing, caching strategies such as prompt caching and intelligent prompt routing, and balancing model complexity with latency and cost [16]. Notably: "Consistent response times, even if slightly slower, often lead to better user satisfaction than highly variable response times with occasional quick replies" [16].

### 1.6 Taxonomy of Latency Regimes and Flow State Impact

The analysis of latency across local and cloud architectures reveals distinct regimes with measurable flow state implications:

| Latency Regime | Typology | Flow State Impact |
|----------------|----------|-------------------|
| 15–80ms | Local TTFT (RTX 5090, M4 Ultra, high-end consumer GPU) | Feels instantaneous; no perceptible pause |
| 180–600ms | Cloud TTFT (GPT-4.1, Claude Sonnet 4, Gemini 2.5 Pro) | Noticeable but acceptable for complete suggestions |
| 700–1,400ms | Frontier model cloud TTFT (GPT-5.5, Claude Opus 4.7) | Borderline for flow state; perceptible delay |
| 1,200–2,400ms | Cloud TTFT for complex generation (GPT 5.5 Instant M365 Copilot) | Disruptive for rapid completions |
| >2,500ms | Pre-optimization Copilot SDK; reasoning mode | Highly disruptive; breaks flow state |

Sources: [6][18][20]

**Tail Latency Considerations:** The Digital Applied AI Development Latency Benchmark report emphasizes that "Most teams design SLOs against P50, ship to production, and discover the P95 outliers ruin perceived UX. Anchor on P95 from day one" [20]. Tail latency inflates between 1.6 to 3.2 times over median latency, making P95-focused service level objectives essential.

**Regional Variance:** Users in Asia-Pacific face an additional **180–220 ms TTFT** compared to US-East users, urging provider deployment near user concentration for latency-sensitive applications [20].

**Reasoning Mode Classification:** "Reasoning mode is not a latency increment — it is a different latency category. Sub-2-second UX simply can't use it" [20]. Reasoning modes inflate TTFT dramatically by 5 to 30 times, making them unsuitable for interactive code completion but feasible for asynchronous or batch use cases.

### 1.7 Interruption Recovery Benchmarks

Foundational research establishes the massive cost of interruptions to developer flow, directly informing when and how AI suggestions should be presented:

- **Gloria Mark (UC Irvine, 2005):** "It takes an average of twenty-three minutes and fifteen seconds to return to the original task after an interruption" [31]. For a ten-engineer team losing two hours per day from distraction, this could cost nearly $375,000 annually [24].

- **Parnin & Rugaber (2010):** Analysis of 10,000 recorded sessions from 86 programmers found that **only 10% of sessions resumed programming activity within one minute** after interruption, and approximately **30% showed resumption delays of over 30 minutes** [25][26]. Developers visit many locations rapidly in "navigation jitter" when trying to recall a location. Only 7.5% of sessions involve editing without prior navigation; most require navigating multiple code locations before editing [26].

- **Duke & Vanderbilt (2025):** A study of twenty participants exposed to six types of interruptions found that **developers need 10–15 minutes to return to editing code** after interruption and **30–45 minutes to recover full context**; interruptions can **erase up to 82% of productive work time** when developers face frequent disruptions from meetings, messages, and quick questions [28]. Code writing tasks exhibited higher stress measures than comprehension or review tasks [28].

- **University of Technology Sydney Survey (141 developers):** **81% perceive task switching as equally disruptive as interruptions**; developers need at least 15 minutes to regain flow after interruption [29][37]. Daily stand-up meetings were frequently reported as disruptive, particularly when unplanned or poorly timed [29].

- **Shakeri Hossein Abad et al. (2018):** Analysis of 4,910 recorded tasks from 17 professional developers found that **self-interruptions were more disruptive than external interruptions**, contrary to developer beliefs [38]. Developers switch about two-thirds (59%) of their daily tasks, with 40% requiring context switching, and **29% of interrupted tasks never resumed** [38]. Afternoon interruptions cause more task fragments and longer resumption lags, with about half being self-initiated [38].

---

## 2. Acceptance Rates & Interruption Timing by Task Type

### 2.1 Proactive vs. On-Demand Delivery Timing

The **ProAIDE study** (JetBrains/Fleet IDE, accepted to IUI 2026) conducted a five-day in-the-wild study with 15 professional developers, capturing **229 AI interventions** across **5,732 interaction points**. This is the most rigorous field study available on proactive AI suggestion timing [1].

Key findings on timing:
- **At workflow boundaries (e.g., post-commit):** Achieved **52% engagement rates**—the highest engagement context
- **Mid-task interventions (e.g., on declined edit):** **Dismissed 62% of the time**, indicating that interruptions during active coding flow are poorly received
- Well-timed proactive suggestions required significantly less interpretation time than reactive suggestions: **45.4 seconds versus 101.4 seconds**, indicating enhanced cognitive alignment when suggestions arrive at natural workflow pauses [1]

One developer expressed: *"I separate writing code and optimizing it... Reviewing should happen after I've written something and feel it's ready for review"* [1]. Developers preferred lightweight, in-editor pop-ups to confirm chat invocation, noting that suggestion rejection often stemmed from task switching rather than disapproval [1].

The study established four design principles: (1) timely proactive AI assistance emphasizing anticipation without disruption, (2) contextually relevant suggestions targeting code quality, (3) explainability and transparency to build trust, and (4) preserving user control to balance AI autonomy [1].

The **Codellaborator study** (arXiv 2502.18658v4) compared three interface variants: a prompt-only system (user-initiated), a proactive agent without visual representation (CodeGhost), and Codellaborator (with visible AI presence and contextual interactions) involving 18 participants [11].

Results showed:
- **Proactive agents increase efficiency** compared to prompt-only paradigms
- However, they **also incur workflow disruptions**—a fundamental trade-off
- **Presence indicators** (AI caret, cursor, status indicators) and interaction context alleviated disruptions and improved users' awareness of AI processes
- Some participants "felt ambivalent to adopt highly proactive programming assistants, embracing efficiency but expressing concerns on maintainability and extendability of the code artifact" [11]
- The study operationalized **six design heuristics for timely intervention** based on: user inactivity, task boundaries, implicit signals like code comments, and code selection [11]

### 2.2 Acceptance Rates Stratified by Task Type

**Task-stratified PR acceptance analysis** (AIDev dataset, 7,156 PRs from 2026 study by Pinna et al.) revealed that **task type overwhelmingly influences acceptance rates**, more so than differences between AI agents [3]:

| Task Type | Overall Acceptance Rate | Best Agent | Best Agent Rate |
|-----------|----------------------|------------|-----------------|
| Documentation | **82.1%** | Claude Code | 92.3% |
| Feature Additions | **66.1%** | Claude Code | 72.6% |
| Bug Fixes | — | **Cursor** | **80.4%** |
| Fix (general) | — | OpenAI Codex | 83.0% |
| Refactor | — | OpenAI Codex | 74.3% |

Key quotes: "PR task type is a dominant factor influencing acceptance rates: documentation tasks achieve 82.1% acceptance compared to 66.1% for new features—a 16 percentage point gap that exceeds typical inter-agent variance" [3]. "No single agent performs best across all task types" [3]. The study recommends task-stratified evaluations become standard practice.

### 2.3 General Acceptance Rate Benchmarks Across Tools

Broad industry benchmarks reveal substantial variation across tools and study conditions:

| Tool | Acceptance Rate | Source |
|------|----------------|--------|
| **GitHub Copilot** | **27–30%** (rising to ~34% after 6 months) | [4][5] |
| **Tabnine** | **~24%** | Industry benchmarks |
| **CodeGeeX** | **~49%** (likely due to single-line recommendations) | Industry benchmarks |
| **JetBrains** (baseline, before filter) | **~33%** | [6] |
| **JetBrains** (after CatBoost filter) | **~50%** (boosted by ~50%) | [6] |
| **Amazon Q Developer** | Varies (BT Group 37%, National Australia Bank 50%) | [6] |
| **ZoomInfo Copilot case study** | **33%** suggestions, **20%** lines of code | [25] |
| **Overall: code generated** | **41% of all new code** is AI-generated | [7] |
| **Overall: actually accepted** | **Only 27–30% accepted** | [7] |

**The JetBrains CatBoost Filter Innovation:** JetBrains observed a significant increase in acceptance rates without retraining their core LLM. Instead, they implemented a **lightweight local filter model built with CatBoost**, running on users' machines with predictions in **1–2 milliseconds** and a compact **~2.5 MB file** [6]. The filter model boosted acceptance rate by **~50%** and cut explicit cancel rate by **~40%**, maintaining steady completion ratios [6]. As JetBrains summarized: "Even if your LLM is already doing a great job, there's always room for improvement. Sometimes, the smart use of extra data like logs can do the trick" [6].

The **SoftwareSeni analysis** highlights the paradox: "41% of professional code is now AI-generated, yet developers only accept 27-30% of AI suggestions in production code" [7]. Developers retain approximately **88% of accepted code** in final submissions [4].

### 2.4 Interruption Costs During Debugging vs. New Feature Development

The cost of interrupting developers during debugging is amplified because interruptions during peak cognitive load cause the biggest disruption. Key findings:

- An interrupted task is estimated to take **twice as long and contain twice as many errors** as uninterrupted tasks [10]
- On complex tasks: **15-30 minutes lost per interruption** [28]
- **Code writing interruptions** exhibit higher stress measures than comprehension or review tasks [28]
- The **METR study** found AI tools increased task completion time for **experienced developers by 19%** despite developers believing they were 20% faster—a 39% perception gap, particularly pronounced in **complex debugging and architectural decisions** [9][13]
- AI **accelerates repetitive and boilerplate tasks** but **hinders complex debugging and architectural decisions**, especially in large, complex codebases [9]
- **Management System Development** tasks saw a **400% increase in task completion rates** with ACAT assistance—the highest impact of any task type tested [5]

**Requirements Engineering tasks were found most vulnerable to interruptions** in the Shakeri Hossein Abad et al. study, with self-interruptions causing more task fragments and longer resumption lags than external interruptions [38].

### 2.5 Developer Preferences for Proactive vs. On-Demand Invocation

In the **"Bridging Developer Needs" study** (35 professionals interviewed, ICSE'26 SEIP), developers expressed strong needs for **proactive AI behavior** but also for **user control** and **context awareness** [18]. Key quotes:

- A44 (Adopter): *"It would be cool if it could search for problems along the way. It could say 'Did you realize you just created an out-of-bounds condition?'"* [18]
- C159 (Churner): *"If I felt the AI system was really reliable, it would have saved me time because I wouldn't need to double-check and could just apply the right solution right away"* [18]

The study revealed a **"conservatism gap"** : "Features that are straightforward to implement are correctly anticipated and realized, while proactive and maintenance-related features are underestimated despite user interest, revealing a gap between demand and perceived feasibility" [18]. Proactive features such as automated documentation, debugging, refactoring, and real-time optimization were consistently underestimated and remain largely unaddressed.

**The Stack Overflow 2025 Developer Survey** (49,000+ developers) provides large-scale preference data:
- 84% of developers use or plan to use AI tools
- 46% actively **distrust** the accuracy of AI tools (up from 31% in 2024)
- Only **33% trust** the accuracy of AI tools
- Only **3% "highly trust"** the output
- The biggest frustration, cited by **66% of developers**, is "AI solutions that are almost right, but not quite" [4]
- **75%** of developers would still ask a person for help when they don't trust AI's answers [4]

---

## 3. Developer Experience Levels: Junior vs. Senior

### 3.1 The "Experience Paradox" in Trust Calibration

A key finding across multiple studies reveals what researchers term the **"Experience Paradox"** : junior developers exhibit **78% trust in AI specificity** compared to **39% for senior engineers** [1][2]. This creates a situation where less experienced developers are more reliant on AI, potentially impairing their growth in architectural judgment and system design.

The **Fastly July 2025 survey** of 791 professional developers provides detailed context on how experience shapes AI usage [2][3]:

- **Senior developers (10+ years) ship approximately 2.5 times more AI-generated code** than junior developers (0-2 years)
- **32% of seniors** report over half of their code is AI-generated, compared to **13% of juniors**
- **59% of senior developers** say AI tools help them ship faster [3]
- However, nearly **30% of seniors** report spending enough time fixing AI-generated code to offset most time savings, compared to **17% of juniors** [3]

Austin Spires, Fastly's senior director of developer engagement, explained: "Senior developers probably have a better understanding of how to build requirements, how to write prompts, how to work with those tools" [3]. A junior developer respondent noted: "It's always hard when AI assumes what I'm doing and that's not the case, so I have to go back and redo it myself" [3].

### 3.2 Productivity Gains by Experience Level

The definitive study on this topic is **"The Effects of Generative AI on High-Skilled Work: Evidence from Three Field Experiments with Software Developers"** (Peng, Demirer et al., MIT/Microsoft/Princeton/UPenn, February 2025). The study involved 3 randomized controlled trials with **4,867 developers** across Microsoft, Accenture, and a Fortune 100 electronics company [5][6].

Key findings:

| Finding | Junior/Less Experienced | Senior/More Experienced |
|---------|------------------------|------------------------|
| Productivity gain (pull requests) | **27% to 39%** | **8% to 16%** |
| Adoption rate | Higher | Lower |
| Acceptance of AI suggestions | More frequent | More selective |

- Combined analysis shows a **26.08% increase** (SE: 10.3%) in completed software development tasks among Copilot users
- Secondary outcomes: **13.55% increase in commits**, **38.38% increase in builds** (compilations)
- **No significant negative effects on code quality** as measured by build success rates
- **30-40% of engineers did not try Copilot**—access alone does not ensure adoption [5]

The researchers note: "Less experienced developers accept AI-generated suggestions more frequently than do their more experienced counterparts," raising the need for education rather than usage restriction [6].

### 3.3 Microsoft's 3-Week Randomized Controlled Trial and Trust Evolution

Microsoft conducted a three-week randomized controlled trial involving **over 200 engineers** to study the impact of GitHub Copilot on productivity and perceptions [7].

**Trust evolution findings:**
- **Initially**, only **20% trusted the code** and **30% believed tools were reliable**
- **65% believed AI couldn't replace them**
- **After three weeks**, users reported significantly higher enjoyment and perceived usefulness, viewing generative AI as inevitable and future-oriented
- Copilot was used in **56% of daily coding sessions**
- Primary uses: boilerplate code, documentation, comments, and API exploration
- **Telemetry data showed no significant productivity gains in this short period**—the study may have been "too short for learning and getting proficient with a new tool" [7]
- **Developers with prior experience were significantly more likely to view Copilot as useful and enjoyable** [7]

One key observation: "Code can now be generated more quickly than ever before, but it is 'faster with more mistakes' and therefore requires a greater degree of critical analysis" [7]. The study suggests productivity gains may require approximately **11 weeks** for adoption and learning.

### 3.4 Barke et al.'s Bimodal Interaction Framework

The **"Grounded Copilot"** study by Barke, James, and Polikarpova (UC San Diego, 2023, 772 citations) presents the first grounded theory analysis of how programmers interact with AI code-generating models, based on observing 20 participants solving tasks across Python, Rust, Haskell, and Java [7][11].

Two distinct interaction modes were identified:

**Acceleration Mode** (predominantly used by experienced developers who have clear mental models):
- The programmer knows what to do next and uses Copilot to speed up coding **without breaking flow**
- Accepts mostly short, line-level suggestions accepted almost instantly
- Enters after decomposing tasks into microtasks
- Validates suggestions using quick pattern matching
- Long, multi-line suggestions often break programmer flow and are dismissed
- Participant quote: *"I think of Copilot as an intelligent autocomplete... I already have the line of code in mind and I just want to see if it can do it, type it out faster than I can"* [7]

**Exploration Mode** (more common for junior developers or unfamiliar tasks):
- The programmer is unsure how to proceed and uses Copilot to explore options through natural language comments
- Reviews multiple suggestions via Copilot's multi-suggestion pane
- Validates generated code more deliberately through execution, static analysis, or thorough code examination
- Participant quote: *"Copilot feels useful for doing novel tasks that I don't necessarily know how to do. It is easier to jump in and get started with the task"* [7]
- Participants switch fluidly between modes throughout tasks depending on task familiarity and complexity

The **"Taking Flight with Copilot"** (CACM, 2023) adds: "As AI-powered tools are integrated into more software development tasks, developer roles will shift so that more time is spent assessing suggestions related to the task than doing the task itself" [12].

### 3.5 The "Productivity Hallucination" / Productivity Paradox

Multiple studies document a disconnect between perceived and actual productivity:

- **METR study (July 2025):** Developers using AI took **19% longer** to complete tasks but believed they were **20-24% faster**—labeled a "productivity hallucination" [13][14]

- **Reasons for the slowdown:** Over-optimism about AI benefits, AI's lack of deep context understanding for complex codebases (averaging 1.1 million lines), low acceptance rate of suggestions (less than 44%), and developers spending approximately **9% of their time just reviewing and cleaning AI-generated outputs** [13]

- **Context-dependent nature:** AI aids best in **boilerplate code, unfamiliar areas, documentation, test generation, and simple tasks** but struggles with **complex systems, security-critical code, and scenarios where developers have deep knowledge** [13]

- **Faros AI's longitudinal analysis** of telemetry data from over 10,000 developers across 1,255 teams found that AI adoption enables handling more concurrent tasks, with developers interacting with 9% more tasks and 47% more PRs daily, leading to 21% more tasks completed and 98% more PRs merged. However, this increased output did not translate to faster task completion because downstream processes like code review and testing become bottlenecks [14]

- The **Cognitive Biases in LLM-Assisted Software Development** paper (arXiv 2601.08045, January 2026): **48.8% of total programming actions are biased**; LLM interactions account for **56.4% of biased actions**; LLM-related actions are statistically more likely to be biased (53.7%) and more likely to be reversed (29.4%) [12]

### 3.6 The Comprehension Gap

Anthropic conducted a randomized controlled trial with **52 mostly junior Python developers** to evaluate AI assistance impact on coding skills, finding a **17-point comprehension gap** [19][20].

**Core finding:**
- Developers who use AI coding assistance **score 17 percentage points lower on code comprehension tests** compared to developers who code by hand — **50% vs. 67% (p=0.01)** [19][20]
- The largest skill gap was in **debugging**—the ability to identify and diagnose errors in code [20]
- Participants in the AI group did not even save significant time, finishing only about two minutes faster on average, and spending up to **30% of their time composing queries** [20]

**How developers use AI matters enormously:**
- **Low scorers (below 40%):** Those who delegated code generation entirely to AI or used AI mainly for debugging—resulting in poor understanding [19][20]
- **High scorers (65%+):** Those who used AI to generate code but **actively sought explanations and engaged with concepts**—leading to better comprehension and faster task completion [19][20]
- The key distinction: using AI for **conceptual guidance** (asking "why") vs. using AI to **delegate code generation** (asking "what") [20]

Core quotes: "How you use AI matters more than whether you use it" [20]. "Use AI to explain, not just to generate. Ask why, not just what. Understand before you ship" [20].

### 3.7 JetBrains Survey Data on Experience-Level Benefits

**JetBrains State of Developer Ecosystem 2025** (24,534 respondents, 194 countries):
- **85% of developers regularly use AI tools** for coding
- **62% rely on at least one AI coding assistant**
- Nearly **nine out of ten developers save at least one hour weekly** using AI, with **one in five saving eight hours or more** [21][22]
- **68% expect employers to require AI tool proficiency soon** [21]
- 15% remain hesitant due to concerns about code quality, security, and skill impact [21]

**JetBrains AI Pulse Survey (January 2026)** (10,000+ professional developers):
- **90% of developers regularly use at least one AI tool** at work
- **74% had already adopted specialized AI tools** for developers
- **GitHub Copilot leads adoption** at **29% workplace adoption** (76% awareness), though growth has plateaued
- **Cursor and Claude Code** both at **18% workplace adoption**
- **Claude Code** shows rapid gains: 57% awareness, 18% adoption globally, **24% adoption in the US and Canada**, and the **highest customer satisfaction (91%)** and net promoter score (54) among AI coding tools [23]

---

## 4. Explanation Availability & Trust Calibration

### 4.1 Foundational Trust Research in AI Code Generation

The **Microsoft Research FAccT 2024 paper** (Wang, Cheng, Ford, Zimmermann)—"Investigating and Designing for Trust in AI-powered Code Generation Tools"—is the most comprehensive study on this topic [1][2][3]. Through interviews with 17 developers experienced with GitHub Copilot and Tabnine, the study found:

- Developers' trust is rooted in the AI tool's perceived **ability, integrity, and benevolence**, and is **situational**, varying according to the context of usage
- **Ability:** Practical benefits and time saved
- **Benevolence:** Alignment with developers' short- and long-term goals, learning aspirations, and career aspirations
- **Integrity:** Understanding model mechanisms and privacy assurances
- Three main challenges: **building appropriate expectations, configuring AI tools, and validating AI suggestions**
- "Without proper support, developers can find it challenging to form accurate mental models of what AI tools can do or not, or determine the quality of specific AI suggestions; thus becoming vulnerable to **over- or under-trusting the AI**" [1][2][3]

A design probe study with 12 developers tested three sets of design concepts:
1. **Usage statistics dashboards** presenting personalized AI performance metrics—participants found this helpful for aligning expectations with AI's actual ability [1]
2. **Quality indicators** providing confidence levels of AI suggestions at solution, token, and file levels—participants noted interpretation challenges, cautioning that numerical confidence scores "might bias users toward over-reliance or premature rejection of valid suggestions" [1]
3. **Control mechanisms** enabling developers to set intentions and preferences for AI behavior at project initialization and during sessions—developers expressed that "the ability to set a boundary and have it respect that boundary is the core of building trust" [1]

### 4.2 Trust Dynamics in Practice

The **Amazon Science study** (Sabouri et al., ICSE 2025) on "Trust Dynamics in AI-Assisted Development" provides quantitative and qualitative evidence [5][6]:

- **48% of AI code suggestions initially accepted were later altered or removed**, frequently because of **blind trust or misjudged correctness**
- Developers prioritize **correctness and comprehensibility** when evaluating code suggestions, but often rely on **proxy characteristics** in practice due to lack of real-time support for assessing trustworthiness
- Nine out of ten participants reported trust grew after positive experiences; negative experiences did not significantly impact trust

Four validated guidelines emerged:
1. **Enhance comprehensibility** of AI suggestions
2. **Signal perceived correctness** transparently
3. **Support dynamic trust re-evaluation** as context changes
4. **Enable developer-control alignment** through configurable behavior

### 4.3 Forms of Explanation and Their Differential Impact

The **UC Berkeley study** on "Calibrating Trust in AI-Assisted Decision Making" (Turner et al.) provides critical findings on explanation types through empirical experiments where participants performed a high-uncertainty decision task categorizing plants using AI recommendations [4][7]:

- **Confidence scores** helped calibrate trust and led to **higher accuracy** than other explanation types. They serve as "a more direct representation of reliability" [4]
- **Data availability explanations** (local explanations showing what data informed the decision) **reduced trust calibration**, leading to increased but often incorrect switching to AI recommendations. These explanations "unexpectedly decreased accuracy and increased users' tendencies to over-rely on AI" [4]
- When users had **low self-competence**, data availability explanations amplified the tendency to switch responses, further diminishing accuracy
- "Explanations do not always aid trust calibration, and can actually hurt it, especially in the face of novice users who have low self-competence" [4]

The **Google PAIR Guidebook** on Explainability + Trust emphasizes:
- "Help users calibrate their trust" by clarifying AI abilities, data sources, and situational stakes
- "Displaying model confidence can sometimes help users calibrate their trust and make better decisions, but it's not always actionable"
- "The process to build the right level of trust with users is slow and deliberate"
- "Partial explanations intentionally leave out parts that are unknown, highly complex, or simply not useful" [5]

### 4.4 Automation Bias and Overreliance

The **Microsoft Overreliance on AI Literature Review** (Passi & Vorvoreanu, June 2022) synthesized approximately **60 interdisciplinary papers** to understand overreliance [8][9]:

- **Overreliance** defined as "users accepting incorrect AI recommendations—i.e., making errors of commission"
- Users with **low AI literacy** are most affected by AI recommendations
- **Automation bias** causes users to favor automated recommendations and disregard non-automated information (Logg et al. 2019)
- **Detailed explanations often lead users to develop overreliance** despite the risk of trusting incorrect recommendations (Bussone et al. 2015)
- **Confirmation bias**, **ordering effects**, and **overestimation of AI explanations** all contribute to overreliance

Mitigation strategies recommended:
1. **Effective onboarding** that transparently communicates AI strengths and limitations, showing examples of both correct and incorrect recommendations
2. **Personalized adjustments** based on user traits like confidence and AI literacy
3. **Cognitive forcing functions** to stimulate analytical thinking—"Cognitive forcing functions significantly reduce overreliance on incorrect AI recommendations by prompting user engagement and reflection" (Buçinca et al. 2021) [8]
4. **Real-time feedback** and tailored explanations
5. **Adjusting AI response speed** to encourage reflection
6. **Giving users choice** regarding AI recommendations

As Mihaela Vorvoreanu noted: "In many cases it's our job to decrease trust in AI, not to increase it - to create *appropriate* trust" [8].

### 4.5 The BSI/ANSSI Joint Report on Security Risks

The German Federal Office for Information Security (BSI) and French Cybersecurity Agency (ANSSI) jointly published a report (September 2024) with alarming findings [10][11]:

- "AI coding assistants are no substitute for experienced developers"
- "An unrestrained use of the tools can have severe security implications"
- About **40% of AI-generated programs contained security vulnerabilities**
- **19.7% of imported packages** in AI-generated code were **hallucinated**, leading to supply chain attack risks—a study at University of Texas at San Antonio found GPT models had a 5.2% hallucination rate compared to 21.7% for open-source models [11]

Risk categories identified:
1. Exposure of sensitive input data due to provider policies
2. Variable and often low quality of generated source code with common security vulnerabilities
3. Novel attack vectors: package hallucination, indirect prompt injections, data and model poisoning, attacks facilitated by malicious extensions
4. Automation bias and over-reliance on AI outputs without critical review

Mitigation strategies: restricting uncontrolled cloud access, performing risk assessments, enforcing strict data confidentiality policies, scaling quality assurance teams proportionally with AI-generated code output, rigorous code review and testing, whitelisting packages, sandboxing development environments, and awareness training for employees [10].

### 4.6 Factors Influencing Trust in Code Completion

The **Google Research AIware 2024 paper** (ACM SIGSOFT Distinguished Paper Award) on "Identifying the Factors that Influence Trust in AI Code Completion" combined qualitative interviews with over **1 million logged code completion suggestions from 59,000 developers** at Google [12][14]:

Three main factors emerged:

1. **Characteristics of AI suggestions:** Quality and length—better model quality scores strongly predict higher acceptance, with a **23.1% increase in odds per standard deviation increase** in model quality score
2. **Characteristics of developers:** Language expertise and tool familiarity—developers with **readability in a language** (expertise certification) were **30.5% more likely** to accept suggestions in that language
3. **Development context:** Suggestions in test files were **20.8% less likely** to be accepted; priority of work did not significantly predict acceptance

"The strongest positive predictor of accepting a multi-line code suggestion was whether or not the developer had been awarded readability in the language of the suggestion" [12][14]. The study underscores the need for personalization and user control to help developers build an appropriate level of trust.

### 4.7 On the Need to Rethink Trust in AI Assistants

The critical review paper "On the Need to Rethink Trust in AI Assistants for Software Development" (arXiv 2504.12461v3) argues that SE research routinely equates trust with acceptance of AI-generated content, which oversimplifies trust and neglects its complexity [26]. Key arguments:

- "Trust is **rarely defined or conceptualized** in SE articles"
- "Trust is a **multidimensional construct** centered around trusting beliefs and trusting intentions"
- "People's behavior can be influenced not only by the perceived trustworthiness of the AI but also by other factors like situation and task"
- "**Ill-calibrated trust** may lead to disuse or misuse of a system"
- "Without a scientifically validated trust model, trust becomes **an empty marketing concept or, even worse, a concept used for ethics washing**" [26]

The authors advocate adopting established trust models from foundational disciplines: Mayer et al. (1995) ability-benevolence-integrity framework, Lee and See (2004) trust calibration model, and philosophical perspectives on preconditions and justification of trust.

---

## 5. Code Quality Outcomes

### 5.1 The Downstream "Productivity Tax"

Research reveals that AI's speed gains come with hidden downstream costs across multiple dimensions:

**GitClear's 2025 AI Copilot Code Quality Report** (211 million changed lines of code from Google, Microsoft, and Meta 2020-2024):
- **Refactoring collapsed from 25% to under 10%** of changed lines (2021-2024), meaning the proportion of "moved" (refactored/reused) code lines declined dramatically [1][2]
- **Code duplication rose from 8.3% to 12.3%** in the same period—2024 marked the first year where "Copy/Pasted" lines exceeded "Moved" lines [1][2]
- **Eightfold increase in duplicate code blocks** in 2024 compared to two years prior [1]
- AI promotes "copy/pasted" code rather than modular, DRY practices [1][2]
- **91% increase in PR review time**—AI-assisted pull requests require significantly more review time [1][2]
- **Google DORA correlation:** DORA's 2024 survey projects a **7.2% decrease in delivery stability** for every 25% increase in AI adoption. The DORA report states: "AI adoption brings some detrimental effects. We have observed reductions to software delivery performance" [1]

The report warns: "As we 'AI enable' programmers with tab-completed code suggestions, we're going to see reams of code generated by AI instead of well-maintained codebases" [1].

**CodeRabbit analysis** of 470 GitHub PRs (320 AI-co-authored, 150 human-only) found AI-generated code contains [6][7][8]:
- **1.7 times more issues overall** (10.83 findings per 100 PRs vs. 6.45 in human submissions)
- **Logic and correctness errors 75% more common**
- **Readability problems 3 times higher**
- **Security vulnerabilities 2.74 times greater** than human-written code, including 2.5x more vulnerabilities rated CVSS 7.0 or above
- **Performance inefficiencies nearly 8x more often**
- **Error handling gaps nearly double**
- **60% of AI code faults are "silent failures"** —compile and pass tests but produce incorrect results in production. An example: Amazon's March 2026 incident where AI code corrupted delivery estimates, leading to 6.3 million orders lost in six hours [8]

AI-generated code has **41% higher churn rate** compared to human-written code [2][15].

### 5.2 Microsoft Research Productivity Studies

**Microsoft's "AI and Productivity" Report (First Edition, December 2023):**
- Across more than 30 studies, Copilot demonstrated productivity gains: users completed tasks in **26% to 73% of the time** without Copilot
- Perceived daily time savings **averaging 14 minutes**
- **70% of respondents** believed Copilot increased their productivity
- Users with Copilot reported **16% lower agreement** with "This task was a lot of effort" and 83% agreement that Copilot reduced effort
- In some complex tasks, LLM-based tools may **trade speed for marginally lower quality** [9][10]

**Google's Randomized Controlled Trial (96 engineers, arXiv 2410.12944):**
- AI assistance significantly decreased completion time by approximately **21%** (96 minutes with AI vs. 114 minutes without)
- Developers who spend more hours coding daily and those of higher seniority tend to complete tasks faster
- Developers coding five or more hours daily were 32% faster regardless of AI use and potentially benefited more from AI [12][13]
- Neither seniority nor prior AI experience significantly predicted speed improvements [12][13]

### 5.3 Time-Warp Study: Workweek Alignment

Microsoft Research's **Time-Warp Developer Productivity Study** (484 developers, 2024) found:
- Developers spend **more time than preferred on communication, security, debugging, and support tasks**, and less on core activities (coding, system design, documentation, learning)
- **Better alignment between actual and ideal workweeks leads to higher productivity and satisfaction**
- **Daily AI users report highest productivity**—83.7% reporting to be "productive" [15]
- Developers expressed greatest interest in automating: documentation creation/updating, environment setup, test authoring, task management, security/compliance
- **Notably, debugging was not among the top automation desires** [15]
- Over-allocation of communication, development environment maintenance, and security tasks negatively impacts productivity and satisfaction [15]

### 5.4 JetBrains HAX Team's Two-Year Study

The **JetBrains Human-AI Experience (HAX) Team study** (2024-2026) analyzed anonymized telemetry data from **800 developers** (400 AI users, 400 non-users) plus surveys and interviews with 62 professionals, presented at ICSE 2026 [16][17][18]:

**Five workflow dimensions analyzed:**

1. **Productivity:** AI users consistently typed approximately **600 extra characters per month** compared to non-users (587 characters increase for AI users vs. 75 for non-users). **82.3% of respondents** reported AI tools increased productivity [16][17].

2. **Code quality:** No significant behavioral change in debugging instances was observed among AI users, though nearly half perceived slight improvements. Telemetry data show stable debugging activity among AI users, contrasting with a decline among non-users [16][17].

3. **Code editing:** AI users produced substantially more code but also **deleted significantly more** code, indicating heightened editing activity. Developers spend **over 50% of their time evaluating and editing AI-generated output** [16][17].

4. **Code reuse:** Remained relatively stable over time, with both groups showing no large change [17].

5. **Context switching:** **Paradoxically increased for AI users**, contrary to expectation that in-IDE AI would reduce it. **74% of developers surveyed said context switching hadn't increased**, but telemetry on **151 million IDE window activations** revealed a hidden rise [16][17].

The study highlights an important distinction: **verification switches** (checking AI output, beneficial) and **lost-thread switches** (productivity loss). However, telemetry counts both the same without differentiation [18]. The overall conclusion: "AI redistributes and reshapes developers' workflows in ways that often elude their own perceptions" [16].

### 5.5 The "70% Problem" and LLM Confidence

SoftwareSeni's analysis identifies the **"70% Problem"** : AI code is approximately **70% correct** but requires significant human effort to debug, refine, and complete the remaining 30% [3]. Key findings:

- The **"productivity placebo effect"** : Instant code generation triggers dopamine responses, creating a sense of progress without delivering faster task completion [3]
- **89% of engineering leaders** report improved developer productivity, while **88% report increased developer satisfaction**, yet **81% observed developers spending more time on manual tasks** like code review [4]
- Approximately **31% of developer time** is now spent on tasks such as reviewing AI-generated code, fixing bugs, and switching between tools [4]
- **Only 6%** believe current measurement frameworks are sufficient to capture AI's impact on productivity [4]

The **"LLM Confidence in Code Completion" study** (arXiv 2508.16131) exploring code perplexity across **14 programming languages** found:
- **Strongly-typed languages (Java, C#, Go)** exhibit lower perplexity (higher model confidence), while dynamically typed languages (Shell, Perl) show higher perplexity [19][20]
- Larger models achieve higher performance but also higher Effective Calibration Error (ECE at 0.247 for Code Llama 70B), indicating a **trade-off between accuracy and calibration**—larger models are more performant but also more overconfident [20]
- The paper title derives from the finding that LLMs exhibit systematic overconfidence in easy scenarios while being poorly calibrated in harder scenarios [19][20]

---

## 6. Practical Design Recommendations

### 6.1 Latency Targets by Suggestion Type

Based on the accumulated evidence, the following latency thresholds are recommended:

| Suggestion Type | Target Latency | Rationale |
|----------------|---------------|-----------|
| Single-line autocomplete | **<200ms P50, <500ms P95** | Must match or exceed local IDE autocomplete; Copilot's proven target; anchor on P95 from day one |
| Multi-line suggestions | **200-600ms P50, <1,500ms P95** | Acceptable cloud TTFT range; longer suggestions justify slightly higher latency |
| Complex code generation (comments→code) | **<2,500ms** | Longer generations allow higher latency; cloud throughput advantage at >200-300 tokens |
| Explanations/rationales | **<1,000ms** | Should not delay suggestion display; can be loaded asynchronously |
| Asynchronous batch analysis | No real-time constraint | Can leverage reasoning mode without UX impact |

Sources: [4][5][6][18][20]

**Implementation guidance:**
- **Design SLOs against P95 from the start**, not P50. Tail latency inflates 1.6-3.2x over median and P95 outliers ruin perceived UX [20]
- **Prioritize local inference for autocomplete-heavy workflows.** "Any developer whose workflow is autocomplete-heavy, which describes most coding patterns, should prioritize local inference for the latency characteristics alone" [18]
- **Use cloud for longer generations** where the crossover point (>200-300 tokens) gives cloud throughput advantage [18]
- **Implement session pooling and client reuse** to avoid lifecycle overhead. Kevin Tan showed this cuts approximately 1.4 seconds per request [6]
- **Use streaming to improve perceived responsiveness** even when total latency is unchanged [6]

### 6.2 When to Show Suggestions Proactively vs. On-Demand by Task Type

Based on the ProAIDE, Codellaborator, and Bridging Developer Needs studies:

**Show suggestions proactively at:**
- **Workflow boundaries** (post-commit, after file save, on file open): 52% engagement demonstrated by ProAIDE study [1]
- **During boilerplate generation**: Known patterns with high acceptance (management system development saw 400% completion rate increase) [5]
- **During documentation generation**: 82.1% acceptance rate [3]
- **During idle periods** (user stops typing for >3 seconds): Low disruption risk [11]
- **When implicit signals suggest intent** (writing comments, opening new files, cursor in empty function body) [11]

**Require on-demand invocation for:**
- **Mid-debugging sessions**: 62% dismissal rate for proactive suggestions during active coding [1]
- **During complex architectural decisions**: AI tools can hinder experienced developers on complex tasks [9]
- **During code review**: Developers explicitly separate writing from reviewing [1]
- **When user has just declined a suggestion**: Repeated interruptions after rejection have near-zero acceptance [1]
- **When developer is actively navigating multiple code locations** (navigation jitter pattern indicating recovery from interruption) [26]

**Design principles for timing:**
- Use **implicit signals**—cursor position, typing speed, navigation patterns, and task context—to determine when to be proactive [11]
- Codellaborator suggests triggering based on: user inactivity, task boundaries, implicit signals like code comments, and code selection [11]
- Proactive agents should have **visible presence indicators** to alleviate disruption and improve user awareness of AI processes [11]
- As Austin Z. Henley argues: "Why aren't the commercial tools in this space pushing harder on the user experience? They seem to be taking the shotgun approach: more suggestions in more places and more often. But I don't want more suggestions!" [12]

### 6.3 Adaptation by Developer Seniority

**For Junior Developers (2-5 years):**
- **More structured proactive suggestions** with confidence indicators to build appropriate trust—they exhibit 78% trust in AI specificity and are more susceptible to over-trust [1][2]
- **Explanation availability is critical** but should emphasize confidence scores rather than data availability explanations (which can reduce calibration) [4]
- **Need structured training** on AI limitations (Microsoft recommends onboarding showing examples of both correct and incorrect recommendations) [8]
- **More frequent nudges toward verification and comprehension checking**—using AI for conceptual guidance ("why") rather than delegation ("what") [19][20]
- **Shorter suggestions** to prevent comprehension gaps and anchoring bias. Long, multi-line suggestions break programmer flow and are dismissed in acceleration mode [7]
- **Designed for exploration mode** (Barke et al.): natural language prompts, multi-suggestion pane, deliberate validation support [7]

**For Senior Developers (10+ years):**
- **More on-demand invocation preferred**—they have clear mental models and benefit from acceleration mode [7]
- **Less frequent proactive suggestions** to avoid disrupting flow; they already know what to do and just need AI to execute faster [7]
- **Richer explanation options** (code rationale, source attribution) for the 39% who demonstrate selective trust [1]
- **Confidence scores at token/file level** for efficient validation—they have the expertise to interpret and act on uncertainty information [1][12]
- **Control mechanisms** to set context and preferences (role settings, context sliders, project initialization panels) [1]
- **Support for decomposition into microtasks**—experienced developers who can break problems into small units benefit most from acceleration mode [7]

### 6.4 Explanation Design Principles

Based on the Microsoft FAccT 2024 paper, UC Berkeley calibration study, Amazon Science research, and Google AIware study:

1. **Use confidence scores, not data availability explanations:** Confidence scores help calibrate trust; data availability explanations can actually reduce trust calibration, especially for novice users [4]

2. **Provide quality indicators at multiple granularity levels:** Solution-level confidence scores, token-level uncertainty highlights, and file-level familiarity signals allow developers to assess trustworthiness at appropriate abstraction levels [1]

3. **Design for engagement, not bypass:** Prevent users from skipping explanations through attention guidance and appropriate friction. However, note that design concepts that reduce overreliance most may receive the least favorable subjective ratings [8][22]

4. **Balance control complexity without overwhelming:** Developers value control but can be overwhelmed—design adjustable transparency rather than always-visible explanations [1]

5. **Support training and learning:** Build users' mental models of AI behavior over time through progressive disclosure. The "ability to set a boundary and have it respect that boundary is the core of building trust" [1]

6. **Avoid numerical overconfidence:** Participants cautioned that numerical confidence scores "might bias users toward over-reliance or premature rejection of valid suggestions" [1]. Consider using qualitative indicators (high/medium/low) alongside quantitative scores

7. **Personalize explanations based on user expertise:** Novice users benefit differently from different explanation types than experts. Confidence scores are more universally helpful [4]

### 6.5 Mitigating Automation Bias Through Cognitive Forcing Functions

To combat the 48.8% bias rate in LLM-assisted development [12], multiple mitigation strategies are recommended:

1. **Cognitive forcing functions (CFFs):** Disrupt fast, intuitive reasoning to promote deliberate analysis. Three types proven effective:
   - **On-demand explanation requests** requiring users to engage with AI reasoning before accepting
   - **Decision update after initial choice** (requiring users to make initial decision before seeing AI recommendation)
   - **Forced waiting** (imposed delay before receiving AI recommendation) [22][25]

2. **Slow-down mechanisms:** Adjust AI response speed for complex tasks to encourage reflection [8]. Note that CFFs "consistently reduce overreliance relative to XAI-only explanation approaches" but "people assigned the least favorable subjective ratings to the designs that reduced overreliance the most" [22][25]

3. **Dual-model verification:** Split workflow between two models—one drafts, one reviews. Microsoft's Critique and Council features implement this approach [11]

4. **Transparency about limitations:** Disclose model limitations during onboarding to calibrate trust. "Effective onboarding should show examples of both correct and incorrect recommendations to help developers develop appropriate first impressions" [8]

5. **Quality validation workflows:** Integrate testing support and static analysis verification alongside suggestions. Automate the detection of "silent failures" (60% of AI code faults) [8]

6. **Differentiated CFF application:** CFFs benefit users higher in Need for Cognition (NFC) more, creating "intervention-generated inequalities." Adaptive CFFs that adjust to user traits may be necessary [22]

7. **Managing AI response time** to encourage reflection: Slowing recommendations for complex tasks can nudge developers toward more careful evaluation [8]

8. **Offering users choices** about when to receive AI recommendations: on-demand systems show less automation bias than always-on proactive systems [8]

### 6.6 Enterprise Deployment Considerations

- **Hybrid architectures (Tabnine model)** offer the best latency balance: instantaneous local suggestions for simple completions, cloud-based for complex generations [8][18]. The crossover point (~200-300 tokens) is "the single most important number for deciding how to split workloads" [18]

- **Global deployment** with smart routing and failover (Copilot model) is essential for enterprise-scale latency. "If a region is down or overloaded, traffic just flows somewhere else" [5]. Regional latency varies by 180-200+ ms [20]

- **Privacy-preserving local inference** for sensitive codebases, with cloud augmentation for accuracy. "Ollama supports true air-gapped operation with no telemetry, no phone-home behavior, and full offline capability" [18]

- **Metrics to track beyond acceptance rate:** Time-to-task-completion, code churn rate, rework percentage, comprehension assessments, flow state disruption frequency, PR review time changes, and organizational DORA metrics (deployment frequency, lead time, mean time to recovery, change failure rate)

- **Security architecture:** Restrict uncontrolled cloud access, enforce data confidentiality policies, whitelist packages, sandbox development environments, and scale quality assurance teams proportionally with AI-generated code output [10]

- **Organizational adoption strategy:** Recognize the J-curve of AI adoption (excitement → learning dip → mastery over 3-6 months) [16]. Productivity gains may require approximately 11 weeks for full adoption [7]

- **Inference architecture selection:** Consider "training might happen in the cloud... but something unexpected is happening with inference. It's moving back on-prem and out to the edge" [23]. Benefits include predictable costs, reduced latency, enhanced data control and compliance

---

## 7. Research Gaps

The literature reveals several significant gaps where evidence is thin or inconclusive:

1. **Specific latency thresholds for different suggestion modalities:** While Copilot's 200ms target exists, no comparable published data exists for Tabnine or CodeWhisperer/Q Developer. There is no published research directly comparing different latency regimes (50ms vs. 500ms vs. 2,000ms) and their impact on flow state in controlled experiments. The taxonomy of latency regimes needs empirical validation with developer mental models.

2. **Acceptance rate norms at moment-by-moment granularity:** While the AIDev study provides PR-level data, there is limited granular data on inline suggestion acceptance rates specifically during debugging versus feature development at the moment-by-moment interaction level. The ProAIDE study's 5,732 interaction points provide initial data, but larger-scale telemetry data exploring acceptance timing within individual task contexts is needed.

3. **Longitudinal effects on junior developer skill development:** The comprehension gap (17 points) from Anthropic is documented cross-sectionally, but there are no long-term studies (>12 months) tracking how AI-dependent junior developers develop architectural judgment, debugging skills, and system design capabilities. The METR follow-up experiment noted that selection effects are already emerging—developers now refuse to work without AI [18].

4. **Explanation effectiveness in production environments:** The trust calibration research is primarily from lab studies or design probes. There are no large-scale A/B tests comparing different explanation modalities (confidence scores vs. data availability vs. hybrid approaches) in production IDE environments across diverse developer populations.

5. **Proactive vs. on-demand trade-offs at scale:** The ProAIDE study (15 developers) and Codellaborator study (18 participants) provide initial evidence, but larger-scale studies with more diverse task types and developer populations are needed. The ProCodeBench benchmark shows "existing proactive coding assistants perform poorly on real-world data, suggesting simulation-based benchmarks overestimate abilities" [10].

6. **Interaction between latency and trust:** No studies directly examine how different latency regimes (50ms vs. 500ms vs. 2,000ms) affect trust calibration and suggestion adoption behavior. Does faster inference increase trust (fluency effect) or decrease it (suspicion about quality)?

7. **Absence of independent head-to-head comparisons:** There is a notable absence of independent, peer-reviewed, head-to-head comparisons of Copilot, Tabnine, and CodeWhisperer/Q Developer on the same tasks with the same developer populations under controlled conditions.

8. **Cognitive forcing function effectiveness in real-world settings:** While CFFs show promise in lab studies, their effectiveness in real-world development environments—where developers may bypass or resent forced friction—remains unstudied. The "intervention-generated inequalities" where high-NFC users benefit more need adaptive solutions.

9. **Code quality metrics consensus:** The industry lacks consensus on which code quality metrics matter most for AI-generated code. GitClear focuses on refactoring and duplication, CodeRabbit on defect rates, and Academics on build success rates. No unified framework exists.

10. **Organizational productivity measurement:** Only 6% of engineering leaders believe current measurement frameworks capture AI's impact on productivity. The field lacks validated instruments for measuring AI's true organizational productivity impact beyond individual developer perceptions.

---

## 8. Sources

[1] ProAIDE Study — "Developer Interaction Patterns with Proactive AI: A Five-Day Field Study" (JetBrains/ACM/IUI 2026): https://arxiv.org/html/2601.10253v1

[2] Fastly July 2025 Developer AI Trust Survey: https://www.fastly.com/blog/developer-ai-trust-survey-2025

[3] The New Stack — "Fastly: Senior Devs Ship 2.5x More AI Code Than Juniors": https://thenewstack.io/fastly-senior-devs-ship-2-5x-more-ai-code-than-juniors

[4] Stack Overflow 2025 Developer Survey — AI Section: https://survey.stackoverflow.co/2025/ai

[5] Peng, Demirer et al. — "The Effects of Generative AI on High-Skilled Work" (MIT/Microsoft/Princeton/UPenn, Feb 2025): https://economics.mit.edu/sites/default/files/inline-files/draft_copilot_experiments.pdf

[6] GetDX Newsletter — "What three experiments tell us about Copilot's impact on productivity": https://newsletter.getdx.com/p/copilot-impact-on-productivity

[7] Barke, Bird, Ford et al. — "Grounded Copilot: How Programmers Interact with Code-Generating Models": https://shraddhabarke.github.io/raw/copilot.pdf

[8] Barke et al. — "Taking Flight with Copilot" (CACM): https://cacm.acm.org/practice/taking-flight-with-copilot/

[9] SoftwareSeni — "The AI Code Productivity Paradox: 41% Generated but Only 27% Accepted": https://www.softwareseni.com/the-ai-code-productivity-paradox-41-percent-generated-but-only-27-percent-accepted

[10] Sabouri et al. — "Trust Dynamics in AI-Assisted Development" (Amazon Science/ICSE 2025): https://assets.amazon.science/99/78/f02aeaa049b4ba514d7f2790ade7/trust-dynamics-in-ai-assisted-development-definitions-factors-and-implications.pdf

[11] Codellaborator Study — "Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support" (arXiv 2502.18658v4): https://arxiv.org/html/2502.18658v4

[12] Google Research — "Identifying the Factors that Influence Trust in AI Code Completion" (AIware 2024): https://storage.googleapis.com/gweb-research2023-media/pubtools/7831.pdf

[13] METR Study — AI Productivity Paradox (DigitalApplied, July 2025): https://www.digitalapplied.com/blog/ai-productivity-paradox-developer-guide

[14] Faros AI — "What METR's Study Missed About AI Productivity in the Wild": https://www.faros.ai/blog/lab-vs-reality-ai-productivity-study-findings

[15] Microsoft Research — "Time-Warp Developer Productivity Study" (2024): https://www.microsoft.com/en-us/research/wp-content/uploads/2024/11/Time-Warp-Developer-Productivity-Study.pdf

[16] JetBrains Research — "Understanding AI's Impact on Developer Workflows" (HAX Team, ICSE 2026): https://blog.jetbrains.com/research/2026/04/ai-impact-developer-workflows

[17] arXiv 2601.10258 — "Evolving with AI: A Longitudinal Analysis of Developer Logs" (JetBrains): https://arxiv.org/html/2601.10258v1

[18] SitePoint — "Local vs Cloud AI Coding: Latency, Privacy & Performance Guide" (2026): https://www.sitepoint.com/local-vs-cloud-ai-coding-latency-privacy-performance/

[19] Wang, Cheng, Ford, Zimmermann — "Investigating and Designing for Trust in AI-powered Code Generation Tools" (Microsoft/FAccT 2024): https://www.microsoft.com/en-us/research/publication/investigating-and-designing-for-trust-in-ai-powered-code-generation-tools

[20] Digital Applied — "AI Model Latency Benchmarks 2026: TTFT & TPS Data" (April 2026): https://www.digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput

[21] Chris Dowin — "The $2M Cost of Workplace Distraction" (Substack, November 2025): https://chrisdowin.substack.com/p/the-cost-of-distraction-why-fragmented

[22] JetBrains — "State of Developer Ecosystem 2025": https://www.jetbrains.com/lp/devecosystem-2025/

[23] JetBrains — "Which AI Coding Tools Do Developers Actually Use at Work?" (January 2026 Survey): https://blog.jetbrains.com/research/2026/04/which-ai-coding-tools-do-developers-actually-use-at-work

[24] Microsoft Research — "Findings from Microsoft's 3-week study on Copilot use" (GetDX): https://newsletter.getdx.com/p/microsoft-3-week-study-on-copilot-impact

[25] Parnin & Rugaber — "Resumption Strategies for Interrupted Programming Tasks" (SQJ 2011): http://chrisparnin.me/pdf/parnin-sqj11.pdf

[26] Parnin & Rugaber — "Resumption Strategies for Interrupted Programming Tasks" (ICPC 2009): https://chrisparnin.me/pdf/parnin-icpc09.pdf

[27] ZoomInfo Copilot Case Study (arXiv 2501.13282): https://arxiv.org/html/2501.13282v1

[28] Duke & Vanderbilt — "Breaking the Flow: A Study of Interruptions During Software Engineering Activities" (2025): https://shiftmag.dev/do-not-interrupt-developers-study-says-5715

[29] UTS — "Software Developers' Perceptions of Task Switching and Task Interruption": https://opus.lib.uts.edu.au/bitstream/10453/132915/1/1805.05504.pdf

[30] Shakeri Hossein Abad et al. — "Task Interruption in Software Development Projects" (2018, GetDX): https://getdx.com/research/task-interruption-in-software-development

[31] Trunk.io — "Context Switching in Software Engineering: Reduce Distractions" (April 2025): https://trunk.io/learn/context-switching-in-software-engineering-how-developers-lose-productivity

[32] UC Berkeley — "Calibrating Trust in AI-Assisted Decision Making" (Turner et al.): https://www.ischool.berkeley.edu/sites/default/files/sproject_attachments/humanai_capstonereport-final.pdf

[33] Passi & Vorvoreanu — "Overreliance on AI Literature Review" (Microsoft Research, June 2022): https://www.microsoft.com/en-us/research/wp-content/uploads/2022/06/Aether-Overreliance-on-AI-Review-Final-6.21.22.pdf

[34] BSI/ANSSI — "AI Coding Assistants" Joint Report (September 2024): https://www.bsi.bund.de/SharedDocs/Downloads/EN/BSI/KI/ANSSI_BSI_AI_Coding_Assistants.pdf

[35] UNU Campus Computing Centre — "The Invisible Threat in Your Code Editor: AI's Package Hallucination Problem": https://c3.unu.edu/blog/the-invisible-threat-in-your-code-editor-ais-package-hallucination-problem

[36] CodeRabbit — "State of AI vs Human Code Generation Report": https://coderabbit.ai/blog/state-of-ai-vs-human-code-generation-report

[37] GitClear — "AI Copilot Code Quality: 2025 Research": https://www.gitclear.com/ai_assistant_code_quality_2025_research

[38] Microsoft Research — "AI and Productivity Report First Edition" (December 2023): https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/AI-and-Productivity-Report-First-Edition.pdf

[39] Microsoft Research — "New Future of Work Report 2023": https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/NFWReport2023.pdf

[40] Google RCT — "How much does AI impact development speed? An enterprise-based randomized controlled trial" (arXiv 2410.12944): https://arxiv.org/html/2410.12944v2

[41] "Bridging Developer Needs and Feasible Features for AI Assistants in IDEs" (arXiv 2410.08676, ICSE'26 SEIP): https://arxiv.org/html/2410.08676v2

[42] "Comparing AI Coding Agents: A Task-Stratified Analysis of PR Acceptance" (arXiv 2602.08915): https://arxiv.org/html/2602.08915v1

[43] "Cognitive Biases in LLM-Assisted Software Development" (arXiv 2601.08045): https://arxiv.org/abs/2601.08045

[44] "The Fools are Certain; the Wise are Doubtful: Exploring LLM Confidence in Code Completion" (arXiv 2508.16131): https://arxiv.org/abs/2508.16131

[45] "On the Need to Rethink Trust in AI Assistants for Software Development" (arXiv 2504.12461): https://arxiv.org/html/2504.12461v3

[46] Anthropic — AI Coding Skills Comprehension Gap Study (2026): https://contral.ai/blog/anthropic-study-ai-coding-skills-gap

[47] Microsoft Research Blog — "New Future of Work: AI is driving rapid change, uneven benefits": https://www.microsoft.com/en-us/research/blog/new-future-of-work-ai-is-driving-rapid-change-uneven-benefits

[48] Amazon Bedrock — "Optimizing AI responsiveness: A practical guide to latency-optimized inference": https://aws.amazon.com/blogs/machine-learning/optimizing-ai-responsiveness-a-practical-guide-to-amazon-bedrock-latency-optimized-inference

[49] GitHub Copilot latency architecture (David Cheney, QCon SF 2024/InfoQ): https://www.infoq.com/presentations/github-copilot

[50] Kevin Tan — Copilot SDK Profiling: https://blog.jztan.com/i-profiled-the-copilot-sdk-33-percent-latency-avoidable

[51] Tabnine — "Tabnine goes hybrid": https://www.tabnine.com/blog/tabnine-goes-hybrid

[52] Google Cloud Blog — "How Tabnine Accelerates Development with AI": https://cloud.google.com/blog/topics/partners/how-tabnine-accelerates-development-with-ai

[53] JetBrains — "AI Code Completion: Less Is More" (CatBoost filter model): https://blog.jetbrains.com/ai/2025/03/ai-code-completion-less-is-more

[54] JetBrains — "AI Tool Switching Is Stealth Friction" (February 2026): https://blog.jetbrains.com/ai/2026/02/ai-tool-switching-is-stealth-friction-beat-it-at-the-access-layer

[55] SoftwareSeni — "Junior Developers in the Age of AI": https://www.softwareseni.com/junior-developers-in-the-age-of-ai-who-trains-the-next-generation-of-engineers

[56] SoftwareSeni — "The AI Productivity Paradox in Software Development": https://www.softwareseni.com/the-ai-productivity-paradox-in-software-development-and-what-the-research-actually-shows

[57] Microsoft Research — "Overreliance on AI Literature Review" (PDF): https://www.microsoft.com/en-us/research/wp-content/uploads/2022/06/Aether-Overreliance-on-AI-Review-Final-6.21.22.pdf

[58] Digitalisation World — "Decoding the AI Productivity Paradox" (Harness State of Engineering Excellence 2026): https://digitalisationworld.com/news/72318/decoding-the-ai-productivity-paradox-in-software-development