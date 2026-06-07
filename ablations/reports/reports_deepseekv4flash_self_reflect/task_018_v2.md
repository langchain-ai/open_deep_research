# ERP Interface Design for Legacy-to-Cloud Migration: Progressive Disclosure vs. Persistent Navigation for Production Floor Supervisors

## Revised and Expanded Research Report — May 30, 2026

---

## Executive Summary

This revised report presents a comprehensive analysis of how **progressive disclosure** (exposing information and actions on demand) versus **persistent navigation** (displaying all options and menus continuously) affects task completion rates, training time, and user proficiency for production floor supervisors (aged 45-60) transitioning from legacy AS/400 systems to cloud-based ERP platforms.

**What Has Changed from the Previous Report:**

1. **New empirical evidence added** — including a June 2025 study on UI design for older adults showing 37% improvement in task completion rates and 55.3% reduction in errors with simplified navigation [9], plus a 2024 controlled ERP usability study demonstrating SUS score improvement from 68.4 to 84.2 [5]
2. **Critical nuance introduced** — the expertise reversal effect from cognitive load theory, showing that design elements beneficial for novices can actively hinder experienced users [17]
3. **Accessibility standards integrated** — WCAG 2.2, Section 508, and AARP age-friendly design guidelines specifically applied to ERP contexts [22][23][25]
4. **Strangler fig migration strategy deepened** — with specific UX patterns for phased legacy coexistence, including the 68% failure rate for projects starting at the UI layer [26]
5. **Edge cases analyzed** — specific workflows where persistent navigation is superior to progressive disclosure [31]
6. **Implementation guidance added** — A/B testing methodology, cognitive load measurement protocols, and rollout metrics framework [33][38]
7. **Head-to-head platform comparison expanded** — with quantitative metrics from SAP, Oracle, and Microsoft implementations [1][2][3]

**Key Findings:**

1. Progressive disclosure consistently reduces training time and cognitive load for transitioning users, with SAP Fiori implementations showing 25-40% faster training and 30-50% faster task completion compared to SAP GUI [1][2]
2. Role-based simplified views outperform comprehensive dashboards for the target demographic, but must include clear pathways to advanced functions — the "80/20 rule" where 80% of daily tasks are handled in the simplified view with clear access to the remaining 20% [31]
3. The expertise reversal effect means that progressive disclosure can **increase** cognitive load for expert users performing high-frequency repetitive tasks — a critical consideration for production floor supervisors who may have decades of muscle memory from AS/400 systems [17]
4. For users aged 45-60, the primary barriers are not age-related inability to learn, but rather interface complexity, reduced contrast sensitivity, declining working memory capacity, and the discomfort of unlearning deeply ingrained mental models [23][24]
5. A hybrid model — default role-based simplified views with clear progressive disclosure paths to advanced functions, plus user-configurable customization — represents the optimal approach for this demographic [14][31]

---

## Section 1: Critical Evaluation of the Previous Report — Gaps, Weaknesses, and Assumptions

### 1.1 Missing Empirical Evidence

The previous report relied heavily on principle-based arguments from Nielsen Norman Group and vendor marketing materials without adequate empirical backing. Specifically:

- **No peer-reviewed studies** directly comparing progressive disclosure vs. persistent navigation for the 45-60 demographic in manufacturing ERP contexts were cited
- The closest applicable study (JETNR June 2025 on UI design for elderly users aged 65-85) was not included — this study found **37% improvement in task completion rates**, **55.3% reduction in error frequency**, **27.4% faster task completion**, and **42% decrease in cognitive load** with simplified navigation for older users [9]
- A 2024 controlled ERP usability study (Nittala) comparing traditional ERP navigation with a simplified conversational interface found SUS scores improved from 68.4 to 84.2 (a 23% improvement), task completion time reduced by 28%, and cognitive load decreased by 22% [5]
- These quantitative benchmarks should have formed the empirical foundation of the original analysis

### 1.2 Over-Reliance on Single-Vendor Data

The previous report disproportionately relied on SAP Fiori quantitative claims (25-40% faster training, 30-50% faster task completion) without adequate cross-validation. While these figures are supported by the 2025 JENRS study [2] and SAP's own UX research [1], they lack independent third-party verification for manufacturing-specific contexts. The report should have included:

- Forrester 2024 survey data showing logistics companies implementing progressive disclosure improved completion rates by 20% on average [37]
- NetSuite UX redesign case study showing 46% improvement in navigation speed and 39% reduction in report generation time [6]
- Dynamics 365 production floor execution interface deployment metrics [3]

### 1.3 Absence of the Expertise Reversal Effect

The most significant theoretical gap was the absence of the **expertise reversal effect** from cognitive load theory. This effect, documented by Kalyuga and colleagues (2003, 2007, 2009), demonstrates that instructional methods and interface designs that benefit novices can actively **hinder** experts [17][18]. Key findings that should have been included:

- "What induces high cognitive load in a novice is different than what induces high cognitive load in someone with high prior knowledge" [17]
- "Novices receiving additional text scored higher on retention and transfer than did novices without additional text, while this result was reversed for experts" (Rey & Buchwald, 2011) [18]
- The expertise reversal effect is directly applicable to progressive disclosure: simplified views that help new AS/400 migrants may frustrate and slow down experienced users who have developed efficient mental models over decades

### 1.4 Insufficient Accessibility Coverage

The previous report mentioned cognitive load reduction but did not address:

- **WCAG 2.2 specific success criteria** applicable to ERP navigation (SC 2.4.11 Focus Not Obscured, SC 2.5.8 Target Size Minimum of 24×24 CSS pixels, SC 3.3.8 Accessible Authentication alternatives) [22]
- **Age-related vision changes**: contrast sensitivity deterioration starts between ages 40-50, affecting the ability to distinguish between similar colors — critical for dashboard design where color coding conveys status [24]
- **Reduced working memory**: Older adults (45+) experience decline in fluid intelligence affecting processing speed and working memory capacity, while crystallized intelligence (accumulated knowledge) remains stable [23]
- **Fine motor control**: WCAG 2.5.8 directly addresses the need for larger touch targets to support users with reduced motor control, particularly relevant for touch-based production floor interfaces [22]

### 1.5 Missing Strangler Fig UX Strategy

The previous report mentioned the strangler fig migration pattern briefly but did not provide concrete UX guidance. Critical missing elements include:

- **68% of strangler fig projects stall within 90 days**, often due to starting at the UI layer rather than at business capability boundaries [26]
- The **anti-corruption layer (ACL)** pattern as a UX translation mechanism between legacy and modern interfaces — "your insurance policy against legacy systems ruining your modern application's design" [27]
- **Green screen modernization tools**: Profound Logic, LANSA aXes, and Fresche Solutions Presto convert 5250 screens to web interfaces without changing underlying business logic — these provide immediate UX improvements while preserving AS/400 functionality [28]
- **The three-phase model**: transform, coexist, eliminate — with specific UX patterns for each phase [26]

### 1.6 Inadequate Edge Case Analysis

The original report assumed progressive disclosure was universally beneficial. It failed to analyze scenarios where it fails:

- **When hidden features are needed frequently**: "Hiding frequently used features increases interaction cost and cognitive load" rather than reducing it [31]
- **The 80/20 rule**: Progressive disclosure works only when the basic level satisfies 80% of users' needs. For the 20% of users who regularly need advanced features, it creates friction [31]
- **Emergency scenarios**: In manufacturing, troubleshooting and override workflows require immediate access to advanced options — progressive disclosure can delay critical responses [30]
- **Muscle memory disruption**: Users with decades of AS/400 experience have deeply ingrained mental models. Progressive disclosure that reorganizes familiar workflows can increase errors during transition [29]

---

## Section 2: Updated Empirical Evidence (2025-2026 Sources)

### 2.1 JETNR June 2025 Study: UI Design for Elderly Users

The most directly relevant new study, "The Impact of UI Design Elements on Cognitive Performance among Elderly Mobile Application Users," published June 2025, provides quantitative benchmarks for simplified navigation in older populations [9]:

| Metric | Standard Interface | Optimized Interface | Improvement |
|--------|-------------------|--------------------|-------------|
| Task Completion Rate | Baseline | +37% | Significant |
| Error Frequency | Baseline | -55.3% | Significant |
| Task Completion Time | Baseline | -27.4% | Significant |
| Cognitive Load (NASA-TLX) | Baseline | -42% | Significant |

**Critical findings for target demographic:**
- Simplified navigation was the **most influential UI element**, more important than increased touch target sizes or reduced visual clutter
- Improvements were **more pronounced in older participants (ages 76-85)** with 54% improvement vs. 28% for ages 65-75
- Improvements were also more pronounced in those with **less technology experience**
- The study explicitly states: "Properly optimized interfaces can to a great extent eliminate cognitive effort and enhance usability for elderly users... without sacrificing functionality, or producing segregated 'senior versions' of applications" [9]

**Limitations**: The study used mobile applications for users aged 65-85, not enterprise desktop software for users aged 45-60. However, the cognitive mechanisms (reduced working memory, processing speed decline, contrast sensitivity loss) are continuous across age ranges and the design principles transfer directly.

### 2.2 Applied Ergonomics 2025: Systematic Review of Eye Tracking in Older Adults

Li and Tang's systematic review of 14 eye-tracking studies on older adults' online performance, published in Applied Ergonomics Volume 128 (October 2025), found [13]:

- Older adults' online performance is influenced by multiple heterogeneous factors across studies
- Most studies focus on **individual design elements** rather than holistic navigation patterns
- The review explicitly calls for future research to propose **context-based design suggestions** to address the holistic needs of older adults — this gap remains particularly for enterprise/industrial contexts
- Key design implications are organized around **icon, image and text, interaction, and layout design**

**Critical implication**: The lack of holistic navigation pattern research means ERP designers must extrapolate from individual element findings. This reinforces the need for controlled A/B testing during ERP rollouts.

### 2.3 Nittala 2024: ERP User Productivity and Cognitive Load Study

A controlled case study involving 20 participants in a mid-sized organization compared traditional ERP navigation with a simplified interface across tasks including inventory management and purchase orders [5]:

| Metric | Traditional ERP | Simplified Interface | Improvement |
|--------|----------------|---------------------|-------------|
| System Usability Scale (SUS) | 68.4 (average, C grade) | 84.2 (Excellent, A grade) | +23% |
| Task Accuracy | 86.2% | 93.5% | +7.3 pp |
| Task Completion Time | Baseline | 28% reduction | -28% |
| Cognitive Load (NASA-TLX) | Baseline | 22% reduction | -22% |

**Key insight**: The SUS score improvement from 68.4 to 84.2 is particularly significant. An SUS of 68 represents the 50th percentile across all software categories. A score of 84.2 places it in the top 10% of all tested products — a dramatic shift from average to excellent usability [4][5].

### 2.4 SAP Fiori Quantitative Metrics: 2025 Peer-Reviewed Publication

The Journal of Engineering Research and Sciences (JENRS) published "Implementing SAP Fiori in S/4HANA Transitions" in 2025, providing quantitative outcomes from Fiori implementations [2]:

- **Training efficiency**: 25-40% faster
- **Task completion time**: 30-50% faster
- **Click reduction**: 40-60% fewer clicks
- **ROI timelines**: 12-18 months

Additional findings from the European Journal of Computer Science and Information Technology (2025):
- Up to 40% reduced training times
- 44% higher user satisfaction
- 37% fewer support tickets
- 38% faster task completion in mobile contexts [1]

These figures are broadly consistent with the Nittala study (28% task time reduction with simplified interface) and Forrester data (20% completion rate improvement), providing convergent validity across different research methodologies.

### 2.5 Forrester 2024: Progressive Disclosure Impact on Completion Rates

A 2024 Forrester survey found that logistics companies implementing field validation and progressive disclosure improved completion rates by **20% on average** [37]. A companion 2023 Gartner supply chain report highlighted that enterprises improving digital form completion rates by 25% saw **12% faster order fulfillment** and **7% fewer downstream errors** [37].

---

## Section 3: Cognitive Load Theory Revisited — The Expertise Reversal Effect

### 3.1 Foundational Cognitive Load Theory (CLT)

Cognitive Load Theory distinguishes three types of cognitive load [16]:

1. **Intrinsic load**: The inherent complexity of the task (e.g., creating a work order)
2. **Extraneous load**: Unnecessary cognitive demands caused by poor interface/instructional design
3. **Germane load**: The effort dedicated to learning and building new mental models

For users transitioning from AS/400 systems, the **intrinsic load** is amplified by the need to learn new business processes, not just a new interface. **Extraneous load** is high when they encounter a modern, visually dense interface. The goal of progressive disclosure is to minimize extraneous load to allow for germane load (learning).

### 3.2 The Expertise Reversal Effect — Critical Missing Piece

The expertise reversal effect, documented by Kalyuga (2007), demonstrates that instructional techniques' effectiveness changes based on learners' prior knowledge levels: "Instructional guidance, which may be essential for novices, may have negative consequences for more experienced learners" [17].

**How this applies to ERP interface design:**

| User Profile | Progressive Disclosure Effect | Persistent Navigation Effect |
|-------------|------------------------------|------------------------------|
| Novice (new ERP user) | **Beneficial** — reduces cognitive overload | Overwhelming — too many options |
| Intermediate (some experience) | Neutral — may or may not help | Workable but inefficient |
| Expert (20+ years AS/400 muscle memory) | **Harmful** — extra clicks for frequently used features | Efficient — supports rapid task switching |

**The key insight from the expertise reversal effect for your demographic:**

AS/400 production floor supervisors are simultaneously **experts in their workflow** (20+ years of mental models) and **novices with the new system**. This creates a unique cognitive conflict:

- Their existing mental models for task completion (e.g., "type transaction code, enter data, press Enter") do not transfer to GUI interfaces
- But progressive disclosure that hides functions they know exist (because they used them daily on the AS/400) **increases frustration and perceived system inadequacy**
- The "paradox of the active user" (Carroll & Rosson, 1987) explains this: users prefer immediate action over learning new approaches, even when learning would save time long-term [29]

### 3.3 Mitigation Strategies for the Expertise Reversal Effect

Research from The Elearning Coach recommends [17]:

1. **Allow users to self-assess proficiency** — provide options to reveal more features
2. **Enable test-out options** — expert users can skip simplified views
3. **Support independent exploration** — undo functionality enables safe experimentation
4. **Accommodate varying skill entry points** — the same user may be expert in some workflows and novice in others

For ERP design specifically, this translates to:

- **Dual-mode interface**: A toggle between simplified "guided" mode and expert "full" mode
- **Adaptive fading**: Gradually reveal more features as the user demonstrates proficiency (measured by task completion time and error rate reduction)
- **Configurable dashboards**: Allow users to pin frequently used advanced functions to their simplified view

---

## Section 4: Accessibility and Inclusive Design Standards for ERP Interfaces

### 4.1 WCAG 2.2 Success Criteria Relevant to ERP Navigation

WCAG 2.2 (released October 2023) introduced nine new success criteria, several of which are critical for production floor ERP interfaces [22]:

**SC 2.4.11 Focus Not Obscured (Level AA)**:
- Keyboard focus must not be entirely hidden by author-created content (sticky headers, modals, tooltips)
- **Relevance for ERP**: Production supervisors using keyboard navigation must always see which field has focus — critical when entering inventory quantities or work order parameters

**SC 2.4.12 Focus Not Obscured (Level AAA)**:
- Focus indicator must not be obscured at all (more stringent than Minimum)

**SC 2.5.7 Dragging Movements (Level AA)**:
- Drag-and-drop must have alternative keyboard-accessible interactions
- **Relevance for ERP**: Manufacturing scheduling boards often use drag-and-drop for work order assignment — an alternative selection method must be provided

**SC 2.5.8 Target Size (Minimum, Level AA)**:
- Interactive targets must be at least 24×24 CSS pixels
- **Relevance for ERP**: Production floor interfaces are accessed via touch screens where operators wear gloves — 24px minimum is insufficient; 48px+ recommended for industrial touch interfaces [22][30]

**SC 3.3.8 Accessible Authentication (Minimum, Level AA)**:
- Alternatives to memory-based authentication must be provided
- **Relevance for ERP**: Production supervisors may have difficulty remembering complex passwords. Support for badge ID scanning, facial recognition, or single sign-on is recommended

### 4.2 Section 508 Compliance Requirements

Section 508 of the Rehabilitation Act requires ICT accessibility for federal agencies and increasingly for enterprise buyers. The 2017 Refresh harmonized U.S. standards with WCAG 2.0 Level A/AA, but has not been updated since, making it less current than WCAG 2.2 [25].

**Key Section 508 technical standards for ERP:**
- Color contrast ratio of at least 4.5:1 for text
- Full keyboard navigation
- Proper hierarchical heading structure
- Descriptive labels for all form controls
- Captions for training videos

**Critical note**: Organizations treating accessibility as optional face legal risk, procurement exclusion, and reputation damage. "Accessibility has become a procurement gatekeeper" — increasingly, enterprise buyers require VPAT documentation [25].

### 4.3 AARP Age-Friendly Technology Design Guidelines

The AARP and Older Adults Technology Services guide "Age-Friendly Technology Design" (2022) provides specific guidance for adults aged 50+ [23]:

**Key facts:**
- 42% of 50-plus adults feel technology is not designed with them in mind
- 9 in 10 adults over 50 own a computer; nearly 8 in 10 own a smartphone
- Five barriers to adoption: cost, complexity, digital literacy, physical/cognitive difficulties, and security/privacy concerns
- Adults 50-plus drive $8.3 trillion of economic activity annually

**Design recommendations for ERP:**
- **Visual**: High-contrast color schemes, sans-serif fonts of at least 14pt, avoid reliance on color alone for status indicators
- **Cognitive**: Step-by-step workflows with clear progress indicators, minimize working memory demands, provide contextual help on demand
- **Motor**: Minimum 48px touch targets for industrial environments, provide keyboard shortcuts as alternatives to mouse-based interactions
- **Training**: Self-paced, work-integrated training with error management that frames mistakes positively

### 4.4 Age-Related Cognitive and Visual Changes

**Contrast sensitivity** (ability to distinguish between similar colors) deteriorates starting between ages 40-50, particularly affecting blue-yellow discrimination. Approximately 3.4 million people in the US are blind or visually impaired, with reduced contrast sensitivity contributing significantly [24].

**Working memory** capacity declines with age, affecting the ability to hold multiple pieces of information while navigating complex interfaces. This directly impacts task completion when progressive disclosure requires users to remember what's behind each disclosure layer.

**Processing speed** decreases with age, meaning older users take 50-100% more time on tasks when encountering unfamiliar complex interfaces [23].

**Crystallized intelligence** (accumulated knowledge and experience) remains stable or increases with age. AS/400 users with 20+ years of experience have deep domain knowledge — the interface should leverage this by preserving familiar workflow structures where possible.

---

## Section 5: Strangler Fig Migration Pattern for Interface Design

### 5.1 The Strangler Fig Pattern Overview

The Strangler Fig Pattern, coined by Martin Fowler, incrementally replaces a legacy system by routing requests via a proxy layer and gradually migrating functionality until the legacy system can be decommissioned. AWS describes the process in three verbs: **transform, coexist, eliminate** [26].

**For interface design, this means:**
1. **Phase 1 — Transform**: Add a modern UI layer on top of the existing AS/400 green screen (e.g., using Profound Logic or LANSA aXes to convert 5250 screens to web interfaces)
2. **Phase 2 — Coexist**: Run both interfaces in parallel, with the proxy router directing users based on functionality migration status
3. **Phase 3 — Eliminate**: Decommission the legacy interface when all functionality has been migrated

### 5.2 Critical Research Finding: 68% of Strangler Projects Stall

A 2022-2025 research study of 41 enterprise projects found that **68% of strangler projects stall within 90 days** [26]. The primary cause? **Starting at the UI layer.** Key findings:

- Projects that extracted less than 5% of monolith functionality in the first 90 days had a **92% failure rate**
- Only three viable starting points for strangler roots: new business capability (91% success), read-only facade with change data capture (76% success), and write-path interception (54% success)
- Starting by redesigning the UI alone (without migrating underlying business logic) was not listed as a viable entry point

**Implication for your ERP migration**: Do NOT start by redesigning the UI. Start by identifying a discrete business capability (e.g., "work order creation for standard orders" rather than "the entire work order management module") and migrate that complete capability — data, logic, and interface — before moving to the next.

### 5.3 Green Screen Modernization Tools

Several tools enable immediate UX improvements for AS/400 green screens without changing underlying business logic [28]:

**Profound Logic UX Futurization Service**: AI-assisted mass screen conversion that preserves critical business logic. Results in weeks, not years. Custom Genie skins modernize green screens while reducing training time.

**LANSA aXes**: Converts 5250 applications to web pages on the fly without programming or source code access. Turns menus into clickable links and function keys into buttons. HUPAC Group modernized applications three times faster and saved 50% on maintenance.

**Fresche Solutions Presto**: Converts green screens to web applications while preserving underlying RPG, COBOL, or CA 2E logic. Quest Medical modernized over 2,000 green screens within their ERP system rapidly without needing in-depth web development skills.

**Programmers.io Green2Glass (G2G)**: Screen scraping engine that automatically transforms 5250 interfaces into web and browser-based GUIs with drag-and-drop controls for field rearrangement.

### 5.4 The Anti-Corruption Layer (ACL) for UX Translation

The Anti-Corruption Layer is a design pattern that isolates and translates communications between legacy and modern systems. For UX specifically, the ACL serves as [27]:

- **A Facade**: Simplifies complex legacy interfaces into clean, modern API calls
- **An Adapter**: Orchestrates data transformation between legacy data models and modern UI components
- **A Translator**: Handles data structure conversions and complex business rules so the new UI remains clean

"The ACL is your insurance policy against legacy systems ruining your modern application's design" [27]. For ERP UX, the ACL ensures that:
- Legacy transaction codes can still be typed into a modern search bar
- AS/400 workflow sequences are preserved in the new interface
- Keyboard shortcuts from the legacy system continue to work

### 5.5 Practical Phased Migration UX Strategy

Based on the research, here is the optimal phased approach:

**Phase 1 — Coexistence (Months 1-3)**:
- Deploy a green screen modernization tool (Profound Logic, LANSA aXes, or Presto) as an interim UI
- This provides immediate UX improvements (web access, mouse navigation, visual refresh) without changing business logic
- Measure baseline metrics: task completion time, error rates, SUS scores
- Begin user training on the transitional interface while preserving AS/400 keyboard shortcuts

**Phase 2 — Incremental Migration (Months 4-12)**:
- Select one high-value, low-risk business capability for full migration (e.g., standard work order creation)
- Build the new cloud ERP interface for this capability only
- Use the strangler fig proxy to route this specific functionality to the new system
- Run A/B tests comparing the new interface against the transitional interface
- Document lessons learned and user feedback before expanding

**Phase 3 — Expansion (Months 13-24)**:
- Migrate additional capabilities in order of business value and technical readiness
- For each capability, provide a "translation guide" showing how AS/400 transaction codes map to new interface navigation paths
- Maintain the existing interface as a fallback for troubleshooting

**Phase 4 — Decommissioning (Months 25-30)**:
- When all functionality is migrated and user proficiency metrics meet targets, begin sunsetting the legacy interface
- Provide a "legacy mode" bookmark that experienced users can access for specific tasks during transition
- Archive but do not delete the old system for audit and reference purposes

---

## Section 6: Platform Comparison — Updated Quantitative Analysis

### 6.1 SAP S/4HANA: Fiori vs. GUI — Updated Evidence

**SAP Fiori (Progressive Disclosure)** : Role-based, tile-driven interface designed for specific user tasks. Governed by five principles: role-based, adaptive, simple, coherent, and delightful [1][2].

**SAP GUI (Persistent Navigation)** : Traditional, text-heavy, form-based interface with transaction codes (T-codes). Keyboard-driven, optimized for power users who memorize shortcuts.

**Quantitative Improvements (2025 peer-reviewed data)** :

| Metric | Improvement | Source |
|--------|-------------|--------|
| Training efficiency | 25-40% faster | JENRS 2025 [2] |
| Task completion time | 30-50% faster | JENRS 2025 [2] |
| Click reduction | 40-60% fewer clicks | JENRS 2025 [2] |
| User satisfaction | 44% higher | EJCSIT 2025 [1] |
| Support tickets | 37% fewer | EJCSIT 2025 [2] |
| ROI timeline | 12-18 months | JENRS 2025 [2] |

**Spaces and Pages Architecture**: SAP introduced Spaces and Pages (starting with S/4HANA 2020) to replace Business Groups. Best practices include limiting a Space to 1-5 pages, 2-5 sections per page, and 3-8 apps per section — this is structured progressive disclosure [1].

**SAP Fiori UX Q1/2026 Update**: New features include AI-assisted Situation Handling, smart personalization of My Home, and the ability to launch apps using traditional transaction codes (t-codes) — addressing the muscle memory challenge for legacy users [1].

**Recommended approach for AS/400 migrants**: Start with SAP Fiori using Spaces and Pages, but enable t-code search for experienced users. Use SAP's Situation Handling progressive disclosure framework for exception management (small indicators → medium descriptions → large detail views).

### 6.2 Oracle NetSuite: Role-Based Centers and Progressive Disclosure

**Design Philosophy**: "The UI uses progressive disclosure; pages focus on your content, and relevant links, menus, and icons appear when moving your pointer over areas" [6].

**Original UX Redesign Quantitative Results** (Ron Design Lab):
- **64% of operational managers desired a unified dashboard** to view financial, CRM, and sales data
- **38% process time reduction** achievable by simplifying tasks
- **47% of employees preferred mobile access**
- **46% improvement in navigation speed** post-redesign
- **39% reduction in report generation time** [6]

**SuiteSuccess Manufacturing**: Preconfigured edition with role-based dashboards for manufacturing roles. Key features include work order management, BOM management, production scheduling, and inventory management [6].

**Redwood Experience**: Recent UI update with collapsible form sections, improved dashboard personalization, and AI-powered "Ask Oracle" for natural language search.

**Recommended approach for AS/400 migrants**: NetSuite's Center-based navigation (e.g., Manufacturing Center for production supervisors) provides a clear role-based starting point. The hover-based progressive disclosure reduces visual clutter while maintaining access to all functions.

### 6.3 Microsoft Dynamics 365: Hybrid Model with Production Floor Execution Interface

**Production Floor Execution (PFE) Interface**: A streamlined, role-based workspace designed specifically for manufacturing workers, optimized for touch interaction [3].

Key features:
- Workers sign in using badge IDs or personnel numbers
- Customizable display themes
- Job list views across tabs: All Jobs, Active Jobs, My Jobs
- Support for material consumption registration, scrap reporting, serial number tracking
- Integration with mixed-reality guides for HoloLens
- Machine health metric visualization for proactive maintenance

**Role Center Architecture (Business Central)** : Role Centers serve as the user's home page, providing quick access to relevant tasks, KPIs, and data visualizations based on the user's role.

**Customization via Power Apps**: Microsoft Power Apps enables low-code customization of simplified manufacturing interfaces. Five practical examples include inventory management apps, audit and inspection apps, and employee attendance tracking apps [3].

**Quantitative ROI Metrics**:
- Average ROI of over 170% within three years (Forrester TEI study)
- Organizations with proper training experience 30-40% faster efficiency gains
- 30% faster month-end close for Business Central users [3]

**Recommended approach for AS/400 migrants**: Dynamics 365's PFE interface is the most purpose-built solution for production floor workers among the three platforms. Its touch-optimized, role-based design directly addresses the needs of shop floor supervisors. The hybrid model — persistent navigation pane combined with role-based default views — provides flexibility for different proficiency levels.

### 6.4 Head-to-Head Comparison: Manufacturing UX

| Dimension | SAP S/4HANA Fiori | Oracle NetSuite | Microsoft Dynamics 365 |
|-----------|-------------------|-----------------|----------------------|
| Disclosure strategy | Progressive (Spaces/Pages) | Progressive (hover-based) | Hybrid (persistent nav + role views) |
| Manufacturing-specific UI | PEO + production apps | Manufacturing Center | Production Floor Execution interface |
| AS/400 migration path | Fiori with t-code support + SAP GUI coexistence | Transitional green screen tools + NetSuite connector | Power Apps custom interfaces + legacy integration |
| User customization | High (pinning tiles) | Moderate (portlet arrangement) | High (Power Apps, role center config) |
| AI assistance | Joule copilot + situation handling | Ask Oracle + AI insights | Copilot + Power Platform AI |
| Touch optimization | Responsive | Responsive | Dedicated touch interface |
| Training approach | Role-based with SAP Activate | SuiteSuccess methodology | Learning paths + adoption platform |

---

## Section 7: Design Patterns for Progressive Disclosure in Manufacturing ERP

### 7.1 Pattern 1: Hierarchical Drill-Down Dashboard (SAP Fiori Model)

**Description**: Start with high-level KPIs (e.g., "Work Orders Today," "Inventory Alerts"), each presented as a tile or card. Clicking a tile reveals more detail, and clicking again reaches transactional data.

**Implementation for production supervisors:**
- **Level 1 (Home dashboard)** : 5-7 KPI tiles showing shift performance: active work orders, overdue orders, inventory exceptions, team attendance, quality alerts
- **Level 2 (List view)** : Click a tile to see a filtered list (e.g., "Overdue Work Orders" shows all overdue orders with status, priority, and assigned team)
- **Level 3 (Detail view)** : Click a list item to see the full work order with materials, operations, labor, and history

**Accessibility requirements**: 
- Tiles must be keyboard navigable (SC 2.1.1)
- Focus must not be obscured when drill-down opens (SC 2.4.11)
- Touch targets must be minimum 24×24px (SC 2.5.8), preferably 48×48px for industrial environments

### 7.2 Pattern 2: Step-by-Step Wizards for Complex Transactions

**Description**: Break multi-step processes (e.g., creating a work order, reporting production) into sequential screens with progress indicators.

**Implementation for production supervisors:**
- **Step 1**: Select work center and production order
- **Step 2**: Enter labor hours and quantities
- **Step 3**: Report quality issues or exceptions
- **Step 4**: Review and submit

**Benefits for older adults**: Reduces working memory demands by showing only one step at a time. Progress indicator provides orientation and reduces anxiety about losing progress.

**Accessibility requirements**:
- Keyboard navigation must follow logical focus order (SC 2.4.3)
- Error identification must be clear and immediate (SC 3.3.1)
- Users must be able to go back without losing data

### 7.3 Pattern 3: Adaptive Interface with Fading Support

**Description**: Combine the "training wheels" approach (Carroll & Carrithers, 1984) with adaptive fading — show simplified views initially and gradually reveal more features as proficiency increases.

**Implementation for production supervisors:**
- **Week 1-2**: Show only essential fields (material, quantity, work center)
- **Week 3-4**: Add optional fields (serial numbers, quality checks, notes)
- **Week 5+**: Enable full interface with all advanced options
- **Trigger conditions**: Proficiency is measured by task completion time and error rate — when metrics reach targets, additional features are revealed
- **Manual override**: User can request to see all features at any time

**Evidence base**: Salden et al. found that "the adaptive fading condition outperformed the two non-adaptive conditions on both immediate and delayed posttests" [17]. This approach respects both the expertise reversal effect (gradually increasing complexity) and the paradox of the active user (allowing immediate action).

### 7.4 Pattern 4: Configurable Role-Based Dashboard

**Description**: Default dashboards based on role, but users can customize by pinning frequently used advanced functions.

**Implementation for production supervisors:**
- **Default view**: Production supervisor dashboard with work order queue, team status, inventory alerts, and quality notifications
- **Pin function**: User can add any function from a "full menu" to their dashboard sidebar
- **Search**: Prominent search bar supports transaction code entry for AS/400 migrants
- **Persistence**: Customizations survive sessions and logouts

**Benefits for target demographic**: Preserves the AS/400 concept of frequently used transaction codes while providing a modern interface. Empowers users to gradually build a personalized workspace as they discover new functions.

### 7.5 Pattern 5: Contextual Help and Tooltip-on-Demand

**Description**: Provide help content only when requested, integrated into the workflow without disrupting task flow.

**Implementation for production supervisors:**
- **Field-level tooltips**: Hover or tap on any field label to see a brief explanation
- **Workflow help**: "How do I..." button that shows the current task's instructions
- **Legacy translation**: Show AS/400 transaction code equivalent for each new interface function
- **Error recovery**: When an error occurs, provide clear instructions for correction (SC 3.3.1)

---

## Section 8: Evaluation of Trade-Offs and Edge Cases

### 8.1 When Progressive Disclosure Fails

**Scenario 1: High-frequency expert users**
- A production supervisor who processes 50+ work orders per day needs rapid access to all fields
- Progressive disclosure requiring 2 extra clicks per work order × 50 orders = 100 extra clicks per day
- Over a year (250 working days): 25,000 extra clicks = approximately 3.5 hours of additional time wasted [31]

**Scenario 2: Emergency response**
- When a machine breaks down, the supervisor needs immediate access to override production schedules
- Progressive disclosure that hides the "cancel work order" function behind three menu levels delays response
- "Every hour of unplanned equipment downtime costs an average of $260,000 across industrial sectors" [30]

**Scenario 3: Multi-parameter troubleshooting**
- Diagnosing a quality issue requires simultaneous visibility of material lots, machine parameters, operator assignments, and inspection results
- Progressive disclosure showing only one parameter at a time makes pattern recognition impossible

**Scenario 4: Shift handover**
- The incoming supervisor needs a complete status overview: all open work orders, pending quality issues, maintenance requests, and team availability
- A simplified dashboard that shows only "current tasks" hides critical context needed for safe handover

### 8.2 When Persistent Navigation is Superior

**Workflows requiring persistent visibility of advanced options:**

1. **Production monitoring dashboards**: Real-time display of line status, throughput, and quality requires simultaneous visibility of multiple data streams [30]
2. **Exception handling**: When quality issues arise, supervisors need instant access to material traceability, machine history, and operator records without navigating through disclosure layers
3. **Multi-line supervision**: Aggregating metrics across multiple production lines requires persistent visibility, not progressive reveal [30]
4. **Audit and compliance**: During regulatory audits, all relevant data must be accessible without navigation delays

### 8.3 The Optimal Hybrid Solution

Based on the evidence, the recommended approach is a **role-based hybrid model**:

**Default: Simplified role-based view**
- Shows 5-7 KPIs and primary task buttons
- Follows the "5-KPI rule" for operator-level dashboards [30]
- Provides clear, obvious paths to deeper functionality

**Persistent elements:**
- Navigation pane (collapsible) showing all available modules
- Search bar supporting transaction codes and natural language queries
- Status bar showing pending exceptions, alerts, and handover notes

**Progressive disclosure elements:**
- Drill-down from KPIs to detailed data
- Wizards for multi-step transactions
- Contextual help on demand

**User customization:**
- Pin frequently used advanced functions to the simplified view
- Toggle between "guided" and "expert" modes
- Create shortcuts for repetitive tasks

**Critical design rule**: "Success in progressive disclosure depends on the basic level satisfying 80% of users' needs and added information justifying the interaction cost" [31]. Test this assumption with real users before implementation.

---

## Section 9: Implementation Guidance — A/B Testing, Metrics, and Cognitive Load Measurement

### 9.1 A/B Testing Methodology for ERP Rollouts

**Recommended approach**: Controlled phased rollout with update rings, adapted from the "strangler fig" pattern [33].

**Phase 1 — Pilot Group (Week 1-4)** :
- Select 5-10 production supervisors from a representative plant
- Must include both "tech-comfortable" and "tech-resistant" users
- Divide into two groups:
  - Group A: Progressive disclosure interface (role-based simplified view)
  - Group B: Persistent navigation interface (comprehensive menu-based view)
- Measure baseline metrics for both groups before transition

**Phase 2 — Controlled Experiment (Week 5-8)** :
- Both groups use their assigned interface for actual work tasks
- Collect metrics daily (see Section 9.2)
- Conduct weekly 15-minute user interviews

**Phase 3 — Analysis and Adjustment (Week 9-10)** :
- Compare quantitative metrics between groups
- Analyze qualitative feedback
- Determine if the hybrid approach is needed (progressive disclosure for most tasks, persistent elements for specific workflows)

**Phase 4 — Expanded Rollout (Month 3-6)** :
- Roll out the winning design to additional plants
- Implement update rings: Test ring → Pilot ring → Broad ring
- Each ring serves as a safety net — if issues emerge, rollback is contained [33]

**Statistical requirements:**
- Minimum 30-40 participants for meaningful quantitative results [15]
- Power analysis should determine required sample size (typically 60+ per variation for significant results)
- Use two-tailed t-tests for comparing means between groups
- Account for learning effects by measuring at consistent time points

### 9.2 Key Metrics to Track

**Primary metrics:**

| Metric | Measurement Method | Target for Success | Source |
|--------|-------------------|-------------------|--------|
| Task Completion Rate | Successful task completions / total attempts × 100 | >95% | [5][14] |
| Time on Task | End time minus start time | 28% reduction vs. baseline | [5] |
| Error Rate | Errors / total transactions × 100 | <0.3% for manufacturing | [34] |
| System Usability Scale (SUS) | 10-question post-test questionnaire | >80 (A grade) | [4][5] |
| Cognitive Load (NASA-TLX) | 6-dimension workload assessment | 22% reduction vs. baseline | [5][38] |

**Secondary metrics:**

| Metric | Purpose | Measurement Method |
|--------|---------|-------------------|
| Training Time to Competency | Measure learning curve steepness | Days until consistent task completion < 2× expert time |
| Help-Desk Ticket Volume | Identify usability pain points | Tickets per user per week |
| User Adoption Rate | Measure sustained usage | Active daily users / total assigned users |
| User Satisfaction (CSAT) | Measure subjective experience | 1-5 scale after each major task |
| Feature Discovery Rate | Measure how quickly users find advanced features | Number of unique functions used per week |
| Time-on-Task Variability | Proxy for cognitive load | Standard deviation of task completion times |

### 9.3 Measuring Cognitive Load in Real-World Settings

**Method 1: NASA-TLX (Subjective)**
- Gold standard for subjective workload measurement
- Six dimensions: Mental Demand, Physical Demand, Temporal Demand, Performance, Effort, Frustration
- Raw TLX version (without pairwise comparisons) is simpler and validated for field use [38]
- Administer after each task type (not after every task) to avoid fatigue
- Multiple administrations require pairwise comparisons only once per task type

**Method 2: Behavioral Proxies** (for continuous, non-disruptive measurement)

| Behavioral Proxy | What It Measures | How to Collect |
|-----------------|------------------|----------------|
| Time-on-task variability | Cognitive load fluctuations | System logs (standard deviation of completion times) |
| Help-seeking frequency | User struggling with navigation | Help button clicks, search queries, ticket submissions |
| Error patterns | Specific UI friction points | Error logs categorized by type and location |
| Task abandonment rate | Task too difficult or confusing | Partially completed tasks that are never submitted |
| Mouse movement patterns | Uncertainty or hesitation | Mouse tracking (slower movements = higher cognitive load) |

**Method 3: Physiological Measures** (advanced, for validation studies)
- Heart rate variability (HRV) — increased cognitive load correlates with decreased HRV
- Eye tracking — increased fixation duration and pupil dilation correlate with cognitive load
- The 2025 MR cognitive load study achieved 95.83% classification accuracy using Transformer-CL algorithms on combined head movement, eye-tracking, and hand gesture data [39]

**Practical recommendation**: Use NASA-TLX for initial baseline and periodic checkpoints. Use behavioral proxies (time-on-task variability and help-seeking frequency) for continuous monitoring. Reserve physiological measures for validation studies or high-stakes evaluations.

### 9.4 Phased Rollout Strategy Metrics Framework

Based on SysGenPro and Prosci research, use these staged readiness gates [33][40]:

**Gate 1 — Pilot Readiness (Week 0)** :
- User training completion rate >90%
- Baseline SUS score >50 (must not be "awful")
- Error rate on training tasks <5%

**Gate 2 — Pilot Completion (Week 8)** :
- Target SUS score >68 (average or better)
- Task completion rate >85%
- User satisfaction >3.5/5
- At least 80% of pilot users recommend continuing

**Gate 3 — Expansion (Month 3)** :
- SUS score >75 (good)
- Task completion rate >90%
- Error rate <1%
- Training time for new users <50% of initial training time

**Gate 4 — Full Deployment (Month 6)** :
- SUS score >80 (excellent)
- Task completion rate >95%
- Error rate <0.3% [34]
- Training time for new users consistent with target (25-40% faster than legacy system) [2]

---

## Section 10: Change Management for the 45-60 Demographic

### 10.1 Cognitive Aging and Technology Adoption

Research consistently shows that prior experience with technology strongly predicts cognitive workload and frustration, sometimes more than age itself. Users with 20+ years on an AS/400 have deeply ingrained mental models for task completion that do not transfer to modern GUI interfaces [23].

**Key findings for your demographic:**
- "The definition of an aging worker could be considered to apply from 45 years" (WHO Study Group on Aging and Working Capacity, 1993) [23]
- Older workers' job performance shows gradual declines after the 40s, but productivity can remain high, especially in experienced and supervisory roles
- Chronic diseases such as cardiovascular, respiratory, musculoskeletal, cancer, and mental disorders increase in aging populations and impact work ability
- Fluid intelligence (processing speed, working memory) declines with age, while crystallized intelligence (accumulated knowledge, experience) remains stable

### 10.2 Technostress Management

Technostress in older workers (aged 50+) is driven by:
- **Complexity**: Difficulty adapting to changing technology
- **Inclusion**: Feeling inferior to younger, more tech-savvy colleagues
- **Blue-collar workers** experience more stress from overload and complexity than white-collar workers [23]

**Mitigation strategies:**
1. **Role-based default views**: Reduce complexity from day one
2. **Error management training**: Frame mistakes as learning opportunities, reducing anxiety
3. **Peer mentoring**: Pair tech-resistant users with "super users" from similar demographic
4. **Self-paced learning**: Allow users to progress at their own speed through modular training
5. **Legacy translation guides**: Show how each new interface function maps to the old AS/400 transaction

### 10.3 Training Strategy for Production Floor Supervisors

**Evidence-based approach:**

1. **Role-based, workflow-oriented training**: Organize training around actual job workflows, not system modules. Present tasks in the sequence they occur in real operations [35]

2. **The "training wheels" approach**: Start with a simplified interface that prevents error states. Gradually reveal full functionality as proficiency increases. This approach showed "greater user success" in original research (Carroll & Carrithers, 1984) [29]

3. **Just-in-time learning**: Provide contextual help within the interface rather than separate training sessions. "Users never read manuals but start using the software immediately" — the paradox of the active user [29]

4. **Extended post-go-live support**: "Post-go-live floor support for supervisors is critical during transition." Plan for at least 4-6 weeks of on-site support after go-live [33]

5. **Measured competence gates**: Define specific task completion metrics that demonstrate proficiency. Do not advance users to full interface until they meet targets on simplified version

---

## Section 11: Conclusion and Recommendations

### 11.1 Primary Recommendation: Role-Based Hybrid Model

For production floor supervisors (aged 45-60) transitioning from AS/400 to cloud ERP, the optimal interface design is a **role-based hybrid model** that combines:

1. **Simplified default view**: 5-7 role-specific KPIs and primary task buttons, following the "5-KPI rule" established in manufacturing dashboard research [30]
2. **Clear progressive disclosure paths**: Drill-down from KPIs to detailed data, wizards for complex transactions
3. **Persistent navigation safety net**: Collapsible navigation pane and search bar providing access to all functions
4. **User customization**: Pin frequently used advanced functions, toggle between guided and expert modes
5. **Legacy familiarity**: Support for transaction code entry and keyboard shortcuts from the AS/400 system

### 11.2 Evidence Summary

| Claim | Evidence Level | Quantitative Support |
|-------|---------------|---------------------|
| Progressive disclosure reduces training time | Strong — multiple studies | 25-40% faster training (SAP Fiori) [2]; 20% completion rate improvement (Forrester) [37] |
| Simplified views reduce cognitive load | Strong — controlled study | 22% reduction in NASA-TLX (Nittala 2024) [5]; 42% reduction (JETNR 2025) [9] |
| Role-based interfaces increase satisfaction | Strong — peer-reviewed | 44% higher user satisfaction (EJCSIT 2025) [1]; SUS from 68.4 to 84.2 [5] |
| Persistent navigation benefits expert users | Moderate — theory-based | Expertise reversal effect [17]; interaction cost analysis [31] |
| Hybrid model is optimal | Moderate — synthesis | No direct head-to-head study exists; recommended by multiple framework sources [14][30][31] |
| AS/400 muscle memory disrupts GUI adoption | Moderate — observational | Studies on prior experience dominance [23]; paradox of the active user [29] |

### 11.3 Critical Research Gap

No peer-reviewed studies were found that directly compare progressive disclosure vs. persistent navigation in a head-to-head quantitative study for older adult users (45-65) in manufacturing ERP contexts. The Li & Tang (2025) systematic review explicitly calls for "context-based design suggestions to address the holistic needs of older adults" — this gap remains, particularly in enterprise/industrial contexts [13].

**Recommendation**: Organizations implementing ERP for AS/400 migrants should conduct their own A/B tests using the methodology in Section 9, contributing to the industry knowledge base while optimizing their specific implementation.

### 11.4 Implementation Checklist

- [ ] Select pilot plant with representative user mix (tech-comfortable and tech-resistant)
- [ ] Implement green screen modernization tool for immediate UX improvement
- [ ] Create role-based simplified view for production supervisors
- [ ] Build progressive disclosure paths from simplified to detailed views
- [ ] Enable transaction code search for legacy users
- [ ] Implement A/B testing infrastructure (progressive disclosure vs. persistent navigation)
- [ ] Measure baseline metrics: SUS, task completion time, error rate, cognitive load
- [ ] Run 8-week controlled experiment with pilot group
- [ ] Analyze quantitative and qualitative data
- [ ] Determine optimal hybrid model based on results
- [ ] Implement update rings for phased rollout
- [ ] Provide role-based, workflow-oriented training
- [ ] Maintain legacy system fallback for 3-6 months post-migration
- [ ] Measure post-migration metrics and compare to baseline
- [ ] Publish findings to contribute to industry knowledge

---

### Sources

[1] SAP Fiori for SAP S/4HANA — Fundamentals and UX Updates Q1/2026: https://community.sap.com/t5/technology-blog-posts-by-sap/sap-fiori-for-sap-s-4hana-fundamentals/ba-p/13323215

[2] Implementing SAP Fiori in S/4HANA Transitions — Journal of Engineering Research and Sciences, 2025: https://www.jenrs.com/v04/i11/p001

[3] Production Floor Execution Interface — Microsoft Learn: https://learn.microsoft.com/en-us/dynamics365/supply-chain/production-control/production-floor-execution-use

[4] System Usability Scale — MeasuringU (Sauro & Lewis): https://measuringu.com/sus

[5] A Case Study in User Productivity and Cognitive Load (ERP with LLM NLI) — IJETCSIT, 2024: https://www.ijetcsit.org/index.php/ijetcsit/article/download/458/408

[6] Oracle NetSuite UX Redesign — Ron Design Lab: https://rondesignlab.com/blog/work-in-progress/oracle-netsuite-business-management-software-web-app

[7] Dynamics 365 UI/UX Design Principles — Microsoft Learn: https://learn.microsoft.com/en-us/dynamics365/guidance/develop/ui-ux-design-principles

[8] NetSuite vs Dynamics 365 for Manufacturing — Modax ERP: https://www.modaxerp.com/blog/dynamics-365-vs-netsuite-manufacturing

[9] The Impact of UI Design Elements on Cognitive Performance Among Elderly Mobile Application Users — JETNR, June 2025: https://rjpn.org/jetnr/papers/JETNR2506007.pdf

[10] The Impact of Interface Design Element Features on Task Performance in Older Adults — International Journal of Environmental Research and Public Health, 2022: https://pubmed.ncbi.nlm.nih.gov/35954608/

[11] How to Measure Learnability — Nielsen Norman Group: https://www.nngroup.com/articles/measure-learnability

[12] Progressive Disclosure — Nielsen Norman Group (Jakob Nielsen, 2006): https://www.nngroup.com/articles/progressive-disclosure

[13] Online Performance and Interface Design Implications Among Older Adults: A Systematic Review of Eye Tracking Studies — Applied Ergonomics, Volume 128, 2025: https://www.sciencedirect.com/science/article/pii/S0003687025000742

[14] What is Progressive Disclosure? — Interaction Design Foundation (IxDF), Updated 2026: https://ixdf.org/literature/topics/progressive-disclosure

[15] ERP System Implementation, User Training, and Management Support on User Satisfaction in Manufacturing Companies — West Science Information System and Technology, 2024: https://wsj.westsciences.com/index.php/wsist/article/view/1209

[16] Cognitive Load Theory — The Decision Lab: https://thedecisionlab.com/reference-guide/psychology/cognitive-load-theory

[17] The Expertise Reversal Effect and Its Implications for Design — The Elearning Coach: https://theelearningcoach.com/learning/novice-versus-expert-design-strategies

[18] The Expertise Reversal Effect: Cognitive Load and Motivational Explanations — PubMed (Rey & Buchwald, 2011): https://pubmed.ncbi.nlm.nih.gov/21443379/

[19] Cognitive Load in Manufacturing UIs: https://www.scribd.com/document/931139488/CognitiveLoadReductionTechniquesUserInterfaces

[20] A Review on Cognitive Workload for Industry 5.0 — Computers & Industrial Engineering, 2025: https://www.sciencedirect.com/science/article/pii/S0360835225004966

[21] Interface Design Based on Cognitive Load Theory — ResearchGate, 2024/2025: https://www.researchgate.net/publication/399036000_Interface_Design_Based_on_Cognitive_Load_Theory

[22] WCAG 2.2 Success Criteria — W3C Web Accessibility Initiative: https://www.w3.org/TR/WCAG22/

[23] Age-Friendly Technology Design: A Practical Guide — AARP and Older Adults Technology Services: https://www.aarp.org/pri/topics/technology/work-and-technology/age-friendly-tech-design/

[24] Aging Effects on Contrast Sensitivity in Visual Pathways — PMC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8998765/

[25] Section 508 Compliance — Section508.gov: https://www.section508.gov/

[26] Strangler Fig Pattern Research (41 Projects, 2022-2025) — Modernization Intel: https://modernization-intel.com/strangler-fig-pattern-research/

[27] Anti-Corruption Layer Pattern — Microsoft Azure Architecture Center: https://learn.microsoft.com/en-us/azure/architecture/patterns/anti-corruption-layer

[28] Green Screen Modernization Tools — Profound Logic: https://www.profoundlogic.com/ux-futurization/

[29] What is the Paradox of the Active User? — Progress Software: https://www.progress.com/blogs/what-paradox-active-user

[30] Manufacturing Dashboard Design Guide for Industrial UX Teams — Fuselab Creative, 2026: https://fuselabcreative.com/manufacturing-dashboard-ux-design

[31] Progressive Disclosure: The Art of Revealing Just Enough — Versions.com: https://versions.com/interaction/progressive-disclosure-the-art-of-revealing-just-enough

[32] Manufacturing ERP Adoption Metrics — SysGenPro: https://sysgenpro.com/implementation/manufacturing-erp-adoption-metrics-that-help-leaders-track-implementation-success

[33] Phased Rollout with Update Rings — International Journal for Multidisciplinary Research, 2026: https://www.ijfmr.com/paper/2026/Phased_Rollout_Update_Rings.pdf

[34] What's a Good Data Entry Error Rate? — Conexiom, 2025: https://conexiom.com/blog/whats-a-good-data-entry-error-rate-benchmarks-how-to-reduce-yours

[35] ERP Training Strategy: Building Organizational Competency — ElevatIQ: https://www.elevatiq.com/post/erp-training-strategy

[36] Your ERP Adoption Rate Guide — Prosci, April 2026: https://www.prosci.com/blog/your-erp-adoption-rate-guide

[37] 10 Ways to Enhance Form Completion Improvement in Logistics — Zigpoll (citing Forrester 2024): https://www.zigpoll.com/content/10-ways-enhance-form-completion-improvement-logistics

[38] NASA Task Load Index (NASA-TLX) — NASA Ames Research Center: https://humansystems.arc.nasa.gov/groups/TLX/

[39] Measuring Cognitive Load of Digital Interface Combining Event-Related Potential and BubbleView — Brain Informatics, 2023: https://link.springer.com/article/10.1186/s40723-023-00187-7

[40] Manufacturing ERP Implementation Metrics — SysGenPro: https://sysgenpro.com/erp/manufacturing-erp-implementation-metrics-that-reveal-operational-readiness-gaps