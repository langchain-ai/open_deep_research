Based on the comprehensive research conducted, here is the detailed answer to your research brief.

# ERP Interface Design for Legacy-to-Cloud Migration: Progressive Disclosure vs. Persistent Navigation for Production Floor Supervisors

## Executive Summary

This report presents a comprehensive analysis of how **progressive disclosure** (exposing information and actions on demand) versus **persistent navigation** (displaying all options and menus continuously) affects task completion rates, training time, and user proficiency for production floor supervisors (aged 45-60) transitioning from legacy AS/400 systems to cloud-based ERP platforms.

The research draws on documented implementations from **SAP S/4HANA (Fiori vs. GUI)**, **Oracle NetSuite**, and **Microsoft Dynamics 365**, alongside change management literature, cognitive load studies, and metrics from successful migrations.

**Key Findings:**
1. Progressive disclosure consistently reduces training time and cognitive load for novice or transitioning users, but can limit efficiency for experienced power users.
2. Role-based views significantly outperform comprehensive dashboards for the target demographic, particularly when they include clear pathways to access advanced functions.
3. Structured competency management, combined with progressive interface design, can reduce time-to-competency by up to 38% and errors by up to 90%.
4. For users aged 45-60, the primary barrier is not age-related inability to learn, but rather interface complexity, technostress, and the discomfort of unlearning deeply ingrained mental models from legacy systems.

---

## 1. ERP Platform Implementations: Progressive Disclosure vs. Persistent Navigation

### 1.1 SAP S/4HANA: Fiori (Progressive Disclosure) vs. GUI (Persistent Navigation)

SAP S/4HANA presents the clearest contrast between the two navigation paradigms because it offers two distinct interfaces: the modern **SAP Fiori** and the legacy **SAP GUI**.

#### Design Philosophy and Navigation
**SAP Fiori** is built on a progressive disclosure model. It is a role-based, tile-driven interface where users see only the apps relevant to their job function. The SAP Fiori design system is governed by five core principles: role-based, adaptive (responsive), simple, coherent, and delightful [SAP Community - SAP Fiori Fundamentals] [2]. The interface employs a 1:1:3 approach—one user, one use case, and three screens (desktop, tablet, mobile)—and uses an "intuitive drill-down chain" where users start with high-level indicators (KPIs, tiles) and progressively interact to reach granular transactional data [SAP Community - Situation Handling - Progressive Disclosure] [1].

**SAP GUI**, in contrast, represents persistent navigation. It is a traditional, text-heavy, form-based interface that provides access to all system functionalities at all times via a menu-driven structure and transaction codes (T-codes). It is primarily keyboard-driven and optimized for power users who memorize shortcuts [Pathlock - SAP GUI vs SAP Fiori] [9]. This interface requires installation on a local machine and is desktop-centric.

#### Evidence from Implementations
**Neogen Corporation Case Study (PwC/ASUG):** This 18-month project migrated from heavily customized SAP ECC (SAP GUI) to SAP S/4HANA Fiori. The company achieved 90% Fiori usage with minimal customization. Key challenges included "resistance to transition from ECC transactions to Fiori apps" and a "cultural shift from Excel dependency." The documented benefits included a **reduction in transaction time**, **reduced training costs**, and improved user satisfaction [ASUG Insights - Simplified Manufacturing through S/4HANA Fiori Apps] [16].

**Sandvik Mining and Rock Technology (KTH Thesis):** Research on this implementation found that technical stabilization was not the primary driver of early success; rather, **user readiness** was the decisive factor. The study noted a "heavy reliance on super users" and insufficient support from key users, underscoring the need to strengthen user competence before and during the transition [DIVA Portal - Larsson 2025] [17].

#### Specific Workflow Analysis (Inventory & Work Orders)
SAP Fiori offers specific apps for these workflows:
- **Inventory Management:** Apps like `Stock Multiple Material`, `Stock Information`, and `Post Goods Receipts` provide a simplified, visual interface compared to the GUI's multi-screen transaction.
- **Work Orders:** The `Manage Process Orders` app helps supervisors manage order progress and related details with enhanced visualization and embedded analytics [SAP Community - Highlights for Manufacturing in S/4HANA 2020] [30].

**Key Takeaway:** Fiori's progressive disclosure reduces the visual chaos of the GUI. However, the learning curve for the *conceptual* shift from a system-centric (SAP GUI) to a user-centric (SAP Fiori) model is significant, especially for long-term users of legacy systems.

### 1.2 Oracle NetSuite: Progressive Disclosure with Role-Based Centers

Oracle NetSuite's official documentation confirms that progressive disclosure is its core design philosophy: *"The UI uses progressive disclosure; pages focus on your content, and relevant links, menus, and icons appear when moving your pointer over areas"* [Oracle NetSuite - Navigating NetSuite] [1]. This approach reduces clutter by showing only the most important content until hovered over.

#### Role-Based Centers
NetSuite's navigation is structured around **Centers** (e.g., Accounting Center, Sales Center). These are tabbed pages that display a variable set of dashboards based on the user's assigned role [Oracle NetSuite - Centers Overview] [20]. This allows for a highly tailored experience where a production floor supervisor might have a "Manufacturing Center" showing only relevant inventory and work order dashboards.

- **Progressive Disclosure Mechanism:** NetSuite utilizes center tabs, custom links, and hover-based menus. The global search field supports progressive disclosure by revealing options upon hover [NetSuite Training Video] [15].
- **The Redwood Experience:** The recent Redwood UI update (2024-2025) further enhances this with collapsible form sections, improved dashboard personalization, and AI-powered tools like "Ask Oracle" for natural language search [Houseblend.io] [6].

#### Evidence from Implementations
**Cornell Manufacturing (Terillium Case Study):** Transitioning from an outdated JD Edwards system to NetSuite, the company eliminated spreadsheet dependency and improved operational visibility. The role-based dashboards were central to this success, providing finance and supply chain staff with relevant, real-time data [Terillium - Cornell Manufacturing] [19].

**Mishimoto (Folio3):** This automotive manufacturer implemented SuiteSuccess Manufacturing Premium. The implementation included advanced order routing and multi-warehouse support, with **up to 30 hours of training** provided to ensure smooth adoption of the new role-based interface [Folio3 - Mishimoto Case Study] [16].

#### Specific Workflow Analysis (Inventory & Work Orders)
- **Inventory Management:** NetSuite's 2026.1 release includes a new Consigned Inventory Management feature and enhanced Warehouse Management capabilities like landed cost validation, barcode scanning, and inbound shipment reversals [Oracle NetSuite - Inventory Management 2026.1] [4][5].
- **Work Orders:** The system supports a full lifecycle from planning (via MRP) to BOM explosion, component picking, completion, and closing. NetSuite allows for both simple assembly builds and full work orders with multi-step routing [BrokenRubik - NetSuite Work Orders Guide] [8].

**Key Takeaway:** NetSuite provides a strong progressive disclosure model that is highly configurable through Centers. Its primary challenge for AS/400 migrants is the shift from a keyboard-driven "green screen" to a mouse-driven, graphical role-based interface.

### 1.3 Microsoft Dynamics 365: Hybrid Model with Production Floor Focus

Microsoft Dynamics 365 offers a hybrid model that combines persistent navigation elements (like the global navigation pane) with powerful progressive disclosure mechanisms, especially in the manufacturing context.

#### Core Navigation Concepts
- **Dashboard:** The main landing page.
- **Navigation Pane:** A persistent, expandable left-side menu providing access to Favorites, Recent, Workspaces, and Modules [Microsoft Learn - Navigation Concepts] [12].
- **Workspaces:** These are activity-oriented pages that provide targeted information for a specific role.
- **Action Pane:** A persistent toolbar that can be collapsed, pinned, or expanded, representing persistent navigation. It also features an **Action Search** (Ctrl+’), which is a progressive disclosure tool for executing commands [Microsoft Learn - Action Controls] [13].

#### The Role Center (Business Central)
Dynamics 365 Business Central uses **Role Centers** as the user’s home page, based on a user-centric design model. These centers provide quick access to relevant tasks, KPIs, and data visualizations [Microsoft Learn - Designing Role Centers] [12]. For example, a Finance Manager sees cash flow and overdue invoices, while a Warehouse Worker sees real-time inventory movements and pending picks [EazyDynamics] [21].

#### Progressive Disclosure in Practice: The Production Floor Execution Interface
This is the most significant example of progressive disclosure for your target demographic. The **production floor execution interface** in Supply Chain Management is designed specifically for shop floor workers and supervisors. It is optimized for touch, has visual contrast for accessibility, and focuses on job registration (start, report feedback, register indirect activities) [Microsoft Learn - How workers use the production floor execution interface] [15].

- **Configuration:** The interface is highly configurable. An administrator can toggle optional features on/off to control what is shown—material consumption registration, scrap reporting, serial number tracking, etc. This creates a "progressive enabling" system where complexity is added as needed for specific roles or locations [Microsoft Learn - Configure the production floor execution interface] [16].
- **Navigation:** Workers sign in using badge IDs, and the interface auto-loads configurations specific to their device, presenting a simplified, task-focused view.

#### Evidence from Implementations
**Planar and MCIA (Sikich Case Studies):** Modernized global operations and improved efficiency using Dynamics 365 F&SCM. The focus was on simplifying complex processes for end-users [Sikich] [2].
**Omega Industries (Encore Business Solutions):** Transitioned from Dynamics GP to Business Central, achieving a "30% reduction in manual work" and full user adoption due to streamlined, role-based workflows [Encore Business Solutions] [4].

**Key Takeaway:** Dynamics 365 is arguably the most flexible of the three platforms for managing the transition. Its production floor execution interface is a textbook example of progressive disclosure for manufacturing workers. The "Role Center" approach effectively serves as a middle ground between a completely stripped-down view and a full "power user" dashboard.

---

## 2. Change Management & Cognitive Load During System Transitions

### 2.1 Cognitive Load Theory (CLT) Applied to ERP Transitions

Cognitive Load Theory (CLT) explains why system transitions are so difficult for users. It distinguishes between three types of cognitive load:

1.  **Intrinsic Load:** The inherent complexity of the task (e.g., creating a work order).
2.  **Extraneous Load:** Unnecessary cognitive demands caused by poor interface/instructional design (e.g., a cluttered dashboard, confusing menu structure).
3.  **Germane Load:** The effort dedicated to learning and building new mental models.

For users transitioning from AS/400 systems, the **intrinsic load** is amplified by the need to learn new business processes, not just a new interface. **Extraneous load** is high when they encounter a modern, visually dense interface like a comprehensive dashboard. The goal of progressive disclosure is to minimize extraneous load to allow for germane load (learning) [The Decision Lab] [1].

### 2.2 Interface Complexity and Older Users (Ages 45-60)

Research shows that cognitive aging primarily affects *fluid intelligence* (processing speed, working memory), while *crystallized intelligence* (accumulated knowledge, experience) remains stable. This leads to specific challenges:

- **Prior Experience Dominates:** Prior experience with technology strongly predicts cognitive workload and frustration, sometimes more than age itself. Users with 20+ years on an AS/400 have deeply ingrained mental models for task completion that do not transfer to modern GUI interfaces [PDXScholar - Technology-Based Training for Older Employees] [25].
- **50-100% More Time:** Older adults often take 50-100% more time on tasks than younger adults when navigating complex, unfamiliar interfaces [PDXScholar] [25].
- **Technostress:** This is a major barrier. Studies show that technostress in older workers (aged 50+) is driven by **complexity** (difficulty adapting to changing tech) and **inclusion** (feeling inferior to younger, more tech-savvy colleagues). Blue-collar workers experience more stress from overload and complexity than white-collar workers [MyOSH - Technostress in Older Workers] [18].

### 2.3 Key Findings for Your Demographic (AS/400 Transition)

- **The "Strangler Fig" Approach:** Purely replacing an AS/400 system creates maximum cognitive and emotional strain. A more humane approach involves creating an orchestration layer that allows new systems to grow alongside the legacy one, giving teams a shared foundation and time to adapt [The Business News] [28].
- **Training Style is Critical:** For older learners, training should be self-paced, work-integrated, and offer high instructional coherence. Error management training (framing mistakes positively) reduces anxiety [IZA World of Labour] [38].
- **The Paradox of the Active User:** Users often plateau at a "mediocre" performance level because they stick to methods they know. Persistent navigation (like AS/400 menus) can trap users in beginner mode. Progressive disclosure is designed to prevent this by gradually increasing functionality [Interaction Design Foundation] [7].

---

## 3. Metrics & Methodologies for Measuring Success

Successful migrations use a combination of academic frameworks and granular operational metrics.

### 3.1 Primary Frameworks for Evaluation

- **Kirkpatrick's Four Levels:** Essential for evaluating training effectiveness.
    - **Level 1 (Reaction):** User satisfaction surveys.
    - **Level 2 (Learning):** Pre/post tests on system knowledge.
    - **Level 3 (Behavior):** Observation of on-the-job system usage and workflow application.
    - **Level 4 (Results):** Business impact, e.g., inventory accuracy, order processing time [Epilogue Systems] [18].
- **Technology Acceptance Model (TAM) / UTAUT:** These models focus on *perceived usefulness* and *perceived ease of use* as drivers of adoption. Studies show "effort expectancy" (how easy the system is to use) is a primary predictor of behavioral intention to use an ERP system [Asia-Pacific Management Accounting Journal] [11].
- **DeLone & McLean IS Success Model:** Evaluates success through System Quality, Information Quality, Service Quality, User Satisfaction, and Net Benefits [Emerald Insight] [11].

### 3.2 Specific Operational Metrics & Benchmarks

- **Error Reduction:**
    - *Target:* Reduce from typical manual data entry error rates of **1-4%** [Lido.app] [13].
    - *Evidence:* Aberdeen Group found that well-trained workforces experience **41% fewer process errors** [ECI Solutions] [8]. Structured competency management can lead to a **90% reduction in errors and incidents** [Seertech Solutions] [7].
- **Time-to-Competency:**
    - *Target:* Structured competency management can reduce time-to-productivity by **38%** , translating to up to $1M in yearly savings [Seertech Solutions] [7].
    - *Benchmark:* Video-based work instructions can cut time-to-competency by **40%** [Manual.to] [6].
- **Training Time & Budget:**
    - *Nielsen PwC Case Study:* Fiori implementation led to "reduced training costs" [ASUG Insights] [16].
    - *Best Practice:* Projects allocating **7-17% of their budget** to training are significantly more successful [SAP Community - The ERP Training Life Cycle] [14].
- **User Adoption Rate:**
    - *Target:* Project with strong change management and role-based dashboards can achieve **90% user adoption** within 6 months [OCM Solution] [12].
    - *Risk:* A study of NetSuite found that only **26% of employees** use the ERP system on average, often because dashboards are not role-specific [Kimberlite Partners] [13].

---

## 4. Synthesis: Role-Based Views vs. Comprehensive Dashboards

### 4.1 The Core Trade-off

This is the central question of your design challenge.

- **Comprehensive Dashboards (Persistent Navigation):** Provide full functionality and data visibility at all times. They are excellent for power users who need to perform a wide range of tasks and value a high degree of situational awareness. However, they suffer from **feature overload**, leading to choice paralysis, higher error rates, and significantly longer training times for novice users.

- **Role-Based Views (Progressive Disclosure):** Simplify the interface to show only the data and actions relevant to a specific job function. This dramatically reduces extraneous cognitive load, speeds up primary task completion, and shortens training. The risk is that they can "lock users in," making it difficult to discover and access advanced or infrequently used features.

### 4.2 Evidence for Your Target Demographic

For the core user (production floor supervisor, 45-60, legacy system experience), the evidence strongly favors **role-based views**.

- **Cognitive Load Reduction:** Role-based design reduces support escalations by **30-50%** and compresses onboarding time by **40-65%** [Reloadux] [7]. Applying CLT principles to dashboards (e.g., showing only 5-7 KPIs) can enable users to find information over **50% faster** [Fegno] [9].
- **The "20% Rule":** The most effective role-based views satisfy roughly 80% of a user's daily tasks. The key is designing for the other 20%—the occasional need for advanced functions.
- **Occasional Advanced Functions:** Best practices for this "20% case" include:
    - **Clear Progressive Disclosure Paths:** Use "drill-down" links and "more options" buttons that are clearly labeled. The test of a dashboard is whether it answers the *next* question without leaving the current view [Nielsen Norman Group] [3].
    - **Ephemeral Adaptation:** Techniques like "ephemeral adaptation" (showing important commands immediately while fading others) help maintain discoverability without cluttering the interface [Interaction Design Foundation] [7].
    - **Toggle/Switching:** Provide an expert mode toggle or context-aware help that guides users to the advanced feature they need.

### 4.3 The Recommended Hybrid Model for Your Brief

The most successful implementations for your demographic use a **hybrid progressive model**:

1.  **Default: Simplified Role-Based View.** The supervisor's home screen should show a clean dashboard with only the most critical tasks (e.g., "View Today's Work Orders," "Check Inventory for Order X," "Report Overtime").
2.  **Clear Path to Depth:** Every element on this dashboard should be clickable or tappable to drill down into a more comprehensive view. A KPI tile for "Inventory Alerts" should lead to the full inventory management module. This is **progressive disclosure**.
3.  **Persistent Navigation as a Tool:** The persistent navigation pane (e.g., Dynamics 365's module pane or SAP Fiori's Launchpad) should be available but not necessary for primary tasks. It serves as a safety net for users who are ready to explore or need to perform an unusual action.
4.  **"Pinning" and "Creating Shortcuts":** Allow users to pin frequently used advanced functions to their simplified view. This empowers them to gradually build a persistent, customized workspace without being overwhelmed from day one.

**Conclusion:** Do not design a "dumbed-down" interface with no way to advance. Design a **smart default interface** that reduces cognitive load for the primary 80% of tasks, while providing clear, intuitive, and reversible pathways to the remaining 20% of advanced functionality. This directly addresses the needs of a transitioning workforce: it offers the simplicity required to get started and the power needed to grow.

---

### Sources

[1] SAP Community - Situation Handling – Navigation Concept & Progressive Disclosure: https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-sap/situation-handling-navigation-concept-progressive-disclosure/ba-p/13476639
[2] SAP Community - SAP Fiori for SAP S/4HANA – Fundamentals: https://community.sap.com/t5/technology-blog-posts-by-sap/sap-fiori-for-sap-s-4hana-fundamentals/ba-p/13323215
[3] Nielsen Norman Group - Interface Design and Research: https://www.nngroup.com/
[4] Oracle NetSuite - Navigating NetSuite: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_N474404.html
[5] Oracle NetSuite - Centers Overview: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/chapter_N404881.html
[6] Houseblend.io - The Redwood Experience: https://www.houseblend.io/blog/redwood-experience
[7] Interaction Design Foundation - Progressive Disclosure: https://www.interaction-design.org/literature/topics/progressive-disclosure
[8] Microsoft Learn - Navigation concepts - Finance & Operations: https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/fin-ops/get-started/navigation-concepts
[9] Microsoft Learn - How workers use the production floor execution interface: https://learn.microsoft.com/en-us/dynamics365/supply-chain/production-control/production-floor-execution-use
[10] Microsoft Learn - Designing Role Centers - Business Central: https://learn.microsoft.com/en-us/dynamics365/business-central/dev-itpro/developer/devenv-designing-role-centers
[11] Pathlock - SAP GUI vs SAP Fiori - A Comprehensive Guide: https://pathlock.com/blog/sap-fiori/sap-gui-vs-sap-fiori
[12] ASUG Insights - Simplified Manufacturing through S/4HANA Fiori Apps (PwC/Neogen): https://blog.asug.com/hubfs/Chapter%20Events/PwC.pdf
[13] DIVA Portal - Mastering SAP S/4HANA implementation in manufacturing (Larsson 2025): https://www.diva-portal.org/smash/get/diva2:2012164/FULLTEXT01.pdf
[14] SAP Community - Highlights for Manufacturing in SAP S/4HANA 2020: https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-sap/highlights-for-manufacturing-in-sap-s-4hana-2020/ba-p/13481951
[15] Folio3 - Mishimoto Case Study (NetSuite): https://netsuite.folio3.com/success-stories/mishimoto
[16] Terillium - Cornell Manufacturing Case Study: https://terillium.com/case-study/cornell-manufacturing-transforms-operations-with-transition-from-jd-edwards-to-netsuite
[17] The Decision Lab - Cognitive Load Theory: https://thedecisionlab.com/reference-guide/psychology/cognitive-load-theory
[18] PDXScholar - Technology-Based Training for Older Employees: A Literature Review: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1126&context=honorstheses
[19] MyOSH - Technostress in Older Workers: https://www.myosh.com/news/technostress-in-older-workers-a-growing-challenge
[20] IZA World of Labour - Is training effective for older workers?: https://wol.iza.org/articles/is-training-effective-for-older-workers/long
[21] Lido.app - Data Entry Error Rates: https://www.lido.app/blog/data-entry-error-rates
[22] ECI Solutions - Maximize Manufacturing ROI: How ERP Training Boosts Efficiency: https://www.eci.com/blog/maximize-manufacturing-roi-erp-training
[23] Seertech Solutions - How Manufacturers Can Accelerate Time-to-Productivity: https://seertechsolutions.com/blog/accelerate-time-to-productivity-with-competency-management/
[24] Manual.to - Manufacturing Training ROI: https://www.manual.to/blog/manufacturing-training-roi
[25] SAP Community - The ERP Training Life Cycle: https://community.sap.com/t5/enterprise-resource-planning-blog-posts/the-erp-training-life-cycle/ba-p/13457815
[26] Epilogue Systems - The Kirkpatrick Model: 4 Levels Of Training Evaluation: https://epiloguesystems.com/blog/the-kirkpatrick-model
[27] OCM Solution - How to Do Change Management for ERP Software Implementation: https://www.ocmsolution.com/best-change-management-for-erp-implementation
[28] Kimberlite Partners - Best Practices for NetSuite Dashboards with Role-Based Functions: https://www.kimberlitepartners.com/blog/netsuite-role-based-dashboards
[29] Reloadux - Role-Based Design Systems for SaaS: Reduce Cognitive Overload: https://reloadux.com/blog/role-based-design-systems-for-saas-reduce-cognitive-overload
[30] Fegno - Designing Enterprise Dashboards with Cognitive Load Theory: https://www.fegno.com/designing-enterprise-dashboards-with-cognitive-load-theory
[31] The Business News - You don't have to replace your old system to escape it: https://thebusinessnews.com/northeast/you-dont-have-to-replace-your-old-system-to-escape-it
[32] Asia-Pacific Management Accounting Journal - ERP Post-Implementation Phase: UTAUT Model: https://apmaj.uitm.edu.my/
[33] Emerald Insight - E-government system evaluation: The case of users' performance: https://www.emerald.com/insight/content/doi/10.1108/tg/.../full/html
[34] EazyDynamics - Role Centres in Microsoft Dynamics 365 Business Central: https://www.eazydynamics.com/blog/role-centres-in-microsoft-dynamics-365-business-central
[35] Sikich - Smarter Manufacturing Production Operations with Dynamics 365: https://www.sikich.com/insight/smarter-production-leaner-manufacturing-operations-the-case-for-dynamics-365-finance-supply-chain-management
[36] Encore Business Solutions - Case Studies (Omega Industries): https://www.encorebusiness.com/company/case-studies
[37] Microsoft Learn - Action controls - Finance & Operations: https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/fin-ops/get-started/action-controls
[38] Microsoft Learn - Configure the production floor execution interface: https://learn.microsoft.com/en-us/dynamics365/supply-chain/production-control/production-floor-execution-configure
[39] Oracle NetSuite - Inventory Management 2026.1: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_1508852924.html
[40] Oracle NetSuite - New NetSuite 2026.1 Capabilities: https://www.netsuite.com/portal/resource/articles/erp/new-netsuite-2026-1-inventory-pricing-connector-and-warehouse-management-ai-capabilities-help-optimize-business-operations.shtml
[41] BrokenRubik - NetSuite Work Orders: Manufacturing Production Guide: https://www.brokenrubik.com/netsuite-work-orders-manufacturing-guide
[42] NetSuite Training Video - Navigating NetSuite: https://www.youtube.com/watch?v=FovtaTAPIZw