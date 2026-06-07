# Revised Research Report: ERP Interface Design for Legacy-to-Cloud Migration
## Progressive Disclosure vs. Persistent Navigation for Production Floor Supervisors

---

## Executive Summary

This report provides a comprehensive, evidence-based analysis of how **progressive disclosure** (revealing information and actions on demand) versus **persistent navigation** (displaying all options continuously) affects task completion rates, training time, and user proficiency for production floor supervisors (aged 45-60) transitioning from legacy AS/400 systems to cloud-based ERP platforms.

The research integrates **granular quantitative usability data** from peer-reviewed studies, **detailed platform-specific UI analysis** of SAP S/4HANA Fiori, Oracle NetSuite Redwood, and Microsoft Dynamics 365, and **comprehensive change management frameworks** (Kirkpatrick's Four Levels, UTAUT, Cognitive Load Theory) mapped to concrete, time-bound metrics.

**Key Findings:**

1. **Progressive disclosure significantly reduces cognitive load and training time for transitioning users.** Peer-reviewed studies show a 42% reduction in NASA-TLX cognitive load scores and a 37% increase in task completion rates when simplified navigation is used for older adults [4]. Hidden navigation increases task completion time by 39% on desktop compared to visible navigation [1].

2. **Older adults (aged 45-60+) strongly benefit from simplified role-based views over comprehensive dashboards.** Working memory capacity declines from 5-7 items (younger adults) to approximately 3.8 items (older adults), making interface complexity a critical barrier [5, 16, 20]. The Journal of Emerging Trends study (2025) found that simplified navigation improved task completion by 37% and reduced errors by 55% in older users [4].

3. **The key design insight is not age-related inability to learn, but the mismatch between interface complexity and reduced working memory capacity.** Older adults maintain domain expertise from AS/400 use (crystallized intelligence) but require interface designs that respect age-related declines in fluid intelligence and visual search efficiency [16, 19, 20].

4. **SAP S/4HANA Fiori offers the strongest role-based progressive disclosure model; Oracle NetSuite Redwood provides the most modern consumer-grade experience with AI integration; Microsoft Dynamics 365 offers the most flexible hybrid approach with business process-driven progressive disclosure.**

5. **A hybrid progressive model—simplified role-based views with clear pathways to advanced functions—is the optimal design for the target demographic, supported by all three change management frameworks with specific numeric targets and phase-gate milestones.**

---

## 1. Quantitative Usability Data: Progressive Disclosure vs. Persistent Navigation

### 1.1 Core Comparative Study: Hidden vs. Visible Navigation

The most directly relevant quantitative comparison of hidden/progressive vs. visible/persistent navigation comes from the **Nielsen Norman Group study by Pernice and Budiu (2016)** [1, 2, 3]. This study provides specific numeric findings that directly inform the design tradeoff:

**Study Design:**
- 179 participants (aged 20+)
- 6 live websites tested with hidden, visible, and combination navigation patterns
- Remote testing platform (WhatUsersDo)
- 6 tasks per participant (3 requiring navigation)
- Data analyzed using R statistical software

**Key Quantitative Findings:**

| Metric | Hidden (Progressive) Navigation | Visible (Persistent) Navigation | Impact |
|--------|--------------------------------|--------------------------------|--------|
| Navigation discoverability (desktop) | 27% usage rate | ~50% usage rate | 23 percentage point reduction |
| Time to access navigation (desktop) | 5-7 seconds longer | Baseline | Significant delay |
| Content discoverability | >20% lower | Baseline | Significant reduction |
| Perceived task difficulty | 21% higher | Baseline | Significant increase |
| Task completion time (desktop) | 39% longer | Baseline | Significant increase |
| Task completion time (mobile) | 15% longer | Baseline | Moderate increase |
| Hidden navigation usage (mobile) | 57-86% | N/A | More discoverable on mobile |

**Key Quote:** "Hidden navigation significantly decreases user experience both on mobile and on desktop, leading to more than 20% drop in content discoverability, higher task difficulty ratings, and longer task times." [1]

**Critical Implications for ERP Design:**
- On desktops (the primary interface for production floor supervisors), hiding navigation degrades the experience more than on mobile
- Hidden navigation causes: low salience, poor information scent, extra interaction cost to expand menus, lack of design standards
- However, this study tested general websites, not enterprise ERP systems. Enterprise users may have different motivations and tolerance for navigation complexity

### 1.2 Elderly-Optimized UI Study (2025): Statistical Breakdown by Age

The **Journal of Emerging Trends and Novel Research (JETNR) study published in June 2025** provides the most directly relevant quantitative data for older adult users [4]:

**Study Design:**
- 127 participants (aged 65 to 85)
- Compared standard UI designs vs. elderly-optimized UIs
- Four application types: medication management, messaging, news reading, public transportation
- Elderly-optimized designs: 16-18pt sans-serif fonts (7:1 contrast), simplified navigation (flat hierarchies), larger touch targets (minimum 12mm² with 4mm spacing), multimodal feedback, reduced visual clutter, extended timing parameters

**Granular Quantitative Findings:**

| Metric | Standard UI | Optimized UI | Improvement |
|--------|-------------|--------------|-------------|
| Task completion rate | 63.2% | 86.7% | **+37% increase** |
| Error frequency | Baseline | Reduced by 55% | **55% reduction** |
| Task completion time | Baseline | Decreased by 27% | **27% faster** |
| Navigation efficiency | Baseline | Improved by 50% | **50% improvement** |
| NASA-TLX cognitive load | Baseline | Decreased by 42% | **42% reduction** |
| Mental demand (NASA-TLX subscale) | Baseline | Decreased by 42% | **42% reduction** |
| Frustration (NASA-TLX subscale) | Baseline | Decreased by 48% | **48% reduction** |

**Subgroup Analysis:**
- Older participants (76-85) benefited **most** from UI optimizations
- Participants with lower prior technology experience benefited **most**
- Simplified navigation had the **strongest positive impact** on task completion
- Followed by increased touch target size and reduced clutter

**Qualitative Findings:**
- Decreased navigation uncertainty
- Reduced timing pressure
- Improved user confidence through multimodal feedback
- Enhanced visual discrimination via better typography and contrast
- Preference for **progressive disclosure of information**

**Study Limitations (flagged):**
- Laboratory testing conditions (not field study)
- Excluded users with significant cognitive impairments
- Short-term performance assessment (no longitudinal data)
- Tested mobile applications, not enterprise/ERP systems

### 1.3 Depth vs. Breadth Tradeoff: Age-Specific Quantitative Findings

**Zaphiris, Kurniawan & Ellis (2003)** investigated age-related differences in the depth vs. breadth tradeoff in hierarchical online information systems [5, 6]:

**Study Design:**
- 24 younger adults (aged ≤36)
- 24 older adults (aged ≥57)
- Two web-based hierarchical structures: expandable and non-expandable menus
- Three depth levels: 2, 3, and 6 levels
- Health-related information search tasks

**Key Quantitative Findings:**

| Finding | Numeric Detail |
|---------|----------------|
| Overall preference | Shallow hierarchies preferred over deep hierarchies |
| Older adult speed | Older participants took **1.37-1.49x longer** to find information (generalized aging ratio of 1.48) |
| Error rates by age | Older adults did **not make significantly more errors** than younger participants |
| Error rates by depth | Error rates increased from **4.0% to 34.0%** as depth increased from 1 to 6 levels |
| Expandable vs. non-expandable | Older adults **preferred non-expandable** hierarchies (all options visible) |
| Speed-accuracy tradeoff | Older adults **sacrifice speed to maintain accuracy** |
| Effect of computer experience | Computer experience did **not alter** age-related differences |

**Critical Finding:** "Age and depth had significant effects on search effectiveness, with seniors disproportionately slower in deeper hierarchies." [5]

**Kurniawan, Zaphiris & Ellis (2002)** further found [7, 8]:

| Metric | Finding |
|--------|---------|
| Traversal time ratio | Older users were 1.37-1.49x slower than younger users |
| Error rates (expandable vs. sequential) | Expandable hierarchy resulted in **fewer errors** than sequential hierarchies |
| Speed-accuracy tradeoff | Older adults sacrifice speed to maintain accuracy |
| A 4x16 structure (breadth=16 at 4 levels) | Had the fastest response times and fewest errors |

### 1.4 NASA-TLX Reliability in Older Adults

The psychometric properties of NASA-TLX were validated for older adults in a 2020 study [9]:

- **38 participants aged 65+** (including cognitively normal individuals with/without elevated amyloid PET scans and those with mild cognitive impairment)
- NASA-TLX demonstrated **good to excellent test-retest reliability** (ICCs: 0.71-0.81)
- Moderate correlations with event-related potential P3 component (Pearson's r: 0.30-0.33)
- **Key Quote:** "Subjective self-recall of cognitive workload is reliable across the spectrum of cognitive aging and has the potential to be used as a measure of attention allocation in this population"

### 1.5 Enterprise/ERP-Specific Quantitative Usability Findings

**Adaptive UI Personalization in ERP Systems (Boulevard System)** [10]:
- A proposed adaptive interface for SAP ERP using progressive disclosure with ephemeral visualization
- "Adaptive disclosure with ephemeral visualization reduces the visual search time in complex screens while maintaining spatial consistency"
- Brief questionnaire with **6 professional SAP users** showed positive reception (very small sample, flagged)
- Separates adaptive elements from static ones to reduce user disorientation

**ERP Usability Evaluation Tools** [11, 12]:
- Morae Manager provides key metrics for SAP ERP usability: number of mouse clicks, keystrokes, window dialogs opened, task completion time, error rates, and task scores
- ERP usability evaluation criteria: Navigation, Presentation, Task Support, Learnability, Customization

**Comprehensive ERP Usability Study (Asif et al., 2022, MDPI)** [13]:
- 67 unique usability problems identified from 11 research papers
- Six major usability problem topics, generalizable across ERP systems
- Most common problems: "difficulty searching and finding desired item/information in interface and error handling" and "missing data and information"
- **Recommendation:** "Projects need to spend at least 10% of their budget on usability to increase their effectiveness by 100% on sales, 161% on user productivity, and 202% on specific target features"

**UI/UX Design in Enterprise Systems Review (Purwandari & Dewi, 2025)** [30]:
- From 82 empirical articles (2019-2024)
- User-Centered Design (UCD) methodologies resulted in approximately **20-30% better usability scores** and faster task completion
- Adaptive interfaces (AI-driven) showed potential to **increase task efficiency by about 25-30%** and reduce error rates

**SAP Fiori Tools UX Benchmarking** [15]:
- SAP Fiori tools users made **fewer mistakes**, reducing overall development time
- Implementing a smart bullet micro chart: development time reduced from **up to 5 hours manually to under 10 minutes** using the Guided Development extension (progressive disclosure-based tool)
- Usability measured using **UMUX-lite** survey (two-question standard tool)

### 1.6 Systematic Review of Age-Friendly Mobile App Design (2025)

A systematic review published in **Aging Clinical and Experimental Research** (August 2025) [17]:
- 132 eligible studies reviewed out of 1,556 records
- Target population: older adults aged 60 and above
- Key usability barriers: **cognitive overload**, digital literacy gaps, and accessibility challenges
- Essential design features: simplified navigation, enlarged text and touch targets, voice interaction, and error-tolerant interfaces
- Participatory, user-centered design involving co-creation with older users "significantly enhances satisfaction and adoption"

### 1.7 Older Adults and Menu Depth: Navigation Regressions and Working Memory

**Accessible Web Design for Older Adults (TACCESS-2025 preprint)** [18]:
- "Older adults exhibit **significantly more navigation regressions when menu depth exceeds their working memory span**, rather than due to issues with the ambiguity of link labels"
- This directly supports the need for **shallow navigation hierarchies** (maximum 2-3 levels) for older adult users

---

## 2. Platform-Specific UI Features and Navigation Structures

### 2.1 SAP S/4HANA Fiori: Progressive Disclosure Architecture

#### Fiori Elements (Metadata-Driven Floorplans)

SAP Fiori elements is a UI development framework that uses **predefined floorplans** and metadata annotations to streamline application development [3, 6]. The key page types are:

- **List Report Page** – filter, view, and work with items in list/table format
- **Object Page** – work with objects in detail with collapsible sections
- **Analytical List Page** – combined charts, drill-down, and Excel export
- **Overview Page** – analytical dashboard with chart drill-down
- **Worklist Page** – task-focused list of items requiring action

Fiori elements are metadata-driven: the UI is generated dynamically based on service annotations, minimizing manual UI coding. This ensures design consistency while enabling rapid development. Freestyle SAPUI5 development is reserved for requirements beyond standard templates.

#### Spaces and Pages

The **Spaces and Pages** paradigm [1] replaced the older Groups/Catalogs paradigm:

- A **space** is a logical container for apps and content related to a specific topic or work area (e.g., "Manufacturing")
- **Pages** are like tabs within a space, grouping apps by specific tasks
- Spaces are assigned to users based on their work profile (user role)
- Users can customize spaces and pages based on preferences and assigned job roles
- The user experience features smoothly animated lateral movements mimicking a panoramic view (SAP Fiori 2.0 "View Point Metaphor")

**Progressive Disclosure Mechanism:** Role-based space assignment ensures users see only apps relevant to their job function, with progressive drill-down through pages and then individual apps.

#### My Home (Product Home Page)

The **My Home** page serves as a personalized entry point in SAP S/4HANA Cloud [4]. Key sections:

- **To Dos** – workflow tasks from My Inbox, ordered by priority with visual cards
- **News** – business line-specific feeds
- **Pages** – up to 8 favorite pages displayed as tiles
- **Apps** – favorite, most used, recently used, and SAP Business AI-recommended apps
- **Insights Tiles** – analytical charts and KPIs
- **Insights Cards** – overview and list-report-based application cards

Users can personalize via **My Home Settings**: adding apps to favorites, creating app groups, modifying icons and colors, enabling/disabling AI recommendations, exporting/importing settings, and resetting to defaults.

#### My Inbox

The **My Inbox** Fiori app is a mobile-friendly SAP Portal application for all SAP workflow approvals [1, 4]. Key features:

- Filter approval tasks by type, priority, date
- View order details
- Approve or reject with one click
- Handle related attachments and comments
- Universal task consolidation point for workflow items across modules

**Supervisor navigation flow:** SAP Fiori Launchpad → My Inbox app → Filter by task type → Select approval task → View details → Click Approve/Reject (5 clicks)

#### Inventory Management Apps

From the SAP Fiori Apps Reference Library [2, 4]:

| App | App ID | Function |
|-----|--------|----------|
| Stock - Single Material | F1076/MMBE | Overview of material stock by plant/storage location |
| Stock - Multiple Materials | F1075 | Stock levels and values across plants with drill-down |
| Post Goods Receipt for Purchase Order | F0843A/F0843 | Post goods receipts against purchase orders |
| Post Goods Receipt for Inbound Delivery | F2502 | Post goods receipts for inbound deliveries |
| Goods Movement Analysis | F1872 | Complex interpretation of goods movements |
| Inventory Turnover Analysis | F2381 | Identify materials with turnover issues |
| Dead Stock Analysis | F2382 | Monitor unconsumed stock |
| Slow or Non-Moving Materials | F2383 | Identify materials with little movement |
| Material Documents Overview | F0701A | Detailed list of material movements (like MB51) |
| Overdue Materials | F2384 | Focused on blocked stock and stock in transit |

**Navigation Paths:**

- **Check stock levels:** SAP Fiori Launchpad → Stock - Single Material app → Enter material number → View stock by plant/storage location (3 clicks)
- **Post goods receipt for PO:** SAP Fiori Launchpad → Post Goods Receipt for Purchase Order app → Enter PO number → Select items → Click "Post" (4 clicks)

#### Work Order (Maintenance) Apps

Key SAP Fiori applications for maintenance/work orders [11, 13]:

| App | App ID | Function |
|-----|--------|----------|
| Perform Maintenance Jobs | F5104A | Receive, start, pause, resume, complete jobs; records time and materials |
| Find Maintenance Order | F2175 | Filter by phase, mass status changes, dispatch operations |
| Find Maintenance Order Confirmation | F2174 | List maintenance order confirmations |
| Find Maintenance Order and Operation | F2173 | Mass changes and time confirmations |
| Process Maintenance Orders | W0017 | Creation, change, display of orders |
| Resource Scheduling for Planners | – | Real-time insights into work center utilization |
| Maintenance Scheduling Board | – | Graphical Gantt chart-based scheduling |
| Assign Maintenance Order Operations | – | Assignment of maintenance tasks to technicians |

**Supervisor navigation flow:** Find Maintenance Order (F2175) → View order list with filters → Select order → View details → Find Maintenance Order Confirmation (F2174) → View confirmations (Note: Navigation between F2175 and F2174 may require additional configuration)

#### Fiori Shell Bar and Navigation

The **Shell Bar** is a persistent and responsive UI element at the top of the launchpad [13, 14, 15]:

- **Back button** – navigation
- **Branding area** – company logo/name
- **Page title**
- **Navigation menu** – access to Spaces/Pages
- **Search** – global search functionality
- **Notifications**
- **User menu (Me Area)** – user profile, settings, sign out

The **Me Area** contains user-specific actions and app-specific settings. The **App Finder** can be enabled via transaction `/n/ui2/flp_cus_conf` (parameter APPFINDER_ENABLED set to true).

### 2.2 Oracle NetSuite Redwood UI: Modern Progressive Disclosure

#### Redwood UI Components

Per the Oracle NetSuite Redwood UI Migration Guide [16]:

**Sticky Header:**
- Remains fixed when scrolling
- Centered AI-powered "Ask Oracle" search box
- Customizable "Create New" (+) menu for quick record creation
- Home icon with preferences

**Collapsible Field Groups:**
- On forms, field groups are collapsible
- Greatly enhances usability for data-rich forms
- Personalize panel enables drag-and-drop customization

**Card Layout and Inline Editing:**
- Card-based layouts for visual information presentation
- Items can be displayed in card, list, or table layouts
- Inline editing capabilities within list views

**Navigation Menus:**
- Enlarged, responsive hover menus with multi-level expansion
- Hover-activated flyout menus

**Progressive Disclosure:**
- "The UI uses progressive disclosure; pages focus on your content, and relevant links, menus, and icons appear when moving your pointer over areas" [1]
- Content and contextually revealed links, menus, and icons

The Redwood Experience theme provides different colors, icons, and fonts. Users can enable it via Home > Set Preferences > Appearance. Open Sans is the default font.

#### "Ask Oracle" AI Feature

**Ask Oracle** is a conversational AI assistant embedded throughout NetSuite [19, 20]:

- Appears prominently centered in the global header search box
- Natural language query capabilities: users can type conversational English questions
- Generates real-time insights, summaries, and actionable workflows
- Uses retrieval-augmented generation (RAG) for accuracy
- Every response includes citations to data sources, reasoning steps, and options to refine or reject suggestions
- Contextualized to each user role, providing personalized, proactive responses
- Supports both text and voice interactions
- Can generate visualizations (charts, KPIs)

**NetSuite Next** (announced at SuiteWorld 2025) places Ask Oracle at the center, giving users "a single, conversational interface to search, analyze, and act across your entire system" [21].

#### Inventory Management in Redwood

**Item Quantities Page** [22, 23]:

Navigation: Inventory Management landing page → Item Quantities task

Functionality:
- View on-hand item quantities across multiple organizations
- Search by item name, description, manufacturer part number (MPN), lot, or serial number
- Filter results by organization using filter chips
- Toggle buttons: available to reserve, available to transact
- View inbound quantities (in-transit and unshipped) with drill-down by document types
- Select between primary or stocking units of measure
- Download information onto a spreadsheet
- Tabs for Item, Lot, and Serial navigation
- For lot-controlled items: expiration date, expiration action date, origination type
- For serial-controlled items: search by serial number to view location and status

**NetSuite Inventory Page Portlets** [31]:
- Inventory by Location – real-time on-hand quantities with Available to Allocate, Unallocated Demand, Overdue Demand, Overdue on Order
- Projected On Hand – forecasts future availability
- Inbound/Outbound Transactions – pending receipt/fulfillment transactions
- Inventory by Status, Lot, and Bin
- Open Transfer Orders, Open Inter-Co Transfer Orders, Planned Transfer Orders

**Receipts Work Area** [26, 27, 28]:

Item Receipts in NetSuite WMS: Records receipt of items into warehouse, links to transfer order, purchase order, or return authorization. When posted, system marks the item as received and updates stock level.

Step-by-step workflow:
1. **Enable Deferred Item Receipt Posting** (optional)
2. **Mobile app:** Menu → Post Item Receipt → Select receipt type (Purchase Order, Transfer Order, Inbound Shipment, or RMA Receipt) → Choose order → Review items → Post
3. **UI (WMS Warehouse Manager role):** WMS Inbound > Post Item Receipt → Filter by location/transaction/number → Review putaway items → Submit

#### Navigation Structure

**Navigation Menu Bar** [23, 24]:
- Located at bottom of header
- Tabs depend on user role
- Three tabs use icons: Recent Records, Shortcuts, and Home
- Ellipsis indicates more menus available
- Tabs automatically expand to multi-level navigation menus on hover

**SuiteBar (Header)** [16]:
- Sticky global header
- Centralized "Ask Oracle" AI search
- Create New (+) menu
- Larger hover menus
- Home icon

**Center Tabs** [24]:
- Different set of tabbed pages based on user role
- Each center displays tabbed pages with dashboards composed of portlets
- Administrators can create custom center tabs

**Quick Links / Shortcuts** [23]:
- Added via star icon
- Shortcuts portlet and menu offer access to frequently used items
- Options to add, reorder, rename, or remove

**Global Search** [32, 33, 34]:
- Accessed via search field or **Alt+G** keyboard shortcut
- Supports prefixes: "cus:" (customers), "invo:" (invoices), "empl:" (employees)
- Accepts wildcards (% for partial matches, _ for specific patterns)
- Capitalized prefixes open records in edit mode

**Navigation Paths:**

- **Check item quantities:** Inventory Management work area → Item Quantities task → Search by item/MPN → View on-hand quantities (3 clicks)
- **Post item receipt (UI):** WMS Inbound > Post Item Receipt → Filter by location/transaction → Review items → Submit (4 clicks)
- **Post item receipt (mobile):** WMS app → Menu → Post Item Receipt → Select type → Select order → Review → Post (5 taps)
- **Global search:** Press Alt+G → Type search term (with optional prefix) → Select result (2 keystrokes)

#### Role-Based Dashboards [35, 36]

Dashboards provide a visual workspace with instant access to accurate information. Content appears in **portlets** (dynamic data display windows: raw data, KPIs, scorecards, graphs, RSS feeds). NetSuite shows different centers based on user role. Available portlet types:

- KPIs
- Analytics charts
- Saved searches
- Reminders
- Project tasks
- **Analytics Portlet** (chart-based workbook data)
- **Trend Graph** (up to 3 KPIs)
- **Condensed KPI portlet** format

For a supervisor role: Dashboards can be tailored with KPIs for inventory accuracy, order fulfillment, and operations monitoring. The Redwood UI introduces collapsible portlets and a slide-out **Personalize panel** for drag-and-drop customization.

### 2.3 Microsoft Dynamics 365: Hybrid Business Process-Driven Model

#### Unified Interface Sitemap

The **Unified Interface** provides a consistent and responsive user experience across devices [39, 40]. Navigation changed from top of screen (classic web client) to **left-hand side** in Unified Interface. Key elements:

- **App-selector menu** – switch between apps
- **Work-area menu** – navigate within an app
- **Side navigator (site map)** – displays entities and pages on left side
- **Recent and favorite records** – available from navigation
- **Reference panel** – related item lookup without leaving current screen
- **App message bar** – three notification types: informational, warning, error

For Supply Chain Management, navigation sidebar provides access to modules: Warehouse Management, Inventory Management, Production Control, Master Planning, Procurement and Sourcing, Sales and Marketing, Asset Management, and Cost Management [46].

On mobile devices, navigation adapts to collapsed/hamburger menu pattern while maintaining same structure.

#### Business Process Flows (BPFs)

A Dynamics 365 **Business Process Flow** is a step-by-step visual guide displayed in the screen header that leads users through an established business process [48, 49, 50]:

- Displayed as multi-stage progress bar at top of screen
- Created via **no-code, drag-and-drop interface** in Power Automate or PowerApps
- Can be simple sequential steps or complex, conditional multi-stage processes
- Enforce required fields and optional conditional steps requiring approvals
- Assignable to specific security roles
- Support triggering automations at various stages

**Progressive Disclosure Mechanism:** The BPF bar reveals the next step only when the current step is complete, providing guided, staged workflow progression.

For production supervisors, BPFs guide standardized processes: production order creation → scheduling → release → execution → completion. BPFs ensure data uniformity, reduce errors, and improve user adoption without requiring extensive retraining.

#### Production Floor Execution Interface (PFE)

The **Production Floor Execution Interface** enables shop floor workers to register their daily work [40, 41, 42]:

- Workers can: start jobs, report feedback about jobs, register indirect activities, report absence
- Registrations track progress and cost on production orders and calculate worker pay
- Interface is automatically configured per browser and device
- Optional features enabled via **Feature management** page
- Supports custom color themes

**Job card registration:** Workers start jobs, report progress (time and quantity), and complete jobs. Time and goods quantity are registered in two separate journals automatically.

**Clock in/out:** Configurable via Login tab settings in Configure production floor execution. The interface supports worker clock-in and session management.

**Setup:** User with "Maintain time supervision" duty (usually "Shop floor supervisor" role) signs in and selects device-specific configurations and job filters. Interface can run in full-screen mode hiding navigation.

**Progressive Disclosure in PFE:**
- Automatic configuration loading per browser and device
- Badge ID sign-in loads configurations specific to device
- Configurable features: material consumption registration, scrap reporting, serial number tracking
- Creates a **"progressive enabling"** system where complexity is added as needed

**Optional Features:**
- Clock in and out only (simplified interface)
- Material consumption registration (WMS and non-WMS items)
- Batch/serial number tracking
- Multi-job progress reporting
- Streamlined indirect activity registration (removes confirmation dialogs)
- Test results recording
- Automatic completion of secondary operations with primary operations

#### Warehouse Management Mobile App (ProcessGuide Framework)

The **ProcessGuide** framework guides users through warehouse business processes step by step [42, 43]:

**Framework Components:**
- **ProcessGuideController** – orchestrates execution
- **ProcessGuideStep** – represents a single step
- **ProcessGuidePageBuilder** – constructs the UI
- **ProcessGuideAction** – represents user actions (e.g., ProcessGuideOKAction executes the step's `executeOKAction` method)
- **ProcessGuideDataProcessor** – processes user input data
- **ProcessGuideNavigationAgent** – manages step transitions

Each step has:
- **Title** – short description
- **Instruction** – longer description shown when step opens (with "Don't show again" option)

The process starts when XML requests are deserialized into `ProcessGuideRequest` objects, then the controller creates responses by invoking steps and page builders sequentially.

The **Warehouse Management mobile app** (versions 4.0+) supports Windows, Android, and iOS. Features include GS1 barcode support, hardware keyboard enhancements, Entra brokered authentication, and battery optimization.

#### Role Centers

**Role Centers** are personalized start pages tailored to specific business roles [41]. The **Production Supervisor Role Center** displays:

- **Headline area** – aggregated business data KPIs (production order status, work center utilization)
- **Activity Queues** – divided into Data Queues (aggregated business data) and Action Queues (links to operations)
- **KPIs and alerts** – real-time production monitoring, quality issues, overdue orders
- **Actionable data** – links to key pages: Production Orders, Resource Scheduling, Quality Management

Role Centers follow consistent structure: application bar, navigation menu, headline area, actions on right, and activity queues. Users enter **Design mode** via gear icon to reposition elements using drag and drop, add new fields, and switch between monitor, tablet, and phone views.

#### Action Search (Keyboard Shortcuts)

For Dynamics 365 Finance and Operations apps (Supply Chain Management) [45]:

- Press **Ctrl+'** to find actions and pages via search
- Press **Alt+Shift+K** to view available shortcuts for current context
- Shortcuts grouped into: action shortcuts, date picker shortcuts, FactBox shortcuts, filtering shortcuts, page shortcuts, page navigation, grid operations, input controls, messaging, navigation, personalization, segmented entry, and task recorder
- Some shortcuts are key chords (require two consecutive key combinations)
- Version-specific shortcuts marked with asterisk available from version 10.0.32+

**Natural language search** is available through **Microsoft Copilot** integration, enabling users to ask questions in natural language for tasks like demand analysis and procurement impact analysis (new features arriving in 2026 Wave 1, with generative demand analysis available June 2026).

**Navigation Paths:**
- **View production orders:** Navigation sidebar → Production Control → Production Orders → View list (3 clicks)
- **Report progress (PFE):** Production Floor Execution interface → Select job → Report Progress → Enter quantity → Confirm (4 clicks)
- **Clock in (PFE):** Production Floor Execution interface → Clock In → Enter worker ID/scan badge → Confirm (3 clicks)
- **Find action using keyboard:** Press Alt+Shift+K to view shortcuts, or use search functionality (2 keystrokes)

### 2.4 Cross-Platform Navigation Philosophy Comparison

| Platform | Primary Philosophy | Progressive Disclosure Strength | Persistent Navigation Strength | How Tension is Managed |
|----------|-------------------|--------------------------------|-------------------------------|------------------------|
| **SAP S/4HANA Fiori** | **Strong Progressive Disclosure** | Role-based tiles hide non-relevant apps; Spaces/Pages hierarchy; collapsible object page sections; smart filters | My Inbox consolidates pending tasks; shell header provides global actions | Role-based filtering + hybrid GUI coexistence; power users can access all GUI transactions via launchpad |
| **Oracle NetSuite Redwood** | **Hybrid – Leans Progressive Disclosure** | Sticky header with "Ask Oracle" AI search; collapsible field groups; inline editing; card-based layouts; hover-activated menus | Global search provides persistent navigation entry point; Create New (+) always visible | Role-based Centers + progressive disclosure within forms + AI-powered search |
| **Microsoft Dynamics 365** | **Hybrid – Business Process-Driven** | Business Process Flow bar is strongest mechanism; Production Floor Execution interface with progressive enabling; Action Search (Ctrl+') | Sitemap provides persistent left-side navigation; module categories always visible | Role-based sitemap filtering + BPF-guided workflows + configurable PFE interface + AI Copilot |

---

## 3. Change Management and Training Frameworks with Concrete Metrics

### 3.1 Kirkpatrick's Four Levels of Training Evaluation

The Kirkpatrick Model was developed by Dr. Donald Kirkpatrick in 1959 and updated over the years. It comprises four levels: Reaction, Learning, Behavior, and Results [1, 2, 3, 4, 5, 14].

#### Level 1: Reaction – User Satisfaction and Engagement

**Definition:** Measures whether learners find training engaging, favorable, and relevant to their jobs [1, 2, 3, 4, 5].

**Validated Instruments:**
- Traditional "smile sheet" questionnaire [11]
- New World Learner-Centered Reaction Sheets emphasize: understanding of objectives, clarity of expectations, ability to apply on the job, anticipated barriers [11]
- A study using Kirkpatrick for evaluating an exercise rehabilitation program (2025) used 6 items for Reaction with Cronbach alpha of 0.958 for overall questionnaire [11]

**Specific Numeric Targets (for supervisors aged 45-60):**

| Metric | Target | Timeline | Measurement Method |
|--------|--------|----------|-------------------|
| Relevance-to-job rating | ≥4.2/5.0 | End of each training module | "The training content was relevant to my job on the production floor" |
| Confidence rating | ≥4.0/5.0 | End of each training module | "I am confident I can perform this task in the new ERP system" |
| Clarity of expectations | ≥4.0/5.0 | End of each training module | "I am clear about what is expected of me after this training" |
| Overall favorable response rate | ≥85% (responses of 4 or 5) | End of each training module | Aggregated satisfaction items |

**Timeline:** Pre-go-live weeks 1-8, measured immediately after each training module (post-session survey).

#### Level 2: Learning – Knowledge and Skill Acquisition

**Definition:** Measures increase in knowledge, skills, or attitudes through pre- and post-training assessments [1, 2, 3, 4, 5].

**Specific Assessment Methods (for production floor supervisors):**
- Knowledge tests (multiple-choice/true-false on navigation paths, transaction codes, data entry rules)
- Simulated goods receipt transaction in sandbox ERP environment
- Hands-on demonstrations of inventory counts, material transfers, production order confirmations
- Pre- and post-training assessments aligned with learning objectives

**Specific Numeric Targets:**

| Metric | Target | Timeline | Measurement Method |
|--------|--------|----------|-------------------|
| Simulated goods receipt pass rate | ≥85% first attempt | End of week 4 training | Sandbox simulation |
| Post-training knowledge assessment | ≥80% | End of each training module | Written knowledge test |
| Retention score at go-live | ≥75% | Week 8 (go-live readiness) | Same test re-administered |
| Pre-training baseline improvement | ≥30% knowledge gain | Pre-test week 1, post-test week 4 | Comparative assessment |

**Anchor Tasks for Assessments:**
1. **Goods Receipt:** Locate "Post Goods Receipts" app/transaction, enter PO number, confirm quantities, assign storage location, post. Target time: <3 minutes.
2. **Stock Transfer:** Locate stock transfer function, select material, specify source/destination locations, enter quantity, post. Target time: <2 minutes.
3. **Work Order Creation:** Create production order with material, quantity, production version, start date; release order. Target time: <5 minutes.
4. **Work Order Confirmation:** Report yield, confirm operations, complete order. Target time: <3 minutes.

**Timeline:** Pre-go-live weeks 1-8, measured immediately after each training session (post-training test). Re-measured at go-live (Week 8) for retention.

#### Level 3: Behavior – On-the-Job Application

**Definition:** Evaluates whether participants apply learned skills in the workplace [1, 2, 3, 4, 5].

**Measurement Methods:**
- System audit logs tracking transaction accuracy and navigation path compliance
- Observation checklists completed by super-users or floor trainers
- Supervisor ratings (from plant managers or shift managers)
- Kirkpatrick's four conditions for behavior change: desire to change, knowledge of what to do/how to do it, right climate, reward for changing [15]

**Specific Behavioral Metrics:**

| Metric | Target | Timeline | Measurement Method |
|--------|--------|----------|-------------------|
| Correct navigation path usage | ≥90% of inventory transactions | Days 1-30 post go-live | System audit logs |
| Core transaction error rate | ≤5% | Days 1-30 post go-live | Transaction error log |
| Independent transaction completion | ≥80% of supervisors without job aids | Day 30 post go-live | Super-user observation |
| System login compliance | ≥95% of supervisors logged in at least once per shift | Days 1-30 post go-live | Login audit logs |
| Peer coaching adoption | ≥50% of supervisors can demonstrate to peers | Months 2-3 post go-live | Self-report + peer verification |

**Timeline:** Go-live Days 1-30 and Months 2-3.

#### Level 4: Results – Organizational Impact

**Definition:** Measures how training contributes to organizational success through reduced cost, improved quality, increased productivity, and employee retention [1, 2, 3, 4, 5].

**Specific Business Impact Metrics (for manufacturing ERP context) [3]:**

| Metric | Target | Timeline | Measurement Method |
|--------|--------|----------|-------------------|
| Order accuracy | ≥99.8% picking accuracy; ≥98% on-time delivery | Quarter 2 post go-live | Order fulfillment metrics |
| Inventory accuracy | ≥95% (cycle count reconciliation) | Quarter 2 post go-live | Physical inventory comparison |
| Error rate on ERP transactions | ≤3% | Month 6 post go-live | Transaction audit logs |
| Order processing cycle time | 10-20% reduction (e.g., 48h → 12h) | Quarter 2 post go-live | Process time tracking |
| Inventory carrying cost reduction | 5-15% reduction | Months 6-12 post go-live | Finance system analysis |
| Training ROI | 15-25% within 3-5 years | Annual measurement | Phillips ROI Methodology |
| User adoption balance | >80% departmental usage balance | Month 6 post go-live | System usage analytics |
| User satisfaction score | >7.5/10 | Month 6 post go-live | Satisfaction survey |
| Manual data entry error reduction | Up to 90% reduction | Quarter 3 post go-live | Before/after comparison |

**Timeline:** Months 2-12 (quarterly measurement).

### 3.2 UTAUT (Unified Theory of Acceptance and Use of Technology)

The UTAUT model was developed by Venkatesh et al. (2003). It synthesizes eight prior models and explains ≈70% of variance in behavioral intention and ≈50% in actual use [6, 7, 8, 9, 10]. UTAUT2 (Venkatesh et al., 2012) extends to consumer contexts, explaining ≈74% of behavioral intention [8, 9].

The four core determinants are moderated by age, gender, experience, and voluntariness of use [6, 7, 8, 9, 10].

#### Performance Expectancy

**Definition:** "The degree to which an individual believes that using the system will help him or her to attain gains in job performance" [8, 9].

**Survey Items (adapted from Venkatesh et al., 2003):**
1. "Using the new ERP will help me complete inventory checks more quickly"
2. "Using the new ERP will improve my job performance as a production floor supervisor"
3. "Using the new ERP will increase my productivity in managing shift operations"
4. "Using the new ERP will enhance my effectiveness on the job"

**Likert Scale:** 1 (Strongly Disagree) to 7 (Strongly Agree). **Target: >5.5** for supervisors aged 45-60.

**Measurement Timeline:**
- T0 (Baseline, Week 1): Pre-training survey
- T1 (Post-training, Week 8): After all training modules
- T2 (Go-live, Day 30): During stabilization
- T3 (Month 3), T4 (Month 6), T5 (Month 12): Quarterly follow-ups

**Age-Specific Findings:**
- Performance expectancy had the greatest influence on Behavioral Intention in an EMR adoption study, increasing intention by 50.7% [3]
- A meta-analysis of health care technology acceptance in older adults (PMC, 2025): Perceived usefulness had significant positive correlation with behavioral intention (r=0.607) among older adults (mean age 67.58) [10]
- However, age and gender showed **no significant moderating effects** in this meta-analysis [10]
- For the oldest generation (born 1900-1946), effort expectancy and facilitating conditions were the strongest predictors, not performance expectancy [10]

#### Effort Expectancy

**Definition:** "The degree of ease associated with the use of the system" [8, 9].

**Survey Items (adapted from Venkatesh et al., 2003):**
1. "Learning to use the new ERP system will be easy for me"
2. "My interaction with the new ERP system will be clear and understandable"
3. "I will find the new ERP system easy to use"
4. "It will be easy for me to become skillful at using the new ERP system"

**Likert Scale:** 1 (Strongly Disagree) to 7 (Strongly Agree). **Target: >5.5** for supervisors aged 45-60.

**Age-Specific Findings:**
- Experience with technology decreases the effect of effort expectancy [8]
- For older adults with limited software experience (AS/400 users), effort expectancy is a **stronger predictor** of behavioral intention compared to younger or more experienced users [8]
- The multigenerational tablet adoption study found: "Effort expectancy and facilitating conditions were the only determinants that positively predicted tablet use intentions after controlling for age, gender, and tablet use" [10]
- A meta-analysis (PMC, 2025): Perceived ease of use had significant positive correlation with behavioral intention among older adults (r=0.525) [10]
- Geographic region significantly moderated this relationship [10]

#### Social Influence

**Definition:** "The degree to which an individual perceives that important others believe he or she should use the new system" [8, 9].

**Survey Items (adapted from Venkatesh et al., 2003):**
1. "People who influence my behavior (plant manager, shift supervisor) think that I should use the new ERP system"
2. "People who are important to me (colleagues, team leads) think that I should use the new ERP system"
3. "The senior management of this plant has been helpful in getting me to use the new ERP system"
4. "In general, the organization has supported the use of the new ERP system"

**Likert Scale:** 1 (Strongly Disagree) to 7 (Strongly Agree). **Target: >5.0** for supervisors aged 45-60.

**Age-Specific Findings:**
- Social influence was **not significant** in the final model predicting tablet use intention after controlling for moderators [10]
- "Consistent generational differences in UTAUT determinants, most frequently between the oldest and youngest generations" [10]
- A meta-analysis (PMC, 2025): Social influence had significant positive correlation with behavioral intention among older adults (r=0.551) [10]
- **Visual demonstrations significantly enhanced** the effect of social influence [10]
- In mandatory settings (ERP adoption at work), social influence factors were more significant [10]
- Social influence has a stronger effect on behavioral intention for **older workers** in mandatory adoption settings [10]

#### Facilitating Conditions

**Definition:** "The degree to which an individual believes that an organisation's and technical infrastructure exists to support the use of the system" [7, 8, 9].

**Survey Items (adapted from Venkatesh et al., 2003):**
1. "I have the resources necessary to use the new ERP system (e.g., computer, internet access, headset)"
2. "I have the knowledge necessary to use the new ERP system"
3. "The new ERP system is compatible with other systems I use"
4. "A specific person (or help desk) is available for assistance with system difficulties"

**Likert Scale:** 1 (Strongly Disagree) to 7 (Strongly Agree). **Target: >5.5** for supervisors aged 45-60 (this construct is **critical** for this demographic).

**Age-Specific Findings:**
- For older workers (age 45-60+), **facilitating conditions is the strongest or among the strongest predictors** of technology adoption [10]
- The multigenerational tablet study found: "Effort expectancy and facilitating conditions were the only determinants that positively predicted tablet use intentions" [10]
- UTAUT2 research: Older women in early stages of experience rely more on facilitating conditions [9]
- A study of e-NAPSA adoption: Correlation coefficient of 0.312 (p=0.002) for facilitating conditions [7]

**Organizational Support Factors That Matter for This Demographic:**
- Training programs tailored to their experience level
- Responsive help desk with <30 minute response time
- Peer support (buddy system or floor champions)
- Access to job aids and quick reference cards
- Patient, non-judgmental support environment

#### Summary of UTAUT Targets by Timeline

| Construct | T0 (Week 1) | T1 (Week 8) | T2 (Day 30) | T3 (Month 3) | T4 (Month 6) | T5 (Month 12) |
|-----------|-------------|-------------|-------------|--------------|--------------|---------------|
| Performance Expectancy | Baseline | 4.5 | 5.0 | 5.5 | 5.8 | 5.8 |
| Effort Expectancy | Baseline | 4.0 | 4.5 | 5.0 | 5.5 | 5.8 |
| Social Influence | Baseline | 4.5 | 5.0 | 5.0 | 5.0 | 5.0 |
| Facilitating Conditions | Baseline | 5.0 | 5.5 | 5.8 | 5.8 | 5.8 |

*All targets on 7-point Likert scale.*
*Note: T0 establishes baseline only; targets represent minimum acceptable means at each measurement point.*

### 3.3 Cognitive Load Theory (CLT) Applied to ERP Training for Ages 45-60

Cognitive Load Theory (CLT), developed by John Sweller in the late 1980s, explains how limited working memory capacity affects learning and instructional effectiveness [11, 12, 14, 16, 18, 19].

#### Core Working Memory Limits

**Traditional findings:**
- Miller (1956): 7±2 items [16, 18]
- Cowan (2001): Approximately 4 elements at a time [16, 18]

**Age-Specific Findings:**

| Age Group | Working Memory Capacity (Items) | Source |
|-----------|--------------------------------|--------|
| Younger adults | 5.08 items | Brown University/Stanford CCN 2024 [20] |
| Older adults | Approximately 3.8 items (SD=1.0) | Brown University/Stanford CCN 2024 [20] |
| Age-related decline cause | Reduced storage capacity (not chunking), linked to parietal glutamate decline | Brown University study [16] |

**The CRUNCH Hypothesis:** The Compensation-Related Utilization of Neural Circuits Hypothesis proposes that brain activation and memory load have an inverted-U relationship shifted left in older adults, who reach capacity limits sooner. Throughput plateaus at approximately 3.8 items compared to 5.08 items in younger adults [20].

**Key Quote from Brown University Study (2024):** "Our results show that age-related memory deficits are driven primarily by limitations in storage capacity linked to glutamate declines in parietal regions" [16].

#### Three Types of Cognitive Load

1. **Intrinsic load:** Inherent complexity of the task (e.g., creating a work order involves multiple fields and steps)
2. **Extraneous load:** Unnecessary cognitive demands caused by poor interface/instructional design (e.g., cluttered dashboard, confusing menu structure)
3. **Germane load:** Cognitive effort directed toward learning and building new mental schemas

**Goal:** Minimize extraneous load and optimize intrinsic load to free capacity for germane load.

#### Intrinsic Load Reduction Recommendations

| Recommendation | Rationale | Metric/Target |
|----------------|-----------|---------------|
| Maximum 3-5 elements per screen | Older adults plateau at ~3.8 items WM capacity [16, 20] | Limit screens to 3-5 chunks |
| Session duration: 20-30 minutes hands-on practice | Cognitive fatigue sets in; clinical trial used two 20-min sessions/day [1] | Timed training blocks; mandatory 10-min breaks |
| Maximum 45 minutes total classroom time | Chandler & Sweller (1991), Mayer (2001) support shorter, focused sessions [16] | Trainer-enforced session caps |
| Chunk complex workflows into ≤5 steps per sub-task | Working memory limited to ~4 elements [16, 18] | Maximum 5 steps per sub-task |
| Pre-training on interface vocabulary (15 minutes) | Advance organizers improve learning outcomes for older adults | Fiori terminology session before first transaction |
| Task decomposition: max 3-5 steps per sub-task | "Chunking information helps optimize intrinsic load" [14] | Structured micro-flows |

#### Extraneous Load Reduction Recommendations

| Recommendation | Rationale | Metric/Target |
|----------------|-----------|---------------|
| Maximum 5 navigation options per screen | Age-specific: reduces from 7±2 for younger to ~3-5 for older [16, 19] | Navigation limit per screen |
| Information reachable within 3 clicks | Minimize click-distance to reduce navigation regressions [18] | Max 3 clicks from home screen |
| Integrate text with visuals | Avoid split-attention effects [14] | Labelled diagrams, not separate legends |
| Use labeled diagrams | "A labelled diagram places a lower demand on your working memory" [19] | All visuals integrated with labels |
| Use non-expandable menus for initial views | Older adults prefer all options visible rather than hidden [5, 6] | Default: flat hierarchy with progressive drill-down |
| Consistent interface layout across all simulations | Simple, consistent interfaces benefit older workers [4] | Same button positions in training and production |
| Remove redundant information | Keep explanations simple [14] | ≤20 words per instruction step |

**Critical Finding from Zaphiris (2003):** Older adults preferred **non-expandable hierarchies** (all options visible) over expandable ones. This suggests that while progressive disclosure is beneficial, the initial default view should show all primary options rather than hiding them behind expandable controls [5, 6].

#### Germane Load Enhancement Recommendations

| Recommendation | Rationale | Metric/Target |
|----------------|-----------|---------------|
| Provide 3 worked examples per transaction type first | Worked-example effect: studying solved problems enhances learning more than problem-solving for novices [14, 17] | Minimum 3 examples before independent practice |
| Scaffolded training with phased fade-out | Full scaffolding (Weeks 1-4) → Intermediate (Weeks 5-8) → Minimal (Go-live Days 1-30) → Independence (Months 2-3) | Phased tooltip/guide removal |
| Scenario-based training: 80% of time in realistic scenarios | "Manufacturing users experience ERP through work, not modules" | Realistic scenarios reflecting production constraints |
| Dual-channel presentation: audio narration + visual demonstration | Visual + auditory channels do not compete, expanding cognitive capacity [CLT] | All training materials with narration |
| Self-paced with mastery gates: ≥80% score to advance | Adaptive difficulty based on performance | Module-level mastery assessments |
| Use completion problems for intermediate learners | "Completion problems support new learners without removing thought" [12] | Partial scaffolding during Phase 2 |

#### The Expertise Reversal Effect for AS/400 Users

**Critical peer-reviewed finding (Kalyuga, 2007):** "The reversal in the relative effectiveness of instructional methods as levels of learner knowledge in a domain change has been referred to as an expertise reversal effect" [12]. Instructional formats optimal for novices may hinder performance of more experienced learners.

**Implications for AS/400 migrants:**
- Deep domain expertise (inventory management, work orders) – crystallized intelligence is intact
- Interface novices – fluid intelligence for learning new interaction paradigms is declining
- Training must provide **strong guidance for the interface** while **respecting and leveraging domain knowledge**
- "Excessive instructional guidance" for experienced workers in the domain "uses up valuable cognitive resources" [12]
- Use "tailored fading of worked examples to individual students' growing expertise levels" [14]

**Recommendation:** Strong interface guidance initially, then fade as the user becomes proficient in the new paradigm. This directly supports the hybrid progressive model: simplify the interface, not the work.

### 3.4 Detailed Timeline with Phase-Gate Milestones

#### Phase 1: Pre-Go-Live – Weeks 1-8

**Weeks 1-2: Assessment Phase**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Administer baseline UTAUT survey (T0) | >90% completion rate | Standardized UTAUT questionnaire |
| Conduct pre-training knowledge assessment | Establish baseline (<40% expected) | Written knowledge test |
| Identify individual learning needs | 100% of supervisors assessed | Computer literacy self-assessment + interview |
| Map user personas for supervisor cohort | All supervisors categorized | Role analysis |
| Establish baseline KPIs for Level 4 | Document current order accuracy, inventory accuracy, processing times, error rates | Production system data |
| Conduct stakeholder interviews | 100% of shift-level supervisors interviewed | Structured interview protocol |

**Week 3-4: Foundation Training**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Introductory ERP concepts training | ≥85% attendance | Training roster |
| Basic computer skills refresher (if needed) | 100% of identified low-literacy group complete | Skills checklist |
| Navigation basics training (≤5 options per screen) | Level 1 Reaction ≥4.0/5.0 | Post-session survey |
| Training sessions limited to 30-45 min with 10-min breaks | 100% compliance with session limits | Trainer log |
| Worked examples: minimum 3 per transaction type | All trainees complete all examples | Training completion tracking |
| Level 2 knowledge test | ≥80% pass rate | Written test after each module |

**Week 5-6: Role-Specific Simulation**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Hands-on sandbox practice: goods receipt, stock transfer, work order creation | ≥85% simulation pass rate | Sandbox simulation |
| Task decomposition: max 5 steps per sub-task | All modules designed with micro-flows | Training content audit |
| Intermediate scaffolding (cue cards available, tooltips reduced) | Phase 2 scaffolding implemented | Training materials audit |
| Supervisors practice all role-specific transactions | 100% of required transactions practiced | Training completion log |

**Week 7-8: Go-Live Preparation**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Administer T1 UTAUT survey (post-training) | PE >5.5, EE >5.5, SI >5.0, FC >5.5 | Standardized questionnaire |
| Conduct go-live readiness assessment | ≥75% retention score | Level 2 knowledge test re-administered |
| Full simulation of 3 consecutive days' work | ≥85% transaction accuracy rate | Integrated workflow simulation |
| Cutover dry run | 100% of supervisors can log in, see correct data, complete 1 test transaction | System access verification |
| Final knowledge assessment | Mean score ≥85% | Comprehensive assessment |
| Distribute quick reference cards and job aids | 100% of supervisors receive materials | Distribution log |
| Assign floor "buddies" and super-user champions | 100% of shifts have designated support | Staffing roster |

**Phase 1 Go/No-Go Gate:**
- No red-flagged supervisors (any scoring <70% on readiness assessment gets 1:1 coaching)
- All supervisors can log in and complete at least 1 test transaction
- All shifts have designated floor support personnel

#### Phase 2: Go-Live – Days 1-30

**Days 1-7: Hypercare**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| 24/7 on-floor support personnel | <5 minute response time | Support ticket SLA tracking |
| Floor buddies/super-users available every shift | 100% coverage | Staffing roster |
| Direct supervisor observation of first 5 transactions | 100% of supervisors observed | Observation checklist |
| Quick reference cards at each workstation | 100% of workstations | Physical audit |
| Daily 15-minute huddles with plant manager | 100% of days | Meeting minutes |
| System login compliance | >95% of supervisors logged in | Login audit logs |
| Core transaction error rate | ≤5% | Transaction error log |

**Days 8-14: Stabilization**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| On-floor support reduced but available peak hours | 50% reduction in hotline calls from week 1 | Help desk tracking |
| Independent transaction completion begins | 70% of supervisors independent | Observation |
| Continued buddy support and observation | Check-ins continue daily | Supervisor sign-off |
| Additional 1-on-1 coaching for low performers | 100% of below-target supervisors receive coaching | Training records |

**Days 15-21: Transition**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| On-floor support on-call only | >80% of supervisors independent | Observation |
| Supervisors perform all core transactions independently | ≥80% without job aids | Super-user verification |
| Peer-to-peer support encouraged | ≥20% of supervisors demonstrate to peers | Self-report + peer verification |
| Error rate sustained | <5% | Transaction error log |

**Days 22-30: Review**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Administer T2 UTAUT survey (30-day) | PE >5.0, EE >5.0, FC >5.5 | Standardized questionnaire |
| Re-administer Level 1 reaction survey | Satisfaction ≥3.8/5.0 | Post-go-live survey |
| 30-day review meeting with plant manager | 100% of stakeholders attend | Meeting minutes |
| Review system audit data | 90% correct navigation paths | Audit log analysis |
| Plan Month 2-3 interventions for at-risk users | Individual improvement plans created | Training records |

#### Phase 3: Post-Go-Live – Months 2-12

**Months 2-3: Consolidation**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Kirkpatrick Level 3 (Behavior) assessment | ≥90% work orders created in ERP; ≥85% independent goods receipts; <5% error rate | System audit logs + observation |
| UTAUT T3 survey | PE >5.5, EE >5.5, FC >5.8 | Standardized questionnaire |
| Phase 4 independence (no tooltips, exception-based support) | Implemented for all users | Training materials audit |
| Additional training for low performers | Targeted sessions for bottom quintile | Training records |
| Advanced features training | Introduction of exception handling | Training completion log |

**Months 4-6: Progressive Adoption**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Kirkpatrick Level 3 (Behavior) | Error rate ≤3%; all supervisors demonstrate independent proficiency; peer mentors emerge | System audit + observation |
| UTAUT T4 survey (6-month) | PE >5.8, EE >5.8, FC >5.8 | Standardized questionnaire |
| Inventory accuracy reconciliation | ≥95% | Cycle count comparison |
| Order accuracy tracking | ≥99.8% picking accuracy; ≥98% on-time delivery | Order fulfillment metrics |
| User satisfaction score | >7.5/10 | Satisfaction survey |

**Months 7-9: Optimization**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| Kirkpatrick Level 3 (Behavior) | Error rate ≤2%; supervisors serve as peer mentors; independently identify improvement opportunities | System audit + observation |
| Inventory discrepancy reduction | 5% reduction in cycle count adjustments | Physical inventory comparison |
| Month-end close time improvement | Trending toward 3-day target (from 5 days baseline) | Process time tracking |
| Order fulfillment cycle time | 10-20% reduction from baseline | Process time tracking |

**Months 10-12: Self-Sufficiency**

| Activity | Target/KPI | Measurement Method |
|----------|------------|-------------------|
| UTAUT T5 survey (12-month) | All constructs ≥5.8/7.0 | Standardized questionnaire |
| Kirkpatrick Level 4 (Results) | 15-25% ROI emerging; inventory accuracy ≥95% sustained; order accuracy ≥99.8% sustained | Finance + production data |
| Training cost per user declining | Peer-training replacing formal instructor-led training | Training cost tracking |
| Full self-sufficiency | Comfortable to serve as change champions for future updates | Supervisor feedback + observation |

---

## 4. Evidence Quality Differentiation

### 4.1 Source Type Classification Framework

| Source Type | Description | Reliability Level |
|-------------|-------------|-------------------|
| **Peer-Reviewed Research** | Studies published in academic journals with rigorous peer review; includes theses and dissertations from accredited institutions | **Highest** – especially for cognitive load, aging, usability claims |
| **Practitioner Case Study** | Documented implementations from consulting firms, industry blogs, or conference presentations with specific metrics | **Medium-High** – valuable for real-world metrics but may lack methodological rigor |
| **Vendor Documentation** | Official documentation, white papers, and training materials from SAP, Oracle, Microsoft | **Medium** – useful for understanding design intent and features; treat claims with caution |
| **Industry Benchmark Report** | Market analysis, survey data, and industry statistics from research firms and analyst groups | **Medium** – useful for context and trends; treat specific claims with caution |

### 4.2 Peer-Reviewed Sources for Older Adult Cognitive Load (Prioritized)

The following peer-reviewed sources are **prioritized** for claims about cognitive load, interface usability, and performance for older adults:

1. **NN Group (Pernice & Budiu, 2016)** – "Hamburger Menus and Hidden Navigation Hurt UX Metrics" – 179 participants; compared hidden vs. visible navigation; found 39% longer task time on desktop, 21% higher perceived difficulty, >20% lower content discoverability [1, 2, 3]

2. **Journal of Emerging Trends and Novel Research (2025)** – "THE IMPACT OF UI DESIGN ELEMENTS ON COGNITIVE PERFORMANCE AMONG ELDERLY MOBILE APPLICATION USERS" – 127 participants aged 65-85; 37% increase in task completion, 55% error reduction, 42% NASA-TLX reduction [4]

3. **Zaphiris, Kurniawan & Ellis (2003)** – "Age Related Differences and the Depth vs. Breadth Tradeoff" – 24 older adults (aged 57+); found older adults 1.37-1.49x slower, preferred non-expandable hierarchies; error rates increased from 4% to 34% as depth increased [5, 6]

4. **Kurniawan, Zaphiris & Ellis (2002)** – "Expandable and Sequential Hierarchies: Older and Younger Adults' Time Spent and Errors" – 24 older adults (average age 67.5); older adults 1.37-1.49x slower; expandable hierarchy produced fewer errors but no time advantage [7, 8]

5. **Psychometric Properties of NASA-TLX in Older Adults (2020, PMC)** – 38 participants aged 65+; NASA-TLX ICCs 0.71-0.81; valid for measuring cognitive load in older adults [9]

6. **UTAUT Meta-Analysis (PMC, 2025)** – Health care technology acceptance in older adults (mean age 67.58); perceived usefulness r=0.607, ease of use r=0.525, social influence r=0.551 [10]

7. **Multigenerational Tablet Adoption Study (PMC, 2015)** – 899 participants aged 19-99; effort expectancy and facilitating conditions were strongest predictors for oldest generation [10]

8. **Brown University/Stanford CCN (2024)** – Age-related declines in visual short-term memory due to reduced capacity (3.8 vs. 5.08 items), linked to parietal glutamate decline [16, 20]

9. **Cowan (2010)** – "The Magical Mystery Four" – Working memory limited to approximately 4 meaningful items [18]

10. **Gilchrist, Cowan & Naveh-Benjamin (2008)** – Working memory capacity for spoken sentences decreases with adult aging; older adults recall fewer chunks but maintain chunk completion rates [19]

11. **Asif et al. (2022, MDPI)** – "A Comprehensive Approach of Exploring Usability Problems in Enterprise Resource Planning Systems" – 67 usability problems from 11 research papers; recommendation: spend 10% of budget on usability [13]

12. **Purwandari & Dewi (2025)** – "Advancements in UI and UX Design within Enterprise Systems" – 82 empirical articles (2019-2024); UCD resulted in 20-30% better usability scores; adaptive interfaces increased task efficiency by 25-30% [30]

13. **Systematic Review (Aging Clinical and Experimental Research, 2025)** – 132 eligible studies; key barriers: cognitive overload, digital literacy gaps, accessibility challenges [17]

14. **TACCESS-2025 Preprint** – "Accessible Web Design for Older Adults" – Older adults exhibit significantly more navigation regressions when menu depth exceeds working memory span [18]

15. **CRUNCH Hypothesis** – Compensation-Related Utilization of Neural Circuits; older adults reach capacity limits sooner, plateau at ~3.8 items [20]

### 4.3 Vendor Claims: Flagged and Separated

The following claims from vendor sources should be treated with caution as they lack independent peer-reviewed validation:

| Claim | Source | Caution Level |
|-------|--------|---------------|
| SAP Fiori "transforms casual users into SAP experts" | SAP documentation | **High** – no peer-reviewed evidence |
| SAP Fiori "delivers a consumer-grade user experience" | SOAPeople | **High** – marketing claim |
| "Fiori has proved to be a game-changer" | Gemini Consulting | **High** – practitioner opinion |
| RFgen "shortening training by 80% or more" | RFgen | **High** – unverifiable; no methodology |
| "Companies using customized dashboards have seen a 40% improvement in decision-making efficiency" | ZealousWeb (2025) | **Medium** – no peer-reviewed source cited |
| SAP Fiori "minimal training and low click rates" for Situation Handling | SAP Community | **Medium** – no specific quantitative data provided |
| NetSuite Redwood "complete rethinking of how a business system can embrace better workflows" | Houseblend.io | **Medium** – practitioner interpretation |
| Dynamics 365 Business Process Flows "ensure data uniformity, reduce errors, and improve user adoption without requiring extensive retraining" | Pragmatiq | **Medium** – practitioner claim |
| Adaptive UI Boulevard system "reduces visual search time" | IS MUNI thesis | **Low** – very small sample (n=6) |

### 4.4 Source Quality Summary by Claim Type

| Claim Type | Preferred Source Type | Key Sources |
|------------|----------------------|-------------|
| Cognitive load in older adults | **Peer-reviewed only** | Journal of Emerging Trends 2025 [4], Cowan 2010 [18], Gilchrist 2008 [19], Brown/Stanford 2024 [16,20], TACCESS-2025 [18] |
| Age-related working memory capacity | **Peer-reviewed only** | Cowan 2010 [18], Brown/Stanford 2024 [16,20], CRUNCH hypothesis [20] |
| Depth vs. breadth tradeoff (older adults) | **Peer-reviewed only** | Zaphiris et al. 2003 [5,6], Kurniawan et al. 2002 [7,8], NN Group 2016 [1,2,3] |
| Hidden vs. visible navigation comparison | **Peer-reviewed only** | NN Group 2016 [1,2,3] |
| UTAUT age-specific findings | **Peer-reviewed only** | Venkatesh et al. 2003 [8], PMC meta-analysis 2025 [10], multigenerational study 2015 [10] |
| Kirkpatrick model with metrics | Peer-reviewed + practitioner | Kirkpatrick 1959/2006 [1,2,3,4,5,14,15], practitioner implementation guides |
| Platform navigation patterns | Vendor documentation + practitioner | SAP/Oracle/Microsoft official docs, case studies |
| Implementation success metrics | Practitioner + industry benchmark | MDPI ERP study [13], Purwandari & Dewi 2025 [30] |
| Vendor-specific feature claims | Vendor documentation only | Must be flagged as lacking independent validation |

---

## 5. Synthesis: Role-Based Views vs. Comprehensive Dashboards

### 5.1 The Core Trade-off

The central design question is: **Do simplified "role-based views" or comprehensive "dashboards" better support users who need occasional access to advanced functions while maintaining primary task efficiency?**

**Comprehensive Dashboards (Persistent Navigation):**
- Provide full functionality and data visibility at all times
- Excellent for power users who need broad situational awareness
- Suffer from **feature overload**, leading to choice paralysis, higher error rates, and significantly longer training times
- For older adults (45-60), dense displays impose higher extraneous cognitive load due to age-related declines in visual search efficiency and reduced working memory capacity [4, 5, 16, 19]

**Role-Based Views (Progressive Disclosure):**
- Simplify the interface to show only data and actions relevant to a specific job function
- Dramatically reduce extraneous cognitive load
- Speed up primary task completion and shorten training
- Risk of "locking users in" – making it difficult to discover and access advanced or infrequently used features

### 5.2 Evidence for the Target Demographic

For production floor supervisors (aged 45-60, limited software experience beyond legacy AS/400), the evidence **strongly favors role-based views with progressive disclosure**:

**Quantitative Evidence:**

| Metric | Finding | Source |
|--------|---------|--------|
| Task completion rate increase | **37%** (63.2% → 86.7%) with simplified navigation | Journal of Emerging Trends 2025 [4] |
| Error frequency reduction | **55%** with elderly-optimized UI | Journal of Emerging Trends 2025 [4] |
| Cognitive load (NASA-TLX) reduction | **42%** with optimized UI | Journal of Emerging Trends 2025 [4] |
| Navigation efficiency improvement | **50%** with optimized UI | Journal of Emerging Trends 2025 [4] |
| Hidden navigation task time increase | **39%** longer on desktop vs. visible navigation | NN Group 2016 [1] |
| Hidden navigation perceived difficulty increase | **21%** higher vs. visible navigation | NN Group 2016 [1] |
| Hidden navigation content discoverability drop | **>20%** lower vs. visible/combo navigation | NN Group 2016 [1] |
| Error rates as depth increases | From **4% (1 level) to 34% (6 levels)** | Zaphiris 2003 [5] |
| Older adult navigation speed | **1.37-1.49x slower** than younger adults | Kurniawan et al. 2002 [7] |
| UCD usability improvement in enterprise systems | **20-30%** better usability scores | Purwandari & Dewi 2025 [30] |
| Adaptive interface task efficiency increase | **25-30%** in enterprise systems | Purwandari & Dewi 2025 [30] |

**Cognitive Aging Evidence:**

| Age-Related Factor | Finding | Source |
|--------------------|---------|--------|
| Working memory capacity (older adults) | ~3.8 items (vs. 5.08 for younger adults) | Brown University 2024 [16,20] |
| Cause of age-related WM decline | Reduced storage capacity, not chunking; linked to parietal glutamate decline | Brown University 2024 [16] |
| Navigation regressions | Significantly more when menu depth exceeds working memory span | TACCESS-2025 [18] |
| Speed-accuracy tradeoff | Older adults sacrifice speed to maintain accuracy | Kurniawan et al. 2002 [7] |
| CRUNCH hypothesis | Older adults reach capacity limits sooner, brain activation shifts left | CRUNCH [20] |
| Preference for non-expandable menus | Older adults prefer all options visible rather than hidden | Zaphiris 2003 [5] |

### 5.3 The "20% Rule" and Advanced Functions

The most effective role-based views satisfy approximately **80% of a user's daily tasks**. The critical design challenge is handling the remaining **20%** – occasional needs for advanced functions.

**Best Practices for the "20% Case":**

1. **Progressive Drill-Down:** Every element on the simplified view should be clickable to drill down. A KPI tile for "Inventory Alerts" leads to the full inventory management module. This respects the finding that hidden navigation causes 39% longer task times, while progressive drill-down from a visible starting point maintains discoverability [1].

2. **"More Options" Buttons:** Clearly labeled buttons that reveal advanced functionality on demand. Must be visually distinct and predictable.

3. **Non-Expandable Default with Expandable Depth:** Start with non-expandable menus showing all primary options (older adults prefer this), with clear drill-down paths to secondary and tertiary details [5, 6].

4. **Toggle/Expert Mode Switch:** Provide an "Expert Mode" toggle that reveals the full interface. This empowers users to gradually transition from simplified to comprehensive views at their own pace.

5. **Pinning and Shortcuts:** Allow users to pin frequently used advanced functions to their simplified view. This empowers gradual personalized workspace building.

6. **Action Search (Ctrl+'):** Type-ahead command search for power users (like Dynamics 365's Ctrl+') provides a progressive disclosure tool for executing commands without navigating menus [45].

7. **AI-Powered Natural Language Search:** "Ask Oracle" or Microsoft Copilot allows users to ask questions in plain language rather than navigating menus, reducing the need to learn complex menu hierarchies [21, 47].

### 5.4 The Recommended Hybrid Model

The most successful implementations for the target demographic use a **hybrid progressive model**:

**1. Default: Simplified Role-Based View**
- Clean dashboard with critical tasks only: "View Today's Work Orders," "Check Inventory for Order X," "Report Overtime"
- Maximum 5 navigation options per screen (respecting working memory limits of ~3-5 elements) [16, 20]
- Non-expandable menus showing all primary options (older adults prefer this) [5, 6]
- Large fonts (16-18pt sans-serif), 7:1 contrast ratio, high visual hierarchy [4]
- Minimum 12mm² touch targets with 4mm spacing [4]

**2. Clear Path to Depth**
- Every tile, KPI, and list item is clickable for drill-down
- "More Details" links clearly labeled and visually distinct
- Maximum 3 clicks from home screen (minimizing navigation regressions) [18]
- Avoid exceeding 2-3 levels of depth (error rates increase from 4% to 34% as depth increases) [5]

**3. Persistent Navigation as Safety Net**
- Navigation pane (Dynamics 365 sitemap, Fiori Spaces/Pages, NetSuite header) available but not required
- Serves users who are ready to explore or need unusual actions
- Global search (Alt+G for NetSuite, Ctrl+' for Dynamics 365) provides alternative access

**4. AI-Powered Search as Alternative Navigation**
- "Ask Oracle" natural language queries for users who cannot find functions via menus [21]
- Microsoft Copilot integration for Dynamics 365 [47]
- Reduces the need to learn complex navigation hierarchies

**5. Gradual Empowerment**
- Allow users to pin advanced functions to simplified view
- Provide "Show All Options" toggle for users ready to graduate
- Track user proficiency and automatically suggest advanced features when appropriate
- Scaffolded training with phased fade-out: full guidance → intermediate scaffolding → independence

**Conclusion:** Do not design a "dumbed-down" interface with no way to advance. Design a **smart default interface** that reduces cognitive load for the primary 80% of tasks, while providing clear, intuitive, and reversible pathways to the remaining 20% of advanced functionality. This directly addresses the needs of a transitioning workforce: it offers the simplicity required to get started and the power needed to grow.

As the peer-reviewed research from the Journal of Emerging Trends (2025) concludes: "Such success could be accomplished without sacrificing functionality, or producing segregated 'senior versions' of applications, hinting at how many elderly-friendly design imperatives are good design principles for all users" [4].

---

## 6. Platform-Specific Implementation Recommendations

### 6.1 SAP S/4HANA Fiori: Recommended Implementation

**Strengths for Target Demographic:**
- Strongest role-based progressive disclosure model among the three platforms
- Spaces/Pages paradigm provides clear hierarchical organization
- My Home personalized landing page with KPIs and task lists
- My Inbox consolidates all approval tasks in one place
- Fiori Elements ensure design consistency across apps

**Implementation Recommendations:**

1. **Configure role-based Spaces for each supervisor role:** Create separate spaces for "Inventory Management" and "Production Orders" with only relevant pages and apps visible.

2. **Maximize use of My Home:** Configure To Dos, favorite apps, and KPI tiles. Limit to 5-7 KPIs maximum (respecting working memory limits).

3. **Enable smart filters with defaults:** Hide advanced filter criteria by default, show on demand. Set useful default filters (e.g., "Today's Work Orders").

4. **Use collapsible sections on Object Pages:** Ensure Object Page sections are collapsed by default, with only essential fields visible initially.

5. **Pin frequently used apps to launchpad:** Ensure goods receipt, stock check, and work order confirmation apps are prominent.

6. **Train supervisors on My Inbox first:** This is the simplest entry point and provides immediate value.

**Navigation Path Optimization:**
- Check stock: 3 clicks (Launchpad → Stock app → View)
- Post goods receipt: 4 clicks (Launchpad → Post GR → Enter PO → Post)
- Approve work order: 5 clicks (Launchpad → My Inbox → Filter → Select → Approve)

### 6.2 Oracle NetSuite Redwood: Recommended Implementation

**Strengths for Target Demographic:**
- Most modern consumer-grade experience
- "Ask Oracle" AI provides alternative navigation via natural language
- Collapsible field groups on forms reduce visual clutter
- Inline editing enables task completion without page navigation
- Sticky header with persistent access to search and Create New

**Implementation Recommendations:**

1. **Maximize "Ask Oracle" for navigation:** Train supervisors to ask natural language questions (e.g., "Show me inventory for Part XYZ") rather than navigating menus. This reduces the need to learn menu hierarchies.

2. **Configure role-based Centers:** Ensure each supervisor role has a customized Center with only relevant tabs and links.

3. **Enable card-based layouts:** Use card views for inventory lists and order lists rather than dense tables.

4. **Configure collapsible field groups on inventory forms:** Set essential fields visible by default, group secondary fields into collapsible sections.

5. **Set up Quick Access Regions:** Configure for inventory management and work order tasks to eliminate additional navigation steps.

6. **Enable inline editing for inventory transactions:** Allow supervisors to edit fields directly within list views without opening separate forms.

**Navigation Path Optimization:**
- Check item quantities: 3 clicks (Inventory Management → Item Quantities → Search)
- Post item receipt (UI): 4 clicks (WMS Inbound → Post Item Receipt → Filter → Submit)
- Post item receipt (mobile): 5 taps (WMS app → Menu → Post Item Receipt → Select Type → Post)
- Global search: Alt+G + search term (2 keystrokes)

### 6.3 Microsoft Dynamics 365: Recommended Implementation

**Strengths for Target Demographic:**
- Most flexible hybrid approach with business process-driven progressive disclosure
- Production Floor Execution interface with progressive enabling
- Business Process Flows guide users step by step
- ProcessGuide framework for warehouse mobile app
- Action Search (Ctrl+') for efficient power user navigation
- Role Centers with personalized KPIs and alerts

**Implementation Recommendations:**

1. **Create Business Process Flows for inventory management and work orders:** Guide supervisors step by step through goods receipt, stock transfer, and work order confirmation processes. BPFs should have no more than 5-7 stages each.

2. **Configure Production Floor Execution interface with minimal optional features initially:** Start with essential features only (clock in/out, job start/complete, basic progress reporting). Enable advanced features (serial tracking, batch recording) after initial proficiency is established (30-60 days).

3. **Design Role Center for Production Supervisor:** Include headline KPIs (order status, work center utilization), activity queues for pending tasks, and actionable links to key pages. Limit to 5-7 KPIs maximum.

4. **Configure warehouse mobile app ProcessGuide flows:** Ensure each workflow (receive, pick, put away) has clear step-by-step guidance with step IDs, titles, and optional instructions. Start with the "Don't show again" option disabled.

5. **Train supervisors on Action Search (Ctrl+')** for efficient navigation after initial proficiency.

6. **Use Role Centers as default landing page:** Navigate supervisors directly to their personalized Role Center upon login.

**Navigation Path Optimization:**
- View production orders: 3 clicks (Navigation sidebar → Production Control → Production Orders)
- Report progress (PFE): 4 clicks (PFE interface → Select job → Report Progress → Confirm)
- Clock in (PFE): 3 clicks (PFE interface → Clock In → Enter ID → Confirm)
- Find action via Ctrl+': 2 keystrokes

### 6.4 Cross-Platform Recommendations

**Universal Design Principles for All Three Platforms:**

1. **Implement non-expandable primary menus with progressive drill-down:** Show all primary navigation options by default (older adults prefer this), with drill-down to details only when users click [5, 6].

2. **Limit navigation depth to 2-3 levels maximum:** Error rates increase from 4% to 34% as depth increases from 1 to 6 levels [5].

3. **Provide multiple navigation pathways:** Offer menu navigation, global search, AI-powered natural language search, and keyboard shortcuts (Ctrl+', Alt+G) to accommodate different user preferences and task contexts.

4. **Enable role-based views with clear "expert mode" toggle:** Default to simplified role-based view with clear pathways to full functionality for users who want or need it.

5. **Use AI-powered search as progressive disclosure tool:** "Ask Oracle" and Microsoft Copilot can serve as progressive disclosure mechanisms, revealing functions only when users explicitly search for them.

6. **Implement phased scaffolding:** Full guidance → intermediate scaffolding → minimal scaffolding → independence, with clear phase-gate milestones and 30-60 day transition periods.

7. **Provide 5-7 KPIs maximum:** Respecting working memory limits of ~3-5 items for older adults [16, 20, 18].

8. **Design for touch as primary input:** Larger touch targets (minimum 12mm² with 4mm spacing), high contrast (7:1 ratio), sans-serif fonts [4].

---

## Sources

[1] NN Group – Hamburger Menus and Hidden Navigation Hurt UX Metrics: https://www.nngroup.com/articles/hamburger-menus

[2] NN Group – Hamburger Menus Methodology: https://www.nngroup.com/articles/hidden-navigation-methodology

[3] NN Group – Hamburger Menus Video: https://www.nngroup.com/videos/hamburger-menus

[4] THE IMPACT OF UI DESIGN ELEMENTS ON COGNITIVE PERFORMANCE AMONG ELDERLY MOBILE APPLICATION USERS (JETNR, 2025): https://rjpn.org/jetnr/papers/JETNR2506007.pdf

[5] Zaphiris, Kurniawan, & Ellis (2003) – Age Related Differences and the Depth vs. Breadth Tradeoff: https://users.soe.ucsc.edu/~srikur/files/2003_ercim.pdf

[6] Zaphiris (2003) – Age Differences and the Depth - Breadth Tradeoff in Hierarchical Online Information Systems: https://ktisis.cut.ac.cy/bitstream/20.500.14279/2995/5/Age%20differences%20and%20the%20depth-breadth.pdf

[7] Kurniawan, Zaphiris, & Ellis (2002) – Expandable and Sequential Hierarchies: Gerontechnology, Vol. 2, No. 2: https://ktisis.cut.ac.cy/handle/20.500.14279/1971

[8] Kurniawan, Zaphiris, & Ellis (2002) – Full PDF: Older and Younger Adults' Time Spent and Errors: https://users.soe.ucsc.edu/~srikur/files/gerontech.pdf

[9] Psychometric Properties of NASA-TLX and Index of Cognitive Activity in Older Adults (2020, PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC7766152

[10] UTAUT Meta-Analysis and Multigenerational Studies: Venkatesh et al. (2003), PMC 2025, PMC 2015

[11] Kirkpatrick Model – Original and New World Reaction Sheets: Dr. Donald Kirkpatrick, Jim Kirkpatrick, PhD

[12] Adaptive User Interface Personalization in ERP Systems – IS MUNI: https://is.muni.cz/th/yktxa/Adaptive_User_Interface_Personalization_in_ERP_System.pdf

[13] Comprehensive ERP Usability Study – Asif et al. (2022, MDPI Applied Sciences): https://www.mdpi.com/2076-3417/12/5/2293

[14] SAP Situation Handling – Navigation Concept & Progressive Disclosure: https://community.sap.com/t5/enterprise-resource-planning-blog-posts-by-sap/situation-handling-navigation-concept-progressive-disclosure/ba-p/13476639

[15] SAP Fiori Tools UX Benchmarking: https://community.sap.com/t5/technology-blog-posts-by-sap/hitting-the-benchmark-ensuring-a-best-in-class-user-experience-for-sap/ba-p/13499245

[16] Brown University/Stanford CCN (2024) – Age-related declines in visual short-term memory: CCN 2024 proceedings; Frontiers in Aging Neuroscience

[17] Optimizing mobile app design for older adults: systematic review (Aging Clinical and Experimental Research, 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12350549

[18] Accessible Web Design for Older Adults: Challenges and Solutions (TACCESS-2025 preprint): https://digibug.ugr.es/bitstream/handle/10481/106560/TACCESS-2025-05-25-preprint.pdf

[19] Gilchrist, Cowan & Naveh-Benjamin (2008) – Working Memory Capacity for Spoken Sentences Decreases with Adult Aging, Memory journal

[20] CRUNCH Hypothesis – Compensation-Related Utilization of Neural Circuits (Brown University, various publications)

[21] Oracle NetSuite Redwood UI Migration Guide: https://houseblend.io/articles/pdfs/netsuite-redwood-ui-migration-guide.pdf

[22] Oracle NetSuite – "Ask Oracle" AI Interface: https://www.houseblend.io/articles/netsuite-next-ask-oracle-ai-interface

[23] Oracle NetSuite – Navigation Menu: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_4323290738.html

[24] Oracle NetSuite – Navigating NetSuite / Center Tabs: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_N474404.html

[25] Oracle NetSuite – Dashboards Overview: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/chapter_N576403.html

[26] Oracle NetSuite – Manually Posting Item Receipts: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_155993562169.html

[27] Oracle NetSuite – Item Receipts in WMS: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_1541435993.html

[28] Oracle NetSuite – Inbound Processing: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/chapter_1541435915.html

[29] Oracle SCM Cloud – View Item Quantities Using a Redwood Page: https://docs.oracle.com/en/cloud/saas/readiness/scm/24d/inv24d/24D-inventory-wn-f34063.htm

[30] Purwandari & Dewi (2025) – Advancements in UI and UX Design within Enterprise Systems: https://journal.idscipub.com/index.php/data/article/download/732/688/11200

[31] Oracle NetSuite – Inventory Page: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_1121043715.html

[32] Oracle NetSuite – Global Search Shortcuts: https://www.gma-cpa.com/blog/netsuite-shortcuts

[33] Oracle NetSuite – Global Search Tips: https://blog.proteloinc.com/netsuite-global-search-tips

[34] Oracle NetSuite – Global Search Tips (SuiteDynamics): https://www.suitedynamics.io/netsuite-global-search-tips

[35] Oracle NetSuite – Dashboard FAQ: https://blog.proteloinc.com/faq-netsuite-dashboards

[36] Oracle NetSuite – Best Practices for Role-Based Dashboards: https://www.kimberlitepartners.com/blog/netsuite-role-based-dashboards

[37] SAP Fiori – Spaces and Pages: https://help.sap.com/docs/btp/sap-fiori-launchpad-for-sap-btp-abap-environment/spaces-and-pages

[38] SAP Fiori – List Report and Object Page: https://help.sap.com/docs/ABAP_PLATFORM_NEW/468a97775123488ab3345a0c48cadd8f/c0eec49db81a441e878f528c8f3d28de.html

[39] Microsoft Dynamics 365 Supply Chain Management documentation: https://learn.microsoft.com/en-us/dynamics365/supply-chain

[40] Microsoft Dynamics 365 – Configure production floor execution interface: https://learn.microsoft.com/en-us/dynamics365/supply-chain/production-control/production-floor-execution-configure

[41] Microsoft Dynamics 365 – Production floor execution interface setup: https://learn.microsoft.com/en-us/dynamics365/supply-chain/production-control/production-floor-execution-setup

[42] Microsoft Dynamics 365 – Process Guide Framework: https://learn.microsoft.com/en-us/dynamics365/supply-chain/supply-chain-dev/process-guide-framework

[43] Microsoft Dynamics 365 – Warehouse Management mobile app: https://learn.microsoft.com/en-us/dynamics365/supply-chain/warehousing/warehouse-app-whats-new

[44] Microsoft Dynamics 365 – Install and configure Warehouse Management mobile app: https://learn.microsoft.com/en-us/dynamics365/supply-chain/warehousing/install-configure-warehouse-management-app

[45] Microsoft Dynamics 365 – Keyboard shortcuts: https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/fin-ops/get-started/shortcut-keys

[46] Microsoft Dynamics 365 – Supply Chain Management product page: https://www.microsoft.com/en-us/dynamics-365/products/supply-chain-management

[47] Microsoft Dynamics 365 – 2026 Wave 1 Release Plan: https://learn.microsoft.com/en-us/dynamics365/release-plan/2026wave1/enterprise-resource-planning/dynamics365-supply-chain-management/planned-features

[48] Microsoft Dynamics 365 – Business Process Flows overview: https://learn.microsoft.com/en-us/power-automate/business-process-flows-overview

[49] Microsoft Dynamics 365 – Business Process Flows (LevelShift): https://levelshift.com/blogs/business-process-flow-in-d365-explained

[50] Microsoft Dynamics 365 – Business Process Flows (Stoneridge): https://stoneridgesoftware.com/everything-you-need-to-know-about-dynamics-365-business-process-flow

[51] Microsoft Dynamics 365 – Production Floor Execution Interface (Dynamics Communities): https://dynamicscommunities.com/ug/dynamics-fo-ax-ug/boost-manufacturing-efficiency-with-d365-finance-production-floor-execution

[52] SAP Fiori – Stock Apps (F1076/MMBE, F1075): SAP Fiori Apps Reference Library

[53] SAP Fiori – Post Goods Receipt for Purchase Order (F0843A): SAP Fiori Apps Reference Library

[54] SAP Fiori – Post Goods Receipt for Inbound Delivery (F2502): SAP Fiori Apps Reference Library

[55] SAP Fiori – Perform Maintenance Jobs (F5104A): SAP Help Portal

[56] SAP Fiori – Find Maintenance Order (F2175): SAP Help Portal

[57] SAP Fiori – Find Maintenance Order Confirmation (F2174): SAP Help Portal

[58] SAP Fiori – Process Maintenance Orders (W0017): SAP Help Portal

[59] SAP Fiori – Shell and Shell Bar: https://help.sap.com/docs/SAP_S4HANA_CLOUD/4fc8d03390c342da8a60f8ee387bca1a/1fcec711535845fda50228cb294f6640.html

[60] SAP Fiori – Overview: Apps for Maintenance Orders: https://help.sap.com/docs/SAP_S4HANA_CLOUD/2dfa044a255f49e89a3050daf3c61c11/e8652a0c8733427d9cf120b60467f3b8.html

[61] Oracle NetSuite – NetSuite Next AI-native ERP: https://getcoai.com/news/oracle-netsuite-next-brings-ai-first-erp-with-natural-language-assistant

[62] Oracle NetSuite – Artificial Intelligence features: https://www.netsuite.com/artificial-intelligence.shtml

[63] Oracle NetSuite – Redwood Experience Theme documentation: https://docs.oracle.com/en/cloud/saas/netsuite/ns-online-help/section_N474404.html

[64] Oracle NetSuite – Card Layout with Progressive Disclosure: Oracle NetSuite online help

[65] Oracle NetSuite – Inline Editing with Progressive Disclosure: Oracle NetSuite online help