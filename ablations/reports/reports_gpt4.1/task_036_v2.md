# Comparative Analysis of Asynchronous and Store-and-Forward Telehealth Deployments in Rural Primary Care: Uganda, Kenya, and Tanzania under Low-Connectivity Conditions

## Introduction

Expanding health access in rural East Africa requires telehealth solutions that are both robust to low-bandwidth (2G/3G) and intermittent connectivity, and carefully integrated into existing care models. Asynchronous and store-and-forward telehealth paradigms offer a way to bridge gaps in healthcare quality, provider capacity, and health system reach, but their real-world effectiveness and sustainability depend on workflow design, robust escalation protocols, interoperability with national health systems, and alignment with local priorities. This report investigates several major deployments in Uganda, Kenya, and Tanzania (excluding Babylon Health Rwanda and mPharma), providing detailed workflow descriptions, outcome metrics, escalation protocols, standards alignment, authoritative endorsements, and analysis of operational viability—including the clinical-business paradox.

## Overview of Challenges and Context in Rural Telehealth

Rural primary care in Uganda, Kenya, and Tanzania faces chronic shortages of skilled providers, patchy digital infrastructure, and a patient population that may have low digital or health literacy and limited financial means. Asynchronous/store-and-forward models are essential due to:

- Frequent reliance on 2G/3G networks and regular connectivity outages.
- The need to support frontline workers (often Community Health Workers, nurses, or general clinicians) who must consult with specialists or health centers remotely.
- Barriers such as user reluctance, provider skepticism, systemic underfunding, gender and socioeconomic divides, and weak health IT policy implementation. 

Best-practice telehealth deployments explicitly account for these realities—both technically, through offline-first and store-and-forward architectures, and operationally, through context-sensitive escalation and workflow adaptation[1][2][3].

## Real-World Asynchronous and Store-and-Forward Telehealth Deployments

### A. Uganda: Community Health Worker (CHW) Telehealth Support via Asynchronous Call Center

- **Deployment Overview**: In the COVID-19 pandemic, Uganda implemented large-scale asynchronous telehealth support for approximately 3,500 CHWs. The system was tailored for rural, low-bandwidth (2G/3G) conditions[4].
- **Workflow**: CHWs used mobile phones (voice/SMS/data) to submit cases, clinical questions, and logistics queries asynchronously to a central telehealth center staffed by clinicians and supervisors.
    - **Data Flow**: Case details were logged with locally available connectivity (SMS/voice data fallback); clinical supervisors provided asynchronous guidance, triage, and referrals as needed.
    - **Response Cycle**: Responses logged when next online or via store-and-forward, ensuring minimal delays even during network interruptions.
    - **Escalation** (see protocols below): Unresolved or urgent cases were referred up the chain (in-person visits or next-level facility).
- **Outcome Metrics**:
    - 35,553 consults over 16 months (87% clinical, 10% logistics, 2.6% COVID-related)
    - Major reduction in CHW professional isolation
    - Enhanced routine care and medicine stockout responses
    - High ongoing usage indicates operational sustainability
    - Initial user reluctance and public skepticism overcame via persistent community engagement, contextual adaptation
- **Operational Sustainability**: Enabled through rapid, low-cost scaling, donor and government buy-in, and flexibility to context[4].

### B. Tanzania: Asynchronous Teleradiology Using ATM

- **Deployment Overview**: The “Asynchronous Transmission Mode” (ATM) teleradiology project in Tanzania targeted cost-effective, reliable radiological interpretation for remote/rural centers, operating under low connectivity[5].
- **Workflow**:
    - Digital radiological images were acquired (DICOM-compliant).
    - Images/data were stored and forwarded (asynchronously) when connectivity allowed.
    - Radiologists at distant centers retrieved and interpreted cases during availability; structured reports were returned for physician review.
    - The workflow was deliberately bandwidth-agnostic; data were queued locally and transmitted in background sessions.
- **Outcome Metrics**:
    - Demonstrated continuous operation under 2G/3G conditions
    - Improved accessibility of specialist reports in districts without in-person radiology (metrics of accuracy, timeliness, and quality in line with global standards)
    - Platform-independence, scalability, and broad applicability validated
- **Operational Sustainability**:
    - Technology adaptable to the national context, driven by local partnerships
    - Integrated into regional referral workflows[5].

### C. Kenya & Tanzania: Asynchronous Teledermatology Networks

- **Deployment Overview**: EpIC/iPath and similar platforms enabled providers in rural Kenya/Tanzania to consult dermatologists asynchronously, bypassing synchronous video/voice limitations[2][6].
- **Workflow**:
    - Local providers captured high-quality photos and clinical histories, uploaded via web/app as bandwidth allowed.
    - Remote dermatology teams reviewed and responded asynchronously (within defined turnaround windows, typically 24-72h).
    - Case status and expert advice were relayed to the frontline provider, with protocols in place for escalation (see below).
- **Outcome Metrics**:
    - >90% diagnostic concordance (agreement) between teledermatologists and in-person clinicians in comparable LMIC settings
    - Substantial reduction in access and response delays
    - High patient and provider satisfaction rates
    - Slightly lower follow-up rates in asynchronous care vs in-person: 57.8% vs 92.7% (a potential patient safety concern offset by rapid access and triage for urgent cases)
    - Demonstrated improved HIV case-finding and outpatient management continuity during pandemic-related disruptions[2][6].
- **Operational Sustainability**:
    - Integration with Ministry of Health initiatives drove long-term platform use, particularly when contextually adapted
    - Challenges included device availability, user training, and maintaining engagement in regions with severe infrastructure deficits[6].

### D. Cross-Border EHR and HIV Clinic Store-and-Forward Systems

- **Deployment Overview**: A regional EHR platform was developed in Kenya and replicated in six HIV clinics in Tanzania and Uganda, using store-and-forward principles suited to rural, low-infrastructure environments[7].
- **Workflow**:
    - Health records and patient data were updated asynchronously; when disconnected, CHWs/clinics could work offline and synchronize records when back online.
    - Platform operated in concert with mobile consults, allowing flexible continuity of HIV specialty care.
- **Outcome Metrics**:
    - System adoption and ongoing use depended on strength of local funding control and partnership with in-country academic/technical teams.
    - Demonstrated improvement in data completeness, care continuity, and linkage to national EMR reporting for HIV care[7].
- **Operational Sustainability**:
    - High sustainability when locally owned and integrated, low sustainability when operated solely by international partners.
    - Failures occurred in deployments that failed to adapt to local workflow or systems[7].

## Escalation Protocols for Workflow Resilience

### General Principles

All reviewed deployments maintained strict escalation protocols to handle situations where asynchronous/store-and-forward models could not resolve cases due to connectivity loss, diagnostic uncertainty, or deterioration of patient status.

### Protocols in Practice

- **CHW Telehealth Support (Uganda)**:
    - **Triaging**: All incoming consults triaged by remote clinicians.
    - **Escalation Timing**: Immediate for emergency/deteriorating clinical status; within hours for routine issues; within 24h for non-urgent.
    - **Channels**: Backup available via direct voice call, SMS, or supervisor dispatch; always allowed paper/in-person fallback for technical failure.
    - **Rationale**: Ensured patient safety, mitigated downside of technological interruptions[4].

- **Teleradiology (Tanzania ATM)**:
    - **If Image Not Transmittable/Unreadable**: Protocol required either repeat image acquisition or urgent physical referral to higher-level care.
    - **Escalation Rationale**: No diagnosis delivered without reliable transport or alternate evaluation channel, preventing diagnostic error from technical failure[5].

- **Teledermatology (Kenya/Tanzania)**:
    - **If Case Not Suitable/Not Conclusive**: Providers instructed to physically refer or consult via alternate real-time means.
    - **Timing**: Urgent cases flagged for priority review or direct patient transfer.
    - **Channels**: Phone, SMS, paper if online failed.
    - **Rationale**: Protects against risk of missed serious diagnoses, ensures continuity irrespective of technology[2][6].

- **EHR/HIV Clinics**:
    - **Synchronization Failures**: Local storage until reconnection, on-site care prioritized; formal escalation if synchronization exceeded preset delay.
    - **Backup**: Manual/paper processes for continuity[7].

## Health Data Standards, Interoperability, and Best-Practice Endorsement

- **Core Standards**:
    - **HL7 FHIR**: Increasingly established as the default for mobile/integrated health data exchange across the region.
    - **OpenMRS, Bahmni, DHIS2**: Widely used open-source platforms, enabling interoperability at national scale.
    - **Uganda’s Digital Health Communication Infrastructure (DHCI) Standards**: Contextualized from international baselines (HL7, IHE) to match local constraints and workflows, formally developed and awaiting final Ministry of Health endorsement[8].
- **Importance of Standards**:
    - Adoption of recognized data standards is essential to secure government, donor, and large-scale payer support. “If you’re building EMRs, telemedicine platforms, or patient registries, FHIR compatibility isn’t optional—it’s your path to government contracts and donor funding”[9].
    - Inconsistent or noncontextual adoption leads to platform failure, inefficiency, or inability to scale[3][8][9].
- **Authoritative Endorsements**:
    - WHO and ITU guidelines are embedded in most national digital health strategies, but must be contextually applied[3][8].
    - Kenya: Ministry of Health’s sectoral reports prioritize digital health, innovation, and telehealth as UHC accelerators[10][11].
    - Uganda: Developed and multi-stakeholder validated digital health interoperability standards for government rollout[8].

## The Clinical-Business Paradox in Telehealth Deployments

### Definition and Examples

The clinical-business paradox refers to the tension between platforms that achieve clinical innovation and usability versus those that achieve operational and financial sustainability:

- **Pilot vs Scale Failure**: Deployments featuring “innovative” tools (cutting-edge UI, AI triage) often fail to reach national adoption when they lack proper standards compliance, integration with government workflows, or real health system fit. Conversely, so-called “boring,” standards-based platforms are chosen for scale as they interoperate, comply with procurement, and offer operational predictability[9].
- **Uganda Example**: DHIS2 and similar platforms gained prominence not due to technical superiority alone but because they matched ministry requirements and could integrate with donor and international funders. Projects ignoring contextual adaptation failed to sustain[8].
- **Kenya/Tanzania/Uganda EHR Programs**: Sustainability was highest when local funding and operational control were present, highlighting the need for business alignment rather than pure technical innovation[7].
- **Platform Failures**: Many systems failed or withered when they had insufficient local engagement, did not match user needs, or imported standards without contextualization—leading to underutilization and, ultimately, loss of funding/support[3].

### Lessons for Future Deployments

- **Interoperability and Local Fit** matter more than novel clinical features for true scale and system transformation.
- Platforms must prioritize compliance, integration, and adaptation over pure technical innovation to survive beyond pilot stages.

## Comparative Synthesis and Workflow Design Lessons

- **Asynchronous/store-and-forward telehealth enables high completion rates and diagnostic accuracy in low-connectivity, rural East Africa**—evidenced by CHW telehealth (Uganda), teleradiology (Tanzania), teledermatology (Kenya/Tanzania), and regionally adapted EHR systems.
- **Success factors**: Contextualized workflow design, clear escalation protocols, agent/facilitator involvement, robust fallback modes (voice/SMS/paper), and adaptation to health data standards.
- **Operational sustainability** requires local funding/ownership, full interoperability, and business alignment with government or donor priorities.
- **Escalation protocols** must be rigorous, with multiple redundant communication channels to ensure safety during connection losses.
- **Clinical-business paradox** is common: only platforms matching system, funding, and standards frameworks survive long-term.

## Sources

[1] Unlocking the potential of telehealth in Africa for HIV: opportunities, challenges, and pathways to equitable healthcare delivery: https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2024.1278223/full  
[2] Telemedicine in Africa: Applications, Opportunities, and Challenges: https://www.intechopen.com/chapters/1176535  
[3] African digital health strategic plans analysis: key weaknesses in contextualization, intervention focus, and technological foresight: https://www.nature.com/articles/s41746-025-02121-z  
[4] Using telehealth to support community health workers in Uganda during COVID-19: a mixed-method study: https://pmc.ncbi.nlm.nih.gov/articles/PMC10040915/  
[5] Asynchronous Transmission Mode as the Entry Point of Teleradiology Connectivity in Developing Countries: Case Study Tanzania: https://ui.adsabs.harvard.edu/abs/2021ista.conf....8G/abstract  
[6] Telehealth & Digital Health Innovation In Africa | TAU-GY: https://tau.edu.gy/blog/how-telehealth-is-shaping-africas-public-health-future/  
[7] Experience implementing electronic health records in three East African countries: https://pubmed.ncbi.nlm.nih.gov/20841711/  
[8] Contextualised digital health communication infrastructure standards for resource-constrained settings: Perception of digital health stakeholders regarding suitability for Uganda’s health system: https://pmc.ncbi.nlm.nih.gov/articles/PMC11392385/  
[9] Interoperability Is the Moat: Why Health Data Standards Will Decide the Winners of African Healthtech – Dr. Hamza Asumah: https://hamzaasumah.org/2026/02/21/interoperability-is-the-moat-why-health-data-standards-will-decide-the-winners-of-african-healthtech/  
[10] MINISTRY OF HEALTH Sector Working Group Report - Kenya: https://www.treasury.go.ke/sites/default/files/Sector%20Budget%20Reports/Health-Sector-Report.pdf  
[11] REPUBLIC OF KENYA HEALTH SECTOR REPORT: https://newsite.treasury.go.ke/sites/default/files/Sector%20Budget%20Reports/HEALTH-SECTOR-REPORT.pdf