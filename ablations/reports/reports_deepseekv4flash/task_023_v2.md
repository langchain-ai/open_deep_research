# Comprehensive Data Breach/Cyber Insurance Evaluation: Chubb, Coalition, and Cowbell Cyber

## Executive Summary

This report provides a targeted, gap-filled analysis of cyber insurance coverage from Chubb, Coalition, and Cowbell Cyber for a 15-employee Austin, Texas business handling customer payment data (PCI-DSS) and employee health records (HIPAA), with $2-5M annual revenue. The research addresses three specific areas requiring improvement: (1) precise response time SLAs, (2) premium estimates anchored to specific underwriting conditions, and (3) a structured broker RFP specification with actionable checklists.

**Critical finding across all three carriers: None of them provide binding, contractual service-level agreements (SLAs) for incident response times in their policy forms.** The response time claims—Chubb's "24/7 hotline," Coalition's "5-minute average," and Cowbell's "acknowledgment within minutes"—are marketing descriptions, not contractual guarantees. No remedies, service credits, or penalties exist for SLA breaches. This finding fundamentally changes how a business owner should evaluate incident response capabilities: focus on documented vendor response times and contractual service agreements for incident response providers, not carrier marketing claims.

For a business with dual PCI/HIPAA exposure, premium estimates range from approximately $1,200-$2,500/year with strong security controls to $4,000-$8,000/year with weak controls, with risk of outright declination from Coalition and Cowbell if basic controls like MFA and EDR are missing.

---

## 1. PRECISE RESPONSE TIME GUARANTEES / SLAs FOR INCIDENT RESPONSE SERVICES

### CRITICAL OVERARCHING FINDING

**None of the three carriers—Chubb, Coalition, or Cowbell Cyber—provide binding, contractual service-level commitments (SLAs) for incident response times in their insurance policy forms.** The response time claims found in marketing materials are not written into policy language, endorsements, declarations pages, or separate binding service agreements. There are no remedies, service credits, liquidated damages, or penalties for failure to meet any response time target.

This finding is consistent across all three carriers and represents a fundamental limitation that every business owner must understand when evaluating cyber insurance.

---

### 1A. Chubb: Response Time Guarantees

#### What Chubb States About Response Times

Chubb offers a 24/7 Cyber Incident Response Coach hotline (800-817-2665) and a Cyber Alert app for "immediate assistance." Policyholders can email cyberclaimreport@chubb.com for commercial cyber claims. [1][2][3]

Chubb's European/Swiss marketing flyer references expert support "within 24 hours" and "cost settlement within 48 hours of an incident." [4] However, these are service descriptions in marketing materials, not contractual guarantees.

#### Are These Binding Contractual SLAs?

**NO.** Multiple official Chubb sources contain explicit disclaimers:

- "Chubb has no obligation to provide any cyber services for incident response or loss mitigation." [5][6]
- "Chubb assumes no liability arising out of any services rendered by a cyber service provider. Chubb also assumes no liability arising out of a delay in user access to any cyber service provider portal, or delay in services rendered by a cyber service provider." [7]
- "Chubb assumes no liability for services rendered by Cyber Incident Response Team providers and does not endorse their services." [3]
- "Chubb's cyber services cannot be construed to replace any provisions of your policy." [8]

#### Cyber Incident Response Coach: Guaranteed or Best-Effort?

**Best-effort only.** Chubb explicitly states: "Chubb has no obligation to provide any of the legal, computer forensic, notification, call center, public relations, crisis communications, fraud consultation, credit monitoring and identity restoration advice and services provided by the Cyber Incident Response Team." [5][6]

The Cyber Incident Response Team members are "independent contractors, and not agents of Chubb." Policyholders may "select their own approved firms" and are under no obligation to use Chubb's panel. [3][5]

#### Where Are SLAs Located?

Response time descriptions appear **only in marketing materials**—Chubb's website, service flyers, and the Cyber Alert app descriptions. They are **not** in the Cyber ERM policy form (Version 2.2), any endorsement, declarations page, or a separate binding service agreement. [9][10][11][12]

#### Remedies for SLA Breach

**None.** No service credits, penalty payments, liquidated damages, or premium reductions exist for failure to meet response time targets. Chubb explicitly disclaims all liability for incident response services and their timeliness.

#### Legal Defense Model: Reimbursement (Not Duty to Defend)

Chubb's Cyber ERM operates on a **reimbursement/indemnity model**, not a duty to defend model. Policy language consistently uses "We will pay on Your behalf" rather than "We have the right and duty to defend." [9][10][11][12]

Implications for policyholders:
- You select your own breach counsel and incident response vendors (or use Chubb's panel)
- Chubb reimburses covered expenses after the fact
- You may face cash-flow challenges and potential reimbursement delays
- Defense costs reduce and may exhaust policy limits [13][14]

#### Sublimit Structure for PCI and HIPAA Coverage

From actual policy declarations [15][16]:

| Coverage | La Leche League (2022-2023) | Entigrity (2025-2026) | Danville, IN (2022-2023) |
|----------|---------------------------|----------------------|-------------------------|
| Payment Card Loss (PCI) | $250,000 sublimit | $250,000 sublimit | $1,000,000 (aggregate) |
| Regulatory Proceedings (HIPAA) | $250,000 sublimit | $250,000 sublimit | $1,000,000 (aggregate) |

#### P.F. Chang's Precedent: Critical Warning for PCI-Exposed Businesses

In *P.F. Chang's China Bistro v. Federal Insurance Company* (2016, District of Arizona), a Chubb subsidiary denied coverage for $1.9 million in MasterCard assessments passed through P.F. Chang's acquiring bank. The court held: [17][18][19]

1. **Privacy Injury coverage did not apply** because the acquiring bank (which suffered the loss) did not have its data compromised—the customers' data was compromised.
2. **Contractual liability exclusion barred coverage** because P.F. Chang's had contractually agreed to reimburse the acquiring bank for card brand assessments.
3. **Reasonable expectations argument failed**—the court ruled actual policy terms control over marketing language.

**Implications:** Businesses relying on Chubb for PCI coverage must carefully review how contractual liability exclusions might impact coverage for PCI assessments passed through merchant acquirers. While Chubb now includes Payment Card Loss coverage in the base form, the contractual liability exclusion remains and could limit coverage depending on how assessments are triggered.

---

### 1B. Coalition: Response Time Guarantees

#### What Coalition States About Response Times

Coalition prominently markets "5-minute average claims response time," "64% of closed claims resolved with no out-of-pocket loss," and "two-hour legal consultation at no cost." [20][21][22]

#### Are These Binding Contractual SLAs?

**NO.** The widely cited "5-minute average claims response" is a marketing claim based on averages, not a contractual guarantee with penalties for breach. The actual Coalition Cyber Policy form (SP 14 798 0419) contains **zero language** guaranteeing any specific response timeframe. [23]

The "two-hour legal consultation at no cost" is also a marketing claim. Coalition's coverages page states: "Rapid Response Services provide immediate access to critical resources during an incident, including a free two-hour legal consultation and Coalition Incident Response support." [24] However, the actual policy form contains no mention of "two-hour" or any specific timeframe for legal consultation.

#### Where Are SLAs Located?

Response time claims exist **only on Coalition's website and in promotional materials.** They are NOT embedded in: [23]
- The policy form (SP 14 798 0419)
- The Declarations page
- Any endorsements attached to the policy

The only contractual service-type provision found is "Pre-Claim Assistance" (Item 6 of Declarations, typically $610-$1,670 limit [23][25]), which states the insurer "may, in our discretion, agree to pay for up to the amount shown... in legal, forensic, and IT fees." The phrase "in our discretion" confirms this is not a guaranteed service-level commitment.

#### Remedies for SLA Breach

**None.** The policy contains no service credits, liquidated damages, or penalties for response time failures. Standard insurance remedies (bad faith claims, breach of contract) would apply under state insurance law if Coalition breaches its duties, but no specific SLA penalty provisions exist.

#### Legal Defense Model: Duty to Defend (Defense Inside Limits)

Coalition operates on a **duty to defend** basis. The policy states: "We will have the right and duty to defend any claim against you seeking damages that are payable under the terms of this Policy." [23]

**Defense costs are INSIDE the limits:** "The Limits of Liability of this Policy will be reduced and may be completely exhausted by payment of claim expenses. Our duty to defend ends once the applicable Limit of Liability is exhausted." [23]

This means every dollar spent on legal defense reduces the amount available for settlements, judgments, and other covered losses.

#### Notable Policy Features

**Vanishing Retention (Active Cyber Policy, April 2025):** Retention reduces to zero over three claim-free years (25% Year 1, 50% Year 2, 100% Year 3). [26]

**$0 Retention for Coalition Incident Response:** Policyholders using Coalition's incident response services see no out-of-pocket costs for these services. [26]

**Reduced FTF Retention for 72-Hour Reporting:** Lower retentions for funds transfer fraud reported within 72 hours. [26]

---

### 1C. Cowbell Cyber: Response Time Guarantees

#### What Cowbell States About Response Times

Cowbell markets "acknowledgment within minutes" from claim receipt, "urgent issues addressed within 1 hour," and through its partner CyberClan, "guarantees a response within 15 minutes." [27][28][29]

#### Are These Binding Contractual SLAs?

**NO.** The sample Commercial Cyber Insurance Policy (issued by Spinnaker Insurance Company) contains **no response time SLAs, guarantees, or service level commitments.** [30] The policy form focuses on coverage grants, definitions, conditions, exclusions, and claims reporting requirements—not on carrier response time performance.

Cowbell 365, launched February 28, 2023, provides "24/7 service delivering comprehensive cyber claims handling and risk mitigation support" but these are service descriptions, not contractual SLAs with remedies. [31]

#### CyberClan's "15-Minute Response": What It Actually Is

**This is a vendor-level marketing guarantee, NOT a guarantee written into the Cowbell insurance policy.** [32][33]

CyberClan's Warranty Program guarantees "a 15-minute emergency response time available 24/7/365 across US, Canada, UK, and Australia." However, this warranty:
- Applies only if a third party obtains unauthorized access via a **protected endpoint** resulting in data exfiltration or loss exceeding $8,000
- Requires a separate 12-month managed security service subscription
- Has tiered warranty caps ($100K Basic, $500K Enhanced, $2M Complete)
- Is CyberClan's own vendor commitment, not Cowbell's

Cowbell policyholders who use CyberClan through Cowbell RX receive a discounted rate ($295/hour, 14.5% off standard), but the 15-minute response is not a contractual obligation of Cowbell. [32][33][34]

#### Where Are SLAs Located?

Response time commitments appear **on Cowbell's website, partner pages, and marketing materials.** They are **not** in the insurance policy form, any endorsement, declarations page, or separate binding service agreement. [30][34]

#### Remedies for SLA Breach

**None.** No evidence exists of service credits, remedies, or penalties for breached response time SLAs. Since no binding SLAs exist in the policy, no penalty provisions are included.

#### Legal Defense Model: Duty to Defend (Defense Inside Limits)

Cowbell uses a **duty to defend** model. The policy states: "We shall have the right and duty to select counsel and defend the Insured against any Claim covered under Security Breach Liability, even if allegations are groundless, false or fraudulent." [30]

**Defense costs are INSIDE limits:** "Defense expenses, where applicable, are included in the limits of insurance and will erode the limits." [30][34] This means defense expenditures reduce available coverage for settlements and other losses.

#### Legal Precedent: Perry & Perry Builders v. Cowbell Cyber

In a March 2026 decision by the Western District of Texas, the court enforced a $250,000 cybercrime sublimit in a Cowbell policy despite the policyholder suffering nearly $875,000 in fraudulent electronic fund transfers. [35]

The court concluded that the sublimit capped recovery across the policy and did not permit limits to be multiplied based on the number of transfers. This case is directly relevant to Texas businesses and demonstrates that Cowbell's sublimit endorsements are strictly enforced.

---

## 2. PRECISE PREMIUM ESTIMATES ANCHORED TO SPECIFIC UNDERWRITING CONDITIONS

### National Premium Benchmarks (2025-2026)

| Source | Monthly Average | Annual Average | Notes |
|--------|----------------|----------------|-------|
| MoneyGeek (2026) | $83/mo | $999/yr | For $1M aggregate, small business average |
| Insureon (2026) | $129/mo | $1,552/yr | Based on 40,000+ policies [36] |
| Security.org (2026) | ~$120/mo | $1,438/yr | Business policies [37] |
| Pro Insurance Group (2026) | $100-$300/mo | $1,200-$3,600/yr | For $1M coverage [38] |
| Mitchell-Joseph (2025) | ~$145/mo | $1,740/yr | Typical small business [39] |

**Texas-Specific Context:** Texas ranks second nationwide for reported cyber incidents, with North Texas accounting for 37% of all cases. [40] Texas House Bill 3746 requires breach notification within 60 days. [40] Texas accounts for 9.70% of total U.S. cyber insurance Direct Written Premium, ranking 4th among all states. [41]

### Scenario A: Weak Controls (Baseline/High Risk)

**Assumptions:** No MFA deployed, no EDR/antivirus, no full-disk encryption, no written incident response plan, no security awareness training, infrequent or untested backups, outdated patches.

| Carrier | Estimated Annual Premium | Deductible/Retention | Key Considerations |
|---------|------------------------|---------------------|-------------------|
| **Chubb** | $4,500 - $7,500 | $10,000 - $25,000 | Likely to quote with significant surcharge. 10-25% credit opportunity missed. [16][15] |
| **Coalition** | Likely **Declined** or $5,000 - $8,000+ | $10,000 - $25,000 | Coalition has **strict essential requirements** including MFA, EDR, training, and backups. Without these, they will likely **decline coverage entirely.** [42][43] |
| **Cowbell** | $3,500 - $6,000 | $2,500 - $10,000 | Poor Cowbell Factors score will drive premium to high end. May require manual underwriting. [44][45] |
| **The Hartford** | $3,000 - $5,500 (if coverable) | Varies | May decline weak-control risks [46] |
| **Hiscox** | $3,000 - $6,000 | Varies | Generally requires basic controls [47] |

**Scenario A Expected Range: $4,000 - $8,000/year with significant risk of outright declination from Coalition and potentially Cowbell.**

**Underwriting Reality:** "Missing basic controls like MFA or EDR can add 25% to 50% to your quote or disqualify you entirely." [36] "73% of SMBs fail cyber insurance assessments in 2026 due to weak controls, missing documentation, and reactive security." [48]

### Scenario B: Strong Controls (Best-Case/Low Risk)

**Assumptions:** MFA enabled on all systems (email, VPN, remote access, privileged accounts), EDR deployed on all endpoints, full-disk encryption on all devices, weekly offline/immutable backups tested and verified, written and tested incident response plan, phishing/social engineering training completed for all employees annually, all patches current, documented access controls, vendor risk management in place.

| Carrier | Estimated Annual Premium | Deductible/Retention | Applicable Premium Credits | Key Considerations |
|---------|------------------------|---------------------|--------------------------|-------------------|
| **Chubb** | $1,800 - $3,200 | $1,000 - $5,000 (likely $2,500) | 10-25% for documented controls (MFA, EDR, encryption, IR plan, training); Neglected Software Exploit 45-day grace period [49][10] | BriteProtect XDR available at $72/device/yr (~$1,080 for 15 devices) [50] |
| **Coalition** | $1,200 - $2,500 | $1,000 - $2,500 | Up to 12.5% for MDR (Coalition MDR, CrowdStrike, SentinelOne); Vanishing Retention (up to 100% reduction over 3 years); $0 retention for IR services [51][52][26] | Ideal risk profile. All five essential requirements met. 73% fewer claims for Control users. [21][42] |
| **Cowbell** | $1,100 - $2,000 | $1,000 - $2,500 | Up to 5% for Connector activation (Prime 250 only); Improved Cowbell Factors score at renewal [53][54][55] | Minimum premium starts at $1,100 for $1M limit [44] |
| **The Hartford** | $1,200 - $2,500 | $1,000 - $2,500 | Standard control-based credits | Strong control profile [46] |
| **Hiscox** | $1,000 - $2,000 | $1,000 - $2,500 | Starting at $30/mo for best-case [47] | Good fit for well-controlled small business |

**Scenario B Expected Range: $1,100 - $3,200/year with best-case being $1,100-$1,500 from Cowbell or $1,200-$2,000 from Coalition.**

**Underwriting Reality:** "MFA alone often saves 10-20% on premiums." [39] "Implementing security measures such as MFA and regular training can reduce premiums by up to 25% in Texas." [40] "Strong security controls can reduce quotes by 20-30%." [36]

### Premium Credits Summary Table

| Carrier | Credit Type | Amount | Requirements |
|---------|-------------|--------|--------------|
| **Chubb** | Standard security controls | 10-25% estimated | MFA, EDR, encryption, IR plan, training, patching [49][36][39] |
| **Chubb** | Neglected Software Exploit | Risk-sharing shift | Patch within 45 days of CVE publication [10] |
| **Coalition** | MDR Premium Credit | Up to 12.5% | Coalition MDR, CrowdStrike Falcon Complete, SentinelOne Vigilance Respond [51][52] |
| **Coalition** | Security Awareness Training | Up to $100K FTF increase | Purchase Coalition SAT [56] |
| **Coalition** | Vanishing Retention | 25%/50%/100% reduction over 3 years | Claim-free consecutive years [26] |
| **Cowbell** | Connector Credit | Up to 5% (Prime 250 only) | Activate one or more Cowbell Connectors [53][54][55] |
| **Cowbell** | Improved Risk Rating | Variable at renewal | Better Cowbell Factors score from implementing recommendations [55] |

### Dual PCI/HIPAA Exposure Impact

The business's dual exposure creates a **moderate-high** risk classification. Key underwriting considerations:

- **PCI-DSS (Level 4 Merchant):** Requires annual SAQ and quarterly ASV vulnerability scans [57]. "While using a third party does not exempt a company from PCI compliance, it can simplify the PCI compliance process. However, a third-party breach means your client is still legally obligated to notify their clients." [57]
- **HIPAA:** Requires documented compliance with Security Rule, Privacy Rule, and Breach Notification Rule. Healthcare sector prices have risen slightly due to claims experience [58].
- **Dual Exposure Premium Impact:** Compared to a general small business, expect a 20-40% loading for regulated data exposure. This aligns with both Scenario A and B estimates above.

### Texas Market Conditions

- Cyber insurance rates in the U.S. declined an average of 5% in Q4 2024—first quarterly decrease after seven years of rising rates [41].
- U.S. cyber market DWP was ~$9.14 billion in 2024, a 7% decrease from 2023 [41].
- Despite 40% rise in reported claims (nearly 50,000), average ransomware payments dropped 77% in 2024 [41].
- Texas House Bill 3746 requires notification within 60 days, making cyber insurance crucial for compliance [40].

---

## 3. STRUCTURED BROKER RFP SPECIFICATION / STEP-BY-STEP CHECKLIST

### A) Coverage Target Checklist

Use this table to compare specific coverage items across carriers. Check whether each carrier provides the coverage, note the sublimit/limit, and verify conditions.

| Coverage Item | Chubb (Cyber ERM) | Coalition (Active Cyber Policy) | Cowbell (Prime 100/Pro) | Your Notes |
|---------------|-------------------|--------------------------------|------------------------|------------|
| **1. PCI Fines & Assessments** | Yes - "Payment Card Loss" coverage. Sublimit typically $250K-$1M (varies by policy). [15][16] | Yes - Coverage D. $1M limit in sample policy (shared aggregate). [23] | Yes - "PCI Fines & Penalties." Sublimit set at underwriting. [59] | ________ |
| **2. HIPAA/HITECH Regulatory Fines & Penalties** | Yes - "Regulatory Proceedings" coverage. Sublimit typically $250K-$1M. [15][16] | Yes - Coverage B. Regulatory Defense and Penalties. $1M sample. [23] | Yes - "Regulator Defense & Penalties." Sublimit set at underwriting. [59] | ________ |
| **3. Regulatory Defense Costs** | Included in Regulatory Proceedings sublimit. Defense **inside limits.** [60][61] | Included in Coverage B. Defense **inside limits.** [23] | Included. Defense **inside limits.** [30] | ________ |
| **4. Breach Response / Incident Response Expenses** | Yes - "Cyber Incident Response Fund." Retention may apply ($0 if quoted). [12][15] | Yes - Coverage E. Breach Response. **$0 retention** when using Coalition IR. [26] | Yes - Security Breach Expense. Retention applies. [59] | ________ |
| **5. Credit Monitoring** | Yes - **No time cap** (covered as required by law). [60] | Yes - Covered as breach response cost. Duration per policy terms. [23] | Yes - Typically 1-year identity protection. Duration per policy. [59] | ________ |
| **6. Legal Defense** | **Reimbursement model.** You choose counsel (or use panel). Chubb reimburses covered costs. Defense **inside limits.** [60][61] | **Duty to Defend.** Insurer selects panel counsel. Defense **inside limits.** [23] | **Duty to Defend.** Insurer selects counsel. Defense **inside limits.** [30] | ________ |
| **7. Ransomware/Extortion** | Yes - Ransomware Encounter Endorsement. Sublimit varies. [10][12] | Yes - Coverage G. Cyber Extortion. $1M sample. [23] | Yes - Extortion Threats & Ransom Payments. Sublimit varies. [59] | ________ |
| **8. Business Interruption** | Yes - Business Interruption Loss. Waiting period applies. [12] | Yes - Coverage H. Business Interruption & Extra Expenses. 8-hour waiting period. [23] | Yes - Business Income & Extra Expense. 6-hour waiting period. [59] | ________ |
| **9. Social Engineering / Funds Transfer Fraud** | Yes - Via endorsement (computer fraud, FTF, social engineering). [10] | Yes - Coverage J. Funds Transfer Fraud. $250K sublimit / $12.5K retention in sample. [23] | Yes - Social Engineering ($250K sublimit typical); Computer & Funds Transfer Fraud. [62][35] | ________ |
| **10. Notification Costs** | Yes - Covered under Cyber Incident Response Fund. [12] | Yes - Covered under Breach Response (Coverage E). [23] | Yes - Covered under Security Breach Expense. [59] | ________ |
| **11. Public Relations / Crisis Management** | Yes - Included in incident response services. [3] | Yes - Coverage F. Crisis Management & PR. [23] | Yes - Public Relations Expense. [59] | ________ |
| **12. Call Center Services** | Yes - Included in incident response. [3] | Yes - Covered under Breach Response. [23] | Yes - Covered under Security Breach Expense. [59] | ________ |
| **13. Forensic Investigation Costs** | Yes - Covered under Cyber Incident Response Fund. [12] | Yes - Covered under Breach Response. [23] | Yes - Covered under Security Breach Expense. [59] | ________ |
| **14. Data Restoration** | Yes - Digital Data Recovery. [10] | Yes - Coverage I. Digital Asset Restoration. [23] | Yes - Restoration of Electronic Data. [59] | ________ |
| **15. Dependent/Business Partner Loss** | Yes - Contingent Business Interruption. [10] | Ask broker - Not explicitly named in sample. | Ask broker - Prime 250 has Contractual Damages Endorsement. [63] | ________ |
| **16. System Failure (Non-Malicious)** | Yes - Preventative Shutdown endorsement. [12] | Included in Business Interruption (systems failure). [23] | **Excluded** in Prime 100. Included in Prime 100 Pro. [59] | ________ |

### B) Data Requirements Checklist

Organize the following information before requesting quotes:

#### Company Information
- [ ] Legal business name, DBA (if any), and entity structure (LLC, S-Corp, etc.)
- [ ] Years in operation
- [ ] Annual revenue for current and prior year ($2-5M)
- [ ] Number of employees (15 full-time equivalent)
- [ ] Physical address (Austin, TX)
- [ ] NAICS code (for industry classification)
- [ ] Prior cyber insurance carrier, policy limits, deductibles, and premium
- [ ] Prior claims history (last 5 years): date, type of incident, cost, resolution, impact on premium

#### Security Controls
- [ ] **MFA:** Document where MFA is deployed (email, VPN, remote access, privileged accounts, all systems). Provide screenshots or configuration reports.
- [ ] **EDR:** Specify which EDR solution, coverage across all endpoints, alert handling procedures, and mean time to respond.
- [ ] **Encryption:** Full-disk encryption status on laptops, desktops, mobile devices. Encryption in transit (TLS/SSL) for data transmission.
- [ ] **Backups:** Frequency, type (full/incremental), storage location (offsite, immutable, cloud), retention period, and documented restore testing results.
- [ ] **Patch Management:** Formal patching policy, frequency, evidence of timely application of critical patches within 14-30 days. Document CVE tracking process.
- [ ] **Incident Response Plan:** Written plan with defined roles, escalation procedures, containment steps, evidence preservation, communications plan, and recovery procedures. Date of last tabletop exercise.
- [ ] **Access Controls:** Documented identity and access management, privileged access management (PAM), principle of least privilege enforcement.
- [ ] **Email Security:** DMARC, DKIM, SPF configuration. Security awareness training completion rates.
- [ ] **Network Security:** Firewall configuration, network segmentation, RDP exposure status, VPN usage.
- [ ] **Vulnerability Scanning:** Regular internal and external scans with remediation tracking.

#### Data Handling
- [ ] **Types of data stored:** Payment card data (PCI), protected health information (PHI/PHI)
- [ ] **Volume of records:** Number of customer payment records, number of employee health records
- [ ] **Data storage locations:** Cloud providers (AWS, Azure, Google Cloud, etc.), on-premise servers, third-party platforms
- [ ] **Third-party vendors with data access:** Payment processors, cloud services, healthcare clearinghouses, billing services, payroll providers
- [ ] **Data retention policies:** Documented schedules for data retention and destruction
- [ ] **Data disposal procedures:** Secure destruction methods for physical and digital media

#### Compliance Posture
- [ ] **PCI-DSS:** SAQ type completed, date of last assessment, ASV scan results, compliance validation status
- [ ] **HIPAA:** Security Risk Assessment completion date, documented policies and procedures, training completion records, breach notification procedures
- [ ] **Recent audits/assessments:** Penetration test results, vulnerability assessment reports, third-party security audit findings (with remediation status)
- [ ] **Framework alignment:** NIST CSF, CIS Controls, or other framework alignment documentation

### C) Key Questions to Ask Each Carrier/Broker

#### Exclusion Questions (General)
1. "Does your policy have a contractual liability exclusion? Please provide the exact wording so we can evaluate how it would apply to our PCI and HIPAA contractual obligations."
2. "How would your contractual liability exclusion apply to PCI assessments passed through our merchant acquiring bank?"
3. "Are there any exclusions specific to unencrypted data? What constitutes 'encrypted' under your policy?"
4. "Does your policy exclude coverage for voluntary system shutdowns or preventative measures?"
5. "Are there any exclusions related to third-party vendor liability or cloud service provider incidents?"
6. "What is your policy's treatment of prior knowledge or known vulnerabilities?"
7. "Does your policy include a 'war exclusion' and how might it apply to state-sponsored cyber attacks?"

#### Chubb-Specific Questions
8. "Given the P.F. Chang's precedent, please confirm in writing how your current Payment Card Loss coverage handles contractual PCI assessments passed from merchant acquirers."
9. "Please confirm that the Cyber Incident Response Coach services are contractual obligations, not best-effort offerings."
10. "What is the specific sublimit for Payment Card Loss and Regulatory Proceedings in the quote you are providing?"
11. "Does the Neglected Software Exploit Endorsement apply to our policy? What are the coinsurance percentages at each milestone?"
12. "Please explain how the reimbursement model works—specifically, what are the timelines for reimbursement approval and payment?"

#### Coalition-Specific Questions
13. "Please confirm in writing whether the '5-minute average claims response' and 'two-hour legal consultation' are contractual SLAs with remedies for breach."
14. "What is the specific sublimit for Funds Transfer Fraud? Is the $12,500 retention shown in sample policies standard for our profile?"
15. "Does the $0 retention for Coalition Incident Response apply to all breach response services or only certain services?"
16. "How does the Vanishing Retention feature apply to each coverage agreement? Does it apply to Funds Transfer Fraud separately?"
17. "Please confirm that our business is not excluded under the 'payment processor' exclusion category. We accept credit cards as a merchant, not as a third-party processor."

#### Cowbell-Specific Questions
18. "Please confirm in writing whether the CyberClan 15-minute response guarantee is a contractual obligation of Cowbell or a vendor-level guarantee that requires a separate subscription."
19. "What is the specific sublimit for Social Engineering coverage? Is it an aggregate cap for the policy period or per-incident?"
20. "Does our business qualify for the Prime 100 Pro product, which includes system failure coverage? What would the premium difference be?"
21. "What is the minimum premium for our profile? Cowbell markets a $1,100 minimum—does that apply to our dual-exposure business?"
22. "How does the Cowbell Connector premium credit work for Prime 100 (not Prime 250) policyholders?"
23. "Please explain how defense costs are allocated between coverages when a claim involves multiple insuring agreements."

#### Broker/Carrier Comparative Questions
24. "Please provide a side-by-side comparison of sublimits for each coverage item across all three carriers."
25. "For each carrier, please confirm whether defense costs are inside or outside the aggregate limit."
26. "What is the maximum aggregate limit available from each carrier for our business profile?"
27. "Are there any premium credits or discounts available for our specific security controls beyond standard MFA/EDR incentives?"
28. "What is the claims handling philosophy—dedicated claims adjuster per claim? In-house team? Outsourced?"
29. "How does each carrier handle allocation of defense costs between covered and uncovered claims?"
30. "Are there any mandatory arbitration provisions or choice-of-law clauses that affect claims in Texas?"

---

## Summary Comparison Table

| Dimension | Chubb | Coalition | Cowbell Cyber |
|-----------|-------|-----------|---------------|
| **Policy Form** | Cyber ERM V2.2 (claims-made) | Active Cyber Policy (claims-made, April 2025) | Prime 100/100 Pro/250 (claims-made) |
| **Response Time SLAs** | NO binding SLAs. Marketing only. | NO binding SLAs. "5-minute avg" is marketing. | NO binding SLAs. CyberClan 15-min is vendor-level. |
| **SLA Breach Remedies** | None | None | None |
| **Legal Defense Model** | **Reimbursement** (you pay, Chubb reimburses) | **Duty to Defend** (insurer controls defense) | **Duty to Defend** (insurer controls defense) |
| **Defense Costs Location** | Inside limits | Inside limits | Inside limits |
| **PCI Fines Coverage** | Yes (Payment Card Loss) - but P.F. Chang's precedent limits contractual assessments | Yes - Affirmative Coverage D | Yes - PCI Fines & Penalties |
| **HIPAA Regulatory Coverage** | Yes - Regulatory Proceedings | Yes - Coverage B | Yes - Regulator Defense & Penalties |
| **Credit Monitoring Duration** | No time cap (as required by law) | Per policy terms (typically limited) | Per policy terms (typically 1 year) |
| **Vendor Selection** | Your choice (panel or own) | Panel with prior approval | Curated panel (terms govern) |
| **Weak Controls Premium** | $4,500-$7,500 | Likely Declined or $5,000-$8,000+ | $3,500-$6,000 |
| **Strong Controls Premium** | $1,800-$3,200 | $1,200-$2,500 | $1,100-$2,000 |
| **Texas Availability** | Yes (Top 3 carrier) [36] | Yes (Top 3 carrier) [36] | Yes (expanded to TX 2020) [64] |
| **Financial Strength** | A++ (Superior) AM Best | Multiple A-rated carriers | A/A- AM Best (Palomar/Spinnaker) |

---

## Recommendations

### Immediate Actions

**1. Institute Minimum Security Controls Immediately:**
Before seeking quotes, implement MFA on all systems, deploy EDR across all endpoints, enable full-disk encryption, and document backup procedures with restore testing. These controls are table stakes for insurability and can reduce premiums by 25-30%.

**2. Request Quotes with Both Scenarios:**
Ask each broker to provide quotes for both weak-control and strong-control scenarios to quantify the financial benefit of security improvements.

**3. Focus on Policy Language, Not Marketing:**
Given that no carrier provides binding response time SLAs, evaluate incident response capabilities based on:
- Actual vendor contracts with pre-approved incident response providers
- Documented response time guarantees from forensic vendors (e.g., CrowdStrike, Mandiant, CyberClan)
- Independent incident response retainer agreements

**4. Prioritize Coalition for Strong-Control Scenario:**
With all controls in place, Coalition offers the most innovative features for your profile:
- Vanishing Retention rewards good security hygiene
- $0 retention for Coalition Incident Response services
- Unlimited reinstatements for first-party incidents
- Explicit PCI and HIPAA coverage (avoiding Chubb's P.F. Chang's issues)

**5. Review Chubb's Contractual Liability Exclusion Carefully:**
If considering Chubb, obtain written confirmation from the broker on how PCI assessments triggered through merchant acquirers would be treated under the current policy form.

**6. Investigate Cowbell for Price-Sensitive Scenarios:**
Cowbell's $1,100 minimum premium offers the lowest entry point, but verify that the Prime 100 product (which excludes system failure) is appropriate for your needs. The Prime 100 Pro includes system failure coverage but may cost more.

**7. Engage a Specialized Cyber Insurance Broker:**
Work with a broker who specializes in cyber insurance for small businesses with regulated data exposure. Provide them with the completed Data Requirements Checklist and Coverage Target Checklist from Section 3.

---

### Sources

[1] Chubb - Cyber Incident Response Solutions & Support (U.S.): https://www.chubb.com/us-en/business-insurance/products/cyber-insurance/cyber-incident-response-solutions.html

[2] Chubb's Cyber Service Solutions (PDF): https://mccmeetingspublic.blob.core.usgovcloudapi.net/watertwnwi-meet-ddd92075b6f148dd8311d2d1b7f3f4b9/ITEM-Attachment-001-0edb40b235f04e74acefb67d9c01f8d7.pdf

[3] Chubb - Cyber Services for Incident Response (PDF): https://www.fcbanking.com/media/2333/chubb-cyber-services-incident-response-services.pdf

[4] Chubb Cyber ERM - Incident Response Hotline (Switzerland/PDF): https://www.chubb.com/content/dam/chubb-sites/chubb-com/ch-en/our-solutions-n/cyber-product/documents/pdf/cyber_erm_incident_response_flyer_ch_en2.pdf

[5] Chubb - Cyber Partners for Mitigation and Response (U.S.): https://www.chubb.com/us-en/business-insurance/products/cyber-insurance/cyber-partners-for-mitigation-and-response.html

[6] Chubb Malaysia - Cyber Incident Response Vendors: https://www.chubb.com/my-en/business-insurance/products/cyber-erm/cyber-incident-response-index.html

[7] Chubb Singapore - Cyber Services Forms: https://www.chubb.com/sg-en/business-insurance/cyber-erm/cyber-services-forms.html

[8] Chubb U.K. - Cyber Incident Response Vendors: https://www.chubb.com/content/chubb-sites/chubb-com/emea/uk/en/business/products/cyber-erm/cyber-incident-response-index.html

[9] Chubb Cyber Enterprise Risk Management V2.2 Policy - Lloyd's Market Association (PDF): https://lmalloyds.com/wp-content/uploads/2025/09/Chubb-Cyber-ERM-V2.2-Policy-Documents-1-003-International.pdf

[10] Chubb - Cyber Enterprise Risk Management (PDF): https://www.chubb.com/content/dam/chubb-sites/chubb-com/us-en/business-insurance/cyber-enterprise-risk-management-cyber-erm/documents/pdf/17010185-cyber-erm-12.17.pdf

[11] Chubb Cyber Enterprise Risk Management V2.2 Policy - Australia (PDF): https://www.chubb.com/content/dam/chubb-sites/chubb-com/au-en/business/cyber-insurance/documents/pdf/cyber-erm-version-2-2-policy-sme-marketplace-platform-au.pdf

[12] Town of Danville, IN - Chubb Cyber ERM Policy Quotation (PDF): https://danvillein.gov/egov/documents/1644588217_45448.pdf

[13] La Leche League International - Chubb Cyber ERM Declarations (PDF): https://llli.org/wp-content/uploads/D52824035-LA-LECHE-CYBER-CHUBB.pdf

[14] Entigrity Solutions LLC - Chubb Cyber ERM Declarations (PDF): https://my-cpe.com/erp-my-cpe/custom-docs/Cybersecurity-Insurance-Coverage.pdf

[15] La Leche League Cyber ERM Declarations - Full document: https://llli.org/wp-content/uploads/D52824035-LA-LECHE-CYBER-CHUBB.pdf

[16] Entigrity Cyber ERM Declarations - Full document: https://my-cpe.com/erp-my-cpe/custom-docs/Cybersecurity-Insurance-Coverage.pdf

[17] McGuireWoods - "Arizona Court Rules That Chubb Cyber Policy Does Not Cover Credit Card Theft Losses": https://www.mcguirewoods.com/client-resources/alerts/2016/6/arizona-court-rules-chubb-cyber-policy-not-cover-credit-card-theft

[18] Inside Privacy - "P.F. Chang's Ruling Highlights Potential Pitfalls of Cyber Insurance": https://www.insideprivacy.com/data-security/p-f-changs-ruling-highlights-potential-pitfalls-of-cyber-insurance

[19] Dentons - "Arizona court applies traditional exclusion in modern 'cyber' coverage form": https://www.dentons.com/en/insights/articles/2016/july/11/arizona-court-applies-traditional-exclusion-in-modern-cyber-coverage-form

[20] Coalition - 2025 Cyber Claims Report (May 7, 2025): https://www.coalitioninc.com/announcements/2025-cyber-claims-report

[21] Coalition - Active Cyber Insurance & Security for Business: https://www.coalitioninc.com/business

[22] Coalition - 5 Key Takeaways from Activate 2025: https://www.coalitioninc.com/blog/cyber-insurance/5-key-takeaways-from-activate-2025

[23] Ventura Country Club - Coalition Cyber Insurance Policy (Dec 2024-Dec 2025): https://venturacc.org/wp-content/uploads/2025/06/Ventura-Country-Club-2024-2025-Cyber-Policy.pdf

[24] Coalition - Broad Cyber Coverage Designed for Digital Risk: https://www.coalitioninc.com/coverages

[25] ATU Local 1596 Pension Plan - Coalition Cyber Policy (2023-2024): https://www.resourcecenters.com/pension%20funds/Select%20Your%20Fund/ATU%20Local%201596%20Pension%20Plan/22Meeting%20Archives/Archives/56Quarterly%20Meeting%20Agenda%20and%20Materials%20(May%2023,%202023)/Meeting%20Items/10May%2023,%202023%20Meeting/Item%208d-1.%20Coalition%2023-24%20Cyber%20Policy.pdf

[26] Coalition - Launches New Active Cyber Policy (April 9, 2025): https://www.coalitioninc.com/announcements/coalition-launches-new-active-cyber-policy

[27] Cowbell - Cyber Roundup 2026 Claims Report (PDF): https://cowbell.insure/wp-content/uploads/pdfs/CB-US-CyberRoundup-2026ClaimsReport.pdf

[28] Cowbell - Incident Response Panel (PDF): https://cowbell.insure/wp-content/uploads/pdfs/CB-US-Claims-IncidentResponsePanel.pdf

[29] Cowbell - Rx Solutions for Incident Management (Respond): https://cowbell.insure/rx-controls/rs-im

[30] Sample Cowbell Commercial Cyber Insurance Policy (Spinnaker Insurance Company): https://home.sayatalabs.com/cnc-wb/get_resource/carriers/SAMPLE_POLICY_FORM/COWBELL

[31] Cowbell 365 - PR Newswire: https://cowbell.insure/news-events/pr/cowbell-365

[32] CyberClan - Warranty Program Overview: https://cyberclan.com/warranty-program-overview

[33] Cowbell - Rx All Partners: https://cowbell.insure/rx-all

[34] Cowbell - Incident Response Panel (PDF) - Alternative: https://cowbell.insure/wp-content/uploads/pdfs/CB-US-Claims-IncidentResponsePanel.pdf

[35] Anderson Kill - "When Sublimits Bite: Perry & Perry Builders v. Cowbell Cyber" (2026): https://www.andersonkill.com/when-cyber-sublimits-work-cowbell-and-the-drafting-lessons-after-cici.html

[36] MoneyGeek - "Average Cyber Insurance Cost (2026 Report)": https://www.moneygeek.com/insurance/business/cyber/cost

[37] Security.org - "The Best Cyber Insurance of 2026": https://www.security.org/insurance/cyber/best

[38] Pro Insurance Group - "Cyber Liability Insurance Cost 2026": https://www.proinsgrp.com/business/cyber-liability-insurance/cost

[39] Mitchell-Joseph Insurance Agency - Cyber Insurance Data: https://www.mitchelljoseph.com/cyber-insurance

[40] The Agent's Office (North Texas) - Cyber Insurance Data: https://www.theagentsoffice.com/cyber-insurance-north-texas

[41] NAIC - 2025 Cybersecurity Insurance Market Report: https://content.naic.org/sites/default/files/inline-files/2025-cybersecurity-insurance-market-report.pdf

[42] Coalition - Do Small Businesses Need Cyber Insurance?: https://www.coalitioninc.com/blog/cyber-insurance/do-small-businesses-need-cyber-insurance

[43] Coalition - Cyber Policy Application (MassAgent): https://massagent.com/wp-content/uploads/2025/01/Cyber_Application_Agency.pdf

[44] Cowbell - Prime 100 Minimum Premium ($1,100): https://www.prnewswire.com/news-releases/cowbell-cyber-demystifies-cyber-insurance-with-cowbell-prime-100-300991101.html

[45] Cowbell - Factors Risk Rating System: https://cowbell.insure/cowbell-factors

[46] The Hartford - Business Insurance Average Cost: https://www.thehartford.com/business-insurance/average-cost

[47] Hiscox - Cyber Liability Insurance: https://www.hiscox.com/business-insurance/cyber-insurance

[48] AlphaCIS - "2026 Cyber Insurance Requirements": https://www.alphacis.com/2026-cyber-insurance-requirements

[49] IADC Law - "The Cyber Insurance Conundrum" by Elizabeth S. Fitch: https://www.iadclaw.org/assets/1/7/8.1-_Fitch-_IADC_Cyber_Conundrum_Article.pdf

[50] Chubb U.S. - Cyber Services Small Business Programs: https://www.chubb.com/us-en/business-insurance/products/cyber-insurance/cyber-services.html

[51] Coalition Blog - "Premium Credits for MDR Customers": https://www.coalitioninc.com/blog/cyber-insurance/premium-credits-mdr

[52] Coalition - MDR Premium Credit Overview (PDF): https://assets.ctfassets.net/o2pgk9gufvga/4NfPI7QssnPh5nRyuwx4gd/0ae1f0bd0f2a26966e69284f140dbd60/Coalition_MDR-Premium_Credit-Overview-US.pdf

[53] Cowbell - Connectors Overview (PDF): https://cowbell.insure/wp-content/uploads/2021/10/Cowbell-Connectors-Overview.pdf

[54] Cowbell - Microsoft Secure Score Connector - PR Newswire: https://www.prnewswire.com/news-releases/cowbell-cyber-introduces-microsoft-secure-score-connector-to-improve-policyholders-cyber-risk-profile-301445265.html

[55] Cowbell Report - Policyholders Experience Significant Risk Improvement - PR Newswire: https://www.prnewswire.com/news-releases/cowbell-report-finds-policyholders-experience-significant-risk-improvement-upon-renewal-301624709.html

[56] Coalition - Coverage Advantage Checklist (PDF): https://assets.ctfassets.net/o2pgk9gufvga/6GDAqNBNDON9cmkVdLCzd5/363be97c2125f8b89a7ca2a0515fd786/Coalition_Coverage-Advantage-Checklist-US.pdf

[57] ProWriters - "Small Business PCI Compliance": https://prowritersins.com/small-business-pci-compliance

[58] Gallagher - 2026 Cyber Insurance Market Outlook: https://www.gallagher.com/us/cyber-insurance-market-outlook-2026

[59] Sonora ISD - Cowbell Cyber Quote (PDF): https://meetings.boardbook.org/Documents/DownloadPDF/e2bb4bf7-190d-400a-abf1-b242477fbe0c?org=1303

[60] Chubb - "Commercial Cyber Policy Guide for Agents and Brokers" (PDF): https://www.chubb.com/content/dam/chubb-sites/chubb-com/us-en/cyber-risk-management/agent-center/pdfs/2020-06.23-14-01-1295-commercial-cyber-policy-guide-for-agents-brokers.pdf

[61] Oswald Companies - "Understanding Duty to Defend vs. Reimbursement": https://www.oswaldcompanies.com/media-center/understanding-the-difference-duty-to-defend-vs-reimbursement-insurance-policy-forms

[62] Cowbell - Social Engineering Coverage Announcement - PR Newswire: https://www.prnewswire.com/news-releases/cowbell-adds-social-engineering-coverage-to-its-cyber-insurance-program-301040689.html

[63] Cowbell - Prime 250 Announcement: https://www.prnewswire.com/news-releases/cowbell-extends-cyber-and-tech-eo-offerings-with-prime-250-for-upmarket-smes-301839386.html

[64] Cowbell - Expansion to 18 U.S. States (including Texas) - PR Newswire: https://www.prnewswire.com/news-releases/cowbell-cyber-expands-cyber-insurance-program-to-18-states-301124308.html