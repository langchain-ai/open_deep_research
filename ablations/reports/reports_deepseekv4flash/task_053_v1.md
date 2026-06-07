# Comprehensive IAM Platform Comparison: Okta Workforce Identity vs. Microsoft Entra ID Premium P2 vs. Ping Identity PingOne

**For a European Pharmaceutical Company (2,500 Employees, Germany/France/Poland)**

**Date: May 26, 2026**

---

## 1. Executive Summary

This report provides a detailed, dimension-by-dimension comparison of three leading enterprise Identity and Access Management (IAM) platforms—Okta Workforce Identity, Microsoft Entra ID Premium P2, and Ping Identity PingOne—for a European pharmaceutical company with approximately 2,500 employees operating across Germany, France, and Poland. The analysis covers MFA and phishing-resistant methods, Privileged Access Management (PAM) for SOX compliance, API rate limits for SAP/Salesforce integration, EU data residency options, licensing models, and 5-year Total Cost of Ownership (TCO) drivers. Each platform presents distinct strengths and trade-offs relevant to the pharmaceutical sector's strict regulatory and security requirements.

---

## 2. MFA Options & Phishing-Resistant Methods

### 2.1 Okta Workforce Identity

Okta supports a comprehensive range of MFA authenticators categorized by factor type (possession, knowledge, biometric) [1]. The platform's phishing-resistant methods are a key differentiator:

**Phishing-Resistant Methods:**
- **Okta FastPass**: Okta's flagship passwordless, phishing-resistant authentication. Uses device trust and biometric checks (Face ID, fingerprint). As of early 2025, approximately 91% of all daily Okta authentications use FastPass, saving an estimated 36,000 staff hours annually compared to traditional MFA [5].
- **Passkeys (FIDO2/WebAuthn)**: Supports both platform authenticators (Windows Hello, Touch ID) and roaming authenticators (YubiKey). Passkeys use public key cryptography and are phishing-proof [2][3]. Administrators can block synced passkeys to enforce managed-device-only policies [4].
- **Smart Cards (PIV/CAC)**: Certificate-based authentication using smart cards, meeting FedRAMP High and NIST AAL3 requirements [5].
- **Okta Verify**: When configured with phishing-resistant policies, works in conjunction with FastPass to provide phishing resistance [2].

**Other Supported Methods:**
- Okta Verify (push notifications, TOTP, biometrics)
- Google Authenticator (TOTP)
- YubiKey OTP
- Duo Security (third-party)
- Email OTP
- Phone (SMS/voice OTP) — Okta explicitly recommends moving away from these methods [3]
- Security Questions

**Important Caveat:** Some apps using WebView do not support Okta phishing-resistant authentication, potentially leading to access denial if policies require phishing resistance [2].

### 2.2 Microsoft Entra ID Premium P2

Microsoft Entra ID supports a wide range of authentication methods with strong phishing-resistant capabilities built into the platform natively [6][7].

**Phishing-Resistant Methods:**
- **Passkeys (FIDO2)**: Supports both device-bound passkeys (stored on one physical device) and synced passkeys (encrypted and stored in the cloud via Apple iCloud Keychain, Google Password Manager). 99% of users successfully register synced passkeys, and sign-in is 14x faster (3 seconds vs. 69 seconds) compared to password + traditional MFA [3].
- **Windows Hello for Business (WHfB)**: Phishing-resistant biometric/PIN-based authentication integrated with Windows devices [6].
- **Certificate-Based Authentication (CBA)**: Smartcard and X.509 certificate-based authentication, suitable for legacy systems [7].
- **Microsoft Authenticator**: Supports passwordless phone sign-in, number matching, and passkey storage in the app [2].
- **Temporary Access Pass (TAP)**: Short-lived codes enabling passwordless bootstrap for new or recovering users to register passwordless methods [4]. Microsoft reports an 80% reduction in password reset volume where Identity Pass (combining TAPs, Verified ID, and automated onboarding) is in use [4].

**Authentication Strength in Conditional Access:**
Microsoft provides three built-in authentication strengths [11]:
1. **Multifactor authentication strength** (least restrictive) — supports any MFA method
2. **Passwordless MFA strength** — methods without passwords
3. **Phishing-resistant MFA strength** (most restrictive) — FIDO2 security keys and Windows Hello for Business

Organizations can also create custom authentication strengths. Microsoft recommends applying phishing-resistant MFA to high-privilege roles (Global Administrator, Security Administrator, etc.) [13].

**Other Supported Methods:**
- OATH hardware/software tokens (TOTP)
- SMS sign-in and voice calls
- Email OTP
- Legacy methods (will be deprecated September 30, 2025) [2][3]

### 2.3 Ping Identity PingOne

PingOne provides MFA through the PingID mobile app, with phishing-resistant capabilities available through FIDO2/WebAuthn and, via PingFederate, certificate-based authentication [1][4].

**Phishing-Resistant Methods:**
- **FIDO2/WebAuthn (Passkeys)**: Supports passkeys on multiple platforms including Windows Hello and Apple Touch ID via platform authenticators, as well as hardware security keys. Passkeys can be synced across devices using iCloud Keychain or Google Password Manager [2]. Registration involves creating a key pair; the public key is stored in the user's profile, while the private key never leaves the client device.
- **YubiKey with FIDO2**: YubiKey 5 Series provides phishing-resistant security using hardware-based public key cryptography, effectively stopping Man-in-the-Middle and phishing attacks [3].
- **Certificate-Based Authentication (CBA)**: **Not native to PingOne MFA**. Requires PingFederate with the X.509 Certificate Integration Kit for smart card/PKI authentication [4][3]. This is a limitation for organizations wanting fully integrated CBA within the core MFA platform.

**Other Supported Methods (PingID Mobile App):**
- Push notifications with number matching
- Biometrics (fingerprint, facial recognition)
- One-time passcodes (OTP)
- Smart watch integration
- Offline/manual authentication
- VPN access

**PingID Desktop App:**
- OTP generation on Windows and Mac

**Third-Party TOTP:**
- Google Authenticator, Microsoft Authenticator

**Non-Phishing-Resistant Methods:**
- SMS and voice OTP
- Email OTP
- WhatsApp OTP (customer-only)

### 2.4 Comparative Summary — MFA & Phishing Resistance

| Feature | Okta Workforce Identity | Microsoft Entra ID P2 | Ping Identity PingOne |
|---|---|---|---|
| FIDO2/WebAuthn (Passkeys) | ✓ (platform + roaming) | ✓ (synced + device-bound + profiles coming Nov 2025) | ✓ (via PingID app & Advanced Identity Cloud) |
| Passwordless Phishing-Resistant | Okta FastPass (91% of authentications) | Windows Hello, Microsoft Authenticator, TAP | FIDO2 passkeys |
| Smart Card / CBA | ✓ (PIV/CAC natively) | ✓ (CBA natively) | ✗ (requires PingFederate separately) |
| Native CBA in MFA | ✓ | ✓ | ✗ |
| TOTP (Google Auth, etc.) | ✓ | ✓ | ✓ |
| SMS/Voice OTP | ✓ (deprecated recommendation) | ✓ | ✓ |
| Push with Number Matching | ✓ (Okta Verify) | ✓ (Microsoft Authenticator) | ✓ (PingID app) |

---

## 3. PAM Capabilities for SOX Compliance

### 3.1 Okta Workforce Identity — Okta Privileged Access

Okta Privileged Access is the company's current PAM offering, unifying privileged access management within the Okta Workforce Identity Cloud [10]. **Important context:** As of May 1, 2026, Okta will cease selling and renewing Advanced Server Access (ASA). Existing ASA customers must migrate to Okta Privileged Access within one year of their next scheduled renewal [11].

**Core PAM Features:**
- **Just-In-Time (JIT) Access**: Grants privileged access only when needed and for a limited time, significantly reducing the attack surface. Includes time-based access, approval workflows, and duration limits [12][13].
- **Credential Vaulting & Secrets Management**: Secrets vaulting with continuous local account discovery and scheduled password rotation. Manages non-federated SaaS service accounts [12].
- **Infrastructure Access Management**: Extends SSO to Linux and Windows servers, eliminating static SSH keys and passwords. Works with existing SSH and RDP tools [11][12].
- **Session Management & Recording**: SSH and RDP session recording with native integration with Okta System Log for audit trails [12].
- **Privileged Access Governance**: Multi-step approvals, justifications, duration limits, and integration with Okta Access Requests [12].
- **Break-Glass Accounts**: Emergency access mechanisms for shared accounts [12].

**SOX Compliance Mapping:**
- **SOX Section 302/404 (Internal Controls)**: Okta Identity Governance (OIG) provides automated access certifications and security reviews designed for SOX compliance. The platform generates audit-ready reports and SysLog events for SOX, GDPR, HIPAA, and PCI DSS [14].
- **Access Certification**: Security access reviews and campaigns to review user access and automate revocations [16].
- **Separation of Duties (SoD)**: SoD rules help reduce risks by defining conflicting entitlements, preventing users from having excessive access [16].
- **Audit Trail**: Session recording, System Log integration, and detailed audit logging provide "who, what, when, where" evidence [12][11].

### 3.2 Microsoft Entra ID Premium P2 — Privileged Identity Management (PIM)

Privileged Identity Management (PIM) is a service within Microsoft Entra ID that enables organizations to manage, control, and monitor access to important resources [9][11].

**Core PAM Features:**
- **Just-In-Time (JIT) Privileged Access**: Eliminate persistent access and enforce time-limited access for critical roles. Set start and end dates for role assignments [9][11].
- **Time-based and Approval-based Role Activation**: Users can be eligible (requiring activation with MFA, approval, and justification) or active (immediate use) [9].
- **PIM for Groups**: JIT membership and ownership activation of Microsoft Entra security and Microsoft 365 groups, regulating access to Azure roles, applications (Azure SQL, Key Vault, Intune), and third-party apps [14].
- **MFA Enforcement**: Require MFA to activate any role [9].
- **Usage Justification + Notifications**: Understand why users activate roles and get alerts [9].
- **Access Reviews**: Conduct periodic reviews to ensure users still need roles. Supports multistage approval workflows, separation of duties policies, and machine-learning recommendations [2][3].

**Entitlement Management:**
- Bundles resources (groups, apps, SharePoint sites) into access packages
- Automates workflows for access requests, assignments, reviews, and expirations [1]
- Enforces time-limited assignments and eligibility policies

**SOX Compliance Mapping:**
- **Segregation of Duties (SoD)**: Entitlement management supports SoD policies that can disable access requests for users assigned to incompatible groups and generate alerts for inappropriate access [3].
- **Access Certifications**: Access reviews enable administrators to manage access recertification with multistage reviews and machine-learning-derived decision recommendations [3].
- **Audit Trails**: PIM audit history is available for 30 days by default; longer retention requires Azure Monitor and storage costs [6]. Audit logs detail requestor, subject, action, domain, and target role [10].
- **Least Privilege**: PIM enforces least privilege by periodically reviewing, renewing, and extending access to resources. JIT access eliminates persistent access [7][11].

**Licensing Note:** All users who fall under the scope of governance features (including reviewers, approvers, and subjects of access reviews) require P2 or Entra ID Governance licenses [12][11].

### 3.3 Ping Identity PingOne — PingOne Privilege

Ping Identity launched PingOne Privilege on August 18, 2025, following its acquisition of Procyon [22][23]. This is a native JIT privileged access platform, though it takes a **vault-light approach** compared to traditional PAM.

**Core PAM Features:**
- **Just-In-Time (JIT) Privileged Access**: Users can securely request and obtain time-bound access to cloud environments (AWS, GCP, Azure), servers, databases, Kubernetes, and other critical resources. Adheres to zero trust best practices [23].
- **Zero Standing Privilege (ZSP)**: Eliminates static credentials and enforces an operating model granting ephemeral, task-specific access automatically revoked after session completion. Ping claims "eliminate static credentials for 95% of human access" [11].
- **Passwordless Access**: Authentication to all resources (SSH, RDP, IAM) is passwordless, eliminating the need for static credentials like SSH keys and RDP passwords [23].
- **Hardware-Bound Session Assurance (TPM)**: Each privileged session is cryptographically bound to both a verified identity and a trusted physical device using Trusted Platform Module (TPM) technology [11][21].
- **Session Recording and Audit Logs**: Session recordings and audit logs for privileged access support compliance with regulations including SOX, SOC2, GDPR, HIPAA, and PCI-DSS [23].
- **Break-Glass Access**: Vault integration only for narrow break-glass scenarios (95/5 model) [11].

**PAM Partnerships (for traditional credential vaulting):**
- **CyberArk Partnership**: Ping Identity and CyberArk have partnered to deliver SSO, MFA, and PAM [6].
- **BeyondTrust Partnership**: BeyondTrust Password Safe combines privileged account and session management (PASM) with secrets management. Integration with PingOne DaVinci allows real-time sharing of identity threat detections and automated session termination [7][8][9].

**SOX Compliance Mapping:**
- **Access Reviews**: Ping Identity's platform provides automated access reviews, detailed audit trails, and role-based access controls, supporting frameworks including SOX [16].
- **Session Recording & Audit Trails**: Continuous audit trails and session recordings aligned with SOX, CIS, HIPAA, ISO, SOC 2, and PCI DSS [11].
- **JIT/ZSP**: Ensures least privilege, a core SOX requirement for financial systems access.

**Limitation vs. Traditional PAM:** PingOne Privilege does not natively include traditional credential vaulting with password rotation. It relies on the "95/5 model" (95% JIT, 5% vault). For highly regulated environments requiring traditional vaulting, the CyberArk or BeyondTrust partnership is necessary.

### 3.4 Comparative Summary — PAM & SOX Compliance

| Capability | Okta Privileged Access | Microsoft Entra ID P2 (PIM) | PingOne Privilege |
|---|---|---|---|
| JIT Privileged Access | ✓ | ✓ (roles + groups) | ✓ (JIT + ZSP) |
| Credential Vaulting | ✓ (with rotation) | ✗ (not natively; uses Azure Key Vault for secrets) | ✗ (vault-light; partners with CyberArk/BeyondTrust) |
| Session Recording (SSH/RDP) | ✓ | ✗ (PIM doesn't record sessions; requires third-party) | ✓ (TPM-bound) |
| Access Reviews / Certifications | ✓ (via OIG) | ✓ (native in P2/Governance) | ✓ (via Identity Governance) |
| Separation of Duties (SoD) | ✓ (via OIG) | ✓ (via Entitlement Management) | ✓ (via Identity Governance) |
| Break-Glass Access | ✓ | ✓ | ✓ (vault integration) |
| MFA Enforcement for Privilege | ✓ | ✓ | ✓ |
| Audit Trail (Native) | ✓ (System Log) | ✓ (30-day default) | ✓ (audit logs + recordings) |
| Native Integration with IAM Platform | ✓ | ✓ | ✓ (through DaVinci) |

---

## 4. API Rate Limits for SAP/Salesforce Integration

### 4.1 Okta Workforce Identity

Okta uses a **bucket-based rate limiting system** where rate limiting buckets are collections of one or more endpoints sharing a defined quota per unit of time [18][19]. Buckets are scoped at organizational level, specific clients, authenticated users, or non-authenticated users.

**Key Rate Limits:**
- **Authentication endpoint (`/api/v1/authn`)**: 600 requests per minute (org-scoped). Warning at 60% (360 req/min), violation at 3,000 requests per minute [20].
- **Concurrency limit**: 75 simultaneous transactions (for both Office 365 and other traffic in Workforce orgs). Most requests process in milliseconds [21].
- **Per-User Limits**: Authenticate the same user: 4 requests per second. OAuth 2.0 token generate/refresh: 4 requests per second. These prevent brute force attacks [22].
- **Email Limits**: No more than 30 emails per recipient per minute [22].

**Rate Limit Scaling for Workforce Identity:**
Okta provides automatic rate limit multipliers based on license count [24]:

| Licenses Purchased | Rate Limit Multiplier |
|---|---|
| < 10,000 | 1x (base) |
| 10,000–100,000 | 5x |
| > 100,000 | 10x |

**For a company with ~2,500 users, the base 1x multiplier applies** [24].

**Specific Endpoint Limits (from community):**
- `/api/v1/apps/{id}/users/{id}` (App Users API): ~25 requests per second [25]
- `/api/v1/users/` endpoint: Higher rate limit [25]
- `/api/v1/logs` endpoint: Varies by tier [26]

**Throttling Behavior:** When quota is exceeded, requests are rejected with **HTTP 429 Too Many Requests** until the quota resets. Counters typically reset every 60 seconds [19]. Okta provides a Rate Limit Dashboard (Admin Console: Reports > Rate Limits) for monitoring [23]. Customers can request manual rate limit increases for expected high-traffic events [24].

### 4.2 Microsoft Entra ID Premium P2

Microsoft Graph concurrently imposes two categories of throttling: **Global limits** (apply to all services) and **Service-specific limits** (apply to individual services). The first limit to be reached triggers throttling [3].

**Global Limits:**
- Any request type: **130,000 requests per 10 seconds per app across all tenants** [3][6]
- **Tenant-level**: 8,000 Resource Units (RU); App + Tenant pair: 8,000 RU [10]

**Entra ID / Identity & Access Management Specific Limits:**
- **Entra ID Audit Logs API**: **5 requests per 10 seconds** for Directory, Sign-In, and Provisioning logs [1]
- Throttling is based on a token bucket algorithm where write operations (POST, PATCH, DELETE) cost more. Using `$select` decreases cost by 1; using `$expand` increases cost by 1 [12]
- **Token acquisition** via OAuth 2.0 endpoints is not counted toward Graph API request limits; these have separate Azure AD throttling policies [5]
- **Subscription (webhook) operations**: 500 requests per 20 seconds per app per tenant [5]

**Important Considerations:**
- Throttling limits are **not tied to Microsoft 365 license type** (E3, E5, Business Standard, etc.). They are enforced per API/service, per app, per user, and/or per tenant [13][8].
- Standard APIs are free within usage thresholds; High-Capacity and Advanced APIs may be metered and incur additional costs [13].
- **Throttling limits cannot be increased** and are not influenced by subscription plans. For large-scale data extraction, Microsoft recommends using **Microsoft Graph Data Connect** instead of increasing Graph API calls [5].
- Microsoft Graph returns HTTP 429 with a **Retry-After header** specifying wait time [5].
- **No differences reported between regions or tiers** for throttling limits; limits are globally consistent [8].

### 4.3 Ping Identity PingOne

PingOne's rate limiting is organized by **rate groups tied to purchased product entitlements**. Enforcement of rate entitlement values began at some point after September 2025 [7][16].

**Key Rate Limits:**
- **Per-IP Limit**: An IP address is limited to **35% of the overall license rate** by default. For Trial licenses, an IP can use 100% of the overall license rate [18].
- **Gateway Limits**: **500 requests per second per instance** [7].
- **Community Estimate**: Approximately **100–150 requests per second per IP** [6].
- **HTTP API Request Header Size**: Maximum of **6 KB** [7].

**Rate Limiting Architecture:**
- Rate limits are defined per License and shared by all environments assigned to that license [16]
- Base rate limits can be increased through the **Maximum Throughput Assurance** program [16]
- Requests exceeding limits are rejected with **HTTP 429 "REQUEST_LIMITED"** error [16]
- Server-Sourced Traffic IP allow-listing can exempt corporate servers from per-IP limits [18]

**Integration Considerations for SAP/Salesforce:**
- Calls from SAP/Salesforce to PingOne: Subject to PingOne's per-license rate limits (default per-IP limited to 35% of license rate). Use the "Server-Sourced Traffic" feature to white-list corporate server IPs [18].
- Calls from PingOne to SAP/Salesforce: Subject to **SAP's 50 concurrent requests per second** limit (SAP Cloud Identity Services) [8] or **Salesforce's 100,000 daily API request limit** (Enterprise Edition, plus 1,000 additional requests per user license) [9].

**Monitoring:** PingOne provides an API Usage Dashboard for monitoring peak usage [18][7].

### 4.4 Comparative Summary — API Rate Limits

| Metric | Okta Workforce Identity | Microsoft Entra ID P2 | Ping Identity PingOne |
|---|---|---|---|
| Authn Endpoint | 600 req/min (1x for <10k users) | N/A (Graph API) | Per-license rate groups |
| Concurrency | 75 simultaneous transactions | 130k req/10 sec (global); 8k RU/tenant | 500 req/sec per gateway instance |
| Per-User Limit | 4 req/sec (authn & token) | N/A | ~100-150 req/sec per IP |
| Audit Logs API | Varies by tier | 5 req/10 sec | Limited by license rate group |
| Throttling Response | HTTP 429 | HTTP 429 + Retry-After header | HTTP 429 "REQUEST_LIMITED" |
| Limit Increases | Request-based (for events) | Cannot be increased | Maximum Throughput Assurance program |
| Regional Differences | None documented | None documented | None documented |
| Integration (SAP/SF) | Subject to SAP/SF limits; Okta rate limits apply | Subject to SAP/SF limits; Graph limits apply | Subject to SAP/SF limits; PingOne license limits apply |

---

## 5. EU Data Residency Options

### 5.1 Okta Workforce Identity

Okta uses a **cell architecture** allowing customers to select local data cells for storing customer data [28][29]. Data residency refers to the actual geographic location where an organization's content is stored.

**EMEA Cell Locations:**
- **Primary data center in Germany**: Okta operates a data centre in Germany, enabling compliance with German data localization requirements [30][31].
- **Disaster recovery site in Ireland**: Uses a data centre in Germany with a disaster recovery site in Ireland [30].

**Specific Country Coverage:**
- **Germany**: Primary data center in Germany [30][31].
- **France**: Served from the Germany-based primary EMEA location. No dedicated France-specific data center confirmed in available documentation.
- **Poland**: Served from the Germany primary + Ireland DR infrastructure. No dedicated Poland data center.

**Guaranteeing Data Residency:**
Customers should contact Okta Sales to request an EMEA cell org creation: "We do offer having your Okta org created in our EMEA cells if this is a requirement of yours, but our Sales team should be able to provide you more information about that" [30].

**GDPR Compliance:**
Okta states: "At Okta, we are committed to our customers' success and assist with their GDPR compliance through comprehensive privacy and security protections" [32]. Key features include:
- EU-dedicated architecture with data centers in Germany/Ireland
- Data Processing Addendum (DPA)
- Encryption for data at rest and in transit
- ISO 27001/27017/27018 and GDPR certifications
- Okta does not monetize or sell customer data [28][29][32]

### 5.2 Microsoft Entra ID Premium P2

**Microsoft Azure EU Data Centers:**
Microsoft operates multiple Azure regions in Europe [1][2][3][5]:
- **Germany West Central (Frankfurt)**: Three availability zones. Customer data stored at rest in Germany. Opened 2019 [1][5].
- **Germany North (Berlin)**: Available as a region [1].
- **France Central (Paris)**: Azure region with availability zones [2].
- **France South (Marseille)**: Additional Azure region [2].
- **Poland Central (Warsaw)**: Launched in April 2023 — Microsoft's first cloud datacenter region in Central and Eastern Europe, with three Azure availability zones [3].

**EU Data Boundary Initiative:**
Microsoft completed the EU Data Boundary in three phases [7][8][9]:
- **Phase 1 (January 2023)**: Customer data from core cloud services stored and processed within EU/EFTA.
- **Phase 2 (January 2024)**: Pseudonymized personal data remains within EU/EFTA boundaries.
- **Phase 3 (February 2025)**: Professional Services Data (technical support data) stored and processed within EU/EFTA.

**Critical Caveat for Entra ID:**
Microsoft Entra ID replicates each tenant across datacenters based on criteria including proximity. **The geo-location mapping is fixed and permanent at tenant creation and cannot be changed later** [7][9].

For European customers, most Entra ID customer data is stored in Europe. However:
- **The Core Store for EMEA data is backed by two Azure regions: Amsterdam (Netherlands) and Dublin (Ireland)** [10] — **not necessarily in a specific country** like Germany, France, or Poland
- Some tenants created before 2013 or 2017 may have data stored in US or Asia/Pacific regions based on historical geo-mappings [6]
- Some service components have temporary data transfers outside the EU as work is in progress [6][8]

**Impact for Pharmaceutical Company:**
Organizations with strict sector-specific data residency rules (e.g., pharmaceutical regulated data) should be aware that Entra ID data location is immutable and may not match workload deployment regions. This is a significant consideration for GDPR data localization requirements.

### 5.3 Ping Identity PingOne

**PingOne Advanced Identity Cloud Data Regions (Europe) [11][12]:**

| Region | Location | Services Available |
|---|---|---|
| Frankfurt (europe-west3) | Germany | Advanced Identity Cloud + IGA + Autonomous Identity |
| Paris | France | Advanced Identity Cloud + IGA + Autonomous Identity |
| Belgium | Europe | Advanced Identity Cloud + IGA + Autonomous Identity |
| Netherlands | Europe | Advanced Identity Cloud + IGA + Autonomous Identity |
| London (europe-west2) | UK | Advanced Identity Cloud + IGA + Autonomous Identity |
| Finland | Europe | Advanced Identity Cloud only |
| Zurich | Switzerland | Advanced Identity Cloud only |

**Data Residency Guarantee:**
- **Germany (Frankfurt)**: Confirmed data region [12].
- **France (Paris)**: Confirmed data region [12].
- **Poland**: **No** data center in Poland. Closest would be Frankfurt (Germany) or Netherlands.

**Important CLOUD Act Consideration:**
Ping Identity is a US-based company (Delaware corporation, owned by US private equity firm Thoma Bravo, merged with ForgeRock) [14]. The CLOUD Act allows US authorities to access data stored abroad by US companies without EU oversight. This poses a critical consideration for EU organizations using PingOne cloud services, even with EU data residency options, because:
- Ping Identity's US jurisdiction means contractual measures may not fully mitigate compliance risk
- PingOne is built on AWS or Google Cloud (depending on deployment type)
- For strict EU data sovereignty, on-premises PingFederate deployments reduce CLOUD Act risks but increase operational burdens [14]

**PingOne Cloud Platform vs. Advanced Identity Cloud:**
- **PingOne Cloud Platform**: Multi-tenant SaaS, self-service admin, 99.99% uptime SLA, single-region deployment [16]
- **PingOne Advanced Identity Cloud**: Built on Google Cloud, single-tenant VPC on AWS (Advanced Services option), deeper customization, active-active multi-region [18][20]

### 5.4 Comparative Summary — EU Data Residency

| Factor | Okta Workforce Identity | Microsoft Entra ID P2 | Ping Identity PingOne |
|---|---|---|---|
| Germany Data Center | ✓ (primary) | ✓ (Frankfurt, Berlin) | ✓ (Frankfurt) |
| France Data Center | ✗ (served from Germany) | ✓ (Paris, Marseille) | ✓ (Paris) |
| Poland Data Center | ✗ (served from Germany) | ✓ (Warsaw, launched April 2023) | ✗ (closest: Frankfurt) |
| Data Guaranteed in EU | ✓ (EMEA cell option) | ✓ (EU Data Boundary) but Core Store = Amsterdam/Dublin | ✓ (selectable region) |
| Data Location Immutable After Creation | ✗ (can request EMEA cell) | ✓ (fixed at tenant creation) | ✗ (region selectable) |
| US Company / CLOUD Act Risk | US-based | US-based | US-based (Thoma Bravo) |
| Certifications | ISO 27001, GDPR, FedRAMP High | ISO, EU Data Boundary, GDPR | ISO, SOC 2, GDPR, PCI DSS |

---

## 6. Licensing Models

### 6.1 Okta Workforce Identity

**Pricing Tiers (Published) [33][34][35][36]:**

| Tier | Published Price | Key Features |
|---|---|---|
| Starter Suite | **$6/user/month** | Single Sign-On, Multi-Factor Authentication, Universal Directory, 5 Workflows |
| Essentials Suite | **$17/user/month** | Adaptive MFA, Privileged Access, Lifecycle Management, Access Governance, 50 Workflows |
| Professional | **Custom pricing** (contact sales) | Device Access, Identity Threat Protection |
| Enterprise | **Custom pricing** (contact sales) | API Access Management, Access Gateway, full suite |

**Additional Pricing Details:**
- Minimum annual contract of **$1,500** [34][36]
- Annual billing required; monthly billing not available for enterprise plans
- Volume discounts available, typically at 500, 1,000, 5,000 user thresholds [35]
- Multi-year contracts (2-3 year terms) offer better discounts [35]
- Discounts negotiable — "buyers who negotiate all components together often achieve 10–20% better overall pricing" [35]

**Add-On Module Pricing:**
- Okta Privileged Access: Resource-based pricing (Resource Units)
- Okta Identity Governance: Starting at $9/user/month [37]
- Okta Workflows: Tiered by usage (5 included in Starter, 50 in Essentials)
- API Access Management: $2/user/month [37]
- Device Access: Enterprise tier feature
- Identity Threat Protection: Enterprise tier

**Guidance:** Professional and Enterprise tiers require contacting Okta Sales. Timing negotiations around Okta's fiscal year-end (January 31) or quarter-ends may yield better discounts [35].

### 6.2 Microsoft Entra ID Premium P2

**Standalone Pricing [5][8][12][13]:**
- **Microsoft Entra ID P2**: **$9.00/user/month** (with annual payment commitment)
- **Microsoft Entra ID P1**: **$6.00/user/month**
- **Microsoft Entra Suite**: **$12.00/user/month** (bundles ID Protection, full ID Governance, Internet Access, Private Access, and Verified ID premium)

**Bundled with Microsoft 365:**
- **Microsoft 365 E5** (~$54.75/user/month without Teams): Includes Entra ID P2, Microsoft Defender suite, Microsoft Purview, Power BI Pro, enterprise voice [14]
- **Microsoft 365 E3**: Includes Entra ID P1 (not P2) [11]
- **Enterprise Mobility + Security (EMS) E5**: Includes Entra ID P2 [14]

**Minimum License Requirements:**
No specific minimum license requirements for standalone Entra ID P2. However, for full governance features (access reviews, entitlement management), **all users who fall under the scope** — including subjects of reviews, reviewers, and approvers — require P2 or Governance licenses [11][13].

**Billing Options:**
- Annual commitment required for $9/user/month pricing [5]
- Starting April 1, 2025, Microsoft applies a 5% uplift to monthly-billed annual subscriptions [14]
- Volume-based pricing tiers (Levels B-D) under Enterprise Agreements were retired November 1, 2025, moving all customers to standard Level A pricing [4][6]

**Synergy Considerations:**
- If already on M365 E3: Adding Entra ID P2 standalone at $9/user/month (upgrading from P1 included in E3) [11]
- If already on M365 E5: Entra ID P2 included at no additional cost [11]
- Entra Suite ($12/user/month) offers ~29% savings over buying components individually [11]
- Buying M365 E5 purely for Entra ID P2 may not be cost-effective; M365 E3 + standalone P2 is ~24% cheaper than E5 [14]

### 6.3 Ping Identity PingOne

**Workforce Plans (for internal employee base) [13][17][20]:**

| Tier | Price | Key Features |
|---|---|---|
| Essential (Workforce) | **$3/user/month** | SSO and basic orchestration without adaptive MFA |
| Plus (Workforce) | **$6/user/month** | Adaptive MFA and Microsoft ecosystem integration |

**Critical Minimum Commitment:**
Ping Identity requires a **5,000-user minimum with annual contracts** across all Workforce tiers [13]. For a ~2,500 employee company:
- Essential tier: Minimum commitment = 5,000 × $3 × 12 = **$180,000/year**
- Plus tier: Minimum commitment = 5,000 × $6 × 12 = **$360,000/year**
- You pay for 5,000 users regardless of actual headcount [13]

**Customer Identity Plans:**
- Essential (Customer): $20,000–$35,000/year
- Plus (Customer): $50,000/year

**Volume Discounts:**
- 15–50% common for large multi-year commitments [13]
- Multi-year commitments (2–3 years) and bundling can yield 15–40% discounts [20]

**Hidden Costs and Add-Ons:**
- Professional services: 20–50% of first-year costs [13][20]
- Premium support: adds 5–15% uplift [20]
- Add-on modules: fraud detection, privileged access (PingOne Privilege), API intelligence
- PingFederate (if needed): starts around $50,000–$75,000 annually [20]
- SMS/MFA delivery costs via third-party providers
- Overage fees billed at 2× the annual per-MAU rate [13]

**DaVinci Orchestration Costs:**
Pricing is not publicly listed and typically bundled with PingOne subscriptions or sold as an add-on module. Specific pricing requires engaging Ping Identity sales.

**Guidance:** Official quotes must be obtained through Ping Identity sales, partner/reseller network, or certified systems integrators. The average annual contract value in 2026 is ~$50,690 based on 30 purchases [20].

### 6.4 Comparative Summary — Licensing

| Factor | Okta Workforce Identity | Microsoft Entra ID P2 | Ping Identity PingOne |
|---|---|---|---|
| Published Price (Mid-Tier) | $17/user/month (Essentials) | $9/user/month (P2 standalone) | $6/user/month (Plus) *see minimum |
| Minimum Commitment | $1,500 annual | None | **5,000 users mandatory** |
| Annual Cost (2,500 users) | $510,000 (Essentials) | $270,000 (P2 standalone) | $360,000 (Plus, 5k min) |
| Bundling Savings | N/A | M365 E5 includes P2; Entra Suite at $12/user | Volume discounts 15-50% |
| Add-On Costs | Governance (+$9/user/mo), PAM (resource-based) | Governance included in P2 with limitations | PAM (PingOne Privilege, quote-based), DaVinci (quote-based) |
| Annual Escalations | 5-10% typical | 5% uplift for monthly billing | 3-7% typical |

---

## 7. 5-Year Total Cost of Ownership (TCO) Drivers

### 7.1 Okta Workforce Identity — 5-Year TCO Framework

**Licensing Costs (Base):**
For ~2,500 users at Essentials Suite ($17/user/month published), negotiate to approximately $10/user/month [34][35]:
- Year 1: 2,500 × $10 × 12 = **$300,000**
- Years 2-5 (escalating 5-10% annually): **$315,000–$330,000/year**

**Implementation & Professional Services:**
- "Implementation services typically run 2.5x your annual license cost in year one" [34]
- AD migration, systems integration, configuration, training: **~$250,000 one-time**

**Support Costs:**
- Premier Support: 15–25% of license value [35]: **$45,000–$75,000/year**

**Add-On Modules:**
- PAM (Okta Privileged Access), Governance (OIG), API Management: **~$100,000/year**

**Hidden Costs:**
- **SSO Tax**: SaaS vendors charge a premium (15% to over 100% more per user) when connecting a third-party SSO provider [39]. For a 100-person company using 80 SaaS tools, the true annual cost was $220,400 vs. $20,400 sticker price.
- Internal IT operations: **~$50,000/year**
- Training & change management: **~$30,000 year 1, $10,000 thereafter**

**5-Year TCO Estimate [34][35][41]:**

| Cost Category | Year 1 | Years 2-5 (Annual) | 5-Year Total |
|---|---|---|---|
| License (Essentials, ~$10/user negotiated) | $300,000 | $300,000 | $1,500,000 |
| Implementation Services (2.5x license) | $250,000 | $0 | $250,000 |
| Premier Support (20% of license) | $60,000 | $60,000 | $300,000 |
| Add-ons (PAM, Governance, API Mgmt) | $100,000 | $100,000 | $500,000 |
| SSO Tax / SaaS Premiums | Variable | Variable | $200k–$1M+ |
| Internal IT Operations | $50,000 | $50,000 | $250,000 |
| Training & Change Management | $30,000 | $10,000 | $70,000 |
| **Total** | **~$790,000** | **~$520,000+** | **~$3.07M+** |

### 7.2 Microsoft Entra ID Premium P2 — 5-Year TCO Framework

**Licensing Costs (Base):**
Standalone P2 at $9/user/month for 2,500 users:
- Year 1: 2,500 × $9 × 12 = **$270,000**
- Years 2-5 (with ~5% annual escalation): **$283,500–$328,000/year**

**Implementation & Professional Services:**
- Consultant costs: $200–$500/hour for specialized Microsoft identity consultants
- Full Entra ID deployment (PIM, Conditional Access, authentication method migration, integration): **$100,000–$250,000** [14]
- AD Connect/Cloud Sync: Tool is free but requires on-premises server infrastructure
- Training: **$20,000–$50,000**

**Ongoing Operational Costs:**
- PIM administration, access review management, Conditional Access maintenance: **$50,000–$100,000/year**
- Access reviews (quarterly): **$20,000–$40,000/year**
- Azure Monitor for audit log retention (31+ days): Varies based on volume

**Infrastructure:**
- Azure AD Connect servers (if not already deployed): **$5,000–$15,000 one-time**

**Governance Licensing Nuance:**
All users who are subjects of access reviews require P2 or Governance licenses. This includes all 2,500 employees if they are evaluated in access reviews [11]. "If you have 400+ users, the total cost would be $4,000 per month" [7].

**5-Year TCO Estimate (Standalone P2):**

| Cost Category | Year 1 | Years 2-5 (Annual) | 5-Year Total |
|---|---|---|---|
| License (P2, $9/user/month) | $270,000 | $270,000 | $1,350,000 |
| Implementation/Consulting | $150,000 | $0 | $150,000 |
| Internal IT Operations | $75,000 | $75,000 | $375,000 |
| Infrastructure (AD Connect) | $15,000 | $2,500 | $25,000 |
| Azure Monitor (audit logs) | $5,000 | $5,000 | $25,000 |
| Training | $35,000 | $10,000 | $75,000 |
| **Total** | **~$550,000** | **~$362,500** | **~$2.0M** |

**If Already on M365 E5:** P2 is included at no additional cost. TCO reduces to implementation and operational costs only (~$100,000–$400,000 over 5 years).

### 7.3 Ping Identity PingOne — 5-Year TCO Framework

**Licensing Costs (Base):**
Workforce Plus at $6/user/month, but with **5,000-user minimum** for 2,500 actual employees:
- Year 1: 5,000 × $6 × 12 = **$360,000**
- Years 2-5 (escalating 3-7% annually): **$370,800–$450,000/year**

**Implementation & Professional Services:**
- "Professional services typically represent 20–50% of first-year costs for complex deployments" [13][20]
- Fees range from **$20,000 to $200,000+** depending on scope
- Example: "First-year expenses reaching up to $1.5 million for a 10,000-user Workforce Plus deployment" [13]
- AD migration, DaVinci flow creation, SAP/Salesforce integration

**PingFederate (if needed for CBA/legacy integration):**
- Self-hosted: Starts around **$50,000–$75,000 annually** [20]

**Premium Support:**
- 5–15% uplift: **$18,000–$54,000/year**

**Add-On Modules:**
- PingOne Privilege (PAM): Quote-based
- DaVinci orchestration: Quote-based
- API intelligence: Quote-based
- Fraud detection: Quote-based

**Hidden Costs:**
- SMS/MFA delivery via third-party (Twilio, etc.)
- Overage fees at 2× annual per-MAU rate [13]
- DaVinci flow maintenance and connector development

**5-Year TCO Estimate:**

| Cost Category | Year 1 | Years 2-5 (Annual) | 5-Year Total |
|---|---|---|---|
| License (Plus, 5k min) | $360,000 | $360,000 | $1,800,000 |
| Implementation (35% of first year) | $126,000 | $0 | $126,000 |
| PingFederate (if needed) | $75,000 | $75,000 | $375,000 |
| Premium Support (10%) | $36,000 | $36,000 | $180,000 |
| Add-ons (PAM, DaVinci, etc.) | $80,000 | $80,000 | $400,000 |
| Internal IT Operations | $60,000 | $60,000 | $300,000 |
| Training | $30,000 | $10,000 | $70,000 |
| **Total** | **~$767,000** | **~$621,000** | **~$3.25M** |

### 7.4 Comparative 5-Year TCO Summary

| Platform | Estimated 5-Year TCO | Key Cost Drivers |
|---|---|---|
| **Okta Workforce Identity** (Essentials) | **~$3.07M+** | Higher per-user pricing ($17/user/mo pub.), implementation 2.5x license, SSO Tax, annual escalations 5-10% |
| **Microsoft Entra ID P2** (Standalone) | **~$2.0M** | Lower per-user pricing ($9/user/mo), implementation costs moderate, included in M365 E5 if already owned, governance licensing can surprise |
| **Ping Identity PingOne** (Workforce Plus) | **~$3.25M** | 5,000-user minimum forces higher base cost ($360k/yr for 2,500 employees), additional PingFederate costs if CBA needed, US CLOUD Act risk may require on-prem investment |

**Key TCO Observations:**
1. **Microsoft Entra ID P2** has the lowest base licensing cost for 2,500 users ($270,000/year vs. Okta's $510,000 at list or PingOne's $360,000 with 5k minimum)
2. **PingOne's 5,000-user minimum** is a significant disadvantage for a 2,500-employee company, effectively doubling the licensing cost
3. **Okta's SSO Tax** — hidden cost of SaaS vendors charging more when Okta is the SSO provider — can add $200,000–$1,000,000+ over 5 years
4. **Microsoft synergy** — if already on M365 E5, P2 is included, making it the most cost-effective option
5. **Professional services** are significant for all platforms (20-50% of first-year costs for PingOne and Microsoft; ~2.5x license for Okta)

---

## 8. Overall Comparative Summary

| Dimension | Okta Workforce Identity | Microsoft Entra ID P2 | Ping Identity PingOne |
|---|---|---|---|
| **MFA & Phishing Resistance** | Strong (FastPass dominates, FIDO2, Smart Cards) | Strong (FIDO2 synced/device-bound, WHfB, CBA, TAP) | Good (FIDO2, YubiKey; CBA requires PingFederate) |
| **PAM for SOX** | Comprehensive (JIT, vaulting, session recording, OIG) | Strong (PIM, Entitlement Mgmt, Access Reviews; no native session recording) | Modern JIT/ZSP approach; vault-light; partnerships for traditional vaulting |
| **API Rate Limits** | 600 req/min authn; 75 concurrent; 4 req/sec per user; 1x multiplier for <10k users | 130k req/10 sec global; 5 req/10 sec audit logs; **cannot be increased** | Per-license rate groups; 500 req/sec gateway; ~100-150 req/sec per IP |
| **EU Data Residency** | Germany (primary) + Ireland (DR); EMEA cell available on request | EU Data Boundary; Poland data center (Warsaw); BUT Entra ID Core Store = Amsterdam/Dublin, immutable | Frankfurt (Germany), Paris (France) confirmed; CLOUD Act risk (US parent company) |
| **Licensing** | $6-17/user/mo; min $1,500; volume discounts | $9/user/mo (P2 standalone); included in M365 E5; no minimum | $3-6/user/mo but **5,000-user minimum** required |
| **5-Year TCO (est.)** | ~$3.07M+ | ~$2.0M (standalone); ~$100-400k (if on M365 E5) | ~$3.25M (5k minimum drives cost) |

---

## 9. Recommendations and Key Considerations

**For a European Pharmaceutical Company (2,500 employees, Germany/France/Poland):**

### Primary Recommendation: Microsoft Entra ID Premium P2
- **Lowest 5-year TCO** at ~$2.0M standalone, or significantly less if already on M365 E5
- **Strongest EU data residency options** with a confirmed data center in Poland (Warsaw) and France (Paris), plus the completed EU Data Boundary initiative
- **No minimum user commitment** — pay for exactly 2,500 users
- **Robust phishing-resistant MFA** with FIDO2 passkeys, Windows Hello, and CBA natively
- **Comprehensive PIM** for SOX compliance (JIT, access reviews, SoD, audit trails)

**Critical Caveat for Microsoft:** Entra ID tenant data location is fixed at creation and the Core Store is in Amsterdam/Dublin, not necessarily in a specific country. Organizations with strict data localization requirements should validate this with Microsoft before proceeding.

### Strong Alternative: Okta Workforce Identity
- **Best phishing-resistant experience** with Okta FastPass (91% of authentications)
- **Comprehensive PAM** with native credential vaulting and session recording
- **More flexible data residency** — can request EMEA cell creation (Germany + Ireland)
- Higher cost (~$3.07M+ over 5 years) and SSO Tax risks

### Consider with Caution: Ping Identity PingOne
- **5,000-user minimum** is a significant disadvantage for 2,500 employees
- **CLOUD Act risk** for a regulated EU pharmaceutical company — US parent company (Thoma Bravo) means even EU-hosted data may be subject to US access
- If CBA is required, PingFederate must be licensed separately, increasing cost and complexity
- Modern JIT approach may be suitable for cloud-native environments but less mature for traditional pharmaceutical IT landscapes

### Action Items:
1. **Engage Microsoft** to confirm Entra ID data residency guarantees for pharmaceutical regulated data in Germany, France, and Poland
2. **Request Okta quote** for an EMEA cell org and compare TCO with negotiated volume discounts
3. **Validate PingOne's CLOUD Act exposure** with legal counsel specializing in EU data protection
4. **Assess current Microsoft licensing** — if already on M365 E5, Entra ID P2 is included and represents the lowest TCO path
5. **Budget for implementation services** (20-50% of first-year licensing for all platforms) and the SSO Tax (SaaS vendor premiums)

---

## Sources

[1] Okta Identity Engine - Multifactor Authentication: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/about-authenticators.htm

[2] Okta Help - Phishing-resistant authentication: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/phishing-resistant-auth.htm

[3] Okta Support - MFA Authenticator Options: https://support.okta.com/help/s/article/mfa-authenticator-options

[4] Okta Help - FIDO2 (WebAuthn): https://help.okta.com/en-us/content/topics/security/mfa-webauthn.htm

[5] Okta Product - Okta FastPass: https://www.okta.com/products/fastpass

[6] Okta Newsroom - Secure Sign-in Trends Report 2025: https://www.okta.com/newsroom/articles/secure-sign-in-trends-report-2025

[7] Okta Blog - Phishing-Resistant MFA Shows Great Momentum: https://www.okta.com/blog/identity-security/phishing-resistant-mfa-shows-great-momentum

[8] Okta Help - Google Authenticator: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/google-authenticator.htm

[9] Okta Help - Configure the phone authenticator: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/configure-phone.htm

[10] Okta Help - Okta Privileged Access Overview: https://help.okta.com/oie/en-us/content/topics/privileged-access/pam-overview.htm

[11] Okta Help - Get Started with Advanced Server Access: https://help.okta.com/asa/en-us/content/topics/adv_server_access/docs/start-here.htm

[12] Okta Product - Okta Privileged Access: https://www.okta.com/products/privileged-access

[13] Okta Identity 101 - Privileged Access Management Solutions: https://www.okta.com/identity-101/privileged-access-management-solutions

[14] YouTube - Okta Identity Governance (OIG) Explained 2026: https://www.youtube.com/watch?v=et7QJEeAkpM

[15] Okta Help - Identity Governance: https://help.okta.com/oie/en-us/content/topics/identity-governance/iga.htm

[16] Okta Developer - Rate limits overview: https://developer.okta.com/docs/reference/rate-limits

[17] Okta Developer - Burst rate limits: https://developer.okta.com/docs/reference/rl2-burst

[18] Okta Developer - Concurrency limits: https://developer.okta.com/docs/reference/rl2-concurrency

[19] Okta Developer - Additional Rate limits: https://developer.okta.com/docs/reference/rl2-limits

[20] Okta Developer - Increase your rate limits: https://developer.okta.com/docs/reference/rl2-increase

[21] Okta - Data Residency: https://www.okta.com/okta-data-residency

[22] Okta - Starting Your GDPR Journey: https://www.okta.com/resources/whitepaper/starting-your-general-data-protection-regulation-journey-with-okta

[23] Okta Support - Okta datacenter location: https://support.okta.com/help/s/question/0D51Y00009BJSsqSAH/okta-datacenter-location

[24] Digitalisation World - Okta opens EU data centre: https://c.digitalisationworld.com/news/39369/okta-opens-eu-data-centre-nbsp

[25] Okta - GDPR: https://www.okta.com/nl/gdpr

[26] Okta - Plans and Pricing: https://www.okta.com/pricing

[27] CheckThat.ai - Okta Pricing 2026: https://checkthat.ai/brands/okta/pricing

[28] Vendr - Okta Software Pricing & Plans 2026: https://www.vendr.com/marketplace/okta

[29] UnderDefense - Okta Pricing 2026 Ultimate Guide: https://underdefense.com/industry-pricings/okta-pricing-ultimate-guide-for-security-products

[30] AccessOwl Blog - Okta Pricing April 2026: True Cost with SSO Tax: https://www.accessowl.com/blog/okta-cost

[31] MetaCTO - The True Cost of Okta Pricing and Integration: https://www.metacto.com/blogs/the-true-cost-of-okta-a-comprehensive-guide-to-pricing-and-integration

[32] SuperTokens - Okta Pricing Complete Guide 2024: https://supertokens.com/blog/okta-pricing-the-complete-guide

[33] Microsoft Entra Plans and Pricing | Microsoft Security: https://www.microsoft.com/en-us/security/business/microsoft-entra-pricing

[34] Microsoft Entra authentication overview: https://learn.microsoft.com/en-us/entra/identity/authentication/overview-authentication

[35] Authentication methods in Microsoft Entra ID - passkeys (FIDO2): https://learn.microsoft.com/en-us/entra/identity/authentication/concept-authentication-passkeys-fido2

[36] What Are the Phishing Resistant Options in Entra ID and Microsoft 365? | Keytos Blog: https://www.keytos.io/blog/passwordless/phishing-resistant-options-in-entra-id-and-microsoft-365

[37] How to enable passkeys (FIDO2) in Microsoft Entra ID: https://learn.microsoft.com/en-us/entra/identity/authentication/how-to-authentication-passkeys-fido2

[38] Privileged Identity Management (PIM) | Microsoft Security: https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-privileged-identity-management-pim

[39] What is Privileged Identity Management? - Microsoft Entra ID Governance | Microsoft Learn: https://learn.microsoft.com/en-us/entra/id-governance/privileged-identity-management/pim-configure

[40] Privileged Identity Management (PIM) for Groups - Microsoft Entra ID Governance | Microsoft Learn: https://learn.microsoft.com/en-us/entra/id-governance/privileged-identity-management/concept-pim-for-groups

[41] What is entitlement management? - Microsoft Entra ID Governance: https://learn.microsoft.com/en-us/entra/id-governance/entitlement-management-overview

[42] What are access reviews? - Microsoft Entra ID Governance: https://learn.microsoft.com/en-us/entra/id-governance/access-reviews-overview

[43] Microsoft Graph service-specific throttling limits: https://learn.microsoft.com/en-us/graph/throttling-limits

[44] Entra ID Audit logs API - Rate Limits - Microsoft Q&A: https://learn.microsoft.com/en-us/answers/questions/1688849/entra-id-audit-logs-api-rate-limits

[45] Microsoft EU Data Boundary FAQs: https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-product-and-services/security/pdf/eu-data-boundary-for-the-microsoft-cloud-frequently-asked-questions-updated-february-2025.pdf

[46] What is the EU Data Boundary? - Microsoft Privacy: https://learn.microsoft.com/en-us/privacy/eudb/eu-data-boundary-learn

[47] Customer data storage and processing for European customers in Microsoft Entra ID: https://learn.microsoft.com/en-us/entra/fundamentals/data-storage-eu

[48] Microsoft Entra ID and data residency: https://learn.microsoft.com/en-us/entra/fundamentals/data-residency

[49] Microsoft launches its first datacenter region in Poland: https://news.microsoft.com/europe/2023/04/26/microsoft-launches-its-first-datacenter-region-in-poland-bringing-new-opportunities-to-develop-the-digital-economy

[50] Microsoft Entra licensing - Microsoft Entra | Microsoft Learn: https://learn.microsoft.com/en-us/entra/fundamentals/licensing

[51] Ping Identity - PingOne Authentication Methods Overview: https://docs.pingidentity.com/pingone/strong_authentication_mfa/p1_authentication_methods_overview.html

[52] Ping Identity - FIDO2/WebAuthn with Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/am-authentication/authn-mfa-webauthn.html

[53] YubiKey 5 Series for Ping Identity: https://www.yubico.com/products/yubikey-5-series-for-ping-identity/

[54] Ping Identity - PingOne Privilege: https://www.pingidentity.com/en/platform/capabilities/privileged-access.html

[55] Ping Identity - Runtime Privileged Access: https://www.pingidentity.com/en/platform/capabilities/runtime-privileged-access.html

[56] Ping Identity - PingOne Privilege Launch (August 2025): https://www.pingidentity.com/en/news/press-releases/just-in-time-privileged-access.html

[57] Ping Identity - Rate Limits and Allowed IPs: https://docs.pingidentity.com/pingone/settings/p1_rate_limits.html

[58] Ping Identity - Standard Platform Limits: https://docs.pingidentity.com/pingone/getting_started_with_pingone/p1_platform_limits.html

[59] Ping Identity - Data Regions (Advanced Identity Cloud): https://docs.pingidentity.com/pingoneaic/product-information/global-identity-cloud-locations.html

[60] Ping Identity - Data Residency (Advanced Identity Cloud): https://docs.pingidentity.com/pingoneaic/product-information/global-identity-cloud-locations.html

[61] SAP Help Portal - Rate Limiting and Throttling: https://help.sap.com/docs/identity-services/identity-services/rate-limiting-and-throttling

[62] Vendr - Ping Identity Software Pricing & Plans 2026: https://www.vendr.com/marketplace/ping-identity

[63] SuperTokens - Ping Identity Pricing Guide: https://supertokens.com/blog/ping-identity-pricing-guide

[64] sota.io - Ping Identity EU Alternative 2026: https://sota.io/blog/ping-identity-eu-alternative-2026

[65] Ping Identity - Comparison: PingOne Advanced Services and PingOne Cloud Platform: https://docs.pingidentity.com/pingone/overview/pingone_comparison.html

[66] Ping Identity - Customer data storage for European customers: https://docs.pingidentity.com/pingone/overview/pingone_data_residency.html