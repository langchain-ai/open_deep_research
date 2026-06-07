# Comparative Analysis: Okta Workforce Identity, Microsoft Entra ID Premium P2, and Ping Identity PingOne for a 2,500-User European Pharmaceutical Company

---

## Executive Summary

This report delivers a rigorous, source-backed comparison of three leading enterprise IAM platforms—Okta Workforce Identity, Microsoft Entra ID Premium P2, and Ping Identity PingOne—in the context of a 2,500-employee pharmaceutical company operating across Germany, France, and Poland. Each platform is evaluated for 5-year total cost of ownership (TCO), advanced MFA (with special focus on phishing resistance), Privileged Access Management (PAM/PIM) for SOX compliance (including feature packaging/licensing), published API limits and event throughput (especially for SAP/Salesforce batch needs), concrete EU data residency and GDPR alignment, and a licensing/support breakdown with explicit attention to direct/quote-based features. Open questions requiring direct vendor engagement are clearly flagged for full transparency.

---

## 1. 5-Year TCO: Licensing, Professional Services, and Support for 2,500 Users

### Okta Workforce Identity

**Licensing & Feature Packaging**
- Workforce Identity (Essentials level, required for advanced compliance): *List price* $17/user/month; actual price negotiable for enterprise deals, but advanced PAM, Identity Governance, API Access Management, and Workflow Automation are **add-on SKUs** and often "quoted only" in highly regulated, multinational deployments. Multi-factor adaptive and phishing-resistant MFA is standard, but PAM/IGA is tiered extra[1][2][3][4].
- **Base license for 2,500 users:** $17 × 2,500 × 12 = $510,000/year
- **Professional/implementation services (Year 1 estimate):** 2.5x annual license base = $1,275,000[5]
- **Premium support (per year):** 15% = $76,500
- **Annual escalation:** 3% typical industry rate
- **Note:** Substantial components (e.g., Privileged Access, API Access Management) are only available through custom contracts—essential for pharma/SOX contexts[4][5].

**5-Year TCO Calculation (USD)**
| Year | License | Support | Implementation | Yearly Total  | Cumulative Total   |
|------|---------|---------|----------------|---------------|--------------------|
| 1    | $510,000 | $76,500 | $1,275,000   | $1,861,500    | $1,861,500         |
| 2    | $525,300 | $78,795 | –            | $604,095      | $2,465,595         |
| 3    | $540,959 | $81,144 | –            | $622,103      | $3,087,698         |
| 4    | $557,188 | $83,548 | –            | $640,735      | $3,728,433         |
| 5    | $573,904 | $86,014 | –            | $659,918      | $4,388,351         |

**Total (Approx):** $4.4 million (excluding extra add-on modules which can push TCO higher; direct quote essential for compliance modules.)

---
### Microsoft Entra ID Premium P2

**Licensing & Feature Packaging**
- Entra ID P2: $9/user/month list price (direct); includes core SSO, MFA, PIM, conditional access, basic governance. Advanced Identity Governance is add-on, but for core SOX, PIM/PAM features are in base P2[6][7][8].
- **Base license for 2,500 users:** $9 × 2,500 × 12 = $270,000/year
- **Professional/implementation services (Year 1 estimate):** 15% × $270,000 = $40,500[9]
- **Premium support (per year):** 10% = $27,000
- **Annual escalation:** 3% standard
- **Note:** Large enterprises can negotiate discounts via EA; bundling with Microsoft 365 can further lower real world cost[7][8][9].

**5-Year TCO Calculation (USD)**
| Year | License | Support | Implementation | Yearly Total  | Cumulative Total   |
|------|---------|---------|----------------|---------------|--------------------|
| 1    | $270,000 | $27,000 | $40,500      | $337,500      | $337,500           |
| 2    | $278,100 | $27,810 | –            | $305,910      | $643,410           |
| 3    | $286,443 | $28,644 | –            | $315,087      | $958,497           |
| 4    | $295,036 | $29,504 | –            | $324,540      | $1,283,037         |
| 5    | $303,887 | $30,389 | –            | $334,276      | $1,617,313         |

**Total (Approx):** $1.6 million (P2 covers PAM/PIM for most SOX use; costs increase if advanced governance or supplemental modules are needed.)

---
### Ping Identity PingOne

**Licensing & Feature Packaging**
- Published list price (Workforce “Plus”): $6/user/month, **5,000-user minimum contract**; for 2,500-user deals, direct quote is required, and pricing/features may differ[10][11][12][13].
- **If forced to 5,000-minimum price:** $6 × 5,000 × 12 = $360,000/year (if vendor agrees to adjust minimum, TCO proportional for 2,500)
- **Professional/implementation services (Year 1 estimate):** 35% × $360,000 = $126,000[14]
- **Premium support (per year):** 10% = $36,000
- **Annual escalation:** 3%
- **Note:** Advanced PAM/PIM features (PingOne Privilege) are **modular and "quote only"**; pharmaceutical/SOX use mandates explicit scoping and negotiation[14][15].

**5-Year TCO Calculation (USD, at 5k-min price)**
| Year | License | Support | Implementation | Yearly Total  | Cumulative Total   |
|------|---------|---------|----------------|---------------|--------------------|
| 1    | $360,000 | $36,000 | $126,000      | $522,000      | $522,000           |
| 2    | $370,800 | $37,080 | –             | $407,880      | $929,880           |
| 3    | $381,924 | $38,192 | –             | $420,116      | $1,349,996         |
| 4    | $393,382 | $39,338 | –             | $432,720      | $1,782,716         |
| 5    | $405,183 | $40,518 | –             | $445,701      | $2,228,417         |

**Total (5k-min):** $2.2 million (true 2,500-user deals are likely "quote only"—TCO is not public and must be confirmed.)

---
**Summary Table:**  
| Platform | 5-Year TCO (2,500 users) | Compliance Modules | Implementation | Support | Licensing Caveats      |
|----------|--------------------------|-------------------|----------------|---------|------------------------|
| Okta     | ~$4.4M (base only; more for modules) | PAM, Governance quoted add-ons | 2.5x Y1 license | 15%/yr | All serious compliance add-ons “quote only” |
| Entra P2 | ~$1.6M                   | PAM/PIM in P2     | 15% Y1 license | 10%/yr | Lower with M365 bundle  |
| PingOne  | ~$2.2M (@5k min users)   | PAM quoted add-on | 35% Y1 license | 10%/yr | 5k user min contract, real 2,500-user “quote only” |

---

## 2. MFA and Phishing-Resistant Authentication Methods

### Okta Workforce Identity

- **Phishing-resistant methods:**  
  - FIDO2/WebAuthn (YubiKey, platform biometrics)  
  - Okta FastPass (device-bound, passwordless biometrics or PIN)  
  - X.509 smartcards/PIV[16][17][18][19].
- **Policy and enforcement:**  
  - Admins can require phishing-resistant MFA for admin and critical user roles, supporting step-up/conditional policies[20].  
  - Push/QR and fallback factors are available but less secure.
- **Primary Documentation:**  
  [Okta FastPass Product Docs](https://www.okta.com/blog/product-innovation/okta-fastpass-phishing-resistant-mfa/)  
  [Implement Phishing-Resistant Auth](https://learning.okta.com/path/implement-phishing-resistant-authentication)

### Microsoft Entra ID Premium P2

- **Phishing-resistant methods:**  
  - Passkeys (FIDO2)  
  - Windows Hello for Business  
  - Microsoft Authenticator (app passkey/number match)  
  - Certificate-based/smartcard[21][22][23][24].
- **Policy and enforcement:**  
  - Conditional Access allows admins to enforce “phishing-resistant MFA” specifically for privileged roles or across the organization[22][24].
- **Integration:**  
  - Seamless in Windows environments; Admin roles (Global, Security, User Admin, etc.) enforceable by policy[22].

### Ping Identity PingOne

- **Phishing-resistant methods:**  
  - FIDO2 security keys/biometrics (YubiKey, Windows Hello, Touch ID)  
  - PingID app (biometric/push, number match)  
  - Smartcard/X.509 certificate[25][26][27][28].
- **Policy and enforcement:**  
  - Admins can require only phishing-resistant factors for selected users, step-up for high-risk access, and restrict fallback/OTP[28].
- **Device posture:**  
  - Up to 20 paired devices per user; admin console unlock/reset; detection of MFA fatigue attacks in policy[29].
- **Documentation:**  
  [PingOne MFA Methods](https://docs.pingidentity.com/pingone/strong_authentication_mfa/p1_authentication_methods_overview.html)  
  [Yubikey + Ping](https://marketplace.pingone.com/item/yubikey)

---

## 3. Privileged Access Management (PAM/PIM) & SOX Compliance

### Okta Workforce Identity

- **Privileged Access:**  
  - *Okta Privileged Access* is a unified PAM platform supporting SSO to servers, session recording (SSH/RDP), secrets vaulting, just-in-time access, approval workflows, and audit logs, specifically targeting SOX compliance requirements[30][31][32].
- **Feature packaging:**  
  - PAM is an advanced add-on to Workforce Identity Cloud; only available via enterprise quote, not included in base. Precise SKU and cost requires direct Okta engagement for packaging[4][5][33].

### Microsoft Entra ID Premium P2

- **Privileged Identity Management (PIM):**  
  - Core PIM enables just-in-time admin access, time/approval-based role activation, eligibility review, Conditional Access on privilege elevation, and comprehensive audit logs compatible with SOX regulations[21][34][35][36].
- **Feature packaging:**  
  - PIM is included in **Premium P2**. Only users who need privileged access require P2; others can remain on lower-cost SKUs. Identity Governance for more advanced features is a separate add-on[8][37].

### Ping Identity PingOne

- **Privileged Access:**  
  - *PingOne Privilege* enables real-time privilege elevation, JIT tokens, zero standing privilege, session recording/logging, device trust, and full hybrid/cloud resource coverage[38][39][40].
- **Feature packaging:**  
  - PAM/PIM is modular and sold as a distinct tier/add-on; not included in standard Workforce tier—required for SOX/audited pharma workloads. SOX-specific audit reporting needs confirmation via RFP[14][40].
  
**Summary Table:**  
| Platform | PAM/PIM Features       | SOX Audit | Packaging           | SKU Inclusion     |
|----------|-----------------------|-----------|---------------------|-------------------|
| Okta     | Full PAM, session rec | Yes       | Quote-only add-on   | Separate from base|
| Entra P2 | Full PIM, just-in-time| Yes       | In P2, add-ons for IGA| Included in P2   |
| PingOne  | Full PAM/JIT, ZSP     | Yes       | Modular, quote-only | Add-on, not base  |

---

## 4. API Rate Limits & Authentication/Event Throughput (SAP/Salesforce at 2,500-User Scale)

### Okta Workforce Identity

- **API rate limits (2,500 users):**  
  - Default org limit: 1,200 requests/minute (all endpoints)  
  - Per integration service (e.g., SAP/Salesforce): 50% of org limit (600 req/min)[41][42][43].  
  - Per-user cap: 4 login req/sec[44].
- **Scalability:**  
  - Adequate support for daily authentication and SSO, including SAP/Salesforce SAML/SCIM provisioning at pharma enterprise scale. High-volume bulk events (onboarding, mass update) may require job staging to avoid HTTP 429 throttling; higher caps negotiable via support/RFP[45][46].
- **Target platform limits:**  
  - Salesforce: SFDC API quotas (typically 15,000–100,000 calls/day for enterprise licenses) may be a bottleneck in batch provisioning, **not Okta**[47].

### Microsoft Entra ID Premium P2

- **API rate limits:**  
  - Microsoft Graph: 10,000 per 10 minutes per app/org (~17/sec)[48].  
  - App Proxy: 500/sec per app; 200/sec per tenant (External ID)[49][50].  
  - SCIM provisioning: No hard documented per-app cap—subject to performance best practices[51].
- **Scalability:**  
  - Easily sufficient for interactive and batch operations for 2,500 users (including SAP/Salesforce batch sync)[52][53].
- **Target platform limits:**  
  - No Entra bottleneck in SAP/SF integration; must obey SAP/SF-target platform API boundaries for mass updates[53].

### Ping Identity PingOne

- **API rate limits:**  
  - Directory API: 2,500/min (41.6/sec)  
  - SSO: 1,500/min  
  - Per endpoint base rates: Audit API 10/sec; Authorization 150/sec[54][55][56].  
  - User profile: Up to 16 kB  
  - Higher limits and throughput assurance available via contract[57].
- **Scalability:**  
  - All standard authentication, SSO, SCIM flows for 2,500 users accommodated; batch sync jobs should be scheduled or tested with vendor for spikes during onboarding or annual processes. Bulk failures return HTTP 429; higher bursts require support engagement[58].
- **Target platform limits:**  
  - See SAP/Salesforce API quotas; Ping provides integration kits for both[59][60].

**Summary Table:**  
| Platform | API Limit (per min)     | SAP/SF Throughput | Batch Event Guidance    | Extensible via Support  |
|----------|------------------------|-------------------|------------------------|-------------------------|
| Okta     | 1,200 (org)/600 (svc)  | Yes, with staging | Stagger batch/up throttling | Yes (quote)           |
| Entra P2 | ~10,000/10min (Graph)  | Yes               | No platform bottleneck | Yes (MS support)        |
| PingOne  | 2,500 (directory)      | Yes, if jobs staggered | Test batch peaks     | Yes (quote)             |

---

## 5. EU Data Residency, Data Center Locations & GDPR Alignment

### Okta Workforce Identity

- **Data locality:**  
  - EU region selectable at tenant creation; major centers in Germany, with other EU options available[61][62][63].  
  - Data not transferred out of region except per SCC/DPA and explicit customer consent[64][65].
- **GDPR compliance:**  
  - Certified with ISO 27001/17/18, GDPR supporting documentation, SCCs, and published DPA/subprocessor lists[64][65][66].
- **Custom PII protections:**  
  - Optional integration with Datex DataStealth for in-country encryption/tokenization—useful for pharmaceutical clients[67].

### Microsoft Entra ID Premium P2

- **Data locality:**  
  - Tenant provisioned in Germany, France, or EU region; full Azure data boundary support incl. advanced data residency (ADR) options for Germany/France; Poland uses EU default[68][69][70].  
  - Data migration for region change is possible via M365 Admin Center (data location card)[70].
- **GDPR compliance:**  
  - Full compliance, subject to Microsoft's DPA, SCCs, ISO 27001/17/18, and CSA STAR[71][72].
- **Geo control:**  
  - Enforced at tenant creation; region cannot be changed retroactively except via migration project[70].

### Ping Identity PingOne

- **Data locality:**  
  - At provisioning, select from multiple EU GCP data centers (Frankfurt, Paris, London, Netherlands, others; all options mapped in docs)[73][74].  
  - Backups and runtime logs remain located in selected geography[75].
- **GDPR compliance:**  
  - Stated as full, with DPA, subprocessor transparency, and EU data boundary maintained for PII[75][76].
- **Flexibility:**  
  - Advanced Identity Cloud and Governance each support granular region selection; ideal for distributed pharma teams[73].

**Summary Table:**  
| Platform | Regional Data Centers     | Region Selection         | GDPR Alignment | PII Escrow/Encryption |
|----------|--------------------------|-------------------------|---------------|----------------------|
| Okta     | Germany, France, EU      | On setup; fixed per tenant| Full         | Optional (Datex)     |
| Entra P2 | Germany, France, EU      | On setup; ADR for top regions| Full   | No local PII escrow  |
| PingOne  | Frankfurt, Paris, London | On setup per environment| Full          | Built-in; EU resolve |

---

## 6. Licensing and Support Models: Transparency on Quote/Feature Gating

### Okta Workforce Identity

- **Per-user/month model; advanced features (PAM, IGA, API AM, Workflows) are “quote only” at enterprise scale**[4][5][33].
- **Discounts:** Negotiable, especially >5,000 users or with multi-year contracts[3].
- **Professional services:** Standard is 2.5x annual license in year 1, reflecting multinational pharma complexity[5].
- **Support upgrades:** 11–25%/yr typical for premium SLA; advanced compliance reporting and integration often reserved for premium support customers.

### Microsoft Entra ID Premium P2

- **Per-user/month for P2 features; PIM included**
- **Volume/EA discounts:** Negotiable for large orgs; bundles with M365 can reduce price further[7][8][9].
- **Implementation:** 15% of annual license is a standard consulting starting point in year 1.
- **Support:** 10%/yr of license estimate for Enhanced/Premier plans.

### Ping Identity PingOne

- **List model is published but only valid for 5,000+ users—for 2,500-user enterprise, all pricing is “quote only”**[10][11][13][14].
- **Professional services:** 35% of first-year license as minimum; varies by SSO/PAM/complex workload[14].
- **Support tiers:** Add 10%/yr for advanced support/SLA.
- **Advanced features:** PAM/PIM/SOX modules are modular and only available as contract add-ons.

**Key caveat for each platform:**  
**Okta:** Final cost for compliance add-ons, API increases, and some region-specific options only available via direct engagement/RFP.  
**Microsoft:** Full quotes for multi-year, ADR, or non-list tier bundles require EA negotiation.  
**Ping:** All detailed pricing, inclusion of advanced PAM, and non-5,000-user contract scenarios are only available via vendor quote.

---

## 7. Remaining Open Questions & Advisory for Direct Vendor Engagement

- **Okta:**  
  - Confirm enterprise PAM/IGA module SKU/package for SOX (quote-specific).  
  - Get written commitments on burst rate increases for peak SAP/Salesforce event flows.
- **Microsoft Entra:**  
  - Confirm eligibility and pricing for ADR add-ons per country and any specific Identity Governance overlap if future IGA features are important.
  - Map out licensing split if only privileged roles/users need P2 for cost efficiency.
- **Ping Identity:**  
  - Obtain explicit TCO quote for 2,500-user deployment (minimum is 5,000 on published price), incl. advanced PAM/PIM/SOX support.
  - Confirm contract include real SOX compliance audit features and batch authentication/event scaling.
  
Without explicit quotes, precise 2,500-user TCO and compliance add-on coverage cannot be fully established. Vendor engagement is essential for final contract and pricing review.

---

## Conclusion

- **Okta Workforce Identity:** Most flexible on cross-cloud integration, outstanding phishing-resistant MFA portfolio, and deep compliance controls, but highest TCO for full pharma/SOX coverage—mainly due to add-on/“quote only” packaging and professional services.
- **Microsoft Entra ID Premium P2:** Drastically lower TCO, powerful PIM and Conditional Access (all in P2 tier), best choice for Microsoft-centered environments, with some future-proofing required for evolving IGA features and data residency add-ons.
- **Ping Identity PingOne:** Competitive in price if deployed at or above 5,000 users, remarkable architectural flexibility and data residency, best orchestration and policy engine. For <5,000 users, real-world price and compliance features must be individually quoted.

**Platform selection should be finalized after a formal RFP with each vendor, specifying:**  
- Precise PAM/PIM features for SOX  
- Actual service limits and scaling options  
- Contracted data residency scope  
- Final price for target user count, country deployment, and pharmaceutical compliance package

---

### Sources

[1] Okta Pricing 2026: Ultimate Guide for Security Products – UnderDefense: https://underdefense.com/industry-pricings/okta-pricing-ultimate-guide-for-security-products/  
[2] Plans and Pricing – Okta: https://www.okta.com/pricing/  
[3] Okta Pricing 2026: Plans, Costs & Hidden Fees – CheckThat.ai: https://checkthat.ai/brands/okta/pricing  
[4] The Okta Tax: How Much Are You Really Paying for Identity? – Authentik: https://goauthentik.io/blog/2026-02-23-the-okta-tax/  
[5] Okta Software Pricing & Plans 2026: See Your Cost – Vendr: https://www.vendr.com/marketplace/okta  
[6] Microsoft Entra ID Pricing 2026 – TrustRadius: https://www.trustradius.com/products/microsoft-entra-id/pricing  
[7] Microsoft Entra ID Pricing Overview – G2: https://www.g2.com/products/microsoft-entra-id/pricing  
[8] Microsoft Entra ID Licensing: P1 vs P2 & Cost Guide 2026 – Atonement Licensing: https://atonementlicensing.com/blog/entra-id-licensing/  
[9] Microsoft Entra ID Governance licensing fundamentals – Microsoft Learn: https://learn.microsoft.com/en-us/entra/id-governance/licensing-fundamentals  
[10] Ping Identity Pricing 2026: Plans, Costs & Enterprise Breakdown – CheckThat.ai: https://checkthat.ai/brands/ping-identity/pricing  
[11] Ping Identity Pricing 2026: Plans & Cost – PulseSignal: https://getpulsesignal.com/pricing/pingidentity  
[12] PingOne from Ping Identity Pricing 2026 – TrustRadius: https://www.trustradius.com/products/ping-identity-pingone/pricing  
[13] Standard license types | PingOne – Ping Identity Docs: https://docs.pingidentity.com/pingone/getting_started_with_pingone/p1_license_types.html  
[14] Ping Identity Software Pricing & Plans 2026: See Your Cost – Vendr: https://www.vendr.com/marketplace/ping-identity  
[15] Guide to GDPR Data Residency Requirements for Compliance – GDPR Local: https://gdprlocal.com/gdpr-data-residency-requirements/  
[16] Okta FastPass: Phishing-resistant MFA – Okta: https://www.okta.com/blog/product-innovation/okta-fastpass-phishing-resistant-mfa/  
[17] Implement Phishing-Resistant Authentication – Okta: https://learning.okta.com/path/implement-phishing-resistant-authentication  
[18] The Need for Phishing-Resistant Multi-Factor Authentication – Okta: https://www.okta.com/blog/identity-security/the-need-for-phishing-resistant-multi-factor-authentication/  
[19] Multifactor authentication | Okta: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/about-authenticators.htm  
[20] Okta Security Knowledge - Phishing Resistant Factors: https://support.okta.com/help/s/article/Okta-Security-Knowledge-Phishing-Resistance?language=en_US  
[21] Get started with a phishing-resistant passwordless authentication deployment in Microsoft Entra ID – Microsoft Learn: https://learn.microsoft.com/en-us/entra/identity/authentication/how-to-plan-prerequisites-phishing-resistant-passwordless-authentication  
[22] Overview of Conditional Access Authentication Strengths – Microsoft Learn: https://learn.microsoft.com/en-us/entra/identity/authentication/concept-authentication-strengths  
[23] Require phishing-resistant multifactor authentication for Microsoft Entra administrator roles – Microsoft Learn: https://learn.microsoft.com/en-us/entra/identity/conditional-access/policy-admin-phish-resistant-mfa  
[24] Windows Hello for Business overview – Microsoft Learn: https://learn.microsoft.com/en-us/windows/security/identity-protection/hello-for-business/hello-overview  
[25] Overview of authentication methods | PingOne – Ping Identity Docs: https://docs.pingidentity.com/pingone/strong_authentication_mfa/p1_authentication_methods_overview.html  
[26] Configuring MFA settings | PingOne: https://docs.pingidentity.com/pingone/authentication/p1_configure_mfa_settings.html  
[27] YubiKey 5 Series: Phishing-Resistant MFA for the PingOne Platform | Ping Identity Marketplace: https://marketplace.pingone.com/item/yubikey  
[28] Configuring an MFA policy for strong authentication | PingOne: https://docs.pingidentity.com/pingone/strong_authentication_mfa/p1_creating_an_mfa_policy_for_strong_auth.html  
[29] PingOne MFA - Cloud Multi-factor Authentication for Customers: https://www.pingidentity.com/en/product/pingone-mfa.html  
[30] Okta Privileged Access – Okta Product Page: https://www.okta.com/products/privileged-access/  
[31] Okta Privileged Access | Okta Classic Engine: https://help.okta.com/en-us/content/topics/privileged-access/pam-overview.htm  
[32] Okta Privileged Access Datasheet: https://www.okta.com/sites/default/files/2024-11/Datasheet-Okta%20Privileged%20Access_.pdf  
[33] Okta Workforce Identity Tech Specs: https://invgate.com/itdb/okta-workforce-identity  
[34] Privileged Identity Management (PIM) | Microsoft Security: https://www.microsoft.com/en-us/security/business/identity-access/microsoft-entra-privileged-identity-management-pim  
[35] Microsoft Entra ID Governance licensing fundamentals – Microsoft Learn: https://learn.microsoft.com/en-us/entra/id-governance/licensing-fundamentals  
[36] What is Microsoft Entra Privileged Identity Management? – Microsoft Learn: https://learn.microsoft.com/en-us/entra/identity/privileged-identity-management/pim-configure  
[37] Pricing for Microsoft Entra Identity Governance – Microsoft Learn: https://learn.microsoft.com/en-us/entra/id-governance/governance-service-limits  
[38] Key concepts | PingOne Privilege: https://docs.pingidentity.com/privilege/getting-started/key-concepts.html  
[39] Runtime Privileged Access | Ping Identity: https://www.pingidentity.com/en/capability/jit-privileged-access.html  
[40] PingOne Licensing & Capabilities – Quote Form: https://getpulsesignal.com/pricing/pingidentity  
[41] Rate limits | Okta Developer: https://developer.okta.com/docs/reference/rate-limits/  
[42] Additional Rate limits | Okta Developer: https://developer.okta.com/docs/reference/rl2-limits/  
[43] Rate Limits for API Service Integrations: https://support.okta.com/help/s/article/rate-limits-for-api-service-integrations?  
[44] Rate Limit Frequently Asked Questions – Okta: https://support.okta.com/help/s/article/rate-limit-frequently-asked-questions  
[45] Set up and Integrate OKTA with Salesforce Identity: https://help.salesforce.com/s/articleView?id=cc.b2c_account_manager_sso_integrate_okta.htm&language=en_US&type=5  
[46] Okta Integration Network (OIN): https://www.okta.com/integrations/  
[47] Salesforce API Limits Quick Reference Guide: https://help.salesforce.com/s/articleView?id=sf.api_requests_limits.htm  
[48] Service limits and restrictions – Microsoft Entra ID: https://learn.microsoft.com/en-us/entra/identity/users/directory-service-limits-restrictions  
[49] Service limits and restrictions – Microsoft Entra External ID: https://learn.microsoft.com/en-us/entra/external-id/customers/reference-service-limits  
[50] Application Proxy load limits – Microsoft Learn: https://learn.microsoft.com/en-us/azure/active-directory/app-proxy/application-proxy-scalability  
[51] Configure Salesforce for automatic user provisioning with Microsoft Entra ID: https://learn.microsoft.com/en-us/entra/identity/saas-apps/salesforce-provisioning-tutorial  
[52] Microsoft Entra ID and SAP integration: https://learn.microsoft.com/en-us/azure/active-directory/saas-apps/sap-sso  
[53] SCIM integration in Entra for Salesforce – Microsoft Q&A: https://learn.microsoft.com/en-us/answers/questions/5828857/scim-integration-in-entra-for-salesforce  
[54] PingOne standard platform limits | PingOne: https://docs.pingidentity.com/pingone/getting_started_with_pingone/p1_platform_limits.html  
[55] Rate Limits and Allowed IPs | PingOne: https://docs.pingidentity.com/pingone/settings/p1_rate_limits.html  
[56] Rate Limiting | PingOne Platform APIs: https://developer.pingidentity.com/pingone-api/platform/rate-limiting.html  
[57] Base rate limits | PingOne Platform APIs: https://developer.pingidentity.com/pingone-api/platform/rate-limiting/base-rate-limits.html  
[58] PingOne API Response Codes: https://docs.pingidentity.com/pingone/api/p1api_response_codes.html  
[59] SCIM Provisioner | PingFederate Integrations: https://docs.pingidentity.com/integrations/scim/pf_scim_connector.html  
[60] Federating PingOne and Salesforce | Solution Guides: https://docs.pingidentity.com/solution-guides/single_sign-on_use_cases/htg_federate_p1_salesforce.html  
[61] Okta + Data Residency: https://www.okta.com/okta-data-residency/  
[62] GDPR | Okta: https://www.okta.com/legal/gdpr/  
[63] Data residency for User Directory -- where is the data stored? | Okta Developer Community: https://devforum.okta.com/t/data-residency-for-user-directory-where-is-the-data-stored/6244  
[64] Keeping Your Data Safe: Identity, Security and The GDPR | Okta: https://www.okta.com/node/5495/  
[65] Privacy Policy | Okta: https://www.okta.com/legal/privacy-policy/2025-03-archived/  
[66] Workforce Identity Cloud Customer Data Retention Policy: https://support.okta.com/help/s/article/Customer-Data-Retention-Policy?language=en_US  
[67] Okta + Datex: https://www.okta.com/partners/datex/  
[68] Customer data storage and processing for European customers in Microsoft Entra ID: https://learn.microsoft.com/en-us/entra/fundamentals/data-storage-eu  
[69] Microsoft 365 Advanced Data Residency – Microsoft Learn: https://learn.microsoft.com/en-us/microsoft-365/enterprise/advanced-data-residency?view=o365-worldwide  
[70] Microsoft 365 data locations – Microsoft Learn: https://learn.microsoft.com/en-us/microsoft-365/enterprise/o365-data-locations?view=o365-worldwide  
[71] GDPR Compliance Made Easy with Microsoft Entra: https://hoop.dev/blog/gdpr-compliance-made-easy-with-microsoft-entra/  
[72] Microsoft Product Terms: https://www.microsoft.com/licensing/terms  
[73] Data residency | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/tenants/data-residency.html  
[74] Data regions | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/product-information/global-identity-cloud-locations.html  
[75] Security and compliance | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/product-information/security-compliance.html  
[76] Data Supplement | Ping Identity: https://www.pingidentity.com/en-us/docs/legal/data-supplement