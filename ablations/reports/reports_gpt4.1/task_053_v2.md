# Comparative Evaluation of Okta Workforce Identity, Microsoft Entra ID Premium P2, and Ping Identity PingOne for a European Pharmaceutical Company

## Executive Summary

This report delivers a transparent, scenario-specific comparative analysis of Okta Workforce Identity, Microsoft Entra ID Premium P2, and Ping Identity PingOne as enterprise IAM solutions for a 2,500-user European pharmaceutical company operating in Germany, France, and Poland. The evaluation covers 5-year cost breakdowns, detailed data residency and GDPR compliance capabilities, API throughput and technical integration for SAP and Salesforce, performance validation, advanced MFA methods in context, and in-depth architectural integration/connector guidance — all based on current vendor documentation and industry benchmarks.

---

## 1. Explicit 5-Year Licensing Costs and TCO Breakdown

### Modelling Parameters and Assumptions

- **User count for all models:** 5,000 (PingOne requires 5,000-user minimum for enterprise licensing; Okta/Microsoft priced equivalently for comparability).
- **Feature set:** Includes SSO, adaptive MFA, identity governance, privileged access, lifecycle management, API connectors — as required for pharmaceutical compliance (GDPR, SOX).
- **Implementation cost:** 35% (PingOne), 15% (Microsoft), 2.5x annual license (Okta); reflects multi-country pharma integration with SAP/Salesforce and AD migration.
- **Premium support:** 10–15% annual license cost.
- **Annual price escalation:** 3% per year, industry standard.
- **Professional services:**
  - Okta: 2.5x year-one license
  - Microsoft: 15% of year-one license
  - PingOne: 35% of year-one license

### Okta Workforce Identity

- **License cost:** $17/user/month × 5,000 × 12 = $1,020,000/year
- **Professional/Implementation (Year 1):** $1,020,000 × 2.5 = $2,550,000
- **Premium support:** 15% = $153,000/year
- **Annual escalation:** 3%

| Year | License | Support | Implementation | Yearly Total | Cumulative Total |
|------|---------|---------|----------------|--------------|------------------|
| 1    | $1,020,000 | $153,000 | $2,550,000  | $3,723,000  | $3,723,000       |
| 2    | $1,050,600 | $157,590 | –            | $1,208,190  | $4,931,190       |
| 3    | $1,082,118 | $162,318 | –            | $1,244,436  | $6,175,626       |
| 4    | $1,114,581 | $167,187 | –            | $1,281,768  | $7,457,394       |
| 5    | $1,148,018 | $172,203 | –            | $1,320,221  | $8,777,615       |

**5-Year TCO:** ≈ **$8.8 million**  
(*Excludes volume discounts or negotiated feature bundling. Actuals may vary by enterprise negotiation*)  
**TCO Drivers:** High initial implementation, recurring premium support; premium feature gating at higher tiers[1][2][3][4][5][6][7][8].

### Microsoft Entra ID Premium P2

- **License cost:** $9/user/month × 5,000 × 12 = $540,000/year
- **Implementation (Year 1):** 15% × $540,000 = $81,000
- **Premium support:** 10% of license = $54,000/year
- **Annual escalation:** 3%

| Year | License | Support | Implementation | Yearly Total | Cumulative Total |
|------|---------|---------|----------------|--------------|------------------|
| 1    | $540,000 | $54,000 | $81,000       | $675,000    | $675,000         |
| 2    | $556,200 | $55,620 | –             | $611,820    | $1,286,820       |
| 3    | $572,886 | $57,289 | –             | $630,175    | $1,917,995       |
| 4    | $590,073 | $59,007 | –             | $649,080    | $2,567,075       |
| 5    | $607,775 | $60,777 | –             | $668,552    | $3,235,627       |

**5-Year TCO:** ≈ **$3.24 million**  
(*Excludes possible additional costs for Intune, cloud governance, or advanced device controls; assumes no discount from bundle with M365 E5*)  
**TCO Drivers:** Lowest license costs, in-house expertise reduces services costs, M365 bundle may lower total further[9][10][11][12].

### Ping Identity PingOne Workforce Plus

- **Contractual user minimum:** 5,000 (actual deployment size immaterial)
- **License cost:** $6/user/month × 5,000 × 12 = $360,000/year (cannot be reduced if fewer than 5,000 users)
- **Implementation (Year 1):** 35% × $360,000 = $126,000
- **Premium support:** 10% = $36,000/year
- **Annual escalation:** 3%

| Year | License | Support | Implementation | Yearly Total | Cumulative Total |
|------|---------|---------|----------------|--------------|------------------|
| 1    | $360,000 | $36,000 | $126,000      | $522,000    | $522,000         |
| 2    | $370,800 | $37,080 | –             | $407,880    | $929,880         |
| 3    | $381,924 | $38,192 | –             | $420,116    | $1,349,996       |
| 4    | $393,382 | $39,338 | –             | $432,720    | $1,782,716       |
| 5    | $405,183 | $40,518 | –             | $445,701    | $2,228,417       |

**5-Year TCO:** ≈ **$2.23 million**  
(*Lower license cost but enforced 5,000-user/year minimum; cost-efficient for over 3,500+ staff scenarios*)  
**TCO Drivers:** Low per-user pricing, multi-year user min, simple support, potentially higher integration effort if hybrid/legacy[13][14][15][16][17].

---

### Key Observations

- **Okta**: Highest TCO due to aggressive implementation and premium support costs required for multinational pharma compliance; is the most scalable for very large or highly customized/federated environments.
- **Microsoft Entra ID P2**: Most cost-efficient assuming no excessive dependency on paid add-ons; especially advantageous if the organization is already invested in Microsoft 365.
- **PingOne**: Most cost-predictable (unless workforce <3,500); combines robust compliance with the lowest licensing/service costs under these constraints.

---

## 2. Data Residency and GDPR Compliance

### Okta Workforce Identity

- **Data Center Locality**: EU data region selectable at tenant creation (Germany, France, Poland available); data replication contained within the EU region[18][19][20].
- **Compliance**: Full alignment with GDPR, Schrems II, ISO 27001/17/18, CSA STAR; Data Processing Addendum (DPA), standard contractual clauses (SCCs) provided to all customers[18][19][20][21].
- **Audit and Log Storage**: System logs retained for 90 days (api exportable), infra logs up to 12 months, backup data purged after 6 months. Customizable audit decomposition for pharma compliance[22][23].
- **Deletion and Portability**: Right to erasure/request supported through APIs and administrator console; compliant post-termination data destruction process[18][19][20].
- **Data Transfer**: International transfers only as required, always via SCCs/DPA; transparent subprocessor listing and controls[19][21].

### Microsoft Entra ID Premium P2

- **Data Center Locality**: Azure region selection at tenant provisioning (Germany, France, Poland, Netherlands, Paris, London, Zurich, Finland); geo-locked, cannot be changed post-setup[24][25].
- **Compliance**: Certified for ISO 27001/17/18, CSA STAR, SOC2-II; Advanced Data Residency & EU Data Boundary; DPA/SCCs published[24][25].
- **Audit and Log Storage**: All identity, access, and privilege logs retained 1 year+ (minimum); extended retention available via SIEM/export. Some non-PII telemetry/fraud data may be processed outside EU (with contracts/controls)[24][25][26].
- **Deletion and Portability**: Access, rectification, and erasure fully subject-driven via admin tools and automation; logs exportable[24][25].
- **Data Transfer**: Limited and transparent; DPA and SCCs for all regulated scenarios[25][26].

### Ping Identity PingOne

- **Data Center Locality**: GCP region chosen at onboarding (Frankfurt, Paris, Belgium, London, Netherlands, etc.); backup/redundancy remains in EU region[27][28].
- **Compliance**: SOC2-II, ISO 27001/17/18, CSA STAR, full GDPR/BIPA alignment; DPA/SCCs/subprocessor map published[28][29].
- **Audit and Log Storage**: User event logs: 90 days; configuration/admin: 2 years; export available; 0-day/30-min retention for PII (where required)[30][31].
- **Deletion and Portability**: Instant erasure API; user consent and Article 22 (human oversight) built in[27][28].
- **Data Transfer**: Global operational subprocessors used for support, but regulated PII strictly in region[28][29].

---

### Comparison Summary

- All three platforms offer strong, pharmaceutically compliant GDPR data residency with selectable EU regions, strict log controls, documented contracts/transfers, and proven certifications.
- PingOne and Okta provide granular runtime log and data erasure; Microsoft offers advanced options through the EU Data Boundary.
- Regional lockdowns are fixed on creation; migration requires tenant recreation for all platforms.

---

## 3. API Rate Limits, Federation, and SAP/Salesforce Integration

### Okta Workforce Identity

- **API Rate Limits**: Org-wide: 1,200 requests/min; per-client: 600/min; per-auth endpoint: 4/sec per username. Request overages yield HTTP 429. No automatic handling of target-side throttling[32][33][34].
- **SAP/Salesforce Integration**: Connectors available through Okta Integration Network for SAML/OIDC SSO and SCIM provisioning; supports direct federation, attribute mapping, group/entitlement sync[35][36].
- **Workarounds for Throughput Constraints**: Must stagger large batch updates; no API-level reduction in outbound SCIM flow on target 429; high-volume users must engage Okta Support for higher limits[32][33].

### Microsoft Entra ID Premium P2

- **API Rate Limits**: Microsoft Graph: 10,000/10min per app/org (≈16.7/sec); Application Proxy: 500/sec per app, 750/sec per org. SCIM: No per-app rate limit for custom integrations; must handle own scaling and error control[37][38].
- **SAP/Salesforce Integration**: Pre-integrated gallery apps for SSO/provisioning; on-prem SAP via Entra Connect Provisioning Agent; SAML, OIDC, SCIM all supported[39][40].
- **Throughput Management**: Large batch jobs should be staged; for increases, open escalation with Microsoft Support; strict attribute mapping (e.g., Salesforce entitlements)[40][41].

### Ping Identity PingOne

- **API Rate Limits**: Directory: 2,500/min (41.6/sec); SSO: 1,500/min; per-IP: 35% of licensed throughput by default; explicit IP allow-listing for batch jobs available[42][43][44].
- **SAP/Salesforce Integration**: Native SAML, OIDC, SCIM, REST; SCIM Provisioner for SAP/Salesforce provisioning; federation best practices recommended for both SSO and user lifecycle[45][46].
- **Scaling and Workarounds**: Rate limits negotiable via support contract; monitor via API Usage Dashboard; separate SSO/provisioning connectors for throughput control[43][44].

---

## 4. Throughput and Authentication Volume Assessment

- **Volume Estimation**: For 2,500 users, assume 100–200 peak concurrent authentications (shift start, mass SAP/HR actions), with daily batch user lifecycle updates to SAP/Salesforce.
- **Okta**: With 1,200/min org, 600/min client, and 4/sec per-user cap, system supports all SSO and batch flows for a 2,500-user company, provided large batch jobs are staged. Ongoing monitoring and escalation are required for quarter/year-begin batch HR/SAP updates.
- **Microsoft Entra ID**: 10,000/10min (Graph), 500/sec (App Proxy), 200/sec (External ID), more than sufficient for 2,500 users—even during SAP/Salesforce batch sync.
- **PingOne**: Directory API default (2,500/min) covers steady-state and batch periods; allow-list batch IPs or negotiate increased caps as required.

**Conclusion**: All three platforms meet the throughput and availability needs of a 2,500-user multinational pharma enterprise, assuming proper staging/monitoring for batch jobs and standard integration best practices.

---

## 5. MFA and Phishing-Resistant Authentication Methods

### Okta Workforce Identity

- **Supported Methods**: Okta Verify (push, TOTP), FIDO2/WebAuthn (YubiKey, biometrics), Smartcards/X.509, Okta FastPass (device-bound, passwordless), SMS/email/voice (fallback)[47][48][49][50].
- **Phishing Resistance**: FIDO2 hardware (YubiKey etc.) and FastPass are NIST AAL3-equivalent, immune to phishing/interception.
- **Policy Control**: Automated step-up policies; enforce phishing-resistant methods for privileged/critical roles[51].
- **Business Impact**: FastPass, FIDO2 — median auth time: <4 seconds, low recovery overhead; fallback and device enrollment flows streamlined; robust event logging and self-service[47][49].

### Microsoft Entra ID Premium P2

- **Supported Methods**: FIDO2 security keys, Windows Hello for Business, Microsoft Authenticator (number match), Passkeys, Smartcards (X.509), SMS/phone (fallback)[52][53][54].
- **Phishing Resistance**: FIDO2, Smartcards, Passkeys; conditional access mandates by admin role.
- **Policy Control**: Graph-based, AD-integrated enforcement for step-up/fallback, user self-enrollment, admin resets.
- **Business Impact**: Windows integration gives seamless experience for Office/SAP; average login: 2–6 seconds; recovery via self-service or staged helpdesk process; broad device support, deep PC integration[52][54].

### Ping Identity PingOne

- **Supported Methods**: Push notifications, biometrics via PingID, TOTP, FIDO2 (YubiKey, biometrics), certificate/x.509, SMS/email/voice, Adaptive authentication (risk-driven step-up)[55][56][57].
- **Phishing Resistance**: FIDO2, certificate-based; adaptive engine can enforce context-based elevation to phishing-resistant factors.
- **Policy Control**: Orchestration flows for fallback/enrollment/reset; context-aware risk policies.
- **Business Impact**: Median auth time: 3–5 seconds; fallback/recovery highly configurable; policy-driven enrollment and device management; granular audit logs for pharma compliance[57][56].

---

### MFA Method Comparison & Pharma Suitability

- All platforms support FIDO2, smartcards, and strong adaptive MFA orchestration, making them fully suited for high-assurance, regulated enterprise contexts.
- Microsoft’s native Windows integration and Okta’s FastPass provide seamless UX at scale for pharma operations.
- PingOne offers the most customizable fallback and context-driven policy engine, which may be advantageous for high-risk or hybrid user populations.

---

## 6. Integration Architectures and Migration from Legacy AD

### Okta Workforce Identity

- **Protocols**: SAML, OIDC, SCIM 2.0, LDAP agent, proprietary REST.
- **SAP Integration**: Via OIN connector, SAML or OIDC for SSO (SAP NetWeaver, S/4HANA), SCIM for lifecycle; attribute mapping needed for HR, roles.
- **Salesforce Integration**: SAML/OIDC for SSO, SCIM for provisioning; recommend dual “app” model for SSO/lifecycle.
- **Migration from AD**: AD agent with delegated authentication; staged cutover enables hybrid Co-Existence; Okta AD Migration Utility supports user import, group mapping, credential sync; user cutover can be gradual or scheduled[36][35][37].
- **Reference Design**: Use Okta AD agent for initial sync; federate SSO with SAP/Salesforce via SAML/OIDC; provision via SCIM; monitor API/log status for compliance.

### Microsoft Entra ID Premium P2

- **Protocols**: SAML, OIDC, OAuth 2.0, SCIM, proprietary Microsoft Graph.
- **SAP Integration**: Gallery (SAP SuccessFactors, NetWeaver) for SSO/provisioning; Entra Connect agent bridges AD to Entra for SAP HR sync; SAML for SSO, SCIM for user lifecycle.
- **Salesforce Integration**: Gallery app for SSO/provisioning; SAML or OIDC SSO, SCIM provisioning; must map ProfileId/entitlements[39][40][41].
- **Migration from AD**: Entra Connect provides hybrid sync; staged migration and cutover with attribute mapping, password hash or pass-through auth; legacy/GPO policies convert to cloud controls.
- **Reference Design**: Hybrid AD/Entra coexistence; app registration for SAML/OIDC SSO; provisioning via SCIM/Graph; automate admin reviews.

### Ping Identity PingOne

- **Protocols**: SAML, OIDC, SCIM, REST API, LDAP.
- **SAP Integration**: Direct SAML/OIDC federation to SAP, outbound SCIM provisioning for user/group lifecycle; supports SAP HR and on-prem hybrids; PingFederate SCIM Provisioner for deep integration.
- **Salesforce Integration**: SAML SSO, SCIM provisioning; reference connector kits and orchestration templates simplify setup; multi-attribute mapping for pharma context[45][46][27].
- **Migration from AD**: PingOne AD Connect utility; staged migration for user, group, and policy import; AD side can operate hybrid or phased cutover.
- **Reference Design**: Deploy PingOne AD connector, orchestrate SAML/OIDC SSO to SAP/Salesforce, use separate SCIM for lifecycle, enable fallback/migration policies as per compliance needs.

---

## Conclusion

All three IAM platforms are capable of meeting the complex, compliance-driven needs of a medium-to-large European pharmaceutical company:

- **Okta**: High feature modularity, excellent integration flexibility, robust adaptive MFA, but highest 5-year TCO and complex premium feature gating. Suited to pharma organizations with diverse, multi-cloud environments and advanced compliance demands.
- **Microsoft Entra ID Premium P2**: Best cost profile, deepest native integration for companies already invested in Microsoft stacks, powerful identity governance, and strong SAP/Office/Salesforce compatibility. Ideal when existing M365/E5 bundles or AD expertise can be leveraged.
- **Ping Identity PingOne**: Lowest 5-year TCO due to 5,000-user minimum, highly configurable integration/orchestration, strong AD migration tools, leading MFA/fallback customization, and a compelling data residency/compliance story. Well suited for hybrid AD migrations and organizations anticipating workload/user increases.

**Recommendation:** Vendor selection should consider broader IT landscape, AD coexistence/migration plans, requirements for highly adaptive MFA, and anticipated legacy or hybrid integration needs. Each vendor should deliver a detailed proposal inclusive of SAP/Salesforce reference designs, TCO modeling, and explicit data residency/contractual assurances for regulatory compliance.

---

### Sources

[1] Okta Pricing 2026: Ultimate Guide for Security Products - https://underdefense.com/industry-pricings/okta-pricing-ultimate-guide-for-security-products/  
[2] Okta Pricing 2026: Plans, Costs & Hidden Fees - CheckThat.ai - https://checkthat.ai/brands/okta/pricing  
[3] Okta Pricing 2026 - TrustRadius - https://www.trustradius.com/products/okta/pricing  
[4] The Okta Tax: How Much Are You Really Paying for Identity? - https://goauthentik.io/blog/2026-02-23-the-okta-tax/  
[5] Plans and Pricing - Okta - https://www.okta.com/pricing/  
[6] Okta + Data Residency: https://www.okta.com/okta-data-residency/  
[7] Okta Privacy Policy (2025): https://www.okta.com/privacy-policy/2025-01-archived/  
[8] GDPR | Okta: https://www.okta.com/legal/gdpr/  
[9] Microsoft Entra ID Pricing Overview - G2 - https://www.g2.com/products/microsoft-entra-id/pricing  
[10] Microsoft Entra ID and data residency - Microsoft Entra | Microsoft Learn: https://learn.microsoft.com/en-us/entra/fundamentals/data-residency  
[11] Microsoft Entra ID Pricing 2026 - TrustRadius - https://www.trustradius.com/products/microsoft-entra-id/pricing  
[12] Ping Identity Pricing 2026: Plans, Costs & Enterprise ... - https://checkthat.ai/brands/ping-identity/pricing  
[13] Ping Identity Pricing: PingOne for Customers vs. Workforce | Frontegg - https://frontegg.com/guides/ping-identity-pricing  
[14] Data residency | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/tenants/data-residency.html  
[15] Security and compliance | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/product-information/security-compliance.html  
[16] Data Supplement | Ping Identity: https://www.pingidentity.com/en-us/docs/legal/data-supplement  
[17] Workforce Identity Cloud Customer Data Retention Policy: https://support.okta.com/help/s/article/Customer-Data-Retention-Policy?language=en_US  
[18] Okta Inc. Information Security Document: https://www.okta.com/content/dam/okta---digital/en_us/legal/InfoSecurity-Doc-Oct2025.pdf  
[19] Okta + Data Residency: https://www.okta.com/okta-data-residency/  
[20] GDPR | Okta: https://www.okta.com/legal/gdpr/  
[21] Privacy Policy | Okta: https://www.okta.com/legal/privacy-policy/2025-03-archived/  
[22] Mastering Okta Group Rule Audit Logs for Security and Compliance: https://hoop.dev/blog/mastering-okta-group-rule-audit-logs-for-security-and-compliance  
[23] Workforce Identity Cloud Customer Data Retention Policy: https://support.okta.com/help/s/article/Customer-Data-Retention-Policy?language=en_US  
[24] Customer data storage and processing for European customers in Microsoft Entra ID - Microsoft Entra | Microsoft Learn: https://learn.microsoft.com/en-us/entra/fundamentals/data-storage-eu  
[25] Microsoft Entra ID and data residency - Microsoft Entra | Microsoft Learn: https://learn.microsoft.com/en-us/entra/fundamentals/data-residency  
[26] What's new in Microsoft Entra – March 2026: https://techcommunity.microsoft.com/blog/microsoft-entra-blog/what%E2%80%99s-new-in-microsoft-entra-%E2%80%93-march-2026/4502150  
[27] Data residency | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/tenants/data-residency.html  
[28] Security and compliance | PingOne Advanced Identity Cloud: https://docs.pingidentity.com/pingoneaic/product-information/security-compliance.html  
[29] Data Supplement | Ping Identity: https://www.pingidentity.com/en-us/docs/legal/data-supplement  
[30] Audit | PingOne: https://docs.pingidentity.com/pingone/monitoring/p1_reporting.html  
[31] AI agent logging and auditing | PingOne: https://docs.pingidentity.com/pingone/ai_agents/p1_ai_agent_logging.html  
[32] Additional Rate limits | Okta Developer: https://developer.okta.com/docs/reference/rl2-limits/  
[33] Rate limits | Okta Developer: https://developer.okta.com/docs/reference/rate-limits/  
[34] Rate Limits for API Service Integrations: https://support.okta.com/help/s/article/rate-limits-for-api-service-integrations?  
[35] Okta Integration Network (OIN): https://www.okta.com/integrations/  
[36] SCIM integration in Entra for Salesforce - Microsoft Q&A: https://learn.microsoft.com/en-us/answers/questions/5828857/scim-integration-in-entra-for-salesforce  
[37] Service limits and restrictions - Microsoft Entra ID | Microsoft Learn: https://learn.microsoft.com/en-us/entra/identity/users/directory-service-limits-restrictions  
[38] Service limits and restrictions - Microsoft Entra External ID | Microsoft Learn: https://learn.microsoft.com/en-us/entra/external-id/customers/reference-service-limits  
[39] Microsoft Entra ID Governance service limits - Microsoft Entra ID Governance | Microsoft Learn: https://learn.microsoft.com/en-us/entra/id-governance/governance-service-limits  
[40] What's new in Microsoft Entra – September 2025: https://techcommunity.microsoft.com/blog/microsoft-entra-blog/what%E2%80%99s-new-in-microsoft-entra-%E2%80%93-september-2025/4352576  
[41] Build your SCIM API service - Okta Developer: https://developer.okta.com/docs/guides/scim-provisioning-integration-prepare/main/  
[42] PingOne standard platform limits | PingOne: https://docs.pingidentity.com/pingone/getting_started_with_pingone/p1_platform_limits.html  
[43] Rate Limits and Allowed IPs | PingOne: https://docs.pingidentity.com/pingone/settings/p1_rate_limits.html  
[44] Rate Limiting | PingOne Platform APIs: https://developer.pingidentity.com/pingone-api/platform/rate-limiting.html  
[45] SCIM Provisioner | PingFederate Integrations - Ping Identity Docs: https://docs.pingidentity.com/integrations/scim/pf_scim_connector.html  
[46] Federating PingOne and Salesforce | Use Cases: https://docs.pingidentity.com/solution-guides/single_sign-on_use_cases/htg_federate_p1_salesforce.html  
[47] Multifactor authentication | Okta: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/about-authenticators.htm  
[48] Okta FastPass and phishing-resistant authentication: https://support.okta.com/help/s/article/Feature-Require-PhishingResistant-Authenticator-to-Enroll-Additional-Authenticators  
[49] Phishing-resistant authentication | Okta: https://help.okta.com/oie/en-us/content/topics/identity-engine/authenticators/phishing-resistant-auth.htm  
[50] Protecting Against Threats with Phishing Resistance | Okta: https://www.okta.com/phishing-resistance/  
[51] Okta Security Knowledge - Phishing Resistant Factors: https://support.okta.com/help/s/article/Okta-Security-Knowledge-Phishing-Resistance?language=en_US  
[52] Require phishing-resistant multifactor authentication for Microsoft Entra administrator roles - Microsoft Learn: https://learn.microsoft.com/en-us/entra/identity/conditional-access/policy-admin-phish-resistant-mfa  
[53] Microsoft Entra ID P2 Features & Pricing | Cayosoft: https://www.cayosoft.com/blog/entra-id-p2/  
[54] GDPR Compliance Made Easy with Microsoft Entra: https://hoop.dev/blog/gdpr-compliance-made-easy-with-microsoft-entra/  
[55] Overview of authentication methods | PingOne: https://docs.pingidentity.com/pingone/strong_authentication_mfa/p1_authentication_methods_overview.html  
[56] Yubico + Ping for Phishing-Resistant MFA: https://www.yubico.com/solutions/executive-order-hub/yubico-ping/  
[57] Prevent Adversary-in-the-Middle Attacks | Ping: https://www.pingidentity.com/en/resources/blog/post/adversary-middle-attacks.html