# Payment Terminal Deployment for 25-location QSR Chain: Verifone Carbon 8 vs Clover Station Pro vs PAX A920 Pro with Oracle MICROS Simphony in Quebec and New Brunswick

## Executive Summary

This report offers a scenario-specific, actionable comparison of the Verifone Carbon 8, Clover Station Pro, and PAX A920 Pro payment terminals for a 25-location quick-service restaurant (QSR) chain expanding across Quebec and New Brunswick. It provides device-level recommendations for both fixed-lane and mobile/table service deployments with Oracle MICROS Simphony—specifying preferred hardware configurations, exact middleware/integration models, and confirming device and acquirer certification status per official documentation.

Operational benchmarks (NFC/contactless payment limits, transaction speeds), regulatory compliance (Bill 96 French language enforcement, new Canadian payment rules 2024–26), and pre-go-live compliance validation checklists (UI/admin/receipt French and accessibility) are included—backed by authoritative payment network, acquirer, government, and Oracle/vendor sources. Total Cost of Ownership (TCO) modeling is presented with per-location and chain-wide breakdowns and explicit modeling input requirements. Areas of unresolved deployment risk or missing evidence are meticulously flagged.

---

## 1. Hardware and Integration Overview: Device-by-Scenario Analysis with Oracle MICROS Simphony

### 1.1. Verifone Carbon 8

#### Fixed-lane Deployment

- **Hardware Model:** Verifone Carbon 8 countertop terminal (PCI PTS POI certified until April 2027[1]).
- **Integration Method:** The standard integration path for Verifone hardware with Oracle MICROS Simphony in Canada is via the Oracle Payment Interface (OPI), or (when available) certified acquirer middleware/plugins such as Moneris Payment Plugin.
- **Certification and Support:**
    - Carbon 8 is PCI PTS certified; EMV capable; not explicitly listed by Interac or Canadian acquirers as certified for Interac processing with Simphony as of 2026. The Verifone Engage family is referenced in Oracle/Moneris documentation, but Carbon 8 is not [1][2][3][4][5].
    - Moneris Payment Plugin for Oracle Simphony only explicitly lists P400 as supported; Carbon 8 is not named, and Oracle’s official hardware compatibility lists do not reference it.
- **Operational Risk:** High risk due to lack of public, explicit acquirer certification for Carbon 8 with Simphony/Interac in Canada; project-level verification with both Verifone and the selected payment acquirer is mandatory prior to deployment [2][3][6].

#### Mobile/Table Service Deployment

- **Hardware Model:** Verifone Carbon 8 can be used as a mobile device, but its form factor is bulkier, and battery performance is subpar compared to devices designed for true mobility.
- **Integration Method:** Same as above—OPI or custom middleware if available, but with the same device-level certification caveats.
- **Operational Risk:** Operational and support risks further increase due to likely lack of optimized pay-at-table and tip adjustment workflows; limited market evidence or support for Carbon 8 in Canadian QSR mobile deployments.

#### Summary Table

| Scenario       | Preferred Model         | Integration     | Device Certification | Operational Caveat     |
|----------------|------------------------|-----------------|---------------------|------------------------|
| Fixed-lane     | Carbon 8 Countertop    | OPI/Plugin      | Not confirmed       | High deployment risk   |
| Mobile/Table   | Carbon 8 (not optimal) | OPI/Plugin      | Not confirmed       | High risk, not mobile-optimized |

---

### 1.2. Clover Station Pro

#### Fixed-lane Deployment

- **Hardware Model:** Clover Station Pro (EMV/NFC capable, PCI PTS v5.x) [7].
- **Integration Method:** No officially documented, certified integration with Oracle MICROS Simphony for the Canadian market as of 2026. Neither OPI, SecureTablePay, nor acquirer plugins are available or supported for Clover devices with Simphony or Interac in Canada [8][9][10][11][12][13][14].
- **Certification and Support:**
    - PCI PTS compliant; Interac capable in standalone or with Clover software only; lacks Simphony OPI/acquirer plugin certification.
    - Region-specific SDK restrictions: tip adjustment, Interac refunds, and offline/manual operations are not supported in the Canadian version of Clover SDK [10].
- **Operational Risk:** Extreme—deployment with Simphony in Canadian QSRs is unsupported; Interac integration not possible; fails both technical and compliance expectations [8][9][10].

#### Mobile/Table Service Deployment

- **Hardware Model:** Clover Flex could be considered for mobile; however, same lack of OPI OR certified integration applies [8][10][11].
- **Integration Method:** Not available for Simphony or through supported middleware in Canada.
- **Operational Risk:** Deployment is unadvisable due to a total lack of Oracle Simphony/acquirer certification and critical workflow limitations.

#### Summary Table

| Scenario       | Preferred Model         | Integration     | Device Certification | Operational Caveat     |
|----------------|------------------------|-----------------|---------------------|------------------------|
| Fixed-lane     | Station Pro            | Not supported   | Not with Simphony   | Unsupported, do not use|
| Mobile/Table   | Flex                    | Not supported   | Not with Simphony   | Unsupported, do not use|

---

### 1.3. PAX A920 Pro

#### Fixed-lane Deployment

- **Hardware Model:** PAX A920 Pro (PCI PTS 6.x/7.x certified to July 2027; Moneris Go and PayFacto branding in Canada) [15][16][17].
- **Integration Method:** Integration is achieved through:
    - Oracle Payment Interface (OPI): Acquirer-neutral approach for Simphony [4][9][14][15].
    - Certified acquirer plugin (e.g., Moneris Payment Plugin): Explicitly supports Moneris Go (A920/A920 Pro) for in-store payment flows [16].
    - PayFacto SecureTablePay: Used for direct pay-at-table/table management, supports Oracle Simphony [13][15].
- **Certification and Support:**
    - Widely referenced as supported for both in-store (counter) and mobile deployments by Moneris, PayFacto, PAX, and Oracle integration guides [13][15][16][17][18][19].
    - Confirmed PCI/EMV/interac status; vendor and acquirer documentation recognize the A920 Pro as "Moneris Go" for Canadian markets [16][17].
- **Operational Caveat:** Confirm with payment processor/acquirer on device model SKUs, as public Oracle/acquirer lists may lag actual certification.

#### Mobile/Table Service Deployment

- **Hardware Model:** PAX A920 Pro (with base/charging cradle as mobile accessory is available).
- **Integration Method:** Identical to above, but typically paired with PayFacto SecureTablePay middleware for pay-at-table (includes tip, check closing, table management) [13][15].
- **Certification and Support:** Supported through certified integration as above; best-in-class mix of hardware form factor and software integration.
- **Operational Caveat:** Ensure middleware configuration, regular firmware patching, and processor confirmation for pay-at-table and tip flows.

#### Summary Table

| Scenario       | Preferred Model         | Integration                 | Device Certification | Operational Caveat           |
|----------------|------------------------|-----------------------------|---------------------|------------------------------|
| Fixed-lane     | A920 Pro with cradle   | OPI / Moneris plugin        | Certified           | Confirm SKU at install       |
| Mobile/Table   | A920 Pro               | OPI / SecureTablePay        | Certified           | Confirm tip/table config     |

---

## 2. Operational Payment Network Parameters for Canada

### 2.1. Contactless/NFC Tap Payment Limits (2026)

- **Interac Debit:** $250 CAD per tap, after which chip & PIN or device authentication required [20][21][22][23][24].
- **Visa, Mastercard, Amex:** Also $250 CAD per tap; transactions above this require chip/PIN or mobile device authentication [22][23][24][25].
- These limits are set by the payment networks (not hardware) and universally applicable.
- Device must support EMV L1/L2 and NFC—All three models do; however, Clover and Verifone Carbon 8’s ability to process Interac within Simphony is not certified for Canada.

### 2.2. Transaction Speed Benchmarks

- **Visa:** Advertises contactless as completing in as little as 0.5 seconds, 7X faster than chip/PIN [22][25]. No SLA or regulatory mandate exists for QSR speed; real-world results vary by acquirer and network connectivity.
- **Mastercard:** Claims up to 10X faster than chip/PIN—also with no formal QSR benchmark [24].
- **Interac/Amex:** No explicit QSR timing requirement published; promote fast, tap-first flows [21][24].

### 2.3. Compliance Dates and Regulatory Change Windows

- **Contactless/Tap Limit Updates:** All major Canadian card brands as of 2024–2026 set current $250 limit [20][23][24][25].
- **Canadian Payments Act Amendments:** Compliance for new PSPs (merchant processors) by November 15, 2024 (registration); operational compliance by September 8, 2025 [26].
- **Code of Conduct for Payment Card Industry:** Merchant cancellation, notification, and disclosure rules effective April 30, 2025 [27].
- **Quebec Bill 96:** Key French language compliance deadlines entered force July 1, 2025, affecting all payment and UI/admin/receipt workflows [28][29].

---

## 3. Bill 96 (Quebec) Compliance: Pre-Go-Live Validation Checklist

### 3.1. Device-Level and Workflow Obligations

Bill 96 presents strict requirements for French-language usage in all customer- and employee-facing digital systems in Quebec. Failure to comply can result in regulatory action or fines.

#### Validation Steps:

- **Customer UI:** All prompts, approval/decline messages, tip suggestion screens, privacy/legal info must default to French and be of equal prominence and completeness as English flows. Test every customer-facing screen for French translation, correct context, and technical accuracy [28][29][30].
- **Admin/Back Office UI:** All menus, settings, and functions accessible to staff/management in Quebec must be available in French. Some device brands default to English in admin modes—these must be remediated or supplemented with French documentation/workarounds.
- **Printed/Email Receipts:** Default language must be French; all transactional detail, including merchant legal entity, return/refund policy, payment method, taxes, and tip must be in accurate Quebec French, with English available only as a secondary option [28][29][30].
- **Training & Manuals:** All staff-facing materials, quick start guides, and technical support resources must be provided in French; workflows for support/escalation must operate bilingually [29].
- **Accessibility:** French language flows must be maintained regardless of accessibility or voice-over settings.
- **Audit and Validation:** Pre-go-live, conduct:
    - Human translation review (not automated only; compliance requires professional/peer-checked translations).
    - Step-by-step walkthrough, capturing screenshots/video of each user/admin/receipt flow in French.
    - Documentation and issue log for any non-compliant flows; remediation plan for corrections.
    - Retain audit artifacts for OQLF/government inspection [28][29][30].
- **Ongoing Review:** Schedule annual or semi-annual audits to ensure updated firmware/app versions don’t revert or erase French translations or settings.

### 3.2. Device-Specific Bill 96 Readiness

- **Verifone Carbon 8:** No vendor-validated Bill 96 French workflow. Only basic language toggling described. Requires manual validation, professional translation, and possible custom configuration [1][30].
- **Clover Station Pro:** Bilingual user interface, but Bill 96 compliance not vendor-certified; absence of default French in admin/receipts confirmed by public evidence. Manual review and remediation mandatory [8][14][29].
- **PAX A920 Pro:** Full French UI selectable, but Bill 96-level interface/receipt validation unconfirmed. Each OS/app/receipt template must be reviewed and adjusted as necessary (especially for PayFacto and Moneris apps). Document and audit all flows prior to go-live [15][29][30].

---

## 4. Total Cost of Ownership (TCO): Line-Item and Modeling Inputs

### 4.1. Key Cost Components

- **Hardware Unit Cost:** Device price per location (see line-item below).
- **Payment Terminal Accessories:** Stands, cradles, printers, customer-facing screens (as needed for scenario).
- **Processor/Acquirer Fees:** Blended rates: % of transaction or flat fee per payment, specific to acquirer agreements for Interac/credit brands.
- **Integration Software Licenses:** Fees for OPI, Middleware (SecureTablePay, Moneris plugin), and associated server costs.
- **Support/Warranty:** 1–3 year hardware plus optional extended (replacement, break/fix, remote management).
- **Replacement Cycle:** 3–5 years recommended for terminal refresh based on durability, warranty limits, and QSR environment.
- **Compliance/Translation Costs:** Professional French-language translation, compliance audit retainers, and annual/bi-annual workflow reviews (especially in Quebec).

### 4.2. Per-Location, Chain-Wide Breakdown (Sample Input Table)

| Cost Element            | Carbon 8 | Clover Station Pro | PAX A920 Pro                | Notes                                            |
|------------------------|----------|-------------------|-----------------------------|--------------------------------------------------|
| Hardware unit price    | ~$800    | $1,800–2,000      | $700–800                    | per device, bulk may reduce by 10%               |
| Terminal accessories   | $150     | $200              | $200                        | cradles/stands/customer screens as required      |
| OPI/Middleware license | $200     | Not Available     | $200–$300                   | annual license/maintenance (per location/server) |
| Processor/acquirer fee | 2–2.9%+$0.10 | 2.3–2.6%+$0.10 | 2–2.5%+$0.10 + flat Interac | acquirer-specific, Interac is typically flat fee |
| Support/warranty       | $100/yr  | $150/yr           | $100/yr                     | annual, per device                               |
| Compliance/audit       | $500     | $600              | $500                        | translation & audit, per go-live/re-audit        |
| Device replacement     | $160/yr  | $400/yr           | $160/yr                     | assuming 5-yr lifecycle                          |

- **Total 5-year TCO per location** (PAX A920 Pro estimate):  
  - Device and accessories: $800 + $200 = $1,000  
  - Middleware/license: $250 × 5 = $1,250  
  - Acquirer fees: Strongly volume-dependent (see modeling below)  
  - Support: $100 × 5 = $500  
  - Compliance/Audit: $500 initial + $250 periodic = $750  
  - Replacement: $160 × 5 = $800  
  - **Subtotal per location (excl. fees): $4,300**  
  - **For 25 locations:** $107,500 + transaction/processor fees.

- **To model processor/acquirer fees:**
    - Average monthly gross card transaction volume (needed input)
    - Average split between Interac debit and credit (by card brand)
    - Average ticket size
    - Estimated acquirer flat and % fee for each type (obtain from merchant agreement)
    - Multiply monthly/annual value for all locations, include in multi-year projections

#### Required Data Inputs for Scenarios

- Number of devices per location (counter/table service requirements)
- Estimated transaction counts per device per day/week/month (by payment type)
- Merchant acquirer rates by brand
- Expected device lifecycle (years)
- Number of compliance events or re-audits (per year)
- Middleware/integration contract terms and costs

---

## 5. Recommendations and Risks by Scenario

### 5.1. Fixed Lane/Counter Service

- **Preferred Device:** PAX A920 Pro (counter base).
- **Integration:** OPI or Moneris Payment Plugin (for Moneris Go/A920 Pro); confirmed support for Interac, Visa, Mastercard, Amex [15][16][17][18][19].
- **Caveats:** Ensure device SKU (A920 Pro) is referenced in the acquirer certification; insist on in-writing confirmation from processor before purchase.
- **Other Devices:** Verifone Carbon 8 presents high risk: no confirmed Interac/Simphony support in Canada. Clover Station Pro is not supported with Simphony/Interac in Canada for QSR.

### 5.2. Mobile/Table Service

- **Preferred Device:** PAX A920 Pro.
- **Integration:** SecureTablePay (PayFacto) middleware for pay-at-table, OPI integration for Simphony [13][15].
- **Caveats:** Validate table/tip flows operationally, ensure French workflows compliant/Bill 96 tested before rollout.
- **Other Devices:** Neither Carbon 8 nor Clover Station Pro are practically usable for mobile/table Simphony workflows in a Canadian QSR.

### 5.3. Compliance and Go-Live: Critical Steps for Quebec

- **All Devices:** Manual French workflow validation and documentation is essential.
- **PAX A920 Pro:** Audit SecureTablePay and Moneris app French interfaces, edit all receipt templates for Bill 96.
- **Direct Vendor/Acquirer Confirmation:** Required for every device and merchant chain installation.

---

## 6. Kitchen Display System (KDS) Considerations

None of the payment terminals evaluated (Verifone Carbon 8, Clover Station Pro, PAX A920 Pro) can act as a Simphony KDS endpoint. Oracle-certified KDS hardware (Workstation 6/8, KDS Controller 210/166, etc.) must be procured for kitchen integration needs [31].

---

## 7. Outstanding Risks, Unresolved Gaps, and Action Items

- **Carbon 8 and Clover:** Unresolved Simphony/Interac certification for Canada. Do not deploy without written confirmation if local circumstances require.
- **Clover Station Pro:** Total lack of integration/support with Oracle Simphony/Interac in Canada.
- **Bill 96:** No major brand has complete vendor-certified, Quebec-specific French compliance for every admin/user/receipt workflow—manual translation and periodic audit required [28][29][30].
- **TCO Modeling:** Actual per-location costs may deviate based on negotiated acquirer discounts, transaction mix, and compliance overhead.
- **Device Security:** On PAX A920 Pro, apply patches for any identified Android/firmware CVEs, especially for large deployments.

---

## 8. Conclusion

For a 25-location QSR rollout in Quebec and New Brunswick using Oracle MICROS Simphony, only the PAX A920 Pro is recommended for both fixed-lane and mobile/table service deployment. Integrate through OPI (Oracle) or an acquirer-certified plugin (Moneris Payment Plugin or SecureTablePay) for full support of Canadian card brands—including Interac—with tailored configuration for each deployment scenario. Bill 96 pre-go-live compliance requires comprehensive French language review of every device/app/receipt workflow; success depends on professional translation, documenting each step, and annual audits. Critical input data (transaction volumes, acquirer rates) is necessary to finalize TCO projections.

Do not deploy Verifone Carbon 8 or Clover Station Pro with Oracle Simphony in Canada until public or written confirmation is available from the device vendor and the designated acquirer confirming Canadian Simphony/Interac certification and compliance.

---

### Sources

[1] PCI-Approved PTS Device List: Verifone Carbon 8, PCI SSC  
https://www.pcisecuritystandards.org/popups/p2pe_app_device.php?reference=2024-00154.128  
[2] Moneris Payment Plugin for Oracle Simphony: Supported Devices  
https://support.moneris.com/article/moneris-payment-plugin-for-oracle-simphony-installing-and-47710  
[3] Oracle Payment Interface (OPI) Global Overview  
https://docs.oracle.com/en/industries/food-beverage/simphony/19.7/simcm/t_shared_services_overview_opi.htm  
[4] Simphony OPI Payment Driver Installation Guide  
https://docs.oracle.com/cd/E79534_01/docs/E85864.pdf  
[5] Verifone Payment Integration Guide (Oracle)  
https://docs.oracle.com/cd/E69785_01/frs/pdf/1101/VeriFone%20Payment%20Integration%20Guide.pdf  
[6] Verifone Carbon 8 Product Specifications  
https://merchantrolls.com/product/verifone-carbon-8/  
[7] Clover Devices: Technical Specifications  
https://docs.clover.com/dev/docs/clover-devices-tech-specs  
[8] Canada merchants - Clover Developer  
https://docs.clover.com/dev/docs/canadian-merchants  
[9] Payments integration options - Clover Developer  
https://docs.clover.com/dev/docs/paas-integration-options  
[10] Clover Payments SDK—Regional Feature Matrix  
https://docs.clover.com/dev/docs/canadian-merchants  
[11] Simphony Supported POS Client Devices  
https://docs.oracle.com/cd/E76065_01/doc.29/e69880/c_supportandcompatibility_posclientdevices.htm  
[12] Integrate with Simphony Point of Sale—Oracle  
https://www.oracle.com/ca-en/food-beverage/restaurant-pos-systems/pos-integrations/partners/  
[13] PAX A920Pro—Official Device Page  
https://www.paxtechnology.com/a920pro  
[14] PayFacto SecureTablePay Installation Guide  
https://securetablepay.com/en/install-pax-a920  
[15] PAX A920/A920Pro Payment Terminal (Bluefin)  
https://www.bluefin.com/device/pax-a920/  
[16] Moneris Payment Plugin for Oracle Simphony  
https://support.moneris.com/article/moneris-payment-plugin-for-oracle-simphony-setting-up-47709  
[17] PAX Technology and Moneris Solutions Launch A920 in Canada  
https://www.pax.us/about/press-room/pax-technology-and-moneris-solutions-launch-a920-in-canada/  
[18] PayFacto—PAX A920 Support  
https://payfacto.com/en/pax-a920/  
[19] Moneris Partner Integrations  
https://www.moneris.com/en/partners/integrations  
[20] Interac—Accept Interac Payments  
https://www.interac.ca/en/payments/business/accept-interac-debit-payments/  
[21] Interac Debit Contactless Payments FAQ (2022)  
https://www.uni.ca/pdf/interac-debit-contactless-payments-FAQ-17062022.pdf  
[22] Visa Canada Contactless Payment Policy  
https://www.visa.ca/en_CA/pay-with-visa/featured-technologies/contactless-payments.html  
[23] Mastercard Contactless Payment Solutions Canada  
https://www.mastercard.com/ca/en/business/payments/commercial-payments/accept-payments/contactless-payments.html  
[24] American Express Contactless Payments (Canada)  
https://www.americanexpress.com/ca/en/services/ways-to-pay/contactless/  
[25] Visa Contactless Payment Resources—For Merchants  
https://www.visa.ca/en_CA/run-your-business/merchant-resources/contactless-payments.html  
[26] Criteria for registering payment service providers—Bank of Canada  
https://www.bankofcanada.ca/2024/10/criteria-for-registering-payment-service-providers/  
[27] Changes to the Code of Conduct for the Canadian payment card industry  
https://www.dlapiper.com/insights/publications/2024/12/changes-to-the-code-of-conduct-for-the-canadian-payment-card-industry  
[28] Bill 96 (Law 14) Official Text and Requirements—Quebec  
http://legisquebec.gouv.qc.ca/en/document/cs/C-11 (French language charter as amended)  
[29] Bill 96 Compliance Guidance—RWS  
https://www.rws.com/blog/bill-96-compliance-clock-is-ticking/  
[30] Bill 96: Checklist and Translation Needs for Hospitality Tech  
https://alexatranslations.com/blog/navigating-multilingual-compliance-in-2025-with-free-bill-96-checklist/  
[31] Oracle Kitchen Display System—Official Product Sheet  
https://www.oracle.com/a/ocom/docs/industries/hospitality/oracle-micros-kitchen-display-systems-ds.pdf