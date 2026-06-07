# Comparative Analysis: Verifone Carbon 8, Clover Station Pro, and PAX A920 Pro for 25-Location QSR Chain Expansion in Quebec and New Brunswick (with Canadian and Bill 96-specific Requirements)

## Executive Summary

This report presents an in-depth, evidence-based comparison of the Verifone Carbon 8, Clover Station Pro, and PAX A920 Pro payment terminals for deployment in a 25-location quick-service restaurant (QSR) chain expanding across Quebec and New Brunswick. The comparison focuses on (1) exact Oracle MICROS Simphony integration methods and certification, including plugins/middleware and Canadian-specific limitations (especially Interac, SDK support, and deprecated/EOL risks); (2) product lifecycle and roadmap; (3) Canadian five-year Total Cost of Ownership (TCO) including hardware, transaction fees, and support; (4) Bill 96 compliance and risk mitigation workflows for French/English; (5) offline payment functionality for Canadian payment types; (6) transaction speeds on Canadian networks; (7) contactless transaction limits; (8) tip adjustment/table service workflows; and (9) kitchen display system (KDS) integration with Simphony. Explicit areas of risk, documentation gaps, and Canadian/Quebec-specific constraints are identified throughout.

---

## 1. Oracle MICROS Simphony POS Integration and Middleware (Canada), Certification, and Limitations

### Verifone Carbon 8

- **Integration Methods & Certification:**  
  - No public or Oracle documentation confirms that the Carbon 8 is a certified integration partner for Oracle MICROS Simphony POS in Canada. The Verifone Secure Commerce Application (SCA) middleware supports Simphony integration for some Verifone devices, enabling semi-integrated EMV, Interac, credit, and tip workflows, but Carbon 8 is not explicitly listed in the official Oracle or Verifone Simphony partner/device registries for Canadian deployment[1][2][3].
  - No evidence of SKU-level certification for Carbon 8 for Interac/EMV with Oracle[3][4][5].
- **Plugins & Middleware:**  
  - SCA variants (Classic, Enterprise Direct-to-Processor, Lite) enable different connection types, with "Direct" versions designed for Canadian acquirer routing. SCA is the only documented Simphony middleware listed, but direct support for Carbon 8 is not addressed[3].
- **Canadian/Interac Limitations:**  
  - While Verifone SCA supports Interac across supported hardware, there is no evidence that Carbon 8 specifically is Interac-certified for Simphony deployments. This introduces a risk of workflow incompatibility or lack of Interac acceptance at the device level in Canada[3][5].
- **EOL/Deprecated Risk:**  
  - No explicit EOL notice for Carbon 8; some Verifone lines (e.g., VX, MX) are deprecated/EOL[6][7].

### Clover Station Pro

- **Integration Methods & Certification:**  
  - No documented out-of-the-box, certified integration between Clover Station Pro and Oracle MICROS Simphony POS for Canada. Any integration would require custom middleware or third-party development, with no turnkey or supported plugin available from Clover or Oracle[8][9].
- **Plugins & Middleware:**  
  - Canadian SDK support for Clover restricts payment flows. Interac and tip adjustment/post-auth are not supported via the Clover SDK; refunds require physical card presence and several actions are limited or unavailable[10].
- **Canadian/Interac Limitations:**  
  - Clover SDK in Canada does NOT support Interac offline, pre-auth, or post-authorization tip adjustment. Only credit cards and co-branded Interac can be used manually[10]. No certified or plug-and-play middleware is available for Simphony integration in the Canadian market.
- **EOL/Deprecated Risk:**  
  - Several legacy Clover devices (Gen 1) hit End of App Update status on May 15, 2026, though Station Pro itself is current[11].

### PAX A920 Pro

- **Integration Methods & Certification:**  
  - Partners such as Touché, SecureTablePay, and other middleware providers offer integration between Oracle MICROS Simphony and PAX Android SmartPOS devices (including A920 Pro), focused on dining, pay-at-table, and QSR environments[12][13][14].  
  - Direct Oracle Simphony API integration is available for partners who complete the Oracle PartnerNetwork validation process. While integration is demonstrated live, public documentation for Canadian plug-in version certification is not available; “compatibility” is cited, not “certification”[12][14].
- **Plugins & Middleware:**  
  - Middleware platforms (e.g., SecureTablePay) are often used to bridge Simphony and PAX devices but require processor/partner configuration, especially for Interac and tip support[13][14].
- **Canadian/Interac Limitations:**  
  - PAX A920 Pro supports Interac debit and Interac Flash tap via certified Canadian acquirers; however, tip adjustment for Interac and some delayed tips for debit are not universally supported and are processor-dependent[15][16].
- **EOL/Deprecated Risk:**  
  - A920 Pro is active and not scheduled for EOL. EOL/EOSL notices involve other discontinued PAX terminals[17].

#### **Integration Summary Table**

| Terminal         | Simphony Certification (Canada)     | Middleware/Plugin                | Interac Tip Support | Risk |
|------------------|-------------------------------------|----------------------------------|---------------------|------|
| Verifone Carbon 8| Unverified                         | SCA (semicertified, but not for Carbon 8) | Not confirmed      | High (no evidence) |
| Clover Station Pro| Not supported, no plugin           | No Canadian Simphony plugin      | Not supported      | Extreme (no Simphony/Interac) |
| PAX A920 Pro     | Compatible via partners (not certified public) | SecureTablePay, Oracle API     | Limited, processor-defined | Medium (custom validation) |

---

## 2. Product Lifecycle Status and Roadmap

### Verifone Carbon 8
- Carbon 8 is not on current End of Life lists; discontinued devices (VX, MX, Topaz) are, but Carbon 8 is still documented as supported and in normal lifecycle with device management options[6][7][18].
- No roadmap or sunset date is provided by Verifone in public sources; however, the device is not at risk of imminent EOL.

### Clover Station Pro
- Actively sold and supported, but several Gen 1 device models (not Station Pro) will lose app update support after May 2026[11][19].
- No deprecation is planned for Station Pro for the next two years. Monthly software/firmware updates continue[19].

### PAX A920 Pro
- The A920 Pro is current, part of PAX’s North American product strategy for 2025–2026, and not scheduled for end-of-life[17][20].
- New models (A920MAX, others) are introduced, but no phase-out notice for A920 Pro[20].

---

## 3. Comprehensive Canadian Pricing & Five-Year TCO

### Verifone Carbon 8
- Hardware pricing listed at ~$810 CAD per unit; detailed support contracts, multi-site/bulk QSR pricing, or processor/maintenance bundles are unpublished and require direct negotiation[21].
- Typical processor fees: 2.3%–2.9% + $0.10 (but set by acquiring bank, not device vendor)[22].
- No published 5-year total cost of ownership calculator. Hardware-only TCO for 25 devices: $20,250. Add support (variable), transaction fees (volume-based), and replacement (3-5 years cycle)[21][22].

### Clover Station Pro
- Hardware cost is $1,800–$2,000 CAD per unit[23][24][25]. Transaction fee: ~2.3–2.6% + $0.10. Monthly software: $14.95–$89.95; numerous hidden and recurring fees, adding $100–$200+ per month per location[23][24][25][26].
- 36–48 month contracts with early exit penalties. App marketplace, PCI, and compliance fees often unbundled[23][24][25][26].
- Estimated 5-year TCO: $7,000–$10,000 per device; chain-wide, $175,000–$250,000 (hardware, software, and contract fees, excluding merchant discounts)[23][24][25][26].

### PAX A920 Pro
- Hardware acquisition: $500–$800 CAD per terminal[27][28][29]. Rental options available[29].
- Processor fees: set by acquirer, not public; Interac is flat-fee per txn, credit is %/flat plus interchange[15].
- Support: 1-year warranty included, extended support not priced public[28]. Expected device lifespan: ~5 years.
- Estimated 5-year hardware TCO (25 units): $12,500–$20,000, exclusive of transaction processing, which is variable[27][28][29].

---

## 4. Bill 96 (Quebec) Compliance and Bilingual French/English Interface Validation

### Verifone Carbon 8
- No documentation confirms full Bill 96-compliant (French-default, admin/user/receipt) interface. Carbon 8 manuals describe language selection (French/English) but not Quebec French, nor OQLF-validated interface audits or workflows[30][31][32].
- Legal compliance requires human translation, validation for every touchpoint (UI, receipts, admin functions), and periodic audits. Fines for non-compliance up to $30,000[33][34][35].
- **Deployment risk:** No vendor-validated workflow for Bill 96. Manual validation and certified translation/audit are mandatory for QSR rollout in Quebec[33][34].

### Clover Station Pro
- Interface and customer flows can be toggled French/English, but no public evidence exists of Bill 96-level default French compliance, nor deep admin/config UI in French[10][36].
- Quebec law requires that ALL customer and employee-facing screens, contracts, and receipts are in French by default and equally functional. No public translation validation exists for Clover Station Pro[36][37][38].
- Fines and regulatory delays for non-compliance. Organization must perform training/testing, QA and, where necessary, obtain human-reviewed translation for every interface layer[36][37].

### PAX A920 Pro
- Device interface can be switched to French, but Quebec French Bill 96 default, human-checked interface and compliant receipts are not confirmed by PAX or any integration partner[21][39][40].
- Full compliance is a merchant responsibility; expert review, interface, and workflow testing needed before rollout. Staff/service support materials must also be available in French[33][35][39].

---

## 5. Offline Payments (Canadian EMV/Interac Debit)

### Verifone Carbon 8
- No evidence of offline (store-and-forward) capability for Interac debit on Carbon 8 (or, indeed, on any device in Canada due to Interac network/risk policies)[3][5][41].
- Offline card processing may be available for some credit transactions if processor supports it, but for Interac, online authorization is mandatory[41].
- **Operational risk:** Outages mean no Interac sales—business continuity is not supported for offline Interac.

### Clover Station Pro
- Offline payment (store-and-forward) is explicitly **not supported** for Canadian merchants—all Interac and card payments must be processed or authorized online[10][42].
- US Clover devices offer offline mode, but this is not available in Canada due to Interac and processor restrictions.

### PAX A920 Pro
- The terminal supports offline mode (batch upload), but offline Interac payments are not universally supported; this depends on processor policy and is generally NOT allowed for Interac in Canada[15][43].
- Offline credit card may be possible if the processor allows, but Interac always requires online authorization due to fraud risk[43].

---

## 6. Transaction Processing Speed Benchmarks (Canadian Networks)

- No device (Carbon 8, Clover Station Pro, or PAX A920 Pro) publishes Canadian-specific transaction timing benchmarks in seconds for Interac, Visa, Mastercard, or Amex.  
- All three are considered industry standard for fast QSR transaction flow with support for NFC, chip, and mobile wallets. Real-world speed will depend on processor/acquirer network infrastructure, not device hardware alone[44][45][46].
- Performance differences are negligible for modern tap/EMV cards in supported, current hardware; empirical testing is strongly recommended in pilot environments.

---

## 7. Contactless Tap (NFC) Payment Limits (Visa, Mastercard, Interac, Amex - Canada 2026)

- National tap limits (set by networks, not terminals):
    - **Visa/Mastercard/Amex:** $250 CAD[47][48][49][50].
    - **Interac Debit:** $100 CAD. (Has not moved to $250—for fraud/security, Interac keeps a lower limit[47][50]).
- Larger transactions automatically prompt chip & PIN insertion regardless of terminal.
- All three devices accept NFC/tap for major Canadian networks with these limits, provided processor/software supports them[47][50].

---

## 8. Tip Adjustment and Table Service Workflows

### Verifone Carbon 8
- Simphony POS supports "tip calculator" plugin and post-authorization tip adjustment for credit cards[51][52]. No device-specific evidence for closed check adjustment, Interac debit tip, or certified workflow with Carbon 8[3][5].
- Tip adjustment on debit (Interac) is not supported; only credit is supported post-auth.

### Clover Station Pro
- Canadian SDK does **not** support post-authorization or pre-auth tip adjustment for Interac debit (only at sale, if supported) and restricts several related workflows[10].
- Credit card tip adjustment is supported via front-end and dashboard on US SKUs, but Canadian workflows have limits or are blocked due to Interac specifics[10][53].
- MEV Connect (certified for Revenu Quebec) supports counter service only, table service is “in development”[54].

### PAX A920 Pro
- Credit cards: Tip adjustment, pay-at-table, and split bill are supported if application and processor allow[55][56][57]. Operational guides show how staff can adjust tips pre-batch/settlement for credit.
- Interac: Tip must be entered at payment—no delayed/post-auth tips or “add-on” after initial approval per Interac rules[15].
- Workflow: FUNC > Tip Menu > select transaction > Adjust > Confirm—works only for eligible payment types[55][56][57].

---

## 9. Kitchen Display System (KDS) Integration with Simphony

- **Oracle Simphony KDS** requires certified Oracle KDS endpoints (Express Station 400, KDS Controller 210/166, Oracle tablets, etc.)[58][59][60].
- None of Carbon 8, Clover Station Pro, or PAX A920 Pro are on the list of supported KDS clients for Oracle Simphony deployments in Canada.
- KDS functions (prep, display, order flow) must be provided via Oracle’s certified hardware; third-party payment terminals are not eligible as KDS endpoints[58][60].

---

## 10. Comparative Risk Overview and Missing Evidence

- Neither Verifone Carbon 8 nor Clover Station Pro offers direct, certified Simphony/Interac deployment evidence for Canada. PAX A920 Pro provides the strongest “field evidence” for compatibility (with caveat: not public Oracle certification), but all require direct pilot testing, integration validation, and robust Bill 96 review before final rollout.
- No device supports Interac offline, nor provides full Bill 96 interface/receipt validation—this is a critical compliance risk.
- No device supports use as an Oracle MICROS KDS endpoint—additional investment in Oracle hardware for kitchen display is required.
- Public Canadian 5-year TCO is only available as rough estimate; actual costs vary and need direct processor and hardware vendor negotiation.

---

## Conclusion and Deployment Recommendations

PAX A920 Pro is the most flexible and widely adopted device for Oracle MICROS Simphony integration in the Canadian QSR sector, with robust hardware, processor/operator partnerships, and evident market acceptance. However, final deployment in Quebec and New Brunswick should **not proceed** without:

- **Pilot integration**: Live restaurant pilot to validate all device/POS/payment/kitchen flows in a Canadian Simphony environment, especially Interac, tip, and kitchen order transfer.
- **Bill 96 compliance audit**: Full translation and OQLF-reviewed workflow validation for every customer, staff, and admin-facing UI and all receipts in Quebec.
- **Processor and middleware confirmation**: Written certification from processor/integrator that required payment methods—including Interac debit/credit, tap, and tip adjustment workflows—are supported on the deployed configuration.
- **Offline payment policy**: Business process and training for network outages, as offline Interac is not feasible.
- **Kitchen hardware investment**: Use certified Oracle KDS endpoints for kitchen display integration; payment terminals cannot substitute as KDS clients.

The Verifone Carbon 8 and Clover Station Pro both present critical integration, certification, or compliance gaps for Canadian QSR deployment on Simphony with Interac. Only a combination of validated device integration, certified translation/legal compliance, and on-the-ground piloting will ensure a compliant, future-proof QSR infrastructure for Quebec and New Brunswick growth.

---

## Sources

1. [Integrate with Simphony Point of Sale | Oracle](https://www.oracle.com/food-beverage/restaurant-pos-systems/pos-integrations/partners/)
2. [Configuration Guide - Oracle Help Center (PDF)](https://docs.oracle.com/cd/E66669_01/doc.28/e66799.pdf)
3. [Overview - Verifone Documentation (PDF)](https://verifone.cloud/print/pdf/node/19801)
4. [Verifone Carbon 8 | Smart POS Terminal for Businesses - Merchantrolls](https://merchantrolls.com/product/verifone-carbon-8/?srsltid=AfmBOop_74u04khzMppP4t7dZ5oDpvZhWyYXEH4EVePVc2_yU-9suxxX)
5. [Verifone Documentation | Verifone Developer Portal](https://verifone.cloud/)
6. [Verifone EOL Notices](https://bluefin.my.site.com/knowledgebase/s/article/Verifone-EOL-Notices)
7. [Verifone Regulatory and Compliance Sheets (PDF)](https://verifone.cloud/print/pdf/node/14861)
8. [Clover POS Cost & Pricing in 2025 – Limelight Payments](https://shop.limelightpayments.com/blog/clover-pos-cost-pricing-2025-hardware-clover-fees-processing-fees?srsltid=AfmBOopfWsssvaBMotMmXQgDnaUG3UNC-DgrVDkXWOqarUIcDDuBHmc_)
9. [Clover Integration Services](https://www.clover.com/pos/integration-services?srsltid=AfmBOopmsOXFuDdY5uJUvUOFld5_MRNadLuuINzOV3PVkmVqdCmF092D)
10. [Clover Documentation: Canadian Merchants](https://docs.clover.com/dev/docs/canadian-merchants)
11. [Device lifecycle and support – Clover](https://docs.clover.com/dev/docs/device-lifecycle-and-support)
12. [Touche deploys Oracle solution for F&B and Hospitality on PAX Android SmartPOS devices](https://www.paxglobal.com.hk/en/latest-news/touche-deploys-oracle-solution-for-fb-and-hospitality-clients-on-pax-android-smartpos-devices/)
13. [SecureTablePay: PAX integration with Oracle, POS, Canadian Hospitality](https://www.securetablepay.com/en/partners)
14. [Hospitality PMS and POS Integrations | Oracle Canada](https://www.oracle.com/ca-en/hospitality/pms-pos-integration-partners/)
15. [Interac on Integrated Terminals | Amilia Help Center](https://help.amilia.com/en/articles/9580609-interac-on-integrated-terminals)
16. [PAX A920 PRO – Cardium](https://www.cardium.ca/pax-a920-pro)
17. [PAX End-of-Life Notices (2026) - PAX Technology Inc., North America](https://www.pax.us/support/end-of-life-notices/)
18. [Active Lifecycle and Key Indicators | Device Management – Verifone](https://verifone.cloud/docs/device-management/device-management-user-guide/devices/active-lifecycle-and-key-indicators)
19. [Announcements & Developer Communications – Clover](https://docs.clover.com/dev/docs/release-notes-announcements)
20. [2025 Product Roadmap, Exciting New Devices – PAX Technology Inc., North America](https://www.pax.us/2025-product-roadmap/)
21. [Verifone Carbon 8 POS Reader | eBay](https://www.ebay.com/itm/325518544621)
22. [Verifone Carbon POS Review and Profile](https://www.cardfellow.com/product-directory/pos-systems/verifone-carbon-review)
23. [CheckThat.ai: Clover Pricing 2026: Plans, Costs & Actual TCO](https://checkthat.ai/brands/clover/pricing)
24. [Tech.co: Clover POS Pricing 2026](https://tech.co/pos-system/clover-pos-pricing)
25. [SaaSworthy: Clover POS Pricing in 2026](https://www.saasworthy.com/blog/clover-pos-pricing)
26. [SleftPayments.com: Clover Overcharges $3,200/yr](https://www.sleftpayments.com/learning-hub/clover-pos-review-honest-pricing-2026)
27. [PAX-A920 Mobile Payment Terminal Available Through Bluefin](https://www.bluefin.com/device/pax-a920/)
28. [A920 Pro » Extraordinary Mobile POS Device – PAX Technology](https://www.pax.us/products/mobile/a920-pro/)
29. [Pax A920 Pro](https://paybotx.company.site/Pax-A920-Pro-p629364972)
30. [Language | In-Person Payments | Verifone Developer Portal](https://verifone.cloud/docs/in-person-payments/global-payment-application-gpa/verifone-global-payment-application-8)
31. [Multi-Language support – Verifone Cloud](https://verifone.cloud/print/pdf/node/1326)
32. [Verifone Carbon - Installation Guide - FCC Report](https://fcc.report/FCC-ID/B32CARBON10/3100655.pdf)
33. [Bill 96: How Quebec's French Law Shapes Localization](https://www.lingualinx.com/blog/bill-96-how-quebecs-french-law-shapes-localization)
34. [What Quebec's Bill 96 Means for Translation and Compliance](https://www.argosmultilingual.com/blog/what-quebecs-bill-96-means-for-translation-and-compliance)
35. [Reviewing Compliance With Bill 96 In 2026 | RVIA](https://www.rvia.org/news-insights/reviewing-compliance-bill-96-2026)
36. [Megalexis: Quebec Bill 96 (Law 14) Compliance](https://www.megalexis.com/en/quebec-bill-96-translation-compliance/)
37. [RWS: Bill 96 compliance clock is ticking](https://www.rws.com/blog/bill-96-compliance-clock-is-ticking/)
38. [ChemLinked: How Bill 96 Impacts Bilingual Labelling in Canada](https://cosmetic.chemlinked.com/expert-article/how-bill-96-impacts-bilingual-labelling-in-canada)
39. [Navigating Multilingual Compliance in 2025 [Bill 96 Checklist]](https://alexatranslations.com/blog/navigating-multilingual-compliance-in-2025-with-free-bill-96-checklist/)
40. [A920 Pro - Datacap Systems (PDF)](https://datacapsystems.com/wp-content/uploads/A920-Pro-Data-Sheet_May2021.pdf)
41. [Offline payments guidance | Verifone Documentation 2Checkout](https://verifone.cloud/docs/2checkout/Documentation/11Emails/Email_variables/Offline_payments_guidance)
42. [Clover: Set up offline payments](https://www.clover.com/help/set-up-offline-payments)
43. [How to process offline transactions on a standalone PAX A920 – VostroPay](https://support.vostropay.com.au/hc/en-au/articles/11242405458319-How-to-process-offline-transactions-on-a-standalone-PAX-A920-terminal)
44. [Quick Service Restaurant (QSR) POS & Software - Clover](https://www.clover.com/pos-solutions/quick-service-restaurant?srsltid=AfmBOor7OXqBhBMXYkVFKWdajJKrGjAIr_zb7iyhjf1-odDXqIKm0wEG)
45. [How to improve restaurant service times at your QSR - Clover Blog](https://blog.clover.com/how-to-improve-restaurant-service-times-at-your-qsr)
46. [PAX A920 Smart Terminal: Features, Setup, and Pricing (2026)](https://soireeinc.com/blog/everything-you-need-to-know-about-pax-a920-smart-terminal)
47. [Card Networks Up Canadian Contactless Transaction Limits To Limit Physical Contact – Digital Transactions](https://www.digitaltransactions.net/card-networks-up-canadian-contactless-transaction-limits-to-limit-physical-contact/)
48. [Moneris: Canadians quickly adjust to higher tap limits](https://www.moneris.com/en/blog/posts/insights-trends/canadians-quickly-adjust-to-higher-tap-limits-with-american-express-visa-and-mastercard)
49. [Ways to Pay | Pay the Way That’s Right For You | Amex CA](https://www.americanexpress.com/ca/en/services/ways-to-pay/)
50. [American Express Contactless Payments | American Express Canada](https://www.americanexpress.com/ca/en/services/ways-to-pay/contactless/)
51. [Oracle MICROS Simphony: add a tip with a Workstation 6 and a P400](https://www.youtube.com/watch?v=7t1MtrN2OwU)
52. [Team Service and Check Adjustment](https://docs.oracle.com/cd/F14820_01/doc.191/f15053/c_checks_team_service_adjustment.htm)
53. [Adjust a tip: Clover API](https://docs.clover.com/dev/docs/android-payments-api-tip-adjust)
54. [Clover MEV Connect: Quebec Sales Recording](https://mevconnect.ca/en/)
55. [How to Adjust Tips on a Pax A920 - Talus Pay](https://taluspay.zendesk.com/hc/en-us/articles/27031070834579-How-to-Adjust-Tips-on-a-Pax-A920)
56. [PAX A920 - Adjust A Tip - YouTube](https://www.youtube.com/watch?v=BUE549heG98)
57. [PAX A920 - How to Add Tips? - ZBS Helpdesk](https://help.zbspos.com/en/article/pax-a920-how-to-add-tips-qk4k8g/)
58. [Kitchen Display Systems (KDS) for Restaurants | Oracle](https://www.oracle.com/food-beverage/restaurant-pos-systems/kds-kitchen-display-systems/)
59. [Supported KDS Display Client Peripheral Devices | Oracle](https://docs.oracle.com/cd/E89797_01/doc.210/e89806/c_supportandcompatibility_kdsclientdevices.htm)
60. [Oracle MICROS Kitchen Display Systems – Official Product Sheet (PDF)](https://www.oracle.com/a/ocom/docs/industries/hospitality/oracle-micros-kitchen-display-systems-ds.pdf)