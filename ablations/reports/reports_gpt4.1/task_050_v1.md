# Comparative Evaluation: Verifone Carbon 8, Clover Station Pro, and PAX A920 Pro for a 25-Location QSR Chain in Quebec and New Brunswick

## Executive Summary

This report delivers a comprehensive, criterion-by-criterion comparison of the Verifone Carbon 8, Clover Station Pro, and PAX A920 Pro payment terminals as potential solutions for a quick-service restaurant (QSR) chain expanding across Quebec and New Brunswick. The evaluation specifically addresses transaction processing speed during peak hours, offline payment capabilities, seamless Oracle MICROS POS integration, Bill 96 bilingual compliance, total five-year cost of ownership (TCO), current Canadian contactless payment limits, tip adjustment workflows for table service, and kitchen display system (KDS) compatibility. Where data remains unavailable or open-ended, this is explicitly stated to guide strategic decision-making.

---

## 1. Transaction Processing Speeds During Peak Service Hours

### Verifone Carbon 8
- Built for high-speed operation with an Intel chipset, designed for fast checkout in hospitality and quick-service environments, targeting scenarios such as pay-at-the-table, curbside, and busy chain service[1].
- No independent or industry-standard benchmarks available, but manufacturer claims focus on reliability and efficiency even during rush periods[1].
- Contactless transactions (industry-wide) generally process in about one-tenth the time of traditional card transactions[2].

### Clover Station Pro
- Dual-screen design allows concurrent order taking and customer payment/interaction, enhancing peak throughput[3].
- POS is optimized for high-velocity environments, supporting contactless/NFC payments that are fast and minimize wait times, assuming consistent network connectivity[4][5].
- Known to power thousands of busy QSRs, with processing speed on par with leading industry hardware[3][4].

### PAX A920 Pro
- Quad-core 1.4GHz processor and advanced hardware, supporting stable, high-volume transaction processing for up to hundreds of transactions per device daily[6][7].
- Designed for high-load, secure, and responsive operation, common in large-scale retail and restaurants[8][9].
- Field reports indicate successful use in "busy" restaurant and retail scenarios; no third-party benchmarks provided[7].

**Summary:** All three devices are engineered to support fast, efficient transaction flows in busy quick-service environments. The Clover Station Pro offers a slight operational edge for counter service with its optimized workflow and dual-screen setup; all three support modern, quick tap processing.

---

## 2. Offline Payment Capabilities

### Verifone Carbon 8
- No public documentation confirming true offline card payment capability for Carbon 8 as of 2026. Other Verifone models have offline/fallback modes, but Carbon 8 support is unclear[1][10].
- Device supports customizable failed payment messaging; any offline handling would depend on payment application/processors and should be validated before rollout[11][12].

### Clover Station Pro
- Supports full offline payment mode for up to 7 days, with configurable per-transaction and cumulative offline limits[13].
- Offline mode must be enabled per device; offers varying levels of risk management (merchant-prompted, automatic, or "force" approvals)[14].
- Offline transactions are securely stored and uploaded for settlement upon reconnection; visual logs help track pending payments[14].

### PAX A920 Pro
- Can operate in offline (standalone) mode with batch upload of stored transactions, but this must be specifically enabled by the acquirer or payment processor[15].
- Built-in SIM card and automatic failover to cellular supports business continuity, though true offline card auth requires processor support[16].
- User-activated as required, not enabled out of the box[17].

**Summary:** Clover Station Pro offers documented, configurable offline processing. PAX A920 Pro can support offline transactions pending acquirer/software support. Carbon 8’s offline capability is not confirmed and requires further validation.

---

## 3. Integration with Oracle MICROS POS Systems

### Verifone Carbon 8
- No official confirmation of out-of-the-box Oracle MICROS Simphony integration. Verifone family terminals commonly integrate with Oracle, but Carbon 8 is not specifically listed in Oracle hardware compatibility documents[18][19][20].
- Integration possible via APIs and SDKs, but real-world compatibility should be validated via pilot program[20][21].

### Clover Station Pro
- Offers integration services for leading POS software (including MICROS), but no evidence of turnkey, certified Simphony or Oracle KDS integration[22][23].
- Typically requires middleware or custom development; chief strengths are in native Clover environments rather than legacy Oracle integrations[24][25].

### PAX A920 Pro
- Officially supported for integrated payments with Oracle MICROS Simphony via DNA Payments' axept® PRO solution; supports pay-at-table, split billing, and secure transaction functions[26].
- Frequently used with third-party integrators (Datacap, Payroc, etc.) that bridge POS and payment environments[27][28].
- Acts primarily as the payment device; does not function as a full-featured workstation or KDS display within MICROS[29].

**Summary:** PAX A920 Pro has proven Oracle MICROS Simphony integration through third party and partner channels, making it the strongest fit for deep Oracle deployment. Verifone Carbon 8 and Clover require custom middleware and are less plug-and-play.

---

## 4. Compliance with Quebec's Bill 96 for Bilingual (French/English) Interfaces

### All Terminals – Legal Context
- Bill 96, in force from 2025, requires all digital, signage, contract, and public interfaces in Quebec to feature French as the markedly predominant language[30][31].
- Failure results in fines up to $30,000 and risk of business license penalties[30].

### Verifone Carbon 8
- Supports both English and French interfaces. Merchant configures device language from a management menu; receipts print in the language of the cardholder, if supported[32][33].
- Complete compliance requires that merchants and system integrators ensure all UI/UX, receipts, and printouts meet “twice as prominent” French legal standard[30][32].

### Clover Station Pro
- User interface and receipts can be switched between French and English, supporting bilingual operation[34].
- No confirmation of certification for Bill 96; configuration is up to the merchant and may require legal review to assure prominence standards[30][34].

### PAX A920 Pro
- Interface and menus are customizable for language, with most POS/payment apps supporting French and other languages. Manual configuration required; not marked as inherently “Bill 96 certified”[35][36].
- Responsibility to assure French predominance resides with merchant and software provider; some system integrators offer translation/oversight services[31][36].

**Summary:** All three terminals can support French/English operation, but ongoing legal compliance is the responsibility of the merchant. None are certified “Bill 96 compliant” out-of-the-box. Configuration and frequent audits are strongly recommended.

---

## 5. 5-Year Total Cost of Ownership (TCO): Hardware, Processing, Replacement, and French Technical Support

### Verifone Carbon 8
- Hardware price: ~$810 CAD/unit; standard 1-year warranty, optional 2-year add-on[1][37].
- Hardware replacement: Typical POS cycle is 3-5 years. Out-of-scope (e.g., misuse, wear, environment) not covered[38].
- Payment processing: Fees set by processor (Worldpay, Elavon, First Data, etc.), not Verifone. Commercial rates vary, commonly 2.3–2.9% + $0.10/txn[39].
- French support: Canadian-based Verifone support includes French-language service lines and documentation[40].
- No public 5-year TCO calculator; final cost depends on actual merchant processing fees and support requirements[1][38][40].

### Clover Station Pro
- Hardware: $1,799–$2,099 CAD per terminal (outright) or up to $7,200 over 4-year lease[41][42].
- Software: Restaurant QSR package averages $135–$189/month per site (committed contract); additional add-ons $100–$300/month[43][44].
- Processing: Standard is 2.3% + $0.10/txn, negotiable for high volume chains[43].
- Replacement: 3–5 year expected terminal lifespan.
- French technical support varies by reseller and location; some offer French-first service, others prioritize English[45].
- 5-year per-location TCO: Can reach ~$200,000 including fees, software/hardware, and add-ons for high-turnover QSRs, but highly sensitive to transaction volume and add-on complexity[43][44][45].

### PAX A920 Pro
- Hardware: $400–$900 USD/unit retail, plus additional for accessories and upgrades[6][46][47].
- Replacement: Durable, all-day operation; planning for a 3-5 year replacement window in 24/7 settings[47].
- Processing: Determined by Canadian acquirer/processor; rates similar to above[6][46].
- French technical support provided by resellers/acquirers, not by PAX directly; available through most large acquirers and partners[46][48].
- No published complete 5-year TCO; merchants must factor in local support contracts, compliance/training, and potentially software licensing[46][47][48].

**Summary:** TCO is location/model/provider dependent. Clover is highest up-front and annually, with deeply integrated workflows. Verifone and PAX require merchant-led processor negotiation and quoting. All have variable French support depending on channel.

---

## 6. Current Contactless Payment Transaction Limits

- **Canada-wide:** The tap (contactless) limit is set by networks, not hardware. As of 2024, the standard limit is $250 per transaction for Visa, MasterCard, Amex; Interac debit remains at $100[49][50].
- All three terminals support industry-standard contactless wallets: Apple Pay, Google Pay, Samsung Pay, etc.
- Larger transactions require chip-and-PIN (or other authentication).

---

## 7. Tip Adjustment Workflows for Table Service Environments

### Verifone Carbon 8
- Supports flexible tipping at payment time or via post-payment adjustment, including custom amounts, tip thresholds, and reporting[51].
- Split and pay-at-table options available through POS integration; no Carbon 8-specific post-close tip adjustment workflow documented—would depend on POS setup[51][52].

### Clover Station Pro
- Robust tip adjustment workflows: Staff can enter/adjust tips before batch close, on the device, through the web dashboard, or via integrated workflow[53][54].
- Both customer-facing and programmable post-authorization tip adjustments supported via device interface or API[55].

### PAX A920 Pro
- Efficient tip workflows, including real-time (on-device) and post-authorization adjustments. User locates transaction, selects “tip” or “adjust,” enters amount[56][57].
- Suitable for fast-paced, pay-at-table QSR and casual/table service settings; process requires appropriate manager/staff access[58].

**Summary:** All three terminals support QSR and restaurant-standard tip adjustment workflows including post-authorization adjustment before batch settlement, supporting operational needs for dine-in/table service restaurants.

---

## 8. Compatibility with Kitchen Display Systems (KDS)

### Verifone Carbon 8
- Not listed in Oracle MICROS’s official supported hardware for KDS. Possibility of third-party integration exists but is not guaranteed; requires custom implementation/pilot[19][21][59].

### Clover Station Pro
- Native KDS integration with Clover KDS and Fresh KDS (Android/iOS-based); not directly compatible with Oracle MICROS KDS, which requires proprietary hardware[60][61].
- Parallel KDS environments possible, but integration between Clover and MICROS KDS not natively supported[62].

### PAX A920 Pro
- Serves as payment device only; not a KDS display per Oracle documentation[26][29][47].
- Oracle MICROS KDS requires Oracle-certified Windows client hardware; PAX can interface for payment but not as a kitchen display[59][63].

**Summary:** None of these payment terminals act as a direct client for Oracle MICROS KDS. For full Oracle KDS integration, use of officially supported MICROS client devices is necessary; payment terminals are not substitutes but can be payment endpoints feeding into the POS ecosystem.

---

## Comparative Table: Feature Summary

| **Criteria**                         | **Verifone Carbon 8**                                      | **Clover Station Pro**                                    | **PAX A920 Pro**                                         |
|--------------------------------------|------------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
| Peak Transaction Speed               | High (no benchmarks)                                       | High; dual screen; proven in QSR                         | High; robust specs; proven for busy environments         |
| Offline Payment                      | Not confirmed for Carbon 8 / requires validation           | Yes; up to 7 days; configurable                          | Yes; depends on acquirer/software configuration          |
| MICROS POS Integration               | APIs available; not certified for Simphony                 | Possible w/ custom work, not plug-and-play               | Certified partner integrations for Simphony payments     |
| Bill 96 Bilingual Compliance         | Full French/English configurable; legal burden on merchant | Bilingual configurable; legal burden on merchant         | Bilingual configurable; legal burden on merchant         |
| 5-Year TCO/Hardware                  | $810/unit + fees; support add-ons; variable                | $1799–$2099/unit; $135–$189/mo + add-ons; higher total   | $400–$900/unit; fees via acquirer; variable TCO          |
| Contactless Payment Limits           | $250 CAD (industry standard)                               | $250 CAD (industry standard)                             | $250 CAD (industry standard)                             |
| Tip Adjustment Workflow              | Supported; POS integration required                        | Robust, customer/staff workflows via device/API           | Supported, efficient both at device and post-authorization|
| Kitchen Display System Compatibility | Not Oracle KDS certified; requires pilot                   | Native Clover KDS; not for Oracle KDS                    | Payment only; no KDS function                            |

---

## Conclusion

All three evaluated terminals are fit for high-volume QSR environments and can meet most operational requirements of a growing chain in Quebec and New Brunswick if configured correctly and integrated via experienced partners. The **PAX A920 Pro** emerges as the strongest candidate for seamless Oracle MICROS Simphony POS payment integration and flexible offline operation (pending processor alignment). The **Clover Station Pro** dominates in workflow and customer interaction for QSRs using the Clover platform, with robust offline and tip functionality, albeit with less direct Oracle compatibility. The **Verifone Carbon 8** is technologically competitive but would require substantial custom validation for Oracle POS integration and confirmed offline payment operation.

Legal compliance with Quebec’s Bill 96 is a merchant responsibility, not a device feature. All reviewed terminals can support French/English bilingualism but must be properly configured and audited. None act as KDS displays for Oracle MICROS; investment in Oracle-certified KDS hardware will be needed for full kitchen integration.

Total cost of ownership varies: Clover is highest in upfront and ongoing costs for a vertically integrated solution; Verifone and PAX offer more a la carte/hybrid pricing depending on partnerships, with significant flexibility for acquirer and integration choices.

**Recommendation:** Prioritize operational pilots in live restaurants before chain-wide rollout, validating POS and KDS workflows, offline processing, and end-to-end Bill 96 compliance.

---

## Sources

[1] Verifone Carbon 8 | Smart POS Terminal for Businesses - Merchantrolls: https://merchantrolls.com/product/verifone-carbon-8/?srsltid=AfmBOorS94Ha0sgF64Dx2rLgCL1Ropx2Uz4t90-oPPh0LVLk3LbnbHZZ  
[2] Contactless Transactions Take the Lead in Canada's In- ...: https://ffnews.com/newsarticle/paytech/contactless-transactions-take-the-lead-in-canadas-in-store-purchases/  
[3] Quick Service Restaurant (QSR) POS & Software - Clover: https://www.clover.com/pos-solutions/quick-service-restaurant?srsltid=AfmBOor7OXqBhBMXYkVFKWdajJKrGjAIr_zb7iyhjf1-odDXqIKm0wEG  
[4] Clover POS Review 2026: Pricing, Hidden Fees & What Reps Won't Tell You: https://www.sleftpayments.com/learning-hub/clover-pos-review-honest-pricing-2026  
[5] How to improve restaurant service times at your QSR - Clover Blog: https://blog.clover.com/how-to-improve-restaurant-service-times-at-your-qsr  
[6] PAX A920Pro - Android mobile payment terminal: https://www.paxtechnology.com/a920pro  
[7] PAX A920 Pro vs A920 Max: Which one to Pick?: https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick/  
[8] A920 Pro » Extraordinary Mobile POS Device - PAX Technology: https://www.pax.us/products/mobile/a920-pro/  
[9] Describing The Difference Between A920 And A920 Pro: https://thriftypayments.com/blog/describing-the-difference-between-a920-and-a920-pro  
[10] Offline Payments - Knowledge Hub - Surfboard Payments: https://www.surfboardpayments.com/knowledge-hub/carbon-drop-9  
[11] Payment failed (offline payment methods) | Documentation | 2Checkout: https://verifone.cloud/docs/2checkout/Documentation/11Emails/Email_variables/Payment_failed_offline_payment_methods  
[12] Offline payments guidance | Documentation | 2Checkout: https://verifone.cloud/docs/2checkout/Documentation/11Emails/Email_variables/Offline_payments_guidance  
[13] Using Station Duo in Offline Mode - Clover Sport Help Center: https://bypassmobile.zendesk.com/hc/en-us/articles/30680012022676-Using-Station-Duo-in-Offline-Mode  
[14] Handle offline payments: https://docs.clover.com/dev/docs/handling-offline-payments  
[15] How to process offline transactions on a standalone PAX A920: https://support.tillpayments.com/hc/en-us/articles/6981642304143-How-to-process-offline-transactions-on-a-standalone-PAX-A920-terminal  
[16] Configuring an Internet Connection for the PAX Terminal: https://developer.cybersource.com/docs/cybs/en-us/pax-a920pro/activation/all/pax-a920pro/pax-a920pro/pax-terminal-configure-connect-intro/pax-terminal-configure-wifi-connection.html  
[17] PAX A920Pro User Manual - PostFinance Checkout: https://checkout.postfinance.ch/s/1480/resource/manual/pax-a920pro-en.pdf  
[18] Supported POS Client Devices: https://docs.oracle.com/cd/E76065_01/doc.29/e69880/c_supportandcompatibility_posclientdevices.htm  
[19] Supported Peripheral Devices: https://docs.oracle.com/cd/E76065_01/doc.29/e69880/c_supportandcompatibility_peripheraldevices.htm  
[20] Integrate with Simphony Point of Sale | Oracle: https://www.oracle.com/food-beverage/restaurant-pos-systems/pos-integrations/partners/  
[21] Verifone Carbon POS Review and Profile: https://www.cardfellow.com/product-directory/pos-systems/verifone-carbon-review  
[22] Clover Integration Services: https://www.clover.com/pos/integration-services?srsltid=AfmBOopmsOXFuDdY5uJUvUOFld5_MRNadLuuINzOV3PVkmVqdCmF092D  
[23] Clover Payment Device Integration for Software Developers: https://www.installpos.com/integrate-clover-payment-system-with-software  
[24] Payments integration options: https://docs.clover.com/dev/docs/paas-integration-options  
[25] Clover Dining vs Oracle MICROS Simphony POS for Restaurant Management Software in 2025: https://www.taloflow.ai/guides/comparisons/clover-vs-micros-rms  
[26] Pay at Reception, Pay at Counter and Pay at Table now available on Oracle Cloud Marketplace: https://www.dnapayments.com/news/pay-reception-pay-counter-and-pay-table-now-available-oracle-cloud-marketplace  
[27] PAX A920 Pro - Datacap: https://datacapsystems.com/pax-a920-pro  
[28] Pairing your Pax A920Pro, A920, or A80 with Retail POS (X-Series): https://x-series-support.lightspeedhq.com/hc/en-us/articles/25533755858203-Pairing-your-Pax-A920Pro-A920-or-A80-with-Retail-POS-X-Series  
[29] Oracle MICROS POS Systems for Restaurants: https://go.oracle.com/LP=91689  
[30] Bill 96 Compliance Guide for Quebec Businesses: Checklist | 2727 Coworking: https://2727coworking.com/articles/quebec-bill-96-business-compliance  
[31] Wordly Quebec Bill 96 Compliance Guide: https://offers.wordly.ai/bill-96-compliance-guide  
[32] Language | In-Person Payments | Verifone Developer Portal: https://verifone.cloud/docs/in-person-payments/global-payment-application-gpa/verifone-global-payment-application-8  
[33] Multi-Language support: https://verifone.cloud/print/pdf/node/1326  
[34] Quebec Bill 96 (Law 14) Compliance: https://www.megalexis.com/en/quebec-bill-96-translation-compliance/  
[35] PAX A920Pro User Manual - PostFinance Checkout: https://checkout.postfinance.ch/s/1480/resource/manual/pax-a920pro-en.pdf  
[36] Quebec Businesses Exposed by Bill 96: Aria's Bilingual Solution | Aria AI: https://www.linkedin.com/posts/heyaria_quebec-bill96-bilingualbusiness-activity-7440062956375052290-Equx  
[37] Shop Verifone Warranties | PC-Canada.com: https://www.pc-canada.com/cat/services-training/services/warranty/verifone?srsltid=AfmBOor4LPNVziSdcBXY7-5QhLgSSioikYPUgooNJ4_C3A8amq8MfYJf  
[38] Warranty Information - Customer Service - Verifone: https://my.verifone.com/s/warrantyinformation  
[39] Verifone Carbon POS Review and Profile: https://www.cardfellow.com/product-directory/pos-systems/verifone-carbon-review  
[40] Verifone | Legal Information: https://www.verifone.com/legal/standard-warranty-terms-verifone-payment-devices  
[41] Clover POS Cost & Pricing in 2025: Clover Hardware Pricing, Clover Monthly Fees, Processing Fees — Limelight Payments: https://shop.limelightpayments.com/blog/clover-pos-cost-pricing-2025-hardware-clover-fees-processing-fees?srsltid=AfmBOor4l6PBiVGGCToJVn4Ii3vUDuh6Vf31UTqOMOZRQV2ZJMHNRQXm  
[42] Clover POS Pricing: How Much Does Clover Cost?: https://tech.co/pos-system/clover-pos-pricing  
[43] Clover POS Pricing Breakdown: Fees & Hidden Costs (2026) | UpMenu: https://www.upmenu.com/blog/clover-pos-pricing/  
[44] Clover POS Pricing 2026: Plans, Costs & Actual TCO - CheckThat.ai: https://checkthat.ai/brands/clover/pricing  
[45] Clover POS Review 2026: Pricing, Hidden Fees & What Reps Won't Tell You: https://www.sleftpayments.com/learning-hub/clover-pos-review-honest-pricing-2026  
[46] PAX Wireless Terminals and Payment Devices for Sale: https://www.discountcreditcardsupply.com/collections/pax-wireless-terminals/printer-a920?srsltid=AfmBOorMujXgtbFefVYfXtYQBfbzOG57hA6RN0qB8Ne6oeOIJtwZvtA6  
[47] PAX Technology A920 Payment Terminals | POSGuys.com: https://posguys.com/payment-terminals_37/PAX-Technology-A920_4119/?srsltid=AfmBOooZAECHSTCTVt-aLjfRPyY5H6sc-GwvY_Rqi0xrTHWIkJB5e42J  
[48] PAX-A920 Mobile Payment Terminal Available Through Bluefin: https://www.bluefin.com/device/pax-a920/  
[49] Card Networks Up Canadian Contactless Transaction Limits To Limit Physical Contact – Digital Transactions: https://www.digitaltransactions.net/card-networks-up-canadian-contactless-transaction-limits-to-limit-physical-contact/  
[50] Accept Payments, Card Processing Services | Clover Canada: https://www.clover.com/ca/pos-systems/accept-payments?srsltid=AfmBOor-DMr1SdnuILt3AiamMRME8mYU7g9k-t5cgLG5WZGojyHbkNKl  
[51] Tipping - Verifone Documentation: https://verifone.cloud/sites/default/files/inline-files/%5Bcurrent-date%3Ahtml_month%5D/Tipping.pdf  
[52] How to use Pay @ Table (e.g. split bill) function on an Integrated Verifone T650p (mx51) – Nuvei Help Centre: https://support.tillpayments.com/hc/en-us/articles/6369045235727-How-to-use-Pay-Table-e-g-split-bill-function-on-an-Integrated-Verifone-T650p-mx51  
[53] Adjust a tip: https://docs.clover.com/dev/docs/android-payments-api-tip-adjust  
[54] Authorize and capture a tip-adjusted payment on paper: https://docs.clover.com/dev/docs/authorizing-a-tip-adjusted-payment-on-paper  
[55] Capture and tip adjust: https://docs.clover.com/dev/docs/capture-and-tip-adjust  
[56] How to Adjust Tips on a Pax A920 - Talus Pay: https://taluspay.zendesk.com/hc/en-us/articles/27031070834579-How-to-Adjust-Tips-on-a-Pax-A920  
[57] PAX A920 - Adjust A Tip - YouTube: https://www.youtube.com/watch?v=BUE549heG98  
[58] PAX A920 - How to Add Tips? - ZBS Helpdesk: https://help.zbspos.com/en/article/pax-a920-how-to-add-tips-qk4k8g/  
[59] Supported KDS Display Client Peripheral Devices: https://docs.oracle.com/cd/E89797_01/doc.210/e89806/c_supportandcompatibility_kdsclientdevices.htm  
[60] Run your restaurant more efficiently with the Clover Kitchen Display System (KDS) - Clover Blog: https://blog.clover.com/introducing-clover-kitchen-display-system-kds/  
[61] Kitchen Display System for Clover | Fresh KDS Integration: https://www.fresh.technology/kds/pos/clover  
[62] Clover POS Cost & Pricing in 2025: Clover Hardware Pricing, Clover Monthly Fees, Processing Fees — Limelight Payments: https://shop.limelightpayments.com/blog/clover-pos-cost-pricing-2025-hardware-clover-fees-processing-fees?srsltid=AfmBOor4l6PBiVGGCToJVn4Ii3vUDuh6Vf31UTqOMOZRQV2ZJMHNRQXm  
[63] Supported KDS Display Client Peripheral Devices: https://docs.oracle.com/cd/E76065_01/doc.29/e69880/c_supportandcompatibility_kdsclientdevices.htm