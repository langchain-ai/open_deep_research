# Comprehensive Comparison Report: Payment Terminal Solutions for 25-Location QSR Chain Expanding Across Quebec and New Brunswick

**Date:** May 30, 2026
**Prepared for:** Quick-service restaurant chain (25 locations)
**Terminals Evaluated:** Verifone Victa Portable, Clover Station Pro/Duo, PAX A920 Pro, PAX A920MAX

---

## Executive Summary

This revised report provides a comprehensive comparison of payment terminal solutions for your 25-location quick-service restaurant chain expanding across Quebec and New Brunswick, reflecting current market conditions as of May 30, 2026. The Verifone Carbon 8 has been removed from consideration (end-of-life, Android Lollipop, PCI approval expiring April 2027) and replaced with the **Verifone Victa Portable** (launched February 2026 via Moneris).

**Critical finding: Oracle Payment Interface (OPI) support ended October 31, 2025.** Any terminal integration with Oracle MICROS Simphony must now use the **Simphony Payment Interface (SPI)** or a certified payment processor plugin. This fundamentally changes the integration landscape.

**Primary Recommendation: PAX A920 Pro (standard locations) + PAX A920MAX (highest-volume locations)** — deployed through **PayFacto** (Montreal-based) with touché middleware for Oracle Simphony integration. This combination offers the strongest certified integration pathway for Oracle MICROS Simphony, lowest 5-year total cost of ownership, proven Quebec market presence (serving chains like St-Hubert and Benny & Co.), full French-language interface support via Android/PayDroid, and the A920MAX's proven 40% faster transaction throughput for peak-hour volumes.

**Secondary Recommendation: Verifone Victa Portable via Moneris** — excellent for chains preferring a rental model with predictable costs and the newest PCI 7-ready hardware, but with the critical caveat that the Moneris Go Restaurant KDS app is not available in Quebec and is English-only.

**Not Recommended: Clover Station Pro** — Clover is fundamentally a competing full POS system, not a payment-only peripheral for an existing Oracle MICROS setup. No certified direct Oracle Simphony integration was found. The 5-year TCO is the highest of all options, and hardware is processor-locked, preventing future switching.

---

## 1. Terminal Overview and Canadian Availability

### 1.1 Verifone Victa Portable (Moneris Go Terminal)

**Launch:** February 3, 2026 — Moneris is "the first commerce solutions provider in Canada to launch the Verifone Victa Portable terminal" [Moneris - Expands Go Commerce Suite](https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite).

**Hardware Specifications:**

| Specification | Detail |
|---|---|
| Processor | Qualcomm QCM2290 Quad Core A53 @ 2GHz |
| RAM | 4 GB |
| Storage | 32 GB Flash |
| Display | 6.7-inch HD+ touchscreen |
| Battery | 5000mAh Li-ion, 12+ hours continuous use (processing every 2 minutes) |
| Connectivity | Wi-Fi dual-band, Bluetooth 5.0, 4G LTE with eSIM and SIM slot, USB-C, Ethernet port |
| Scanner | Integrated Honeywell barcode scanner (2MP) |
| Printer | Built-in thermal printer (58mm x 30mm), NFC e-receipts |
| Dimensions | 205mm x 82mm x 58mm, ~456g |
| Durability | IP52 dust/water resistance, IK04 impact resistance |
| Security | PCI PTS 6.x approved, PCI 7-ready technology |
| OS | Android 13 (upgradeable to 14), Verifone Secure OS |

[Verifone Victa Portable](https://www.verifone.com/en-us/hardware-product/verifone-victa-portable) | [Verifone Victa Portable Plus Datasheet](https://cdn.prod.website-files.com/6877dbe2d81008ec40dd7770/69ab54bd5582db93331a50be_Victa%20Portable%20Plus%20Datasheet.pdf)

**Canadian Availability:** Exclusively through Moneris as the "Moneris Go Terminal." Moneris states: "We know Quebec because we're rooted here. With offices and local support teams within the province" [Moneris - Quebec](https://www.moneris.com/en/canada/quebec). Moneris has an office in Sackville, New Brunswick. The device supplies "over 12 hours of continuous usage processing payments every 2 minutes" [Moneris - Expands Go Commerce Suite](https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite).

### 1.2 Clover Station Pro / Station Duo

**Overview:** Clover is a wholly owned subsidiary of Fiserv. The Station Duo (current generation, sometimes called Station Pro) combines a 14-inch merchant-facing HD display with an 8-inch customer-facing touchscreen, built-in receipt printer, and cash drawer.

**Hardware Specifications (Station Duo):**

| Specification | Detail |
|---|---|
| Processor | Qualcomm Snapdragon 660 octa-core |
| RAM | 2 GB (some sources cite 4 GB) |
| Storage | 16 GB Flash |
| Display | 14" merchant-facing HD (1920x1080) + 8" customer-facing touchscreen |
| Camera | Dual 5 MP cameras with Zebra barcode scanning |
| Connectivity | Ethernet, Wi-Fi, 4G/LTE (optional), Bluetooth, 4 USB ports, 2 RJ12 cash drawer ports |
| Security | PCI PTS-6, fingerprint login, end-to-end encryption |
| OS | Android 10 (AOSP) |
| Dimensions | 13.0" x 9" x 12", ~6 lbs |

**Canadian Availability:** Available through Global Payments Canada (Fiserv's primary Canadian channel), Gravity Payments Canada ($1,995 CAD), and other resellers. Clover has a dedicated Canadian website at clover.com/ca. The Clover name and logo are used by Fiserv Canada Ltd. [Clover Canada](https://www.clover.com/ca/pos-solutions/quick-service-restaurant) | [Gravity Payments Canada - Clover Station Duo](https://gravitypayments.com/devices/clover-station-duo-canada)

**Critical limitation:** Gen 1 devices (Clover Station C010) reached End-of-App-Update on March 30, 2026. Current-generation devices (Station Solo, Station Duo 2) are not impacted [Clover Gen 1 End-of-App-Update](https://docs.clover.com/dev/docs/clover-devices-tech-specs).

### 1.3 PAX A920 Pro

**Overview:** The PAX A920 Pro is a mid-range Android smartPOS terminal designed for growing businesses processing 50-200 transactions daily [OctoPOS - A920 Pro vs A920 Max](https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick).

**Hardware Specifications:**

| Specification | Detail |
|---|---|
| Processor | ARM Cortex A53 Quad-Core, 1.4GHz + Secure Processor |
| RAM / Storage | 1GB DDR + 8GB eMMC (standard); 2GB DDR + 16GB eMMC (optional); microSD expandable |
| Display | 5.5-inch HD IPS (720 x 1440) |
| Battery | 5150mAh Li-ion (7.5 to 9.5 hours per charge) |
| Camera | 5MP rear autofocus + barcode scanner (1D/2D) |
| Printer | 2-inch thermal, 80mm/sec |
| Connectivity | 4G LTE, dual-band Wi-Fi, Bluetooth, dual SIM, USB-C OTG |
| Dimensions | 178.3 x 78 x 54.2mm, 390g |
| Security | PCI PTS 5.x SRED (Android 8.1) or 6.x (Android 10) |
| OS | Android 8.1 or 10 (PayDroid) |

[PAX A920 Pro Official Datasheet (PDF)](https://www.pax.us/wp-content/uploads/2023/12/A920Pro-Datasheet.pdf) | [PAX A920 Pro Product Page](https://www.paxtechnology.com/a920pro)

**Canadian Availability:** Available through PayFacto (Montreal-based, Class A Certification for PAX in Canada), Moneris (as Moneris Go), and Global Payments. PayFacto is "the first Canadian payment company to achieve Class A Certification for this device" [PayFacto - Launches PAX A920 in Canada](https://payfacto.com/payfacto-launch-pax-a920-android-payment-terminals-canada).

**Canadian Pricing:** $506.99 CAD at PC-Canada [PC-Canada PAX A920 Pro](https://www.pc-canada.com/item/pax-a920-pro-pos-terminal/a920pro-0aw-rd5-30ea).

### 1.4 PAX A920MAX

**Overview:** The A920MAX is PAX's flagship high-performance terminal, purpose-built for merchants processing 200+ daily transactions. It offers 40% faster processing and 216% faster reading speed compared to the original A920 [OctoPOS Blog](https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick).

**Hardware Specifications:**

| Specification | Detail |
|---|---|
| Processor | Cortex A53 Quad-Core 1.3GHz + Secure Processor (Octa-core variant available in some regions) |
| RAM / Storage | 1-2GB DDR3 + 8-16GB eMMC (standard); up to 4GB DDR4 + 64GB eMMC (regional variants); microSD expandable |
| Display | 6.0" HD+ (4G) or 6.5" (5G) capacitive touchscreen |
| Battery | 2500mAh LiFePO4 (equivalent 5000mAh/3.2V), 2.5 hours longer than A920 |
| Camera | Up to 13MP rear + 1MP front; optional professional barcode scanner |
| Printer | 80mm/sec thermal |
| Connectivity | 5G/4G LTE, WiFi 6 (5.0), Bluetooth 5.0/5.4, USB-C OTG, GPS |
| Dimensions | 186.3 x 80 x 54.9mm, 413g |
| Security | PCI PTS 6.x SRED, EMV L1 & L2 |
| OS | Android 10/11/13 (PayDroid) |

[PAX A920MAX Official Product Page](https://www.paxtechnology.com/a920max) | [Qualcomm Device Finder - PAX A920MAX](https://www.qualcomm.com/internet-of-things/device-finder/pax-a920-max)

**Canadian Availability:** Available through PayFacto, which has already deployed 25,000+ PAX terminals across Canada. PayFacto plans to "adopt next-gen PAX devices like the A920 Pro PCI 7 and Elys Solution" [PAX - Partner Success Story PayFacto](https://www.pax.us/partner-success-story-payfacto). US retail pricing is approximately $455 USD (~$620 CAD at current rates), with Canadian pricing to be confirmed directly with PayFacto.

---

## 2. Transaction Processing Speeds During Peak Hours

### 2.1 Verifone Victa Portable

The Victa Portable is powered by a Qualcomm QCM2290 quad-core processor at 2GHz with 4GB RAM, double the memory of previous generations [Moneris - Expands Go Commerce Suite](https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite). It processes payments approximately every 2 minutes during continuous use, delivering over 12 hours of battery life at this rate [Verifone Victa Portable](https://www.verifone.com/en-us/hardware-product/verifone-victa-portable). The device is described by Moneris as "built to handle high transaction volumes and diverse environments" [Moneris - Expands Go Commerce Suite](https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite).

For a QSR environment with peak-hour rushes, the Victa's 4GB RAM and quad-core processor provide adequate performance. However, no specific transactions-per-second benchmarks are published for peak-hour conditions.

### 2.2 Clover Station Pro/Duo

The Clover Station Duo features a Qualcomm Snapdragon 660 octa-core processor. The dual-screen design (14" merchant-facing + 8" customer-facing) is specifically engineered to speed checkout during peak traffic by allowing customers to select tips, sign, and choose receipt preferences in parallel with the merchant's workflow [Clover Station Pro Bundle](https://commercetech.com/eblog/clover-station-pro-bundle-with-customer-facing-screen).

Clover's own documentation recommends:
- **Ethernet over Wi-Fi** for reliable and fast transactions during peak hours
- **4G/LTE failover** as critical backup during network disruptions
- **Digital wallets (Apple Pay, Google Pay)** for faster checkout (contactless transactions take 1-2 seconds vs. 10-20 seconds for chip + PIN)

[Clover Insights - Faster Payments](https://uk.clover.com/insights/accept-payments-faster-and-get-paid-faster) | [Clover Blog - Contactless Payments](https://blog.clover.com/creating-a-contactless-payment-experience)

### 2.3 PAX A920 Pro

The A920 Pro's quad-core 1.4GHz ARM Cortex A53 processor handles 50-200 daily transactions effectively [OctoPOS Blog](https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick). It is described as "better suited for high-volume merchants needing faster processing and scalability" [Discount Credit Card Supply](https://www.discountcreditcardsupply.com/products/pax-a920-pro-payment-terminal). Battery life is rated at 7.5 to 9.5 hours of continuous use, which is adequate for a single shift but may require charging stations for extended operations.

### 2.4 PAX A920MAX (Upgrade for Highest-Volume Locations)

The A920MAX delivers **40% faster processing speed** compared to the A920 Pro, making it "crucial during rush periods" [OctoPOS Blog](https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick). Key speed advantages:

- **216% faster reading speed** compared to the original A920 [PAX A920MAX Product Page](https://www.paxtechnology.com/a920max)
- **WiFi 6 (5.0)** delivers "six times the speed of WiFi 4.0" [PAX A920MAX Product Page](https://www.paxtechnology.com/a920max)
- **5G connectivity** provides "faster, smoother, and more stable connections" in crowded environments [PAX A920MAX Product Page](https://www.paxtechnology.com/a920max)
- **40% faster install speed** via Bluetooth compared to previous generation

**Real-world benchmark:** "The A920 Max transformed a retail chain's checkout experience. Transaction times dropped 30%" [OctoPOS Blog](https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick).

For high-volume QSR locations (200+ daily transactions), the A920MAX is the recommended choice. The standard A920 Pro is better suited for locations with 50-200 daily transactions.

### 2.5 Verdict

| Terminal | Peak-Hour Capacity | Speed Benchmark |
|---|---|---|
| Verifone Victa | Adequate for most QSR volumes | No published benchmarks |
| Clover Station Duo | Good, dual-screen parallel processing | Optimized for counter service |
| PAX A920 Pro | Good for 50-200 transactions/day | Standard quad-core |
| **PAX A920MAX** | **Excellent for 200+ transactions/day** | **40% faster than A920 Pro; case study shows 30% faster checkouts** |

---

## 3. Offline Payment Capabilities (Store-and-Forward)

### 3.1 Verifone Victa Portable

Verifone terminals support Store and Forward (SAF) functionality, where "the pinpad approves and temporarily stores this transaction offline" and forwards transactions once connectivity is restored [Verifone Cloud - SAF](https://verifone.cloud/print/pdf/node/20078). Key limitations:

- **Storage limit:** Maximum of 1,000 transactions stored in SAF mode [Bank of America - SAF](https://merchanthelp.bankofamerica.com/Payments_Application_Store_and_Forward__SAF__Mode)
- **Merchant liability:** "Merchants are fully liable for the risk of failed captures related to payments processed offline" [Oolio Support](https://support.oolio.com/offlinepayments)
- **Authorization risk:** "A 5% average rate of declined transactions" when processed offline
- **Card restrictions:** Chip card purchases with correct PIN entry only; magnetic stripe, AMEX, cashouts, and refunds are typically excluded

For the Victa specifically, SAF availability depends on the processor's implementation (Moneris). Interac debit transactions are **not supported offline** by Moneris [Moneris - Offline Payments](https://www.moneris.com/help/V400m-WH-EN/Transactions/Offline_Payments_(formally_known_as_Store_and_Forward)_processing.htm).

### 3.2 Clover Station Pro/Duo

Clover supports offline payments when enabled, stating: "When enabled, devices can take offline payments for up to 7 days" [Clover Help Center](https://www.clover.com/help/set-up-offline-payments). Default limits:

- **Maximum per-transaction:** $500 (configurable) [Lightspeed Retail POS](https://shopkeep-support.lightspeedhq.com/hc/en-us/articles/47479984822299-Offline-Credit-Payments)
- **Maximum total offline payment amount:** $5,000 (configurable)
- **7-day offline window**

**Critical limitation for Canadian merchants:** "Offline payments and manual closeout are NOT supported" for Canadian merchants [Clover Developer Docs - Canadian Merchants](https://docs.clover.com/dev/docs/canadian-merchants). This means Clover's offline payment capability may not be available in Canada, or is severely restricted.

The Station Pro includes **native 4G/LTE connectivity** as a backup option for when Wi-Fi or Ethernet is unavailable, providing additional reliability beyond store-and-forward capability.

### 3.3 PAX A920 Pro / A920MAX

PAX terminals support store-and-forward as an optional feature that must be explicitly activated by the payment processor, with the merchant assuming associated risks [Moneris - Offline Payments](https://www.moneris.com/help/V400m-WH-EN/Transactions/Offline_Payments_(formally_known_as_Store_and_Forward)_processing.htm). Key conditions:

- Transactions are automatically forwarded once connectivity is restored
- Only chip card purchases with correct PIN entry and non-expired cards are eligible
- Merchant must enter a valid user ID and passcode for each offline transaction (fraud mitigation)
- **Daily cumulative limit:** Maximum total dollar value capped at 20% of projected monthly volume [Moneris - Offline Payments](https://www.moneris.com/help/V400m-WH-EN/Transactions/Offline_Payments_(formally_known_as_Store_and_Forward)_processing.htm)
- **Excluded transactions:** Interac debit, DCC, Gift and Loyalty, UnionPay, EMV fallback, cashback

Global Payments documentation notes that PAX A920 units require firmware version **01.01.11E or higher** for store-and-forward functionality [Heartland/Global Payments - PAX Store and Forward](https://heartlandpos.zendesk.com/hc/en-us/articles/1260806110890-PAX-Store-and-Forward-Setup-Guide).

The A920MAX's **5G connectivity** provides a significant advantage here — it is less likely to need offline mode because it maintains stable connections even in crowded network environments [PAX A920MAX Product Page](https://www.paxtechnology.com/a920max).

### 3.4 Verdict

| Terminal | Offline Support | Canadian Availability | Interac Offline |
|---|---|---|---|
| Verifone Victa | SAF (1,000 transactions) | Processor-dependent | Not supported |
| Clover Station Duo | 7-day window, $500/txn | **Not supported for Canadian merchants** | N/A |
| PAX A920 Pro | Configurable limits | Yes (processor-dependent) | Not supported |
| **PAX A920MAX** | **Configurable limits + 5G reduces offline need** | **Yes (processor-dependent)** | Not supported |

**A note on offline payments:** None of these solutions guarantee approval for offline transactions — that risk is always borne by the merchant. The A920MAX's 5G connectivity reduces the likelihood of needing offline mode in the first place, which is a practical advantage for high-volume QSR locations.

---

## 4. Oracle MICROS Simphony Integration

### 4.1 CRITICAL FINDING: Oracle Payment Interface (OPI) End-of-Life

**As of October 31, 2025, Oracle has ended support for the Oracle Payment Interface (OPI).** The official Simphony Cloud Service Compatibility Matrix (F36594-07, updated February 2026) explicitly states: "Support for Oracle Payment Interface ended on October 31, 2025" [Oracle Simphony Cloud Service Compatibility Matrix](https://docs.oracle.com/en/industries/food-beverage/fbcom/F36594_07.pdf).

The replacement is the **Simphony Payment Interface (SPI)**, described as "a resilient version of the OPI" that is part of the Simphony client application. SPI formats messages for the PSP (terminal or middleware) and processes responses for the client directly, eliminating the need for the OPI server and reducing LAN traffic [Oracle - The Simphony Payment Interface](https://docs.oracle.com/cd/F14820_01/doc.191/f15052/c_payments_spi.htm).

**This means any integration strategy based on OPI is no longer supported.** All three terminals must now integrate via SPI or through a certified payment processor plugin that handles the SPI interface.

### 4.2 Verifone Victa Portable Integration

Verifone terminals integrate with Oracle Simphony through certified payment processor partnerships. Key integrators:

- **Global Payments (Heartland):** "Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7" [Global Payments Developer - OPI](https://developer.globalpayments.com/heartland/payments/in-store/pos-middleware/oracle-payment-interface). Note: This was certified for OPI; SPI compatibility must be confirmed.
- **Shift4:** Integration with OPI supporting "multiple verification methods like chip-and-PIN, chip-and-signature, and Quick Chip" [Shift4 - Oracle OPI](https://www.shift4.com/news/shift4-announces-integration-to-the-oracle-payment-interface)
- **Adyen:** Adyen implements OPI 6.2 to integrate with Oracle Simphony, supporting Pay at Counter and Pay at Table [Adyen Docs - Oracle Simphony](https://docs.adyen.com/plugins/oracle-simphony)

**The Victa Portable is not the issue — the integration path exists through certified processors.** The question is whether those processor integrations have migrated from OPI to SPI.

Verifone's own documentation shows a "Verifone Payments Connector (FIPay OPI)" but this is for Oracle Retail (Xstore/EFTLink), **not** for Simphony F&B [Oracle Help Center - Verifone FIPay](https://docs.oracle.com/en/industries/retail/retail-eftlink/25.0/reopg/verifone-fipay.htm).

### 4.3 Clover Station Pro/Duo Integration

**No certified direct integration between Clover and Oracle MICROS Simphony was found.** This remains a critical gap.

Clover is fundamentally a competing POS platform owned by Fiserv. Clover Dining and Oracle MICROS Simphony POS are compared as **competing** restaurant management solutions, not integrated systems [Taloflow - Clover vs Oracle Simphony](https://taloflow.com/comparisons/clover-dining-vs-oracle-micros-simphony-pos).

Available indirect pathways do not solve the core problem:
- **Deliverect** offers two-way integration with Oracle Micros Simphony for online order management, but this is for order routing (not payment terminal integration) [Deliverect](https://www.deliverect.com/integrations/oracle-micros-simphony)
- **Global Payments OPI middleware** works with Oracle Simphony but uses Global Payments' own terminals (not specifically Clover)

Using Clover terminals with an existing MICROS Simphony setup would likely require extensive custom middleware development — a high-risk, high-cost approach.

### 4.4 PAX A920 Pro / A920MAX Integration

The PAX A920 series has the **strongest documented Oracle Simphony integration pathway** through the **Touché middleware solution**:

- PAX Technology has partnered with **Touché**, a software provider for Food & Beverage and Hospitality sectors, to integrate Touché's solution on PAX Android-based SmartPOS devices with **Oracle's MICROS Simphony POS system** [PAX Technology Blog - Touché](https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android).
- "This integration enables merchants to streamline ordering, payments, loyalty programs, and customer interactions more efficiently using a single platform" [PAX Technology Blog - Touché](https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android).
- The Touché solution supports **Pay@Table** (bill review, tip addition, discounts, payments, digital receipts, and automatic POS reconciliation), **Order@Table**, and **Order&Pay** (quick-service order and payment processing).
- The partnership is "being rolled out globally, with initial deployments in Italy, the Middle East, Europe, and **North America**" [PAX Technology Blog - Touché](https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android).
- Supported PAX Android devices include "the A920Pro, A930, and A77 Payment Smartphone" [PAX Technology Blog - Touché](https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android). The A920MAX runs the same PayDroid OS and PAXSTORE platform, making it compatible.

**Additionally, PayFacto (Montreal-based) offers integration through:**
- **SecureTablePay middleware** for connecting PAX terminals with various POS systems [PayFacto - Integrated Solutions](https://payfacto.com/integrated-solutions)
- **Maitre'D** and **Veloce** POS systems (acquired by PayFacto) for the hospitality industry

**The Moneris Payment Plugin for Oracle Simphony** is also available: "The Moneris payment plugin connects your Oracle Simphony solution to your Moneris payment terminal" [Moneris - Payment Plugin for Oracle Simphony](https://support.moneris.com/article/moneris-payment-plugin-for-oracle-simphony-getting-started-47707). This works with Moneris terminals (including PAX devices deployed through Moneris Go).

### 4.5 Verdict

| Terminal | Oracle Simphony Integration | Status |
|---|---|---|
| Verifone Victa | Via certified processor partners (Global Payments, Adyen, Shift4) | Possible but SPI migration must be confirmed |
| **Clover Station Duo** | **No certified direct integration found** | **High-risk; requires custom middleware** |
| **PAX A920 Pro / A920MAX** | **Touché middleware (validated globally); PayFacto SecureTablePay; Moneris plugin** | **Strongest documented pathway** |

**The PAX A920 Pro/A920MAX is the only terminal with a documented, globally deployed integration pathway for Oracle MICROS Simphony that is actively being rolled out in North America.**

---

## 5. Bilingual Interface Compliance with Quebec's Bill 96

### 5.1 Bill 96 Requirements Overview

Bill 96 (An Act respecting French, the official and common language of Québec) modified the Charter of the French Language, with major provisions effective June 1, 2025. Key requirements for payment terminals:

- **Customer-facing screens** on payment terminals must display French as the primary language
- **Receipts** must be printed in French (or bilingually with French predominant)
- **Software interfaces** used by employees must be available in French
- **Businesses with 25+ employees** in Quebec must register with the Office québécois de la langue française (OQLF) and undertake a francization process
- **Fines:** $3,000 to $30,000 per violation per day for first offenses; up to $90,000 per day for repeat offenses

[McCarthy Tétrault - Bill 96 Common Misconceptions](https://www.mccarthy.ca/en/insights/blogs/consumer-markets-perspectives/french-language-requirements-bill-96-and-june-1-2025-common-misconceptions) | [CFIB - Quebec's Law 14](https://www.cfib-fcei.ca/en/site/qc-law-14-bill-96)

### 5.2 OQLF Enforcement Findings (May 1, 2026)

On May 1, 2026, the OQLF released findings from inspections of nearly 1,000 businesses in Greater Montreal:
- 98% provided service in French
- 89% met requirements on invoices
- **87% complied for payment terminals**
- 23% of establishments had at least one compliance issue
- Over 10,000 complaints related to the Charter in the past year

[CTV News Montreal - OQLF Survey](https://www.ctvnews.ca/montreal/article/most-montreal-area-businesses-complying-with-french-language-rules-oqlf-survey-finds)

**Payment terminals are specifically monitored by the OQLF** as part of compliance inspections.

### 5.3 Terminal-Specific French Language Support

**Verifone Victa Portable (via Moneris):**
- Moneris terminals support bilingual operation with configurable Merchant Language and Customer Language settings [Moneris - Selecting Terminal Language](https://www.moneris.com/help/820_4.32f_webhelp/features_and_procedures/procedures/set_language.htm)
- Customer Language is set per transaction based on the Language Code on the customer's card
- Moneris explicitly states: "From language to local presence, our POS systems are made for Quebec" [Moneris - Quebec](https://www.moneris.com/en/canada/quebec)

**Critical concern:** The Moneris Go Restaurant KDS app is "currently available across Canada **except Quebec and in English only**" [Moneris Go Restaurant](https://www.moneris.com/en/solutions/pos-systems/go-restaurant). This means the native KDS solution does not support French-language operation and is not available in Quebec.

**Clover Station Pro/Duo:**
- Clover supports both English and French for payment flows [Clover Developer Docs - Canadian Merchants](https://docs.clover.com/dev/docs/canadian-merchants)
- The Clover Adobe Commerce payments extension supports "Canadian French" as a default locale [Clover Marketplace - Adobe Commerce](https://marketplace.clover.com/apps/adobe-commerce-payments)
- Clover has a dedicated French-language Canadian website at clover.com/ca/fr/[Clover Canada French](https://www.clover.com/ca/fr/pos-systems/online-ordering)
- **Quebec MEV WEB compliance:** Clover is compatible with Quebec's mandatory Sales Recording Module (WEB-SRM/MEV WEB) system through a dedicated app available on the Clover App Market [Clover MEV WEB](https://clovermevweb.ca)
- For Quebec-specific requirements: "Every merchant and customer-facing screen and prompt must be available in French and English" [Clover Developer Docs - Canadian App Market](https://docs.clover.com/dev/docs/international-app-market-readiness)

**PAX A920 Pro / A920MAX (via PayFacto):**
- PAX terminals run on Android/PayDroid with full language localization capabilities
- PayFacto documentation confirms PAX terminals support French-language display settings: Settings → Configure application → Terminal options → Display → Merchant Language → Choose English or French [PayFacto Docs - Display Language](https://docs.payfacto.com/payfacto-knowledge/canada-doc-center/applications/secure-payment/payment-standalone-mode/terminal-configuration/display-language-and-theme-settings)
- **PAX France division** provides "French-language UX/UI designs for Android payment terminals including the A920 and A920Pro" [Teddy Graphics - PAX France](https://teddygraphics.com/pax-france.html)
- PayFacto is headquartered in Montreal and serves Quebec restaurant chains (St-Hubert, Benny & Co.)
- PayFacto provides French-language customer support and documentation [PayFacto French Contact](https://payfacto.com/fr/contactez-nous) [PayFacto Docs](https://docs.payfacto.com/payfacto-knowledge/canada-doc-center)

### 5.4 Verdict

| Terminal | French Interface | Quebec-Specific Compliance |
|---|---|---|
| Verifone Victa | Configurable bilingual | **KDS not available in Quebec** |
| Clover Station Duo | Canadian French supported; MEV WEB certified | MEV WEB compatible |
| **PAX A920 Pro / A920MAX** | **Full Android localization; PAX France expertise** | **PayFacto is Montreal-based; serves Quebec chains** |

**The PAX A920 Pro/A920MAX through PayFacto offers the strongest Quebec compliance position** because:
1. PayFacto is a Montreal-based company with deep Quebec market experience
2. PAX has a dedicated France division with French-language UI expertise
3. French-language customer support and documentation are available
4. The Android platform allows full customization for Bill 96 compliance

---

## 6. New Brunswick Regional Considerations

### 6.1 Official Languages Act

New Brunswick is Canada's only officially bilingual province. However, **the Official Languages Act applies to government services, not private businesses** [Government of New Brunswick - Official Languages Act](https://laws.gnb.ca/en/document/cs/o-0.5). Key points:

- "The Act does not apply to schools, nor generally to private companies unless they serve on behalf of the government" [NB Media Co-op](https://nbmediacoop.org/2026/01/13/the-french-language-must-prevail-one-year-later)
- Private sector businesses are not legally required to provide bilingual signage or French-language interfaces
- A review of the Official Languages Act has been accelerated to be completed by **December 31, 2026**, which may extend requirements to the private sector

### 6.2 Municipal Signage Regulations

Some New Brunswick municipalities have taken independent action:
- **Dieppe** (75% francophone) introduced By-law Z-22 requiring new exterior commercial signs to be bilingual (French and English) or solely in French [OCOLNB - Dieppe Signage By-law](https://officiallanguages.nb.ca/newsroom/signage-by-law-in-dieppe-comments-from-the-commissioner)
- This applies to descriptive content on signs (e.g., "shoe store"), not business names

### 6.3 Federal French Language Regulations (UFPBA)

On April 15, 2026, the Canadian federal government introduced draft regulations under the **Use of French in Federally Regulated Private Businesses Act (UFPBA)** [Norton Rose Fulbright - UFPBA](https://www.nortonrosefulbright.com/en-ca/knowledge/publications/94ef6435/new-language-rules-for-federally-regulated-private-businesses-announced). This will initially apply to federally regulated businesses in Quebec, then extend to regions with strong francophone presence (including potentially New Brunswick) two years later.

### 6.4 Practical Recommendation for New Brunswick

For your New Brunswick locations, French-language support on payment terminals is not legally required for private businesses at this time, but it is strongly recommended for:
1. Serving the 32.4% francophone population
2. Future-proofing against potential regulatory changes (OLA review, UFPBA extension)
3. Consistent brand experience across Quebec and New Brunswick locations

**All three terminal options support bilingual interfaces**, so you can deploy the same terminal across both provinces with consistent French-language configuration.

---

## 7. Total 5-Year Cost of Ownership (TCO)

### 7.1 Verifone Victa Portable (Moneris Rental Model)

**Hardware:**
- Purchase: Approximately $535 CAD (US pricing reference) [POSGuys Extended Catalog](https://posguys.com/ExtendedCatalog/VeriFone/Verifone%20Portable%20Payment%20Devices/M571-350-22-NAA-6)
- Rental: $49.95/month (promotional: $0 for up to 1 year ending June 2, 2026) [Moneris - Save on Go](https://go.moneris.com/saveongo)
- Monthly account fee: $24.95 to $49.95 CAD

**Processing Fees (Moneris Flat Rate):**
- Credit card (card-present): 2.65% + $0.10 per transaction [Moneris Pricing](https://www.moneris.com/en/pricing)
- Interac debit: $0.12 per transaction

**5-Year TCO (per location, purchase model):**
- Hardware (purchase): $535 one-time
- Monthly fees: $24.95-$49.95/month = $1,497-$2,997 over 5 years
- **Total hardware + fees: ~$2,032 - $3,532** (excluding processing fees)

**5-Year TCO (per location, rental model):**
- Rental: $49.95/month x 60 months = $2,997 (after promotional period)
- Account fee: $24.95/month x 60 months = $1,497
- **Total rental: ~$4,494** (excluding processing fees)

### 7.2 Clover Station Pro/Duo

**Hardware:**
- Station Duo purchase: $1,995 CAD (Gravity Payments Canada) [Gravity Payments Canada](https://gravitypayments.com/devices/clover-station-duo-canada)
- Amazon.ca: $1,227.97 CAD (requires Powering POS account) [Amazon.ca](https://www.amazon.ca/Clover-Station-White-407-4040-0001/dp/B08QYNYHF5)

**Monthly Software Plans:**
- Register + Table Service: $84.95 - $89.95/month
- Advanced: $189/month
- Hidden fees commonly add $100-$200/month per location [CheckThat.ai - Clover Hidden Fees](https://checkthat.ai/blog/clover-hidden-fees)

**Typical hidden fees:**
- PCI compliance: $9.95-$10/month
- Platform access: $27.95/month
- Statement charges: $5-$15/month
- Monthly minimums: ~$49/month
- Essential app subscriptions: $150-$950+/month per location

**Processing Fees (Clover Direct):**
- In-person: 2.3% + $0.10 [Clover Pricing](https://www.clover.com/pricing)
- Online/keyed-in: 3.5% + $0.10-$0.15

**5-Year TCO (per location, moderate estimate):**
- Hardware: $1,995 one-time
- Software subscriptions: $84.95-$189/month = $5,097-$11,340 over 5 years
- Hidden fees: $100-$200/month = $6,000-$12,000 over 5 years
- **Total: ~$13,092 - $25,335** (excluding processing fees)

### 7.3 PAX A920 Pro

**Hardware:**
- Purchase: $506.99 CAD (PC-Canada) [PC-Canada](https://www.pc-canada.com/item/pax-a920-pro-pos-terminal/a920pro-0aw-rd5-30ea)
- Battery replacement: ~$50-100 every 2 years

**Monthly Fees (PayFacto/Moneris):**
- Minimal platform fees (typically $10-20/month)
- No mandatory software subscription (Android platform)
- Extended warranty: ~$50-100/year

**Processing Fees (Moneris or PayFacto):**
- Negotiable interchange-plus pricing recommended for high-volume merchants
- Moneris flat-rate: 2.65% + $0.10 credit; $0.12 Interac debit
- PayFacto: Negotiable volume-based pricing

**5-Year TCO (per location):**
- Hardware: $507 one-time
- Battery replacement (2x): ~$100
- Monthly fees: $10-$20/month = $600-$1,200 over 5 years
- **Total: ~$1,207 - $1,807** (excluding processing fees)

### 7.4 PAX A920MAX

**Hardware:**
- Purchase: ~$455 USD (~$620 CAD at current rates) [PAYBOTX](https://paybotx.company.site/Pax-A920-Max-p796101115)
- Canadian pricing to be confirmed through PayFacto
- Battery (LiFePO4): ~$29 USD replacement [Amazon - A920MAX Battery](https://www.amazon.com/FYIOGXG-Battery-Pax-A920-MAX/dp/B0F8BFZ7N7)

**5-Year TCO (per location, estimated):**
- Hardware: ~$620 CAD
- Battery replacement: ~$40
- Monthly fees: $10-$20/month = $600-$1,200 over 5 years
- **Total: ~$1,260 - $1,860** (excluding processing fees)

### 7.5 25-Location TCO Comparison (5-Year Total, Excluding Processing Fees)

| Terminal | Per Location (5-Year) | 25 Locations (5-Year) |
|---|---|---|
| Verifone Victa (purchase) | $2,032 - $3,532 | $50,800 - $88,300 |
| Verifone Victa (rental) | ~$4,494 | ~$112,350 |
| Clover Station Duo | $13,092 - $25,335 | $327,300 - $633,375 |
| **PAX A920 Pro** | **$1,207 - $1,807** | **$30,175 - $45,175** |
| PAX A920MAX | $1,260 - $1,860 | $31,500 - $46,500 |

**The PAX A920 Pro offers the lowest 5-year TCO by a wide margin** — approximately 70-80% less than Clover. The A920MAX premium over the A920 Pro is minimal (~$53 per location) while providing significantly higher performance.

**Important:** These estimates exclude payment processing fees, which represent the largest cost component. Processing fees vary based on negotiated rates and transaction volumes. Interchange-plus pricing (recommended for high-volume QSRs) can significantly reduce costs compared to flat-rate pricing.

---

## 8. Contactless Payment Limits in Canada

### 8.1 Current Limits (as of 2026)

| Network | Per-Tap Limit | Notes |
|---|---|---|
| Interac Debit | **Up to $500** (increased September 2025) | Varies by bank; TD, RBC, Scotiabank, BMO, CIBC offer $500 |
| Visa | **$250** | Set by Visa's Easy Payment Service |
| Mastercard | **$250** | Issuer-dependent |
| American Express | **$250** | Standard contactless limit |
| Apple Pay / Google Pay | **No per-tap limit** | Biometric authentication replaces PIN; governed only by card's standard limits |

[PaymentGateway.ca - Canada Contactless Limits 2026](https://paymentgateway.ca/contactless-payment-limits-canada-2026) | [Interac - Tap Payment Limits](https://www.interac.ca/en/consumers/faqs/what-is-the-contactless-limit/)

### 8.2 Implications for QSR

- **80%+ of in-person card transactions in Canada are now tap payments** (2024 data) [PaymentGateway.ca](https://paymentgateway.ca/contactless-payment-limits-canada-2026)
- A contactless transaction takes 1-2 seconds vs. 10-20 seconds for chip + PIN
- Contactless acceptance can increase restaurant tips by 15-25% through automatic on-screen tipping
- **Merchants do not need to update POS terminals** — limits are enforced by card issuers

### 8.3 Terminal Capabilities

All three terminals handle Canadian contactless limits identically:
- **Verifone Victa:** "Behind-the-display contactless optimization for better tap performance" [Verifone Victa Portable](https://www.verifone.com/en-us/hardware-product/verifone-victa-portable)
- **Clover Station Duo:** Integrated EMV/NFC acceptance [Clover Station Pro Bundle](https://commercetech.com/eblog/clover-station-pro-bundle-with-customer-facing-screen). Note: Station Solo requires additional hardware for contactless
- **PAX A920 Pro / A920MAX:** Full NFC support including Apple Pay, Google Pay, Samsung Pay

---

## 9. Tip Adjustment Workflows

### 9.1 Verifone Victa Portable

Verifone terminals offer flexible tipping features configurable through the payment software:
- Tip prompt appears on the PIN pad
- Tip amounts can be displayed as percentages (max 99.99%) or fixed values (max $99,999.99) [Verifone Cloud - Tipping](https://verifone.cloud/sites/default/files/inline-files/%5Bcurrent-date%3Ahtml_month%5D/Tipping_0.pdf)
- Options to allow custom tip amounts and include a "No Tip" option
- Tip adjustment holds the transaction until a tip is added manually, then sends to settlement
- All tips must be adjusted prior to batch settlement

Via Moneris Go: The Go platform supports pay-at-table workflows with tip entry at the table [Moneris - Expands Go Commerce Suite](https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite).

### 9.2 Clover Station Pro/Duo

Clover provides per-transaction tip settings through the `tipMode` SDK parameter [Clover Developer Docs - Tip Mode](https://docs.clover.com/dev/docs/tip-mode):
- **ON_SCREEN_BEFORE_PAYMENT** — Tip prompt shown before payment authorization (QSR mode)
- **ON_SCREEN_AFTER_PAYMENT** — Tip prompt shown after payment authorization (table service mode)
- Customizable tip suggestions (e.g., 18%, 20%, 25%) through Clover Dashboard
- **Scan to Pay** feature: Customers can pay and tip via digital wallet by scanning a QR code

**Limitation:** Clover offers manual tip pooling only — no automated tip pooling for operations with pooled tip arrangements [Expert Market - Clover Review](https://www.expertmarket.com/pos/clover-review).

### 9.3 PAX A920 Pro / A920MAX

PAX terminals support comprehensive tip adjustment workflows:

**Post-authorization tip adjustment:**
1. Navigate to the transaction batch
2. Select the transaction to adjust
3. Press "ADJUST" 
4. Enter the tip amount and press "CONFIRM"
5. Confirm the updated total

[Commerce Technologies - PAX A920 Tip Adjustment](https://commercetech.com/pax-a920) | [Talus Pay - Tips on PAX A920](https://taluspay.zendesk.com/hc/en-us/articles/27031070834579-How-to-Adjust-Tips-on-a-Pax-A920)

**Key features:**
- Tips can be added during authorization (pre-auth) or after as a post-auth adjustment
- Customizable tip prompts (15%, 18%, 20%, or custom) through the payment application
- Tip report history available same-day before batch processing
- Tips only adjustable before batch settlement

### 9.4 Verdict

All three terminals support the essential tip adjustment workflows required for QSR table service elements. The PAX A920 Pro/A920MAX offers the most flexible post-auth adjustment workflow with clear step-by-step processes documented by multiple processors.

---

## 10. Kitchen Display System (KDS) Compatibility

### 10.1 Key Consideration: KDS is Decoupled from Payment Terminal

Since your chain already uses Oracle MICROS Simphony, the KDS compatibility question is largely **decoupled from the payment terminal decision**:
- MICROS Simphony handles order entry and routing to the kitchen
- Your existing KDS displays orders from MICROS
- The payment terminal handles payment processing only
- The payment terminal does NOT route orders to the kitchen

The critical integration is between the payment terminal and MICROS Simphony (Section 4), not between the payment terminal and KDS.

### 10.2 Terminal-Specific KDS Options

**Verifone Victa (Moneris Go):**
- Moneris offers a free iPad KDS app: "Moneris Kitchen Display" [Moneris Go Restaurant](https://www.moneris.com/en/solutions/pos-systems/go-restaurant)
- KDS add-on: $9/month
- **Critical limitation:** The KDS app is "currently available across Canada **except Quebec and in English only**" [Moneris Go Restaurant](https://www.moneris.com/en/solutions/pos-systems/go-restaurant)

**Clover Station Duo:**
- Clover offers a native KDS with 14" and 24" display options [Clover Kitchen Display System](https://www.clover.com/kitchen-display-system)
- **14-inch display:** "Highest heat tolerance (122°F) on the market" [Clover KDS 14](https://www.clover.com/kds-14)
- Supports Wi-Fi and LAN connections
- Replaces paper tickets, aggregates all orders
- **But:** This requires adopting Clover as your full POS system, which is incompatible with your existing MICROS Simphony

**PAX A920 Pro / A920MAX:**
- **PAX Elys Display** (K2160: 16-inch, K2220: 22-inch Android-powered touchscreen): IP55 waterproof/dustproof, PoE support [PAX Technology - KDS and Bump Bar](https://www.paxtechnology.com/blog/pax-technology-launches-kitchen-displays-and-bump-bar-for-restaurants)
- **PB20 Bump Bar:** Configurable keyboard with magnetic keys
- **OrderPin KDS APP:** Cloud-based POS suite "specifically optimized for the A920 MAX" with a KDS module [OrderPin](https://www.orderpin.co/hardware/pax-a920-max)
- **retailcloud KDS:** Compatible with PAX A920 Pro hardware [retailcloud](https://retailcloud.com/hardware/Pax-A920-Pro-p123474230)
- **Critical:** PAX Elys KDS requires separate integration with MICROS Simphony, not direct connection to the A920 terminal

### 10.3 Verdict

| Terminal | KDS Solution | Quebec Availability | MICROS Integration |
|---|---|---|---|
| Verifone Victa | Moneris Go KDS ($9/mo) | **Not available in Quebec** | Via Moneris API |
| Clover Station Duo | Native 14"/24" KDS | Available | **Requires full Clover POS** |
| **PAX A920 Pro / A920MAX** | **Elys Display + OrderPin/retailcloud** | **Available** | **Requires separate MICROS integration** |

**Your best strategy: Maintain your existing Oracle MICROS KDS setup.** The payment terminal decision is independent of KDS compatibility.

---

## 11. Summary Comparison Table

| Dimension | Verifone Victa Portable | Clover Station Pro/Duo | PAX A920 Pro | PAX A920MAX |
|---|---|---|---|---|
| **Transaction Speed** | ✅ Quad-core, 4GB RAM | ✅ Dual-screen parallel processing | ✅ 50-200 txns/day capacity | ✅ 40% faster than A920 Pro |
| **Offline Payments** | ⚠️ SAF (1,000 txns); no Interac | ❌ **Not supported for Canadian merchants** | ⚠️ SAF; no Interac | ⚠️ SAF + 5G reduces offline need |
| **Oracle MICROS Integration** | ⚠️ Via processor partners (OPI EOL; SPI needed) | ❌ **No certified integration found** | ✅ **Touché middleware (validated globally)** | ✅ **Same Touché integration** |
| **Bill 96 Compliance** | ⚠️ French UI available; **KDS not available in Quebec** | ⚠️ Canadian French supported; MEV WEB certified | ✅ **Full Android localization; PayFacto Montreal-based** | ✅ **Same platform** |
| **New Brunswick Readiness** | ✅ Bilingual support | ✅ Bilingual support | ✅ Bilingual support | ✅ Bilingual support |
| **5-Year TCO (25 locations)** | ~$50,800-$88,300 (purchase) / ~$112,350 (rental) | **$327,300-$633,375 (highest)** | **$30,175-$45,175 (lowest)** | **$31,500-$46,500 (slight premium)** |
| **Contactless Limits** | ✅ $500 Interac / $250 credit | ✅ Same | ✅ Same | ✅ Same |
| **Tip Workflows** | ✅ Pre/post-auth adjustment | ✅ Most granular control | ✅ Post-auth adjustment documented | ✅ Same workflow |
| **KDS Compatibility** | ⚠️ KDS not available in Quebec | ✅ Comprehensive but requires full Clover POS | ✅ Elys Display (separate MICROS integration needed) | ✅ Elys Display + OrderPin |

---

## 12. Recommendations and Action Plan

### 12.1 Primary Recommendation: PAX A920 Pro + PAX A920MAX (Tiered Deployment)

**Standard locations (50-200 daily transactions): PAX A920 Pro**
- 25-location hardware cost: ~$12,675 CAD
- 5-year TCO: ~$30,175-$45,175

**Highest-volume locations (200+ daily transactions): PAX A920MAX**
- Premium of ~$113-$117 per unit over A920 Pro
- 40% faster processing, 5G connectivity, 30% faster checkouts (case study)
- Ideal for downtown, mall, or high-traffic locations

**Rationale:**
1. **Strongest Oracle MICROS Simphony integration** through Touché middleware (globally validated, North America deployment underway)
2. **Lowest 5-year TCO** by a wide margin (70-80% less than Clover)
3. **Proven Quebec market presence** through PayFacto (serves St-Hubert, Benny & Co.)
4. **Full French-language capability** via Android/PayDroid with PAX France division expertise
5. **Deployment flexibility** — tiered A920 Pro + A920MAX model matches performance to location volume
6. **Multi-channel availability** — PayFacto, Moneris, Global Payments all offer PAX terminals

### 12.2 Secondary Recommendation: Verifone Victa Portable (via Moneris)

**Best for:**
- Chains preferring a rental model with predictable costs
- Operations not heavily dependent on Moneris KDS (or using existing MICROS KDS)
- Chains wanting the newest PCI 7-ready hardware

**Critical caveat:** The Moneris Go Restaurant KDS app is not available in Quebec and is English-only. You would need to maintain your existing MICROS KDS.

### 12.3 Not Recommended: Clover Station Pro/Duo

**Reasons:**
1. **No certified direct integration with Oracle MICROS Simphony** — this is a critical gap requiring custom middleware development
2. **Highest 5-year TCO** — $327,300-$633,375 for 25 locations (10-20x PAX TCO)
3. **Processor-locked hardware** — cannot switch processors without replacing all terminals
4. **Long-term contracts** (36-48 months) with substantial early termination penalties
5. **Clover is a competing POS system**, not a payment-only peripheral for your existing MICROS setup

### 12.4 Action Plan

**Immediate (Next 30 Days):**

1. **Contact Canadian processors for PAX quotes:**
   - **PayFacto** (Montreal): Inquire about PAX A920 Pro and A920MAX pricing for 25-location deployment. Request interchange-plus pricing over flat-rate. Ask about Touché middleware for Oracle Simphony integration. Phone: payfacto.com/fr/contactez-nous
   - **Moneris**: Request PAX A920 Pro/A920MAX pricing through Moneris Go. Ask about Oracle Simphony plugin. Phone: 1-855-463-5669
   - **Global Payments Canada**: Request PAX terminal pricing and OPI/SPI integration status. French support: 1-800-263-2970

2. **Verify Oracle Simphony integration pathway:**
   - Confirm your Simphony version (2.9.2+ or 19.x) and SPI compatibility
   - Request a demo of Touché middleware integration from PayFacto
   - Verify Moneris Payment Plugin for Oracle Simphony works with SPI (not just deprecated OPI)

3. **Initiate OQLF registration:**
   - Register with the Office québécois de la langue française (required for 25+ employees)
   - Contractually require your payment processor to provide terminals pre-configured with French as the default language

**Short-Term (30-60 Days):**

4. **Pilot program:** Deploy PAX A920 Pro terminals at 2-3 locations (one Quebec, one New Brunswick, one highest-volume location with A920MAX) for 30-day testing

5. **Audit all customer-facing screens and receipts** for complete French coverage per Bill 96 requirements

6. **Negotiate volume discounts** for 25-location deployment (anticipate 10-20% bulk discount on hardware)

**Medium-Term (60-90 Days):**

7. **Rollout schedule:** Deploy in phases of 5-8 locations per week

8. **Staff training:** Ensure all employees understand bilingual terminal operation and Bill 96 compliance

9. **KDS strategy confirmation:** Maintain existing Oracle MICROS KDS setup; confirm that payment terminal choice does not affect KDS operations

---

## 13. Sources

[1] Moneris - Expands Go Commerce Suite (Feb 3, 2026): https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite

[2] Verifone Victa Portable - Official Product Page: https://www.verifone.com/en-us/hardware-product/verifone-victa-portable

[3] Verifone Victa Portable Plus Datasheet: https://cdn.prod.website-files.com/6877dbe2d81008ec40dd7770/69ab54bd5582db93331a50be_Victa%20Portable%20Plus%20Datasheet.pdf

[4] Moneris - Quebec Landing Page: https://www.moneris.com/en/canada/quebec

[5] PAX A920 Pro Official Datasheet (PDF): https://www.pax.us/wp-content/uploads/2023/12/A920Pro-Datasheet.pdf

[6] PAX A920 Pro Product Page: https://www.paxtechnology.com/a920pro

[7] PAX A920MAX Official Product Page: https://www.paxtechnology.com/a920max

[8] OctoPOS Blog - A920 Pro vs A920 Max: https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick

[9] Qualcomm Device Finder - PAX A920MAX: https://www.qualcomm.com/internet-of-things/device-finder/pax-a920-max

[10] PayFacto - Launches PAX A920 in Canada: https://payfacto.com/payfacto-launch-pax-a920-android-payment-terminals-canada

[11] PAX Technology Blog - Touché deploys Oracle solution: https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android

[12] Oracle Simphony Cloud Service Compatibility Matrix: https://docs.oracle.com/en/industries/food-beverage/fbcom/F36594_07.pdf

[13] Oracle - The Simphony Payment Interface (SPI): https://docs.oracle.com/cd/F14820_01/doc.191/f15052/c_payments_spi.htm

[14] Moneris - Payment Plugin for Oracle Simphony: https://support.moneris.com/article/moneris-payment-plugin-for-oracle-simphony-getting-started-47707

[15] Global Payments Developer - Oracle Payment Interface: https://developer.globalpayments.com/heartland/payments/in-store/pos-middleware/oracle-payment-interface

[16] Adyen Docs - Oracle Simphony Integration: https://docs.adyen.com/plugins/oracle-simphony

[17] Oracle Help Center - Verifone FIPay OPI: https://docs.oracle.com/en/industries/retail/retail-eftlink/25.0/reopg/verifone-fipay.htm

[18] Moneris - Offline Payments Processing: https://www.moneris.com/help/V400m-WH-EN/Transactions/Offline_Payments_(formally_known_as_Store_and_Forward)_processing.htm

[19] Clover Help Center - Set Up Offline Payments: https://www.clover.com/help/set-up-offline-payments

[20] Clover Developer Docs - Canadian Merchants: https://docs.clover.com/dev/docs/canadian-merchants

[21] Heartland/Global Payments - PAX Store and Forward: https://heartlandpos.zendesk.com/hc/en-us/articles/1260806110890-PAX-Store-and-Forward-Setup-Guide

[22] McCarthy Tétrault - Bill 96 Common Misconceptions: https://www.mccarthy.ca/en/insights/blogs/consumer-markets-perspectives/french-language-requirements-bill-96-and-june-1-2025-common-misconceptions

[23] CFIB - Quebec's Law 14 (Bill 96): https://www.cfib-fcei.ca/en/site/qc-law-14-bill-96

[24] CTV News Montreal - OQLF Survey (May 1, 2026): https://www.ctvnews.ca/montreal/article/most-montreal-area-businesses-complying-with-french-language-rules-oqlf-survey-finds

[25] Moneris - Selecting Terminal Language: https://www.moneris.com/help/820_4.32f_webhelp/features_and_procedures/procedures/set_language.htm

[26] Clover Developer Docs - Canadian App Market: https://docs.clover.com/dev/docs/international-app-market-readiness

[27] PayFacto Docs - Display Language Settings: https://docs.payfacto.com/payfacto-knowledge/canada-doc-center/applications/secure-payment/payment-standalone-mode/terminal-configuration/display-language-and-theme-settings

[28] PayFacto French Contact: https://payfacto.com/fr/contactez-nous

[29] Government of New Brunswick - Official Languages Act: https://laws.gnb.ca/en/document/cs/o-0.5

[30] Norton Rose Fulbright - UFPBA Regulations: https://www.nortonrosefulbright.com/en-ca/knowledge/publications/94ef6435/new-language-rules-for-federally-regulated-private-businesses-announced

[31] PaymentGateway.ca - Canada Contactless Limits 2026: https://paymentgateway.ca/contactless-payment-limits-canada-2026

[32] Clover Station Pro Bundle - Commerce Technologies: https://commercetech.com/eblog/clover-station-pro-bundle-with-customer-facing-screen

[33] Clover Pricing: https://www.clover.com/pricing

[34] CheckThat.ai - Clover Hidden Fees: https://checkthat.ai/blog/clover-hidden-fees

[35] Expert Market - Clover Review 2026: https://www.expertmarket.com/pos/clover-review

[36] Moneris Pricing: https://www.moneris.com/en/pricing

[37] Clover Canada - QSR Solutions: https://www.clover.com/ca/pos-solutions/quick-service-restaurant

[38] Gravity Payments Canada - Clover Station Duo: https://gravitypayments.com/devices/clover-station-duo-canada

[39] PC-Canada - PAX A920 Pro: https://www.pc-canada.com/item/pax-a920-pro-pos-terminal/a920pro-0aw-rd5-30ea

[40] Moneris Go Restaurant: https://www.moneris.com/en/solutions/pos-systems/go-restaurant

[41] Moneris - Save on Go Promotion: https://go.moneris.com/saveongo

[42] Clover Kitchen Display System: https://www.clover.com/kitchen-display-system

[43] PAX Technology - KDS and Bump Bar Launch: https://www.paxtechnology.com/blog/pax-technology-launches-kitchen-displays-and-bump-bar-for-restaurants

[44] OrderPin - PAX A920MAX Hardware: https://www.orderpin.co/hardware/pax-a920-max

[45] PayFacto - Partner Success Story at PAXCON 2025: https://www.pax.us/partner-success-story-payfacto

[46] PayFacto - About Us: https://payfacto.com/about-us

[47] PAX Compare A920 Models: https://www.pax.us/compare-a920-models

[48] Global Payments Canada Contact: https://www.globalpayments.com/fr-ca/contactez-nous

[49] Moneris - Contact Support: https://www.moneris.com/en/contact

[50] PayFacto - Integrated Solutions: https://payfacto.com/integrated-solutions