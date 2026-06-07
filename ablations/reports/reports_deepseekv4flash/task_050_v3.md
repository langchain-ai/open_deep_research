# Comprehensive Revised Comparison: Payment Terminal Solutions for a 25-Location QSR Chain Expanding Across Quebec and New Brunswick

**Date: May 28, 2026**

---

## Executive Summary

This comprehensive report provides a revised comparison of payment terminal solutions for your 25-location quick-service restaurant chain expanding across Quebec and New Brunswick, addressing specific gaps identified in previous analyses—particularly around Canadian market-specific device configurations, SDK limitations, and acquirer integration plugins for Verifone and PAX devices in Oracle MICROS Simphony environments.

**Primary Recommendation: PAX A920 Pro / A920MAX / A920Pro PCI 7** with interchange-plus pricing through PayFacto Montreal or Adyen Canada. This solution offers the strongest certified Oracle MICROS Simphony OPI 6.2+ integration pathway, the lowest 5-year total cost of ownership ($3.86M chain-wide versus $3.91–$3.97M for alternatives), active Canadian certification through multiple processors, and full French-language support for Bill 96 compliance.

**Secondary Recommendation: Verifone Victa Portable (Moneris Go Terminal)** via rental model for locations prioritizing zero upfront investment and predictable monthly costs, though OPI integration requires verification through Moneris specifically.

**Not Recommended:**
- **Verifone Carbon 8** — End-of-life, PCI PTS certification expired, unavailable from authorized Canadian channels
- **Clover Station Pro / Station Duo** — No certified Oracle MICROS Simphony OPI integration, processor-locked to Fiserv, higher TCO

---

**Critical Findings That Change Previous Assessments:**

| Finding | Detail |
|---------|--------|
| **Carbon 8 EOL Confirmed** | Original hardware (SUB179-xx1) PCI PTS expired April 30, 2024. Android Lollipop (2014). Not available from Moneris or Verifone. |
| **Victa Portable is Active** | Launched in Canada February 3, 2026 as Moneris Go Terminal. PCI PTS 6.x (ready for 7.x). Android 13 upgradable to 14. |
| **Clover Has No OPI Integration** | Clover is absent from Oracle's official payment integration partner list. Custom middleware development required for Oracle Simphony. |
| **PAX Has Strongest OPI Path** | Adyen confirms OPI 6.2 integration. PayFacto achieved Class A Certification for A920 in Canada. Multiple certified processor pathways. |
| **Interac Debit: No Offline** | Interac requires real-time online authorization. No store-and-forward capability exists on any terminal for Interac. |
| **Bill 72 Effective May 7, 2025** | Tips must be calculated on pre-tax subtotal in Quebec. All three terminals support tip-on-subtotal configuration with proper setup. |
| **Bill 96 Penalties** | $3,000–$30,000 per offense for corporations; up to $90,000/day for repeat offenses. Registration with OQLF mandatory for 25+ employees. |
| **Processing Fees Dominate TCO** | Processing represents 97–99% of 5-year costs. Interchange-plus pricing saves ~$1.36M over 5 years chain-wide versus flat-rate. |

---

## 1. Verifone Carbon 8 — End-of-Life Status and Replacement

### 1.1 PCI PTS Certification Status

The Verifone Carbon 8 has two PCI PTS certification entries in the PCI Security Standards Council's official PTS device listing [1]:

**Entry 1: PCI PTS Listing #4-10209 — "Verifone, Inc., X10, Carbon 8, Carbon 10"**
- Hardware #s: SUB179-xx1-01-A, SUB179-xx1-02-A
- **PTS Approval Expiry Date: April 30, 2024 (EXPIRED)**

**Entry 2: PCI PTS Listing #4-10241 — "Verifone, Inc., X10, Carbon 8, Carbon 10"**
- Hardware #: SUB179-xx2-02-A
- **PTS Approval Expiry Date: April 30, 2027**

Only units with the **SUB179-xx2-02-A** hardware revision remain PCI PTS approved. Units with original hardware (SUB179-xx1) have expired certification, meaning they cannot be used for new PCI-compliant deployments.

### 1.2 Android Version

The Verifone Documentation Portal confirms the Carbon 8 runs **Android Lollipop (Android 5.x)** [2]. Android Lollipop was released in 2014 and is no longer listed in Android Security Bulletins, meaning it has reached end-of-life and no longer receives security patches [3]. For comparison:

- Verifone Victa Portable: Android 13 (upgradable to Android 14 or 16)
- PAX A920 Pro: Android 8.1/10/11 depending on variant
- PAX A920Pro PCI 7: Android 14

### 1.3 End-of-Life Confirmation

Multiple lines of evidence confirm the Carbon 8 is discontinued for new deployments:

- **Verifone's current retail product page** lists only "Verifone Carbon Mobile 5" — the Carbon 8 and Carbon 10 are not listed [4]
- **Moneris Canada's device catalog** (moneris.com/en/support/devices/all-devices) lists 25+ devices including the Moneris Go Terminal (Victa Portable) but **does not include the Carbon 8** [5]
- The Carbon 8 was originally launched on **May 10, 2017** — approximately **9 years old** as of 2026 [6]
- Moneris announced on **February 3, 2026** that it is "the first in Canada to launch the Verifone Victa Portable terminal" as the Moneris Go Terminal [7]
- Verifone's Application Development Kit (ADK) Version 4.7 release notes state: "Support for Carbon 8 and Carbon 10 is discontinued with ADK-4.7.10" [8]

**Conclusion**: The Carbon 8 is definitively end-of-life and not available for new deployments from authorized Canadian channels. Existing installations may still receive support through specific contracts, but new deployment is not viable. Existing units should be replaced as soon as practical due to PCI compliance risks from the expired certification and obsolete Android OS.

---

## 2. Verifone Victa Portable — Current Canadian Offering

### 2.1 Official Specifications

The Verifone Victa Portable is Verifone's current-generation portable payment terminal, documented on the Verifone Documentation Portal and product pages [9][10][11]:

| Specification | Value |
|---------------|-------|
| **Display** | 6.7-inch HD+ capacitive multi-touch color LCD |
| **Processor** | Qualcomm QCM2290 (A53 Quad-core @ 2GHz) |
| **Memory** | 4GB RAM, 32GB flash memory |
| **Operating System** | Factory-selectable: VAOS Android 13 (upgradable to 14) or VOS3 |
| **Battery** | 5000 mAh, over 12 hours continuous use (processing payments every 2 minutes) |
| **Printer** | Built-in thermal printer (58mm x 30mm) |
| **Connectivity** | Wi-Fi (2.4GHz and 5GHz), Bluetooth 5.0, Cellular (2G/3G/4G LTE CAT 4), eSIM and physical SIM slots, USB-C, Ethernet dongle support |
| **Security** | PCI PTS 6.x approved (ready for PCI PTS 7.x) |
| **Environmental** | IK04 impact rating, IP52 ingress protection |
| **Barcode Scanner** | Integrated with Honeywell decoder |
| **Biometric Authentication** | Integrated capabilities (palm vein and facial recognition) |

The Verifone official website states: "Ensures end-to-end protection with PCI 7.x compliance and integrated biometric authentication" [10]. At TRANSACT 2026, Moneris confirmed: "PCI 7 certified devices not only assure the best security but also the longest life for those devices in the field" [12].

### 2.2 Canadian Availability

The Verifone Victa Portable is **available in Canada exclusively through Moneris** as of February 2026. Moneris is the **first provider in Canada to bring Victa Portable to market** [7][12][13].

Key announcements:
- **February 3, 2026**: Moneris announces the Moneris Go Terminal, which is the Verifone Victa Portable branded as a Moneris product [7]
- Jordan Williamson, Vice President, Core Products at Moneris: "Moneris is proud to partner with Verifone and be the first in Canada to bring the Victa Portable solution to market as part of our Moneris Go commerce suite" [7]
- Skip Hinshaw, EVP, Financial Institutions at Verifone: "Victa Portable was built for the next generation of commerce, and Moneris is setting the pace by being first to introduce it to Canadian businesses" [7]
- Available in ivory and onyx finishes [7]
- The device is described as having "a nearly seven-inch display, fast processing, PCI 7-ready security technology, built-in independent cellular connectivity, and a battery lasting over 12 hours" [7]

The Moneris Go Terminal operates on the "unified Moneris Go platform, helping businesses integrate once and scale over time while maintaining consistent security and performance" [13].

### 2.3 Verifone SDK and VOS3 Platform Limitations

**VOS3 (Verifone Operating System 3)** is one of the factory-selectable operating systems for the Victa device line, alongside VAOS Android 13 [9]. The **Verifone Engage** platform is the payment engine layer that provides security on top of the Android OS [2].

**Key SDK Capabilities:**

- **Application Development Kit (ADK) Version 4.7**: The current SDK for developing applications on Verifone devices [8]
- **XPI Integration API**: An API developed to "facilitate secure payment transactions on Engage and UX payment devices," supporting magnetic stripe, contactless, and contact EMV card transactions [14]
- **Verifone Cloud Device Integration (VCDI)**: Enables POS systems to communicate with Verifone payment terminals via the cloud, eliminating hardwired connections [15]
- **eCommerce API**: Customizable payment forms, support for digital wallets and BNPL options [16]

**Known Limitations:**

1. **Discontinued Carbon 8/10 Support**: Support discontinued as of ADK-4.7.10 [8]
2. **XPI Support Varies by Device and Region**: "XPI support varies by device and region" is a documented limitation [14]
3. **eVo/Verix Terminal Deprecation**: Verifone stopped delivering, supporting, and repairing eVo terminals by March 31, 2023 [17]
4. **Must Upgrade from PCI 3/4**: Merchants must use Verifone Engage (VOS) or Android terminals that support at least PTS PCI 5 approval [17]
5. **API Restrictions**: Special characters disallowed in customer API parameters include the backslash (\) [16]
6. **No Open-Source SDK**: Verifone's development ecosystem is proprietary, requiring ADK licensing and partner agreements [8]

### 2.4 OPI Integration Pathways

The Oracle Payment Interface (OPI) is the payment integration standard for Oracle Hospitality platforms, enabling secure, integrated payment processing for Oracle's hospitality systems including MICROS Simphony [18][19].

**OPI through Canadian Acquirers:**

**Moneris Canada + OPI**: Moneris provides specific integration support for Oracle Simphony through its ERP integration program. Users can "learn how to set up your Moneris Go terminal and configure the Moneris payment plugin for Oracle Simphony" [20]. This confirms Moneris supports OPI integration for Oracle Simphony environments.

**Global Payments Canada + OPI**: "Our Oracle Payment Interface (OPI) enables EMV payment processing for Micros point of sale (POS) systems. Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7" [21].

**Adyen Canada + OPI**: "Adyen payment terminals implement Oracle Payment Interface (OPI) 6.2 to integrate to Oracle Simphony" [22]. The integration supports **Pay at Counter** and **Pay at Table** features. OPI currently supports more than 80 countries through 45 Oracle Partners [18].

**Important Note**: OPI integration is terminal-agnostic at the interface level. The Verifone Victa Portable integrates with Oracle MICROS through OPI via the processor's middleware, not directly. Merchants must verify with their chosen processor that the Victa Portable is supported in their specific OPI deployment.

### 2.5 French Language Configuration

The Victa Portable runs Android 13, which inherently supports French language locale settings. The VOS3 operating system also supports French. For Quebec deployments, the terminal must be configured with French as the default language on:
- Customer-facing screens (payment prompts, tip prompts, signature capture)
- Admin/back-office screens
- Receipt templates (paper and digital)

Moneris, as a Canadian company with Quebec offices, can configure terminals with French language images before deployment. However, the contract should explicitly require French as the default language for Quebec locations.

---

## 3. Clover Station Pro — Discontinuation and Replacement Analysis

### 3.1 Discontinuation Status Confirmed

**The Clover Station Pro has been discontinued.** Key evidence:

- The "Station Pro" (along with "Station 2 (2018)") ended support on **September 30, 2024** [23][24]
- Clover has announced an End of Life/Support policy imposing a **$99.95 per device End of Support Fee** (effective May 1, 2025) [23][24]
- According to Limelight Payments (2026), the following Clover models are **phased out** (no software patching, no support): Clover Station 1.0, Clover Station 2.0, **Clover Station Pro**, Clover Station Duo Gen. 1, Clover Mini Gen. 1 & 2, Clover Flex Gen. 1 & 2, Clover Mobile [25]
- Clover's official developer documentation states: "Gen 1 devices, including Clover Station (C010), Mobile (C020/C021), Mini (C030/C031), and Flex (C041/C042), are approaching End-of-App-Update (EOAU) on March 30, 2026" [26]

### 3.2 Successor Devices

**Clover Station Duo Gen 2 (Duo 2)** is the direct successor:

| Specification | Clover Station Duo Gen 2 |
|---------------|-------------------------|
| **Merchant Display** | 14-inch HD |
| **Customer Display** | 8-inch touchscreen (Gen 2) with chemically strengthened glass |
| **Operating System** | Android 10.x |
| **PCI Certification** | PCI PTS 6.x |
| **Connectivity** | Ethernet, WiFi, 4G/LTE, Bluetooth 5.0 |
| **Payments** | Magnetic stripe, EMV chip card, NFC contactless |
| **Printer** | Integrated high-speed thermal printer |
| **Camera** | Dual 5-megapixel cameras with barcode scanning |
| **Security** | Fingerprint login, NFC employee cards |
| **Launch** | June 2023 |

Source: [26][27][28]

**Clover Station Solo** is a single-screen alternative with a 14-inch HD display, built-in printer, and cash drawer, designed for businesses with "moderate customer volume" that don't need a customer-facing display [26][29].

### 3.3 Oracle MICROS Simphony OPI Integration — Definitive Status

**No official, certified direct integration exists between Clover and Oracle MICROS Simphony via OPI 6.2+.** This is based on the following evidence:

1. **Oracle's official POS integrations page** lists payment platform partners including Adyen, Elavon, and FreedomPay — **Clover is notably absent** [30]

2. **Clover vs Oracle Simphony** is listed as a comparison between competing products, not integrated systems, on multiple comparison sites [31]

3. **Certified OPI Partners** that DO have verified OPI 6.2+ certification with Oracle MICROS:
   - **Adyen**: Confirmed OPI 6.2 integration supporting Pay at Counter and Pay at Table [22]
   - **Elavon (Global Payments)**: "Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7" [21][32]
   - **FreedomPay**: OPI integration with OPERA, MICROS 3700, 9700, Simphony 1.x and 2.x, and Xstore [33]
   - **Viva.com**: OPI integration with Oracle OPERA 5 and Micros Symphony [34]

4. **Oracle Simphony Integrations Program** allows partners to enroll via the Oracle PartnerNetwork (OPN), but Clover does not appear in the published partner list for payment integrations [30][35]

### 3.4 Semi-Integration Options via CloverConnector SDK

Clover **does** offer "semi-integration" where a Clover device acts as a payment-only peripheral. According to Clover's official developer documentation:

"Semi-integrated solutions run on a combination of Clover and third-party hardware; the Clover device handles payment processing" [36]

**CloverConnector SDK Availability:**
- **Android** — Available [36][37]
- **iOS** — Available [36][37]
- **JavaScript** — Available [36][37]
- **.NET (C#)** — Available via NuGet (Clover.RemotePayWindows version 4.0.0) [38][36][37]
- **Java** — Available [37]

**Key SDK Methods for Canada:**
- Canadian merchants must use SDK version 2.0 or higher [39]
- "Only credit cards and co-branded Interac cards can be entered manually" [39]
- Interac transactions: auth/pre-auth functions not allowed, offline payments not supported, vaulting cards not permitted, manual closeouts not supported [39]
- "Canadian merchants cannot use their Clover devices to process any payments in offline mode" [39]
- Refunds require the card to be inserted into the device [39]

**Development Effort for Oracle Simphony Integration:**

There is **no existing middleware or ISV solution** that bridges Clover payment terminals with Oracle Simphony. To achieve semi-integration:

1. A custom software layer (middleware) would need to be built, likely in .NET or Java
2. This middleware would need to interface with Oracle Simphony on one side — using Oracle's Transactional Services API (Gen 1 or Gen 2) or ISL scripting via SIM/PSM modules [40]
3. The Simphony Transaction Services Gen2 (STSG2) is "a RESTful API transaction interface that is both scalable and responsive" [35]
4. For Clover side, the "C# source code is available on Clover's GitHub Project and can be read, altered, and built to your needs" [38]

This would be a significant enterprise software development engagement, estimated at 3-6 months for development and certification, with ongoing maintenance costs for OPI version updates.

### 3.5 Processor-Lock Limitations in Canada

**Clover terminals can ONLY process through Fiserv (First Data) in Canada.**

Multiple sources confirm this:
- "Clover's rules do not allow any non-fiserv processing. There isn't even a way to set up any alternate processing" [41]
- "When you get a Clover POS system, the processor that comes with it is the only processor that can process payments on it" [42]
- Fiserv acquired First Data in 2019, and Clover is now owned by Fiserv [43]

**Canada-specific situation:**
- **Fiserv Canada** distributes Clover in Canada with a dedicated website (merchants.fiserv.com/en-ca) [44]
- **TD Bank** signed a multi-year agreement with Fiserv on July 23, 2025, adding approximately 30,000 Canadian locations to Clover's portfolio [45]
- **Moneris**, Canada's largest processor (a joint venture of RBC and BMO), is **NOT compatible** with Clover terminals [45][46]
- **Global Payments** has OPI integration with Oracle Simphony [21], but uses Global Payments' own terminals, not Clover
- **Adyen Canada** has OPI 6.2+ integration with Oracle Simphony [22], but uses Adyen's own terminals, not Clover

**This means:**
- You cannot switch processors without replacing all Clover terminals
- You cannot use Moneris, Global Payments, or Adyen with Clover terminals
- You are locked into Fiserv's processing rates and contract terms

### 3.6 Canadian Availability and French Language

Clover is available in Canada through Fiserv Canada, which states: "Fiserv has been delivering POS solutions to tens of thousands of Canadian small businesses for more than 20 years with locally managed support teams" [44].

**Quebec-specific requirements:**
- Revenue Quebec SRM (Sales Recording Module) mandate: "Each app in the Canadian hospitality, full service restaurant, or quick service restaurant sectors must be certified with Revenue Quebec's SRM mandate" [47]
- Bilingual support: "All app market listings must be localized, supporting both English and French to accommodate merchants and customers, especially in Quebec" [47]
- Support for Canadian postal codes, tax requirements (GST/QST), and PIPEDA compliance [47]

Clover devices support both English and French payment flow [39]. Users can change the device language on each Clover device [48]. The Clover Payments plugin for WooCommerce "supports English and Canadian French" [49].

---

## 4. PAX A920 Pro / A920MAX / A920Pro PCI 7 — Detailed Analysis

### 4.1 PAX Models Available in Canada

**PAX A920 Pro (Original Model)** [50][51][52]:

| Specification | Value |
|---------------|-------|
| **OS** | PAXBiz® on Android 8.1 / 10 / 11 (variant-dependent) |
| **Processor** | ARM Cortex A53 Quad-Core 1.4GHz + Secure Processor |
| **Memory** | 8GB eMMC Flash + 1GB DDR RAM (optional 16GB + 2GB) |
| **Display** | 5.5-inch IPS WXGA HD touchscreen, 720x1440 |
| **Battery** | 5150mAh, 7.5–9.5 hours depending on usage |
| **Printer** | Built-in thermal, up to 80mm/sec |
| **Connectivity** | 4G LTE, Wi-Fi (2.4GHz + 5GHz), Bluetooth 5.0, GPS |
| **Camera** | 5MP rear with autofocus, 0.3MP front |
| **PCI Certification** | PCI PTS 5.x SRED (older) / PCI PTS 6.x SRED (newer) |
| **Additional** | 1D/2D barcode scanner, optional fingerprint reader |

**PAX A920MAX** [53][54]:

| Specification | Value |
|---------------|-------|
| **OS** | Android 10 / 11 / 13 |
| **Processor** | Cortex A53 Quad-Core, 1.3GHz + Qualcomm secure processor |
| **Memory** | 2GB RAM, 16GB Flash (expandable via microSD to 128GB) |
| **Display** | 6-inch HD+ Infinity Edge (4G) or 6.5-inch (5G) |
| **Battery** | 2500mAh Lithium Iron Phosphate (LiFePO4) — 2.5 hours longer than A920 |
| **Printer** | 80mm/sec thermal printer |
| **Connectivity** | 5G / 4G LTE, Wi-Fi 5.0, Bluetooth 5.0 |
| **Camera** | 13MP rear with autofocus |
| **PCI Certification** | PCI PTS 6.x SRED |
| **Speed Claim** | "216% faster reading speed when compared to the A920" [53] |

**PAX A920Pro Duo (PCI 7)** [55]:

| Specification | Value |
|---------------|-------|
| **OS** | Android 14 with PAXBiz enhancements |
| **Processor** | Multi-core (Cortex A75 and A55) |
| **Memory** | Upgraded (details not specified) |
| **Display** | 6.56-inch HD+ main + 1.99-inch customer-facing screen (Duo model) |
| **Battery** | 6000mAh with magnetic charging |
| **Printer** | Up to 80mm/sec thermal printer |
| **Connectivity** | 5G / 4G LTE, dual-band Wi-Fi, Bluetooth 5.0, GPS |
| **PCI Certification** | PCI PTS 7.x with SRED |
| **PAXSTORE** | Remote application management, updates, device monitoring |

### 4.2 PCI PTS Certification Status

| Device | PCI PTS Version | Approval Expiry |
|--------|----------------|-----------------|
| PAX A920 (original) | PCI PTS 5.x | 30 April 2027 |
| PAX A920 Pro (PCI 5) | PCI PTS 5.x | 30 April 2027 |
| PAX A920 Pro (PCI 6) | PCI PTS 6.x | 30 April 2032 |
| PAX A920MAX | PCI PTS 6.x | Active |
| PAX A920Pro Duo (PCI 7) | PCI PTS 7.x | Active (2030s) |

Source: [56]

The PAX A77 Android MiniPOS was the "world's first payment terminal to receive PCI PTS Version 7.0 certification, certified through April 2035," demonstrating PAX's leadership in PCI certification [57].

### 4.3 Oracle MICROS Simphony OPI Integration — Strongest Pathway

**The PAX A920 Pro has the strongest documented OPI integration pathway of all three terminal options:**

**1. Adyen — Confirmed OPI 6.2 Integration** [22]:

"Adyen payment terminals implement Oracle Payment Interface (OPI) 6.2 to integrate to Oracle Simphony. OPI sends all transaction messages directly to Adyen's payment terminal."

The integration supports:
- Pay at Counter (default option) — "Pay at counter enables transactions directly at the counter with POS sending billing info to the terminal"
- Pay at Table — "With Pay at table, you can get the bill directly on the payment terminal, print the bill, split the amount, and return the payment to the POS system"

Setup requirements:
- First deploy the Oracle Payment Interface by following Oracle's guide
- Order terminals from Adyen and contact Support to enable OPI in your Customer Area
- Key technical detail: "Adyen has deprecated the classic API integration (JNI) in favor of Terminal APIs" [58]

**2. Global Payments Canada — OPI Certified** [21]:

"Our Oracle Payment Interface (OPI) enables EMV payment processing for Micros point of sale (POS) systems. Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7 with Heartland's out-of-scope semi-integrated EMV payment application."

Global Payments lists PAX terminals among supported hardware for OPI.

**3. PayFacto — Class A Certification for A920 in Canada** [59][60][61]:

PayFacto is "the **first Canadian payments company to complete a Class A Certification** of PAX's A920 mobile device, providing Canadian merchants now with a perfectly compact and powerful tool to conduct seamless payments."

- What **Class A Certification** means: The terminal has been fully tested, certified, and approved for deployment on PayFacto's payment processing network
- PayFacto integrates with hospitality POS systems via **SecureTablePay** middleware [60]
- PayFacto serves "over 50,000 merchants" across Canada [59]
- PayFacto is headquartered in **Montreal, Quebec** [59]

**4. Touché Middleware — PAX + Oracle Integration** [62]:

PAX Technology has partnered with **Touché**, a software provider for F&B and Hospitality, to "integrate Touché's software on PAX's range of Android-based SmartPOS devices and Oracle's MICROS Simphony POS system."

This integration streamlines "ordering, payment, and loyalty management" processes, providing an additional integration pathway through third-party middleware.

### 4.4 PAXSTORE Marketplace

PAXSTORE is a comprehensive, Android-based device management platform and marketplace [63][64]:

| Metric | Value |
|--------|-------|
| **Connected Devices** | Over 8,500,000 worldwide |
| **Software Applications** | Over 8,000 (2,400+ Android-based) |
| **Developers** | Over 2,700 |
| **Countries** | 100+ |

**Key Features:**
- App Management: Application approval, distribution controls, subscription from global marketplace, cloud configuration of payment app parameters [64]
- Terminal Management: Remote app installation and control, terminal monitoring including geolocation and hardware status, firmware OTA upgrades [64]
- Remote Assistance: AirViewer offering options to view terminals or take full remote control [63]
- Geolocation: Locate devices in real-time and receive alerts if devices go out of range [63]
- Remote Key Injection: Download encryption keys remotely, eliminating the need to ship terminals to secure third parties [63]
- Triple Signing: Apps authenticated via developer + PAX + optional reseller signatures [65]

**Restaurant/QSR Apps Available:**
- BPayd App [66]
- TableTurn — restaurant management/payment app [66]
- MealsyPay — restaurant payment solution [66]
- Taliup POS — POS application [66]
- AirLauncher — system configuration and brand customization [63]
- GoInsight — detailed terminal analytics [63]

### 4.5 PAX SDK Capabilities and Limitations

**Available SDKs:**

1. **PAX SDK**: "Built for software developers programming a local or cloud based application in the retail or restaurant environment" [67]

2. **PAX POSLink SDK**: "Pax payment processing machine systems work great with cloud based software applications that can easily enable CORS cross origin resource sharing through the POSlink SDK" [67]

3. **PAX SI SDK (Semi-Integrated SDK)**: "Developers, combine your POS app running on a PC with a PAX terminal to start accepting card-present payments. Build a completely custom app" [68][69]

**PAX Developer Portal**: "Serves as a technical community and support hub for developers working with PAX payment solutions. Powered by Android, our devices are built for flexibility and innovation" [70].

**Documented Limitations:**

1. **SRED Cannot Be Disabled**: "The device always provides SRED functionality and doesn't support the disablement (turning off) of SRED functionality" [71] — This is a critical security constraint from PCI PTS security policy.

2. **App Signing Requirements**: "All PAXSTORE applications go through a two-step process, developer and PAX signature, to properly authenticate and verify applications prior to pushing live on the PAXSTORE" [65]

3. **Marketplace Locking**: "Locks terminal to a Marketplace (so the terminal cannot freely move from Marketplace to Marketplace) and prevents the offline loading of an APK" [65]

4. **Tamper Response**: "If the device is in tampered state, the user must contact the device maintenance or authorized center immediately, remove it from service" [71]

5. **Secure Firmware Updates**: "Any security related update and/or patch loaded into PAX terminals must be signed using RSA certificate" [71]

6. **No Offline APK Loading**: The platform prevents offline loading of APK files [65]

### 4.6 French Language Support

PAX terminals **do support French language interfaces**. Key evidence:

- The **PAX France UX/UI project** (behance.net) shows a comprehensive French-language interface design for PAX terminals including the A50, A77, A80, A35, A920, and A920Pro [72]
- The design covers: payment interaction screens, amount-input screens, transaction type selection, currency configuration, service type selection, and administrative control interfaces [72]
- PAX Android-based terminals run on Android, which inherently supports French locale settings
- Canadian distributors (NPS Canada, MOBOPAY, PayFacto) operate in Quebec and support bilingual deployments
- Cardium, a Montreal-based distributor at 1184 Rue Sainte-Catherine, offers PAX terminals pre-configured for Canadian businesses [73]

---

## 5. Canadian-Specific Constraints

### 5.1 Offline Payment Support

#### Interac Debit — NO Store-and-Forward Support

**Interac debit does NOT support true offline/store-and-forward mode** on any terminal platform. This is a fundamental architectural constraint of the Interac network [74][75][76].

**Why Interac cannot operate offline:**
- Interac requires **real-time online authorization** with the cardholder's financial institution for every single transaction
- Interac uses a "good funds" model that verifies sufficient funds instantly [75]
- Technical documentation from Paylosophy explains that Interac transactions require a unique **Message Authentication Code (MAC)** — an encrypted block of transaction data generated during verification [77]
- This MAC "cannot be calculated at the POS/gateway level" because it depends on specific terminal hardware and session keys stored within the terminal [77]
- Hardware-level MAC validation requires online connectivity to the Interac network

**Practical Implication:** If a terminal loses network connectivity, it **cannot process Interac debit transactions**. The terminal will decline the transaction or require an alternative payment method (cash or credit card, which may offer limited offline capabilities).

#### Credit Card Networks — Limited Offline Support

Visa, Mastercard, and American Express CAN support offline/store-and-forward mode in Canada, but with important caveats [78][79]:

| Network | Offline Support | Notes |
|---------|----------------|-------|
| **Visa** | ✅ Limited | EMV offline data authentication (ODA) for limited amounts. Store-and-forward available through certain processors. |
| **Mastercard** | ✅ Limited | EMV supports offline authorization with issuer-defined limits. Store-and-forward available. |
| **American Express** | ✅ Limited | Closed-loop network; offline authorization depends on issuer policies. |
| **Interac Debit** | ❌ NO | Real-time online authorization only. No store-and-forward. |

**Important Canadian vs. US Differences:** Canadian acquiring configurations **default to online-only authorization** for chip/contactless transactions. Canadian EMV is strongly "chip & PIN" (CVM-required), which further limits offline capability. Store-and-forward functionality exists in Canada but must be explicitly configured by the processor/acquirer and is less commonly deployed in Canadian QSR environments compared to the US.

### 5.2 Contactless Payment Limits (2026)

As of 2026, the contactless payment limits in Canada are [80][81]:

| Network | Contactless Limit | Notes |
|---------|------------------|-------|
| **Visa** | $250 CAD | Raised from $100 during COVID-19 pandemic |
| **Mastercard** | $250 CAD | Raised from $100 effective April 2, 2020 |
| **American Express** | $250 CAD | Matched Visa/Mastercard increase |
| **Interac Flash** | $100 CAD (some issuers up to $250) | Interac cited risk control for not matching credit card limits |

**Historical Context:** Mastercard raised its limit from $100 to $250 effective April 2, 2020, in response to the COVID-19 pandemic. Visa confirmed the same increase. Sasha Krstic, president of Mastercard Canada, stated: "With safety and social distancing top of mind for all Canadians, today's announcement is one way we're helping cardholders to shop easily, securely and with more peace of mind during this difficult time" [81].

The Retail Council of Canada wanted Interac to raise its limit as well, but Interac responded: "These limits are security measures to protect Canadians against theft and fraud through unauthorized use of their debit card... by not raising transaction limits, we are doing our part to protect Canadians" [81].

**Mobile Wallets:** Apple Pay and Google Pay can exceed physical card limits entirely because biometric authentication replaces the PIN [80].

**Terminal Handling — All Three Terminals Identical:**

PaymentGateway.ca explicitly states: "Merchants don't need to change anything on their terminals. Payment limits are issuer-enforced, not terminal-enforced" [80]. When limits are exceeded, the user is prompted to insert the card into the EMV chip reader and enter their PIN. All three terminal types (Verifone Victa, PAX A920 Pro, Clover Station Duo) handle this identically — they support NFC/contactless payments including MiFare and NFC/CTLS schemes.

### 5.3 Tip Adjustment Workflows

#### Quebec's Bill 72 — Effective May 7, 2025

**Bill 72** (An Act to protect consumers against abusive commercial practices) introduced specific requirements for tip calculation on payment terminals [82][83]:

1. **Tips must be calculated based on the pre-tax amount (subtotal)**, not on the final total that includes taxes
2. **Tips must be presented neutrally** without influencing specific tip amounts

From MYR POS: "Quebec enacted Bill 72 on November 7, 2024, updating consumer protection laws by changing how tips are calculated in the hospitality industry. Effective May 7, 2025, all businesses in Quebec that accept tips—such as restaurants, bars, cafés, and taxis—must calculate suggested tip amounts based on the pre-tax subtotal rather than the total amount including taxes" [83].

**Example:** A 15% tip on a $100 pre-tax subtotal is $15, instead of $17.25 on a $115 total with tax.

**Terminal Configuration for Tip-on-Subtotal:**

**Adyen Implementation (Specific Technical Reference)** [84]:

"To comply with regulation in Quebec, Canada, tips can be calculated on pre-tax amounts by overriding the tip amount on the payment request."

The technical implementation requires:
1. Sending a payment request with the `AskGratuity` tender option
2. Including a Base64-encoded `Operation` JSON object containing `AmountToTipOn` and `TipSuggestions` parameters
3. The `AmountToTipOn` parameter "specifies the part of the total transaction amount that shoppers can add a tip on"
4. The `TipSuggestions` define "up to three preset tipping options by fixed amount or percentage" and optionally "allow custom tip entry"
5. Requires "terminal software version 1.103 or later"
6. "Not supported for pay-at-table or split check scenarios"

**Moneris Implementation** [85]:

"Bill 72 aims to make key changes to the Consumer Protection Act focusing on tips, food pricing, lease contracts, credit agreements, and itinerant merchants."

Moneris confirms:
- **Moneris Core Semi-Integrated & Moneris Core Restaurant**: Already support tip-on-subtotal functionality
- **Moneris Go (Integrated)**: Update developed to support tip-on-subtotal; devices automatically updated on May 7, 2025
- **Moneris Go (Standalone)**: Updated to support tip-on-subtotal and advanced tax management

**PayFacto Implementation** [86]:

"Bill 72 officially came into force in Quebec on May 7, 2025, requiring tips to be calculated on the amount before taxes. Establishments must adjust their sales management systems and point-of-sale (POS) software to automatically calculate tips before provincial and federal taxes are applied."

PayFacto reports that they have "been able to rapidly deploy solutions compliant with the new regulations for our customers using Maitre'D and Veloce point-of-sale solutions."

**All three terminals support tip-on-subtotal configuration** with proper setup through their respective processors.

#### Critical Distinction: Credit Card vs. Interac Debit Tip Adjustment

| Aspect | Credit Card (Visa/MC/Amex) | Interac Debit |
|--------|---------------------------|---------------|
| **Tip Timing** | Tip can be added BEFORE or AFTER authorization | Tip MUST be included BEFORE transaction finalization |
| **Post-Auth Adjustment** | ✅ Supported — authorization adjustment allows increasing authorized amount before capture (up to ~20% of original) | ❌ NOT supported — "good funds" real-time model; no post-transaction adjustments possible |
| **Technical Mechanism** | Pre-authorize base amount, adjust authorization upward to include tip, then capture | Include tip amount in the initial (and only) authorization request |
| **Overcapture** | Supported (up to ~15-20% over original for most schemes) | N/A — no capture step separate from authorization |
| **Chargebacks** | Yes — transactions can be disputed post-settlement | No — "No chargebacks" model |

Source: [84][87][88]

**For Bill 72 compliance on Interac:** The tip prompt MUST appear on the terminal BEFORE the Interac transaction is finalized. The total (pre-tax subtotal + tip) is sent as a single authorization request. The `AmountToTipOn` must be set to the pre-tax subtotal.

**For Bill 72 compliance on credit cards:** Two approaches are possible: (1) Pre-auth + tip adjustment — authorize base amount, present tip prompt, adjust authorization upward, then capture; (2) Single transaction — include tip in initial authorization.

### 5.4 Bill 96 / French Language Compliance

#### Overview

Bill 96 (enacted June 1, 2022 as "An Act respecting French, the official and common language of Québec"), with final provisions effective **June 1, 2025**, strengthens the Charter of the French Language [89][90][91].

#### French Language Requirements for Payment Terminals

**A. Customer-Facing Screens (Mandatory French):**

French must be the default language on ALL customer-facing screens on payment terminals [89][90]:

- Welcome/payment initiation screen: "Bienvenue" / "Insérez ou effleurez votre carte"
- Amount display: "Montant à payer : $XX.XX"
- Tip selection: "Pourboire" labels; percentages in French ("15%", "18%", "20%")
- Tip amount: "Pourboire suggéré : $X.XX"
- PIN entry: "Entrez votre NIP"
- Processing: "Traitement en cours..."
- Approval: "Approuvé" / "Transaction approuvée"
- Decline: "Refusé" with reason in French
- Receipt prompt: "Souhaitez-vous un reçu?" or "Reçu : Oui / Non"
- Error messages: All errors in French
- Contactless instructions: "Effleurez votre carte ou appareil mobile"
- Language toggle button: Label in French or bilingual

**B. Admin/Employee Screens (Must be available in French for Quebec employees):**

- Terminal configuration menus
- Settlement/batch reports
- Transaction lookup screens
- Error/Diagnostic screens
- Staff login screens

From Montréal International: "Software and work tools must be available in French in Québec, though bilingual interfaces are permitted" [92].

From Preply Business: "Both consumers and business clients in Quebec have the right to be informed and served in French by the enterprise they do business with" [91].

**C. Receipts (French-first):**

All customer-facing text on receipts must be in French [89][90]:
- Merchant name and address
- Transaction items and amounts
- Tip line ("Pourboire")
- Subtotal, taxes ("TPS" for GST, "TVQ" for QST), total
- Payment method labels ("Débit Interac" / "Crédit Visa" / "Crédit Mastercard")
- Return/exchange policies, terms and conditions, footer messages

#### OQLF Registration Process

Since your 25-location chain employs 25 or more employees in Quebec, you must register with the OQLF [89][90][91]:

**Step 1 — Registration:**
- Register with the OQLF via the online portal (https://www.oqlf.gouv.qc.ca)
- Provide: business name, address, Quebec Establishment Number (NEQ), number of employees, description of activities, list of all locations
- Receive a **certificate of registration**

**Step 2 — Linguistic Self-Evaluation (within 3 months):**
- Submit a linguistic self-evaluation form to the OQLF
- This assessment evaluates French integration into: internal communications, hiring practices, training, signage, IT systems (including payment terminals), external communications

**Step 3 — OQLF Assessment and Outcome:**
- **If compliant**: OQLF issues a **francization certificate**. Your company must maintain French usage and submit a **triennial report** (every three years)
- **If non-compliant**: OQLF issues a notice requiring a **francization program** to be established and implemented

**Francization Committee:** If your company has 100+ employees in Quebec, a francization committee must be formed (minimum 6 members including worker and management representatives), meeting at least once every six months [89].

#### Penalty Amounts Under Bill 96

| Offense Type | Individual (Natural Person) | Corporation (Legal Entity) |
|--------------|----------------------------|---------------------------|
| **First offense** | $700 – $7,000 | $3,000 – $30,000 |
| **Second offense** | $1,400 – $14,000 | $6,000 – $60,000 |
| **Third+ offense** | $2,100 – $21,000 | $9,000 – $90,000 |

**Daily penalties**: Each day of non-compliance constitutes a separate offense. Maximum potential fines can reach **$90,000 per day** for a corporation on a third offense [89][90][91].

**Other consequences of non-compliance** [91]:
- Suspension or cancellation of business permits or certificates
- Removal or destruction of non-compliant exterior commercial advertising
- Injunctions from the Quebec Superior Court
- Contracts that violate Bill 96 rules may be declared null and void
- Public registry of non-compliance: The OQLF will publish a list of organizations with denied, suspended, or canceled certificates

**OQLF enforcement powers** [89][90]:
- Conduct investigations to verify compliance
- Enter any location at any reasonable time
- Take pictures during inspections
- Access data stored on electronic devices during inspections
- Apply for injunctions with the Superior Court of Quebec

#### Stepwise Validation Procedure

**Step A: Contractually Require French as Default Language**

Before signing any agreement, include the following contractual requirements:

1. **Terminal software image**: Require that all terminals deployed in Quebec locations are pre-configured with **French as the default language** on all customer-facing screens, admin screens, and receipt templates
2. **Language availability**: Contractually require the processor to confirm that French language is available on the terminal model chosen, the French translation is complete and accurate, the terminal can switch between French and English without data loss, and any software updates maintain French language support
3. **Non-compliance liability**: Include provisions holding the processor responsible if their software fails to meet Bill 96 requirements, including any fines or penalties incurred
4. **Support in French**: Require that technical support for Quebec locations is available in French, with defined response times

**Step B: Audit Checklist for Complete French Coverage**

Conduct a comprehensive audit of all customer-facing and employee-facing touchpoints (see detailed checklist in Section 5.4 above).

**Step C: OQLF Registration**

Complete the registration process as outlined above.

**New Brunswick Note**: Bill 96 does **not** apply in New Brunswick. New Brunswick is Canada's only officially bilingual province, but the Official Languages Act applies to **government institutions**, not private businesses [93]. Private businesses in New Brunswick are not required by provincial law to have French as the default language on payment terminals, signage, or menus.

---

## 6. Kitchen Display System (KDS) Compatibility

### 6.1 Moneris Go Restaurant KDS — Not Compatible with Oracle Simphony

The Moneris Go Restaurant KDS is an iPad-based Kitchen Display System designed specifically for the Moneris Go Restaurant POS ecosystem [94][95].

**Key Facts:**
- Purpose-built for "small, single-location restaurants" using Moneris Go Restaurant POS [96]
- Does **NOT** integrate with Oracle MICROS Simphony
- Moneris explicitly states the system integrates with "Moneris Go Restaurant POS" and TouchBistro, not Oracle Simphony [96]
- Gad Elharrar, VP of Small-to-Medium Business Product at Moneris: "Our expanded partnership with TouchBistro allows us to support full-service restaurants and those scaling up" [96]

**Pricing**: $9/month add-on to Moneris Go Restaurant ($30/month base) [95]

**Availability in Quebec**: Moneris has a dedicated Quebec page at moneris.com/en/canada/quebec stating: "From language to local presence, our POS systems are made for Quebec" [97]. However, the KDS app itself (ca.moneris.kitchendisplay) is not confirmed to have French language support in its interface.

**Verdict**: Not compatible with Oracle Simphony. Do not consider for this deployment.

### 6.2 Oracle MICROS Express Station 400 — Recommended

The Oracle MICROS Express Station 400 is Oracle's purpose-built, durable Kitchen Display System designed for demanding kitchen environments [98][99][100].

**Native Simphony Integration:**
- Designed from the ground up to run Simphony KDS software
- Per Oracle's installation guide: "Once up and running, the system will be ready to start fielding orders from a Simphony restaurant point of sale system" [100]
- Setup involves entering the Oracle MICROS Simphony Home URL, selecting the property, and choosing "KDS" as the device type [100]

**Technical Specifications** [98][99]:

| Specification | Value |
|---------------|-------|
| **Display** | 23.8-inch widescreen HD (1920x1080) |
| **Touchscreen** | PCAP (Projected Capacitive) |
| **Processor** | Intel Atom x5 Dual-Core (Station 400) / Quad-Core (Station 410) |
| **Memory** | 4 GB RAM standard (expandable to 8 GB) |
| **Storage** | 64 GB SATA SSD (Station 400) / 128 GB (Station 410) |
| **OS** | Windows 10 IoT Enterprise LTSC OR Oracle Linux for MICROS |
| **IP Rating** | IP-54 (with port plugs) — sealed against humidity, grease, dust |
| **Temperature** | 0°C to 60°C |
| **Cooling** | Fanless, ventless design (passive cooling) |
| **Connectivity** | 4 USB Type-A, 1 GbE RJ45, 2 Serial ports, USB Type-C with DisplayPort, Oracle IDN port for kitchen printer |
| **Mounting** | VESA 100 compatible (ceiling, wall, countertop) |
| **Warranty** | 15-year Intel processor lifecycle availability |

**French Language Support — CONFIRMED:**

Oracle's official "Translation for KDS" documentation explicitly lists French as a supported language across ALL KDS Display platforms (Windows 32-bit, Windows CE, and RDC devices) [101].

Key details:
- "English is the default language. Each property can support up to four languages (1–4)" [102]
- Privileged users can add and set up to four alternate languages per property
- The KDS Configuration and User Guide confirms that configuring languages is part of the KDS Controller setup [103]
- French is supported on all platforms (unlike Arabic, Chinese, Japanese, Korean, and Thai which are supported only on Windows 32-bit devices) [101]

**Pricing:**
- Specific pricing not publicly listed
- Oracle offers "Upgrade Your POS Hardware for $1" program for existing customers [104]
- Third-party resellers suggest "KDS monitors average $700 each with discounts based on hardware quantity" [105]

**Verdict: Recommended.** Native integration with Simphony, confirmed French language support, ruggedized for kitchen environments.

### 6.3 PAX Elys Display K20/K21 — Indirect Integration via Middleware

The PAX Elys Display is an Android-based Kitchen Display System available in two models: K2160 (16-inch) and K2220 (22-inch) [106][107][108].

**Technical Specifications** [106][107]:

| Specification | Value |
|---------------|-------|
| **OS** | Android 11 |
| **Processor** | Quad-core Cortex-A55 2.0GHz |
| **RAM** | 2 GB |
| **Storage** | 32 GB internal (expandable via microSD to 256 GB) |
| **Display** | 16" or 22" touchscreen LCD |
| **IP Rating** | IP55 water-resistant |
| **Connectivity** | USB Type-C, USB Type-A, LAN, HDMI Out, dual-band Wi-Fi, Bluetooth 5.0 |
| **Power** | Built-in power AND Power over Ethernet (PoE) |
| **Mounting** | Countertop or wall mount (VESA compatible) |
| **Additional** | Built-in gravity sensor (gyroscope) for portrait/landscape orientation, dual speakers |

**Oracle Simphony Integration:**

The available documentation does **NOT** explicitly confirm direct integration between the PAX Elys Display KDS and Oracle MICROS Simphony [106][107][108].

**Indirect Integration via Touché Middleware:**

PAX Technology has partnered with **Touché**, a software provider for F&B and Hospitality, to "integrate Touché's solution with PAX's Android-based SmartPOS devices and Oracle's MICROS Simphony POS system" [62].

However, this Touché+PAX+Oracle partnership is specifically about payment/POS terminals (PAX A920, A930, A77, A920Pro SmartPOS devices) for Pay@Table, Order@Table, loyalty, and Order&Pay functionality — it is **NOT specifically about the Elys Display KDS**[62].

**French Language Support:**
The PAX Elys Display runs Android 11, which supports French as a system language. However, the KDS software running on the device determines the language of the interface. No explicit confirmation of French-language UI for the PAX Elys KDS software itself was found.

**Verdict:** Integration is uncertain and would likely require custom middleware development. The Touché partnership provides a potential bridge for PAX payment terminals to Simphony, but the Elys Display KDS is not part of that confirmed integration path.

### 6.4 Alternative KDS Solutions

**TechRyde AnyPOSConnector** [109]:
- POS integration tool that connects restaurant POS systems with third-party KDS
- "Especially Perfect for Oracle Simphony POS Integration — But Not Limited to It"
- "We've done hundreds of Simphony POS integrations and understand how to make them work with other restaurant systems"

**Ordering Stack POS-Integrator** [40]:
- "Retrieving order status from Oracle Simphony KDS" as a confirmed feature
- Integrates via Oracle's Transactional Services Gen 1 and Gen 2
- Handles "tens of millions of transactions annually for restaurant chains such as Burger King, Popeyes, and others"

**Partner Tech Corp KDS A7** [110]:
- Rugged, all-in-one KDS for restaurant environments
- 21.5-inch touchscreen, IP54 certified
- Can run Windows 10 IoT, Linux Ubuntu, or Android 11
- Could function as a KDS running Oracle Simphony KDS software if configured with Windows and the Simphony KDS client

---

## 7. Transaction Processing Speeds

### 7.1 Industry-Standard Benchmarks

The following benchmarks represent industry-standard transaction processing speeds [111][112][113]:

| Transaction Type | Typical Time | Best Case Time | Notes |
|-----------------|--------------|----------------|-------|
| **Contactless Tap (Visa/MC)** | 1–3 seconds | 0.5–1 second | EMVCo recommends ~500ms processing time |
| **EMV Chip + PIN (Dip)** | 8–15 seconds | 3–5 seconds (with Quick Chip / M/Chip Fast) | Standard EMV is slower due to online authorization |
| **Magnetic Stripe Swipe** | 2–4 seconds | 1–2 seconds | Being phased out; limited support on newer terminals |
| **Interac Debit Tap** | 1–3 seconds | 1 second | Same as contactless credit; online authorization required |
| **Interac Debit Chip+PIN** | 8–15 seconds | 3–5 seconds | Real-time MAC validation adds processing overhead |

### 7.2 Terminal-Specific Speed Capabilities

**Verifone Victa Portable (Moneris Go Terminal):**
- "Fast processing" confirmed by Moneris [7]
- Qualcomm QCM2290 processor with 4GB RAM [9]
- Battery supports "over 12 hours of continuous operation processing payments every 2 minutes" [9]
- No specific per-transaction speed published; estimated contactless tap time: 1–3 seconds
- Optimized for both "mobile and in-store experiences" [7]

**Clover Station Duo Gen 2:**
- Described as "our fastest, most customer-engaging POS solution ever, designed for speed and engagement" by Fiserv Canada [27]
- "Clover Station Duo is POS at its best, offering unparalleled transaction speed and high-level security" [114]
- 2GB RAM, 16GB flash memory, Qualcomm Snapdragon 660 octa-core processor [28][114]
- No specific per-transaction speed published; estimated contactless tap time: 1–3 seconds

**PAX A920 Pro / A920MAX:**
- PAX A920MAX claims "216% faster reading speed when compared to the A920" [53]
- PAX A920 Pro: Cortex A53 quad-core 1.4GHz processor [50]
- PAX A920MAX: Cortex A53 1.3GHz with 5G connectivity, "40% faster install speed and 216% faster reading speed" [53]
- PAX A920Pro Duo (PCI 7): Multi-core Cortex A75 and A55 [55]
- No specific per-transaction speed in seconds published; estimated contactless tap time: 1–2 seconds for A920MAX/PCI 7

### 7.3 Factors Affecting Speed in Canadian QSR Environments

1. **Processor Network Speed**: Dial-up connections slow EMV transactions due to larger data packages. Broadband (Ethernet, WiFi) is optimal [115]

2. **EMV Configuration**: Visa's Quick Chip technology "cuts as much as 18 seconds from transaction times" through software updates [116]. MasterCard's M/Chip Fast provides similar optimization [115]

3. **POS Application Optimization**: "POS applications vary in EMV transaction optimization affecting speed regardless of EMV certification" [115]

4. **Connectivity Type**: 
   - Ethernet (wired LAN) — most stable and fastest
   - Wi-Fi (2.4GHz and 5GHz) — good but subject to interference
   - 4G/LTE and 5G — reliable for mobile terminals; PAX A920MAX "5G internet, WiFi 5.0" [53]
   - The Verifone Victa Portable has "built-in independent cellular connectivity" [7]

5. **Adyen's Impact**: Adyen's unified commerce platform is shown to "reduce checkout times by up to 50%" [117]

6. **Interac Specific**: Interac transactions require MAC validation, adding processing overhead compared to credit card transactions [77]

### 7.4 QSR-Specific Context

- "In QSR, speed isn't just a feature. It's the brand promise." — IW Technologies [118]
- "Top-performing brands keep total drive-thru times under 3 minutes" [118]
- "Even a 30-second reduction in wait time can significantly improve customer satisfaction scores and drive repeat visits" — DTiQ [119]
- "More than 70% of QSR revenue comes from drive-thru customers" [119]
- The new benchmark: "Faster than the drive-thru" — completing order processing faster than the vehicle reaches the pickup window [118]

### 7.5 Peak Hour Benchmark Recommendations

For your QSR chain, the following benchmarks should be measured during peak lunch (11:30 AM – 1:30 PM) and dinner (5:30 PM – 7:30 PM) rushes:

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Contactless Tap (credit) | < 3 seconds average | Measure 50 consecutive transactions during peak |
| Contactless Tap (Interac) | < 3 seconds average | Measure 30 consecutive transactions during peak |
| Chip+PIN (credit) | < 12 seconds average | Measure 20 consecutive transactions during peak |
| Chip+PIN (Interac) | < 12 seconds average | Measure 20 consecutive transactions during peak |
| Total Checkout Time (counter) | < 60 seconds from order to completion | Time from order entry to receipt printing |
| Drive-thru Window Time | < 3 minutes total (QSR benchmark) | Time from vehicle arrival to departure |

All three terminals should meet these targets when properly configured with broadband internet connectivity (Ethernet preferred) and Quick Chip/M/Chip Fast enabled.

---

## 8. Total Cost of Ownership (5-Year Financial Model)

### 8.1 Input Assumptions

| Parameter | Value |
|-----------|-------|
| **Transactions per location per day** | 200 |
| **Operating days per month** | 30 |
| **Average ticket** | $25 CAD |
| **Credit card share** | 70% (4,200 transactions/month per location) |
| **Interac debit share** | 30% (1,800 transactions/month per location) |
| **Monthly volume per location** | $150,000 CAD |
| **Chain monthly volume (25 locations)** | $3,750,000 CAD |
| **Chain annual volume** | $45,000,000 CAD |
| **Time horizon** | 5 years (60 months) |

### 8.2 Pricing Assumptions

**Hardware Costs (CAD):**

| Device | Purchase Price | Rental Model |
|--------|---------------|--------------|
| Verifone Victa (Moneris Go Terminal) | Not available for purchase | $29.95–$34.95/month per terminal [120][121] |
| Clover Station Duo Gen 2 | $1,800–$1,995 CAD [27][28][122] | Not available |
| PAX A920 Pro | $750–$900 CAD [123][124][125] | Not available |
| PAX A920MAX | ~$900–$1,100 CAD (estimated) | Not available |

**Monthly Software/Platform Fees (CAD):**

| Device | Monthly Fee | Details |
|--------|-------------|---------|
| Verifone Victa (Moneris) | $9.95 | Account Service Package Fee [120] |
| Clover Station Duo (Fiserv) | $49.95–$89.95 | Restaurant plan tier [27][122] |
| PAX A920 Pro (PayFacto/Adyen) | $0 | No monthly software fee [126][127] |

**Processing Fee Scenarios:**

**Flat Rate Scenario** (Moneris published flat rates) [128]:

| Card Type | Rate |
|-----------|------|
| Credit (card-present) | 2.65% + $0.10 per transaction |
| Interac Debit | $0.12 per transaction |

**Interchange-Plus Scenario** (Canadian benchmarks) [129][130][131]:

| Card Type | Effective Rate |
|-----------|---------------|
| Credit (card-present) | ~1.85% + $0.10 per transaction |
| Interac Debit | ~$0.10 per transaction |

**Additional Fees:**

| Fee Type | Monthly Cost |
|----------|-------------|
| PCI Compliance Fee | $19.95/month [132][133] |
| Early Termination Fee | $300–$500 (varies by processor) |
| Battery Replacement | $50–$100 once during 5-year lifecycle |

### 8.3 Monthly Processing Cost Calculations

**Flat-Rate Scenario (Per Location):**

| Component | Calculation | Monthly Cost |
|-----------|-------------|-------------|
| Credit (70%): $105,000 volume, 4,200 trans | 2.65% × $105,000 + $0.10 × 4,200 | $2,782.50 + $420.00 = $3,202.50 |
| Interac Debit (30%): $45,000 volume, 1,800 trans | $0.12 × 1,800 | $216.00 |
| **Total Processing Fees** | | **$3,418.50** |

**Interchange-Plus Scenario (Per Location):**

| Component | Calculation | Monthly Cost |
|-----------|-------------|-------------|
| Credit (70%): $105,000 volume, 4,200 trans | 1.85% × $105,000 + $0.10 × 4,200 | $1,942.50 + $420.00 = $2,362.50 |
| Interac Debit (30%): $45,000 volume, 1,800 trans | $0.10 × 1,800 | $180.00 |
| **Total Processing Fees** | | **$2,542.50** |

**Savings from Interchange-Plus: $3,418.50 - $2,542.50 = $876.00/month per location**

### 8.4 5-Year TCO Tables

#### Per-Location TCO (Flat-Rate Scenario)

| Cost Component | Verifone Victa (Moneris Rental) | Clover Station Duo (Fiserv) | PAX A920 Pro (PayFacto) |
|----------------|-------------------------------|---------------------------|------------------------|
| **Hardware (Year 1)** | $0 (rental) | $1,800–$1,995 | $750–$900 |
| **Monthly Software/Platform** | $9.95/account fee | $49.95–$89.95 | $0 |
| **Processing (Flat Rate)** | $3,418.50/mo | $3,418.50/mo | $3,418.50/mo |
| **PCI Compliance** | $19.95/mo | $19.95/mo | $19.95/mo |
| **Battery Replacement** | $100 (once) | N/A (countertop) | $80 (once) |
| **5-Year TCO (Flat)** | **$208,804** | **$211,504** | **$207,137** |

#### Per-Location TCO (Interchange-Plus Scenario)

| Cost Component | Verifone Victa (Moneris Rental) | Clover Station Duo (Fiserv) | PAX A920 Pro (PayFacto) |
|----------------|-------------------------------|---------------------------|------------------------|
| **Hardware (Year 1)** | $0 (rental) | $1,800–$1,995 | $750–$900 |
| **Monthly Software/Platform** | $9.95/account fee | $49.95–$89.95 | $0 |
| **Processing (Interchange+)** | $2,542.50/mo | $2,542.50/mo | $2,542.50/mo |
| **PCI Compliance** | $19.95/mo | $19.95/mo | $19.95/mo |
| **Battery Replacement** | $100 (once) | N/A (countertop) | $80 (once) |
| **5-Year TCO (I+)** | **$156,244** | **$158,944** | **$154,577** |

#### Chain-Wide TCO (25 Locations) — Interchange-Plus Scenario

| Cost Component | Verifone Victa (Moneris Rental) | Clover Station Duo (Fiserv) | PAX A920 Pro (PayFacto) |
|----------------|-------------------------------|---------------------------|------------------------|
| **Hardware/Non-Processing** | $92,350 | $159,850 | $50,675 |
| **Processing (Interchange+)** | $3,813,750 | $3,813,750 | $3,813,750 |
| **Total 5-Year Chain TCO** | **$3,906,100** | **$3,973,600** | **$3,864,425** |

#### Chain-Wide TCO (25 Locations) — Flat-Rate Scenario

| Cost Component | Verifone Victa (Moneris Rental) | Clover Station Duo (Fiserv) | PAX A920 Pro (PayFacto) |
|----------------|-------------------------------|---------------------------|------------------------|
| **Hardware/Non-Processing** | $92,350 | $159,850 | $50,675 |
| **Processing (Flat Rate)** | $5,127,750 | $5,127,750 | $5,127,750 |
| **Total 5-Year Chain TCO** | **$5,220,100** | **$5,287,600** | **$5,178,425** |

### 8.5 Key TCO Insights

1. **Processing Fees Dominate TCO (97–99%)**: Across all three solutions and both pricing scenarios, processing fees represent 96% to 99% of total 5-year cost [129][130].

2. **Interchange-Plus Saves ~$1.36M Over 5 Years**: Switching from flat-rate to interchange-plus pricing saves approximately $876/month per location = $10,512/year per location = $262,800/year chain-wide = **$1,314,000–$1,360,000 over 5 years chain-wide**.

3. **Lowest TCO Solution**: PAX A920 Pro via PayFacto (Interchange-Plus) at **$3,864,425 over 5 years chain-wide**, due to lowest hardware cost ($750/terminal vs $1,800 Clover or $1,800+ rental Moneris), no monthly software/platform fees, and competitive Canadian interchange-plus pricing.

4. **Adyen Alternative**: If using Adyen (interchange + 0.60% + $0.13) [126], the processing cost would be higher than Canadian processors: credit effective ~2.10% + $0.13, narrowing the savings vs flat-rate.

5. **Hidden Fees Are Minor**: PCI compliance fees (~$29,925 over 5 years chain-wide) and account service fees (~$14,925 over 5 years chain-wide for Moneris) represent <1% of total TCO.

---

## 9. Technical Support in French

### 9.1 Ranking by French-Language Support Quality

**1st: PayFacto (Montreal) — Strongest French Support**

- **Headquarters**: 1 Place du Commerce, Verdun, QC, Canada H3E 1A2 [134]
- **CEO**: Martin Leroux (French Canadian) [135]
- **24/7 Technical Support**: 1-888-800-6622 [136]
- **Canada General Support**: 1-877-341-8293 [137]
- **French Language**: Canadian French is the company's default language
- **Serves**: "Large Quebec restaurant chains" explicitly [138]
- **Reputation**: 4.0/5 Glassdoor rating in Montreal [139]; described as "a leader in payment solutions and hospitality technology" by Restaurants Canada [140]
- **PAX Expertise**: First Canadian company to achieve Class A Certification for PAX A920 [59]; successfully migrated 25,000 terminals in Quebec within 10 weeks [138]
- **Additional**: In-person training available, comprehensive documentation in French [136]

**2nd: Fiserv Canada (Clover) — Established French Infrastructure**

- **French Canadian Website**: merchants.fiserv.com/fr-ca [141]
- **24/7 Phone Support**: 1-888-263-1938 (French available) [142][143]
- **Canadian Presence**: "Hundreds of Canada-based highly experienced and knowledgeable staff" [44]
- **20+ Years**: "Delivering POS solutions to tens of thousands of Canadian small businesses for more than 20 years" [44]
- **Bilingual Staffing**: French-bilingual Client Operations Support Associate positions confirmed [144]
- **Code of Conduct Process**: Available in French [145]
- **Note**: Overall customer support quality rated 2.5/5; complaints about slow response times [146]

**3rd: Moneris Canada — French Services Confirmed**

- **24/7 Customer Care**: 1-866-319-7450 [147]
- **French Language Services**: Listed as an advantage on PaymentGateway.ca [148]
- **Bilingual Staff**: Bilingual Customer Service Representative - Technical positions confirmed for Quebec, QC [149]
- **Field Services**: National Field Services team for on-site support [150]
- **Note**: Overall customer support quality rated 1.7/5 on Google; 25% issue resolution rate [151]

**4th: Global Payments Canada — Quebec Office Present**

- **24/7 Support**: 1-800-361-8170 [152]
- **Quebec Office**: 1155 Rene-Levesque Blvd. West, Suite 1007, Montreal, QC H3B 2J2 [153][154]
- **Bilingual Staff**: Bilingual (FR/EN) Customer Success Manager positions confirmed [155]
- **Note**: Overall customer satisfaction very low (1.0-1.1/5 ratings) [156]; complaints about deceptive fees and contract lock-ins

**5th: Adyen Canada — Limited French Support**

- **No dedicated French-language support confirmed** for Canada [157][158]
- **No phone support or live chat** available [157][158]
- **Support**: Primarily ticket/email-based via contact form [157][158]
- **Canadian Office**: Toronto (opened 2023) — relatively recent presence [159]
- **Terminal UI**: Supports French on device interface [160]
- **Best Suited For**: Enterprise-level global merchants, not day-to-day French-language technical support [161]

### 9.2 Canadian French vs. European French

All five providers confirmed to offer **Canadian French** support:
- **PayFacto**: Montreal-headquartered, Quebec-based workforce — Canadian French default
- **Fiserv Canada**: Dedicated Canadian French website with Quebec-specific terminology — Canadian French
- **Moneris Canada**: Canadian company with Quebec-specific job postings — Canadian French
- **Global Payments Canada**: Quebec office in Montreal — Canadian French
- **Adyen Canada**: As a Dutch company, any French support would default to European French unless specifically arranged

---

## 10. Pilot Deployment Plan (30–60 Days)

### Phase 1: Preparation (Days 1–10)

**Objective**: Select pilot locations, configure terminals, establish baseline metrics.

**Selection Criteria:**
- 2–3 pilot locations: 1 high-volume Quebec location (200+ transactions/day), 1 medium-volume Quebec location (150 transactions/day), 1 New Brunswick location
- Mix of peak-hour transaction volumes
- Locations with stable Wi-Fi and cellular coverage

**Pre-Deployment Checklist:**

| Task | Responsible | Timeline |
|------|-------------|----------|
| Terminal hardware ordered and received | Procurement | Days 1–5 |
| Processor contract signed with interchange-plus pricing | Legal/Finance | Days 1–7 |
| French language image configured on terminals (Quebec locations) | Processor/VAR | Days 3–8 |
| Oracle MICROS Simphony OPI integration configured in sandbox | IT/POS Team | Days 1–7 |
| Tip-on-subtotal setting enabled for Quebec locations | Processor | Days 3–8 |
| Receipt templates audited for French compliance | Compliance | Days 1–5 |
| OQLF registration initiated (if not already done) | Compliance | Days 1–5 |
| Staff training materials prepared in French and English | Operations | Days 5–10 |
| KDS integration tested (Oracle Express Station 400) | IT/POS Team | Days 5–10 |
| Battery charging stations installed at each pilot location | Operations | Days 5–10 |
| Network connectivity verified (Wi-Fi and cellular backup) | IT | Days 5–10 |
| Baseline metrics recorded (current transaction speed, decline rates) | Operations | Days 8–10 |

### Phase 2: Soft Launch (Days 11–25)

**Objective**: Deploy terminals in a controlled environment with monitoring.

**Deployment:**
- Install terminals at 1–2 counter positions per location
- Run parallel with existing payment system for first 3 days
- Monitor transaction success rates, speed, and staff comfort

**Daily Monitoring:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Contactless tap speed | < 3 seconds average | Measure 50 peak-hour transactions |
| Chip+PIN speed | < 12 seconds average | Measure 30 peak-hour transactions |
| Transaction success rate | > 99% | Track declines + timeouts |
| Battery drain rate | < 15% per hour | Log battery % hourly |
| French language accuracy | 100% screens, 0 errors | Manual audit at each location |
| French receipt accuracy | 100% French-first with proper accents | Print and verify 20 receipts |
| Decline rate | < 3% | Compare to baseline |
| Staff comfort rating | > 7/10 | Weekly staff survey |
| Customer feedback | Positive | Informal feedback collection |

**Issue Resolution:**
- Document all issues with timestamps, screenshots, and severity
- Escalate critical issues (payment failures, integration errors) within 2 hours
- Track resolution time per issue category

### Phase 3: Full Launch at Pilot Sites (Days 26–45)

**Objective**: Full deployment at pilot locations with comprehensive testing.

**Testing Protocols:**

| Evaluation Dimension | Test Protocol | Success Metric |
|---------------------|---------------|----------------|
| **Transaction Speed** | Measure 50 consecutive transactions during peak lunch (11:30–1:30) | Average < 3 sec tap; < 12 sec chip+PIN |
| **Offline Mode** | Simulate internet outage; process 10 credit transactions | All 10 stored and forwarded successfully; no data loss |
| **Interac Debit Offline** | Attempt Interac transaction during simulated outage | Transaction declined with clear French error message |
| **OPI Integration** | Process 100 transactions through Oracle MICROS OPI | 0 OPI-related errors; all transactions settle correctly |
| **French Language** | Audit all screens (customer + admin) for French accuracy | 100% customer-facing screens in French; 0 translation errors |
| **Bill 72 Tip Calculation** | Process 50 transactions with tips in Quebec location | Tips calculated on pre-tax subtotal |
| **Battery Life** | Run terminal from full charge to shutdown during normal operations | Minimum 8 hours before low-battery warning |
| **KDS Integration** | Process 50 orders and verify KDS display | 100% orders displayed correctly; 0 missed orders |
| **Receipt Accuracy** | Print 50 receipts and verify French content | 100% French-first with proper accents |
| **Cellular Backup** | Disable Wi-Fi; verify cellular connectivity | Automatic failover within 30 seconds |

### Phase 4: Decision Gate and Evaluation (Days 46–60)

**Objective**: Analyze pilot data and make go/no-go decision for full rollout.

**Quantitative Metrics:**

| Metric | Target | Weight |
|--------|--------|--------|
| Average contactless tap speed | < 3 seconds | 15% |
| Average chip+PIN speed | < 12 seconds | 10% |
| Transaction success rate | > 99% | 15% |
| Offline store-and-forward success | 100% | 10% |
| French language compliance | 100% | 20% |
| Battery life | > 8 hours | 5% |
| KDS integration accuracy | 100% | 10% |
| Staff satisfaction | > 7/10 | 10% |
| Customer satisfaction | > 8/10 | 5% |

**Decision Gate Criteria:**

- **Go for full rollout**: All critical metrics meet or exceed targets (weighted score > 90%)
- **Conditional go**: Minor issues identified with clear remediation plan (weighted score > 75%)
- **No-go**: Major integration failures, critical language compliance gaps, or unacceptable performance (weighted score < 75%)

**Full Rollout Plan (if approved):**
- 5 locations per week (5 weeks total for 25 locations)
- 2-day training per location (pre-deployment + go-live support)
- 2-week hyper-care period per location
- Monthly compliance audits for first 6 months
- Quarterly OQLF reporting for Quebec locations

---

## 11. Final Recommendations

### 11.1 Primary Recommendation: PAX A920 Pro / A920MAX / A920Pro PCI 7

**Chosen Configuration: PAX A920 Pro or A920MAX** with interchange-plus pricing through **PayFacto** (Montreal-based) or **Adyen Canada**.

**Rationale:**

| Strength | Detail |
|----------|--------|
| **Strongest OPI Integration** | Certified OPI 6.2+ integration through Adyen [22], Global Payments [21], and PayFacto's SecureTablePay middleware [60]. Multiple certified processor pathways provide flexibility. |
| **Lowest 5-Year TCO** | $3,864,425 chain-wide (Interchange-Plus) — lowest of all three solutions. Savings of ~$1.36M vs. flat-rate pricing. |
| **Active Canadian Certification** | PayFacto achieved Class A Certification for A920 in Canada [59]. Available through multiple Canadian distributors. |
| **Android Platform** | Full French language localization support. Android 14 on A920Pro PCI 7. PAXSTORE marketplace with 8,000+ apps. |
| **Future-Proof** | A920Pro PCI 7 (PCI PTS 7.x) provides longest field life. 6000mAh battery supports full double shifts. |
| **No Processor Lock-In** | Terminals can be switched between processors (PayFacto, Adyen, Global Payments) if needed. |
| **French Support** | PayFacto Montreal provides best-in-class French-language technical support. |

**Recommended Configuration by Location Type:**

| Location Volume | Terminal Model | Processor |
|----------------|---------------|-----------|
| High-volume (200+ trans/day) | PAX A920MAX (5G) or A920Pro PCI 7 | PayFacto (interchange-plus) |
| Standard volume (100–200 trans/day) | PAX A920 Pro | PayFacto or Adyen (interchange-plus) |
| New Brunswick locations | PAX A920 Pro | PayFacto or Adyen (interchange-plus) |

**KDS Strategy:**
- Maintain existing Oracle MICROS KDS setup for immediate deployment
- For new KDS deployments: **Oracle MICROS Express Station 400** (native integration, French language confirmed) [98][99][101]
- Monitor PAX Elys Display + Touché middleware integration as future option [62][106]

### 11.2 Secondary Recommendation: Verifone Victa Portable (Moneris Go Terminal)

**Chosen Configuration: Verifone Victa Portable (Moneris Go Terminal)** via rental model with interchange-plus pricing (if available) or flat-rate processing.

**Rationale:**
- **Zero upfront hardware cost**: Rental model at $29.95–$34.95/month [120][121]
- **Newest device in Canada**: Launched February 2026, PCI 7-ready, Android 13 upgradable to 14 [7][9][10]
- **12+ hour battery claim**: Supports full day of operations [9]
- **Moneris' Canadian footprint**: Processes one in three transactions in Canada; Quebec offices for on-site support [7][97]

**Critical Caveats:**
- Moneris Go Restaurant KDS is **not available in Quebec and is English-only** — you must maintain your existing MICROS KDS setup or use Oracle Express Station 400 [94][95][96]
- Verifone Victa is too new to the market for independent user reviews on battery life and reliability
- OPI integration requires verification through Moneris specifically — processor-dependent
- Higher 5-year TCO than PAX ($3,906,100 vs. $3,864,425 chain-wide)

### 11.3 Not Recommended

**Verifone Carbon 8:**
- 9-year-old device running obsolete Android Lollipop (no security updates)
- Original hardware revision's PCI PTS certification expired April 30, 2024 [1]
- Not available from any authorized Canadian channel for new deployments [2][4][5]
- Replaced by Verifone Victa Portable [7]

**Clover Station Pro / Station Duo Gen 2:**
- **No certified direct integration with Oracle MICROS Simphony via OPI** — Clover is absent from Oracle's payment integration partner list [30]
- Custom middleware development would be required (CloverConnector SDKs) with ongoing maintenance costs [36][38]
- Higher TCO: $3,973,600 chain-wide vs. $3,864,425 (PAX) — $109,175 more over 5 years
- **Processor-locked to Fiserv** — cannot switch processors without replacing all terminals [41][42]
- 36-month contracts with early termination penalties ($500–$2,400) [25]

### 11.4 Action Plan

| Phase | Timeline | Actions |
|-------|----------|---------|
| **Immediate** | Next 2 weeks | 1. Contact PayFacto + Adyen Canada for PAX A920 Pro/A920MAX quotes reflecting 25-location volume 2. Request interchange-plus pricing (not flat-rate) — potential savings of $1.36M over 5 years 3. Verify OPI 6.2+ integration with your specific Oracle MICROS Simphony version |
| **Bill 96 Compliance** | Next 4 weeks | 1. Register with OQLF (mandatory for 25+ employees in Quebec) 2. Contractually require French as default language on all terminal software images 3. Conduct full audit of all customer-facing screens, admin screens, and receipt templates 4. Implement tip-on-subtotal for Quebec locations (Bill 72 compliance) |
| **KDS Strategy** | Next 4 weeks | 1. Maintain existing Oracle MICROS KDS setup for immediate deployment 2. Order Oracle Express Station 400 for any new KDS deployments 3. Monitor PAX Elys Display + Touché middleware integration as future option |
| **Pilot Program** | Days 1–60 | 1. Deploy PAX A920 Pro terminals at 2–3 locations as outlined in Pilot Deployment Plan 2. Include 1 high-volume Quebec, 1 medium-volume Quebec, and 1 New Brunswick location 3. Use Oracle Express Station 400 for KDS at pilot locations 4. Evaluate against defined success metrics before full rollout |
| **Full Rollout** | Weeks 9–13 | 1. 5 locations per week (5 weeks total) 2. 2-day training per location 3. 2-week hyper-care period per location 4. Monthly compliance audits for first 6 months |
| **Ongoing** | Quarterly | 1. Quarterly OQLF reporting for Quebec locations 2. Annual PCI compliance review 3. Semi-annual processing rate renegotiation 4. Annual French language compliance audit |

---

## Sources

[1] PCI Security Standards Council — PTS Device Listing (Carbon 8 approval 4-10209): https://listings.pcisecuritystandards.org/popups/pts_device.php?appnum=4-10209

[2] Verifone Documentation Portal — Carbon 8 Device Specs: https://docs.verifone.com/device-installation-guides/installation-guides/device-installation-guides/android-devices/carbon

[3] Android OS endoflife.date — Lollipop EOL: https://endoflife.date/android

[4] Verifone Retail Product Page (current): https://www.verifone.com/retail

[5] Moneris All Devices Catalog: https://www.moneris.com/en/support/devices/all-devices

[6] Financial IT — Verifone Launches Carbon 8 (May 10, 2017): https://financialit.net/news/payments/verifone-launches-carbon-8-portable-pos-terminal

[7] Moneris Press Release — Expands Go Commerce Suite (Feb 3, 2026): https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite

[8] Verifone Documentation — Application Development Kit (Version 4.7) Release Notes: https://verifone.cloud/docs/application-development-kit-version-47/Release_Notes_ADK_4.7.45

[9] Verifone Documentation Portal — Victa Installation Guides: https://docs.verifone.com/device-installation-guides/installation-guides/device-installation-guides/victa

[10] Verifone — Victa Portable Product Page: https://www.verifone.com/en-us/hardware-product/verifone-victa-portable

[11] Verifone — Victa Product Family Overview: https://www.verifone.com/en-us/verifone-victa

[12] YouTube — TRANSACT 2026: Moneris + Verifone Victa Portable: https://www.youtube.com/watch?v=unkkPdYrqbs

[13] LinkedIn / Verifone Post — "Congratulations to Moneris on expanding the Moneris Go": https://www.linkedin.com/posts/verifone_congratulations-to-moneris-on-expanding-the-activity-7424533971180417024-DDdp

[14] Scribd — Verifone XPI Integration Overview: https://www.scribd.com/document/676020740/XPI

[15] Verifone Cloud Device Integration (VCDI) Documentation: https://verifone.cloud/docs

[16] Verifone — Seamless API Integration: https://www.verifone.com/en-us/verifone-api

[17] Adyen Docs — Deprecation of eVo terminals: https://docs.adyen.com/point-of-sale/user-manuals/terminal-deprecation

[18] Oracle Hospitality — Payment Interface System: https://www.oracle.com/hospitality/products/payment-interface

[19] Oracle Hospitality Simphony Configuration Guide — OPI: https://docs.oracle.com/cd/E76065_01/doc.29/e69879/c_payments_opi.htm

[20] Moneris — ERP Integration: https://www.moneris.com/en/support/products/erp-integration

[21] Global Payments Developer Portal — OPI: https://developer.globalpayments.com/heartland/payments/in-store/pos-middleware/oracle-payment-interface

[22] Adyen Docs — Oracle Simphony Integration: https://docs.adyen.com/plugins/oracle-simphony

[23] Clover devices — Technical specifications: https://docs.clover.com/dev/docs/clover-devices-tech-specs

[24] Free Clover Equipment Replacements / Clover End-of-Support Fee: https://nationwidepaymentsystems.com/free-clover-equipment-replacements

[25] Limelight Payments — Clover POS Cost & Pricing 2026: https://www.limelightpayments.com/blog/clover-pos-cost-pricing-2025-hardware-clover-fees-processing-fees

[26] Clover Devices — Technical Specifications (Updated): https://docs.clover.com/dev/docs/clover-devices-tech-specs

[27] Clover Station Duo 2 Specifications (PDF): https://www.merchantindustry.com/wp-content/uploads/2023/12/Clover-Station-Duo-2-Specifications-012024.pdf

[28] Clover Station Duo | Brilliant POS: https://brilliantpos.com/clover/clover-station-duo

[29] Clover Station Solo | Brilliant POS: https://brilliantpos.com/clover/clover-station-solo

[30] Oracle — POS Integrations: https://www.oracle.com/food-beverage/restaurant-pos-systems/pos-integrations

[31] HotelTechReport — Clover vs Oracle Simphony Comparison: https://hoteltechreport.com/compare/clover-vs-oracle-micros-pos

[32] Elavon WorksWith — Oracle Food & Beverage: https://workswith.elavon.com/en-CA/apps/392187/oracle-food-beverage

[33] FreedomPay — Oracle OPI Support for OPERA and MICROS: https://corporate.freedompay.com/about-us/news/freedompay-announces-oracle-payment-interface-opi-support-for-opera-and-micros-products

[34] Viva.com — Oracle OPI Integration: https://developer.viva.com/apis-for-point-of-sale/card-terminals-devices/oracle-opi

[35] Oracle — Integrate with Simphony Point of Sale: https://www.oracle.com/food-beverage/restaurant-pos-systems/pos-integrations/partners

[36] Clover Developer Docs — Semi-Integration Basics: https://docs.clover.com/dev/docs/clover-development-basics-semi

[37] Clover Developer Docs — PAAS Integration Options: https://docs.clover.com/dev/docs/paas-integration-options

[38] CloverConnector Overview (v4.0.0): https://clover.github.io/remote-pay-windows/4.0.0/cloverconnector/html/index.html

[39] Clover Developer Docs — Canada merchants: https://docs.clover.com/dev/docs/canadian-merchants

[40] Ordering Stack — Oracle Simphony Integration: https://orderingstack.com/oracle-simphony-integration

[41] Reddit — Clover only Fiserv processing: https://www.reddit.com/r/smallbusiness/comments/11dtgxz/clover_fiserv_only

[42] NerdWallet — Clover POS Review: https://www.nerdwallet.com/article/small-business/clover-pos-review

[43] Payments Dive — Fiserv Clover growth goals: https://www.paymentsdive.com/news/fiserv-clover-growth-goals-square-jack-dorsey-smb-merchant-pos-software-services/711202

[44] Fiserv Canada — Clover Solutions: https://merchants.fiserv.com/en-ca

[45] LinkedIn — TD Bank partners with Fiserv, RBC/BMO explore Moneris sale: https://www.linkedin.com/posts/fathom4sight_td-fiserv-clover-activity-7363615779008761861-s0Sv

[46] NCFA Canada — TD Partners with Fiserv (July 23, 2025): https://ncfacanada.org/td-partners-with-fiserv-and-sells-merchant-portfolio

[47] Clover Developer Docs — Launch in the Canadian App Market: https://docs.clover.com/dev/docs/international-app-market-readiness

[48] Clover — Change device language: https://www.clover.com/en-US/help/change-clover-device-language

[49] Clover Developer Docs — Update multi-lingual support (Canadian French): https://docs.clover.com/dev/docs/multilingual-support-1

[50] PAX A920Pro Datasheet (Official PDF): https://www.pax.us/wp-content/uploads/2023/12/A920Pro-Datasheet.pdf

[51] PAX Technology — A920Pro product page: https://www.paxtechnology.com/a920pro

[52] PayFacto — Mobile Terminals: https://payfacto.com/mobile-terminals

[53] PAX Technology — A920MAX Product Page: https://www.paxtechnology.com/a920max

[54] PAX.us — A920MAX product page: https://www.pax.us/products/mobile/a920max

[55] PAX Technology — A920Pro Duo PCI 7: https://www.pax.us/product/a920pro-duo

[56] PCI Security Standards Council — PTS POI Device Listing (PAX): https://www.pcisecuritystandards.org/popups/p2pe_app_device.php?reference=2020-01242.002

[57] PAX — First to Achieve PCI PTS POI v7.0: https://www.pax.com.cn/PAX-First-to-Achieve-PCI7-PTS-POI-Certification-in-Payment-Terminals

[58] Adyen Help — API Integrated Terminals: https://help.adyen.com/knowledge/in-person-payments/get-started/how-can-i-start-with-api-integrated-terminals

[59] PayFacto — Launches PAX A920 in Canada: https://payfacto.com/payfacto-launch-pax-a920-android-payment-terminals-canada

[60] Newswire — PayFacto and PAX Announce A920 Launch in Canada: https://www.newswire.ca/news-releases/payfacto-and-pax-announce-the-launch-of-the-pax-a920-android-payment-terminals-in-canada-848325098.html

[61] The Paypers — PayFacto, PAX launch A920 in Canada: https://thepaypers.com/payments/news/payfacto-pax-launch-pax-a920-payment-terminals-in-canada

[62] PAX Technology Blog — Touché Deploys Oracle Solution on PAX Android: https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android

[63] PAX Technology — PAXSTORE Marketplace: https://www.pax.us/marketplace

[64] PAXSTORE Features PDF: https://uploads.strikinglycdn.com/files/ad04d97b-f19d-4172-9195-c5328bde725d/PAXSTORE%20Features_V10.pdf

[65] PAXSTORE Security Documentation: https://www.paxstore.net/security

[66] PAXSTORE App Marketplace: https://www.paxstore.net/apps

[67] Integrate Payments — Pax SDK API Integration: https://www.integratepayments.com/payment-gateway/pax-payment-machine-api-sdk-integration

[68] North Developer — PAX SI SDK: https://developer.north.com/products/in-person/semi-integrated/pax-si-sdk

[69] Chetu — PAX Integration Services: https://www.chetu.com/payments/pax.php

[70] PAX Technology — Developer Portal: https://www.pax.us/support/developer-portal

[71] PAX A920 Pro Security Policy: https://www.pax.us/security

[72] Behance — PAX France UX/UI Design: https://www.behance.net/gallery/PAX-France-UX-UI

[73] Cardium — Montreal PAX Distributor: https://www.cardium.ca

[74] Interac — Interac Debit 101: https://www.interac.ca/en/consumers/debit-101

[75] Clearly Payments — How Interac and Debit Cards Work in Canada: https://www.clearlypayments.com/blog/an-introduction-to-interac-and-debit-cards-in-canada

[76] Square — Interac Debit Guide: https://squareup.com/ca/en/the-bottom-line/operating-your-business/interac-debit

[77] Paylosophy — Handling Interac Payments in Canada: https://paylosophy.com/handling-interac-payments-canada

[78] Global Payments Integrated — Store and Forward Application: https://www.globalpaymentsintegrated.com/store-and-forward

[79] EMVCo — EMV Specifications: https://www.emvco.com/emv-technologies

[80] PaymentGateway.ca — Canada Contactless Payment Limits in 2026: https://paymentgateway.ca/contactless-payment-limits-canada-2026

[81] Digital Transactions — Card Networks Up Canadian Contactless Limits: https://www.digitaltransactions.net/card-networks-up-canadian-contactless-transaction-limits-to-limit-physical-contact

[82] Moneris Blog — Bill 72 Compliance: https://www.moneris.com/en/blog/posts/compliance/bill-72

[83] MYR POS — Quebec's Bill 72 Law: New Tipping Rule Update: https://myrpos.com/quebec-bill-72-tipping-law

[84] Adyen Docs — Override the amount to tip on: https://docs.adyen.com/point-of-sale/design-your-payment-flow/tip-on-terminal

[85] Moneris Blog — Everything you need to know about Bill 72 in Quebec: https://www.moneris.com/en/blog/posts/compliance/bill-72-2025

[86] PayFacto — Tip Before Taxes: Bill 72 Compliant Solutions: https://payfacto.com/newsroom/bill-72-tip-before-taxes

[87] Adyen Docs — Pre-authorization and authorization adjustment: https://docs.adyen.com/online-payments/pre-authorization

[88] Adyen Partner Help Center — Tipping Options: https://help.adyen.com/knowledge/tipping-options

[89] Éducaloi — Language Laws and Doing Business in Quebec: https://educaloi.qc.ca/en/capsules/language-laws-and-doing-business-in-quebec

[90] LanguageIO — Understanding Bill 96: A Guide for Businesses in Quebec: https://languageio.com/resources/blogs/understanding-bill-96-a-guide-for-businesses-operating-in-quebec-canada

[91] Preply Business — Bill 96 Quebec (Canada) French Language Law Explained: https://preply.com/en/blog/b2b-bill-96-quebec-explained

[92] Montréal International — French is Québec's official and common language: https://www.montrealinternational.com/en/business-environment/french-language

[93] OCOLNB — Frequently Asked Questions: https://officiallanguages.nb.ca/content/frequently-asked-questions

[94] Moneris Kitchen Display App — Apple App Store (Canada): https://apps.apple.com/ca/app/moneris-kitchen-display/id1610511113

[95] Moneris Support — KDS App: https://support.moneris.com/article/moneris-go-restaurant-the-kitchen-display-system-kds-41785

[96] Yahoo Finance Canada — Moneris launches Go Restaurant POS (Oct 6, 2025): https://ca.finance.yahoo.com/news/moneris-launches-restaurant-point-sale-120500458.html

[97] Moneris — POS Systems for Quebec: https://www.moneris.com/en/canada/quebec

[98] Oracle — KDS for Restaurants: https://www.oracle.com/food-beverage/restaurant-pos-systems/kds-kitchen-display-systems

[99] Oracle MICROS Express Station 400 Datasheet (PDF): https://www.oracle.com/a/ocom/docs/industries/food-beverage/oracle-micros-express-station-400.pdf

[100] YouTube — Install Simphony on the Oracle MICROS Express Station 400: https://www.youtube.com/watch?v=3wkm8vl7yzg

[101] Oracle Docs — Translation for KDS (French Language Support): https://docs.oracle.com/cd/F14820_01/doc.191/f15055/c_internationalization_kds.htm

[102] Oracle Docs — Setting the Language for KDS Displays: https://docs.oracle.com/cd/F32325_01/doc.192/f32331/t_internationalization_kds.htm

[103] Oracle MICROS Simphony KDS Configuration and User Guide (19.7): https://docs.oracle.com/en/industries/food-beverage/simphony/19.7/kdscu/F97103_03.pdf

[104] Oracle — Upgrade Your POS Hardware for $1: https://www.oracle.com/food-beverage/restaurant-pos-systems/pos-hardware-upgrade

[105] Micros Integrated Payments — KDS: https://microsintegratedpayments.com/kitchen-display-systems-for-micros-3700-or-micros-simphony

[106] PAX Technology — Elys Display: https://www.pax.us/elys-display

[107] PAX Technology — Launches Innovative KDS and Bump Bar: https://www.pax.us/about/press-room/pax-technology-inc-launches-innovative-kds-and-bump-bar-for-restaurants

[108] PRWeb — PAX Technology Inc. Launches Innovative KDS and Bump Bar: https://www.prweb.com/releases/pax-technology-inc-launches-innovative-kds-and-bump-bar-for-restaurants-302148601.html

[109] TechRyde — AnyPOSConnector: https://www.techryde.com/anyposconnector

[110] Partner Tech Corp — KDS A7: https://www.partnertechcorp.com/us/products-detail/kitchen-display-system

[111] FIS Global — Making the Case for Contactless Payments: https://www.fisglobal.com/-/media/fisglobal/files/pdf/ebook/contactless-cards-ebook.pdf

[112] SCORE.org — Mobile and Cashless Payments Fast-Start: https://www.score.org/resource/article/mobile-and-cashless-payments-fast-start

[113] Security StackExchange — EMV Contactless Transaction Times: https://security.stackexchange.com/questions/emv-contactless-transaction-times

[114] Clearly Payments — Clover Station Duo: https://www.clearlypayments.com/products/pos/clover/clover-station-duo

[115] Ingenico — 3 Reasons Why EMV Transactions Seem Slower: https://ingenico.com/us-en/newsroom/blogs/3-reasons-why-emv-transactions-seem-slower

[116] Retail Dive — Visa Quick Chip technology: https://www.retaildive.com/news/visa-quick-chip-emv-transaction-time

[117] Financial Post — Adyen unified commerce reduces checkout times: https://financialpost.com/adyen-unified-commerce

[118] IW Technologies — QSR POS Speed: https://www.weareiw.com/qsr-pos-speed

[119] DTiQ — Enhancing QSR Speed of Service: https://www.dtiq.com/enhancing-qsr-speed-of-service

[120] Moneris — Get Started with Payment Solutions: https://www.moneris.com/en/get-started

[121] Moneris — Go Payment Terminal: https://www.moneris.com/en/solutions/terminals/moneris-go

[122] NextGen Payment Solutions — Clover Station Duo Canada: https://npscanada.com/clover-station-duo

[123] Levata Canada — PAX A920 PRO SCAN: https://www.levata.com/ca/pax-a920-pro-scan

[124] eHopper — PAX A920 Pro: https://www.ehopper.com/pax-a920-pro

[125] MOBOPAY Canada — PAX Terminals: https://www.mobopay.ca/pax-terminals

[126] Adyen — Pricing: https://www.adyen.com/pricing

[127] PayFacto — Integrated Solutions: https://payfacto.com/integrated-solutions

[128] Moneris — Your Payment Processing Costs Explained: https://www.moneris.com/en/blog/posts/pricing/keeping-it-simple--your-payment-processing-costs-explained

[129] Clearly Payments — Canadian Interchange Rates 2026: https://www.clearlypayments.com/resources/canadian-interchange-rates-2026

[130] PaymentGateway.ca — Moneris vs Helcim 2026: https://paymentgateway.ca/moneris-vs-helcim-canada

[131] Canada.ca — Code of Conduct for Payment Card Industry: https://www.canada.ca/en/financial-consumer-agency/services/industry/code-conduct-payment-card-industry.html

[132] PayFacto — PCI Compliance: https://payfacto.com/pci-dss-compliance

[133] Paystone — PCI Compliance: Why You Need It: https://www.paystone.com/resources/pci-compliance-why-you-need-it-and-how-to-do-it-right

[134] PayFacto — Contact Us: https://payfacto.com/contact-us

[135] PayFacto — About: https://payfacto.com/about

[136] PayFacto — Support: https://payfacto.com/support

[137] PayFacto — Obtaining Support Documentation: https://documentation.payfacto.com/PF_Payment/UG/CA-EN/Misc/Obtaining_Support.htm

[138] PAX Technology — Partner Success Story PayFacto: https://www.pax.us/partner-success-story-payfacto

[139] Glassdoor — PayFacto Montreal Reviews: https://www.glassdoor.com/Reviews/PayFacto-Montr%C3%A9al-Reviews-EI_IE4003351.0,8_IL.9,17_IC2296722.htm

[140] Restaurants Canada — PayFacto Supplier: https://www.restaurantscanada.org/supplier/payfacto

[141] Fiserv Canada French Website: https://merchants.fiserv.com/fr-ca

[142] Fiserv Canada — Support (French): https://merchants.fiserv.com/fr-ca/contact/support

[143] Clover Canada — Contact: https://www.clover.com/ca/contact

[144] Fiserv Careers — French-bilingual Client Operations Support: https://careers.fiserv.com/us/en/job/FFFYJUSR10389517EXTERNALENUS/French-bilingual-Client-Operations-Support-Associate

[145] Fiserv Canada — Code of Conduct: https://merchants.fiserv.com/en-ca/code-of-conduct

[146] Trustpilot — Clover Canada Reviews: https://ca.trustpilot.com/review/clover.com/ca?page=3

[147] Moneris — Customer Care Contact: https://www.moneris.com/help/MGo-A920-R3-EN/Troubleshooting/Contact_us.htm

[148] PaymentGateway.ca — Moneris Review: https://paymentgateway.ca/moneris-review

[149] Moneris — Bilingual Customer Service Job (Quebec): https://www.ziprecruiter.com/c/Moneris/Job/Representant(e)-bilingue-du-service-a-la-clientele-Technique/-in-Qu%C3%A9bec,QC?jid=757f8197595bd44a

[150] Retail Council of Canada — Moneris Solutions: https://directory.retailcouncil.org/listing/moneris-solutions-2

[151] Clearly Payments — Moneris Review: https://www.clearlypayments.com/resources/moneris-review-for-payment-processing

[152] Global Payments Canada — Contact: https://www.globalpayments.com/en-ca/contact-us

[153] Global Payments Canada — French Contact Page: https://gpnprodprilegacys3.blob.core.windows.net/corpsite/GlobalPayInc/Canada/French/contactUs-Canada.html

[154] Global Payments Canada — Office Locations: https://gpnprodprilegacys3.blob.core.windows.net/corpsite/GlobalPayInc/USA/contactUs-Canada.html

[155] Global Payments — Bilingual Customer Success Manager: https://www.tealhq.com/job/bilingual-fr-en-customer-success-manager_7ea1a56448b85bb5ddfe1974fbf7e7788fb52

[156] BestCompany.com — Global Payments Reviews: https://bestcompany.com/merchant-services/global-payments

[157] Adyen — Contact: https://www.adyen.com/contact

[158] Adyen Help — Contact: https://help.adyen.com/en_US/contact

[159] LinkedIn — Adyen Canadian Headquarters Expansion: https://www.linkedin.com/posts/julienbrault_dutch-payments-giant-adyen-has-launched-capital-activity-7349744728646598656-nxnh

[160] Adyen Docs — Supported Languages (POS terminals): https://docs.adyen.com/point-of-sale/what-we-support/supported-languages

[161] Ecommerce-Platforms.com — Adyen Review: https://ecommerce-platforms.com/ecommerce-reviews/adyen-reviews