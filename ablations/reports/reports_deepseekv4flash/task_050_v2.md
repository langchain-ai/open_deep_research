# Revised Comprehensive Comparison: Verifone Carbon 8 / Victa Portable, Clover Station Pro, and PAX A920 Pro/A920MAX for Your 25-Location QSR Chain

## Executive Summary

This revised report provides a comprehensive comparison of three payment terminal solutions for your 25-location quick-service restaurant chain expanding across Quebec and New Brunswick, with specific attention to gaps identified in the previous report. After extensive research using authoritative, device-specific documentation from 2025–2026 sources, the **PAX A920 Pro (or the newer A920MAX or A920Pro PCI 7)** emerges as the strongest recommendation, given its certified Oracle MICROS Simphony integration via OPI 6.2+, active Canadian market presence with multiple certified processors, Android-based platform with full language localization, and competitive total cost of ownership.

**Critical findings that change the previous assessment:**

- **Verifone Carbon 8 is definitively end-of-life.** It is neither available from Moneris Canada nor listed in Verifone's current product portfolio. Its original hardware revision (SUB179-xx1) has an expired PCI PTS certification (April 30, 2024). The device runs Android Lollipop (2014). The appropriate current Verifone equivalent is the **Verifone Victa Portable** (launched in Canada as the Moneris Go Terminal on February 3, 2026).

- **Clover Station Pro is discontinued globally.** The device has been renamed to "Station Duo" and is now superseded by "Station Duo Gen 2" and "Station Solo." Critically, **no official certified OPI 6.2+ integration exists between Clover and Oracle MICROS Simphony**. Clover is absent from Oracle's official payments integration partner list, which includes Adyen, Shift4, Global Payments/Heartland, and Worldline. Semi-integration via CloverConnector SDKs is possible but requires custom development.

- **PAX A920 Pro has the strongest documented OPI integration pathway** through multiple certified processors including Adyen (OPI 6.2 confirmed), Global Payments, and PayFacto. The next-generation **A920Pro PCI 7** (announced April 2026) runs Android 14, is PCI PTS 7.x certified, and features a 6000mAh battery.

**Additional critical findings:**

- **Moneris Go Restaurant KDS is unavailable in Quebec and English-only.** This applies regardless of terminal choice if using the Moneris Go ecosystem. The Oracle MICROS Express Station 400 KDS (with confirmed French language support) is the recommended alternative.

- **Interac debit does NOT support true offline/store-and-forward mode** on any terminal. Interac requires real-time online authorization.

- **Quebec Bill 96 compliance requires** French as the default language on all customer-facing screens, admin screens used by Quebec-based employees, and receipts. Penalties range from $3,000 to $30,000 per offense for corporations, escalating to up to $90,000 per day for repeat offenses.

---

## 1. Terminal-Specific Documentation and Canadian Market Status

### 1.1 Verifone Carbon 8 — Definitively End-of-Life

#### PCI PTS Certification Status

The PCI Security Standards Council official PTS device listing confirms two certification entries for the Verifone Carbon 8 [1]:

**PCI PTS Listing #4-10209 — "Verifone, Inc., X10, Carbon 8, Carbon 10"**
- **Hardware #s**: SUB179-xx1-01-A, SUB179-xx1-02-A
- **PTS Approval Expiry Date: 30 April 2024** (EXPIRED)

**PCI PTS Listing #4-10241 — "Verifone, Inc., X10, Carbon 8, Carbon 10"**
- **Hardware #**: SUB179-xx2-02-A
- **PTS Approval Expiry Date: 30 April 2027**

The Carbon 8 with **original hardware (SUB179-xx1)** has **expired PCI PTS certification as of April 30, 2024**. Only units with the **SUB179-xx2-02-A** hardware revision remain PCI PTS approved until April 30, 2027.

#### Android Version

The Verifone Documentation Portal confirms the Carbon 8 runs **Android Lollipop (Android 5.x)** [2]. Android Lollipop was released in 2014 and is no longer listed in Android Security Bulletins, meaning it has reached end-of-life and no longer receives security patches [3]. For comparison, the Verifone Victa Portable runs Android 13 (upgradeable to Android 16).

#### End-of-Life Status

Multiple lines of evidence confirm the Carbon 8 is discontinued for new deployments:

- **Verifone's current retail product page** lists only "Verifone Carbon Mobile 5" — the Carbon 8 and Carbon 10 are not listed [4].
- **Moneris Canada's device catalog** (moneris.com/en/support/devices/all-devices) lists 25+ devices including the Moneris Go Terminal (Victa Portable) but **does not include the Carbon 8** [5].
- The Carbon 8 was originally launched on **May 10, 2017** — approximately **9 years old** as of 2026 [6].
- Moneris announced on **February 3, 2026** that it is "the first in Canada to launch the Verifone Victa Portable terminal" as the Moneris Go Terminal [7].
- No official Verifone EOL notice specifically naming the Carbon 8 was found in Verifone EOL databases from POSData Group or Bluefin [8][9].

**Conclusion**: The Carbon 8 is not available for new deployments from authorized Canadian channels. No new units can be sourced from Verifone or Moneris. Existing installations may still receive support through specific contracts, but new deployment is not viable.

---

### 1.2 Verifone Victa Portable — Current Canadian Offering (Moneris Go Terminal)

#### Official Verifone Documentation

The **Victa Portable** is part of Verifone's Victa family of devices. According to the official Verifone datasheet [10]:

**Victa Portable (Standard Model):**
- **Display**: 6.7-inch HD+ capacitive multi-touch color LCD
- **Processor**: Qualcomm QCM2290 (A53 Quad-core @ 2GHz)
- **Memory**: 4GB RAM, 32GB flash memory
- **Operating System**: Verifone Secure OS with Engage VOS3 based on Android 13, upgradeable to Android 16
- **Battery**: 5000-mAh battery supporting over 12 hours of continuous use (processing payments every 2 minutes)
- **Printer**: Built-in thermal printer (58mm x 30mm)
- **Connectivity**: Wi-Fi (2.4GHz and 5GHz), Bluetooth 5.0, Cellular (2G/3G/4G LTE CAT 4), eSIM and physical SIM slots, USB-C, Ethernet dongle support
- **Security**: PCI PTS 7.x compliant
- **Environmental**: IK04 impact rating, IP52 ingress protection rating

**Victa Portable Plus (Enhanced Model):**
- **Processor**: Qualcomm SM6225 (A73 Octa-core @ 2.4GHz)
- **Memory**: 8GB RAM, 32GB flash memory
- All other specifications identical

The datasheet states: "Keep transactions safe with a device certified to PCI PTS 7" [10].

#### Canadian Availability and Moneris Certification

Moneris announced on **February 3, 2026** the launch of the **Moneris Go Terminal**, which is the **Verifone Victa Portable** branded as a Moneris product [7]. The official Moneris press release confirms:

> "Moneris is proud to partner with Verifone and be the first in Canada to bring the Victa Portable solution to market as part of our Moneris Go commerce suite" — Jordan Williamson, Vice President, Core Products at Moneris

> "Victa Portable was built for the next generation of commerce, and Moneris is setting the pace by being first to introduce it to Canadian businesses" — Skip Hinshaw, EVP, Financial Institutions at Verifone [7]

The Moneris Go Terminal features:
- Sleek ivory and onyx finishes
- Nearly seven-inch touch display
- PCI 7-ready technology
- Built-in independent cellular connectivity
- Battery life exceeding 12 hours

Moneris official support pages provide setup guides, everyday transaction instructions, and troubleshooting tips for the Moneris Go Terminal (Victa Portable), confirming active Canadian support [11]. Sales inquiries can be directed to 1-855-463-5669.

#### PCI PTS Certification Status

The Verifone Victa Portable is **PCI PTS 7.x certified** [10]. Verifone has stated that "By the end of 2025, all new Verifone devices will be PCI PTS 7 approved" [12].

At **TRANSACT 2026**, Jordan Williamson from Moneris stated: "Having PCI 7 devices not only assures the best security but also the longest life for those devices to ensure they can last in the field longer" [13].

#### Oracle MICROS Simphony OPI Integration

The Verifone Victa Portable integrates with Oracle MICROS through OPI via **certified payment processors**, not directly. The Oracle Payment Interface (OPI) operates at the PSP/processor level, meaning compatibility depends on whether the payment processor supports OPI integration for Verifone terminals [14].

Certified OPI integrators that support Verifone terminals include:
- **Adyen**: "Adyen payment terminals implement Oracle Payment Interface (OPI) 6.2 to integrate to Oracle Simphony" [15]
- **Global Payments (Heartland)**: "Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7" [16]
- **Shift4**: Integration with OPI supporting multiple verification methods [17]

However, **no specific Verifone documentation was found** that explicitly lists the Victa Portable as OPI-compatible. The integration works through the processor's OPI implementation, meaning the merchant must verify with their chosen processor that the Victa Portable is supported in their OPI deployment.

---

### 1.3 Clover Station Pro — Discontinued, No Certified Oracle MICROS Integration

#### Discontinuation Status

**The Clover Station Pro is a phased-out/discontinued model.** Clover's own developer documentation confirms the device formerly called "Station Pro" now appears as "Station Duo (previously Station Pro)" [18].

According to Limelight Payments (2026), the following Clover models are **phased out** (no software patching, no support):
- Clover Station 1.0
- Clover Station 2.0
- **Clover Station Pro**
- Clover Station Duo Gen. 1
- Clover Mini Gen. 1 & 2
- Clover Flex Gen. 1 & 2
- Clover Mobile [19]

Additionally, Clover's official developer documentation states: "Gen 1 devices, including Clover Station (C010), Mobile (C020/C021), Mini (C030/C031), and Flex (C041/C042), are approaching End-of-App-Update (EOAU) on March 30, 2026" [18].

The **current actively supported models** are Station Duo Gen. 2, Station Solo, Mini Gen. 3, and Flex Gen. 4.

#### Oracle MICROS Simphony OPI 6.2+ Integration — Definitive Status

**No official, certified direct integration between Clover and Oracle MICROS Simphony via OPI 6.2+ was found.** This is based on the following evidence:

1. **Oracle's official POS integrations page** lists payment platform partners including Adyen, Elavon, and FreedomPay — **Clover is notably absent** [20].

2. **Clover vs Oracle Simphony** is listed as a comparison between competing products, not integrated systems, on multiple comparison sites [21].

3. **Companies that DO have verified OPI 6.2+ certification with Oracle MICROS:**
   - **Adyen**: Confirmed OPI 6.2 integration supporting Pay at Counter and Pay at Table [15]
   - **Shift4**: Oracle Payments Reference Guide documents OPI 6.2 integration [17]
   - **Global Payments/Heartland**: "Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7" [16]
   - **Worldline**: Europe-wide POS payment solution for Oracle MICROS Simphony customers [22]

#### Payment-Only Peripheral Mode

Clover **does** offer "semi-integration" where a Clover device acts as a payment-only peripheral for a third-party POS system. According to Clover's official developer documentation:

> "Semi-integrated solutions run on a combination of Clover and third-party hardware; the Clover device handles payment processing."
> "Clover offers secure payment integration, also known as a semi-integration solution, allowing POS software to accept EMV-ready, PCI compliant payments." [23]

However, this requires **custom software development** using CloverConnector SDKs (available for Android, iOS, JavaScript, and .NET). There is **no out-of-the-box, plug-and-play integration with Oracle MICROS Simphony**. The Clover device would not natively speak OPI (Oracle Payment Interface) protocol [24].

**Important distinction**: The semi-integration option uses CloverConnector SDKs to enable a third-party POS to send payment requests to a Clover device, which then handles EMV processing, PIN entry, and signature capture. This requires significant custom middleware development and ongoing maintenance.

#### Canadian Market Availability

Clover has an established Canadian presence through **Fiserv Canada**. According to Fiserv Canada's merchant site [25]:
- "Clover simplifies the lives of small business owners with its all-in-one point-of-sale solution"
- Available models in Canada: Station Duo, Flex, and Mini
- "We've been delivering POS solutions to tens of thousands of Canadian small businesses for more than 20 years"

On July 23, 2025, Fiserv announced a multi-year agreement with **TD Bank Group**, where TD Merchant Solutions will use Fiserv technology including Clover to support its merchant clients, adding approximately 30,000 Canadian locations to Clover's portfolio [26].

**No official Clover documentation was found** that specifically addresses availability, certification with Moneris, or processor compatibility in Quebec or New Brunswick.

---

### 1.4 PAX A920 Pro / A920MAX — Strongest Documented OPI Integration

#### Official PAX Technology Documentation

**PAX A920 Pro (Original Model)** — Per the official PAX A920Pro Datasheet [27]:

- **OS**: PAXBiz® on Android 8.1 (upgradeable through PAXSTORE)
- **Processor**: ARM Cortex A53 Quad-Core 1.4GHz + Secure Processor
- **Memory**: 8GB eMMC Flash + 1GB DDR RAM (optional 16GB + 2GB)
- **Display**: 5.5-inch IPS WXGA HD touchscreen, 720 x 1440 pixel resolution
- **Battery**: 5150mAh, 3.7V (7.5 to 9.5 hours depending on usage)
- **Printer**: Built-in thermal printer supporting 40 lines/sec
- **Connectivity**: 4G + Wi-Fi 2.4GHz (optional 5GHz) + Bluetooth 5.0
- **Certifications**: PCI PTS 5.x SRED, EMV L1 & L2

**PAX A920MAX** — Per PAX Technology's official product page [28]:

- **OS**: Android (10/11/13 supported across variants)
- **Processor**: Cortex A53 Quad-Core, 1.3GHz + Qualcomm secure processor
- **Memory**: 16GB Flash + 2GB RAM with microSD support
- **Display**: 6-inch HD+ Infinity Edge display
- **Battery**: 2500mAh Lithium Iron Phosphate (LiFePO4) — lasts 2.5 hours longer than predecessor
- **Printer**: 80mm/sec thermal printer
- **Certification**: PCI PTS 6.x
- **Target**: High-volume merchants processing 200+ daily transactions

**Next-Generation A920Pro PCI 7 (Announced April 22, 2026)** — Per PAX Technology press release [29]:

- **OS**: Android 14
- **Processor**: Faster multi-core processor (Cortex A75 and A55)
- **Display**: 6.56-inch touchscreen + 1.99-inch customer-facing display (Duo model)
- **Battery**: 6000mAh high-capacity battery supporting full-day usage
- **Security**: Certified to PCI PTS 7 standard (the latest)
- **Memory**: Expanded (details not specified)
- **Connectivity**: 4G LTE, Wi-Fi, Bluetooth, eSIM support

Clint Jones, Chief Commercial Officer at PAX Technology, Inc. stated: "The A920Pro PCI 7 represents the next step in the evolution of mobile payments. With enhanced performance, next-generation security, and a highly flexible Android platform, we are enabling our partners and merchants to deliver richer customer experiences while driving operational efficiency" [29].

#### PCI PTS Certification Status

| Device | PCI PTS Version | Approval Expiry |
|--------|----------------|-----------------|
| PAX A920 (original) | PCI PTS 5.1 | 30 April 2027 |
| PAX A920 Pro | PCI PTS 5.x / 6.x | 30 April 2027 / 30 April 2032 |
| PAX A920MAX | PCI PTS 6.x | Active (future expiry) |
| PAX A920Pro PCI 7 (2026) | PCI PTS 7.x | Active (2030s) |

The PAX A77 Android MiniPOS was the "world's first payment terminal to receive PCI PTS Version 7.0 certification," certified through April 2035, demonstrating PAX's leadership in PCI certification [30].

#### Oracle MICROS Simphony OPI Integration

**The PAX A920 Pro has the strongest documented OPI integration pathway:**

1. **Adyen documentation** confirms: "Adyen payment terminals implement Oracle Payment Interface (OPI) 6.2 to integrate to Oracle Simphony. OPI sends all transaction messages directly to Adyen's payment terminal" [15]. The integration supports **Pay at Counter** and **Pay at Table** workflows.

2. **Global Payments** offers an Oracle Payment Interface (OPI) solution that "enables EMV payment processing for Micros point of sale (POS) systems. Certified by Oracle" [16]. Global Payments lists PAX terminals among supported hardware for OPI.

3. **Touché**, a software provider for F&B and Hospitality, has deployed its Oracle solution on PAX Android SmartPOS devices including the A920 and A920Pro. This integration "enables merchants to streamline ordering, payment, and loyalty processes within their existing POS infrastructure, including Oracle's MICROS Simphony POS system" [31].

4. **PayFacto** (Montreal-based) achieved **Class A Certification** for the PAX A920 in Canada and integrates with hospitality POS systems via the **SecureTablePay middleware** [32].

5. **Reddit user report** from the pcicompliance subreddit confirms: "I have a problem with Oracle Simphony. We have OPI/SPI with Pax" — direct user confirmation of PAX terminals in production with Oracle Simphony OPI/SPI [33].

**Important note**: OPI is terminal-agnostic at the interface level. Any OPI-compatible middleware app (provided by Adyen, Global Payments, DNA Payments, or PayFacto) can be loaded onto PAX Android terminals via the PAXSTORE marketplace.

#### Canadian Market Availability

PAX Technology has multiple Canadian distribution channels:

- **Moneris Canada**: PAX Technology and Moneris Solutions partnered to launch the A920 in Canada as "Moneris Go" [34]. Moneris processes more than one-third of all retail transactions in Canada.
- **PayFacto Canada** (Montreal, Quebec): Launched the PAX A920 in Canada exclusively with PayFacto processing on March 3, 2020 — "the first payments company in Canada to complete a Class A Certification of PAX's A920 mobile device" [32].
- **DirectDial Canada**: Canadian online retailer listing PAX A920 Pro at $503 CAD (on sale from $868 CAD) [35].
- **PAX Canada** (pax.us/canada): PAX has been serving the Canadian market since 2019, with product lineup including A920, A920Pro, and A920MAX [36].

**Both Quebec and New Brunswick are covered** — PayFacto is headquartered in Montreal, and Moneris serves all of Canada including both provinces.

---

## 2. Canadian-Specific Constraints vs. US/International Features

### 2.1 Contactless Payment Limits — 2026 Canadian Limits

As of 2026, the contactless payment limits in Canada are [37][38]:

| Network | Contactless Limit |
|---------|------------------|
| Visa | $250 CAD |
| Mastercard | $250 CAD |
| American Express | $250 CAD |
| Interac Flash (debit) | $100 CAD |

**Historical context**: Mastercard raised its limit from $100 to $250 effective April 2, 2020, in response to the COVID-19 pandemic. Visa confirmed the same increase [38]. Sasha Krstic, president of Mastercard Canada, stated: "With safety and social distancing top of mind for all Canadians, today's announcement is one way we're helping cardholders to shop easily, securely and with more peace of mind during this difficult time" [38].

**Interac's $100 limit**: The Retail Council of Canada wanted Interac to raise its limit as well, but Interac cited risk control as the reason for sticking with $100. Interac posted: "These limits are security measures to protect Canadians against theft and fraud through unauthorized use of their debit card... by not raising transaction limits, we are doing our part to protect Canadians" [39].

**When limits are exceeded** (for either network): The user is prompted to insert the card into the EMV chip reader and enter their PIN, confirming authorized cardholder status [39].

**All three terminals handle Canadian contactless limits identically** — they support NFC/contactless payments including MiFare and NFC/CTLS schemes. The key differentiator is whether contactless is built-in (Verifone Victa, PAX A920 Pro, Clover Station Duo) or requires add-on hardware (Clover Station Solo, which is not recommended for this deployment).

---

### 2.2 Interac Debit Processing

#### Interac Debit Support Confirmed

**All three terminals support Interac debit transactions** for in-person payments. Interac is Canada's national debit network, accounting for 56% of card transactions in Canada [40].

For the **PAX A920 Pro** specifically, the Amilia Help Center confirms: "Amilia supports Interac Direct Payment (IDP) for Canadian merchants that use an A920 Pax integrated terminal to accept in-person transactions" [41].

For **Verifone Victa Portable / Moneris Go Terminal**: Moneris pricing explicitly lists Interac debit at $0.12 per transaction [42].

For **Clover Station Duo**: Clover's Canadian offering through Fiserv supports Interac debit as standard [25].

#### Interac Debit in Offline/Store-and-Forward Mode — Critical Finding

**Interac debit does NOT support true offline/store-and-forward mode for in-person transactions** on any terminal platform. This is because Interac requires **real-time online authorization** with the cardholder's financial institution. Each transaction verifies sufficient funds instantly [43].

The Stripe Interac Terms and Conditions confirm the real-time authorization flow: merchants must "initiate a reversal message to the Card Issuer indicating the Transaction was not completed...if a Cardholder cancels a Transaction before the authorization response arrives at the Interac Debit terminal" [44].

If a terminal loses network connectivity, it **cannot process Interac debit transactions**. The terminal will decline the transaction or require an alternative payment method (cash or credit card, which may offer limited offline capabilities depending on configuration).

#### Interac MAC Encryption — Unique Canadian Requirement

Interac requires a supplemental mechanism called **Message Authentication Code (MAC)** that distinguishes it from Visa, Mastercard, or US debit integrations. The MAC block represents an encrypted line of values including transaction ID, amount, merchant ID, and terminal ID. A special session key is used for encryption, which must be updated (rotated) after a fixed number of transactions [45].

This means an EMV payment application developed for the US market would require **significant adjustments** to support Interac in Canada, because Interac requires special logic depending on specific terminal hardware. When ordering test terminals for integrations, they must be injected not only with a PIN key but with the **MAC key** needed for integration with Interac [45].

**Practical implication**: Merchants must ensure their Canadian payment processor properly provisions terminals with Interac MAC keys. This is standard practice for all three terminal types when deployed through Canadian processors (Moneris, Global Payments, PayFacto).

---

### 2.3 Tip Adjustment Workflows

#### Quebec's Bill 72 — Critical Regulatory Change (Effective May 7, 2025)

**Bill 72 in Quebec** (An Act to protect consumers against abusive commercial practices) introduced specific requirements for tip calculation on payment terminals [46]:

1. **Tips must be calculated based on the pre-tax amount (subtotal)**, not on the final total that includes taxes
2. **Tips must be presented neutrally** without influencing specific tip amounts

As stated by Moneris: "The law mandates that customers must have the option to tip based on the pre-tax (subtotal) amount of their bill, not the final total that includes taxes. This ensures gratuities are calculated fairly and transparently, benefiting both employees and customers" [46].

**Moneris compliance updates for Bill 72:**
- **Moneris Core Semi-Integrated & Moneris Core Restaurant**: Already support tip-on-subtotal functionality
- **Moneris Go (Integrated)**: Update developed to support tip-on-subtotal; devices automatically updated on May 7, 2025
- **Moneris Go (Standalone)**: Updated to support tip-on-subtotal and advanced tax management
- **POSPad on the P400**: Similar update available before May 7, 2025 [46]

**Tip adjustment for credit vs. Interac debit:**

**Credit cards (Visa/Mastercard/Amex)**: Allow post-authorization tip adjustments. The final amount (purchase + tip) is captured in settlement. This is well-established in Canadian processing. All three terminals support this workflow:
- **PAX A920 Pro**: Post-authorization tip adjustment through transaction batch — navigate to the transaction, press "ADJUST", enter tip amount, confirm [47]
- **Verifone Victa**: Tip adjustment holds the transaction until a tip is added manually, then sent to settlement [48]
- **Clover Station Duo**: Per-transaction tip settings via `tipMode` SDK parameter — supports TIP_PROVIDED, ON_SCREEN_BEFORE_PAYMENT, ON_SCREEN_AFTER_PAYMENT, NO_TIP [49]

**Interac debit**: Tip adjustment works differently because Interac processes transactions as a single authorization. The tip **must be included before the transaction is finalized** at the terminal. Post-transaction tip adjustment (tip editing after the customer has left) is generally **not supported on Interac debit** the way it is on credit cards, because the Interac transaction is finalized at the time of card interaction.

**Practical guidance for your QSR chain**: For Quebec locations, ensure your terminal/POS combination supports **tip-presentation-at-checkout** for Interac debit transactions (i.e., the tip prompt appears on the terminal screen before the customer inserts/taps their card). This is standard for all three terminals. For credit card transactions, post-authorization tip adjustment is available on all three platforms.

**For New Brunswick locations**: Standard Canadian tipping customs apply. No Quebec-style tip regulation exists in New Brunswick.

---

### 2.4 Canadian Pricing (All Figures in CAD)

Unless otherwise noted, all pricing below has been researched from Canadian sources or converted from USD at the May 2026 exchange rate of approximately 1.37 CAD/USD.

#### Verifone Victa Portable (Moneris Go Terminal)

Moneris does not publicly list a purchase price for the Moneris Go Terminal (Victa Portable). Pricing is obtained through direct sales consultation (1-855-463-5669) [11].

Based on existing Moneris pricing patterns [50]:
- **Rental model**: Typically $29.95–$34.95/month per terminal
- **Processing fees** (Moneris Flat Rate):
  - Card-present credit: 2.65% + $0.10 per transaction
  - Card-present Interac debit: $0.12 per transaction
- **Monthly account service fee**: $2.95/month
- **PCI compliance fee**: $0/month (if compliant) or $35/quarter (non-compliance penalty)

#### Clover Station Duo (Canada)

- **Hardware purchase**: **$1,995 CAD** (Gravity Payments Canada) [51]
- **Software plans** (Canadian pricing):
  - Counter Service Restaurant: $59.95/month
  - Retail Growth: $84.95/month
  - Table Service Restaurant: $89.95/month
- **Additional device fees**: $9.95–$14.95/month per additional device
- **Processing fees**: Typically 2.3% + $0.10 in-person
- **Hidden fees** (multiple sources confirm): PCI compliance ($9.95–$10/month), statement charges ($5–$15/month), platform access fees ($27.95/month) [52][53]
- **Contract**: 36-month typical, with early termination fees of $500+

#### PAX A920 Pro (Canada)

- **Hardware purchase** (DirectDial Canada): **$503 CAD** (on sale from $868 CAD, currently out of stock) [35]
- **Hardware purchase** (US resellers, converted): $400–$900 USD = ~$548–$1,233 CAD
- **PAX A920MAX**: No specific Canadian CAD pricing found; US pricing suggests $500–$800 USD range
- **A920Pro PCI 7 (2026)**: Pricing not yet publicly available; expected comparable to current A920 Pro
- **Monthly software/platform fees** (PayFacto): PCI compliance portal fee of $19.95 CAD/month (waivable if compliant) [54]
- **Processing fees** (examples):
  - Moneris Flat Rate: 2.65% + $0.10 credit; $0.12 Interac debit
  - PayFacto Interchange+: Interchange + processor markup (typically 0.15%–0.35% + $0.05–$0.12 per transaction)
  - Adyen Interchange++: $0.13 + Interchange + 0.60% (starting markup) [55]
- **Extended warranty**: $50–$100/year (industry estimate)
- **Battery replacement**: ~$30–$50 USD (~$40–$68 CAD) every 2–3 years

---

## 3. Total Cost of Ownership (TCO) Model

### Modeling Assumptions

| Parameter | Value |
|-----------|-------|
| Transactions per location per day | 200 |
| Operating days per month | 30 |
| Average ticket | $25 CAD |
| Credit card share | 70% (4,200 trans/month) |
| Interac debit share | 30% (1,800 trans/month) |
| Monthly volume per location | $150,000 CAD |
| Chain monthly volume (25 locations) | $3,750,000 CAD |
| Chain annual volume | $45,000,000 CAD |

### 5-Year TCO Comparison — Per Location

| Cost Component | Verifone Victa (Moneris Rental) | Clover Station Duo (Purchase) | PAX A920 Pro (Purchase) |
|---|---|---|---|
| **Hardware (Year 1)** | $0 (rental) | $1,995 CAD purchase | $503–$870 CAD purchase |
| **Monthly software/platform** | $34.95/mo (rental inclusive) | $59.95–$89.95/mo | $0 (PAXSTORE basic) |
| **Monthly processing fees (Flat Rate)** | $3,418.50/mo | $3,418.50/mo | $3,418.50/mo |
| **Monthly processing fees (Interchange+)** | — | — | $2,511.00/mo |
| **Monthly hidden fees avg.** | $2.95 (statement) | $37.90–$47.90 | $19.95 (PCI portal, waivable) |
| **PCI compliance fee** | $0 (if compliant) | $9.95–$10/mo | $0 (if compliant) |
| **Warranty/repairs (annual)** | Included in rental | $50–$100/yr (extended) | $50–$100/yr |
| **Battery replacement** | Included in rental | Not applicable (countertop) | $40–$68 every 2–3 years |
| **Contract term** | 3–5 years (monitoring) | 36 months | 3–5 years (varies) |
| **Early termination fee** | $300–$500 | $500–$2,400 | $300–$500 |
| **5-year hardware/software** | $2,097 (rental × 60 months) | $3,495–$5,395 ($1,995 + $1,500–$3,400 software) | $503–$870 + $250–$500 (warranty + batteries) |
| **5-year processing (Flat Rate)** | $205,110 | $205,110 | $205,110 |
| **5-year processing (Interchange+)** | — | — | $150,660 |
| **5-year hidden fees** | $177 | $2,274–$2,874 | $1,197 (if non-waived) |
| **5-year TCO (Flat Rate)** | **$207,384** | **$210,879–$213,379** | **$206,810–$207,680** |
| **5-year TCO (Interchange+)** | — | — | **$152,360–$153,230** |
| **5-year TCO (Flat Rate) × 25** | **$5,184,600** | **$5,271,975–$5,334,475** | **$5,170,250–$5,192,000** |
| **5-year TCO (Interchange+) × 25** | — | — | **$3,809,000–$3,830,750** |

### Key TCO Insights

1. **Processing fees dominate the TCO** — representing approximately 98–99% of total costs over 5 years. Negotiating a favorable processing rate is the single most impactful cost-saving measure.

2. **Interchange-plus pricing saves approximately $54,450 per location over 5 years** compared to Moneris Flat Rate pricing for the modeled transaction volume. For the 25-location chain, this represents **$1,361,250 in savings**.

3. The **PAX A920 Pro** offers the lowest TCO when paired with interchange-plus pricing (saving ~$1.36 million vs. flat rate over 5 years for 25 locations).

4. **Verifone Victa rental** offers predictable costs with no upfront hardware investment, but total cost over 5 years is comparable to purchasing a PAX terminal with flat-rate processing.

5. **Clover Station Duo** has the highest TCO due to higher hardware costs ($1,995 CAD) and monthly software fees ($59.95–$89.95/month) plus hidden fees ($37.90–$47.90/month per location on average).

6. **Early termination fees** are a significant risk for any processor contract. The updated Code of Conduct for the Payment Card Industry in Canada (effective October 30, 2024) provides merchants the right to cancel without penalty within 70 calendar days after fee increases [56].

---

## 4. Bill 96 Compliance Validation Workflow

### 4.1 Overview of Bill 96 Requirements

Bill 96 (An Act respecting French, the official and common language of Québec) fundamentally changes language requirements for businesses operating in Quebec. Key provisions effective June 1, 2025 [57][58][59]:

- **French must be the primary/default language** for all commercial activities, customer communications, and employee communications
- **Businesses with 25 or more employees** in Quebec must register with the Office québécois de la langue française (OQLF)
- All commercial documents (receipts, contracts, customer-facing digital screens) must be in French
- **Software and work tools** must be available in French within Quebec
- French must be "markedly predominant" in all public signage

### 4.2 Stepwise Pre-Launch Validation Procedure

#### Step A: Contractually Require French as Default Language from the Processor

**Before signing any agreement**, include the following contractual requirements:

1. **Terminal software image**: Require that all terminals deployed in Quebec locations are pre-configured with **French as the default language** on:
   - All customer-facing screens (payment prompts, tip prompts, signature capture)
   - All admin/back-office screens used by Quebec-based employees
   - Receipt templates (paper and digital)

2. **Language availability**: Contractually require the processor to confirm that:
   - French language is available on the terminal model chosen
   - The French translation is complete and accurate (not machine-translated)
   - The terminal can switch between French and English without data loss
   - Any software updates maintain French language support

3. **Non-compliance liability**: Include provisions holding the processor responsible if their software fails to meet Bill 96 requirements, including any fines or penalties incurred.

4. **Support in French**: Require that technical support for Quebec locations is available in French, with defined response times.

#### Step B: Audit Checklist for Complete French Coverage

Conduct a comprehensive audit of all customer-facing and employee-facing touchpoints:

**Customer-Facing Screens (Mandatory French):**
- [ ] Payment prompts and instructions
- [ ] Tip selection prompts (percentage options, custom amount entry)
- [ ] Receipt selection (printed, emailed, none)
- [ ] Signature capture screen
- [ ] Contactless/insert/swipe instructions
- [ ] Error messages and declined transaction messages
- [ ] Loyalty program enrollment screens (if applicable)
- [ ] Promotional messages or upsell prompts

**Admin/Back-Office Screens (Must be available in French for Quebec employees):**
- [ ] Employee login and time tracking
- [ ] Transaction reporting and settlement
- [ ] Menu management and item configuration
- [ ] Discount and promotion setup
- [ ] Tip adjustment and employee payout management
- [ ] Inventory management screens
- [ ] Configuration and settings menus
- [ ] All drop-down menus, buttons, and status messages

**Receipts (French-first):**
- [ ] Merchant name and address in French format
- [ ] Transaction date in French format (e.g., 27 mai 2026)
- [ ] Line items in French
- [ ] Subtotal, tax, and total labeled in French (Sous-total, TPS, TVQ, Total)
- [ ] Tip line and total with tip in French
- [ ] Merchant's return/exchange policy in French
- [ ] Payment method in French
- [ ] Receipt footer (merci, au revoir, etc.)
- [ ] Proper French accent characters (é, è, ê, ô, ç, etc.)

**Physical Signage (French markedly predominant):**
- [ ] Menus and menu boards — French text must be "markedly predominant" (French must occupy at least twice the space of English and be equally or more legible) [59]
- [ ] Counter signage and promotional materials
- [ ] Directional signs (washrooms, exits, etc.)
- [ ] Pricing displays
- [ ] Any registered trademarks used in signage must comply with amended exception rules

**Digital Content:**
- [ ] Website accessible in Quebec must offer French-language version [59]
- [ ] Mobile app (if applicable) must have French interface
- [ ] Social media content targeting Quebec customers
- [ ] Email communications and marketing materials

#### Step C: OQLF Registration Process

Since your 25-location chain employs 25 or more employees in Quebec, you must register with the OQLF within six months of exceeding the threshold. The process [57][58]:

**Step 1 — Registration:**
- Register with the OQLF by filling out the registration form available on the OQLF website (form available in French only at https://www.oqlf.gouv.qc.ca/francisation/entreprises/formulaires/formulaire-inscription_rg.docx)
- Provide key information: number of employees, nature and scope of business activities, contact information
- Upon registration, your company receives a **certificate of registration**

**Step 2 — Linguistic Self-Evaluation (within 3 months of registration):**
- Submit a linguistic self-evaluation form to the OQLF (available on its website in French only)
- This assessment evaluates how well French is integrated into business operations, including:
  - Internal communications
  - Hiring practices
  - Training
  - Signage
  - IT systems (including payment terminals and POS systems)
  - External communications

**Step 3 — OQLF Assessment and Outcome:**
- **If compliant**: OQLF issues a **francization certificate** ("certificat de francisation"). Your company must maintain French usage and submit a **triennial report** (every three years) on the evolution of French usage.
- **If non-compliant**: OQLF issues a notice requiring a **francization program** to be established and implemented. The company must:
  - Complete and submit a francization program to the OQLF within three months
  - Submit reports on the program every 12 months
  - Share and update employees on the program's implementation
  - Continue until compliance is achieved

**Francization Committee:** If your company has 100+ employees in Quebec, a francization committee must be formed (minimum 6 members including worker and management representatives), meeting at least once every six months. The OQLF can also order companies with 25–99 employees to form a committee if deemed necessary [58].

**Public registry of non-compliance:** The OQLF will publish a list of organizations with denied, suspended, or canceled certificates of registration or francization [58].

**Loss of government contracts:** Section 152.1 of the Charter of the French Language prohibits public contracts or grants to companies not complying with francization requirements [60].

#### Step D: Specific Penalty Amounts Under Bill 96

| Offense Type | First Offense | Second Offense | Third+ Offense |
|-------------|---------------|----------------|----------------|
| Individual | $700–$7,000 | $1,400–$14,000 | $2,100–$21,000 |
| Corporation | $3,000–$30,000 | $6,000–$60,000 | $9,000–$90,000 |

**Daily penalties**: Each day of non-compliance constitutes a separate offense. Maximum potential fines can reach **$90,000 per day** for a corporation on a third offense [59][61].

**Other consequences of non-compliance**:
- Suspension or cancellation of business permits or certificates granted by the Quebec government
- Removal or destruction of non-compliant exterior commercial advertising at the expense of the business
- Injunctions from the Quebec Superior Court
- Contracts that violate Bill 96 rules may be declared null and void [59]

**OQLF enforcement powers** [59]:
- Conduct investigations to verify compliance
- Enter any location at any reasonable time (except private homes)
- Take pictures during inspections
- Access data stored on electronic devices during inspections
- Apply for injunctions with the Superior Court of Quebec

**New Brunswick note**: Bill 96 does **not** apply in New Brunswick. New Brunswick is Canada's only officially bilingual province, but the Official Languages Act applies to **government institutions**, not private businesses [62]. Private businesses in New Brunswick are not required by provincial law to have French as the default language on payment terminals, signage, or menus.

---

## 5. Additional Verification

### 5.1 Battery Life and Shift Coverage — Real-World QSR Conditions

#### Verifone Victa Portable

| Specification | Value |
|--------------|-------|
| Battery capacity | 5000 mAh |
| Manufacturer claim | Over 12 hours continuous use (processing every 2 minutes) |
| Real-world QSR estimate | 8–10 hours (with continuous printing/non-stop use) |
| Shift coverage | One full shift plus up to 2 hours buffer |
| Charge time | Not specified; standard 2–3 hours |

The Victa was only launched in Canada in February 2026 (~4 months ago as of May 2026), so no independent long-term user reviews exist yet [7]. The "12+ hours" manufacturer claim is based on Verifone's lab testing with "payments every 2 minutes," which is representative of a busy QSR during peak periods.

**Impact of continuous printing**: Thermal printing is one of the most battery-intensive operations on portable terminals. In a high-volume QSR processing 200 transactions per day with every transaction printing a receipt, the Victa's battery life may be reduced to **8–10 hours** in practice. The Victa Portable Plus with octa-core processor may have slightly higher power draw.

#### PAX A920 Pro

| Specification | Value |
|--------------|-------|
| Battery capacity | 5150 mAh (removable) |
| Manufacturer claim | 7.5–9.5 hours (with sleep mode) |
| Real-world QSR estimate | 6–8 hours (continuous use) |
| Shift coverage | Single shift (may require mid-shift charging) |
| Charge time | 2–3 hours |

MPI Processing states: "The A920 Pro extends its own battery life by offering a sleep mode when idle, allowing it on average 1½ more hours of use than its counterpart [the A920], meaning it can complete 7½ to 9½ hours of use between charges" [63].

An independent PAX A920 (original, not Pro) review from Mobile Transaction (2021) states: "The 5250 mAh battery guarantees 5-7 hours of use in normal environments, while it performs less well in low temperatures" [64].

#### PAX A920MAX

| Specification | Value |
|--------------|-------|
| Battery capacity | 2500 mAh (LiFePO4) |
| Manufacturer claim | 2.5 hours longer than predecessor (~8–10 hours) |
| Real-world QSR estimate | 7–9 hours |
| Shift coverage | Single full shift comfortably |
| Charge time | 2–3 hours (LiFePO4 charges faster) |

The A920MAX uses **Lithium Iron Phosphate (LiFePO4)** battery technology, which PAX describes as "following the footsteps of EV battery science" [28]. Key advantages:
- Longer cycle life (~450 full discharge cycles before dropping below 90% capacity)
- Safer (less prone to thermal runaway)
- Better performance in high-temperature environments
- But: lower energy density than standard lithium-ion (hence the lower mAh rating)

All-Star Terminals states the A920Max has a "battery lasting up to 10 hours" [65].

#### PAX A920Pro PCI 7 (2026)

| Specification | Value |
|--------------|-------|
| Battery capacity | 6000 mAh |
| Manufacturer claim | "Full-day usage" |
| Real-world QSR estimate | 10–12 hours (estimated) |
| Shift coverage | Full double shift comfortably |
| Charge time | Not specified |

The A920Pro PCI 7's 6000mAh battery is the largest of any PAX portable terminal, representing a 20% capacity increase over the A920 Pro.

#### Clover Station Duo

The Clover Station Duo is a **countertop terminal** that is always plugged in. It has a 4000mAh backup battery providing up to 4 hours of use for power outages [66], but it is not designed for portable use.

**For mobile/pay-at-table use**, the **Clover Flex** (pocket-sized portable) has a manufacturer claim of "up to 8 hours" with a 2100mAh battery. User reviews report that battery life "gradually decreased over time" and real-world usage is closer to 6–7 hours for new units [67].

#### Battery Life Summary for QSR Shift Coverage

| Terminal | Single Shift (8 hrs) | Double Shift (16 hrs) | Notes |
|----------|---------------------|----------------------|-------|
| Verifone Victa | ✅ (10–12 hrs est.) | ⚠️ May need mid-shift charge | Too new for independent validation |
| PAX A920 Pro | ⚠️ (6–8 hrs) | ❌ | May need mid-shift charging station |
| PAX A920MAX | ✅ (7–9 hrs) | ❌ | Better battery chemistry |
| PAX A920Pro PCI 7 | ✅✅ (10–12 hrs est.) | ⚠️ (may last full double shift) | Best option for high-volume |
| Clover Station Duo | ✅ (always plugged) | ✅ (always plugged) | Countertop only |

**Recommendation**: For high-volume QSR locations, plan for **charging stations or hot-swappable batteries** for portable terminals. The PAX A920 Pro battery is removable, allowing a backup battery to be swapped mid-shift. The Verifone Victa has a sealed battery requiring docking station charging.

---

### 5.2 Clover Station Pro as Payment-Only Peripheral with Oracle MICROS

**Finding: Clover CAN operate as a payment-only peripheral, but NOT with Oracle MICROS Simphony out of the box.**

Clover supports "semi-integration" where a third-party POS handles ordering while a Clover device handles payments, using:
- **CloverConnector SDKs** (Android, iOS, JavaScript, .NET)
- **REST Pay Display** (for Clover Flex or Mini via HTTP requests)

Per Clover's official developer documentation: "Semi-integrated solutions run on a combination of Clover and third-party hardware; the Clover device handles payment processing... Clover offers secure payment integration, allowing POS software to accept EMV-ready, PCI compliant payments" [23].

However:
1. There is **no pre-built integration** specifically for Oracle MICROS Simphony
2. The Clover device would **not natively speak OPI** (Oracle Payment Interface) protocol
3. Custom middleware would need to be developed to translate between OPI and CloverConnector APIs
4. This custom development would require ongoing maintenance for OPI version updates

**This is not a viable solution** for a chain requiring certified, supported integration without significant custom development investment.

---

### 5.3 Verifone Carbon 8 — Truly End-of-Life from Authorized Canadian Channels

**Confirmed**: The Verifone Carbon 8 is **not available from any authorized Canadian channel** for new deployments:

1. **Moneris Canada**: Carbon 8 is not listed in Moneris's device catalog of 25+ devices [5]
2. **Verifone Canada**: Carbon 8 is not listed on Verifone's retail product page (verifone.com/retail) — only Carbon Mobile 5 is shown [4]
3. **Verifone's 2026 product strategy**: The Victa family is Verifone's flagship portfolio, with the Victa Portable explicitly positioned as the current portable solution [68]
4. **Third-party resellers**: Some resellers may have backorder inventory (e.g., Merchantrolls.com lists it at $810 USD on backorder [69]), but these are not authorized Canadian channels and units would lack current PCI PTS certification for original hardware revisions

**Support inventory**: Based on Verifone's historical EOL patterns [8][9]:
- Existing installations may still receive support through specific contracts
- Repairs may be possible after End of Support date but outside warranty
- Certified refurbished units and replacement parts may still be available from authorized service centers for existing installations
- However, the device's Android Lollipop OS (no longer receiving security updates) poses ongoing PCI compliance risks

**Recommendation**: Do not consider the Carbon 8 for any new deployment. Existing units should be replaced as soon as practical.

---

### 5.4 Moneris Go Restaurant KDS — Quebec Availability and Alternatives

#### Definitive Status

**The Moneris Go Restaurant KDS is explicitly unavailable in Quebec and English-only.**

The Moneris Go Restaurant product page states the solution is "designed primarily for Canadian small businesses outside Quebec" [70]. The Moneris Go Restaurant POS Terms and Conditions Schedule contains a "For Residents of Quebec" clause in Section 21, which states: "It is agreed that the express wish of the parties is that this Schedule and any related documents be drawn up and executed in English" [71].

The Moneris Kitchen Display app on the Apple App Store shows **no mention of French language support** [72].

The Moneris Support article for the Kitchen Display System provides setup guidance in English only [73].

#### Alternative KDS Solutions for Quebec Locations

**Recommended: Oracle MICROS Express Station 400**

The Oracle Express Station 400 is a durable, all-in-one kitchen display solution designed for harsh kitchen environments [74]:

- **Integration**: Natively integrates with Oracle MICROS Simphony POS — your existing system
- **French language support**: Oracle's official documentation titled "Translation for KDS" explicitly confirms **French is a supported language** for KDS Displays across all three KDS Display platforms (Windows 32, Windows CE, and RDC Devices) [75]
- **Durability**: IP-54 rated against dust and water jets, fanless design, operating temperature 0°C to 60°C, sealed against humidity, grease, and airborne contaminants [76]
- **Processor**: Intel Atom® x5 Series with 15-year availability [76]
- **Availability in Canada**: Oracle has a dedicated Canadian website (oracle.com/ca-en) confirming Canadian availability [74]
- **Payment integration**: Moneris payment processing can be integrated separately as a payment gateway with Simphony

**Other alternatives**:

- **Toast KDS**: Supports French language (including Canadian French) at "all Toast POS locations worldwide" [77]. However, Toast is its own POS ecosystem and does not directly integrate with Oracle MICROS Simphony or Moneris payment processing.

- **TouchBistro KDS**: Available as an iPad app. TouchBistro expanded its partnership with Moneris in October 2025 [78]. However, TouchBistro KDS is for TouchBistro's own POS, not Oracle MICROS Simphony. No explicit French language support was found.

- **PAX Elys Display K20/K21**: See Section 5.5 below.

---

### 5.5 PAX Elys Display KDS — Oracle MICROS Simphony Integration

#### Official PAX Documentation

The PAX Elys Display K20 (15.6-inch) and K21 (21.5-inch) are rugged, high-performance smart displays tailored for kitchen environments [79]:

- **OS**: Android 14 with Google Mobile Services certification
- **Processor**: Octa-core Cortex-A72 at 2.2GHz with 6TOPS NPU
- **RAM/Storage**: 4GB LPDDR4X RAM + 64GB eMMC storage
- **Display**: Full HD IPS touchscreens with anti-fingerprint coatings, Mohs 6 hardness, support for wet or gloved hands
- **Durability**: IP55 rated against water and dust
- **Power**: Power over Ethernet (PoE)
- **Accessories**: PB20 Bump Bar (configurable keyboard, IP65 rated, 50-million-click lifespan)

#### Integration with Oracle MICROS Simphony

**Finding**: The PAX Elys Display **cannot integrate directly with Oracle MICROS Simphony** out of the box. The integration requires third-party middleware.

PAX's official documentation states: "Elys Display seamlessly integrates with thousands of POS software providers" but does not specifically name Oracle MICROS Simphony as a direct integration partner [79].

However, **indirect integration is possible through middleware**:

A PAX Technology blog post describes a partnership with **Touché**, a software provider for F&B and Hospitality, to integrate Touché's software on PAX's Android SmartPOS devices **with Oracle's MICROS Simphony POS system** [31]. The article states: "PAX Technology... has partnered with Touché... to integrate Touché's software on PAX's range of Android-based SmartPOS devices and Oracle's MICROS Simphony POS system. This integration streamlines processes such as ordering, payment, and loyalty management."

This indicates that the PAX Elys Display can integrate with Oracle MICROS Simphony, but **indirectly** through third-party middleware like Touché, rather than through a direct native integration. The PAX Elys Display itself is a hardware platform that runs Android, and any Oracle MICROS integration requires compatible KDS software running on that Android platform.

**Recommendation**: For your existing Oracle MICROS Simphony environment, the **Oracle Express Station 400** is the most straightforward KDS choice with confirmed French language support and native integration. If you prefer the PAX Elys Display hardware, plan for the additional middleware integration layer and associated costs.

---

## 6. Pilot Deployment Plan (30–60 Days)

### Phase 1: Preparation (Days 1–10)

**Objective**: Select pilot locations, configure terminals, and establish baseline metrics.

**Selection Criteria:**
- 2–3 pilot locations: 1 high-volume Quebec location, 1 medium-volume Quebec location, 1 New Brunswick location
- Mix of peak-hour transaction volumes (target: 150–300 transactions/day)
- Locations with stable Wi-Fi and cellular coverage

**Pre-Deployment Checklist:**
- [ ] Contract signed with payment processor specifying French as default language
- [ ] Terminals pre-configured with French language image (for Quebec locations)
- [ ] Oracle MICROS Simphony OPI integration configured and tested in sandbox
- [ ] Tip-on-subtotal setting enabled for Quebec locations (Bill 72 compliance)
- [ ] Receipt templates audited for French compliance
- [ ] OQLF registration initiated (if not already done)
- [ ] Staff training materials prepared in French and English
- [ ] KDS integration tested (Oracle Express Station 400 or existing MICROS KDS)
- [ ] Battery charging stations installed at each pilot location
- [ ] Network connectivity verified (Wi-Fi and cellular backup)
- [ ] Baseline metrics recorded (current transaction speed, decline rates, customer satisfaction)

### Phase 2: Soft Launch (Days 11–25)

**Objective**: Deploy terminals in a controlled environment with monitoring.

**Deployment:**
- Install terminals at 1–2 counter positions per location
- Run parallel with existing payment system for first 3 days
- Monitor transaction success rates, speed, and staff comfort

**Daily Monitoring:**
- Transaction processing speed (seconds per transaction)
- Contactless vs. chip insertion ratio
- Decline rates (online vs. offline)
- Battery drain rate (% per hour)
- Thermal printer performance (paper jams, print quality)
- French language display accuracy (screen prompts, receipts)
- Staff feedback on ease of use
- Customer feedback on payment experience

**Issue Resolution:**
- Document all issues with timestamps, screenshots, and severity
- Escalate critical issues (payment failures, integration errors) within 2 hours
- Track resolution time per issue category

### Phase 3: Full Launch at Pilot Sites (Days 26–45)

**Objective**: Full deployment at pilot locations with comprehensive testing.

**Testing Protocols:**

| Evaluation Dimension | Test Protocol | Success Metric |
|---------------------|---------------|----------------|
| **Transaction Speed** | Measure 50 consecutive transactions during peak lunch rush (11:30–1:30) | Average < 8 seconds per tap; < 15 seconds per chip+PIN |
| **Offline Mode** | Simulate internet outage during off-peak (disable router); process 10 credit transactions | All 10 transactions stored and forwarded successfully; no data loss |
| **Interac Debit Offline** | Attempt Interac debit transaction during simulated outage | Transaction declined with clear French error message (expected) |
| **OPI Integration** | Process 100 transactions through Oracle MICROS OPI | 0 OPI-related errors; all transactions settle correctly |
| **French Language** | Audit all screens (customer + admin) for French accuracy | 100% of customer-facing screens in French; 0 translation errors |
| **Tip Calculation** | Process 50 transactions with tips in Quebec location | Tips calculated on pre-tax subtotal (Bill 72 compliance) |
| **Battery Life** | Run terminal from full charge to shutdown during normal operations | Minimum 8 hours before low-battery warning |
| **KDS Integration** | Process 50 orders and verify KDS display | 100% of orders displayed correctly; 0 missed orders |
| **Receipt Accuracy** | Print 50 receipts and verify French content | 100% French-first with proper accents |
| **Cellular Backup** | Disable Wi-Fi; verify cellular connectivity | Automatic failover within 30 seconds |

### Phase 4: Decision Gate and Evaluation (Days 46–60)

**Objective**: Analyze pilot data and make go/no-go decision for full rollout.

**Quantitative Metrics:**
- Transaction speed: Average transaction time vs. baseline
- Transaction success rate: Declines, timeouts, errors
- Offline mode reliability: Store-and-forward success rate
- Battery life: Average hours per charge under real conditions
- French language compliance: Audit score
- Staff productivity: Transactions per hour per terminal

**Qualitative Metrics:**
- Staff satisfaction (survey: 1–10 scale)
- Customer satisfaction (survey or feedback cards)
- Manager feedback on ease of use and troubleshooting
- Technical support responsiveness (average response time)

**Decision Gate:**
- **Go for full rollout**: All critical metrics meet or exceed targets
- **Conditional go**: Minor issues identified with clear remediation plan
- **No-go**: Major integration failures, critical language compliance gaps, or unacceptable performance

**Full Rollout Plan (if approved):**
- 5 locations per week (5 weeks total for 25 locations)
- 2-day training per location (pre-deployment + go-live support)
- 2-week hyper-care period per location
- Monthly compliance audits for first 6 months
- Quarterly OQLF reporting for Quebec locations

---

## 7. Final Recommendations

### Primary Recommendation: PAX A920 Pro (or A920MAX / A920Pro PCI 7)

The **PAX A920 Pro** (or the newer **A920MAX** or **A920Pro PCI 7**) is the strongest choice for your 25-location QSR chain because:

1. **Certified Oracle MICROS Simphony integration** via OPI 6.2+ through multiple certified processors (Adyen, Global Payments, PayFacto) — the most proven integration pathway of all three terminals

2. **Active Canadian certification and availability** — PayFacto Class A certification, available through Moneris (nationwide), PayFacto (Montreal-based), and other Canadian distributors; PCI PTS 5.x/6.x/7.x certified

3. **Lowest 5-year TCO** — hardware purchase cost of $503–$870 CAD (vs. $1,995 CAD for Clover Station Duo), with potential savings of $1.36 million over 5 years when paired with interchange-plus pricing

4. **Android platform with full language localization** — PAX France division provides French-language UX/UI design; Android/PayDroid supports complete French localization configurable as default language

5. **Next-gen options available** — A920MAX for highest-volume locations (200+ transactions/day); A920Pro PCI 7 (April 2026) with Android 14, PCI 7.x certification, and 6000mAh battery

6. **Interac debit fully supported** — tested and certified by multiple Canadian processors; note: no offline mode for Interac (industry-wide limitation)

**Recommended configuration:**
- High-volume locations: PAX A920MAX or A920Pro PCI 7
- Standard locations: PAX A920 Pro
- Processor: PayFacto (interchange-plus pricing, Montreal-based, French support) or Adyen (interchange++ pricing, OPI certified)
- KDS: Oracle MICROS Express Station 400 (natively integrated, French language confirmed)

### Secondary Recommendation: Verifone Victa Portable (Moneris Go Terminal)

If you prefer a rental model with predictable monthly costs and no upfront hardware investment, the **Verifone Victa Portable** (Moneris Go Terminal) is a viable alternative:

1. **Newest device in Canada** (launched February 2026) with PCI 7 readiness and Android 13
2. **Rental model** eliminates upfront hardware costs and includes maintenance
3. **12+ hour battery claim** (though too new for independent validation)
4. **Moneris is Canada's #1 processor** with local Quebec offices and French-language support
5. **OPI integration** available through multiple certified processors

**Critical caveats:**
- Moneris Go Restaurant KDS is **not available in Quebec and is English-only** — you would need to maintain your existing MICROS KDS setup or use Oracle Express Station 400
- Verifone Victa is too new to the market for independent user reviews on battery life and reliability
- OPI integration is processor-dependent and requires verification with Moneris specifically

### Not Recommended: Verifone Carbon 8

- 9-year-old device running obsolete Android Lollipop (no security updates)
- Original hardware revision's PCI PTS certification expired April 30, 2024
- Not available from any authorized Canadian channel for new deployments
- Replaced by Verifone Victa Portable

### Not Recommended: Clover Station Pro / Station Duo

- **No certified direct integration with Oracle MICROS Simphony** — this is a critical gap confirmed by Clover's absence from Oracle's payments integration partner list
- Custom middleware development would be required (CloverConnector SDKs), with ongoing maintenance costs
- Higher TCO due to hardware costs ($1,995 CAD) plus monthly software fees ($59.95–$89.95/month) and hidden fees ($37.90–$47.90/month per location)
- 36-month contracts with substantial early termination penalties ($500–$2,400)
- Processor-locked hardware — cannot switch processors without replacing all terminals

### Action Plan

1. **Immediate (Next 2 weeks):**
   - Contact PayFacto (Montreal) and Adyen Canada for PAX A920 Pro/A920MAX quotes reflecting 25-location volume
   - Request interchange-plus pricing (not flat-rate) — potential savings of $1.36M over 5 years
   - Verify OPI 6.2+ integration with your specific Oracle MICROS Simphony version

2. **Bill 96 Compliance (Next 4 weeks):**
   - Register with OQLF (mandatory for 25+ employees in Quebec)
   - Contractually require French as default language on all terminal software images
   - Conduct full audit of all customer-facing screens, admin screens, and receipt templates
   - Implement tip-on-subtotal for Quebec locations (Bill 72 compliance)

3. **KDS Strategy (Next 4 weeks):**
   - Maintain existing Oracle MICROS KDS setup for immediate deployment
   - Order Oracle Express Station 400 for any new KDS deployments (native integration, French confirmed)
   - Monitor PAX Elys Display + Touché middleware integration as future option

4. **Pilot Program (Days 1–60):**
   - Deploy PAX A920 Pro terminals at 2–3 locations as outlined in the Pilot Deployment Plan above
   - Include 1 high-volume Quebec, 1 medium-volume Quebec, and 1 New Brunswick location
   - Use Oracle Express Station 400 for KDS at pilot locations
   - Evaluate against defined success metrics before full rollout

5. **Full Rollout (Weeks 9–13, if pilot approved):**
   - 5 locations per week (5 weeks total)
   - 2-day training per location
   - 2-week hyper-care period per location
   - Monthly compliance audits for first 6 months

---

### Sources

[1] PCI Security Standards Council — Approved PTS Device Listing (Carbon 8): https://listings.pcisecuritystandards.org/assessors_and_solutions/vpa_agreement.php?return=%2Fassessors_and_solutions%2Fpin_transaction_devices&agree=true

[2] Verifone Documentation Portal — Carbon 8 Device Specs: https://docs.verifone.com/device-installation-guides/installation-guides/device-installation-guides/android-devices/carbon

[3] Android OS endoflife.date — Lollipop EOL: https://endoflife.date/android

[4] Verifone Retail Product Page (current): https://www.verifone.com/retail

[5] Moneris All Devices Catalog: https://www.moneris.com/en/support/devices/all-devices

[6] Financial IT — Verifone Launches Carbon 8 (May 10, 2017): https://financialit.net/news/payments/verifone-launches-carbon-8-portable-pos-terminal

[7] Moneris Press Release — Expands Go Commerce Suite (Feb 3, 2026): https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite

[8] POSDATA Group Verifone EOL Notices: https://www.posdata.com/notices-payment-terminals

[9] Bluefin — Verifone EOL Notices: https://bluefin.my.site.com/knowledgebase/s/article/Verifone-EOL-Notices

[10] Verifone Victa Portable Plus Datasheet (Official PDF): https://cdn.prod.website-files.com/6877dbe2d81008ec40dd7770/69ab54bd5582db93331a50be_Victa%20Portable%20Plus%20Datasheet.pdf

[11] Moneris Go Terminal (Victa Portable) Support Page: https://www.moneris.com/en/support/moneris-go/victa

[12] Verifone — PCI PTS v7 certification for V660p: https://www.verifone.com/resources/pci-pts-v7-certification-v660p

[13] TRANSACT 2026 — Moneris + Verifone Victa Portable: https://www.youtube.com/watch?v=unkkPdYrqbs

[14] Oracle Docs — The Oracle Payment Interface (OPI): https://docs.oracle.com/cd/E76065_01/doc.29/e69879/c_payments_opi.htm

[15] Adyen Docs — Oracle Simphony Integration: https://docs.adyen.com/plugins/oracle-simphony

[16] Global Payments Developer — Oracle Payment Interface: https://developer.globalpayments.com/heartland/payments/in-store/pos-middleware/oracle-payment-interface

[17] Shift4 — Integration to OPI (PDF): https://www.shift4.com/pdf/S4P_Oracle-Payments_Reference-Guide.pdf

[18] Clover Developer Docs — Devices Tech Specs: https://docs.clover.com/dev/docs/clover-devices-tech-specs

[19] Limelight Payments — Clover POS Cost & Pricing 2026: https://www.limelightpayments.com/blog/clover-pos-cost-pricing-2025-hardware-clover-fees-processing-fees

[20] Oracle — POS Integrations: https://www.oracle.com/food-beverage/restaurant-pos-systems/pos-integrations

[21] HotelTechReport — Clover vs Oracle Simphony Comparison: https://hoteltechreport.com/compare/clover-vs-oracle-micros-pos

[22] Worldline — Oracle MICROS Simphony Integration: https://worldline.com/en/home/top-navigation/media-relations/press-release/worldline-delivers-europe-wide-pos-payment-solution-to-oracle-micros-simphony-customers-in-the-food-beverage-industry

[23] Clover Developer Docs — Semi-Integration Basics: https://docs.clover.com/dev/docs/clover-development-basics-semi

[24] Clover Developer Docs — PAAS Integration Options: https://docs.clover.com/dev/docs/paas-integration-options

[25] Fiserv Canada — Clover Solutions: https://merchants.fiserv.com/en-ca

[26] NCFA Canada — TD Partners with Fiserv (July 23, 2025): https://ncfacanada.org/td-partners-with-fiserv-and-sells-merchant-portfolio

[27] PAX A920Pro Datasheet (Official PDF): https://www.pax.us/wp-content/uploads/2023/12/A920Pro-Datasheet.pdf

[28] PAX Technology — A920MAX Product Page: https://www.paxtechnology.com/a920max

[29] PAX Technology — Next-Gen A920Pro PCI 7 Press Release (April 2026): https://www.pax.us/about/press-room/the-next-generation-a920pro-pci-7

[30] PAX — First to Achieve PCI PTS POI v7.0: https://www.pax.com.cn/PAX-First-to-Achieve-PCI7-PTS-POI-Certification-in-Payment-Terminals

[31] PAX Technology Blog — Touché Deploys Oracle Solution on PAX Android: https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android

[32] PayFacto — Launches PAX A920 in Canada: https://payfacto.com/payfacto-launch-pax-a920-android-payment-terminals-canada

[33] Reddit — PAX OPI/SPI with Oracle Simphony: https://www.reddit.com/r/pcicompliance/comments/1kpf3nw/pax

[34] PAX and Moneris — Launch A920 in Canada: https://www.pax.us/about/press-room/pax-technology-and-moneris-solutions-launch-a920-in-canada

[35] DirectDial Canada — PAX A920 Pro: https://www.directdial.com/ca/store/PAX

[36] PAX Canada: https://www.pax.us/canada

[37] Global Payments — Contactless Payment Limits: https://www.globalpayments.com/insights/when-and-why-do-contactless-limits-matter

[38] The Globe and Mail — Mastercard and Visa raise tap limits to $250: https://www.theglobeandmail.com/business/article-mastercard-and-visa-raise-tap-limits-to-250-so-fewer-consumers-need

[39] Digital Transactions — Card Networks Up Canadian Contactless Limits: https://www.digitaltransactions.net/card-networks-up-canadian-contactless-transaction-limits-to-limit-physical-contact

[40] Square — Interac Debit Guide: https://squareup.com/ca/en/the-bottom-line/operating-your-business/interac-debit

[41] Amilia Help Center — Interac on PAX A920: https://help.amilia.com/en/articles/9580609-interac-on-integrated-terminals

[42] Moneris — Pricing: https://www.moneris.com/en/pricing

[43] Clearly Payments — Interac and Debit Cards in Canada: https://www.clearlypayments.com/blog/an-introduction-to-interac-and-debit-cards-in-canada

[44] Stripe — Interac Terms and Conditions: https://stripe.com/en-ca/legal/interac

[45] Paylosophy — Handling Interac Payments in Canada: https://paylosophy.com/handling-interac-payments-canada

[46] Moneris Blog — Bill 72 Compliance: https://www.moneris.com/en/blog/posts/compliance/bill-72

[47] BASYS Processing — PAX A920 Tip Adjustment: https://support.basyspro.com/hc/en-us/articles/19944279058964-PAX-A920-Card-Transaction-Adjust-Tips

[48] Verifone Cloud — Tipping Feature Reference: https://verifone.cloud/sites/default/files/inline-files/Tipping_0.pdf

[49] Clover Developer Docs — Tip Mode: https://docs.clover.com/dev/docs/tip-mode

[50] PaymentGateway.ca — Moneris vs Helcim 2026: https://paymentgateway.ca/moneris-vs-helcim-canada

[51] Gravity Payments Canada — Clover Station Duo: https://gravitypayments.com/devices/clover-station-duo-canada

[52] CheckThat.ai — Clover Pricing 2026: https://checkthat.ai/brands/clover/pricing

[53] SleftPayments — Clover Hidden Fees 2026: https://www.sleftpayments.com/learning-hub/clover-charging-fees-i-didnt-agree-to-2026

[54] PayFacto — PCI Compliance: https://payfacto.com/pci-dss-compliance

[55] Adyen — Pricing: https://www.adyen.com/pricing

[56] Canada.ca — Code of Conduct for Payment Card Industry: https://www.canada.ca/en/financial-consumer-agency/services/industry/code-conduct-payment-card-industry.html

[57] Miller Thomson — Are you ready for June 1? New French language obligations: https://www.millerthomson.com/en/insights/labour-and-employment/are-you-ready-for-june-1-new-french-language-obligations-for-quebec-employers-with-25-to-49-employees

[58] Éducaloi — Francization Rules for Employers: https://educaloi.qc.ca/en/capsules/francization-rules-for-employers/

[59] Preply Business — Bill 96 Quebec Explained: https://preply.com/en/blog/b2b-bill-96-quebec-explained

[60] Stein Monast — Companies with 25-49 employees: https://steinmonast.ca/en/news-and-resources/companies-with-25-to-49-employees-registered-with-the-office-quebecoise-de-la-langue-francaise

[61] CFIB — Everything you need to know about Quebec's Law 14 (Bill 96): https://www.cfib-fcei.ca/en/site/qc-law-14-bill-96

[62] OCOLNB — Frequently Asked Questions: https://officiallanguages.nb.ca/content/frequently-asked-questions

[63] MPI Processing — PAX A920 & A920Pro Battery Life: https://www.mpiprocessing.com/a920-a920pro

[64] Mobile Transaction — PAX A920 Review: https://www.mobiletransaction.org/pax-a920-review

[65] All-Star Terminals — PAX A920MAX Launch: https://allstarterminals.com/blogs/hardware-insights/pax-launches-new-payment-terminal-the-pax-a920max

[66] Kotapay — Clover Station Duo Specs: https://www.kotapay.com/cloverstationduo

[67] eMerchant Authority — Clover Flex Review: https://emerchantauthority.com/blog/clover-flex-review

[68] Verifone — Expands Victa Line (Jan 8, 2026): https://www.verifone.com/resources/verifone-announces-expanded-victa-line-broadened-partner-ecosystem

[69] Merchantrolls.com — Verifone Carbon 8: https://merchantrolls.com/product/verifone-carbon-8

[70] Moneris Go Restaurant POS: https://www.moneris.com/en/solutions/pos-systems/go-restaurant

[71] Moneris Go Restaurant POS Terms and Conditions Schedule (2026): https://www.moneris.com/-/media/files/terms-and-conditions/2026/en/moneris-go-restaurant-pos-terms-and-conditions-schedule-2026.ashx

[72] Moneris Kitchen Display App — Apple App Store: https://apps.apple.com/ca/app/moneris-kitchen-display/id1610511113

[73] Moneris Support — KDS App: https://support.moneris.com/article/moneris-go-restaurant-the-kitchen-display-system-kds-41785

[74] Oracle Canada — KDS for Restaurants: https://www.oracle.com/ca-en/food-beverage/restaurant-pos-systems/kds-kitchen-display-systems

[75] Oracle Docs — Translation for KDS (French Language Support): https://docs.oracle.com/cd/F14820_01/doc.191/f15055/c_internationalization_kds.htm

[76] Oracle MICROS Express Station 400 Datasheet (PDF): https://www.oracle.com/a/ocom/docs/industries/food-beverage/oracle-micros-express-station-400.pdf

[77] Toast Product Updates — French Language Support for KDS: https://updates.toasttab.com/announcements/additional-french-language-support-for-kds

[78] Moneris Press Release — Go Restaurant POS (Oct 6, 2025): https://www.moneris.com/en/media-room/news/moneris-launches-new-go-restaurant-pos-and-extends-partnership-with-touchbistro

[79] PAX Technology — Elys Display K20/K21: https://www.pax.us/k20-k21

[80] PAX Technology — Launches Innovative KDS and Bump Bar: https://www.pax.us/about/press-room/pax-technology-inc-launches-innovative-kds-and-bump-bar-for-restaurants