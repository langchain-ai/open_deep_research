# Comprehensive Comparison: Verifone Carbon 8, Clover Station Pro, and PAX A920 Pro for Your QSR Chain

## Executive Summary

This report provides a detailed comparison of three payment terminal solutions for your 25-location quick-service restaurant chain expanding across Quebec and New Brunswick. After extensive research, the **PAX A920 Pro** (or its newer variant, the A920MAX) emerges as the strongest recommendation for your specific needs, given its certified Oracle MICROS Simphony integration via OPI 6.2+, active Canadian market presence, Android-based platform with full language localization, and competitive pricing. However, critical considerations around Quebec's Bill 96 compliance and KDS compatibility must be addressed separately regardless of terminal choice.

A significant finding: the **Verifone Carbon 8** is effectively end-of-life (Android Lollipop, PCI approval expiring April 2027) and should be removed from consideration. Verifone's current Canadian offering is the **Verifone Victa Portable** (launched February 2026 through Moneris), which is the appropriate Verifone equivalent to evaluate.

---

## 1. Transaction Processing Speeds During Peak Hours

### Verifone Carbon 8 / Verifone Victa (Current Canadian Equivalent)

The Carbon 8 features an Intel Atom x5-Z8300 processor (1.84 GHz) and runs Android Lollipop [1]. No published transactions-per-second data exists for peak-hour QSR conditions. The device was described at launch in 2017 as "designed for small to midsized businesses in high-touch industries like hospitality" [2]. However, the Carbon 8 is now obsolete.

The **Verifone Victa Portable** (the current Canadian model, launched February 2026 through Moneris) is the proper comparable device. It features:
- **Processor**: Qualcomm QCM2290 quad-core (Victa Portable) or SM6225 octa-core (Victa Portable Plus) [3]
- **RAM**: 4GB (Portable) / 8GB (Portable Plus) [3]
- **Storage**: 32GB [3]
- **Operating System**: Verifone Secure OS based on Android 13, upgradeable to Android 16 [3]
- **Battery**: Over 12 hours of continuous usage processing payments every 2 minutes [4]

Moneris describes the Victa as having "doubled system memory" compared to previous generations and being "built to handle high transaction volumes and diverse environments" [4].

### Clover Station Pro

The Clover Station Pro runs on a Qualcomm Snapdragon octa-core processor. Specifications vary by generation:
- **Clover Station 2018**: Qualcomm Snapdragon 810 octa-core CPU with 4GB RAM and 16GB ROM [5]
- **Clover Station Solo**: Qualcomm Snapdragon 660 chip running Android 10 [6]
- **Clover Station Duo**: Qualcomm Snapdragon 8-core processor with 4GB RAM [7]

The **dual-screen design** (14-inch merchant-facing + customer-facing display) is specifically engineered to speed checkout during peak traffic by allowing customers to select tips, sign, and choose receipt preferences in parallel with the merchant's workflow [8]. One processor states transactions are "now twice as fast" on current-gen models compared to prior versions [7].

**Important note**: Clover announced that "Gen 1 devices, including Clover Station (C010), are approaching End-of-App-Update (EOAU) on March 30, 2026" [9]. Current-generation devices (Station Solo, Station Duo 2) are not impacted.

### PAX A920 Pro

The A920 Pro is powered by a **quad-core 1.4 GHz ARM Cortex A53 processor** [10]. It is available with **1GB RAM / 8GB storage** or **2GB RAM / 16GB storage** configurations [10]. The device runs on Android/PayDroid operating system.

For QSR environments, the A920 Pro is described as "better suited for high-volume merchants needing faster processing and scalability" and "ideal for high-volume merchants needing maximum performance, display, and battery life" [10][11]. Battery life is rated at **7.5 to 9.5 hours** of continuous use with a 5150mAh battery and sleep mode optimization [10].

The **A920MAX** (newer flagship model, 2026) features a Cortex A53 quad-core 1.3GHz processor, 16GB flash + 2GB RAM, and is described as "40% faster processing speed" compared to A920 Pro, with "transaction times dropped 30%" according to a retail chain case study [12]. The A920MAX targets merchants processing **200+ daily transactions** [12].

**Verdict**: For a 25-location QSR chain handling lunch/dinner rushes, the **Verifone Victa Portable Plus** (octa-core, 8GB RAM) and **PAX A920MAX** offer comparable high-volume performance. The Clover Station Pro's dual-screen design provides a unique throughput advantage for customer-facing payment workflows.

---

## 2. Offline Payment Capabilities (Store-and-Forward)

### Verifone Terminals (Carbon 8 / Victa)

Verifone terminals support **Store and Forward (SAF)** functionality, where "the pinpad approves and temporarily stores this transaction offline" and "once the connection is restored, the stored transactions are forwarded to the back-end for processing and online approval" [13]. The terminal automatically switches to offline mode without user input when connectivity is lost [14].

Key limitations and risks:
- **Storage limit**: Maximum of 1,000 transactions stored in SAF mode [15]
- **Merchant liability**: "Merchants are fully liable for the risk of failed captures related to payments processed offline" [14]
- **Authorization risk**: "A 5% average rate of declined transactions" when processed offline [14]
- **Card restrictions**: Chip card purchases with correct PIN entry and non-expired cards only; magnetic stripe, AMEX, cashouts, and refunds are typically excluded [15]

For the Victa specifically, SAF availability depends on the processor's implementation (Moneris, Global Payments). Moneris documentation for offline payments on PAX terminals notes that Interac debit transactions are NOT supported offline [16].

### Clover Station Pro

Clover's Help Center states: "When enabled, devices can take offline payments for up to 7 days" [17]. The offline payments option is enabled by default for Clover Station and Station 2 models [17].

Default limits:
- **Maximum per-transaction amount**: $500 (configurable) [18]
- **Maximum total offline payment amount**: $5,000 (default, configurable) [18]

Key conditions:
- The device must be connected to a local network (LAN/Wi-Fi), even if the internet is down [18]
- "Risk of charge declines due to insufficient funds or card cancellations, which means merchants might not receive payment despite an accepted offline transaction" [18]
- Merchants can monitor offline transaction statuses via the Transactions app, which displays approved, pending, or declined statuses with corresponding icons [18]

The Station Pro also includes **native 4G/LTE connectivity** as a backup option for when Wi-Fi or Ethernet is unavailable, providing additional reliability beyond the store-and-forward capability [8].

### PAX A920 Pro

The PAX A920 Pro supports **store-and-forward (SAF) / offline payments** as an optional feature that must be **explicitly activated by the payment processor**, with the merchant assuming associated risks [16][19].

Per Moneris documentation:
- Transactions are automatically forwarded once connectivity is restored [16]
- Only chip card purchases with correct PIN entry and non-expired cards are eligible [16]
- The merchant must enter a valid user ID and passcode for each offline transaction (fraud mitigation) [16]
- **Daily cumulative limit**: Maximum total dollar value capped at 20% of projected monthly volume [16]
- **Maximum transaction amount**: Per-transaction dollar limit must not exceed the daily cumulative limit [16]

**Excluded transactions**: Dynamic Currency Conversion (DCC), Gift and Loyalty, UnionPay, **Interac debit**, EMV fallback, cashback [16]

Global Payments documentation notes that PAX A920 units require firmware version **01.01.11E or higher** for store-and-forward functionality [19].

**Verdict**: All three terminals support offline processing. Clover offers the most generous default limits ($500/txn, $5,000 total, 7-day window) and includes 4G/LTE backup. PAX A920 Pro offers configurable limits but excludes Interac debit. Verifone's SAF supports up to 1,000 stored transactions. For a QSR chain, Clover's approach is most forgiving, but **none of these solutions guarantee approval** for offline transactions — that risk is always borne by the merchant.

---

## 3. Integration with Oracle MICROS Simphony

### The Integration Pathway: Oracle Payment Interface (OPI)

All three terminals integrate with Oracle MICROS Simphony through the **Oracle Payment Interface (OPI)**, which "simplifies credit card payment configuration by enabling Simphony to communicate with payment service providers (PSPs) using a single payment driver" [20]. OPI "enhances security by not handling or storing card holder or sensitive authentication data in Simphony or OPI" and "eliminates credit card batch processing in Simphony by automating end-of-day settlement" [20].

OPI integration requires:
- OPI version 6.2 with the latest patch
- A certified payment processor/middleware vendor (not direct integration)
- Configuration of certificates and IP/port registration

### Verifone Carbon 8 / Victa

**The Carbon 8** runs Android Lollipop (released 2015), which is likely incompatible with modern OPI implementations. This is another reason to exclude it from consideration.

**The Verifone Victa Portable** integrates with Oracle MICROS through OPI via certified payment processors. Moneris states that the Victa provides "integration with existing POS systems via the Moneris Go API" and "a single API and a single experience... ensuring that merchants and partners are not locked into a single option" [4].

Certified OPI integrators that support Verifone terminals include:
- **Global Payments (Heartland)**: "Certified by Oracle, this solution expands payments for Micros RES 3700, Simphony, and E7" [21]
- **Shift4**: Integration with OPI supporting "multiple verification methods like chip-and-PIN, chip-and-signature, and Quick Chip... compatible with a range of EMV-capable devices from Verifone" [22]
- **Adyen**: OPI 6.2 integration supporting "Pay at Counter" and "Pay at Table" [23]

### Clover Station Pro

**No official, certified, direct integration between Clover Station Pro and Oracle MICROS Simphony was found.** This is a critical gap.

Available pathways:
- **Global Payments OPI Middleware**: Heartland/Global Payments offers an Oracle Payment Interface solution for Micros Simphony, but the specific hardware compatibility with Clover terminals is not explicitly confirmed in available documentation [21]
- **Deliverect**: Offers "reliable two-way integration with Oracle's Micros Simphony POS system" for managing online orders, but this is for order management, not payment terminal integration [24]
- **Chetu**: A certified Oracle Partner offering "custom integration services to make MICROS Simphony compatible with any POS system" [25]

The fundamental issue: Clover is typically deployed as a **full POS system** (ordering + payments), not as a payment-only peripheral for an existing POS. Using Clover terminals with an existing MICROS Simphony setup would likely require significant custom middleware development.

### PAX A920 Pro

**The PAX A920 Pro has the strongest documented OPI integration.** Key evidence:

- **Adyen documentation** confirms: "Adyen payment terminals implement Oracle Payment Interface (OPI) 6.2 to integrate to Oracle Simphony. OPI sends all transaction messages directly to Adyen's payment terminal" [23]
- Integration supports **Pay at Counter** and **Pay at Table** workflows, including bill printing, bill splitting, and returning payment to the POS system [23][26]
- **Touché**, a software provider for Food & Beverage and Hospitality, has deployed its Oracle solution on PAX Android SmartPOS devices including the A920 and A920Pro. This integration "enables merchants to streamline ordering, payment, and loyalty processes within their existing POS infrastructure, including Oracle's MICROS Simphony POS system" [27]
- **PayFacto** (Canadian provider) achieved **Class A Certification** for the A920 in Canada and integrates with hospitality POS systems via the **SecureTablePay middleware** [28]

**Verdict**: The **PAX A920 Pro** offers the most proven, documented OPI integration with Oracle MICROS Simphony. Clover Station Pro faces a critical gap — no certified direct integration was found, making it a high-risk choice for your existing MICROS setup. Verifone Victa offers certified integration through multiple processors but requires the OPI middleware pathway.

---

## 4. Bilingual Interface Compliance with Quebec's Bill 96

### Quebec's Bill 96 Requirements (Effective June 1, 2025)

Bill 96 (An Act respecting French, the official and common language of Québec) fundamentally changes language requirements for businesses operating in Quebec [29][30][31]:

- **French must be the primary/default language** for all commercial activities, customer communications, and employee communications
- **Businesses with 25 or more employees** in Quebec must register with the Office québécois de la langue française (OQLF) — your 25-location chain likely exceeds this threshold
- All commercial documents (including receipts, contracts, and customer-facing digital screens) must be in French, with French being "markedly predominant"
- **Fines**: $3,000 to $30,000 per offense, with higher penalties for repeat offenders (up to $90,000 per day for companies)
- The OQLF has expanded authority to investigate complaints, conduct inspections, and issue fines

For payment terminals specifically: "The best POS terminals for Canadian restaurants" guide notes that "Quebec's Bill 96 requires French to appear prominently on receipts and customer-facing screens" [32].

### Verifone Carbon 8 / Victa

**General capability**: Verifone has a long-established presence in French-speaking markets (Verifone France exists as a distinct entity). Verifone terminals globally support multiple languages including French.

For the **Verifone Victa Portable** in Canada:
- Moneris explicitly states: "From language to local presence, our POS systems are made for Quebec" and "We know Quebec because we're rooted here. With offices and local support teams within the province" [33]
- The original Moneris Go (2020) was described as having "a modern and intuitive user interface available in English and French" [34]
- TouchBistro's integration with Moneris Go confirms: "If you need to switch the terminal's default language (e.g., from English to French), you can use the Go app's Settings screen" [35]

**CRITICAL LIMITATION**: The Moneris Go Restaurant KDS app is explicitly "currently available across Canada **except Quebec and in English only**" [36]. This means the native KDS solution for Moneris Go does **not** support French-language operation and is not available in Quebec. For a Quebec-based deployment, this is a significant compliance and operational risk.

### Clover Station Pro

Clover provides multi-lingual support:
- The **Clover Adobe Commerce payments extension** supports "US English, Canadian English, and **Canadian French**" and can be configured to Canadian French as the default [37]
- The **Clover Payments plugin for WooCommerce** supports "English and **Canadian French**" [38]
- The device OS allows language switching through Settings > System > Languages [39]
- Capterra Canada lists "Français Canada (Français)" as an available language option [40]

**Unconfirmed gaps**: The following are NOT explicitly confirmed by available sources:
- Whether the full Clover OS user interface (admin screens, back-office, reporting, employee management, inventory) is fully translated into Canadian French
- Whether the **customer-facing display** prompts, tip prompts, and receipt defaults are available in French by default
- Whether the French translation is complete and accurate (Bill 96 requires high-quality translation: "Low-quality translations can lead to costly legal issues and delays")

### PAX A920 Pro

PAX Technology has a dedicated **PAX France** division that provides "French-language UX/UI designs for Android payment terminals including the A920 and A920Pro" [41]. The Android/PayDroid operating system supports full language localization [41].

Since the PAX A920 Pro runs on Android (PayDroid), it can support full French language localization across:
- Admin screens and back-office functions
- Customer-facing displays
- Receipts (configurable to print in French)
- Signature capture, tip prompts, and all transaction flows

**Key consideration for Bill 96 compliance**: French must be the **default and primary language**, not just an option. The QSR chain should verify with their Canadian payment processor (Moneris, Global Payments, PayFacto) that the terminal software image deployed in Quebec is configured with French as the primary/initial language.

**Verdict**: All three terminals **can** support French-language interfaces through their Android operating systems. However, **none of the manufacturers provide explicit Bill 96 compliance certification** — this must be verified with the specific processor deployment. The PAX A920 Pro's PAX France division and Android platform provide the strongest foundation. The critical warning about Moneris Go Restaurant KDS being unavailable in Quebec and English-only applies regardless of terminal choice if using that ecosystem.

**Action required**: Your chain should:
1. Contractually require your payment processor to provide terminals pre-configured with French as the default language
2. Audit all customer-facing screens, admin screens, and receipt templates for complete French coverage
3. Register with the OQLF (required for 25+ employees)

---

## 5. Total 5-Year Ownership Costs

### Verifone Carbon 8 / Victa Pricing

**Carbon 8**: The Carbon 8 is no longer available as new equipment from authorized Canadian channels. Used units are available on secondary markets (~$200-400 USD) but lack warranties and current certifications [42].

**Verifone Victa Portable (Moneris Go Terminal)**:
- **Hardware**: Available as rental from Moneris. Current promotional offer: "$0 POS rental fees for up to 1 year" through the "Save on Go" promotion (valid until June 2, 2026) [43]
- **Historical rental fee (Moneris Go 2020)**: $29.95/month plus taxes [34]
- **Moneris Go Restaurant POS**: $64.95/month (includes software and terminal) or $29.99/month software-only [36]
- **Transaction fees**: 2.35% + $0.10 per credit transaction; $0.12 per Interac debit transaction [36]
- **Hardware bundles**: Starter Bundle ($499) includes cash drawer + receipt printer; Core Bundle ($999) includes iPad stand + kitchen printer [36]

### Clover Station Pro Pricing

| Cost Component | Price Range (USD) | Source |
|---|---|---|
| Station Pro (hardware package) | $1,649.00 | [8] |
| Station Duo | $1,899 - $1,995 | [7][44] |
| Station Solo | $1,799 | [45] |
| Essentials monthly subscription | $29.95/month | [46] |
| Register (counter/restaurant) | $39.95/month | [46] |
| Register + Table Service | $84.95 - $89.95/month | [46][47] |
| Advanced | $189/month | [48] |
| Per extra device fee | $11.95 - $19.95/month | [46][47] |

**Processing fees** (Clover direct): 2.3% + $0.10 in-person; 3.5% + $0.10-$0.15 online/keyed-in [46][47][48]

**Hidden costs identified by multiple sources**:
- PCI compliance fee: $9.95 - $10/month per location [49]
- Platform access fee: $27.95/month per location [49]
- Statement charges: $5 - $15/month [49]
- Monthly minimums: ~$49/month [49]
- **Essential app subscriptions**: $150 - $950+/month per location (critical for QSR functionality) [49]
- One analysis states: "Hidden fees... commonly add $100–$200/month per location" [49]

**Contract terms**: Typical 36-month or 48-month commitment. Early termination fee equals "the remaining balance on your contract" [49][50]. Cancellation often takes 2-3 months [49].

**Hardware is processor-locked**: "Clover hardware is processor-locked. If you want to switch processors, you generally cannot bring your Clover hardware with you" [50].

**Warranty**: Standard 1-year full swap-out warranty on main devices [51].

**Multi-location discounts**: Clover "does not publish volume-based discount tiers for transaction fees" [49]. Processing fees are negotiable through ISOs but hardware pricing is generally fixed.

### PAX A920 Pro Pricing

| Cost Component | Price Range (CAD equivalent) | Source |
|---|---|---|
| A920 Pro hardware (purchase) | $400 - $900 USD (~$550 - $1,250 CAD) | [10][52] |
| A920 Pro (US reseller pricing) | $360 - $400 USD | [53][54] |
| Monthly rental/lease | ~$20-40/month | [52][53] |
| Extended warranty | ~$50-100/year | [10] |
| Battery replacement (every 2 years) | ~$30-50 | [10] |

**Canadian processing fees (Moneris Flat Rate)**:
- Credit card (card-present): 2.65% + $0.10 [55]
- Interac Debit (card-present): $0.12 per transaction [55]

**Interchange-plus pricing** (recommended for high-volume QSR): Available from Moneris and other Canadian processors; typically more cost-effective than flat-rate for high transaction volumes [55].

**Multi-location discounts**: "Bulk purchases reduce per-unit costs significantly" [56]. No minimum order requirements from wholesale distributors [57]. The 25-location QSR chain should negotiate directly with Canadian processors.

### 5-Year TCO Comparison (Per Location, Excluding Processing Fees)

| Cost Component | Verifone Victa (Moneris Rental) | Clover Station Pro (Purchase) | PAX A920 Pro (Purchase) |
|---|---|---|---|
| Hardware (year 1) | $0 (promotional rental) or ~$360/yr rental | $1,649 purchase | $550 - $1,250 purchase |
| Monthly subscription | $64.95/mo ($779/yr) | $84.95 - $189/mo ($1,019 - $2,268/yr) | $20 - $40/mo ($240 - $480/yr) |
| Hidden/subscription fees | Included in monthly | $100 - $200/mo extra ($1,200 - $2,400/yr) | Minimal |
| Warranty/repairs | Included (rental) | 1-year warranty + extended options | $50 - $100/yr extended warranty |
| Battery replacement | Included (rental) | Not specified | $30 - $50 every 2 years |
| **Estimated 5-year total (per location)** | **~$3,895 (rental, no hardware)** | **$4,900 - $10,800** | **$2,510 - $4,050** |
| **25 locations (5-year total)** | **~$97,375** | **$122,500 - $270,000** | **$62,750 - $101,250** |

**Important note**: These estimates exclude payment processing fees, which will vary significantly based on transaction volumes and negotiated rates. Processing fees typically represent the largest cost component for QSR operations.

**Verdict**: The **PAX A920 Pro** offers the lowest 5-year TCO, particularly when hardware is purchased outright rather than leased. The **Verifone Victa** rental model through Moneris offers predictable monthly costs and includes support/maintenance. The **Clover Station Pro** carries the highest TCO due to higher hardware costs, monthly subscription fees, and hidden fees that "commonly add $100-$200/month per location" [49].

---

## 6. Contactless Payment Limits in Canada

### Current Canadian Contactless Limit

As of 2026, the contactless payment limit across Canada is **CA$250** for all major networks [58][59][60]:
- **Visa**: Up to $250 per tap
- **Mastercard**: Up to $250 per tap
- **American Express**: Up to $250 per tap
- **Interac Flash (debit)**: $100 per tap (remained unchanged)

### Handling Transactions Above the Contactless Limit

All three terminals follow the same standard workflow:
1. Terminal prompts the customer to **insert their card into the EMV chip reader**
2. Customer enters their **PIN** on the terminal's touchscreen
3. Transaction proceeds as a chip-and-PIN transaction

### Terminal-Specific Capabilities

- **Verifone Carbon 8**: Supports NFC/contactless payments including MiFare and NFC/CTLS schemes [1]. Smart card reader handles EMV chip transactions [1].
- **Verifone Victa**: Supports contactless payments with "behind-the-display contactless optimization for better tap performance" [3]. Also supports Magstripe, smartcard/EMV chip, and mobile payments [3].
- **Clover Station Pro**: Integrated EMV/NFC payment acceptance on dual-screen models [8]. The Station **Solo** model "cannot take contactless cards and digital payments on its own; additional hardware is required" [45]. Supports Apple Pay, Google Pay, Samsung Pay, and card tap [61][62].
- **PAX A920 Pro**: Supports contactless NFC, chip & PIN, and magnetic stripe [10][63]. All major contactless methods (Apple Pay, Google Pay, Samsung Pay) supported [10].

**Verdict**: All three terminals handle Canadian contactless limits identically. The key differentiator is whether contactless is built-in (Clover Station Pro dual-screen, Verifone Victa, PAX A920 Pro) or requires add-on hardware (Clover Station Solo).

---

## 7. Tip Adjustment Workflows

### Verifone Terminals

Verifone terminals offer flexible tipping features configurable through the payment software [64]:
- **Tip prompt** appears on the PIN pad
- Tip amounts can be displayed as **percentages** (max 99.99%) or **fixed values** (max $99,999.99) [64]
- Options to **allow custom tip amounts** and include a "No Tip" option [64]
- **Minimum tip thresholds** can be configured, changing prompts based on transaction totals [64]
- Tipping can be **enabled/disabled at the product level** (Department and PLU files) [64]

**Tip adjustment process**: "Tip adjustment is adding a tip after a sale. Tip adjustment holds the transaction until a tip is added manually and then the transaction is sent to settlement" [65]. All tips must be adjusted prior to batch settlement [65].

For the **Verifone Victa** via Moneris Go: The Go platform supports pay-at-table workflows, which typically include tip entry at the table. Business can also access the "Moneris Go App Marketplace" for additional tipping applications [4].

### Clover Station Pro

Clover provides per-transaction tip settings through the `tipMode` SDK parameter [66]:
- **TIP_PROVIDED** – Tip already included
- **ON_SCREEN_BEFORE_PAYMENT** – Tip prompt shown before payment authorization (QSR mode)
- **ON_SCREEN_AFTER_PAYMENT** – Tip prompt shown after payment authorization (table service mode)
- **NO_TIP** – No tip prompt

**Customizable tip suggestions**: Developers can set custom `tipSuggestions` to display tailored tipping options per transaction [66]. Preset options (e.g., 18%, 20%, 25%) are customizable through the Clover Dashboard [67].

**Limitation**: Clover offers **manual tip pooling** only — "Staff management includes fingerprint login but manual tip pooling" [46][47]. No automated tip pooling for operations with pooled tip arrangements.

**Scan to Pay option**: Customers can pay and tip via a digital wallet by scanning a QR code on their bill, currently supporting Apple devices with Android features coming soon [68].

### PAX A920 Pro

The PAX A920 Pro supports comprehensive tip adjustment workflows [69][70][71]:

**Post-authorization tip adjustment**:
1. Navigate to the transaction batch
2. Select the transaction to adjust [69][70]
3. Press "ADJUST" [70][71]
4. Enter the tip amount and press "CONFIRM" [69][70][71]
5. Press "OK" to confirm [71]

**Tips before/after authorization**: Tips can be added during the authorization process (pre-auth) or after as a post-auth adjustment [69][70][71]. For pre-auth completion, the tip is included in the final amount captured.

**Customizable tip prompts**: The Cybersource developer documentation confirms the PAX A920 supports tipping as a standard feature [72]. Tip prompts and preset amounts (15%, 18%, 20%, or custom) can be configured through the payment application running on the Android platform [72].

**Verdict**: All three terminals support tip adjustment workflows essential for your table service elements. The **Clover Station Pro** offers the most granular control (pre-payment vs. post-payment tip prompts) but lacks automated tip pooling. The **PAX A920 Pro** and **Verifone Victa** both support pre-auth and post-auth tip adjustment with customizable prompts.

---

## 8. Kitchen Display System (KDS) Compatibility

### Verifone / Moneris KDS

Moneris offers a **Kitchen Display System (KDS)** as a free iPad app called "Moneris Kitchen Display," available on the App Store [36]. Features:
- Instantly displays new orders the moment they're sent from the POS [36]
- Redesigned order tiles showing more order details, order type, server name, order number, and elapsed time [36]
- Auto-refresh enhancements ensuring orders are not missed [36]
- "The developer, Moneris Solutions Corporation, states that the app does not collect any user data" [36]

**CRITICAL LIMITATION**: The Moneris Go Restaurant KDS app is "currently available across Canada **except Quebec and in English only**" [36]. This is explicitly stated on the Moneris Go Restaurant page. For a Quebec-based deployment, this KDS solution:
1. **Is not available in Quebec**
2. **Is English-only**

This may require a third-party KDS solution that integrates with Moneris Go API.

### Clover Station Pro KDS

Clover offers its own **Kitchen Display System** specifically designed for restaurant environments [73][74][75]:
- **Two screen sizes**: 14-inch and 24-inch [73][74]
- **14-inch display**: "Highest heat tolerance (122°F) on the market," temperature-resistant aluminum with anti-fingerprint coating [74]
- **24-inch display**: Ideal for overhead installation, supports programmable bump bar (KB9000) for hands-free order management [73]
- Supports **Wi-Fi and LAN connections**, compatible with third-party mounting equipment [74]
- Uses "bi-directional speakers to deliver clear alerts in noisy kitchen settings" [74]
- **Automatically connects with Clover POS** for streamlined front- and back-of-house communication [74]

**KDS features**:
- Replaces paper tickets and reduces risk of lost tickets [75]
- Supports high-volume kitchens and multiple stations [75]
- Aggregates all orders including on-premise, off-premise, and third-party channels [74][75]
- Custom ticket layouts routing orders to different prep stations [73]
- Performance reporting including prep time, fulfillment metrics, and staff performance [74][75]

**Third-party KDS options**: Fresh KDS (works on Android/iOS, lower monthly fees) and Simple KDS for Clover (free, turns any Android tablet into a KDS station) are available [76][77].

### PAX A920 Pro KDS

PAX Technology has launched its own **Elys Display Kitchen Display System (KDS)** [78][79]:
- **K2160**: 16-inch Android-powered touchscreen
- **K2220**: 22-inch Android-powered touchscreen
- Both feature **IP55 waterproof and dustproof ratings** for demanding kitchen environments [78][79]
- Support **Power over Ethernet (PoE)** [78][79]
- Flexible positioning with gyroscope sensors for landscape or portrait mode
- Can be tabletop or wall-mounted via VESA brackets [78][79]

**PAX PB20 Bump Bar**:
- Industry's first configurable keyboard with magnetic keys [78][79]
- QuickNav Scroll Wheel using magnetic angle sensor detection technology
- Aluminum housing with sealed cover, IP65 rating
- Estimated 50 million click lifespan [78][79]

**Integration**: The Elys Display seamlessly integrates with PAX's own EPOS products and other PAX payment devices [78][79]. It connects to "thousands of POS software providers" and synchronizes via PAX's **LinkUp application** [78][79].

**Critical consideration**: The PAX A920 Pro is a payment terminal. Its KDS compatibility depends on the POS software integration, not direct connection between the payment terminal and kitchen displays. For your existing Oracle MICROS Simphony KDS setup:
- Orders would be routed from MICROS Simphony to the KDS (not through the PAX terminal)
- The PAX terminal handles payment processing only
- PAX's KDS (Elys Display) would need to integrate with MICROS Simphony, not with the A920 Pro directly

### Recommendation for Your Existing KDS Setup

Since you already have an existing Oracle MICROS KDS setup, the **KDS compatibility question is largely decoupled from the payment terminal decision**:

1. **MICROS Simphony** handles order entry and routing to the kitchen
2. Your **existing KDS** displays orders from MICROS
3. The **payment terminal** (whichever you choose) handles payment processing only
4. The payment terminal does NOT route orders to the kitchen

The critical integration is between the payment terminal and MICROS Simphony (covered in Section 3), not between the payment terminal and KDS.

**Verdict**: The Clover native KDS ecosystem is the most comprehensive, but it requires adopting Clover as your full POS system — incompatible with your existing MICROS Simphony setup. PAX's Elys Display KDS offers industrial-grade hardware but requires integration with your existing MICROS system. The Moneris Go KDS has a critical Quebec availability limitation. **Your best path is to maintain your existing MICROS KDS setup and choose the payment terminal that best integrates with MICROS Simphony.**

---

## Summary and Recommendations

### Overall Assessment

| Dimension | Verifone Carbon 8 | Verifone Victa (Current Equivalent) | Clover Station Pro | PAX A920 Pro |
|---|---|---|---|---|
| **Transaction Speed** | ❌ Obsolete (Android Lollipop) | ✅ Octa-core option, 12hr+ battery | ✅ Dual-screen boosts throughput | ✅ A920MAX 40% faster |
| **Offline Payments** | ✅ SAF (1,000 txns) | ✅ SAF (processor-dependent) | ✅ Best limits ($500/txn, 7 days) | ✅ SAF (no Interac debit) |
| **MICROS Integration** | ❌ Incompatible (old OS) | ✅ Via OPI (Global Payments, Adyen, Shift4) | ❌ No certified integration found | ✅ **Best** — OPI 6.2+ certified |
| **Bill 96 Compliance** | ⚠️ Possible but unverified | ⚠️ French UI available; **KDS not available in Quebec** | ⚠️ French language support partial | ✅ PAX France division, full Android localization |
| **5-Year TCO (25 locations)** | ❌ Not available new | ~$97,375 (rental) | $122,500 - $270,000 | **$62,750 - $101,250** |
| **Contactless Limits** | ✅ $250 CAD | ✅ $250 CAD | ✅ $250 CAD | ✅ $250 CAD |
| **Tip Workflows** | ✅ Pre/post-auth adjustment | ✅ Customizable prompts | ✅ Most granular control | ✅ Pre/post-auth adjustment |
| **KDS Compatibility** | ❌ Not applicable | ⚠️ KDS not available in Quebec | ✅ Comprehensive but requires full Clover POS | ✅ Elys Display (needs MICROS integration) |

### Primary Recommendation: PAX A920 Pro (or A920MAX)

The **PAX A920 Pro** (or the newer **A920MAX** for highest-volume locations) is the strongest choice for your QSR chain because:

1. **Certified Oracle MICROS Simphony integration** via OPI 6.2+ — the only terminal with documented, certified integration
2. **Active Canadian certification** — PayFacto Class A certification, PCI PTS 5.x/6.x/7.x, available through Moneris, Global Payments, and PayFacto
3. **Lowest 5-year TCO** — hardware purchase costs $400-$900 per unit vs. $1,649+ for Clover, with lower monthly subscription fees
4. **Android platform** — full language localization support through PAX France division, configurable for French as default language
5. **PAX Elys Display KDS** available as a complementary product if you choose to expand your KDS hardware

**Caveats to manage**:
- Verify French language default configuration with your processor before deployment
- PAX Elys KDS requires separate integration with MICROS Simphony — your existing KDS may be preferable
- The standard A920 Pro battery (7.5-9.5 hours) is adequate for a single shift but may require charging stations

### Secondary Recommendation: Verifone Victa Portable (via Moneris)

If you prefer a rental model with predictable monthly costs and are willing to accept the Quebec KDS limitation, the **Verifone Victa Portable** (Moneris Go Terminal) is a strong alternative:

1. **Newest device** in Canada (launched February 2026) with PCI 7 readiness
2. **Rental model** eliminates upfront hardware costs and includes maintenance
3. **12+ hour battery** with processing every 2 minutes provides excellent runtime
4. **Moneris is Canada's #1 processor** with local Quebec offices and French-language support
5. **OPI integration** available through multiple certified processors

**Critical caveat**: The Moneris Go Restaurant KDS app is **not available in Quebec and is English-only**. You would need to maintain your existing MICROS KDS setup or find a third-party KDS solution.

### Not Recommended: Verifone Carbon 8

The **Verifone Carbon 8 is not recommended** for these reasons:
- Runs Android Lollipop (released 2015) — obsolete and likely no longer receiving security updates
- PCI PTS POI approval expires **April 30, 2027** — less than one year from today
- May be impossible to source as new equipment from authorized Canadian channels
- Limited official support and replacement parts availability

### Not Recommended: Clover Station Pro

The **Clover Station Pro is not recommended** for these reasons:
- **No certified direct integration with Oracle MICROS Simphony** — this is a critical gap that would require custom middleware development
- Highest TCO due to hardware costs, monthly subscriptions, and hidden fees ($100-$200/month per location)
- **Processor-locked hardware** — cannot switch processors without replacing all terminals
- Long-term contracts (36-48 months) with substantial early termination penalties

### Action Plan for Your 25-Location QSR Chain

1. **Immediate**: Contact Canadian payment processors (Moneris, Global Payments, PayFacto) for PAX A920 Pro / A920MAX quotes reflecting 25-location volume. Request interchange-plus pricing rather than flat-rate.
2. **Bill 96 compliance**: Contractually require French as the default language on all terminal software images. Register with the OQLF. Conduct a full audit of all customer-facing screens, admin screens, and receipt templates.
3. **MICROS integration**: Request a demonstration of OPI 6.2+ integration with your specific Oracle MICROS Simphony version from the processor.
4. **KDS strategy**: Maintain your existing MICROS KDS setup. The payment terminal decision is largely independent of KDS compatibility.
5. **Pilot program**: Deploy PAX A920 Pro terminals at 2-3 locations (one in Quebec, one in New Brunswick) for 30-60 days before full rollout.

---

### Sources

[1] Verifone Documentation Portal - Carbon Installation Guide: https://docs.verifone.com/device-installation-guides/installation-guides/device-installation-guides/android-devices/carbon

[2] Financial IT - Verifone Launches Carbon 8 Portable POS Terminal: https://financialit.net/news/payments/verifone-launches-carbon-8-portable-pos-terminal

[3] Verifone Victa Portable Plus Datasheet: https://cdn.prod.website-files.com/6877dbe2d81008ec40dd7770/69ab54bd5582db93331a50be_Victa%20Portable%20Plus%20Datasheet.pdf

[4] Moneris - Expands Go Commerce Suite (Feb 3, 2026): https://www.moneris.com/en/media-room/news/moneris-expands-its-go-commerce-suite

[5] Clover Station 2018 Specifications (EPSLA PDF): https://www.epsla.com/sites/default/files/files/Clover%20Station%202018.pdf

[6] Clover Station Solo Specifications - Clearly Payments: https://www.clearlypayments.com/products/pos/clover/clover-station-solo

[7] Clover Station Duo POS System - Clearly Payments: https://www.clearlypayments.com/products/pos/clover/clover-station-duo

[8] Clover Station Pro Bundle - Commerce Technologies: https://commercetech.com/eblog/clover-station-pro-bundle-with-customer-facing-screen

[9] Clover Gen 1 End-of-App-Update Announcement: https://docs.clover.com/dev/docs/clover-devices-tech-specs

[10] PAX A920 Pro Payment Terminal - Discount Credit Card Supply: https://www.discountcreditcardsupply.com/products/pax-a920-pro-payment-terminal

[11] PAX A920 vs A920 Pro Comparison: https://aposmerchant.com/pax-a920-vs-a920-pro-comparison

[12] PAX A920 Pro vs A920 Max Comparison: https://blog.octopos.com/2025/07/10/pax-a920-pro-vs-a920-max-which-one-to-pick

[13] Verifone Cloud - Store and Forward (SAF): https://verifone.cloud/print/pdf/node/20078

[14] Oolio Support - Store and Forward Functionality: https://support.oolio.com/offlinepayments

[15] Bank of America - Payments App Store and Forward Mode: https://merchanthelp.bankofamerica.com/Payments_Application_Store_and_Forward__SAF__Mode

[16] Moneris - Offline Payments Processing: https://www.moneris.com/help/V400m-WH-EN/Transactions/Offline_Payments_(formally_known_as_Store_and_Forward)_processing.htm

[17] Clover Help Center - Set Up Offline Payments: https://www.clover.com/help/set-up-offline-payments

[18] Lightspeed Retail POS - Offline Credit Payments: https://shopkeep-support.lightspeedhq.com/hc/en-us/articles/47479984822299-Offline-Credit-Payments

[19] Heartland/Global Payments - PAX Store and Forward Setup Guide: https://heartlandpos.zendesk.com/hc/en-us/articles/1260806110890-PAX-Store-and-Forward-Setup-Guide

[20] Oracle - The Oracle Payment Interface (OPI): https://docs.oracle.com/cd/E76065_01/doc.29/e69879/c_payments_opi.htm

[21] Global Payments Developer - Oracle Payment Interface: https://developer.globalpayments.com/heartland/payments/in-store/pos-middleware/oracle-payment-interface

[22] Shift4 - Integration to Oracle Payment Interface: https://www.shift4.com/news/shift4-announces-integration-to-the-oracle-payment-interface

[23] Adyen Docs - Oracle Simphony Integration: https://docs.adyen.com/plugins/oracle-simphony

[24] Deliverect - Oracle Micros Simphony Integration: https://www.deliverect.com/integrations/oracle-micros-simphony

[25] Chetu - Oracle MICROS Simphony Integration Services: https://www.chetu.com/hospitality/oracle-micros-simphony.php

[26] DNA Payments - Pay at Counter and Pay at Table on Oracle Cloud Marketplace: https://www.dnapayments.com/news/pay-reception-pay-counter-and-pay-table-now-available-oracle-cloud-marketplace

[27] PAX Technology - Touché Deploys Oracle Solution on PAX Android: https://www.paxtechnology.com/blog/touche-deploys-oracle-solution-for-f-b-and-hospitality-clients-on-pax-android

[28] PayFacto - Launches PAX A920 in Canada: https://payfacto.com/payfacto-launch-pax-a920-android-payment-terminals-canada

[29] RWS - Quebec's Bill 96 Compliance Services: https://www.rws.com/localization/services/resources/quebecs-bill-96

[30] Language IO - Understanding Bill 96: https://languageio.com/resources/blogs/understanding-bill-96-a-guide-for-businesses-operating-in-quebec-canada

[31] TransPerfect - Quebec's Bill 96 Key Changes in 2025: https://www.transperfect.com/blog/quebecs-bill-96-key-changes-2025

[32] Rosper - Best POS Terminals for Canadian Restaurants 2026: https://blog.rospertech.com/best-pos-terminals-canadian-restaurants-2026

[33] Moneris - Quebec Landing Page: https://www.moneris.com/en/canada/quebec

[34] Moneris - Launches All-in-One Portable Payment Solution (2020): https://www.moneris.com/en/media-room/news/moneris-launches-all-in-one-portable-payment-solution

[35] TouchBistro Help - Setting Up Moneris Go Plus: https://help.touchbistro.com/s/article/Setting-Up-Moneris-Go-Plus

[36] Moneris Go Restaurant POS Solution: https://www.moneris.com/en/solutions/pos-systems/go-restaurant

[37] Clover Adobe Commerce Payments Extension: https://marketplace.clover.com/apps/adobe-commerce-payments

[38] Clover Payments Plugin for WooCommerce: https://wordpress.org/plugins/clover-payments/

[39] Clover Help - Change Device Language: https://www.clover.com/help/change-device-language

[40] Capterra Canada - Clover POS: https://www.capterra.ca/software/135988/clover

[41] Teddy Graphics - UX for Android Payment Terminals for PAX France: https://teddygraphics.com/pax-france.html

[42] eBay - Verifone Carbon 8 Listings: https://www.ebay.com/itm/318082528489

[43] Moneris - Save on Go Promotion: https://go.moneris.com/saveongo

[44] Gravity Payments Canada - Clover Station Duo: https://gravitypayments.com/devices/clover-station-duo-canada

[45] Clover Station Solo - Expert Market: https://www.expertmarket.com/pos/clover-review

[46] Expert Market - Clover Review 2026: https://www.expertmarket.com/pos/clover-review

[47] Clover Register + Table Service Pricing: https://www.clover.com/pricing

[48] Clover Advanced Plan Pricing: https://www.clover.com/pricing

[49] CheckThat.ai - Clover Hidden Fees Analysis: https://checkthat.ai/blog/clover-hidden-fees

[50] Clover Station Duo Review - Merchant Maverick: https://www.merchantmaverick.com/reviews/clover-station-duo-review

[51] Clover Warranty Information: https://www.clover.com/warranty

[52] Thrifty Payments - PAX A920 Payment Terminal Review: https://thriftypayments.com/blog/pax-a920-payment-terminal-features-price-and-review-2025

[53] PAYBOTX - PAX A920 Pro: https://paybotx.company.site/Pax-A920-Pro-p629364972

[54] Commerce Technologies - PAX A920 Pro: https://commercetech.com/pax-a920-pro

[55] Moneris - Online Payment Fees: https://www.moneris.com/en/pricing

[56] PAX Terminals Guide - Unison Payment Solutions: https://www.unisonpayment.com/blog/pax-terminals-guide

[57] All-Star Terminals - PAX Terminals: https://allstarterminals.com/collections/pax-terminals

[58] Global Payments Canada - Contactless Payment Limits: https://www.globalpayments.com/en-ca/insights/everything-you-need-to-know-about-contactless-payment-limits

[59] Clearly Payments - Contactless Payment in Canada: https://www.clearlypayments.com/blog/time-to-get-a-contactless-payment-canada

[60] Interac - Tap Payment Limits: https://www.interac.ca/en/consumers/faqs/what-is-the-contactless-limit/

[61] Gravity Payments - Clover Contactless Solutions: https://gravitypayments.com/blog/clover-contactless-solutions-for-your-restaurant

[62] Clover Blog - Creating a Contactless Payment Experience: https://blog.clover.com/creating-a-contactless-payment-experience

[63] PAX Technology - A920Pro Specifications: https://www.paxtechnology.com/a920pro

[64] Verifone Cloud - Tipping Feature Reference: https://verifone.cloud/sites/default/files/inline-files/%5Bcurrent-date%3Ahtml_month%5D/Tipping_0.pdf

[65] Bank of America - Restaurant App Tip Adjustment: https://merchanthelp.bankofamerica.com/Restaurant_App_Tip_Adjustment

[66] Clover Developer Docs - Tip Mode Settings: https://docs.clover.com/dev/docs/tip-mode

[67] Clover Support - Tip Percentages: https://support.clover.com/articles/tip-percentages

[68] Clover - Scan to Pay Feature: https://www.clover.com/scan-to-pay

[69] BASYS Processing - PAX A920 Tip Adjustment: https://support.basyspro.com/hc/en-us/articles/19944279058964-PAX-A920-Card-Transaction-Adjust-Tips

[70] Talus Pay - How to Adjust Tips on PAX A920: https://taluspay.zendesk.com/hc/en-us/articles/27031070834579-How-to-Adjust-Tips-on-a-Pax-A920

[71] ZBS Helpdesk - PAX A920 How to Add Tips: https://help.zbspos.com/en/article/pax-a920-how-to-add-tips-qk4k8g

[72] Cybersource Developer Center - PAX A920 Payment Terminal: https://developer.cybersource.com/docs/cybs/en-us/payworks-sdk/developer/all/na/payworks-sdk/card-readers/pax-a920.html

[73] Clover Kitchen Display System: https://www.clover.com/kitchen-display-system

[74] Clover KDS - Heat-Resistant Display: https://www.clover.com/kds-14

[75] Clover - Restaurant KDS Features: https://www.clover.com/restaurant/kds

[76] Fresh KDS for Clover: https://www.freshkds.com/clover

[77] Simple KDS for Clover (Google Play): https://play.google.com/store/apps/details?id=com.simplekds.clover

[78] PAX Technology - Launches Innovative KDS and Bump Bar: https://www.pax.us/about/press-room/pax-technology-inc-launches-innovative-kds-and-bump-bar-for-restaurants

[79] PAX Technology Blog - Kitchen Displays and Bump Bar: https://www.paxtechnology.com/blog/pax-technology-launches-kitchen-displays-and-bump-bar-for-restaurants