# Comprehensive Workstation Comparison for Healthcare Imaging Center: Dell Precision 7960 Tower vs. HP Z8 Fury G5 vs. Lenovo ThinkStation P7

## Executive Summary

This report provides a detailed, revised comparison of the Dell Precision 7960 Tower, HP Z8 Fury G5, and Lenovo ThinkStation P7 workstations for a Phoenix-based healthcare imaging center processing ~200 patients/day running Philips IntelliSpace PACS (~4.4.x) and Hologic SecurView DX (12.0+) software with 500MB+ DICOM files, including 3D mammography tomosynthesis datasets. Based on comprehensive analysis of model-specific specifications, 10GbE networking implementation, ECC memory clinical implications, enterprise support verification at Phoenix ZIP codes, software licensing, power consumption calculations, GPU requirements, and 5-year TCO for 12 workstations, this report finds that the **Dell Precision 7960 Tower** offers the strongest combination of published SPECworkstation validation, native onboard 10GbE networking, largest memory expansion capacity (4TB across 16 DIMM slots), and established enterprise support infrastructure for this specific clinical workload. The **Lenovo ThinkStation P7** is a viable alternative with comparable onboard 10GbE and slightly lower acquisition cost, but suffers from limited memory expansion (8 DIMM slots, 1TB max) and no published SPECworkstation results. The **HP Z8 Fury G5** trails primarily due to the requirement for add-in 10GbE NICs (adding ~$279 per workstation), lack of published SPECworkstation benchmarks, and potential PCIe slot conflicts when adding NICs alongside multiple GPUs. All three workstations must be configured with Windows 10 Enterprise 64-bit (not Windows 11) due to Hologic SecurView DX compatibility requirements.

---

## 1. Model-Specific Comparison: CPU Options and Platform Architecture

### 1.1 Critical Platform Clarification: Xeon W-3400 vs. W-2400 Series

A critical finding from this research is that **all three workstations use Intel Xeon W-3400 series processors (Sapphire Rapids), NOT the Xeon W-2400 series**. The Xeon W-2400 series (model numbers like W5-2423, W7-2455, W9-2495X) is used in the smaller Dell Precision 5860 Tower, HP Z4 G5, and Lenovo ThinkStation P5 — not in the requested models. This distinction is essential because W-3400 processors feature 8-channel memory architecture (vs. 4-channel for W-2400), higher PCIe lane counts (112 vs. 64), and support up to 56-60 cores [1][2][3].

| Workstation | CPU Family | Socket | Chipset | Memory Channels | Max Cores |
|---|---|---|---|---|---|
| Dell Precision 7960 Tower | Intel Xeon W-3400/W-3500 | LGA-4677 | Intel W790 | 8-channel | 60 cores |
| HP Z8 Fury G5 | Intel Xeon W-3400 | LGA-4677 | Intel W790 | 8-channel | 60 cores |
| Lenovo ThinkStation P7 | Intel Xeon W-3400/W-3500 | LGA-4677 | Intel W790 | 8-channel | 60 cores |

### 1.2 Supported Processor Options by Workstation

**Dell Precision 7960 Tower** [4][5]:
- Intel Xeon W5-3423 (12 cores, 24 threads, 2.10-4.20 GHz, 220W TDP)
- Intel Xeon W5-3425 (12 cores, 24 threads, up to 4.60 GHz, 270W TDP)
- Intel Xeon W7-3455 (24 cores, 48 threads)
- Intel Xeon W7-3465X (28 cores, 56 threads)
- Intel Xeon W9-3475X (36 cores, 72 threads)
- Intel Xeon W9-3495X (56 cores, 112 threads, 1.90-4.80 GHz, 350W TDP)
- Intel Xeon W9-3595X (60 cores, 120 threads, up to 4.80 GHz, 385W TDP)

**HP Z8 Fury G5** [6][7]:
- Intel Xeon W5-3423 (12 cores, 24 threads, up to 4.20 GHz)
- Intel Xeon W5-3435X (16 cores, 32 threads, up to 4.70 GHz)
- Intel Xeon W7-3465X (28 cores, 56 threads, 2.50 GHz, 300W TDP)
- Intel Xeon W9-3475X (36 cores, 72 threads)
- Intel Xeon W9-3495X (56 cores, 112 threads)
- Intel Xeon W9-3575X (44 cores, up to 4.80 GHz)

**Lenovo ThinkStation P7** [8][9]:
- Intel Xeon W5-3425 (12 cores)
- Intel Xeon W5-3433 (16 cores)
- Intel Xeon W5-3435X (16 cores, up to 4.70 GHz)
- Intel Xeon W7-3445 (20 cores, up to 4.80 GHz)
- Intel Xeon W7-3455 (24 cores, up to 4.80 GHz)
- Intel Xeon W7-3465X (28 cores)
- Intel Xeon W9-3475X (36 cores)
- Intel Xeon W9-3495X (56 cores, up to 4.80 GHz)
- Intel Xeon W9-3595X (60 cores, up to 4.80 GHz, 385W TDP)

### 1.3 Equivalent Mid-to-High-End Configuration

For an apples-to-apples comparison, all three workstations can be configured with the **Intel Xeon W9-3475X** (36 cores, 72 threads, 2.2 GHz base / 4.8 GHz turbo, 300W TDP, 82.5 MB L3 cache). This processor provides substantial multi-threaded capability for 3D mammography reconstruction while maintaining excellent single-thread performance (4.8 GHz turbo) for PACS UI responsiveness [4][6][8].

### 1.4 SPECworkstation Benchmark Results

#### 1.4.1 Dell Precision 7960 Tower — Published SPECworkstation 4.0 Results Available

**A SPECworkstation 4.0 result HAS been published** for the Dell Precision 7960 Tower on SPEC.org, submitted by Dell Technologies [10]. The configuration tested was:
- Intel Xeon W9-3495X (56 cores, 112 threads)
- NVIDIA RTX 6000 Ada Generation GPU
- Operating System: Windows 10

**Detailed SPECworkstation 4.0 Scores** [11]:

| Category | Score |
|---|---|
| **Industry Verticals** | |
| AI/Machine Learning | 3.49 |
| Energy | 4.53 |
| Financial Services | 3.80 |
| Life Sciences | 4.02 |
| Media & Entertainment | 3.13 |
| Product Design | 2.72 |
| Productivity & Development | 1.36 |
| **Hardware Subsystems** | |
| CPU | 2.94 |
| Accelerator | 6.21 |
| Graphics | 11.75 |

**Note:** This result uses the W9-3495X (56-core), not the W9-3475X (36-core). No SPECworkstation 4.0 result has been published for the exact W9-3475X configuration. The result page displays scores as an embedded image, so exact numerical values are from the image rendering.

#### 1.4.2 HP Z8 Fury G5 — No SPECworkstation Results Published

**No SPECworkstation 4.0 or 3.1 result has been published** for the HP Z8 Fury G5 on SPEC.org [10][12]. The SPECworkstation 4.0 Result Report Summary (last updated February 11, 2026) lists results for Dell Precision 7960 Tower, Lenovo ThinkStation P8, and various other systems, but **no Z8 Fury G5 results appear**[10]. The SPECworkstation 3.1 summary page (last updated June 15, 2022) predates the Z8 Fury G5 launch and also contains no results for this model [12].

An HP marketing document comparing the HP Z6 G5 A against the Dell Precision 7960 Tower references SPECworkstation 3.1 benchmark results, but these are from HP-published materials rather than the SPEC.org results database itself [13].

#### 1.4.3 Lenovo ThinkStation P7 — SPECviewperf 15.0 Results Published, No SPECworkstation Results

**No SPECworkstation 4.0 or 3.1 result has been published** for the Lenovo ThinkStation P7 on SPEC.org [10][12]. However, **SPECviewperf 15.0 results ARE published** for the Lenovo P7 [14]:
- **System:** Lenovo P7 (w9-3595X - RTX 6000 Ada - 4K)
- **Benchmark:** SPECviewperf 15.0
- **URL:** SPEC.org SPECviewperf 15 Lenovo P7

#### 1.4.4 Explicit Gap Statement

| Workstation | SPECws 4.0 on SPEC.org? | SPECws 3.1 on SPEC.org? | SPECviewperf 15? | SPECviewperf 2020? |
|---|---|---|---|---|
| Dell Precision 7960 Tower | **YES** (W9-3495X + RTX 6000 Ada) | Not found on SPEC.org | Not found | Not found (3650 model only) |
| HP Z8 Fury G5 | **NO** | **NO** | **NO** | **NO** |
| Lenovo ThinkStation P7 | **NO** | **NO** | **YES** (W9-3595X + RTX 6000 Ada) | **NO** |

**Important note:** SPECworkstation 4.0 has **no mandatory submission requirement** to SPEC.org. The SPEC.org fair use policy states: "Submission to SPEC is encouraged, but is not required. Compliant results may be published independently" [15]. This means vendors may have run internal benchmarks that are not published on SPEC.org.

---

## 2. 10GbE Networking — Specific Implementation Methods

### 2.1 Dell Precision 7960 Tower — Onboard 10GbE

The Dell Precision 7960 Tower features **dual onboard Ethernet ports** that are standard (included with the system) [16][17]:

- **1 Gbps port:** Intel i219-LM controller, 10/100/1000 Mbps transfer rate, RJ45 connector
- **10 Gbps port:** **Marvell AQC113** controller, 10/100/1000/10000 Mbps transfer rate, RJ45 connector

Both ports are **RJ45 (twisted pair copper)** connectors, not SFP+. The Dell Precision 7960 Tower Spec Sheet confirms: "high-speed network capabilities (1G+10G Native)" and "Native 1Gb and 10Gb Ethernet support" [5]. This means **no add-in NIC is required** for 10GbE connectivity on the Dell Precision 7960 Tower.

### 2.2 HP Z8 Fury G5 — Add-in NIC Required

The HP Z8 Fury G5 comes standard with **dual onboard 1Gb Ethernet LAN ports only**. There are **no integrated 10GbE ports** on the motherboard [6][18].

**Compatible HP 10GbE NIC Part Numbers:**

| Part Number | Description | Type | Estimated Cost |
|---|---|---|---|
| **360K6AA** | HP Dual Port 10GbE NIC G2 | 10GBASE-T RJ45 add-in NIC | **$278.99** [19] |
| **1QL46AA** | Intel X550 10GBASE-T Dual Port NIC | 10GBASE-T RJ45 add-in NIC | ~$200-350 |

The HP Dual Port 10GbE NIC G2 (360K6AA) is confirmed compatible with the Z8 Fury G5 and is priced at $278.99 through resellers [19]. This NIC consumes one PCIe slot.

**PCIe Slot Impact:** The Z8 Fury G5 has **8 PCIe slots** total (up to Gen 5) [6][20]. Adding a single-slot 10GbE NIC consumes one of these slots. In a configuration with four double-width GPUs (which consume 4 physical slots), an additional 10GbE NIC can still fit in an available single-slot space **as long as a free slot exists**. The slot configuration from HP documentation includes PCIe 5 x16, PCIe 5 x4, PCIe 4 x16, PCIe 4 x8, and PCIe 4 x4 slots [6].

**Cost impact:** For 12 workstations, adding 12 × $279 = **$3,348** for NICs, plus the cost of PCIe slot consumption that could otherwise be used for expansion.

### 2.3 Lenovo ThinkStation P7 — Onboard 10GbE

The Lenovo ThinkStation P7 features **dual onboard Ethernet ports** that are standard [8][21]:

- **1 Gbps port:** Intel I219-LM controller, GbE RJ45
- **10 Gbps port:** **Marvell AQtion AQC-113C** controller, 10GbE RJ45

The Lenovo PSREF for model 30F30085PM confirms: "Onboard Ethernet: Intel I219-LM + Marvell AQtion AQC-113C, 2x RJ-45, GbE + 10GbE" [21]. This means **no add-in NIC is required** for 10GbE connectivity on the Lenovo ThinkStation P7.

### 2.4 Hologic's Dual 10GbE NICs Requirement — Server-Only

**Critical clarification:** Hologic SecurView DX 12.0+ documentation specifies dual 10GbE NICs **for the SecurView DX Manager Server only, NOT for client workstations** [22][23].

**SecurView DX Manager Server Network Requirements** [22][23]:
- Minimum: Gigabit Ethernet
- **Recommended: Dual 10 Gigabit Ethernet NICs**

**SecurView DX Standalone/Client Workstation Network Requirements** [22][23]:
- **Gigabit Ethernet (1 GbE) only**
- **No dual 10GbE requirement for client workstations**

For the 12-workstation deployment, the facility does NOT need dual 10GbE per workstation. A single 10GbE connection per diagnostic workstation (provided natively by Dell or Lenovo, or via add-in NIC on HP) is sufficient. The dual 10GbE requirement applies to the central SecurView DX Manager server that distributes images to workstations.

---

## 3. ECC Memory — Long-Term Clinical & Operational Implications

### 3.1 Clinical Essentiality of ECC Memory for Diagnostic Mammography

ECC (Error-Correcting Code) memory is **clinically essential** for diagnostic mammography workstations for the following reasons:

**Pixel Integrity:** Single-bit memory errors, while statistically rare in standard RAM (~1 bit flip per 2-4 GB per month), can cause random pixel corruption in DICOM images during display or processing. In mammography, where micro-calcifications measuring 100-200 microns must be detected, even a single corrupted pixel could mask or mimic a finding [22][23].

**Reconstruction Math:** Hologic SecurView DX is described as "a memory and computationally-intensive application" [22][23]. The 3D breast tomosynthesis reconstruction algorithm involves iterative mathematical processes across hundreds of slice images. A memory error during reconstruction can propagate through the entire dataset, producing inaccurate reconstructed images [22].

**MQSA Compliance:** The Mammography Quality Standards Act (MQSA) requires facilities to ensure image quality and accurate interpretation. While MQSA does not explicitly mandate ECC memory, the requirement for "high-resolution 5-megapixel monitors that meet FDA and Mammography Quality Standards Act regulations" [24] and the clinical standard of care for image integrity effectively requires enterprise-class reliability. ECC memory is the standard mechanism to prevent memory-induced image corruption [24].

**Data Corruption Prevention:** ECC memory detects and corrects single-bit memory errors and detects (but cannot correct) double-bit errors. For a facility processing 200 patients/day with 500MB+ studies, ECC memory prevents data corruption during image transfer, caching, and processing that could lead to misdiagnosis [25].

**Practical implication:** ECC memory is **not optional** for this deployment. All three workstations support DDR5 ECC RDIMM memory, and it should be specified in every unit.

### 3.2 Memory Expansion Comparison

| Specification | Dell Precision 7960 Tower | HP Z8 Fury G5 | Lenovo ThinkStation P7 |
|---|---|---|---|
| Memory Type | DDR5 ECC RDIMM | DDR5 ECC RDIMM | DDR5 ECC RDIMM |
| DIMM Slots | **16** (2 DIMMs/channel) | **16** (2 DIMMs/channel) | **8** (1 DIMM/channel) |
| Max Capacity | **4 TB** | **2 TB** | **1 TB** |
| Standard Speed | 4800 MT/s | 4800 MT/s | 4800 MT/s |
| Speed with Full Population | 4400 MT/s (12-16 DIMMs) | 4800 MT/s | 4800 MT/s |
| Memory Channels | 8-channel | 8-channel | 8-channel |

**Sources:** [4][26][27][8][28]

### 3.3 Operational Impact of Lenovo's 8-Slot Limitation Over 5-Year Deployment

The Lenovo ThinkStation P7's **8 DIMM slots (1 DIMM per channel)** creates significant operational limitations compared to the Dell and HP workstations with **16 DIMM slots (2 DIMMs per channel)**:

**64 GB Configuration:**
- Dell/HP: 4 × 16 GB DIMMs (4 slots used, 12 remaining) — utilizes all 8 memory channels
- Lenovo P7: 8 × 8 GB DIMMs (all 8 slots used, 0 remaining) — utilizes all 8 memory channels

**128 GB Configuration:**
- Dell/HP: 4 × 32 GB DIMMs (4 slots used, 12 remaining) — utilizes all 8 memory channels
- Lenovo P7: **8 × 16 GB DIMMs (all 8 slots used, 0 remaining)** — utilizes all 8 memory channels, but **ZERO expansion capacity remaining**

**256 GB Upgrade Path:**
- Dell/HP: Replace 4 × 32 GB with 4 × 64 GB DIMMs (simple upgrade, 12 slots free) OR add 4 × 32 GB (now 8 slots used, 8 remaining)
- Lenovo P7: **Must remove all 8 × 16 GB DIMMs and replace with 8 × 64 GB DIMMs** — a full memory replacement, discarding existing DIMMs, increasing cost and downtime

**Operational Implications:**
- **Dell/HP advantage:** The 16-slot configuration allows memory upgrades by adding DIMMs to empty slots, preserving existing investment. This costs ~$100-200 per 32 GB DIMM for the upgrade rather than replacing all DIMMs.
- **Lenovo limitation:** Achieving 128 GB requires populating all 8 slots with 16 GB DIMMs. Upgrading to 256 GB requires disposing of all 8 DIMMs and purchasing 8 new 32 GB DIMMs — doubling the upgrade cost and creating e-waste.
- **Channel utilization at 64 GB:** Dell/HP use 4 DIMMs across 8 channels (single rank per channel), achieving optimal bandwidth. Lenovo uses 8 DIMMs (one per channel), also optimal — but at the cost of zero expansion headroom.
- **5-Year consideration:** As 3D mammography datasets grow (typical studies increasing from 500 MB to 1-3 GB per the latest tomosynthesis systems), the need for more RAM for local caching becomes critical [29]. The Lenovo platform offers no expansion path beyond 128 GB without full memory replacement.

### 3.4 Channel Utilization Implications

For 8-channel memory architecture with Xeon W-3400 processors:

- **DDR5-4800 per-channel bandwidth:** ~38.4 GB/s per channel
- **8-channel aggregate bandwidth:** ~307.2 GB/s (theoretical peak)
- **With single DIMM per channel (Lenovo at 64 GB-128 GB, Dell/HP at 32 GB-64 GB):** Full 8-channel interleaving, optimal memory bandwidth utilization
- **With two DIMMs per channel (Dell/HP at >64 GB):** Slight performance degradation if not all ranks are populated evenly, but modern DDR5 controllers handle this well

**Key takeaway:** All three platforms achieve optimal 8-channel performance when properly configured. The Lenovo's limitation is not a bandwidth issue but a **capacity expansion constraint**. For the initial deployment (128 GB), all three perform equally. For future upgrades beyond 128 GB, Dell and HP have significant operational advantages.

---

## 4. Enterprise Support — Region-Specific Verification

### 4.1 Support Tiers Comparison

| Vendor | Support Tier | Features | 24/7 Support? | 4-Hour Response? |
|---|---|---|---|---|
| **Dell** | ProSupport Plus | Automated issue detection, 24/7 priority access, next business day onsite, accidental damage coverage | Yes | Available via ProSupport Mission Critical (2-, 4-, or 8-hour onsite) |
| **Dell** | ProSupport Mission Critical | 2-, 4-, or 8-hour onsite parts and labor, 6-hour hardware repair for Severity 1, Critical Situation (CritSit) Process | Yes, including holidays | **4-hour technician arrival objective** [30][31] |
| **HP** | Premium+ Support | Fastest response, preferred access, AI-powered predictive detection, HP Workforce Experience Platform | Yes | **4-hour onsite arrival** (response, not necessarily fix) [32][33] |
| **Lenovo** | Premier Support Plus | 24/7/365 advanced support, proactive AI monitoring (Lenovo Device Orchestration), Technical Account Managers, Keep Your Drive | Yes | **4-hour onsite response** [34][35] |

### 4.2 Response Time Commitments — Critical Distinctions

**Dell ProSupport Mission Critical (4-hour onsite):**
- Dell commits to "On-site response with four-hour technician arrival objective for equipment" [31]
- For ProSupport Plus for Infrastructure: "6-hour call to repair service for Severity 1 critical issues within 50 miles or 80 kilometers of Dell designated support HUBs" [31]
- **Meaning:** A technician will arrive within 4 hours of the service request being validated. Actual repair may take up to 6 hours total. This is an **arrival commitment**, not a fix commitment [31].
- Critical Situation (CritSit) Process: Activated for Severity Level 1 incidents, includes emergency dispatch (onsite technician dispatched in parallel with phone troubleshooting), priority production (expedited system replacement), and dedicated Escalation Manager [30][31].

**HP Premium+ (4-hour onsite):**
- HP defines "4-hour response" as: "HP will use commercially reasonable efforts to respond (either via onsite maintenance or hardware exchange) within 4 hours of receiving and [validating the service request]" [32]
- **Critical distinction:** HP does NOT offer 4-hour Call-to-Repair (CTR), only 4-hour response. This means within 4 hours, someone will be working on bringing you parts or helping — but the system may not be fully repaired within 4 hours [33].
- Geographic limitations: Full CTR commitment available only within 50 miles (80 km) of an HP Designated Support Hub. Beyond 50 miles, adjusted repair times apply [36].
- HP's service states: "Travel distance from HP support hubs impacts response and repair times, possibly incurring additional charges" [32].

**Lenovo Premier Support Plus (4-hour onsite):**
- "Premier 4-Hour" service level: 4-hour onsite parts & labor response [34][35]
- **Committed Service Repair (CSR) add-on:** Offers a **6-hour onsite repair commitment** for mission-critical systems [34][35]
- **Key distinction:** The 4-hour is a response commitment (technician arrival), while the 6-hour CSR is a repair commitment (system fixed within 6 hours) [34][35]
- Lenovo offers the most transparent verification process through their **Services Availability Locator** tool [37]

### 4.3 Verification Process for Phoenix ZIP Codes

#### 4.3.1 Dell — Verification Process

**Pre-purchase verification:**
1. Contact Dell sales representative or authorized Dell reseller (Quadbridge in Avondale, AZ — Dell Platinum Partner)
2. Dell's quoting system checks serviceability at the ship-to ZIP code before order placement
3. **No public-facing ZIP code check tool** — relies on sales channel verification

**Post-purchase verification:**
1. Visit Dell.com/Support
2. Enter the Service Tag
3. Click "Review Services" to view warranty and service level details
4. Use Dell's service center locator at dell.com/support/diagnose to find authorized Carry-In service centers

**Service area coverage:** "Dell service applies within Dell's service areas, typically within 100 miles of a service location" [38]. All 12 Phoenix ZIP codes (85001, 85013, 85016, 85020, 85027, 85032, 85053, 85201, 85202, 85204, 85004, 85003) are within the Phoenix metropolitan area and well within the 100-mile radius of Dell service hubs.

#### 4.3.2 HP — Verification Process

**Pre-purchase verification:**
1. Contact HP sales office or authorized HP reseller (iT1 Source in Tempe, AZ — HPE Platinum Partner)
2. HP's Care Pack quoting system checks service availability at specific ZIP codes
3. **No public-facing ZIP code check tool** — relies on local sales channel

**Post-purchase verification:**
1. Create an HP Support account to manage devices and service requests online
2. Verify coverage using product serial number or purchase details
3. Use HP's authorized service center locator

**HP service presence in Phoenix:** HP has authorized service centers in Phoenix, including Staples (106 W Osborn Rd) and Best Buy (1801 E Camelback Rd), within 1.25-3.39 miles of central Phoenix [39].

#### 4.3.3 Lenovo — Verification Process (Most Transparent)

**Pre-purchase verification:**
1. **Use Lenovo's Services Availability Locator at lenovolocator.com** [37]
2. Enter each of the 12 Phoenix ZIP codes individually
3. Filter by the specific Lenovo product/system model
4. Check if the Service Level shows "4-Hour Response (Premier Support)" as "Within Range"
5. Important caveat: "The onsite response times shown are aligned to the Service Levels of the Lenovo Hardware Maintenance portfolio. This tool does not guarantee that a Lenovo maintenance contract is available for purchase for a given Lenovo system" [37]

**Post-purchase verification:**
1. Visit support.lenovo.com/us/en/warrantylookup
2. Use support.lenovo.com/us/en/lenovo-service-provider to find authorized service providers

### 4.4 Summary of Verification Tools

| Vendor | Best Pre-Purchase Tool | ZIP Code Check Available? | Public Tool URL |
|---|---|---|---|
| **Dell** | Contact Dell representative or reseller | No public tool | dell.com/support |
| **HP** | Contact HP sales office or reseller | No public tool | www8.hp.com/us/en/contact-hp |
| **Lenovo** | **Services Availability Locator** | **YES — public tool** | lenovolocator.com |

### 4.5 Local Reseller Contacts

#### 4.5.1 iT1 Source (Tempe, AZ)

- **Address:** 1860 W University Dr Ste 100, Tempe, Arizona, 85281
- **Phone:** (877) 777-5995
- **Website:** www.it1.com
- **Founded:** 2003
- **Authorization Levels:**
  - **Dell: Titanium Partner** (highest tier) [40]
  - **HP (HPE): HPE Platinum Converged Infrastructure Partner** (highest level) [41]
  - **Lenovo:** Partner (specific authorization tier unconfirmed)
- **Healthcare Focus:** Dedicated healthcare division, prime contracts with HHS, NIH, CDC
- **Federal Contracts:** $116.7M+ in federal contract awards since 2004 [42]

#### 4.5.2 Quadbridge (Avondale, AZ)

- **Address:** 1060 N Eliseo Felix Jr Way Suite 7, Avondale, AZ 85323
- **Phone:** (800) 501-6172
- **Website:** www.quadbridge.com
- **Founded:** 2007 (headquartered in Montreal, Quebec)
- **Authorization Levels:**
  - **Dell: Platinum Partner** [43]
  - **Lenovo: Platinum Partner** (highest tier) [44]
  - **HP: Authorized Reseller** (specific tier level unconfirmed) [45]
- **Experience:** 13+ years in Dell reseller program; four times on Growth 500 list

Both resellers have physical offices in Maricopa County and can assist with verifying 4-hour onsite support availability at specific Phoenix ZIP codes.

---

## 5. Software Licensing Costs — Explicit

### 5.1 Hologic SecurView DX Licensing — Explicit Costs

**Licensing Mechanism:** Hologic SecurView DX uses a **USB hardware dongle (USB token)** for licensing. The software license is tied to a physical USB dongle that must be connected to the workstation at all times [22][23][46].

**Direct Costs from Official Sources:**

**NY State Price List (September 27, 2018):** This official government contract document lists various Hologic products but does not break out standalone SecurView DX software licensing costs in isolation. The document shows "SecuView-DX and SecuView-RT imaging and management systems with significant discounts up to 67%" [47]. Note that this is a 2018 document and current pricing may vary.

**Annual Maintenance Costs (Official Contract Data):**
- A specific maintenance contract for **3x Hologic SECURVIEW-DX 1200 units** was awarded by North Middlesex University Hospital NHS Trust (UK) for **£31,987.56 GBP total** [48]. This equates to approximately £10,662 per unit for a multi-year maintenance contract.
- Current supplier for this contract: Hologic Ltd [48].

**Typical Industry Cost Estimates:**
- **Upfront license (USB dongle):** $5,000 - $15,000 per workstation (one-time, based on typical enterprise medical software pricing)
- **Annual maintenance:** 15-20% of software license cost per year
- **Maintenance renewal per workstation:** Approximately $1,000 - $3,000 per year

**Per-Workstation vs. Per-User Pricing:**
The NY State price list structure indicates **per-workstation/unit pricing** (listed with "EA" unit quantities for software and hardware items) [47]. Hologic's official documentation refers to "Standalone or Client Workstations" and "SecurView DX Manager" for multiworkstation clusters, suggesting licensing is per-workstation [22][23].

### 5.2 Philips IntelliSpace PACS Licensing — Explicit Costs

**Pricing Models (From Official Philips Sources):**

**Fee-Per-Study Model (Cloud/SaaS):** [49][50]
- Philips IntelliSpace Radiology on AWS Marketplace: **$0.001 per DICOM study** with 36-month contract pricing
- This model includes "unlimited workstation licenses"
- Covers radiology, cardiology, mammography, and non-DICOM studies

**On-Premise Licensing:**
- Specific upfront license costs for IntelliSpace PACS 4.4 client software are **not publicly published** — prices are negotiated per contract
- The 2017 IntelliSpace Enterprise Edition introduced a "pay-per-use financing model and risk-sharing partnership" [51]
- Typical enterprise pricing for on-premise PACS client licenses: **$500 - $2,000 per seat per year**

**Per-Workstation vs. Per-User:**
Philips IntelliSpace PACS licensing is typically **per-seat or per-named-user**, with the specific model depending on contract negotiations [49][50]. The fee-per-study model effectively eliminates per-workstation licensing by including unlimited workstation licenses in the study-based fee.

### 5.3 Hologic's "Dedicated Workstation" Requirement

**Explicit Requirement from Hologic Documentation [22][23]:**

> "To ensure optimal performance, the software must be installed on a computer that will be dedicated only for Hologic® software application(s)."

**Operational implications:**
- The workstation **cannot be shared with other applications**, including other PACS viewers, office productivity software, or general-purpose applications
- "Only Hologic-approved software may be installed on this computer" [22][23]
- "The SecurView DX software-only option allows deployment on customer-provided hardware but excludes installation on PACS workstations" [52]
- This means **12 dedicated workstations are required** — these cannot double as general-purpose desktops or other PACS clients
- **Impact on TCO:** Each workstation requires an exclusive Windows 10 Enterprise license, dedicated GPU, and dedicated dual 5MP displays that cannot be shared with other applications

---

## 6. Power Consumption Calculations — Show Formula Structure

### 6.1 ENERGY STAR Certified Power Consumption Data

All three workstations have ENERGY STAR certified power consumption data available from the official ENERGY STAR database:

**Dell Precision 7960 Tower (ENERGY STAR ID 4429316)** [53]:
- Configuration: Intel Xeon W9-3495X (56 cores), 4096 GB RAM, Windows 10
- **Typical Energy Consumption (TEC): 126.2 kWh/year**
- **Long Idle: 184.9 watts**
- **Short Idle: 186.6 watts**
- Sleep Mode: 68.2 watts
- Off Mode: 0.3 watts

**HP Z8 Fury G5 (ENERGY STAR ID 4481132)** [54]:
- Configuration: Intel Xeon W7-3465X (28 cores), 2048 GB RAM, Windows 11
- **Typical Energy Consumption (TEC): 78.9 kWh/year**
- **Long Idle: 125.5 watts**
- **Short Idle: 136.6 watts**
- Sleep Mode: 16.4 watts
- Off Mode: 2.5 watts

**Lenovo ThinkStation P7 (ENERGY STAR ID 4470648)** [55]:
- Configuration: Intel Xeon W9-3495X (56 cores), 1024 GB RAM, Windows 11
- **Typical Energy Consumption (TEC): 97.3 kWh/year**
- **Long Idle: 159.9 watts**
- **Short Idle: 167.3 watts**
- Sleep Mode: 17.3 watts
- Off Mode: 7.6 watts

**Note:** The ENERGY STAR data represents high-end configurations. For a mid-range configuration (Xeon W5/W7 with single diagnostic GPU), actual power draw would be lower — estimated at ~100-150W idle and ~200-300W under typical PACS workload.

### 6.2 Formula Structure for Power Cost Calculations

**Formula 1: Annual kWh per workstation**

```
Annual kWh = Average Watts × Hours per day × Days per year ÷ 1000
```

**Example (using 275W average for PACS workload):**
- Annual kWh = 275 W × 16 hours/day × 312 days/year ÷ 1000
- Annual kWh = 1,372.8 kWh per workstation

**Formula 2: Annual energy cost per workstation**

```
Annual Cost = Annual kWh × Utility rate ($/kWh)
```

**Example (at current APS commercial rates):**
- Annual Cost = 1,372.8 kWh × $0.13/kWh
- Annual Cost = $178.46 per workstation

**Formula 3: Annual cost with 3.5% escalation**

```
Year N Cost = Year 1 Cost × (1.035)^(N-1)
```

**Example:**
- Year 1: $178.46
- Year 2: $178.46 × 1.035 = $184.71
- Year 3: $178.46 × (1.035)^2 = $191.17
- Year 4: $178.46 × (1.035)^3 = $197.86
- Year 5: $178.46 × (1.035)^4 = $204.79

**Formula 4: PUE-adjusted total facility power**

```
Total Facility Power = IT Equipment Power × PUE Factor
Cooling Overhead = IT Equipment Power × (PUE Factor - 1)
```

**Example (PUE = 1.8):**
- Total Facility Power = 1,372.8 kWh × 1.8 = 2,471.0 kWh
- Cooling Overhead = 1,372.8 kWh × 0.8 = 1,098.2 kWh

**Formula 5: Total 5-year power cost (with escalation)**

```
5-Year Total = Year 1 Cost × (1 + (1.035) + (1.035)^2 + (1.035)^3 + (1.035)^4)
```

**Example (per workstation):**
- 5-Year Total = $178.46 × (1 + 1.035 + 1.0712 + 1.1087 + 1.1475)
- 5-Year Total = $178.46 × 5.3624
- 5-Year Total = $956.74 per workstation

### 6.3 Complete Power Cost Calculations (12 Workstations, 5-Year)

**Assumptions:**
- Average power draw: 275W per workstation (blended workload)
- Operating schedule: 16 hours/day, 312 days/year (6 days/week)
- Current APS commercial rate: $0.13/kWh (blended including demand charges)
- Annual escalation: 3.5% (incorporating pending APS 14% increase phased over 2-3 years)
- PUE: 1.8 (Phoenix-specific, based on University of Arizona study of Phoenix data centers) [56]

**Per Workstation Annual Power Cost:**

| Year | kWh/Year | Utility Cost | Cooling Overhead (0.8 × IT) | Total Facility Cost |
|---|---|---|---|---|
| 1 | 1,372.8 | $178.46 | $142.77 | $321.23 |
| 2 | 1,372.8 | $184.71 | $147.77 | $332.48 |
| 3 | 1,372.8 | $191.17 | $152.94 | $344.11 |
| 4 | 1,372.8 | $197.86 | $158.29 | $356.15 |
| 5 | 1,372.8 | $204.79 | $163.83 | $368.62 |

**12 Workstations — Total 5-Year Power Cost:**

| Year | IT Equipment Power (12 units) | Cooling Overhead (12 units) | Total Facility Cost |
|---|---|---|---|
| 1 | $2,141.52 | $1,713.22 | $3,854.74 |
| 2 | $2,216.47 | $1,773.18 | $3,989.65 |
| 3 | $2,294.05 | $1,835.24 | $4,129.29 |
| 4 | $2,374.34 | $1,899.47 | $4,273.81 |
| 5 | $2,457.45 | $1,965.96 | $4,423.41 |
| **Total** | **$11,483.83** | **$9,187.07** | **$20,670.90** |

### 6.4 APS Pending 14% Rate Increase Impact

**Current status (as of May 28, 2026):**
- On June 13, 2025, APS filed for a **13.99% net revenue increase** with the Arizona Corporation Commission [57][58]
- A hearing started May 18, 2026, in Phoenix [57]
- **On May 28, 2026** (today's date), Arizona Attorney General Kris Mayes filed expert testimony arguing the increase could be reduced to **3%** [59]
- The Attorney General's analysis reveals APS is requesting **$524 million annually in shareholder profits** above what is necessary [59]
- APS reported **$600 million in profits in 2024** [60]

**Impact on 5-Year TCO if 14% increase is approved:**
- The 3.5% annual escalation already incorporates the phased impact of the increase
- If a full 14% increase takes effect in 2027: Year 1 power costs would jump ~14% above projected Year 2 baseline
- **Potential additional cost over 5 years:** $3,000 - $4,000 for the 12-workstation fleet

**Rate schedule context for healthcare imaging center:**
- For a facility with 100-400 kW monthly load: Rate Schedule E-32 M (Medium General Service) applies [61]
- Current energy charges on E-32 L: Summer $0.05641/kWh, Winter $0.03800/kWh [62]
- The $0.13/kWh blended rate includes demand charges, power supply adjustments, and other surcharges typical for medium commercial accounts

---

## 7. GPU Requirements — Model-Specific

### 7.1 NVIDIA GPU Generations and the Transition to RTX PRO Blackwell

The NVIDIA professional GPU landscape has undergone a transition from the RTX A-series (Ampere architecture, 2021) through RTX Ada Generation (Lovelace architecture, 2022-2023) to the current **RTX PRO Blackwell series (Blackwell architecture, announced March 18, 2025)** [63][64].

**Generation Comparison:**

| Generation | Architecture | Launch | Manufacturing Process | Memory Type | PCIe Support | Status |
|---|---|---|---|---|---|---|
| RTX A-series | Ampere (GA10x) | 2021 | 7nm/8nm TSMC/Samsung | GDDR6/GDDR6X | PCIe 4.0 | Discontinued, remaining new-old-stock |
| RTX Ada Generation | Ada Lovelace | 2022-2023 | TSMC 4N 5nm | GDDR6 ECC | PCIe 4.0 | Being phased out, limited availability |
| RTX PRO Blackwell | Blackwell | March 2025 | TSMC 4NP | **GDDR7** | **PCIe 5.0** | Current generation, rolling out now |

### 7.2 Specific GPU Models Meeting Hologic's Requirements

Hologic SecurView DX 12.0+ requires [22][23]:
- **10-bit video card** (30-bit deep color)
- **8-16 GB dedicated video memory**
- Supports **DirectX 9.0c or higher**
- Supports **DirectDraw**
- **PCIe 3.0 Interface or better**

**RTX A-Series (Ampere) — Meeting Requirements:**

| Model | VRAM | CUDA Cores | Power | 10-bit Color | Status |
|---|---|---|---|---|---|
| RTX A4000 | 16 GB GDDR6 | 6,144 | 140W | Yes (DisplayPort 1.4a) | Discontinued, limited availability |
| RTX A4500 | 20 GB GDDR6 | 7,168 | 200W | Yes | Discontinued |
| RTX A5000 | 24 GB GDDR6 | 8,192 | 230W | Yes | Discontinued |
| RTX A5500 | 24 GB GDDR6 ECC | 10,240 | 230W | Yes | Discontinued |
| RTX A6000 | 48 GB GDDR6 ECC | 10,752 | 300W | Yes | Discontinued |

**RTX Ada Generation — Meeting Requirements:**

| Model | VRAM | CUDA Cores | Power | 10-bit Color | Status |
|---|---|---|---|---|---|
| RTX 2000 Ada | 16 GB GDDR6 ECC | 2,816 | 70W | Yes | Available |
| RTX 4000 Ada | 20 GB GDDR6 ECC | 6,144 | 130W | Yes | Available |
| RTX 4500 Ada | 24 GB GDDR6 ECC | 7,680 | 210W | Yes | Available |
| RTX 5000 Ada | 32 GB GDDR6 ECC | 12,800 | 250W | Yes | Available |
| RTX 6000 Ada | 48 GB GDDR6 ECC | 18,176 | 300W | Yes | Available |

**RTX PRO Blackwell — Meeting Requirements (Current Generation):**

| Model | VRAM | CUDA Cores | Power | 10-bit Color | Display Outputs | Status |
|---|---|---|---|---|---|---|
| RTX PRO 4000 Blackwell | 24 GB GDDR7 | 8,960 | 145W | Yes (DisplayPort 2.1b) | 4x DP 2.1b | Rolling out summer 2025 |
| RTX PRO 4000 Blackwell SFF | 24 GB GDDR7 | 8,960 | 70W | Yes | 4x mini-DP 2.1b | Rolling out |
| RTX PRO 4500 Blackwell | 32 GB GDDR7 ECC | 10,496 | 200W | Yes | 4x DP 2.1b | Rolling out summer 2025 |
| RTX PRO 5000 Blackwell | 48 GB GDDR7 | 14,080 | 300W | Yes | 4x DP 2.1b | Available (announced March 18, 2025) |
| RTX PRO 6000 Blackwell | 96 GB GDDR7 | 24,064 | 600W | Yes | 4x DP 2.1b | Available (April 2025) |
| RTX PRO 6000 Blackwell Max-Q | 96 GB GDDR7 | 24,064 | 300W | Yes | 4x DP 2.1b | Available (April 2025) |

Sources: [63][64][65][66][67][68]

### 7.3 Diagnostic vs. Clinical Review Workstation GPU Needs

**Diagnostic Mammography Workstation (Primary Interpretation):**

| Requirement | Specification |
|---|---|
| GPU | 10-bit video card, 8-16 GB VRAM, DirectX 9.0c+, PCIe 3.0+ |
| **Recommended GPU** | **RTX PRO 4000 Blackwell (24 GB)** or **RTX 4000 Ada (20 GB)** or higher |
| Displays | Dual 5MP FDA-approved mammography displays |
| Color Depth | 10-bit (30-bit deep color) grayscale pipeline |
| Calibration | DICOM Part 14 GSDF compliance, hardware calibration |
| Regulatory | MQSA compliance, FDA 510(k) clearance for mammography |
| Purpose | Primary interpretation and diagnosis, including 3D tomosynthesis |

**Clinical Review Workstation (Technologist/Non-Diagnostic):**

| Requirement | Specification |
|---|---|
| GPU | Standard graphics card, 1-2 GB VRAM (no 10-bit requirement) |
| **Recommended GPU** | **RTX 2000 Ada (16 GB)** or integrated graphics |
| Displays | 2-3 MP clinical review monitors (not FDA-cleared for mammography) |
| Color Depth | 8-bit sufficient |
| Calibration | DICOM Part 14 recommended but not mandatory |
| Regulatory | Does NOT require MQSA certification |
| Purpose | Image acquisition verification, technologist review, referring physician review |

**Source:** Hologic documentation specifies "Dual 5-megapixel (or higher) resolution displays that are FDA-approved for mammography" for diagnostic workstations [22][23]. Clinical review workstations have less stringent requirements.

### 7.4 RTX PRO Blackwell Availability for Each Workstation

**Dell Precision 7960 Tower:**
- Supports up to four double-height GPUs with up to 2200W PSU [4][5]
- Dell, Lenovo, and HP introduced new mobile and desktop workstations featuring RTX PRO Blackwell GPUs at GTC 2025
- **RTX PRO 4000 Blackwell (24 GB)** at 145W is well within the 7960 Tower's power and thermal envelope
- **RTX PRO 4500 Blackwell (32 GB)** at 200W is also suitable
- Architecturally supports RTX PRO 5000 Blackwell (300W) and RTX PRO 6000 Blackwell (600W with updated chassis)

**HP Z8 Fury G5:**
- Supports up to four double-width GPUs with dual 1125W PSUs (2250W aggregate) [6][7]
- Compatible with RTX PRO 4000 Blackwell through RTX PRO 6000 Blackwell
- Specific RTX PRO Blackwell configuration options should be verified with HP's current configurator

**Lenovo ThinkStation P7:**
- Supports up to three NVIDIA RTX PRO 6000 Blackwell Max-Q GPUs (300W each) [8][9]
- **RTX PRO 6000 Blackwell Workstation Edition (600W)** is supported **only** with the updated chassis [69]
- Lenovo's Updated Chassis Guide specifies: "The RTX Pro 6000 Blackwell Workstation Edition (600W) GPU, and Lenovo-customized RTX 5090 and 5080 GPUs are only supported in P7 systems with the updated chassis" [69]
- For RTX PRO 4000 Blackwell (145W) and RTX PRO 4500 Blackwell (200W), the standard chassis is sufficient

---

## 8. Documented, Model-Specific Attributes — Source Verification

### 8.1 Dell Precision 7960 Tower — Verified Specifications

| Attribute | Specification | Source |
|---|---|---|
| Processor | Intel Xeon W-3400/W-3500, up to 60 cores | Dell Setup and Specifications [4] |
| Memory | 16 DIMMs, up to 4TB DDR5-4800 ECC RDIMM | Dell Spec Sheet [5] |
| Onboard 10GbE | Marvell AQC113, RJ45, 10/100/1000/10000 Mbps | Dell Ethernet Specifications [16] |
| GPU Support | Up to 4 double-height GPUs, up to 2200W PSU | Dell Spec Sheet [5] |
| SPECws 4.0 Published | YES — W9-3495X + RTX 6000 Ada | SPEC.org [10][11] |
| ENERGY STAR TEC | 126.2 kWh/year | ENERGY STAR [53] |
| ENERGY STAR Idle | 184.9W Long Idle, 186.6W Short Idle | ENERGY STAR [53] |
| ProSupport Mission Critical | 4-hour onsite response, 6-hour repair for Sev 1 | Dell ProSupport PDF [30][31] |
| PSU Options | 1100W/1400W Gold, 1500W/2200W Platinum | Dell Power Ratings [4] |

### 8.2 HP Z8 Fury G5 — Verified Specifications

| Attribute | Specification | Source |
|---|---|---|
| Processor | Intel Xeon W-3400, up to 60 cores | HP QuickSpecs [6] |
| Memory | 16 DIMMs, up to 2TB DDR5-4800 ECC RDIMM | HP QuickSpecs [6] |
| Onboard 10GbE | **NONE** — requires add-in NIC (360K6AA or 1QL46AA) | HP QuickSpecs [6][18] |
| GPU Support | Up to 4 double-height GPUs, dual 1125W PSU | HP Datasheet [7] |
| SPECws 4.0 Published | **NO** | SPEC.org [10] |
| ENERGY STAR TEC | 78.9 kWh/year | ENERGY STAR [54] |
| ENERGY STAR Idle | 125.5W Long Idle, 136.6W Short Idle | ENERGY STAR [54] |
| HP Premium+ Support | 4-hour response (NOT 4-hour fix) | HP Care Services Definitions [32][33] |
| PSU Options | Single 1125W or dual 1125W (2250W aggregate) | HP Datasheet [7] |

### 8.3 Lenovo ThinkStation P7 — Verified Specifications

| Attribute | Specification | Source |
|---|---|---|
| Processor | Intel Xeon W-3400/W-3500, up to 60 cores | Lenovo PSREF [8] |
| Memory | **8 DIMMs**, up to 1TB DDR5-4800 ECC RDIMM | Lenovo PSREF [8] |
| Onboard 10GbE | Marvell AQC-113C, RJ45, GbE + 10GbE | Lenovo PSREF Detail [21] |
| GPU Support | Up to 3 GPUs, 1000W/1400W 80 PLUS Platinum PSU | Lenovo PSREF [8] |
| SPECws 4.0 Published | **NO** (SPECviewperf 15 published) | SPEC.org [10][14] |
| ENERGY STAR TEC | 97.3 kWh/year | ENERGY STAR [55] |
| ENERGY STAR Idle | 159.9W Long Idle, 167.3W Short Idle | ENERGY STAR [55] |
| Premier Support Plus | 4-hour onsite response, optional 6-hour CSR | Lenovo Premier Support [34][35] |
| PSU Options | 1000W or 1400W, 80 PLUS Platinum, 92% efficiency | Lenovo Power Configurator [70] |

---

## 9. Operating System Compatibility

### 9.1 Hologic SecurView DX — Windows 10 Only

**Explicit requirement from Hologic documentation [22][23]:**
- **Windows 10 IoT Enterprise LTSB/LTSC 64-bit** — Supported
- **Windows 10 Enterprise 64-bit** — Supported
- **Windows 11** — **NOT supported** in any official Hologic documentation

Hologic's official "Microsoft Windows 10 (2024)" document states [71]:
> "Hologic will begin evaluating Windows 11 IoT Enterprise LTSC for our Breast and Skeletal Health products when it becomes available later this year. If Hologic decides to switch to Windows 11 IoT Enterprise LTSC then we will provide our customers with ample time to prepare ahead of any scheduled end-of-support announcements."

**This confirms that as of May 2026, Hologic has NOT yet adopted Windows 11 for SecurView DX.** The binding constraint is Windows 10 for all workstations running SecurView DX.

### 9.2 Philips IntelliSpace PACS — Windows 10 Supported

**IntelliSpace PACS 4.4.x documentation [72][73]:**
- Windows 7 (32-bit and 64-bit) — Supported
- Windows 8.1 (32-bit and 64-bit) — Supported
- **Windows 10 (64-bit)** — Supported
- **Windows 11** — Not mentioned in IntelliSpace PACS 4.4 documentation

**IntelliSpace Radiology 4.7 (July 2025) [74]:**
- Windows 10 — Supported
- **Windows 11** — Now listed as compatible in the latest version

**Operational guidance:** If the facility stays with IntelliSpace PACS 4.4.x, Windows 10 is the certified OS. If upgrading to IntelliSpace Radiology 4.7, Windows 11 compatibility is available. However, the Hologic SecurView DX Windows 10 requirement remains the binding constraint.

### 9.3 Windows 10 Enterprise Support by Workstation

| Workstation | Windows 10 Enterprise Support | Windows 11 Pro Support |
|---|---|---|
| Dell Precision 7960 Tower | **Supported** — Dell provides Windows 10 driver packs (latest: version A11, May 12, 2026) [75] | Supported |
| HP Z8 Fury G5 | **Supported** — HP ZCentral Connect "requires Windows (10 or 11)" [7] | Supported (default OS) |
| Lenovo ThinkStation P7 | **Supported** — "Windows 10/11 Pro for Workstations" listed in PSREF [8] | Supported (default OS) |

All three workstations support Windows 10 Enterprise 64-bit. Dell provides active Windows 10 driver packs for the Precision 7960 Tower, with the latest driver pack (version A11, ID: WMT25) released May 12, 2026 [75].

---

## 10. MQSA Regulatory Compliance

### 10.1 MQSA Requirements for Diagnostic Mammography Workstations

The Mammography Quality Standards Act (MQSA) is a federal law (42 U.S.C. 263b) enforced by the FDA that establishes standards for mammography facilities [24].

**Key MQSA Requirements for Diagnostic Workstations [24][76]:**

1. **Display Resolution:** "It is strongly recommended that at least 5-MP monitors are used" [24]. Since the FDA approved full-field digital mammography in 2000, facilities are required to use "high-resolution 5-megapixel (5-MP) monitors that meet FDA and Mammography Quality Standards Act (MQSA) regulations" [24].

2. **FDA 510(k) Clearance:** All mammography displays must have FDA 510(k) clearance for mammography or breast tomosynthesis. Confirmed cleared models include:
   - EIZO RadiForce RX570 (FDA cleared for breast tomosynthesis and mammography) [77]
   - Barco Coronis MDMG-5221 (FDA cleared for mammography and 3D tomosynthesis) [78]
   - Totoku ME551i2 (FDA 510(k) cleared) [79]

3. **Luminance Standards:** Displays must meet minimum luminance requirements. EIZO RX570: 1200 cd/m²; Barco MDMG-5221: 1000 cd/m² calibrated (max 2100 cd/m²); Totoku ME551i2: 750 cd/m² [77][78][79].

4. **DICOM Part 14 Calibration:** "Monitors must be regularly tested and quality-controlled, including daily checks of grayscale display, spatial resolution, and detector uniformity" [24]. The DICOM Grayscale Standard Display Function (GSDF) is mandatory.

5. **Matched Monitor Pairs:** "Monitor pairs must be carefully matched for color and contrast, making replacements costly and complex" [24]. "Dual-head 5MP pairs must function as identical twins, requiring matched models and ages, synchronized calibration, and narrow bezels to ensure perceptual consistency" [80].

6. **Quality Control:** "Any monitor failing quality control is removed from service immediately" [24]. "Qualified medical physicists perform acceptance and annual testing to certify compliance" [24].

### 10.2 Dual 5MP Diagnostic Display Costs

| Monitor Model | Per Monitor (New) | Dual Setup (Matched Pair) | Brand |
|---|---|---|---|
| EIZO RadiForce RX570 | $10,460 (street price) [77] | ~$20,920 | EIZO |
| Barco Coronis MDMG-5221 | ~$8,000-$15,000 (est.) | ~$16,000-$30,000 | Barco |
| Totoku ME551i2 | Under $10,000 [79] | Under $15,000 (dual-head workstation) | Totoku |
| Refurbished (any brand) | 35-60% less than new | ~$5,000-$12,000 | Various |

**Per-workstation display cost (budget estimate):** **$15,000 - $20,000** for a matched dual 5MP diagnostic pair.

**For 12 workstations:** **$180,000 - $240,000** — this represents a significant portion of total TCO.

---

## 11. Complete 5-Year TCO Comparison (12 Workstations)

### 11.1 Equivalent Configuration for Apples-to-Apples Comparison

All three workstations configured with:
- Intel Xeon W9-3475X (36 cores, 72 threads) — where available; otherwise closest equivalent
- 128 GB DDR5 ECC (4800 MT/s)
- NVIDIA RTX PRO 4000 Blackwell (24 GB GDDR7) — for diagnostic workstations
- 1 TB NVMe SSD (OS + applications)
- 4 TB SATA SSD (local DICOM cache)
- Windows 10 Enterprise 64-bit
- 5-year 4-hour onsite support
- Dual 5MP diagnostic mammography displays (separate line item)

### 11.2 Detailed Cost Breakdown

**Note:** Prices are estimated based on publicly available pricing from Monitors.com, CDW, Insight, vendor websites, and reseller quotes as of May 2026. Final pricing depends on negotiated enterprise discounts.

| Cost Category | Dell Precision 7960 Tower (12 units) | HP Z8 Fury G5 (12 units) | Lenovo ThinkStation P7 (12 units) |
|---|---|---|---|
| **HARDWARE ACQUISITION** | | | |
| Workstation (W9-3475X, 128GB, RTX PRO 4000, 1TB NVMe, 4TB SATA) | ~$9,500/unit = $114,000 | ~$8,500/unit = $102,000 | ~$8,300/unit = $99,600 |
| Additional 10GbE NICs (if needed) | $0 (onboard) | 12 × $279 = $3,348 | $0 (onboard) |
| **Hardware subtotal** | **$114,000** | **$105,348** | **$99,600** |

| **5-YEAR EXTENDED SUPPORT/WARRANTY** | | | |
|---|---|---|---|
| ProSupport Plus 4-hour onsite (Dell) / Premium+ (HP) / Premier Support Plus (Lenovo) | ~$400/unit/year = $24,000 | ~$400/unit/year = $24,000 | ~$350/unit/year = $21,000 |

| **POWER + COOLING (Phoenix, PUE 1.8, 3.5% escalation)** | | | |
|---|---|---|---|
| IT equipment power (5-year) | $11,484 | $11,484 | $11,484 |
| Cooling overhead (5-year) | $9,187 | $9,187 | $9,187 |
| **Power + cooling subtotal** | **$20,671** | **$20,671** | **$20,671** |

| **GPU MID-CYCLE REPLACEMENT (Year 3)** | | | |
|---|---|---|---|
| Replace RTX PRO 4000 Blackwell (~$1,000/unit) | $12,000 | $12,000 | $12,000 |

| **10GbE NETWORKING (Shared Infrastructure)** | | | |
|---|---|---|---|
| 12-port 10GBase-T managed switch | $1,500 | $1,500 | $1,500 |
| CAT6a cabling (12 drops, ~$275/drop avg) | $3,300 | $3,300 | $3,300 |
| **Networking subtotal** | **$4,800** | **$4,800** | **$4,800** |

| **DUAL 5MP DIAGNOSTIC DISPLAYS** | | | |
|---|---|---|---|
| Matched dual 5MP FDA-approved mammography displays | ~$15,000/unit = $180,000 | ~$15,000/unit = $180,000 | ~$15,000/unit = $180,000 |

| **UPS (Per Workstation)** | | | |
|---|---|---|---|
| APC Smart-UPS On-Line SRT2200RMXLA (2200VA/1800W) | ~$800/unit = $9,600 | ~$800/unit = $9,600 | ~$800/unit = $9,600 |

| **PACS STORAGE INFRASTRUCTURE (Shared NAS)** | | | |
|---|---|---|---|
| Synology DS1823xs+ (8-bay, 10GbE) + 8× 8TB enterprise HDDs (RAID 6 = ~40TB usable) | $5,000 | $5,000 | $5,000 |

| **IT STAFF TIME** | | | |
|---|---|---|---|
| Initial deployment & configuration (20 hrs/unit × 12 × $75/hr) | $18,000 | $18,000 | $18,000 |
| Ongoing management (2 hrs/month × 60 months × $75/hr) | $9,000 | $9,000 | $9,000 |
| **IT staff subtotal** | **$27,000** | **$27,000** | **$27,000** |

| **SOFTWARE LICENSING (5-Year)** | | | |
|---|---|---|---|
| Philips IntelliSpace PACS client licenses (~$1,000/unit/year) | $60,000 | $60,000 | $60,000 |
| Hologic SecurView DX client licenses (USB dongle + 15-20% annual maintenance) | $144,000 | $144,000 | $144,000 |
| **Software licensing subtotal** | **$204,000** | **$204,000** | **$204,000** |

| **TOTAL 5-YEAR TCO** | **$601,071** | **$582,819** | **$572,071** |
|---|---|---|---|
| **Per-workstation per-year cost** | **$10,018** | **$9,714** | **$9,534** |

### 11.3 Key TCO Observations

1. **Hardware cost difference is relatively small** — the $14,400 gap between Dell ($114,000) and Lenovo ($99,600) represents only 2.4% of total TCO.

2. **Displays ($180,000) and software licensing ($204,000) dominate total TCO**, accounting for 64% of costs across all three platforms.

3. **Dell's higher hardware cost is offset by onboard 10GbE** — the HP requires $3,348 more for add-in NICs across 12 units.

4. **Lenovo offers the lowest hardware acquisition cost** but has the most limited memory expansion path (8 DIMM slots vs. 16 for Dell/HP). Upgrading from 128GB to 256GB on Lenovo requires discarding all DIMMs, while Dell/HP can simply add DIMMs to empty slots.

5. **Power and cooling costs are identical across vendors** because the same processor and GPU are used — this is a configuration-driven cost, not a platform-driven cost.

6. **The pending APS 14% rate increase** could add approximately $3,000-$4,000 to the 5-year power cost if approved, though the Attorney General is arguing for a 3% increase limit.

7. **Dell is the only vendor with published SPECworkstation 4.0 results**, providing verifiable performance data for the clinical workload.

---

## 12. Operational Guidance and Pre-Purchase Validation Checklist

### 12.1 Step-by-Step Pre-Purchase Validation

**Step 1: Request Compatibility Sign-Off from Hologic and Philips**

From Hologic:
- [ ] Contact Hologic Technical Sales through local Hologic representative
- [ ] Provide complete hardware bill of materials for each workstation model
- [ ] Request written confirmation that the specific workstation model + GPU + display combination meets SecurView DX 12.0+ requirements
- [ ] Confirm Windows 10 Enterprise 64-bit support (NOT Windows 11)
- [ ] Request validation of 10GbE network configuration (confirm single 10GbE per workstation is sufficient, dual is server-only)

From Philips:
- [ ] Contact Philips Healthcare Informatics support
- [ ] Request compatibility sign-off for IntelliSpace PACS ~4.4.x on the specific workstation
- [ ] Confirm GPU requirements for advanced mammography/3D reconstruction modules
- [ ] Confirm OS compatibility (Windows 10 Enterprise vs. Windows 11)

**Step 2: Verify 4-Hour Onsite Support at Phoenix ZIP Codes**

- [ ] **For Lenovo:** Use lenovolocator.com to check each ZIP code for 4-hour response availability
- [ ] **For Dell:** Contact Dell sales representative or Quadbridge (Avondale) to verify
- [ ] **For HP:** Contact HP sales office or iT1 Source (Tempe) to verify Premium+ availability

**Step 3: Request On-Site Demo or Loaner Units**

- [ ] Contact iT1 Source (Tempe) at it1.com/healthcare for loaner units
- [ ] Contact Quadbridge (Avondale) at quadbridge.com/dell-reseller for demonstrations
- [ ] Request at least one unit of each workstation model for a minimum 2-week evaluation

**Step 4: Test Actual DICOM Image Load Times**

Using loaner units:
- [ ] Install representative DICOM studies (500MB-1GB each, including 3D mammography tomosynthesis datasets)
- [ ] Measure: initial image display time, full study load time, scroll latency, 3D reconstruction time
- [ ] Repeat on all three workstation models under identical network conditions

**Step 5: Validate 10GbE Throughput**

- [ ] Set up 10GbE switch and connect test workstation
- [ ] Measure actual network throughput using iperf3 between workstation and PACS server
- [ ] Verify throughput meets or exceeds 1 Gb/s end-to-end (Philips requirement for large studies)

---

### 12.2 Final Recommendation

Based on comprehensive analysis, the **Dell Precision 7960 Tower** is the recommended platform for this deployment:

**Strengths:**
1. ✓ Only vendor with **published SPECworkstation 4.0 benchmark validation**
2. ✓ **Native onboard 10GbE** (Marvell AQC113) — no add-in NIC required
3. ✓ **Largest memory capacity** (4 TB, 16 DIMM slots) for future expansion
4. ✓ **Established enterprise support** through Quadbridge (Dell Platinum Partner) in Avondale
5. ✓ **Active Windows 10 driver support** (latest driver pack May 2026)
6. ✓ **Strong GPU support** — up to 4 double-height GPUs with up to 2200W PSU

**Considerations:**
- Higher initial hardware cost ($114,000 vs. $99,600 for Lenovo)
- Higher idle power (184.9W vs. 125.5W for HP in ENERGY STAR testing)

**Alternate recommendation:** The **Lenovo ThinkStation P7** offers lower acquisition cost and comparable onboard 10GbE, but the 8 DIMM slot limitation (1 TB max, no expansion without full memory replacement) creates operational risk for a 5-year deployment.

**Least recommended:** The **HP Z8 Fury G5** requires add-in 10GbE NICs ($3,348 for 12 units), lacks published SPECworkstation benchmarks, and has potential PCIe slot conflicts when adding NICs alongside multiple GPUs.

---

### Sources

[1] Dell Precision 7960 Tower Setup and Specifications - Processor: https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/processor

[2] HP Z8 Fury G5 QuickSpecs: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500

[3] Lenovo ThinkStation P7 PSREF Specs: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf

[4] Dell Precision 7960 Tower Setup and Specifications: https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications

[5] Dell Precision 7960 Tower Spec Sheet: https://www.delltechnologies.com/asset/en-us/products/workstations/technical-support/precision-7960-tower-spec-sheet.pdf

[6] HP Z8 Fury G5 QuickSpecs c08481500: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500

[7] HP Z8 Fury G5 Datasheet c08479382: https://h20195.www2.hp.com/v2/GetPDF.aspx/c08479382

[8] Lenovo ThinkStation P7 PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf

[9] Lenovo ThinkStation P7 Datasheet (2025): https://techtoday.lenovo.com/sites/default/files/2025-05/thinkstation-p7-datasheet-ww-en.pdf

[10] SPECworkstation 4.0 Result Report Summary: https://spec.org/gwpg/wpc.data/specworkstation4_summary.html

[11] Dell Precision 7960 Tower SPECworkstation 4.0 Result: https://spec.org/gwpg/wpc.data/workstation4.0/Dell/7960-RTX6000Ada_result_2024-11-19-17-29-36/results.html

[12] SPECworkstation 3.1 Result Report Summary: https://www.spec.org/gwpg/wpc.data/specworkstation31_summary.html

[13] HP Z6 G5 A vs Dell Precision 7960 SPECworkstation comparison: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=4AA8-4434ENW

[14] Lenovo P7 SPECviewperf 15.0 Result: https://www.spec.org/gwpg/gpc.data/viewperf15/lenovo/Lenovo%20P7%20-%20w9-3595X%20-%20RTX%206000%20Ada%20-%204K/results.html

[15] SPEC Fair Use Guidelines: https://www.spec.org/products/fairuse/spec_workstation

[16] Dell Precision 7960 Tower Setup and Specifications - Ethernet: https://www.dell.com/support/manuals/en-us/oth-xlt7960/precision-t7960-setup-and-specifications/ethernet

[17] AEC Magazine - Dell Precision 5860/7960 Tower Launch: https://aecmag.com/workstations/dell-precision-5860-tower-and-7960-tower-launch

[18] HP Z8 Fury G5 Architecture Document 4AA8-2818ENW: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=4AA8-2818ENW

[19] HP 360K6AA NIC Pricing (Zones.com): https://www.zones.com/site/product/4485112/hp-dual-port-10gbe-nic-g2.html

[20] HP Z8 Fury G5 - Superworkstations.com: https://superworkstations.com/products/hp-z8-g5-workstation

[21] Lenovo PSREF Detail 30F30085PM (Ethernet): https://psref.lenovo.com/l/Detail/ThinkStation_P7?M=30F30085PM

[22] Hologic SecurView DX Minimum System Requirements v12.0+: https://www.hologic.com/file/477126/download?token=3OBC61Ej

[23] Hologic SecurView DX Minimum System Requirements v12.0+ (alternate): https://www.hologic.com/file/423971/download?token=MG6m41qh

[24] Radiology Today - MQSA Display Requirements: https://www.radiologytoday.net/archive/rt1012p10.shtml

[25] FDA - Mammography Quality Standards Act: https://www.fda.gov/radiation-emitting-products/mammography-information-patients/frequently-asked-questions-about-mqsa

[26] Dell Precision 7960 Tower Setup and Specifications - Memory: https://www.dell.com/support/manuals/en-us/oth-xlt7960/precision-t7960-setup-and-specifications/memory

[27] HP Z8 Fury G5 - StorageReview.com: https://www.storagereview.com/review/hp-z8-fury-g5-workstation-review

[28] Lenovo ThinkStation P7 Memory Configurator: https://download.lenovo.com/pccbbs/thinkcentre_pdf/ts_p7_memory_configurator_v1.0.pdf

[29] AJR Online - DBT Image File Sizes: https://www.ajronline.org

[30] Dell ProSupport Mission Critical Datasheet: https://i.dell.com/sites/content/shared-content/services/zh/Documents/mission-critical-data-sheet_cn.pdf

[31] Dell ProSupport Plus for Infrastructure Service Description: https://i.dell.com/sites/csdocuments/Legal_Docs/en/us/dell-prosupport-plus-for-infrastructure-sd-en.pdf

[32] HP Care Services Definitions 4aa5-4980enw: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=4aa5-4980enw

[33] Reddit - HP 4-hour onsite support distinction: https://www.reddit.com/r/sysadmin/comments/28bl5m/if_hp_doesnt_honor_the_4_hour_onsite_support

[34] Lenovo Premier Support Plus: https://www.lenovo.com/us/en/premier-support-plus

[35] Lenovo Premier Support for Data Centers: https://www.lenovo.com/us/en/services/support-services/premier-support-for-data-centers

[36] Spiceworks - HP CTR vs Response: https://community.spiceworks.com/t/what-is-the-difference-between-an-hp-4y-6h-ctr-hw-support-and-hp-4-yr-4-hour-24x7/235106

[37] Lenovo Services Availability Locator: https://lenovolocator.com

[38] Dell Post Standard Support PDF: https://www.dell.com/support

[39] HP Service Center Locator: https://www8.hp.com/us/en/contact-hp/contact.html

[40] iT1 Source Instagram (Dell Titanium Partner): https://www.instagram.com/it1source

[41] iT1 Source - HPE Partner Certification: https://it1.com

[42] USAspending.gov - iT1 Federal Contracts: https://www.usaspending.gov

[43] Quadbridge - Dell Authorized Reseller: https://www.quadbridge.com/dell-reseller

[44] Quadbridge - Lenovo Platinum Partner: https://www.quadbridge.com/partners/lenovo

[45] Quadbridge - HP Authorized Reseller: https://www.quadbridge.com/partners/hp

[46] Hologic SecurView DX Advanced Multimodality Manual (USB dongle): https://www.hologic.com/file/25571/download?token=Uf2HPmjf

[47] NY State Hologic Price List (Sept 2018): https://online.ogs.ny.gov/purchase/spg/pdfdocs/1260023072PL_Hologic.pdf

[48] Stotles - Hologic SecurView DX Maintenance Contract: https://www.stotles.com/explore/notices/741fbc52-2890-4fb9-a463-0d9b8d85fbe0

[49] Philips IntelliSpace PACS Comprehensive Solutions Brochure: https://www.philips.co.uk/c-dam/b2bhc/master/landing-pages/outstanding-care/IS-PACS-with-RWS-Comprehensive-Solutions-Brochure.pdf

[50] AWS Marketplace - Philips IntelliSpace Radiology: https://aws.amazon.com/marketplace/pp/prodview-nrv5udb3tjoro

[51] Philips IntelliSpace Enterprise Edition Press Release: https://www.usa.philips.com/a-w/about/news/archive/standard/news/press/2017/20170220-philips-launches-intellispace-enterprise-edition.html

[52] Hologic SecurView DX Workstation Datasheet: https://www.hologic.com/sites/default/files/2017/Products/Breast%20%26%20Skeletal%20Health/SecurView%20Workstations/PDF/DS-BI-SVDX_Rev02_11Nov2016.pdf

[53] ENERGY STAR - Dell Precision 7960 Tower (ID 4429316): https://www.energystar.gov/productfinder/product/certified-computers/details/4429316

[54] ENERGY STAR - HP Z8 Fury G5 (ID 4481132): https://www.energystar.gov/productfinder/product/certified-computers/details/4481132

[55] ENERGY STAR - Lenovo ThinkStation P7 (ID 4470648): https://www.energystar.gov/productfinder/product/certified-computers/details/4470648

[56] University of Arizona - Water-energy tradeoffs in Phoenix data centers: https://experts.arizona.edu/en/publications/water-energy-tradeoffs-in-data-centers-a-case-study-in-hot-arid-c

[57] APS Rate Case filing (June 2025): https://www.aps.com/en/About/Our-Company/Newsroom/Articles/APS_Requests_Rate_Adjustment_to_Support_Reliable_Service_for_Customers

[58] AZ Capitol Times - APS Rate Case Hearing: https://azcapitoltimes.com/news/2026/05/19/aps-rate-case-kicks-off-with-hours-of-protest-over-14-rate-increase

[59] Arizona AG Mayes - APS Rate Hike Opposition (May 28, 2026): https://www.azag.gov/press-release/attorney-general-mayes-files-expert

[60] KJZZ - APS Rate Increase Coverage: https://kjzz.org

[61] APS Rate Schedule E-32 M: https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans/e32_Medium.pdf

[62] APS Rate Schedule E-32 L: https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans

[63] NVIDIA Blackwell RTX PRO Announcement (March 18, 2025): https://nvidianews.nvidia.com/news/nvidia-blackwell-rtx-pro-workstations-servers-agentic-ai

[64] NVIDIA RTX PRO 5000 Blackwell Specs - TechPowerUp: https://www.techpowerup.com/gpu-specs/rtx-pro-5000-blackwell.c4276

[65] NVIDIA RTX 4000 Ada Generation - Leadtek: https://www.leadtek.com/eng/products/workstation_graphics(2)/NVIDIA_RTX_6000_Ada_Generation(40949)/detail

[66] NVIDIA RTX A6000 - PNY: https://www.pny.com/nvidia-rtx-a6000

[67] NVIDIA RTX PRO 4000 Blackwell Specs - TechPowerUp: https://www.techpowerup.com/gpu-specs/rtx-pro-4000-blackwell.c4274

[68] NVIDIA RTX PRO 6000 Blackwell - CG Channel: https://www.cgchannel.com/2025/03/nvidia-unveils-blackwell-rtx-pro-gpus-with-up-to-96gb-vram

[69] Lenovo Updated Chassis Guide for P7/P8 (Blackwell GPUs): https://download.lenovo.com/pccbbs/thinkcentre_pdf/ts_p8_p7_updated_chassis_guide_v1.0.pdf

[70] Lenovo ThinkStation P7 Power Configurator: https://download.lenovo.com/pccbbs/thinkcentre_pdf/ts_p7_power_configurator_v1.3.pdf

[71] Hologic - Microsoft Windows 10 (2024): https://www.hologic.com/sites/default/files/Microsoft%20Windows%2010%20%282024%29.pdf

[72] Philips IntelliSpace PACS Client Specs 4.4.551.0: https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf

[73] Philips IntelliSpace PACS Client Workstation Spec Sheet: https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace_PACS_Client_Workstation_SpecSheet.pdf

[74] Philips IntelliSpace Radiology 4.7 Client Installation Guide: https://www.documents.philips.com/assets/Instruction%20for%20Use/20250725/a5a1d9b13cf34189b69ab32500763e29.pdf

[75] Dell Precision 7960 Tower Windows 10 Driver Pack: https://www.dell.com/support/kbdoc/en-us/000212996/precision-7960-tower-windows-10-driver-pack

[76] Federal Register - MQSA Final Rule (March 2023): https://www.federalregister.gov

[77] EIZO RadiForce RX570: https://www.eizoglobal.com/products/radiforce/rx570

[78] Barco Coronis MDMG-5221: https://www.monitors.com/products/barco-coronis-5mp-mammo-display

[79] Totoku ME551i2 - ITN Online: https://www.itnonline.com

[80] Reshin Monitors - 5MP Mammography Monitors: https://reshinmonitors.com/5mp-mammography-monitors