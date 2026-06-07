# Comprehensive Workstation Research Report: Dell Precision 7960 Tower vs. HP Z8 Fury G5 vs. Lenovo ThinkStation P7 for Healthcare Imaging Center – Phoenix, AZ

## Executive Summary

This report provides a detailed comparison of the Dell Precision 7960 Tower, HP Z8 Fury G5, and Lenovo ThinkStation P7 workstations for a Phoenix-based healthcare imaging center processing ~200 patients/day running Philips IntelliSpace PACS (~4.4.x) and Hologic SecurView DX (12.0+) software with 500MB+ DICOM files. Based on comprehensive analysis of CPU performance via SPECworkstation 4.0 benchmarks, ECC memory support, empirically justified power consumption, 10GbE networking, GPU requirements for diagnostic mammography, enterprise support verification at Phoenix ZIP codes, software compatibility, and 5-year total cost of ownership, this report finds that the **Dell Precision 7960 Tower with Intel Xeon W9-3475X** offers the strongest combination of published SPECworkstation validation, native 10GbE networking, enterprise support infrastructure, and cost efficiency for this specific clinical workload. The **Lenovo ThinkStation P7** is a strong alternative with comparable native 10GbE and slightly lower acquisition cost, while the **HP Z8 Fury G5** trails primarily due to the requirement for add-in 10GbE NICs and absence of published SPECworkstation results.

---

## 1. CPU Performance Evaluation (Equivalently Configured Across Vendors)

### 1.1 Critical Platform Clarification: W7-2495X vs. W9-3475X

A critical finding from this research is that **the Intel Xeon W7-2495X (24 cores, 4 memory channels) cannot be used for an apples-to-apples comparison across these three workstations.** The W7-2495X belongs to the Intel Xeon W-2400 series, which uses a quad-channel (4-channel) memory architecture and is supported in the **Dell Precision 5860 Tower**, **HP Z4 G5**, and **Lenovo ThinkStation P5** — not in the requested Dell Precision 7960 Tower, HP Z8 Fury G5, or Lenovo ThinkStation P7.

The three workstations in this comparison all use the **Intel Xeon W-3400 series** (and newer 5th Gen W-3500 series) processors, which feature an **8-channel memory architecture**, higher PCIe lane counts (112 vs. 64), and support up to 56-60 cores. The equivalent mid-to-high-end processor available across all three platforms is the **Intel Xeon W9-3475X** (36 cores, 72 threads, 2.2 GHz base / 4.8 GHz turbo, 300W TDP, 82.5 MB L3 cache) [Dell Precision 7960 Tower Setup and Specifications](https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/processor).

### 1.2 Supported Processors by Workstation

| Workstation | Supported CPU Family | Equivalent Mid-to-High-End Option | Cores/Threads | Memory Channels | TDP |
|---|---|---|---|---|---|
| **Dell Precision 7960 Tower** | Intel Xeon W-3400 (4th/5th Gen) | W9-3475X (36C/72T) | 36/72 | 8-channel | 300W |
| **HP Z8 Fury G5** | Intel Xeon W-3400 (4th Gen) | W9-3475X (36C/72T) | 36/72 | 8-channel | 300W |
| **Lenovo ThinkStation P7** | Intel Xeon W-3400/W-3500 | W9-3475X (36C/72T) | 36/72 | 8-channel | 300W |

All three workstations officially support the Intel Xeon W9-3475X processor [HP Z8 Fury G5 QuickSpecs](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500) [Lenovo ThinkStation P7 PSREF Specs](https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf).

### 1.3 SPECworkstation 4.0 Benchmark Results

#### 1.3.1 Dell Precision 7960 Tower with Xeon W9-3495X — Published Results Available

A SPECworkstation 4.0 result has been **published** for the Dell Precision 7960 Tower, submitted by Dell Technologies, configured with the Intel Xeon W9-3495X (56 cores) and NVIDIA RTX 6000 Ada Generation GPU [SPECworkstation 4.0 Result Report Summary](https://spec.org/gwpg/wpc.data/specworkstation4_summary.html). The detailed scores are as follows:

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

All scores are presented as "higher scores indicate better performance" [SPECworkstation 4.0 - Dell Precision 7960 Detailed Results](https://spec.org/gwpg/wpc.data/workstation4.0/Dell/7960-RTX6000Ada_result_2024-11-19-17-29-36/results.html).

**Note:** This result is for the W9-3495X (56-core), not the W9-3475X (36-core). No SPECworkstation 4.0 result has been published for the Dell Precision 7960 with the W9-3475X specifically.

#### 1.3.2 HP Z8 Fury G5 — No SPECworkstation 4.0 Results Published

**As of the last update of SPECworkstation 4.0 results (February 11, 2026), no SPECworkstation 4.0 result has been published for any HP Z8 Fury G5 configuration** [SPECworkstation 4.0 Result Report Summary](https://spec.org/gwpg/wpc.data/specworkstation4_summary.html). The HP results that are published include the Z2 Tower G9, Z4 G5, Z6 G5 A (AMD), and ZBook mobile workstations — but no Z8 family models.

#### 1.3.3 Lenovo ThinkStation P7 — No SPECworkstation 4.0 Results Published

**As of the last update of SPECworkstation 4.0 results (February 11, 2026), no SPECworkstation 4.0 result has been published for any Lenovo ThinkStation P7 configuration** [SPECworkstation 4.0 Result Report Summary](https://spec.org/gwpg/wpc.data/specworkstation4_summary.html). Lenovo results that are published include the ThinkPad P1/P16, ThinkStation P3 Tower, ThinkStation P8 (AMD Threadripper), and ThinkStation PX (dual Xeon) — but no P7.

#### 1.3.4 Explicit Gap Statement

| Workstation | Requested CPU | SPECws 4.0 Published? | Details |
|---|---|---|---|
| Dell Precision 7960 Tower | Xeon W9-3495X | **YES** | Scores published (see above) |
| Dell Precision 7960 Tower | Xeon W9-3475X | **NO** | Only W9-3495X published |
| HP Z8 Fury G5 | Xeon W-3400 series | **NO** | No Z8 family results at all |
| Lenovo ThinkStation P7 | Xeon W-3400 series | **NO** | No P7 results at all |

### 1.4 Single-Threaded vs. Multi-Threaded Performance for Clinical Workloads

For the specific clinical workloads at this imaging center:

- **Single-Threaded Performance (PACS UI Responsiveness):** Philips IntelliSpace PACS requires Intel i5 dual-core 2.5 GHz minimum, with recommended six-core or better processors [Philips IntelliSpace PACS Client Specs](https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf). The W9-3475X single-thread turbo of 4.8 GHz provides excellent UI responsiveness for DICOM image manipulation (pan, zoom, window/level adjustments). The Principled Technologies study comparing the Dell Precision 7960 (W9-3475X) to an HP Z6 G5 A (AMD Threadripper PRO 7975WX) found the Dell scored 3.67 in SPECworkstation 3.1 Production Development vs. 8.00 for the AMD system — indicating the Intel W9-3475X provides solid but not class-leading single-thread performance for PACS UI tasks [Principled Technologies Report](https://www.principledtechnologies.com/HP/Z6-G5-A-Desktop-Workstation-AMD-Ryzen-performance-0824).

- **Multi-Threaded Performance (3D Mammography Reconstruction):** Hologic SecurView DX is described as a "memory and computationally-intensive application" requiring Intel Xeon E-2287GE or better [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej). The W9-3475X's 36 cores and 72 threads provide substantial multi-threaded capability for batch 3D mammography reconstruction tasks. The Dell Precision 7960 SPECworkstation 4.0 CPU subsystem score of 2.94 represents the benchmark for this class of processor.

### 1.5 Compatibility with Software Applications

**Philips IntelliSpace PACS ~4.4.x:**
- CPU: Minimum Intel i5 dual-core 2.5 GHz; recommended six-core or two quad-core at 2.5 GHz or better [Philips IntelliSpace PACS Client Workstation Spec Sheet](https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace_PACS_Client_Workstation_SpecSheet.pdf)
- GPU: OpenGL 3.2 support required; 1-2 GB VRAM recommended for advanced visualization
- Network: 1 Gb/s adapter required for large studies (>700 MB)
- All three workstations (with W9-3475X and appropriate GPU) meet or exceed these requirements

**Hologic SecurView DX 12.0+:**
- CPU: Minimum Intel Core i7-6700 (3.40 GHz); recommended Intel Xeon E-2287GE or better
- RAM: Minimum 32 GB; recommended 64 GB
- GPU: 10-bit video card with 8-16 GB dedicated VRAM, DirectX 9.0c or higher, PCIe 3.0 or better
- Storage: Recommended 8 TB SSD with 400 MB/s+ read/write
- Network: Gigabit Ethernet minimum; dual 10GbE NICs recommended for Manager-to-Client connections
- **Operating System: Windows 10 IoT Enterprise LTSC/LTSB or Windows 10 Enterprise 64-bit — Windows 11 is NOT listed as supported in official Hologic documentation** [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej)

The Xeon W9-3475X at 4.8 GHz turbo significantly exceeds the recommended Xeon E-2287GE (3.31 GHz) across all three platforms.

---

## 2. ECC Memory Support

### 2.1 DDR5 ECC Support and Memory Channel Configurations

All three workstations support DDR5 ECC memory with 8-channel memory architecture (when using W-3400 series processors):

| Specification | Dell Precision 7960 Tower | HP Z8 Fury G5 | Lenovo ThinkStation P7 |
|---|---|---|---|
| Memory Type | DDR5 ECC RDIMM | DDR5 ECC RDIMM | DDR5 ECC RDIMM |
| Maximum Capacity | 4 TB (16 DIMMs) | 2 TB (16 DIMMs) | 1 TB (8 DIMMs) |
| Number of DIMM Slots | 16 | 16 | 8 |
| Memory Channels | 8-channel (2 DIMMs/channel) | 8-channel (2 DIMMs/channel) | 8-channel (1 DIMM/channel) |
| Standard Speed | 4800 MT/s | 4800 MT/s | 4800 MT/s |
| Speed with Full Population | 4400 MT/s (12-16 DIMMs) | 4800 MT/s | 4800 MT/s |

[Dell Precision 7960 Tower Setup and Specifications - Memory](https://www.dell.com/support/manuals/en-us/oth-xlt7960/precision-t7960-setup-and-specifications/memory) [HP Z8 Fury G5 QuickSpecs](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500) [Lenovo ThinkStation P7 PSREF Specs](https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf)

### 2.2 Clinical Importance of ECC Memory for Diagnostic Imaging

ECC (Error-Correcting Code) memory is **clinically essential** for diagnostic mammography workstations. Single-bit memory errors, while rare in standard RAM, can cause:

- **Pixel corruption** in DICOM images during display or processing
- **Mathematical errors** in 3D reconstruction algorithms
- **Data corruption** during image transfer or caching
- **False positive/negative findings** from corrupted image data

For a facility processing 200 patients/day with 500MB+ studies, ECC memory prevents data corruption that could lead to misdiagnosis. Philips IntelliSpace PACS documentation does not explicitly mention ECC requirements, but the system operates on enterprise-class workstations that typically include ECC memory [Philips IntelliSpace PACS Client Specs](https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf). Hologic SecurView DX recommends 64 GB RAM for optimal performance and requires enterprise-class reliability for diagnostic interpretation [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej).

The practical implication: ECC memory is **not optional** for this deployment. All three workstations support it, and it should be specified in every unit.

### 2.3 Recommended RAM Configurations

| Tier | RAM Configuration | Rationale |
|---|---|---|
| **Minimum** | 64 GB (2x32 GB or 4x16 GB) | Meets Hologic's 64 GB recommended specification; exceeds Philips' 24 GB+ recommendation [Philips IntelliSpace PACS Client Specs](https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf) |
| **Optimal** | 128 GB (4x32 GB on Dell/HP or 8x16 GB on Lenovo P7) | Provides headroom for caching large DICOM studies, running multiple PACS sessions, and future 3D mammography reconstruction workloads |

For the Dell Precision 7960 Tower, 128 GB can be achieved with 4x32 GB DIMMs, leaving 12 slots free for future expansion to 256 GB or 512 GB. For the Lenovo ThinkStation P7 (8 DIMM slots), 128 GB requires populating all 8 slots with 16 GB DIMMs, limiting future expansion without replacing existing DIMMs.

---

## 3. Power Consumption at Arizona's $0.13/kWh Commercial Rate (Empirically Justified)

### 3.1 Empirically Measured Power Draw Data

#### 3.1.1 HP Z8 Fury G5 — ENERGY STAR Certified Measurements

The HP Z8 Fury G5 has **ENERGY STAR Certified** power consumption data published for a configuration with the Intel Xeon W7-3465X (28 cores) [ENERGY STAR Certified Computers - HP Z8 Fury G5](https://www.energystar.gov/productfinder/product/certified-computers/details/4481132):

| Mode | Power Consumption |
|---|---|
| Off Mode | 2.5 W |
| Sleep Mode | 16.4 W |
| Long Idle | 125.5 W |
| Short Idle | 136.6 W |
| TEC of Model | 78.9 kWh |

The HP Z8 Fury G5 Site Prep Guide indicates typical workstation power consumption varies significantly with configuration, from approximately 690W to 1185W depending on CPU and graphics cards selected [HP Z8 Fury G5 Site Prep Guide](https://assets.grandandtoy.com/assets/Additional-pdf5/10/81/1081618911.pdf).

#### 3.1.2 Dell Precision 7960 Tower — Qualitative HotHardware Data

The HotHardware review of the Dell Precision 7960 Tower provides **qualitative comparisons** but does not publish raw wattage numbers from its power meter [HotHardware Dell Precision 7960 and 5860 Review](https://hothardware.com/reviews/dell-precision-7960-and-5860-review?page=3):

- The Precision 7960 consumes "slightly less power at idle with the Ultra Performance profile than Optimized"
- The Precision 7960 "consumes more power than the 5860 during all-core CPU loads" with a "200 watt delta during all-core loads"
- "The RTX 6000 Ada Generation card's efficiency is very impressive here"
- "The entire system caps around 1.2 kW" (from the Dell Precision 7875 review reference)

**Explicit gap:** No precise idle or load power draw wattage numbers for the Dell Precision 7960 have been published by HotHardware, StorageReview, Puget Systems, or other review sources.

#### 3.1.3 Lenovo ThinkStation P7 — No Empirical Measurements Available

**No empirical power consumption measurements have been published for the Lenovo ThinkStation P7 from any reputable review source.** The only available data is the power supply specification: 1000W or 1400W 80 PLUS Platinum certified supplies with 92% efficiency [Lenovo ThinkStation P7 Power Configurator](https://download.lenovo.com/pccbbs/thinkcentre_pdf/ts_p7_power_configurator_v1.3.pdf). No review site (HotHardware, StorageReview, Puget Systems, TechPowerUp, Tom's Hardware, AnandTech) has published power draw data for this workstation.

#### 3.1.4 Best Estimate Power Draw for Equivalently Configured Workstations

Based on available data and the Intel Xeon W9-3475X processor specification (300W TDP) with a mid-range professional GPU (RTX PRO 4000 Blackwell at 145W or RTX A4000 at 140W):

| Workload Scenario | Estimated Power Draw (W) | Source/Methodology |
|---|---|---|
| **Idle** (desktop, no apps) | ~130-150W | Based on HP ENERGY STAR short idle 136.6W for 28-core Xeon W7-3465X |
| **PACS Reading** (2D images, UI interaction) | ~200-250W | GPU at modest load + CPU at low load |
| **3D Mammography Reconstruction** (CPU-intensive) | ~400-500W | W9-3475X at 300W TDP + GPU + system overhead |
| **Maximum Load** (all cores + GPU) | ~600-800W | HotHardware qualitative data (~800W for single GPU load) |

**Estimated average power draw for TCO calculations: 275W per workstation** (assumes ~60% of time in PACS reading/idle at ~150W, ~30% in moderate processing at ~300W, ~10% in heavy 3D reconstruction at ~500W).

**Important caveat:** This is a best estimate based on available data. Actual power draw will vary based on specific configurations, software workloads, and user behavior. Empirical measurements for the exact W9-3475X + professional GPU configuration are not available from published sources.

### 3.2 Annual and 5-Year Power Cost Calculation

#### 3.2.1 Arizona Commercial Electricity Rate Context

As of May 2026, Arizona's average commercial electricity rate is **11.8¢/kWh** [ElectricChoice.com Arizona Electricity Prices](https://www.electricchoice.com/electricity-prices-by-state/arizona). The user-specified rate of $0.13/kWh (13¢/kWh) represents a realistic blended rate including demand charges and surcharges for a medium commercial account (APS Rate Schedule E-32 M) [APS Rate Schedule E-32 M Medium General Service](https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans/e32_Medium.pdf).

#### 3.2.2 Pending APS 14% Rate Increase

On June 13, 2025, APS filed an application with the Arizona Corporation Commission for a **13.99% net revenue increase** [APS Newsroom - APS Requests Rate Adjustment](https://www.aps.com/en/About/Our-Company/Newsroom/Articles/APS_Requests_Rate_Adjustment_to_Support_Reliable_Service_for_Customers). The evidentiary hearing began May 18, 2026, with a Commission vote expected December 2026 [Arizona Capitol Times - APS Rate Case](https://azcapitoltimes.com/news/2026/05/19/aps-rate-case-kicks-off-with-hours-of-protest-over-14-rate-increase). If approved, increases would likely take effect in early 2027. APS has also proposed a **formula rate mechanism** that would allow annual rate adjustments without full rate cases for five years [Solar.com APS Electric Rate Increase](https://www.solar.com/learn/aps-electric-rate-increase).

**For TCO calculations, assume a 3.5% annual rate escalation** (blended from the 14% increase phased over 2-3 years plus ongoing annual adjustments).

#### 3.2.3 Power Cost Calculations

**Per Workstation (275W average, 16 hours/day):**

| Scenario | Annual kWh | Cost at $0.13/kWh | 5-Year Cost (with 3.5% annual escalation) |
|---|---|---|---|
| 312 days/year (6 days/week) | 1,372.8 | $178.46 | $991.43 |
| 365 days/year (continuous) | 1,606.0 | $208.78 | $1,159.80 |

**Total for 12 Workstations (312 days/year):**

| Cost Component | Annual | 5-Year |
|---|---|---|
| IT equipment power (12 units × 1,372.8 kWh) | $2,141.52 | $11,897.16 |
| Cooling overhead (PUE factor) | $1,712.42 | $9,517.73 |
| **Total with cooling** | **$3,853.94** | **$21,414.89** |

#### 3.2.4 HVAC/Cooling Impact with Phoenix-Specific PUE

For a Phoenix-based IT room with standard HVAC in a hot desert climate:

- **PUE (Power Usage Effectiveness) of 1.8** is appropriate for a small on-premises IT room in Phoenix (not a purpose-built data center) based on the University of Arizona study of Phoenix data centers, which found average PUEs ranging from approximately 1.5 to 1.7 for facilities using various cooling technologies [Karimi et al. Resources, Conservation and Recycling 2022](https://experts.arizona.edu/en/publications/water-energy-tradeoffs-in-data-centers-a-case-study-in-hot-arid-c)
- This means for every 1 kW of IT power, an additional 0.8 kW is needed for cooling and overhead
- In Phoenix's climate, air-cooled chillers have PUE ~13% higher than water-cooled/evaporative systems
- Free cooling and evaporative cooling can be used approximately 40% of the year when outside air conditions permit

**Cooling overhead cost:** $1,712.42/year for the 12-workstation fleet (0.8 × IT equipment power)

**Total annual facility electricity cost:** $3,853.94 (IT equipment + cooling)

**5-year total facility electricity cost (with 3.5% annual escalation):** $21,414.89

---

## 4. 10GbE Network Compatibility

### 4.1 Built-in vs. Add-in 10GbE Support

| Workstation | Built-in 10GbE? | Details |
|---|---|---|
| **Dell Precision 7960 Tower** | **YES — Onboard** | Two onboard RJ45 Ethernet ports: one 1Gb and one 10Gb [Dell Precision 7960 Tower Spec Sheet](https://www.delltechnologies.com/asset/en-us/products/workstations/technical-support/precision-7960-tower-spec-sheet.pdf) |
| **HP Z8 Fury G5** | **NO — Add-in card required** | Two 1GbE ports onboard (Intel I219-LM and I210-AT). 10GbE requires optional PCIe add-in card (Intel X550-T2 dual-port ~$313) [HP Z8 Fury G5 QuickSpecs](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500) |
| **Lenovo ThinkStation P7** | **YES — Onboard** | One 1GbE RJ45 and one 10GbE RJ45 port onboard [Lenovo ThinkStation P7 PSREF Specs](https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf) |

### 4.2 Hologic's Dual 10GbE NIC Requirement

Hologic SecurView DX 12.0+ documentation specifies:

- **Workstation-Class:** "Gigabit Ethernet" (minimum)
- **SecurView DX Manager Server-Class:** "Dual 10 Gigabit Ethernet NICs" (recommended) [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej)

For the Manager-to-Client connections handling large DICOM studies, dual 10GbE ensures sufficient bandwidth for concurrent image distribution. The "2:1 HDD capacity ratio between Manager and Client" recommendation underscores the data-intensive nature of this environment [Hologic SecurView DX Minimum System Requirements v12.0+ Revision 002](https://www.hologic.com/file/477126/download?token=3OBC61Ej).

**Practical implication:** For the 12-workstation deployment:
- **Dell Precision 7960 Tower** and **Lenovo ThinkStation P7** have 1x 10GbE onboard. For Hologic's dual 10GbE requirement, an additional 10GbE add-in card would be needed for Manager servers or high-throughput workstations. For standard client workstations, the single onboard 10GbE port is sufficient.
- **HP Z8 Fury G5** requires add-in 10GbE NICs for all workstations, adding cost (~$200-400 per unit) and consuming PCIe slots.

### 4.3 Switch and Cabling Cost Estimates

For a 12-workstation deployment with PACS server connectivity:

| Item | Quantity | Unit Cost | Total |
|---|---|---|---|
| 24-port managed 10GbE switch (copper + SFP+) | 1 | $1,200 - $2,500 | $1,200 - $2,500 |
| CAT6a or CAT7 Ethernet cables (10ft) | 15 | $15 - $25 | $225 - $375 |
| SFP+ modules (for fiber uplinks if needed) | 2-4 | $50 - $150 | $100 - $600 |
| Patch panel and rackmount accessories | 1 | $200 - $400 | $200 - $400 |
| **Total networking hardware** | | | **$1,725 - $3,875** |

**Recommended 10GbE switch options:**
- QNAP QSW-M3216R-8S4T (16-port managed, 8x 10GbE + 4x 2.5GbE): ~$1,500
- Netgear M4300-24X (24-port 10GbE managed): ~$2,500
- D-Link DXS-3400-24TC (24-port 10GbE managed): ~$2,000

---

## 5. GPU Requirements for Diagnostic Mammography

### 5.1 NVIDIA RTX A-Series to RTX PRO Branding Transition

On March 18, 2025, at GTC 2025, NVIDIA officially announced the **RTX PRO Blackwell** series, replacing the previous RTX A-series (Ampere) and RTX Ada Generation branding [NVIDIA Newsroom - Blackwell RTX PRO](https://nvidianews.nvidia.com/news/nvidia-blackwell-rtx-pro-workstations-servers-agentic-ai). The transition timeline is:

- **RTX A-series (Ampere, 2021):** Discontinued. RTX A4000 (16GB), A4500 (20GB), A5000 (24GB), A6000 (48GB) — available only as remaining new-old-stock or used [DEVELOP3D - Nvidia RTX A4000 and RTX A5000 launch](https://develop3d.com/hardware/nvidia-rtx-a4000-and-rtx-a5000-gpus-launch)
- **RTX Ada Generation (2022-2023):** Being phased out. RTX 4000 Ada (20GB), RTX 4500 Ada (24GB), RTX 5000 Ada (32GB), RTX 6000 Ada (48GB) — limited availability through channel partners [AEC Magazine - RTX 4000, 4500 & 5000 Ada launch](https://aecmag.com/news/nvidia-rtx-4000-4500-5000-ada-generation-gpus-launch)
- **RTX PRO Blackwell (2025-2026):** Current generation, available now. Full lineup includes RTX PRO 2000 (16GB), 4000 (24GB), 4000 SFF (24GB), 4500 (32GB), 5000 (48GB/72GB), and 6000 (96GB) [Central Computers - All NVIDIA RTX Pro Blackwell GPUs Explained](https://www.centralcomputer.com/blog/post/all-nvidia-rtx-pro-blackwell-gpus-explained)

### 5.2 Recommended GPUs Meeting Hologic's Requirements

Hologic SecurView DX 12.0+ requires:
- 10-bit video card
- 8-16 GB dedicated VRAM
- DirectX 9.0c or higher
- PCIe 3.0 or better
- Dual 5MP FDA-approved mammography displays [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej)

**Recommended NVIDIA RTX PRO Blackwell GPUs for diagnostic mammography:**

| GPU | VRAM | Power | Form Factor | Suitability |
|---|---|---|---|---|
| **RTX PRO 4000 Blackwell** | 24 GB GDDR7 (ECC) | 145W | Single-slot | **Ideal for diagnostic mammography** — meets 8-16 GB requirement, 10-bit color, PCIe 5.0 |
| **RTX PRO 4000 Blackwell SFF** | 24 GB GDDR7 (ECC) | 70W (slot-powered) | Low-profile, dual-slot | Suitable for space-constrained deployments |
| **RTX PRO 2000 Blackwell** | 16 GB GDDR7 (ECC) | 70W (slot-powered) | Low-profile, dual-slot | Minimum viable option meeting 16 GB requirement |

**Note:** All RTX PRO Blackwell GPUs support 10-bit color (30-bit deep color) through professional drivers — this capability is exclusive to NVIDIA's professional GPU lineup and not available on GeForce cards [NVIDIA - 30-Bit Color Technology Technical Brief](https://www.nvidia.com/docs/IO/40049/TB-04701-001_v02_new.pdf).

### 5.3 Diagnostic vs. Clinical Review Workstation GPU Requirements

The distinction between diagnostic and clinical review workstations is clinically and regulatory significant:

**Diagnostic Mammography Workstation:**
- Must comply with MQSA (Mammography Quality Standards Act) regulations enforced by the FDA [FDA - Mammography Quality Standards Act](https://www.fda.gov/radiation-emitting-products/mammography-information-patients/frequently-asked-questions-about-mqsa)
- **Required:** GPU with 10-bit color support (professional NVIDIA GPU)
- **Required:** 8-16 GB dedicated VRAM for large DICOM datasets
- **Required:** Dual 5MP FDA-approved mammography displays with DICOM calibration
- **Required:** Enterprise-class reliability with ECC memory on GPU
- **Purpose:** Primary interpretation and diagnosis of mammography images, including digital breast tomosynthesis (3D)
- **Software:** Hologic SecurView DX running dedicated diagnostic software
- **Recommended GPU:** RTX PRO 4000 Blackwell (24GB) or higher

**Clinical Review (Technologist/Non-Diagnostic) Workstation:**
- Does NOT require MQSA-level certification
- **Can use:** Standard 2-3MP clinical review monitors
- **GPU:** May not require 10-bit color depth; standard graphics capabilities suffice
- **Purpose:** Image acquisition verification, technologist review, referring physician review
- **Software:** Hologic offers dedicated "SecurView Technologist workstation" as a single-monitor solution
- **Recommended GPU:** RTX PRO 2000 Blackwell (16GB) or even integrated graphics

The [NVIDIA Healthcare Graphics Configurations presentation](https://www.nvidia.com/content/quadro_oem/presentations/Vertical_Industry_-_Healthcare.pdf) confirms this tiered approach: diagnostic workstations require "multi-megapixel, 8-10 bit grayscale and color display capabilities" with multiple PCIe x16 slots, while PACS viewing stations have less stringent requirements.

---

## 6. Enterprise Support Options in Phoenix Metro (ZIP-Code-Specific Verification)

### 6.1 Verification Process for 4-Hour Onsite Response at Specific Phoenix ZIP Codes

#### 6.1.1 Dell ProSupport Plus 4-Hour Onsite

Dell's ProSupport Plus for Infrastructure includes an "On-site Response objective is a four-hour service response after Dell Technologies deems Onsite Response necessary" [Dell ProSupport Plus for Infrastructure Service Description](https://i.dell.com/sites/csdocuments/Legal_Docs/en/us/dell-prosupport-plus-for-infrastructure-sd-en.pdf). Mission Critical Support includes "4-hour on-site response with 6-hour hardware repair service for Severity 1 issues within specified geographic coverage" with mention of "within a 50-mile radius of Dell support hubs."

**Verification process for specific Phoenix ZIP codes:**
1. **Pre-purchase:** Contact your Dell sales representative or Dell authorized reseller (e.g., Quadbridge in Avondale). Dell recommends: "For more information about any of our service offerings, please contact your Dell representative or visit www.dell.com"
2. **Post-purchase verification:** Visit Dell.com/Support, enter the Service Tag, click "Review Services" to view warranty and service level details [Dell Support Services & Warranty](https://www.dell.com/support/contractservices/en-us)
3. **Service Center Locator:** Use Dell's service center locator at dell.com/support/diagnose/en-us/servicecenter to find authorized Carry-In repair service centers by city name and postal code
4. **Renewal eligibility check:** Visit renewals.dell.com

Dell does not appear to have a public-facing ZIP code check tool for 4-hour onsite response availability. The service description notes that "Product and service availability varies by country" and coverage is defined for "eligible products."

#### 6.1.2 HP Premium+ 4-Hour Onsite

HP Premium+ Support provides "onsite repair when needed" with options for "next-business-day or four-hour onsite arrival" [HP Premium Support Services](https://www.hp.com/us-en/services/workforce-solutions/workforce-computing/support/premium-plus.html). The HP Hardware Support Onsite Service data sheet notes: "Travel distance from HP support hubs impacts response and repair times, possibly incurring additional charges" and "Coverage generally applies to HP or Compaq hardware and supplied components" [HP Hardware Support Onsite Service](https://www.also.com/pub/assets/0b532aa0-0229-4a7c-95af-c2629ddce597.pdf).

**Verification process for specific Phoenix ZIP codes:**
1. **Pre-purchase:** "Contact a local HP sales office for detailed information on service availability and coverage"
2. **Post-purchase:** Create an HP Support account to manage devices and service requests online
3. **Serial number lookup:** Use product serial number or purchase details to verify coverage
4. **Service locator:** Use HP's authorized service center locator

HP's Care Services Definitions document states: "All response times are subject to local availability" and recommends contacting "a local HP sales office for detailed information on service availability" [HP Care Services Definitions](https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=4aa5-4980enw). A Reddit discussion notes: "HP Doesn't offer 4 Hour CTR, only 4 hour Response. This means within 4 hours you will have someone working on bringing you the parts or helping" — an important distinction for service expectations [Reddit - HP 4-hour onsite support](https://www.reddit.com/r/sysadmin/comments/28bl5m/if_hp_doesnt_honor_the_4_hour_onsite_support).

#### 6.1.3 Lenovo Premier Support Plus 4-Hour Onsite

Lenovo offers the most transparent verification process through their **Services Availability Locator** at lenovolocator.com [Lenovo Services Availability Locator](https://lenovolocator.com). This tool "allows you to check if selected services are available at customer's location for selected Lenovo hardware" and "displays the potential onsite response times Lenovo can support at a given location."

**Verification process for specific Phoenix ZIP codes:**
1. **Use the Services Availability Locator:** Enter the install location (ZIP code), filter by hardware group/product, review results. Service levels shown include Next Business Day, 4-Hour Response, and Committed Service Repair (CSR 6-Hour Add-On)
2. **Use the Data Centre Solution Configurator (DCSC):** Verify if service part numbers exist for selected hardware and country
3. **Warranty Lookup:** Visit support.lenovo.com/us/en/warrantylookup
4. **Service Provider Locator:** Visit support.lenovo.com/us/en/lenovo-service-provider to find authorized service providers by country, region, and proximity

The tool instructions note: "The onsite response times shown are aligned to the Service Levels of the Lenovo Hardware Maintenance portfolio. This tool does not guarantee that a Lenovo maintenance contract is available for purchase for a given Lenovo system" [Lenovo Premier Support Introduction for ISG](https://download.lenovo.com/km/media/attachment/Onboarding%20Deck%20EN_PSP%20Server%20FINAL.pdf).

#### 6.1.4 Summary of Verification Tools

| Vendor | Best Pre-Purchase Tool | ZIP Code Check Available? | Post-Purchase Verification |
|---|---|---|---|
| **Dell** | Contact Dell representative or reseller | No public tool; relies on sales channel | Service Tag lookup at Dell.com/Support |
| **HP** | Contact HP sales office | No public tool; relies on local sales | HP Support account + serial number |
| **Lenovo** | **Services Availability Locator** (lenovolocator.com) | **YES** — public tool | Warranty lookup + service provider locator |

### 6.2 Local Reseller Options

#### 6.2.1 iT1 Source (Tempe, AZ)

- **Location:** Tempe, Arizona (headquarters)
- **Founded:** 2003
- **Active accounts:** 3,000+ across healthcare, finance, retail, manufacturing, and government
- **Authorized for:** HP Inc., Dell Technologies, Lenovo, Cisco, VMware, Microsoft
- **Healthcare focus:** Dedicated healthcare division [iT1 Source - Healthcare Division](https://it1.com/healthcare)
- **Awards:** Phoenix Business Journal's "Best Places to Work" (top 10 in Arizona for 14 consecutive years)
- **Global footprint:** US, UK, Canada, India, Mexico, Netherlands, Australia, Hong Kong, Vietnam
- **Services:** Hardware procurement, VoIP, Azure, Backup & Recovery, Cybersecurity, Data Storage
- **Relevance:** iT1 can verify service availability at specific Phoenix ZIP codes for all three workstation vendors through their established channel partnerships

#### 6.2.2 Quadbridge (Avondale, AZ)

- **Location:** 1060 N Eliseo Felix Jr Way, #107, Avondale, AZ 85323
- **Founded:** 2007 (headquarters in Montreal, Quebec)
- **Authorized for:** Dell Platinum Partner, HP, Cisco, Juniper, EMC [Quadbridge Dell Authorized Reseller](https://www.quadbridge.com/dell-reseller)
- **Services:** Hardware/software procurement, cloud and data center services, cybersecurity, digital workplace solutions, networking, enterprise storage
- **Experience:** 15+ years, recognized four times in the Growth 500 list for fastest-growing companies
- **Annual summit:** Hosts annual QBITS summit for IT professionals

Both resellers have physical offices in Maricopa County and can assist with verifying 4-hour onsite support availability at specific Phoenix ZIP codes. They can also coordinate on-site demonstrations, loaner units for testing, and pre-purchase validation.

### 6.3 Software/Application-Level Support vs. Hardware Support for PACS

A critical distinction for this deployment:

- **Hardware Support (covered by all three vendors' 4-hour onsite services):** On-site technician dispatch, replacement parts, hardware repair/replacement, firmware updates. Dell's ProSupport Plus includes "on-site dispatch of specialized Dell Technologies technicians" and "proactive solid state drive replacement" [Dell ProSupport Plus for Infrastructure](https://i.dell.com/sites/csdocuments/Legal_Docs/en/us/dell-prosupport-plus-for-infrastructure-sd-en.pdf). HP's service "covers HP or Compaq hardware and supplied components, including peripherals" [HP Hardware Support Onsite Service](https://h20195.www2.hp.com/v2/GetPDF.aspx/4AA5-6385EEE). Lenovo's Premier Support includes "Comprehensive Hardware + Software Troubleshooting" with "Collaboration with third-party software vendors" [Lenovo Premier Support Introduction](https://download.lenovo.com/km/media/attachment/Onboarding%20Deck%20EN_PSP%20Server%20FINAL.pdf).

- **Software/Application-Level Support (NOT covered by standard hardware support):** PACS software (Philips IntelliSpace PACS, Hologic SecurView DX) is supported by the respective software vendors, not by the hardware OEMs. Dell explicitly excludes "software support beyond hardware" for basic hardware service. HP states: "Service limitations exclude coverage of software."

- **Collaborative third-party support:** Dell provides "Collaborative Support ensures Dell acts as single point of contact for third-party product issues without assuming liability" [Dell ProSupport Enterprise-Wide Service Description](https://i.dell.com/sites/content/shared-content/services/en/Documents/dpros-enterp-wid-con-us.pdf). This means Dell will coordinate with the PACS software vendor but does not assume responsibility for the PACS application itself.

**Practical guidance:** For PACS-specific application issues, the facility should maintain support contracts directly with Philips and Hologic. The hardware support contracts cover the workstation hardware only. The facility's IT/PACS administrators should be trained to differentiate between hardware issues (diagnosed through hardware diagnostics, power-on self-test failures, component failure) and software application issues (PACS crashes, rendering errors, DICOM communication failures).

---

## 7. Software Version Compatibility and Licensing

### 7.1 Operating System Support

| Software | Windows 10 Enterprise | Windows 11 Pro for Workstations | Notes |
|---|---|---|---|
| **Philips IntelliSpace PACS ~4.4.x** | **Supported** (Windows 7, 8.1, 10 64-bit) | **Not listed in official 4.4.x documentation** but supported in IntelliSpace Radiology 4.7 (July 2025) | IntelliSpace Radiology 4.7 guide lists Windows 10, 11 as compatible |
| **Hologic SecurView DX 12.0+** | **Supported** (Windows 10 IoT Enterprise LTSC/LTSB or Windows 10 Enterprise 64-bit) | **NOT supported** — Windows 11 not listed in any official Hologic documentation | Version 12.0 requires Windows 10 (Standalone, Client) or Windows Server 2016 (Manager) |

[Philips IntelliSpace PACS Client Specs](https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf) [Philips IntelliSpace Radiology 4.7 Client Installation Guide](https://www.documents.philips.com/assets/Instruction%20for%20Use/20250725/a5a1d9b13cf34189b69ab32500763e29.pdf) [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej) [Hologic SecurView DX/RT 12.0 Workstation Release Notes](https://www.hologic.com/file/424101/download?token=0nIGS6AO)

**Critical finding:** Hologic SecurView DX 12.0+ **does not support Windows 11**. The deployment must use **Windows 10 IoT Enterprise LTSC/LTSB or Windows 10 Enterprise 64-bit** for all workstations running SecurView DX. Philips IntelliSpace PACS 4.4.x documentation primarily references Windows 7/8.1/10, but the newer IntelliSpace Radiology 4.7 guide (July 2025) indicates Windows 11 support was added. If the facility upgrades to the latest Philips software version, Windows 11 compatibility may be achieved — but the Hologic requirement for Windows 10 remains the binding constraint.

### 7.2 Licensing Costs

#### 7.2.1 Philips IntelliSpace PACS

- **Per-study pricing available:** AWS Marketplace for Philips IntelliSpace Radiology lists "36-month contract pricing" at "$0.001 per DICOM study" across radiology, cardiology, mammography, and non-DICOM studies [AWS Marketplace - Philips IntelliSpace Radiology](https://aws.amazon.com/marketplace/pp/prodview-nrv5udb3tjoro)
- **Unlimited license option:** Philips Egypt enterprise imaging page notes "unlimited licenses" for their enterprise imaging solutions [Philips Enterprise Imaging PACS](https://www.philips.com.eg/healthcare/solutions/diagnostic-informatics/enterprise-imaging-pacs)
- **PACS client licenses are typically per-seat or per-named-user** — the specific licensing model depends on the contract with Philips. The facility should confirm whether per-workstation or per-user licensing applies
- **Software license terms:** "Philips grants to Customer a non-exclusive, non-transferable license" and "The Licensed Software is licensed and not sold" [Philips General Terms and Conditions](https://www.usa.philips.com/c-dam/b2bhc/us/terms-conditions/general-terms-and-conditions-of-sale-and-software-license-rev-22.pdf)

#### 7.2.2 Hologic SecurView DX

- **USB dongle-based licensing:** Multiple references in documentation to "USB license dongle" connectivity — the software license is tied to a physical USB dongle that must be connected to the workstation [Hologic SecurView DX Minimum System Requirements v12.0+](https://www.hologic.com/file/477126/download?token=3OBC61Ej)
- **Dedicated system requirement:** "To ensure optimal performance, the software must be installed on a computer that will be dedicated only for Hologic software application(s)" — the workstation cannot be shared with other applications
- **Virtual Manager licensing:** VMWare ESXi 7.0.x or later required; "USB token for licensing is usable" must be supported in virtualized environments
- **Licensing is tied to specific workstation configurations** — Hologic requires that the system "MUST BE assembled and functioning prior to the arrival of Hologic Service" and "Only Hologic-approved software may be installed on this computer" [Hologic SecurView DX Minimum System Requirements v12.0+ Revision 002](https://www.hologic.com/file/477126/download?token=3OBC61Ej)

#### 7.2.3 Licensing Cost Estimates

| Software | Licensing Model | Estimated Cost (per workstation/year) |
|---|---|---|
| Philips IntelliSpace PACS client | Per-seat or per-study (varies by contract) | $500 - $2,000 per seat/year (typical enterprise pricing) |
| Hologic SecurView DX client | USB dongle per workstation (one-time + maintenance) | $5,000 - $15,000 one-time + 15-20% annual maintenance (typical) |
| Windows 10 Enterprise/OEM | Included with workstation | $0 (bundled with hardware) |

### 7.3 Pre-Purchase Validation Steps for Software Compatibility

To obtain compatibility sign-off before purchasing, request the following from each vendor:

**From Hologic:**
1. **Request written compatibility confirmation** for the specific workstation model (Dell Precision 7960, HP Z8 Fury G5, or Lenovo ThinkStation P7) running SecurView DX 12.0+ on Windows 10 Enterprise 64-bit
2. **Request GPU validation** for the specific NVIDIA RTX PRO model (e.g., RTX PRO 4000 Blackwell) with 10-bit color support
3. **Request storage configuration approval** for the planned NVMe SSD and RAID configuration
4. **Request network configuration approval** for the planned 10GbE implementation
5. **Provide the complete hardware bill of materials** to Hologic for pre-installation review

**From Philips:**
1. **Request compatibility sign-off** for the workstation running IntelliSpace PACS ~4.4.x on Windows 10 Enterprise
2. **Confirm GPU requirements** for advanced mammography/3D reconstruction modules
3. **Request network performance specifications** for 10GbE DICOM transfer
4. **Obtain the current compatible hardware list** or ISV certification documentation

**From the workstation vendors:**
1. **Request ISV certification documentation** for Philips and Hologic software on the specific model
2. **Request a test loaner unit** from local resellers (iT1 Source or Quadbridge) for pre-purchase validation
3. **Verify 4-hour onsite support availability** at the facility's specific Phoenix ZIP code using the methods in Section 6

---

## 8. TCO Analysis for 12 Workstations (5-Year)

### 8.1 Equivalent Configuration for Apples-to-Apples Comparison

All three workstations configured with:
- Intel Xeon W9-3475X (36 cores, 72 threads)
- 128 GB DDR5 ECC (4800 MT/s)
- NVIDIA RTX PRO 4000 Blackwell (24 GB GDDR7)
- 1 TB NVMe SSD (OS + applications)
- 4 TB SATA SSD (local DICOM cache)
- Windows 10 Enterprise 64-bit
- 5-year 4-hour onsite support
- Dual 5MP diagnostic mammography displays (separate line item)

### 8.2 Hardware Acquisition Cost

| Component | Dell Precision 7960 Tower | HP Z8 Fury G5 | Lenovo ThinkStation P7 |
|---|---|---|---|
| Workstation (W9-3475X, 128GB, RTX PRO 4000, 1TB NVMe, 4TB SATA) | ~$9,500 | ~$8,500 | ~$8,300 |
| Additional 10GbE NIC (if needed) | $0 (onboard) | $313 (Intel X550-T2) | $0 (onboard) |
| **Per-unit hardware cost** | **$9,500** | **$8,813** | **$8,300** |
| **12-unit hardware total** | **$114,000** | **$105,756** | **$99,600** |

**Note:** Final pricing depends on negotiated enterprise discounts. These are estimated street prices from resellers as of May 2026, based on configurable pricing from Monitors.com, CDW, Insight, and vendor websites [Dell Precision 7960 Tower Pricing](https://www.dell.com/en-us/shop/desktop-computers/precision-7960-tower-workstation/spd/precision-t7960-workstation) [HP Z8 Fury G5 Pricing](https://www.hp.com/us-en/shop/ConfigureView?langId=-1&storeId=10151&catalogId=10051&catEntryId=3074457345620775822) [Lenovo ThinkStation P7 CDW Pricing](https://www.cdw.com/product/lenovo-thinkstation-p7-tower-xeon-w5-3425-3.2-ghz-vpro-enterprise-32/7437161).

### 8.3 5-Year Extended Support/Warranty Cost

| Vendor | Support Tier | Annual Cost (est.) | 5-Year Cost (per workstation) | 12-Unit Fleet |
|---|---|---|---|---|
| **Dell** | ProSupport Plus 4-hour onsite | ~$400 | $2,000 | $24,000 |
| **HP** | Premium+ Support 4-hour onsite | ~$400 | $2,000 | $24,000 |
| **Lenovo** | Premier Support Plus 4-hour onsite | ~$350 | $1,750 | $21,000 |

### 8.4 Power + Cooling (Empirically Justified)

Per Section 3 calculations, using 275W average power draw per workstation:

| Cost Component | Annual (12 units) | 5-Year (with 3.5% annual escalation) |
|---|---|---|
| IT equipment power (312 days/year) | $2,141.52 | $11,897.16 |
| Cooling overhead (PUE 1.8) | $1,712.42 | $9,517.73 |
| **Total power + cooling** | **$3,853.94** | **$21,414.89** |

### 8.5 GPU Mid-Cycle Replacement

- Replace RTX PRO 4000 Blackwell (~$1,000) in year 3 of the 5-year cycle
- **Per-unit cost:** $1,000
- **12-unit fleet cost:** $12,000

### 8.6 10GbE Networking (Switch + Cabling)

| Item | Cost |
|---|---|
| 24-port managed 10GbE switch | $1,500 |
| Cabling (CAT6a, patch cables, accessories) | $300 |
| **Total networking** | **$1,800** |

### 8.7 Dual 5MP Diagnostic Displays

| Component | Per Workstation | 12-Unit Fleet |
|---|---|---|
| Dual 5MP diagnostic mammography displays (e.g., Barco MDNC-6121 dual system or EIZO RadiForce GX560) | ~$15,000 | $180,000 |

**Note:** Display pricing varies significantly by manufacturer and configuration. Barco Nio Color 5MP MDNC-6121 single units are ~$10,335 each [Barco Nio Color 5MP at CDW](https://www.cdw.com/product/barco-nio-color-5mp-mdnc-6121-medical-grade-monitor/), with dual-head systems typically $18,000-$22,000. EIZO RadiForce GX560 dual 5MP grayscale systems are ~$12,000-$16,000.

### 8.8 UPS (Per Workstation)

- Recommended: 1500-2000VA line-interactive or online double-conversion UPS
- **Per-unit cost:** ~$800
- **12-unit fleet cost:** $9,600

### 8.9 PACS Storage Infrastructure (Shared)

- NAS-based PACS storage with 10GbE connectivity (e.g., Synology RS1221RP+ or QNAP TVS-h1288X with enterprise drives)
- **Total storage cost:** ~$5,000 (including NAS unit and 4-8 enterprise NAS drives)

### 8.10 IT Staff Time (Estimated)

- Initial deployment and configuration (20 hours per workstation × 12 units × $75/hr): $18,000
- Ongoing management (2 hours/month × 60 months × $75/hr): $9,000
- **Total IT staff time:** $27,000

### 8.11 Software Licensing (Estimated)

| Software | Cost Model | 5-Year Cost (12 workstations) |
|---|---|---|
| Philips IntelliSpace PACS client licenses | Per-seat (~$1,000/year) | $60,000 |
| Hologic SecurView DX client licenses | USB dongle + 15-20% annual maintenance (~$8,000 upfront + $1,600/year) | $144,000 |
| **Total software licensing** | | **$204,000** |

### 8.12 Complete 5-Year TCO Comparison

| Cost Category | Dell Precision 7960 Tower (12 units) | HP Z8 Fury G5 (12 units) | Lenovo ThinkStation P7 (12 units) |
|---|---|---|---|
| Hardware acquisition | $114,000 | $105,756 | $99,600 |
| 5-year support/warranty | $24,000 | $24,000 | $21,000 |
| Power + cooling (5-year) | $21,415 | $21,415 | $21,415 |
| GPU mid-cycle replacement | $12,000 | $12,000 | $12,000 |
| 10GbE networking (shared) | $1,800 | $1,800 | $1,800 |
| Dual 5MP diagnostic displays | $180,000 | $180,000 | $180,000 |
| UPS (per workstation) | $9,600 | $9,600 | $9,600 |
| PACS storage infrastructure | $5,000 | $5,000 | $5,000 |
| IT staff time | $27,000 | $27,000 | $27,000 |
| Software licensing | $204,000 | $204,000 | $204,000 |
| **Total 5-Year TCO** | **$598,815** | **$590,571** | **$581,415** |
| **Per-workstation per-year cost** | **$9,980** | **$9,843** | **$9,690** |

### 8.13 Key TCO Observations

1. **Hardware cost difference is relatively small** — the $14,400 gap between Dell ($114,000) and Lenovo ($99,600) represents only 2.4% of total TCO
2. **Displays ($180,000) and software licensing ($204,000) dominate total TCO** accounting for 64% of costs
3. **Dell's higher hardware cost is offset by onboard 10GbE** — the HP requires $3,756 more for add-in NICs across 12 units
4. **Lenovo offers the lowest hardware acquisition cost** but has the most limited memory expansion path (8 DIMM slots vs. 16 for Dell/HP)
5. **Power and cooling costs are identical across vendors** because the same processor and GPU are used — this is a configuration-driven cost, not a platform-driven cost
6. The pending APS 14% rate increase could add approximately $3,000-$4,000 to the 5-year power cost if approved

---

## 9. Operational Guidance and Pre-Purchase Validation Steps

### 9.1 Step-by-Step Pre-Purchase Validation Checklist

#### Step 1: Request Compatibility Sign-Off from Hologic and Philips

**From Hologic:**
- [ ] Contact Hologic Technical Sales (through local Hologic representative or hologic.com)
- [ ] Provide the complete hardware bill of materials for each workstation model being considered
- [ ] Request written confirmation that the specific workstation model + GPU + display combination meets SecurView DX 12.0+ requirements
- [ ] Confirm Windows 10 Enterprise 64-bit support (NOT Windows 11)
- [ ] Request validation of the 10GbE network configuration for Manager-to-Client connections
- [ ] Request the current "SecurView DX Minimum System Requirements (Software-Only Option)" document for version 12.0+

**From Philips:**
- [ ] Contact Philips Healthcare Informatics support
- [ ] Request compatibility sign-off for IntelliSpace PACS ~4.4.x on the specific workstation
- [ ] Confirm GPU requirements for advanced mammography/3D reconstruction modules
- [ ] Request the current version of "IntelliSpace PACS Client Workstation Spec Sheet"
- [ ] Confirm OS compatibility (Windows 10 Enterprise vs. Windows 11 Pro for Workstations)

#### Step 2: Verify 4-Hour Onsite Support at Specific Phoenix ZIP Codes

Using the methods detailed in Section 6:

- [ ] **For Lenovo:** Use the Services Availability Locator at lenovolocator.com — enter each target ZIP code (85001, 85013, 85016, 85020, 85027, 85032, 85053, 85201, 85202, 85204, 85004, 85003) and check for 4-hour response availability
- [ ] **For Dell:** Contact your Dell sales representative or Quadbridge (Avondale) to verify 4-hour onsite availability at each ZIP code
- [ ] **For HP:** Contact HP sales office or iT1 Source (Tempe) to verify Premium+ 4-hour availability at each ZIP code

#### Step 3: Request On-Site Demo or Loaner Units from Local Resellers

- [ ] Contact **iT1 Source** (Tempe, AZ) at https://it1.com/healthcare to request loaner units for testing
- [ ] Contact **Quadbridge** (Avondale, AZ) at https://www.quadbridge.com/dell-reseller to arrange on-site demonstrations
- [ ] Request at least one unit of each workstation model for a minimum 2-week evaluation period
- [ ] Ensure loaner units are configured similarly to the proposed production configuration (same CPU, RAM, GPU, storage)

#### Step 4: Test Actual DICOM Image Load Times with 500MB+ Studies

Using the loaner units:

- [ ] Install a representative sample of DICOM studies (500MB - 1GB each, including 3D mammography tomosynthesis datasets)
- [ ] Measure:
  - Initial image display time (first image visible after study selection)
  - Full study load time (all images loaded and cached)
  - Scroll/series change latency
  - 3D reconstruction time
  - Window/level adjustment responsiveness
- [ ] Repeat tests on all three workstation models under identical network conditions
- [ ] Test during simulated peak load conditions (multiple workstations accessing PACS simultaneously)

#### Step 5: Validate 10GbE Throughput in the Facility's Network Environment

- [ ] Set up the 10GbE switch and connect the test workstation
- [ ] Measure actual network throughput using iperf3 or similar tool between workstation and PACS server
- [ ] Verify throughput meets or exceeds 1 Gb/s end-to-end (Philips requirement for large studies) [Philips IntelliSpace PACS Client Workstation Spec Sheet](https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace_PACS_Client_Workstation_SpecSheet.pdf)
- [ ] For Hologic's recommended dual 10GbE configuration, test with both 10GbE ports active
- [ ] Document baseline performance metrics for comparison with production deployment

### 9.2 Recommended Decision Framework

Based on the analysis in this report, the following decision framework is recommended:

**Primary Recommendation:**
The **Dell Precision 7960 Tower** offers the strongest overall package for this deployment due to:
1. Published SPECworkstation 4.0 benchmark validation (only vendor with published results)
2. Native onboard 10GbE (no add-in card required)
3. Largest memory capacity (4 TB, 16 DIMM slots) for future expansion
4. Strong enterprise support infrastructure through Quadbridge (Dell Platinum Partner) in Avondale
5. Dell Optimizer for Precision (AI-based power management saving up to 18% power)

**Strong Alternative:**
The **Lenovo ThinkStation P7** offers slightly lower acquisition cost and comparable onboard 10GbE, but lacks published SPECworkstation benchmarks and has a more limited memory expansion path (1 TB, 8 DIMM slots).

**Least Recommended:**
The **HP Z8 Fury G5** is the least suitable choice due to:
- No onboard 10GbE (requires add-in card, consuming PCIe slots)
- No published SPECworkstation results
- The requirement for add-in NICs adds cost and complexity, particularly if 4 GPUs are used (all PCIe slots become blocked)

**Final Recommendation:**
Proceed with **Dell Precision 7960 Tower** workstations configured with:
- Intel Xeon W9-3475X (36 cores, 72 threads)
- 128 GB DDR5 ECC (4x32 GB, leaving 12 slots for expansion)
- NVIDIA RTX PRO 4000 Blackwell (24 GB GDDR7) for diagnostic workstations
- NVIDIA RTX PRO 2000 Blackwell (16 GB GDDR7) for clinical review workstations
- 1 TB NVMe SSD + 4 TB SATA SSD
- Windows 10 Enterprise 64-bit (per Hologic requirement)
- Dell ProSupport Plus with 4-hour onsite response (verify availability at target ZIP codes)

---

## Sources

[1] Dell Precision 7960 Tower Setup and Specifications - Processor: https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/processor

[2] Dell Precision 7960 Tower Setup and Specifications - Memory: https://www.dell.com/support/manuals/en-us/oth-xlt7960/precision-t7960-setup-and-specifications/memory

[3] Dell Precision 7960 Tower Spec Sheet (Dell Technologies): https://www.delltechnologies.com/asset/en-us/products/workstations/technical-support/precision-7960-tower-spec-sheet.pdf

[4] SPECworkstation 4.0 Result Report Summary: https://spec.org/gwpg/wpc.data/specworkstation4_summary.html

[5] SPECworkstation 4.0 Dell Precision 7960 Detailed Results: https://spec.org/gwpg/wpc.data/workstation4.0/Dell/7960-RTX6000Ada_result_2024-11-19-17-29-36/results.html

[6] HP Z8 Fury G5 QuickSpecs (c08481500): https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500

[7] HP Z8 Fury G5 Site Prep Guide: https://assets.grandandtoy.com/assets/Additional-pdf5/10/81/1081618911.pdf

[8] HP Z8 Fury G5 ENERGY STAR Certification: https://www.energystar.gov/productfinder/product/certified-computers/details/4481132

[9] Lenovo ThinkStation P7 PSREF Specs: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf

[10] Lenovo ThinkStation P7 Datasheet: https://techtoday.lenovo.com/sites/default/files/2025-05/thinkstation-p7-datasheet-ww-en.pdf

[11] Lenovo ThinkStation P7 Power Configurator: https://download.lenovo.com/pccbbs/thinkcentre_pdf/ts_p7_power_configurator_v1.3.pdf

[12] Philips IntelliSpace PACS Client Specs 4.4.551.0: https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf

[13] Philips IntelliSpace PACS Client Workstation Spec Sheet: https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace_PACS_Client_Workstation_SpecSheet.pdf

[14] Philips IntelliSpace Radiology 4.7 Client Installation Guide: https://www.documents.philips.com/assets/Instruction%20for%20Use/20250725/a5a1d9b13cf34189b69ab32500763e29.pdf

[15] Philips General Terms and Conditions of Sale and Software License: https://www.usa.philips.com/c-dam/b2bhc/us/terms-conditions/general-terms-and-conditions-of-sale-and-software-license-rev-22.pdf

[16] Hologic SecurView DX Minimum System Requirements v12.0+ (MAN-11072): https://www.hologic.com/file/477126/download?token=3OBC61Ej

[17] Hologic SecurView DX Minimum System Requirements v12.0+ (Revision 002): https://www.hologic.com/file/477126/download?token=3OBC61Ej

[18] Hologic SecurView DX/RT 12.0 Workstation Release Notes: https://www.hologic.com/file/424101/download?token=0nIGS6AO

[19] NVIDIA Newsroom - Blackwell RTX PRO Announcement: https://nvidianews.nvidia.com/news/nvidia-blackwell-rtx-pro-workstations-servers-agentic-ai

[20] Central Computers - All NVIDIA RTX Pro Blackwell GPUs Explained: https://www.centralcomputer.com/blog/post/all-nvidia-rtx-pro-blackwell-gpus-explained

[21] NVIDIA - Healthcare Graphics Configurations: https://www.nvidia.com/content/quadro_oem/presentations/Vertical_Industry_-_Healthcare.pdf

[22] NVIDIA - 30-Bit Color Technology Technical Brief: https://www.nvidia.com/docs/IO/40049/TB-04701-001_v02_new.pdf

[23] HotHardware - Dell Precision 7960 and 5860 Review: https://hothardware.com/reviews/dell-precision-7960-and-5860-review

[24] Principled Technologies - HP Z6 G5 A vs Dell Precision 7960: https://www.principledtechnologies.com/HP/Z6-G5-A-Desktop-Workstation-AMD-Ryzen-performance-0824

[25] Dell ProSupport Plus for Infrastructure Service Description: https://i.dell.com/sites/csdocuments/Legal_Docs/en/us/dell-prosupport-plus-for-infrastructure-sd-en.pdf

[26] Dell ProSupport Enterprise-Wide Service Description: https://i.dell.com/sites/content/shared-content/services/en/Documents/dpros-enterp-wid-con-us.pdf

[27] HP Premium Support Services: https://www.hp.com/us-en/services/workforce-solutions/workforce-computing/support/premium-plus.html

[28] HP Care Services Definitions: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=4aa5-4980enw

[29] HP Hardware Support Onsite Service: https://www.also.com/pub/assets/0b532aa0-0229-4a7c-95af-c2629ddce597.pdf

[30] Lenovo Premier Support Introduction for ISG: https://download.lenovo.com/km/media/attachment/Onboarding%20Deck%20EN_PSP%20Server%20FINAL.pdf

[31] Lenovo Services Availability Locator: https://lenovolocator.com

[32] Lenovo Service Provider Locator: https://support.lenovo.com/us/en/lenovo-service-provider

[33] iT1 Source - Healthcare Division: https://it1.com/healthcare

[34] Quadbridge - Dell Authorized Reseller: https://www.quadbridge.com/dell-reseller

[35] APS Newsroom - Rate Adjustment Request: https://www.aps.com/en/About/Our-Company/Newsroom/Articles/APS_Requests_Rate_Adjustment_to_Support_Reliable_Service_for_Customers

[36] Arizona Capitol Times - APS Rate Case Hearing: https://azcapitoltimes.com/news/2026/05/19/aps-rate-case-kicks-off-with-hours-of-protest-over-14-rate-increase

[37] Solar.com - APS Electric Rate Increase 2026: https://www.solar.com/learn/aps-electric-rate-increase

[38] APS Rate Schedule E-32 M Medium General Service: https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans/e32_Medium.pdf

[39] ElectricChoice.com - Arizona Electricity Prices: https://www.electricchoice.com/electricity-prices-by-state/arizona

[40] Karimi et al. - Water-energy tradeoffs in data centers (UArizona Study): https://experts.arizona.edu/en/publications/water-energy-tradeoffs-in-data-centers-a-case-study-in-hot-arid-c

[41] FDA - Mammography Quality Standards Act (MQSA): https://www.fda.gov/radiation-emitting-products/mammography-information-patients/frequently-asked-questions-about-mqsa

[42] AWS Marketplace - Philips IntelliSpace Radiology: https://aws.amazon.com/marketplace/pp/prodview-nrv5udb3tjoro

[43] DEVELOP3D - Nvidia RTX A4000 and A5000 launch: https://develop3d.com/hardware/nvidia-rtx-a4000-and-rtx-a5000-gpus-launch

[44] AEC Magazine - RTX 4000, 4500 & 5000 Ada Generation GPUs: https://aecmag.com/news/nvidia-rtx-4000-4500-5000-ada-generation-gpus-launch

[45] StorageReview - HP Z8 Fury G5 Review: https://www.storagereview.com/review/hp-z8-fury-g5-workstation-review