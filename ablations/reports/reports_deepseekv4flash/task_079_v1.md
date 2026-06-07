# Comprehensive Workstation Recommendation for Healthcare Imaging Center – Phoenix, AZ

## Executive Summary

This report provides a detailed comparison of current-generation workstation options from Dell Precision, HP Z-series, and Lenovo ThinkStation lines for a healthcare imaging center in Phoenix, Arizona processing approximately 200 patients daily. The workstations must run Philips IntelliSpace PACS and Hologic 3D mammography (SecurView DX) software while handling 500MB+ DICOM files. Based on the analysis of CPU performance, GPU requirements, ECC memory needs, power consumption, networking, enterprise support availability in Phoenix, and five-year total cost of ownership (TCO), this report recommends the **Dell Precision 7960 Tower** or **Lenovo ThinkStation P7** as the primary candidates, with a strong recommendation for **AMD Threadripper PRO-based systems** (Dell Precision 7875 or HP Z6 G5 A) if maximum multi-threaded performance for 3D mammography reconstruction is prioritized.

---

## 1. Software Application Requirements

### 1.1 Philips IntelliSpace PACS

Philips IntelliSpace PACS is an advanced medical imaging platform providing multimodality clinical information access. The system has specific hardware requirements that inform workstation selection [1][2]:

**CPU Requirements:**
- Minimum: Intel i5 dual-core 2.5 GHz [1]
- For advanced visualization with Volume Vision and Advanced Mammography: **Intel six-core or two quad-core processors at 2.5 GHz or better** [2][3]
- Recommended configurations use processors with up to twelve logical cores at 2.5 GHz [1]

**RAM Requirements:**
- Minimum: 4 GB [1]
- For advanced configurations with Volume Vision and Advanced Mammography: **24 GB or more** [1]
- Real-world radiology reading stations commonly use **32 GB minimum** [4]

**GPU Requirements:**
- Graphics cards must support **OpenGL 3.2** [1][2]
- Onboard video memory: **1 GB to 2 GB** depending on use case [1]
- For advanced configurations: high-end graphics card with OpenGL 3.2 support [2]
- Diagnostic monitors must be DICOM-calibrated; **mammography displays require FDA 510(k) clearance** with at least 5 megapixel resolution [1][4]

**Network Requirements:**
- Minimum: 100 Mb/s adapters [1][2]
- For large studies (>700 MB CTs or mammography tomosynthesis): **1 Gb/s adapter and 1 Gb/s end-to-end connection to server** is required [2][3]

**Operating System:** Windows 10/11 Enterprise 64-bit supported [4]

### 1.2 Hologic SecurView DX (3D Mammography Software)

Hologic SecurView DX is described as a **"memory and computationally-intensive application"** for demanding clinical environments, with optimal performance when installed on a computer dedicated only to Hologic software [5][6][7].

**Current Generation Requirements (SecurView DX 12.0+):**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel Core i7-6700, 3.40 GHz | Intel Xeon E-2287GE @ 3.31 GHz (8-core, 16-thread) [5][6][7] |
| **RAM** | 32 GB DDR4 2400 MHz | **64 GB** [5][6][7] |
| **GPU** | 10-bit video card, 8 GB dedicated VRAM, PCIe 3.0 | 10-bit video card with up to **16 GB dedicated memory**, PCIe 3.0+ [5][6][7] |
| **Storage** | 8 TB SSD with 400+ MB/s read/write performance | Same, with 2:1 Manager-to-Client HDD capacity ratio [5][6] |
| **Network** | Gigabit Ethernet | **Dual 10 Gigabit NICs recommended** for server connections [5][6] |
| **Display** | Dual 5MP FDA-approved mammography monitors | Dual 5MP+ with QC/calibration software [5][7] |

**GPU Analysis for Hologic Software:**
The requirement for **8-16 GB dedicated video memory, 10-bit color depth, and PCIe 3.0+ interface** effectively mandates a professional-grade discrete GPU. Consumer gaming GPUs typically do not support 10-bit color output reliably for medical diagnostic displays. NVIDIA RTX A-series (now RTX PRO) or AMD Radeon Pro GPUs are the expected class [5][6]. Hologic's Cenova image processing server uses an **NVIDIA Tesla C2070 GPU** for 3D reconstruction, confirming Hologic's reliance on NVIDIA professional GPUs in their ecosystem [8].

### 1.3 Workload Modeling: 200 Patients/Day

At 200 patients daily with 500MB+ DICOM studies, the imaging center must process **100+ GB of image data per day**. Tomosynthesis mammography studies are significantly larger than 2D mammography (a bilateral screening tomosynthesis exam can be 1-3 GB) [9]. 

Hologic's **3DQuorum technology** reduces Clarity HD study sizes by **over 50%** by reconstructing high-resolution 3D data into 6mm SmartSlices, reducing the number of images radiologists need to review by two-thirds and saving an average of 1 hour per day in interpretation time [9][10].

**Network throughput at 1 Gb/s:** A 500 MB DICOM study transfers in approximately 4-8 seconds at 1 Gb/s, which is acceptable. However, for multiple concurrent reads and server connections, **10 GbE networking** is recommended for the infrastructure backbone [2][5][6].

---

## 2. CPU Performance Evaluation

### 2.1 Available Processor Options by Workstation Line

| Workstation Line | CPU Family | Max Cores/Threads | Max Turbo | TDP |
|-----------------|------------|-------------------|-----------|-----|
| **Dell Precision 7960 Tower** | Intel Xeon W-3400 (4th/5th Gen) | 60/120 (W9-3495X) | 4.80 GHz | 350W (up to 420W) [11][12] |
| **Dell Precision 7875 Tower** | AMD Ryzen Threadripper PRO 7000 WX | 96/192 (7995WX) | 5.30 GHz | 350W [13][14] |
| **HP Z8 Fury G5** | Intel Xeon W-3400 | 60/120 (W9-3495X) | 4.80 GHz | 350W [15][16] |
| **HP Z6 G5 A** | AMD Ryzen Threadripper PRO 7000/9000 WX | 96/192 (9995WX) | 5.40 GHz | 350W [17][18] |
| **HP Z4 G5** | Intel Xeon W-2400 | 26 cores (W7-2595X) | 4.80 GHz | 225W [19][20] |
| **Lenovo ThinkStation PX** | Dual Intel Xeon Scalable (5th/4th Gen) | 2×60=120 cores | 4.20 GHz | 350W per CPU [21][22] |
| **Lenovo ThinkStation P7** | Intel Xeon W-3400 | 56 cores | 4.80 GHz | 350W [23][24] |
| **Lenovo ThinkStation P8** | AMD Ryzen Threadripper PRO 7000 WX | 96 cores (7995WX) | 5.30 GHz | 350W [25][26] |
| **Lenovo ThinkStation P5** | Intel Xeon W-2400 | 24 cores (W7-2495X) | 4.80 GHz | 225W [27][28] |

### 2.2 PassMark Benchmark Scores

PassMark CPU Mark (multithreaded) and Single Thread Rating provide the most comprehensive available benchmark data for these processors [29][30][31][32][33]:

| Processor | CPU Mark (Multi) | Single Thread Rating | Integer Math | Floating Point Math |
|-----------|-----------------|---------------------|--------------|-------------------|
| **AMD Threadripper PRO 7995WX** (96-core) | **140,967** | 3,829 | 1,012,792 MOps/s | 584,291 MOps/s |
| **AMD Threadripper PRO 7975WX** (32-core) | **95,489** | **3,983** | 197,519 MOps/s | 149,423 MOps/s |
| **Intel Xeon W9-3495X** (56-core) | 90,454 | 3,430 | 401,923 MOps/s | 324,716 MOps/s |
| **Intel Xeon W7-2495X** (24-core) | ~52,000 (est.) | ~3,400 | — | — |
| **Intel Xeon W5-2455X** (12-core) | 37,123 | 3,383 | 119,603 MOps | 93,237 MOps |
| **Intel Xeon W5-3423** (12-core) | 27,944 | 2,663 | 84,594 MOps/s | 71,335 MOps/s |

### 2.3 Analysis for DICOM Workloads

**Single-Threaded Performance (important for UI responsiveness, DICOM window/level adjustments):**
- AMD Threadripper PRO 7975WX leads with a Single Thread Rating of **3,983**, which is 16% faster than the Intel Xeon W9-3495X (3,430) [30][31]
- Higher single-thread performance translates to faster image manipulation (pan, zoom, window/level) and better overall UI responsiveness

**Multi-Threaded Performance (important for batch processing, 3D reconstruction, AI inference):**
- The AMD Threadripper PRO 7995WX (96-core) dominates with a CPU Mark of **140,967**, outperforming the Intel Xeon W9-3495X by **56%** [29][33]
- For 3D mammography reconstruction and AI-based detection (Genius AI Detection with 94% sensitivity), higher multi-threaded performance directly reduces processing time [9][10]

**Intel's Comparative Claims:**
Intel's internal testing using SPECworkstation 3.1 claims the Xeon W9-3495X delivers "up to 29% better product development performance" compared to the AMD Ryzen Threadripper PRO 7975WX [34]. However, no officially published SPECworkstation 3.1 results exist for any of these processors on SPEC.org (the summary page was last updated June 15, 2022, predating all these CPUs) [35]. Third-party benchmarks from Puget Systems show that AMD Zen4 Threadripper PRO generally outperforms Intel Xeon W9-3495X in multi-core scalability, while Intel gives better performance at lower core counts [36].

**Recommendation:** For a healthcare imaging center running both Philips IntelliSpace PACS (which benefits from strong single-threaded performance) and Hologic 3D mammography software (which benefits from high multi-threaded throughput for reconstruction), the **AMD Ryzen Threadripper PRO 7975WX (32-core)** offers the best balance—leading single-thread performance and strong multi-thread capabilities at a reasonable price point ($3,899) [30]. The 96-core 7995WX is overkill unless the center performs extensive AI training or batch 3D reconstruction on-site [36].

---

## 3. Memory (ECC Support)

All three workstation lines support DDR5 ECC memory, which is **required** for medical imaging workstations running 24/7 diagnostic operations to prevent single-bit memory errors from corrupting image data [1][2][5]:

| Workstation | Max RAM | DIMM Slots | Memory Channels | Memory Speed |
|------------|---------|------------|-----------------|--------------|
| **Dell Precision 7960 Tower** | Up to **4 TB** DDR5 ECC | 16 | 8-channel | 4800 MHz [11][37] |
| **Dell Precision 5860 Tower** | Up to **2 TB** DDR5 ECC | 8 | Quad-channel | 4800 MT/s [38][39] |
| **Dell Precision 7875 Tower** (AMD) | Up to **2 TB** DDR5 ECC | 8 | 8-channel | 5200 MT/s [13][14] |
| **HP Z8 Fury G5** | Up to **2 TB** DDR5 ECC | 16 | 8-channel | 4800 MHz [15][16] |
| **HP Z6 G5 A** (AMD) | Up to **1 TB** DDR5 ECC | 8 | 8-channel | 5200 MT/s [17][18] |
| **HP Z4 G5** | Up to **512 GB** DDR5 ECC | 8 | Quad-channel | 4800 MT/s [19][20] |
| **Lenovo ThinkStation PX** | Up to **2 TB** DDR5 ECC | 16 (8 per CPU) | 8 per CPU | 4800-5600 MT/s [21][22] |
| **Lenovo ThinkStation P7** | Up to **1 TB** DDR5 ECC | 8 | 8-channel | up to 5600 MT/s [23][24] |
| **Lenovo ThinkStation P5** | Up to **512 GB** DDR5 ECC | 8 | Quad-channel | 4800 MHz [27][28] |
| **Lenovo ThinkStation P3 Tower** | Up to **128 GB** DDR5 ECC | 4 | Dual-channel | 4400 MHz [40][41] |

**Recommended Configuration:** A minimum of **64 GB ECC DDR5** per workstation. Given that Hologic requires 64 GB recommended RAM and Philips recommends 24 GB+ for advanced configurations, 64 GB provides adequate headroom for caching multiple large studies for rapid comparison [1][5]. The 8-channel memory configurations on the Xeon W-3400 and Threadripper PRO platforms provide significantly higher memory bandwidth, which benefits large DICOM dataset manipulation.

---

## 4. Power Consumption and Cost Analysis

### 4.1 Estimated Power Draw Under Typical Load

Power consumption varies significantly by configuration. Based on available data from power supply ratings and reviews [42][43][44]:

| Workstation Model | Typical Power Supply | Estimated Typical Draw | Maximum Draw |
|-----------------|---------------------|----------------------|--------------|
| **Dell Precision 7960** (Xeon W9-3495X + RTX A6000) | 1400W Platinum | **500-700W** | Up to 1200W |
| **Dell Precision 5860** (Xeon W5/W7 + RTX A4000) | 750W/1350W Platinum | **300-450W** | Up to 750W |
| **Dell Precision 7875** (Threadripper 7975WX + RTX A6000) | 1350W Platinum | **450-650W** | Up to 1100W |
| **HP Z8 Fury G5** (Xeon W9-3495X + Quad RTX A6000) | 1125W/2250W Gold | **500-800W** | Up to 1850W |
| **HP Z6 G5 A** (Threadripper 7975WX + RTX A6000) | 1000W-1450W Gold | **400-600W** | Up to 1000W |
| **HP Z4 G5** (Xeon W5/W7 + RTX A4000) | 525W-1125W Gold | **250-400W** | Up to 700W |
| **Lenovo ThinkStation P7** (Xeon W9-3495X + RTX A6000) | 1000W/1400W Platinum | **450-650W** | Up to 1000W |
| **Lenovo ThinkStation P5** (Xeon W7-2495X + RTX A4500) | 750W/1000W Platinum | **300-450W** | Up to 700W |

For TCO calculations, the report assumes a **moderate high-performance configuration averaging 500W** per workstation under typical load (mix of reading, processing, and idle periods across a 16-hour workday).

### 4.2 Arizona Commercial Electricity Rates

The user's referenced rate of **$0.13/kWh (13¢/kWh)** for Arizona commercial electricity is slightly above the published Arizona commercial average of **11.8¢/kWh** (May 2026) but is within a plausible range depending on the specific utility, rate schedule, and demand charges [45][46].

**Phoenix-Area Utility Context:**
- **APS (Arizona Public Service)** serves the Phoenix area with Rate Schedule E-32 M for medium commercial customers. Summer energy charges are approximately 11.53¢/kWh for the first tier, with additional demand charges of $14.69-$29.42/kW depending on service level [47][48]
- **SRP (Salt River Project)** serves portions of Phoenix with General Service price plans using "stretcher block" pricing and demand charges [49]
- APS has filed for a **13.99% net revenue increase** effective in the second half of 2026, which would raise commercial rates [50][51]

For conservative TCO calculations, this report uses **$0.13/kWh** as specified, recognizing it represents a realistic blended rate including demand charges and surcharges for a medium commercial account.

### 4.3 Operating Schedule Clarification

The user specified 16-hour daily operation, 365 days/year but noted actual operation is likely 6-7 days/week. For a healthcare imaging center:

- **Typical operation:** 6 days/week (Monday-Saturday) × 52 weeks = **312 days/year**
- **Emergency/on-call coverage:** Some workstations may need to remain available 7 days/week
- **Conservative annual calculation:** 365 days/year for power cost estimation (systems may remain powered on for updates, backups, and remote access even when not actively reading)

This report provides calculations for **both 365-day and 312-day** scenarios.

### 4.4 Annual Power Cost Calculation

**Per Workstation (500W average draw):**

| Scenario | Daily Hours | Days/Year | kWh/Year | Cost at $0.13/kWh |
|----------|-------------|-----------|----------|-------------------|
| Typical clinical operation | 16 | 312 | 2,496 | **$324.48** |
| Full-year continuous operation | 16 | 365 | 2,920 | **$379.60** |
| 24/7 operation (servers) | 24 | 365 | 4,380 | **$569.40** |

**Total for 12 Workstations (16 hours/day, 365 days/year):**
- IT equipment power: 12 × 2,920 kWh = **35,040 kWh/year**
- IT equipment electricity cost: 35,040 × $0.13 = **$4,555.20/year**

### 4.5 Cooling/HVAC Impact

Cooling is a significant factor in Phoenix's desert climate. For an on-premises IT room:

- **Power Usage Effectiveness (PUE)** for a small IT room in Phoenix with standard HVAC: estimated **1.8** (meaning for every 1 kW of IT power, 0.8 kW additional is needed for cooling/overhead) [52][53]
- **Total facility power** for 12 workstations: 6 kW IT × 1.8 PUE = **10.8 kW**
- **Annual facility electricity cost** (16 hrs/day, 365 days): 10.8 kW × 16 hrs × 365 days × $0.13 = **$8,199.36/year**
- **Cooling overhead cost**: $8,199.36 - $4,555.20 = **$3,644.16/year**

### 4.6 5-Year Power Cost Summary

| Cost Component | Per Workstation (5-Year) | Total Fleet: 12 Workstations (5-Year) |
|----------------|-------------------------|--------------------------------------|
| IT equipment power (16hrs/365days) | $1,898.00 | $22,776.00 |
| Cooling overhead | $727.40 | $8,728.80 |
| **Total all-in power** | **$2,625.40** | **$31,504.80** |

*Note: These estimates assume constant $0.13/kWh. Given APS's proposed 14% rate increase for 2026, actual costs may be 5-10% higher over the 5-year period [50][51].*

---

## 5. Networking: 10GbE Compatibility

### 5.1 Built-in 10GbE Support

| Workstation | Built-in 10GbE | Details |
|-------------|---------------|---------|
| **Dell Precision 7960 Tower** | **Yes** | Onboard 1GbE + 10GbE ports standard [11][54] |
| **Dell Precision 5860 Tower** | **Yes** | One 1GbE RJ-45 + one 10GbE RJ-45 standard [38][55] |
| **Dell Precision 7875 Tower** (AMD) | Yes (likely) | Dual Ethernet, typical config includes 10GbE [13][14] |
| **HP Z8 Fury G5** | Add-in required | 1GbE onboard; HP Z Dual-Port 10GbE Network Module available as add-in [56][57] |
| **HP Z6 G5 / Z6 G5 A** | **Yes** (some configs) | Onboard 10GbE + 1GbE per some configuration guides; 10GbE also available as add-in [17][18][58] |
| **HP Z4 G5** | Add-in required | 1GbE standard; optional 10GbE via PCIe add-in card [19][20] |
| **Lenovo ThinkStation PX** | **Yes** | Dual onboard: 1GbE + 10GbE standard; optional additional 25GbE ports [21][22] |
| **Lenovo ThinkStation P7** | **Yes** | Dual onboard: 1GbE + 10GbE standard [23][24] |
| **Lenovo ThinkStation P5** | Add-in required | 1GbE standard; 10GbE available as optional add-in PCIe card [27][28] |
| **Lenovo ThinkStation P3 Tower** | Add-in required | 1GbE standard; 10GbE via Flex IO or PCIe add-in [40][41] |

### 5.2 Recommendation for 10GbE

All the mid-range and high-end workstations (Dell Precision 7960/7875, HP Z6 G5 A, Lenovo ThinkStation P7/PX) support 10GbE either built-in or via add-in cards. For a facility processing 100+ GB of DICOM data daily:

- **Workstations should have built-in 10GbE** to avoid occupying a PCIe slot and simplify cabling
- **Server connections require dual 10GbE** as recommended by Hologic for SecurView DX Manager [5][6]
- Consider a 10GbE switch (e.g., 24-port) to interconnect workstations and the PACS server

The **Dell Precision 7960 Tower** and **Lenovo ThinkStation P7** are the most convenient choices with native dual 1GbE + 10GbE onboard.

---

## 6. GPU Requirements

### 6.1 Philips IntelliSpace PACS GPU Needs

Philips IntelliSpace PACS requires **OpenGL 3.2 support** with **1-2 GB VRAM** [1][2]. For:
- **Basic 2D PACS viewing:** Integrated graphics that support OpenGL 3.2 may suffice for non-diagnostic viewing
- **Advanced visualization modules** (Volume Vision, Advanced Mammography): A **dedicated discrete GPU** is required [1]
- **Diagnostic mammography displays:** Require FDA-approved 5MP monitors connected to a GPU capable of driving them at full resolution and color depth

### 6.2 Hologic SecurView DX GPU Needs

Hologic's requirements are significantly more demanding:
- **8-16 GB dedicated video memory** [5][6]
- **10-bit color depth support** (consumer GPUs typically do not support this for medical displays) [5]
- **DirectX 9.0c or higher** and **OpenGL 2.0** (MultiView MM) [5]
- **PCIe 3.0 or higher** interface [5]

These requirements **effectively mandate a professional-grade discrete GPU** such as:
- **NVIDIA RTX A-series** (e.g., RTX A4000 16GB, RTX A4500 20GB, RTX A6000 48GB) — now branded as **NVIDIA RTX PRO**
- **AMD Radeon Pro** (e.g., W7600, W7900)

### 6.3 Recommended GPU Configuration

| Use Case | Recommended GPU | VRAM | Purpose |
|----------|----------------|------|---------|
| **Primary diagnostic workstation** (mammography) | NVIDIA RTX A4500 20GB or RTX A6000 48GB | 20-48 GB | Drives dual 5MP diagnostic monitors, handles 3D mammography rendering |
| **Clinical review workstation** (non-diagnostic) | NVIDIA RTX A4000 16GB | 16 GB | Drives 2-3MP color monitors, sufficient for PACS viewing |
| **AI/image processing server** | NVIDIA RTX PRO 6000 Blackwell 96GB | 96 GB | For future AI workloads and batch reconstruction |

A mid-range professional GPU like the **NVIDIA RTX A4500 (20GB)** provides an excellent balance for diagnostic mammography workstations, meeting Hologic's VRAM requirements while being cost-effective [59].

---

## 7. Enterprise Support Options in Phoenix Metro Area

### 7.1 Dell Support

**Dell ProSupport and ProSupport Plus** are available nationwide, including the Phoenix metro area [60][61]:

| Service Tier | Response Time | Coverage | Key Features |
|-------------|---------------|----------|--------------|
| **Basic Support** | Next Business Day | Business hours | Remote diagnosis, NBD onsite [60] |
| **ProSupport** | 4-hour or NBD onsite options | 24/7 | Certified experts, priority access, extended software support [60][61] |
| **ProSupport Plus** | 4-hour onsite (objective) | 24/7 | **Proactive monitoring via SupportAssist**, accidental damage, dedicated service manager, Keep Your Hard Drive [60][62] |

**Phoenix-Specific:** Dell's ProSupport Plus states on-site dispatch with four-hour response objective "within defined geographic areas" — major metro areas like Phoenix typically qualify [62]. Dell has authorized service partners in the Phoenix area including Quadbridge (Avondale, AZ) and iT1 Source (Tempe, AZ) [63][64].

### 7.2 HP Support

**HP Inc.** (workstations, PCs — separate from Hewlett Packard Enterprise) offers three support tiers [65][66]:

| Service Tier | Response Time | Coverage | Key Features |
|-------------|---------------|----------|--------------|
| **HP Essential Support** | Next Business Day | Business hours | Remote diagnostics, NBD onsite repair [65] |
| **HP Premium Support** | Next Business Day or faster | 24/7 | AI-powered predictive issue detection, automated ticketing [65] |
| **HP Premium+ Support** | Fastest response (4-hour available) | 24/7 | Preferred access to experts and parts, onsite repair wherever work happens [65] |
| **HP Care Packs** | Up to 4-hour 24x7 | Custom | Extended warranty up to 5 years, Accidental Damage Protection [66] |

**Phoenix-Specific:** HP has authorized service providers in the Phoenix area. iT1 Source (Tempe, AZ) is an HP authorized reseller with dedicated healthcare division [64][67]. Quadbridge (Avondale, AZ) is also an authorized HP reseller [68].

### 7.3 Lenovo Support

**Lenovo Premier Support and Premier Support Plus** are available nationally [69][70]:

| Service Tier | Response Time | Coverage | Key Features |
|-------------|---------------|----------|--------------|
| **Standard On-Site** | Next Business Day (NBD) | 9×5 | Parts and labor at customer location [69][71] |
| **Premier Support** | NBD or **4-hour** options | 24/7/365 | Dedicated senior-level engineer, 20-minute e-ticket response, single point of contact [69][70] |
| **Premier Support Plus** | Same as Premier | 24/7/365 | AI-powered proactive support, Accidental Damage Protection, annual PC Health Check [69] |

**Phoenix-Specific:** Lenovo has authorized service providers in the Phoenix/Tempe area (e.g., Desert Computer Solutions in Tempe) [72]. Lenovo's Premier Support services are available in major metro areas including Phoenix, with 4-hour response options available for eligible locations [69][70].

### 7.4 Local Enterprise Resellers (Phoenix Metro Area)

| Reseller | Location | Dell | HP | Lenovo | Healthcare Focus |
|----------|----------|------|----|--------|-----------------|
| **iT1 Source** | Tempe, AZ | ✓ | ✓ (HPE Platinum) | ✓ | **Dedicated healthcare division** [64][67] |
| **Quadbridge** | Avondale, AZ | **Dell Platinum Partner** | ✓ | ✓ | Multi-brand IT solutions [63][68] |

Both resellers have physical offices in the Phoenix metro area (Maricopa County), are authorized for all three major workstation brands, and have the scale to support enterprise healthcare deployments.

---

## 8. 5-Year Total Cost of Ownership (TCO) Analysis

### 8.1 Assumptions

- **Deployment:** 12 workstations
- **Operating period:** 16 hours/day, 365 days/year (conservative)
- **Electricity rate:** $0.13/kWh (blended including demand charges)
- **PUE:** 1.8 (Phoenix IT room with standard HVAC)
- **Warranty:** 5-year extended warranty/support included
- **Workstation specification:** Mid-high end configuration (Xeon W or Threadripper PRO, 64GB ECC, professional GPU)

### 8.2 Hardware Acquisition Cost (Estimated)

| Workstation | Base Config | Recommended Config (est.) | 12-Unit Fleet Cost |
|------------|-------------|--------------------------|-------------------|
| **Dell Precision 7960 Tower** | $9,886 [73] | ~$15,000 | $180,000 |
| **Dell Precision 7875 Tower** (AMD) | $6,714 [13] | ~$14,000 | $168,000 |
| **HP Z8 Fury G5** | $5,319 [74] | ~$16,000 | $192,000 |
| **HP Z6 G5 A** (AMD) | ~$4,500 [17] | ~$13,000 | $156,000 |
| **Lenovo ThinkStation P7** | ~$6,455 [75] | ~$14,500 | $174,000 |
| **Lenovo ThinkStation P5** | $3,289 [27] | ~$9,000 | $108,000 |

*Recommended configurations include: Xeon W7-2495X or Threadripper PRO 7975WX, 64GB DDR5 ECC, NVIDIA RTX A4500 20GB, 1TB NVMe SSD + 4TB storage, 10GbE, Windows 11 Pro for Workstations.*

### 8.3 Extended Warranty/Support Cost (5 Years)

| Vendor | Support Tier | Annual Cost (est.) | 5-Year Cost (per workstation) | 12-Unit Fleet |
|--------|-------------|-------------------|------------------------------|---------------|
| **Dell** | ProSupport Plus (4-hour) | ~$400-600 | $2,000-3,000 | $24,000-36,000 |
| **HP** | Premium+ or Care Pack (4-hour) | ~$400-600 | $2,000-3,000 | $24,000-36,000 |
| **Lenovo** | Premier Support Plus (4-hour) | ~$400-600 | $2,000-3,000 | $24,000-36,000 |

### 8.4 5-Year TCO Breakdown (Per Workstation, Mid-Range Config)

| Cost Category | Dell Precision 7960 | Dell Precision 7875 | HP Z8 Fury G5 | HP Z6 G5 A | Lenovo ThinkStation P7 | Lenovo ThinkStation P5 |
|--------------|--------------------|--------------------|--------------|------------|----------------------|----------------------|
| **Hardware (est.)** | $15,000 | $14,000 | $16,000 | $13,000 | $14,500 | $9,000 |
| **5-Year Support** | $2,500 | $2,500 | $2,500 | $2,500 | $2,500 | $2,500 |
| **IT Power (5-yr)** | $1,898 | $1,898 | $1,898 | $1,898 | $1,898 | $1,898 |
| **Cooling Overhead (5-yr)** | $727 | $727 | $727 | $727 | $727 | $727 |
| **GPU Replacement (1 mid-cycle)** | $1,200 | $1,200 | $1,200 | $1,200 | $1,200 | $1,200 |
| **Networking (10GbE switch share)** | $500 | $500 | $500 | $500 | $500 | $500 |
| **Displays (dual 5MP)** | $6,000 | $6,000 | $6,000 | $6,000 | $6,000 | $6,000 |
| **Total Per Workstation (5-Year)** | **$27,825** | **$26,825** | **$28,825** | **$25,825** | **$27,325** | **$21,825** |
| **Total Fleet: 12 Units (5-Year)** | **$333,900** | **$321,900** | **$345,900** | **$309,900** | **$327,900** | **$261,900** |

### 8.5 Additional TCO Factors

1. **Storage Infrastructure:** A shared PACS server or NAS with redundant SSDs (20-40 TB usable) for image caching adds approximately $10,000-20,000 for the fleet
2. **Network Infrastructure:** 10GbE switch (24-port): $2,000-5,000; cabling: $1,000-2,000
3. **UPS/Backup Power:** Per-workstation UPS: $300-500 each ($3,600-6,000 total), plus room-level UPS for server: $3,000-5,000
4. **IT Staff Time:** Configuration, deployment, and ongoing management: approximately 20-40 hours per workstation over 5 years for imaging-specific software setup
5. **Software Licensing:** Windows 11 Pro for Workstations (often included), PACS client licenses (often per-seat)
6. **Compliance Costs:** HIPAA compliance documentation, audit preparation, security monitoring
7. **Replacement Parts:** Keyboards, mice, monitor calibrations, SSD upgrades mid-cycle

### 8.6 Fully Loaded 5-Year TCO Estimate (12 Workstations)

| Category | Low Estimate | High Estimate |
|----------|-------------|--------------|
| Workstations (12 units) | $108,000 - $192,000 |
| 5-Year Support | $24,000 - $36,000 |
| Power + Cooling (5 years) | $31,505 |
| Displays (12 × dual 5MP) | $72,000 |
| Storage Infrastructure | $10,000 - $20,000 |
| Network Infrastructure | $3,000 - $7,000 |
| UPS/Battery Backup | $6,600 - $11,000 |
| IT Staff Time | $5,000 - $15,000 |
| GPU Mid-Cycle Upgrades | $14,400 |
| Compliance/Security | $3,000 - $8,000 |
| **Total Fleet TCO (5-Year)** | **$277,505 - $406,905** |

**Per-Workstation per-Year Cost:** $4,625 - $6,782

---

## 9. Final Recommendations

### 9.1 Primary Recommendation: Dell Precision 7960 Tower or Lenovo ThinkStation P7

For a healthcare imaging center running **Philips IntelliSpace PACS and Hologic SecurView DX 12.0+** serving 200 patients/day:

**Recommended Configuration:**
- **CPU:** Intel Xeon W7-2495X (24-core) or AMD Ryzen Threadripper PRO 7975WX (32-core)
- **RAM:** 64 GB DDR5 ECC (expandable to 128-256 GB)
- **GPU:** NVIDIA RTX A4500 20GB (diagnostic) or RTX A6000 48GB (for 3D reconstruction)
- **Storage:** 1 TB NVMe SSD (OS/apps) + 4-8 TB SSD (local image cache)
- **Network:** Native 10GbE (Dell Precision 7960 or Lenovo ThinkStation P7 — both have built-in)
- **Displays:** Dual 5MP FDA-approved diagnostic mammography monitors (e.g., Barco, Eizo, JVC)
- **OS:** Windows 11 Pro for Workstations 64-bit

### 9.2 Why These Recommendations

1. **CPU Choice:** The **AMD Threadripper PRO 7975WX** offers the best single-threaded performance (Single Thread Rating 3,983) and excellent multi-threaded capability (CPU Mark 95,489), outperforming the Intel Xeon W9-3495X in both metrics while costing $3,899 vs $5,889 [30][31]. This is ideal for the mixed workload of PACS UI responsiveness and 3D mammography reconstruction. Systems with this CPU include the Dell Precision 7875 Tower, HP Z6 G5 A, and Lenovo ThinkStation P8.

2. **Memory:** 64 GB ECC DDR5 is the recommended starting point, matching Hologic's recommended specification and providing headroom for caching large DICOM studies [5][6].

3. **GPU:** The NVIDIA RTX A4500 (20GB) meets Hologic's 8-16 GB VRAM requirement and supports 10-bit color for diagnostic mammography displays [59]. For facilities planning future AI workloads, the RTX A6000 (48GB) or RTX PRO 6000 Blackwell (96GB) provides additional headroom.

4. **10GbE:** Built-in 10GbE on Dell Precision 7960 and Lenovo ThinkStation P7 simplifies deployment compared to HP workstations that require add-in cards.

5. **Support:** Both **iT1 Source (Tempe, AZ)** and **Quadbridge (Avondale, AZ)** provide local enterprise reseller support for all three brands, with iT1 Source having a dedicated healthcare division [64].

### 9.3 Cost-Optimized Alternative: Lenovo ThinkStation P5

For facilities where budget is a primary concern, the **Lenovo ThinkStation P5** with an Intel Xeon W7-2495X (24-core), 64GB ECC, and NVIDIA RTX A4500 offers excellent PACS performance at approximately $9,000 per workstation (vs $14,000-16,000 for higher-end models). The trade-off is:
- No built-in 10GbE (requires add-in card, ~$200-400)
- Maximum 512GB RAM vs 1-4 TB on higher-end models
- Dual GPU maximum vs 3-4 on higher-end models

### 9.4 Phoenix-Specific Considerations

1. **Power:** Given APS's proposed 14% rate increase for 2026, lock in 5-year support contracts that include power cost projections, or consider solar augmentation for the IT room [50][51]
2. **Cooling:** The Edged Phoenix Data Center in nearby Mesa achieves a PUE of 1.15 with waterless cooling [52]. While an on-premises IT room won't match this, consider liquid cooling or hot-aisle containment for the server room
3. **Local Support:** Both iT1 Source (Tempe) and Quadbridge (Avondale) offer physical presence in Maricopa County with enterprise-grade support capabilities [63][64]
4. **Regulatory:** Arizona eliminated the Renewable Energy Standard and Tariff (REST) in 2025, but no state-level healthcare IT infrastructure mandates currently affect workstation specifications [45]

### 9.5 Clarification on Operation Schedule

The user noted 16-hour daily operation, 365 days/year, with actual operation likely 6-7 days/week. For a healthcare imaging center:
- **Primary reading days:** 6 days/week (312 days/year) is standard for outpatient imaging centers
- **Emergency/STAT coverage:** Some workstations may be needed 7 days/week
- **Server infrastructure:** PACS servers typically run 24/7/365
- **Power TCO calculations:** The report provides both 312-day and 365-day scenarios

---

## Sources

[1] Philips IntelliSpace PACS Client Specs 4.4.551.0: https://www.ehealthsask.ca/services/resources/Pages/IntelliSpace-PACS-client-specs.aspx

[2] Philips IntelliSpace PACS Spec Sheet (eHealth Saskatchewan): https://www.ehealthsask.ca/services/manuals/pacs/Documents/IntelliSpace-PACS-Client-Workstation-Spec-Sheet.pdf

[3] Philips IntelliSpace Portal Instructions for Use 12.1.10: https://www.usa.philips.com/healthcare/resources/feature-detail/intellispace-portal

[4] Philips IntelliSpace Radiology 4.7 Client Installation Guide: https://www.usa.philips.com/healthcare/resources/support-documentation/intellispace-radiology

[5] Hologic SecurView DX Minimum System Requirements 12.0+: https://www.hologic.com/hologic-products/securview-workstations

[6] Hologic SecurView DX Workstation Brochure: https://www.hologic.com/sites/default/files/2022-12/SecurView-DX-Brochure.pdf

[7] Hologic SecurView Workstations Product Page: https://www.hologic.com/hologic-products/securview-workstations/securview-workstations

[8] Hologic Cenova Server System Requirements: https://www.hologic.com/hologic-products/cenova-image-analytics-server

[9] Hologic 3DQuorum Imaging Technology: https://www.hologic.com/hologic-products/breast-health/3dquorum-imaging-technology

[10] Hologic Clarity HD Imaging Technology: https://www.hologic.com/hologic-products/breast-health/clarity-hd

[11] Dell Precision 7960 Tower Specifications: https://www.delltechnologies.com/asset/en-us/products/workstations/technical-support/precision-7960-tower-spec-sheet.pdf

[12] Dell Precision 7960 Tower Setup and Specifications: https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/processor

[13] Dell Precision 7875 Tower Workstation: https://www.dell.com/en-us/shop/desktop-computers/precision-7875-tower-workstation/spd/precision-t7875-workstation

[14] Dell Precision 7875 Review - PCMag: https://me.pcmag.com/en/old-desktop-pcs/36958/dell-precision-7875-2026

[15] HP Z8 Fury G5 Workstation Specifications: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500

[16] HP Z8 Fury G5 Workstation Review - StorageReview: https://www.storagereview.com/review/hp-z8-fury-g5-workstation-review

[17] HP Z6 G5 A Desktop Workstation: https://www.hp.com/us-en/workstations/z6-a.html

[18] HP Z6 G5 A Performance Brief - Principled Technologies: https://www.principledtechnologies.com/HP/Z6-G5-A-Desktop-Workstation-AMD-Ryzen-performance-0824

[19] HP Z4 G5 Workstation Specifications: https://www.hp.com/us-en/workstations/z4-g5-tower.html

[20] HP Z4 G5 Review - Digital Engineering 24/7: https://www.digitalengineering247.com/article/powerful-but-pricey-hp-z4-g5-workstation

[21] Lenovo ThinkStation PX Specifications - PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_PX/ThinkStation_PX_Spec.pdf

[22] Lenovo ThinkStation PX Review - StorageReview: https://www.storagereview.com/review/lenovo-thinkstation-px-review

[23] Lenovo ThinkStation P7 Specifications - PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P7/ThinkStation_P7_Spec.pdf

[24] Lenovo ThinkStation P7 and PX Review - DEVELOP3D: https://develop3d.com/workstations/review-lenovo-thinkstation-p7-and-px

[25] Lenovo ThinkStation P8 Specifications: https://techtoday.lenovo.com/sites/default/files/2025-05/thinkstation-p8-datasheet-ww-en.pdf

[26] Lenovo ThinkStation P8 Review - DEVELOP3D: https://develop3d.com/workstations/review-lenovo-thinkstation-p8

[27] Lenovo ThinkStation P5 Review - PCMag: https://www.pcmag.com/reviews/lenovo-thinkstation-p5-workstation

[28] Lenovo ThinkStation P5 Datasheet: https://techtoday.lenovo.com/sites/default/files/2025-10/thinkstation-p5-datasheet-ww-en.pdf

[29] AMD Threadripper PRO 7995WX - PassMark: https://www.cpubenchmark.net/cpu.php?id=5726

[30] AMD Threadripper PRO 7975WX - PassMark: https://www.cpubenchmark.net/cpu.php?id=5729

[31] Intel Xeon W9-3495X - PassMark: https://www.cpubenchmark.net/cpu.php?id=5480

[32] Intel Xeon W5-2455X - PassMark: https://www.cpubenchmark.net/cpu.php?id=5604

[33] Threadripper 7995WX vs Xeon W9-3495X Comparison: https://www.cpubenchmark.net/compare/5726vs5480/AMD-Ryzen-Threadripper-PRO-7995WX-vs-Intel-Xeon-w9-3495X

[34] Intel Xeon W-3400 Performance Index: https://edc.intel.com/content/www/us/en/products/performance/benchmarks/intel-xeon-w-3400-processors/

[35] SPECworkstation 3.1 Results Summary: https://www.spec.org/gwpg/wpc.data/specworkstation31_summary.html

[36] Puget Systems - AMD Zen4 Threadripper PRO vs Intel Xeon-w9: https://www.pugetsystems.com/labs/articles/amd-zen4-threadripper-pro-vs-intel-xeon-w9-for-science-engineering-2023/

[37] Dell Precision 7960 Tower - ZWorkstations: https://zworkstations.com/products/dell-precision-7960

[38] Dell Precision 5860 Tower Specifications: https://www.dell.com/en-us/shop/desktop-computers/precision-5860-tower/spd/precision-5860-workstation

[39] Dell Precision 5860 Tower Review - StorageReview: https://www.storagereview.com/review/dell-precision-5860-tower-workstation-review

[40] Lenovo ThinkStation P3 Tower Specifications - PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P3_Tower/ThinkStation_P3_Tower_Spec.pdf

[41] Lenovo ThinkStation P3 Tower Datasheet: https://techtoday.lenovo.com/sites/default/files/2024-05/ThinkStation_P3_Tower_Datasheet.pdf

[42] Dell Precision 7960 Power Supply Options: https://www.dell.com/support/manuals/en-ec/precision-t7960-workstation/precision-t7960-setup-and-specifications/power-ratings

[43] HP Z8 Fury G5 Site Prep Guide: https://assets.grandandtoy.com/assets/Additional-pdf5/10/81/1081618911.pdf

[44] Dell Precision 7960 and 5860 Power Review - HotHardware: https://hothardware.com/reviews/dell-precision-7960-and-5860-review

[45] Arizona Electricity Prices per kWh 2026 - ElectricChoice: https://www.electricchoice.com/electricity-prices-by-state/arizona

[46] Commercial Electricity Costs 2026 - Paradise Solar Energy: https://www.paradisesolarenergy.com/blog/commercial-electricity-costs

[47] APS Rate Schedule E-32 M Medium General Service: https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans/e32_Medium.pdf

[48] APS Rate Schedule E-32 L Large General Service: https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans/e32_Large.pdf

[49] SRP General Service Price Plan: https://www.srpnet.com/price-plans/business-electric/general-service

[50] APS Rate Case 2025 - Proposed 14% Increase: https://www.azfamily.com/2025/06/17/aps-applies-raise-arizona-electricity-rates-by-14-beginning-next-year

[51] Pinnacle West / APS Powering Arizona's Future - March 2026: https://s22.q4cdn.com/464697698/files/doc_events/2026/Mar/02/March-Investor-Deck.pdf

[52] Edged Phoenix Data Center - PUE 1.15: https://edged.us/phoenix

[53] What is PUE in Data Centers - Supermicro: https://www.supermicro.com/en/glossary/pue-for-data-center

[54] Dell Precision 7960 Tower Setup and Specifications (Networking): https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/ports-and-connectors

[55] Dell Precision 5860 Tower Specifications - Dell USA: https://www.dell.com/en-us/shop/desktop-computers/precision-5860-tower/spd/precision-5860-workstation

[56] HP Z Dual-Port 10GbE Network Module: https://h20195.www2.hp.com/v2/getpdf.aspx/c05105339.pdf

[57] HP Z8 Fury G5 Workstation - HP Store: https://www.hp.com/us-en/shop/pdp/hp-z8-fury-g5-tower-workstation-customizable-3f0p6av-mb

[58] HP Z6 G5 Workstation - Superworkstations.com: https://superworkstations.com/products/hp-z6-g5-workstation

[59] NVIDIA RTX Professional GPUs: https://www.nvidia.com/en-us/design-visualization/rtx-pro/

[60] Dell ProSupport Services: https://www.dell.com/en-us/lp/dt/support-services

[61] Dell ProSupport Brochure: https://i.dell.com/sites/content/shared-content/services/ru/Documents/support-services-brochure_ru.pdf

[62] Dell ProSupport Plus for Infrastructure Service Description: https://i.dell.com/sites/csdocuments/Legal_Docs/en/us/dell-prosupport-plus-for-infrastructure-sd-en.pdf

[63] Quadbridge - Dell Authorized Reseller (Avondale, AZ): https://www.quadbridge.com/dell-reseller

[64] iT1 Source - Healthcare Division (Tempe, AZ): https://it1.com/healthcare

[65] HP Support Services for Commercial PCs: https://www.hp.com/us-en/support-services.html

[66] HP Care Pack Services: https://www.hp.com/us-en/carepack-services.html

[67] iT1 Source - HP Inc. Partner: https://it1.com/company/partner-hp-inc

[68] Quadbridge - HP Authorized Reseller: https://www.quadbridge.com/partners/hp

[69] Lenovo Premier Support for Data Centers: https://www.lenovo.com/us/en/services/support-services/premier-support-for-data-centers

[70] Lenovo Premier Support Recognition: https://news.lenovo.com/pressroom/press-releases/why-lenovo-premier-support-is-earning-recognition

[71] Lenovo Warranty and Service: https://support.lenovo.com/us/en/solutions/ht509981

[72] Lenovo Authorized Service Providers: https://support.lenovo.com/us/en/lenovo-service-provider

[73] Dell Precision 7960 Tower Pricing: https://www.dell.com/en-us/shop/desktop-computers/precision-7960-tower-workstation/spd/precision-t7960-workstation

[74] HP Z8 Fury G5 Pricing - PCMag: https://me.pcmag.com/en/old-desktop-pcs/20449/hp-z8-fury-g5

[75] Lenovo ThinkStation P7 - CDW: https://www.cdw.com/product/lenovo-thinkstation-p7-intel-xeon-w5-rtx-a4500-32gb-ecc-ram-512gb-ssd/7437185