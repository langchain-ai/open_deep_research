# Comprehensive Research Report: Workstation Evaluation for Healthcare Imaging Center – Phoenix, AZ

## Executive Summary

This report provides a detailed comparison of current-generation workstation options from Dell Precision, HP Z-series, and Lenovo ThinkStation lines for a healthcare imaging center in Phoenix, Arizona processing approximately 200 patients daily. The workstations must run Philips IntelliSpace PACS and Hologic 3D mammography (SecurView DX) software while handling 500MB+ DICOM files. Based on comprehensive analysis across CPU performance, GPU requirements, ECC memory needs, power consumption, networking, enterprise support availability in Phoenix, and five-year total cost of ownership (TCO), this report recommends the **Dell Precision 5860 Tower** or **Lenovo ThinkStation P5** as the primary candidates for diagnostic workstations, with a strong recommendation for **AMD Threadripper PRO-based systems** (Dell Precision 7875 or HP Z6 G5 A) if maximum multi-threaded performance for 3D mammography reconstruction is prioritized.

---

## 1. Software Application Requirements

### 1.1 Philips IntelliSpace PACS / IntelliSpace Radiology

Philips IntelliSpace PACS provides access to relevant, multimodality information to support clinical decision making. The system has specific hardware requirements that inform workstation selection [1][2]:

**CPU Requirements:**
- Minimum: Intel i5 dual logical processors at 2.5 GHz [2]
- For advanced configurations (Volume Vision, Advanced Mammography): **12 logical processors or more @ 2.5 GHz or higher** with a turbo frequency of at least 3.0 GHz [2][3]
- IntelliSpace Radiology 4.7 specifies that Advanced Mammography requires higher CPU cores (12 logical processors) and notes that Hyper-Threading should be disabled for performance improvements [3]

**RAM Requirements:**
- Minimum: 4 GB [2]
- For advanced configurations: **24 GB or more** [2]
- IntelliSpace Radiology with Advanced Mammography requires 24 GB RAM [3]

**GPU Requirements:**
- Graphics cards must support **OpenGL 3.2** with dedicated memory (1–2 GB) [1][2]
- For advanced configurations: high-end graphics card supporting OpenGL 3.2 with 2 GB memory [2]
- Diagnostic monitors must be DICOM-calibrated; mammography displays require **FDA 510(k) clearance** with at least 5 megapixel resolution [1][4]

**Network Requirements:**
- Minimum: 100 Mb/s adapters with 100 Mb/s end-to-end connection to server [1][2][3]
- For large studies (>700 MB CTs or mammography tomosynthesis): **1 Gb/s network adapter** and 1 Gb/s end-to-end connection to server required [2][3]

**Operating System:** Windows 10 64-bit (Windows 11 is not listed as supported in the most recent publicly available documentation) [3]

**Key Note:** Philips does not publish a formal "certified workstation" list. The specifications state: "The IntelliSpace PACS Enterprise and IntelliSpace PACS Radiology client workstations must use monitors and video cards that have been validated by Philips Healthcare Informatics" [4].

### 1.2 Hologic SecurView DX (3D Mammography Software)

Hologic SecurView DX is described as a **"memory- and computationally-intensive application"** for demanding clinical environments, with optimal performance when installed on a computer dedicated only to Hologic software [5][6][7].

**Latest Version Requirements (SecurView DX v12.0+) [5][6]:**

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel Core i7-6700, 3.40 GHz | Intel Xeon E-2287GE @ 3.31 GHz or better |
| **RAM** | 32 GB | **64 GB** |
| **GPU** | 10-bit video card, 8 GB dedicated VRAM, PCIe 3.0 | 10-bit video card with **8-16 GB dedicated memory**, DirectX 9.0c+, PCIe 3.0+ |
| **Storage** | 2x8TB enterprise class RAID 1, 200 MB/s combined read/write | **8 TB SSD** with 400+ MB/s read/write performance |
| **Network** | Gigabit Ethernet | Gigabit Ethernet (Dual **10 Gigabit NICs recommended** for server connections) |
| **Display** | Dual 5MP FDA-approved mammography monitors | Dual 5MP+ with QC/calibration software |

**GPU Analysis for Hologic Software:**
The requirement for **8-16 GB dedicated video memory, 10-bit color depth, and PCIe 3.0+ interface** effectively mandates a professional-grade discrete GPU. Consumer gaming GPUs typically do not support 10-bit color output reliably for medical diagnostic displays. All NVIDIA Quadro and RTX professional GPUs support 10-bit color output—the key distinction is that Quadro/RTX professional cards support 10-bit color in OpenGL, whereas GeForce cards support 10-bit color only in DirectX [7][8].

NVIDIA's 10-bit and 12-bit grayscale technology transmits higher depth grayscale data over standard DVI and DisplayPort cables. In a preliminary study with 10 radiologists using 10-bit displays in a mammography application, radiologists' performance was statistically significant on 10-bit enabled displays, with some experiencing **triple the read time speedup** [7].

### 1.3 Workload Modeling: 200 Patients/Day

At 200 patients daily with 500MB+ DICOM studies, the imaging center must process **100+ GB of image data per day**. Tomosynthesis mammography studies are significantly larger than 2D mammography (a bilateral screening tomosynthesis exam can be 1-3 GB) [9].

**Network throughput at 1 Gb/s:** A 500 MB DICOM study transfers in approximately 4-8 seconds at 1 Gb/s, which is acceptable for individual reads. However, for multiple concurrent reads and server connections, **10 GbE networking** is recommended for the infrastructure backbone [5][6].

**Storage Architecture Considerations:** Hologic recommends a 2:1 Manager-to-Client HDD capacity ratio, especially when DICOM image data is uncompressed or multiple prior studies per patient are processed [5]. For 200 patients daily, a 15-64 TB storage infrastructure at 700 MB/s combined read/write performance is recommended for the SecurView DX Manager server [5].

The Hologic **SecurView DX Manager** (server component for multiworkstation clusters) requires [5]:
- Windows Server 2016 or 2019 Standard Edition 64-bit
- Intel Xeon Gold 5120T, 2.2 GHz with 14 cores/28 threads
- 64 GB RAM
- 15-64 TB storage at 700 MB/s combined read/write
- 1 GbE minimum, 10 GbE recommended
- VMware ESXi 7.0.x or later supported for virtual deployments

---

## 2. CPU Performance Evaluation

### 2.1 Available Processor Options by Workstation Line

| Workstation Line | CPU Family | Max Cores/Threads | Max Turbo | TDP |
|-----------------|------------|-------------------|-----------|-----|
| **Dell Precision 7960 Tower** | Intel Xeon W-2400/W-3400 | 56/112 (W9-3495X) | 4.80 GHz | 350W (up to 420W) [10] |
| **Dell Precision 7875 Tower** | AMD Ryzen Threadripper PRO 7000 WX | 96/192 (7995WX) | 5.30 GHz | 350W [11] |
| **Dell Precision 5860 Tower** | Intel Xeon W-2400 series only | 24/48 (W7-2495X) | 4.80 GHz | 225W [12] |
| **HP Z8 Fury G5** | Intel Xeon W-3400 | 60/120 (W9-3495X) | 4.80 GHz | 350W [13] |
| **HP Z6 G5 A** | AMD Ryzen Threadripper PRO 7000/9000 WX | 96/192 (9995WX) | 5.40 GHz | 350W [14] |
| **HP Z4 G5** | Intel Xeon W-2400 | 26 cores (W7-2595X) | 4.80 GHz | 225W [15] |
| **Lenovo ThinkStation PX** | Dual Intel Xeon Scalable (4th/5th Gen) | 128 cores (2×64) | 4.20 GHz | 350W per CPU [16] |
| **Lenovo ThinkStation P7** | Intel Xeon W-3400/W-3500 | 60 cores | 4.80 GHz | 350W [17] |
| **Lenovo ThinkStation P8** | AMD Ryzen Threadripper PRO 7000 WX | 96 cores (7995WX) | 5.30 GHz | 350W [18] |
| **Lenovo ThinkStation P5** | Intel Xeon W-2400/W-2500 | 26 cores (W7-2595X) | 4.80 GHz | 225W [19] |

### 2.2 PassMark CPU Benchmark Results

PassMark CPU Mark (multithreaded) and Single Thread Rating provide the most comprehensive available benchmark data for these processors [20][21][22][23]:

**Multi-Threaded Performance (CPU Mark) – Full Ranking:**

| Processor | CPU Mark (MT) | Overall Rank | Server Rank |
|-----------|--------------|--------------|-------------|
| AMD Threadripper PRO 7995WX (96C/192T) | **140,967** | #12 of 5,895 | #10 of 1,251 |
| AMD Threadripper PRO 7985WX (64C/128T) | **131,837** | #17 of 5,895 | – |
| AMD Threadripper PRO 7975WX (32C/64T) | **95,489** | #51 of 5,890 | #46 of 1,251 |
| Intel Xeon W9-3495X (56C/112T) | **90,454** | #58 of 5,890 | #52 of 1,250 |
| AMD Threadripper PRO 7965WX (24C/48T) | **82,173** | #79 of 5,895 | #72 of 1,251 |
| Intel Xeon W7-3465X (28C/56T) | **61,626** | #164 of 5,895 | #141 of 1,251 |
| Intel Xeon W7-2495X (24C/48T) | **57,524** | #193 of 5,890 | – |
| Intel Xeon W5-3435X (16C/32T) | **44,520** | #316 of 5,895 | #235 of 1,251 |
| Intel Xeon W5-2455X (12C/24T) | **37,118** | #401 of 5,895 | #275 of 1,251 |

**Single-Threaded Performance (Single Thread Rating) – Full Ranking:**

| Processor | Single Thread Rating | Overall Rank |
|-----------|---------------------|--------------|
| AMD Threadripper PRO 7965WX (24C/48T) | **4,010** | #207 of 5,895 |
| AMD Threadripper PRO 7975WX (32C/64T) | **3,983** | #218 of 5,890 |
| AMD Threadripper PRO 7985WX (64C/128T) | **3,960** | – |
| AMD Threadripper PRO 7995WX (96C/192T) | **3,829** | #301 of 5,895 |
| Intel Xeon W7-2495X (24C/48T) | **3,455** | #536 of 5,890 |
| Intel Xeon W9-3495X (56C/112T) | **3,430** | – |
| Intel Xeon W5-2455X (12C/24T) | **3,383** | #580 of 5,895 |
| Intel Xeon W5-3435X (16C/32T) | **3,301** | #640 of 5,895 |
| Intel Xeon W7-3465X (28C/56T) | **3,207** | #702 of 5,895 |

### 2.3 Analysis for DICOM Workloads

**Single-Threaded Performance (important for UI responsiveness, DICOM window/level adjustments):**
- AMD Threadripper PRO 7965WX leads with a Single Thread Rating of **4,010**, which is 17% faster than the Intel Xeon W9-3495X (3,430) [20][21]
- Higher single-thread performance translates to faster image manipulation (pan, zoom, window/level) and better overall UI responsiveness in PACS viewers
- Even the 32-core Threadripper PRO 7975WX (3,983) significantly outperforms all Intel Xeon W-series processors in single-threaded tasks

**Multi-Threaded Performance (important for 3D reconstruction, batch processing, AI inference):**
- The AMD Threadripper PRO 7995WX (96-core) dominates with a CPU Mark of **140,967**, outperforming the Intel Xeon W9-3495X by **56%** [20][22]
- For 3D mammography reconstruction and AI-based detection, higher multi-threaded performance directly reduces processing time
- Even the 32-core 7975WX (95,489) slightly edges out the 56-core Intel W9-3495X (90,454) in multi-threaded performance, demonstrating AMD's architectural advantage

**Key Benchmark Context:**
The most recent published SPECworkstation 3.1 results on SPEC.org are dated June 15, 2022, predating all processors in this comparison [24]. No official SPECworkstation 3.1 published results exist for any Xeon W-2400/W-3400 or Threadripper PRO 7000 series processors. SPECviewperf 2020 v3.0 results were last updated November 30, 2023, and do not contain specific entries for these newest processors either [25]. Therefore, PassMark CPU benchmarks represent the most reliable publicly available comparative data.

**Price and Value Comparison [20][21][22][23]:**

| Processor | Approximate Street Price | CPU Mark per Dollar |
|-----------|-------------------------|-------------------|
| AMD TR PRO 7965WX (24C) | ~$2,699 | **30.44** |
| AMD TR PRO 7975WX (32C) | ~$3,800 | **25.13** |
| AMD TR PRO 7985WX (64C) | ~$7,399 | **17.82** |
| AMD TR PRO 7995WX (96C) | ~$8,788 | **16.04** |
| Intel Xeon W5-2455X (12C) | ~$1,149 | **32.30** |
| Intel Xeon W7-2495X (24C) | ~$2,905 | **19.80** |
| Intel Xeon W7-3465X (28C) | ~$3,078 | **20.02** |
| Intel Xeon W9-3495X (56C) | ~$5,889 | **15.36** |

**Recommendation:** For a healthcare imaging center running both Philips IntelliSpace PACS (which benefits from strong single-threaded performance) and Hologic 3D mammography software (which benefits from high multi-threaded throughput for reconstruction), the **AMD Ryzen Threadripper PRO 7975WX (32-core)** offers the best balance—leading single-thread performance (3,983) and strong multi-thread capabilities (95,489) at a reasonable price point ($3,800). The 96-core 7995WX is overkill unless the center performs extensive AI training or batch 3D reconstruction on-site. For a cost-optimized approach, the **Intel Xeon W7-2495X (24-core)** at ~$2,905 provides excellent performance for PACS workloads at a lower price point.

---

## 3. ECC Memory Support

All three workstation lines support DDR5 ECC memory, which is **required** for medical imaging workstations running 24/7 diagnostic operations to prevent single-bit memory errors from corrupting image data [1][2][5]:

| Workstation | Max RAM | DIMM Slots | Memory Channels | Memory Speed |
|------------|---------|------------|-----------------|--------------|
| **Dell Precision 7960 Tower** | Up to **4 TB** DDR5 ECC | 16 | 8-channel (W-3400) / Quad-channel (W-2400) | 4800 MT/s (drops to 4400 MT/s when 12-16 slots populated) [10] |
| **Dell Precision 5860 Tower** | Up to **2 TB** DDR5 ECC | 8 | Quad-channel | 4800 MT/s [12] |
| **Dell Precision 7875 Tower** (AMD) | Up to **2 TB** DDR5 ECC | 8 | **8-channel** | 5200 MT/s [11] |
| **HP Z8 Fury G5** | Up to **2 TB** DDR5 ECC | 16 | 8-channel | 4800 MHz [13] |
| **HP Z6 G5 A** (AMD) | Up to **1 TB** DDR5 ECC | 8 | **8-channel** | 5200 MT/s [14] |
| **HP Z4 G5** | Up to **512 GB** DDR5 ECC | 8 | Quad-channel | 4800 MT/s [15] |
| **Lenovo ThinkStation PX** | Up to **2 TB** DDR5 ECC | 16 (8 per CPU) | 8 per CPU | 4800-5600 MT/s [16] |
| **Lenovo ThinkStation P7** | Up to **1 TB** DDR5 ECC | 8 | **8-channel** | up to 5600 MT/s [17] |
| **Lenovo ThinkStation P5** | Up to **512 GB** DDR5 ECC (1 TB Gen 2) | 8 | Quad-channel | 4800-6400 MT/s [19] |
| **Lenovo ThinkStation P3 Tower** | Up to **128 GB** DDR5 ECC | 4 | Dual-channel | 4400-5600 MT/s [26] |

**Recommended Configuration:** A minimum of **64 GB ECC DDR5** per workstation. Given that Hologic requires 64 GB recommended RAM and Philips recommends 24 GB+ for advanced configurations, 64 GB provides adequate headroom for caching multiple large studies for rapid comparison [2][5]. The 8-channel memory configurations on the Xeon W-3400 and Threadripper PRO platforms provide significantly higher memory bandwidth (up to 2× that of quad-channel), which benefits large DICOM dataset manipulation.

**Important Consideration:** The AMD Threadripper PRO platform (Dell Precision 7875, HP Z6 G5 A, Lenovo ThinkStation P8) uses 8-channel memory at 5200 MT/s, compared to quad-channel at 4800 MT/s for Xeon W-2400-based systems (Dell Precision 5860, HP Z4 G5, Lenovo ThinkStation P5). This provides approximately 73% more memory bandwidth (8×5200 vs 4×4800), which significantly benefits 3D reconstruction and large dataset handling.

---

## 4. Power Consumption and Cost Analysis at Arizona's $0.13/kWh Commercial Rate

### 4.1 Arizona Commercial Electricity Rates and Context

The user's referenced rate of **$0.13/kWh (13¢/kWh)** for Arizona commercial electricity is slightly above the published Arizona commercial average of **11.8¢/kWh** but is within a plausible range depending on the specific utility, rate schedule, and demand charges [27][28].

**Phoenix-Area Utility Context:**

APS (Arizona Public Service) serves the Phoenix area with Rate Schedule E-32 M for medium commercial customers (monthly loads of 101 kW up to 400 kW) [29]:
- Basic Service Charge: $1.286 per day
- Demand Charges (secondary): First 100 kW at $14.690/kW; all additional kW at $8.068/kW
- Energy Charges: $0.11530/kWh summer, with various adjustments
- Power factor requirements: Must maintain 90% lagging below 69 kV

SRP (Salt River Project) serves portions of Phoenix with the E-36 General Service price plan [30]:
- Monthly Service Charge: $15.16
- Demand Charges (over 5 kW): Summer (May, June, Sept, Oct) $4.73/kW; Summer Peak (July, Aug) $7.13/kW; Winter $4.37/kW
- Energy Charges: approximately $0.0759 to $0.1405 per kWh depending on season and usage block
- Temporary price reduction: $0.0038/kWh during May–October 2026 billing cycles

**APS Proposed 14% Rate Increase for 2026 [31][32]:**
- On May 18, 2026, the Arizona Corporation Commission began hearings on APS's request for a 14% rate increase
- Hearings continue through June 30, 2026
- A recommended order is expected in November 2026
- A final vote is anticipated in December 2026
- If approved, higher rates could take effect in early 2027
- For business customers, increases could range from 6.5% to 30% depending on class and usage
- Arizona Attorney General Kris Mayes has argued APS could maintain reliable service with a rate increase closer to 3% instead of 14%

### 4.2 Estimated Power Draw Under Typical and Maximum Loads

Power consumption varies significantly by configuration. Based on TDP data and published specifications [33][34][35]:

| Workstation Configuration | Typical Power Supply | Estimated Typical Draw | Maximum Draw |
|--------------------------|---------------------|----------------------|--------------|
| **Dell Precision 7960** (Xeon W9-3495X + RTX A6000) | 1400W Platinum | **500-700W** | Up to 1200W |
| **Dell Precision 5860** (Xeon W7-2495X + RTX A4500) | 750W/1350W Platinum | **300-450W** | Up to 750W |
| **Dell Precision 7875** (Threadripper 7975WX + RTX A4500) | 1350W Platinum | **400-600W** | Up to 1000W |
| **HP Z8 Fury G5** (Xeon W9-3495X + RTX A6000) | 1125W/2250W Gold | **500-800W** | Up to 1850W |
| **HP Z6 G5 A** (Threadripper 7975WX + RTX A4500) | 1000W-1450W Gold | **400-600W** | Up to 1000W |
| **HP Z4 G5** (Xeon W7-2495X + RTX A4500) | 525W-1125W Gold | **250-400W** | Up to 700W |
| **Lenovo ThinkStation P7** (Xeon W9-3495X + RTX A6000) | 1000W/1400W Platinum | **450-650W** | Up to 1000W |
| **Lenovo ThinkStation P5** (Xeon W7-2495X + RTX A4500) | 750W/1000W Platinum | **300-450W** | Up to 700W |

**CPU TDP Breakdown:**
- Intel Xeon W9-3495X (56-core): 350W TDP, up to 420W maximum turbo power [33]
- Intel Xeon W7-2495X (24-core): 225W TDP [34]
- AMD Threadripper PRO 7975WX (32-core): 350W TDP [35]
- Intel Core i9-14900K (24-core): 125W TDP, up to 253W maximum turbo power [35]

**GPU TDP Breakdown:**
- NVIDIA RTX A4000 (16GB): 140W TDP [36]
- NVIDIA RTX A5000 (24GB): 230W TDP [36]
- NVIDIA RTX 6000 Ada (48GB): 300W TDP [37]

For TCO calculations, this report assumes a **moderate high-performance configuration averaging 400W** per workstation under typical load (mix of reading, processing, and idle periods across a 16-hour workday) for Xeon W-2400-based systems, and **500W** for Threadripper PRO/Xeon W-3400-based systems.

### 4.3 Operating Schedule Clarification

The user specified 16-hour daily operation, 365 days/year but noted actual operation is likely 6-7 days/week. For a healthcare imaging center:

- **Typical outpatient imaging center operation:** 6 days/week (Monday-Saturday) × 52 weeks = **312 days/year**
- **Emergency/on-call coverage:** Some workstations may need to remain available 7 days/week
- **Conservative annual calculation:** 365 days/year for power cost estimation (systems may remain powered on for updates, backups, and remote access even when not actively reading)

This report provides calculations for **both 365-day and 312-day** scenarios.

### 4.4 Annual Power Cost Calculation

**Per Workstation (400W average draw – Xeon W-2400 based):**

| Scenario | Daily Hours | Days/Year | kWh/Year | Cost at $0.13/kWh |
|----------|-------------|-----------|----------|-------------------|
| Typical clinical operation | 16 | 312 | 1,997 | **$259.58** |
| Full-year continuous operation | 16 | 365 | 2,336 | **$303.68** |
| 24/7 operation (servers) | 24 | 365 | 3,504 | **$455.52** |

**Per Workstation (500W average draw – Threadripper PRO / Xeon W-3400 based):**

| Scenario | Daily Hours | Days/Year | kWh/Year | Cost at $0.13/kWh |
|----------|-------------|-----------|----------|-------------------|
| Typical clinical operation | 16 | 312 | 2,496 | **$324.48** |
| Full-year continuous operation | 16 | 365 | 2,920 | **$379.60** |
| 24/7 operation (servers) | 24 | 365 | 4,380 | **$569.40** |

### 4.5 Cooling/HVAC Impact Analysis

Cooling is a significant factor in Phoenix's desert climate. Phoenix has average high temperatures ranging from 66°F in December to 106°F in July, with extreme highs reaching 122°F [38]. The climate is of a desert type with low annual rainfall and low relative humidity, with sunshine averaging 86% annually [38].

**PUE (Power Usage Effectiveness) Estimates for Phoenix:**

For an on-premises server/workstation room in Phoenix:
- Industry average PUE for data centers: 1.58 (Uptime Institute 2023) [39]
- Well-designed facilities: 1.2-1.5
- For a small on-premises IT room in Phoenix desert climate without economizers (since outside air is too hot for much of the year): estimated **PUE of 1.6-2.0** for reasonably designed rooms [39]
- Air-side economizers can cool data centers for only 35% of the year in hot-arid climates [40]
- Mechanical compression cooling must be relied upon for the majority of annual cooling [40]
- This report uses a conservative PUE estimate of **1.8** for a small IT room with standard HVAC in Phoenix

**Cooling Cost Calculation for 12 Workstations (Xeon W-2400 based, 400W each):**

- IT equipment power: 12 × 400W = 4,800W (4.8 kW)
- Total facility power: 4.8 kW × 1.8 PUE = 8.64 kW
- Annual facility electricity (16 hrs/day, 365 days): 8.64 kW × 16 hrs × 365 days × $0.13 = **$6,559.49/year**
- IT equipment electricity cost: 4.8 kW × 16 hrs × 365 days × $0.13 = **$3,644.16/year**
- Cooling overhead: $6,559.49 - $3,644.16 = **$2,915.33/year**

**Cooling Cost Calculation for 12 Workstations (Threadripper PRO based, 500W each):**

- IT equipment power: 12 × 500W = 6,000W (6.0 kW)
- Total facility power: 6.0 kW × 1.8 PUE = 10.8 kW
- Annual facility electricity (16 hrs/day, 365 days): 10.8 kW × 16 hrs × 365 days × $0.13 = **$8,199.36/year**
- IT equipment electricity cost: 6.0 kW × 16 hrs × 365 days × $0.13 = **$4,555.20/year**
- Cooling overhead: $8,199.36 - $4,555.20 = **$3,644.16/year**

### 4.6 5-Year Power Cost Summary

**Xeon W-2400 Based Configuration (400W average):**

| Cost Component | Per Workstation (5-Year) | Total Fleet: 12 Workstations (5-Year) |
|----------------|-------------------------|--------------------------------------|
| IT equipment power (16hrs/365days) | $1,518.40 | $18,220.80 |
| Cooling overhead | $607.36 | $7,288.32 |
| **Total all-in power** | **$2,125.76** | **$25,509.12** |

**Threadripper PRO / Xeon W-3400 Based Configuration (500W average):**

| Cost Component | Per Workstation (5-Year) | Total Fleet: 12 Workstations (5-Year) |
|----------------|-------------------------|--------------------------------------|
| IT equipment power (16hrs/365days) | $1,898.00 | $22,776.00 |
| Cooling overhead | $728.00 | $8,736.00 |
| **Total all-in power** | **$2,626.00** | **$31,512.00** |

**312-Day Scenario (Xeon W-2400 based, 400W average):**

| Cost Component | Per Workstation (5-Year) | Total Fleet: 12 Workstations (5-Year) |
|----------------|-------------------------|--------------------------------------|
| IT equipment power (16hrs/312days) | $1,297.92 | $15,575.04 |
| Cooling overhead | $519.17 | $6,230.02 |
| **Total all-in power** | **$1,817.09** | **$21,805.06** |

*Note: These estimates assume constant $0.13/kWh. Given APS's proposed 14% rate increase for 2026, actual costs may be 5-10% higher over the 5-year period [31][32].*

---

## 5. 10GbE Network Compatibility

### 5.1 Built-in 10GbE Support

| Workstation | Built-in 10GbE | Details |
|-------------|---------------|---------|
| **Dell Precision 7960 Tower** | **Yes** | Dual Ethernet: 1GbE (Intel i219-LM) + 10GbE (Marvell AQC113) standard [10][41] |
| **Dell Precision 5860 Tower** | **Yes** | 1x 1GbE RJ-45 + 1x 10GbE RJ-45 standard [12][42] |
| **Dell Precision 7875 Tower** (AMD) | **Yes** | Dual Ethernet, typical config includes 10GbE [11] |
| **HP Z8 Fury G5** | Add-in required | 1GbE onboard; HP Z Dual-Port 10GbE Network Module available as add-in [13][43] |
| **HP Z6 G5 A** | **Yes** (some configs) | Onboard 10GbE + 1GbE per some configuration guides; 10GbE also available as add-in [14] |
| **HP Z4 G5** | Add-in required | 1GbE standard; optional 10GbE via PCIe add-in card [15] |
| **Lenovo ThinkStation PX** | **Yes** | Dual onboard: 1GbE (Intel I219-LM) + 10GbE (Marvell ACQ-113C) standard [16] |
| **Lenovo ThinkStation P7** | **Yes** | Dual onboard: 1GbE + 10GbE standard [17] |
| **Lenovo ThinkStation P5** | 2.5GbE standard | 1GbE/2.5GbE standard; 10GbE available as optional add-in PCIe card [19] |
| **Lenovo ThinkStation P3 Tower** | Add-in required | 1GbE standard; 10GbE via add-in PCIe card [26] |

### 5.2 Recommendation for 10GbE

All the mid-range and high-end workstations (Dell Precision 7960/7875, HP Z6 G5 A, Lenovo ThinkStation P7/PX) support 10GbE either built-in or via add-in cards. For a facility processing 100+ GB of DICOM data daily:

- **Workstations with built-in 10GbE** are preferred to avoid occupying a PCIe slot and simplify cabling
- **Dual 10GbE for server connections** as recommended by Hologic for SecurView DX Manager [5][6]
- Consider a 10GbE switch (e.g., 12-24 port) to interconnect workstations and the PACS server

**10GbE Network Component Costs [44][45][46]:**

| Component | Estimated Cost |
|-----------|---------------|
| Intel X550-T2 Dual Port 10GbE NIC (add-in card) | ~$475 |
| Managed 10GbE Switch (12-port SFP+/RJ45) | ~$286-3,143 depending on brand |
| Cat6a cabling (per 1,000 ft) | ~$350-500 |
| Cat6a installation (per drop) | ~$200-350 |

The **Dell Precision 5860 Tower** and **Lenovo ThinkStation P5** (with add-in card) are convenient options with native 10GbE or simple add-in capability. The Lenovo ThinkStation P5 requires a 10GbE add-in card (~$475) but offers excellent value overall.

---

## 6. GPU Requirements

### 6.1 Philips IntelliSpace PACS GPU Needs

Philips IntelliSpace PACS requires **OpenGL 3.2 support** with **1-2 GB VRAM** [1][2]. For:
- **Basic 2D PACS viewing:** Integrated graphics that support OpenGL 3.2 may suffice for non-diagnostic viewing
- **Advanced visualization modules** (Volume Vision, Advanced Mammography): A **dedicated discrete GPU** is required [2]
- **Diagnostic mammography displays:** Require FDA-approved 5MP monitors connected to a GPU capable of driving them at full resolution and color depth

### 6.2 Hologic SecurView DX GPU Needs

Hologic's requirements are significantly more demanding [5][6]:
- **8-16 GB dedicated video memory**
- **10-bit color depth support** (consumer GPUs typically do not support this for medical displays)
- **DirectX 9.0c or higher** and **OpenGL 2.0** (MultiView MM)
- **PCIe 3.0 or higher** interface

These requirements **effectively mandate a professional-grade discrete GPU**. All NVIDIA Quadro and RTX professional GPUs support 10-bit color output in both OpenGL and DirectX, whereas GeForce cards support 10-bit color only in DirectX [7][8].

### 6.3 Recommended GPU Configuration

**NVIDIA Professional GPU Options [36][37][47][48]:**

| GPU | VRAM | TDP | Estimated Price | 10-bit Color | Suitability |
|-----|------|-----|----------------|--------------|-------------|
| **NVIDIA RTX A2000 12GB** | 12GB GDDR6 | 70W | ~$700 | Yes | Marginal (below 8-16GB recommendation) |
| **NVIDIA RTX A4000 16GB** | 16GB GDDR6 ECC | 140W | ~$1,290 | Yes | **Meets minimum Hologic requirement** |
| **NVIDIA RTX A4500 20GB** | 20GB GDDR6 ECC | 200W | ~$2,199 | Yes | **Recommended – good balance** |
| **NVIDIA RTX A5000 24GB** | 24GB GDDR6 ECC | 230W | ~$2,200-4,000 | Yes | Excellent – headroom for future |
| **NVIDIA RTX 6000 Ada 48GB** | 48GB GDDR6 ECC | 300W | ~$7,500 | Yes | Overkill for current needs |

**AMD Radeon Pro Options [49]:**

| GPU | VRAM | TDP | Estimated Price | Notes |
|-----|------|-----|----------------|-------|
| **Radeon Pro W7600** | 8GB GDDR6 | 130W | ~$615 | Insufficient VRAM for Hologic |
| **Radeon Pro W7800** | 32GB GDDR6 | 200W | ~$2,499 | Good option for multi-modal |
| **Radeon Pro W7900** | 48GB GDDR6 | 295W | ~$3,999 | Excellent, comparable to RTX 6000 Ada |

**Recommended Configuration:**

| Use Case | Recommended GPU | VRAM | Rationale |
|----------|----------------|------|-----------|
| **Primary diagnostic workstation** (mammography + PACS) | NVIDIA RTX A4500 20GB | 20 GB | Meets Hologic 8-16GB requirement, supports 10-bit color, cost-effective at ~$2,199 |
| **Clinical review workstation** (PACS only) | NVIDIA RTX A4000 16GB | 16 GB | Sufficient for PACS viewing and basic 3D |
| **High-end / future-proof** | NVIDIA RTX A5000 24GB | 24 GB | Additional VRAM for future AI workloads |
| **AI/image processing server** | NVIDIA RTX 6000 Ada 48GB | 48 GB | For batch reconstruction and AI inference |

Barco MXRT series graphics cards are specifically optimized for PACS applications and deliver accurate, consistent image quality with seamless integration with Barco medical displays [50]. However, they are typically more expensive than comparable NVIDIA RTX professional GPUs.

---

## 7. Enterprise Support Options in Phoenix Metro Area

### 7.1 Dell ProSupport and ProSupport Plus

**Dell ProSupport (Standard Premium Support) [51][52]:**
- 24x7 access to Dell Customer Service
- Onsite technician dispatch based on purchased service levels (4-hour, same day, next business day)
- Access to SupportAssist software for proactive device monitoring using AI/ML
- Collaborative assistance for eligible third-party products
- Comprehensive software support for select Dell OEM applications

**Dell ProSupport Plus (Highest Tier) [51][53]:**
- Everything in ProSupport plus:
- **Priority 24/7 access** to Senior ProSupport Engineers
- **Accidental Damage Service** – covers drops, spills, surges with no deductible
- **Hard Drive Retention** – you keep your drive if replacement is needed
- **Proactive monitoring** via SupportAssist with predictive hardware failure detection
- **Technical Customer Success Manager (CSM)** for lifecycle management, success planning, and escalation assistance
- **4-hour onsite response** objective (within 50 miles of designated support hubs)

**Availability in Phoenix Metro:** Standard ProSupport and ProSupport Plus are available in the Phoenix metropolitan area. Dell provides NBD onsite service across the contiguous US, including Phoenix metro. The 4-hour mission-critical option is typically available in major metro areas including Phoenix. Authorized local service providers include **Quadbridge (Avondale, AZ)** and **iT1 Source (Tempe, AZ)** [54][55].

### 7.2 HP Premium Support and Care Packs

**HP Active Care / Premium Support (Base Tier) [56][57]:**
- 24/7 remote technical support
- Onsite repair services (Next Business Day or faster)
- Basic device health monitoring
- Coverage durations from 3 to 5 years

**HP Premium Support (Mid Tier) [56][57]:**
- All Essential features plus:
- Predictive analytics and AI-powered alerts via HP Workforce Experience Platform (WXP)
- Faster response times
- Automated ticketing for issue resolution

**HP Premium+ Support (Top Tier) [56][58]:**
- All Premium features plus:
- **AI-powered predictive issue detection** and proactive expert support
- **Preferred access** to HP expert support with fastest response
- **Out-of-Band Diagnosis and Remediation** technology – enables remote issue fixing even if the PC won't boot
- Device health data compared against billions of global HP data points using AI
- **Accidental Damage Protection** (optional)

**Availability in Phoenix Metro:** HP has authorized service providers in the Phoenix area. **iT1 Source (Tempe, AZ)** is an HP authorized reseller with a dedicated healthcare division [59]. **Quadbridge (Avondale, AZ)** is also an authorized HP reseller [60].

### 7.3 Lenovo Premier Support and Premier Support Plus

**Lenovo Premier Support [61][62]:**
- Direct access to elite Lenovo engineers
- Dedicated Technical Account Managers
- Prioritized onsite labor and parts service
- Next-business-day onsite service
- Available as 1-year to 5-year extended service agreements

**Lenovo Premier Support Plus [61][63]:**
- Everything in Premier Support plus:
- **24x7x365 advanced technical support** with break/fix services
- **Proactive and predictive issue detection** powered by AI insights from Lenovo Device Intelligence
- **Accidental Damage Protection (ADP)** – covers spills and mishaps, reduces device repair costs by up to 80%
- **Keep Your Drive (KYD)** – data security option to retain drives
- **Sealed Battery (SBTY)** coverage for up to 3 years
- **Services Engagement Manager (SEM)** for escalations and asset performance reporting

**Availability in Phoenix Metro:** Lenovo has authorized service providers in the Phoenix/Tempe area. **iT1 Source (Tempe, AZ)** is a confirmed Lenovo partner [64]. **Quadbridge (Avondale, AZ)** is a **Lenovo Platinum Partner** and authorized reseller [65].

### 7.4 Local Enterprise Resellers (Phoenix Metro Area)

| Reseller | Location | Dell | HP | Lenovo | Healthcare Focus |
|----------|----------|------|----|--------|-----------------|
| **iT1 Source** | 1860 W University Drive, Suite 100, Tempe, AZ 85281 | ✓ Authorized Partner | ✓ HP Inc. Partner | ✓ Lenovo Partner | **Dedicated healthcare division**, serves 3,000+ active accounts, CRN Tech Elite 250 [54][59][64] |
| **Quadbridge** | 1060 N Eliseo Felix Jr Way, #107, Avondale, AZ 85323 | **Dell Platinum Partner** | ✓ Authorized Reseller | **Lenovo Platinum Partner** | Multi-brand IT solutions, managed services, founded 2007, multiple Growth 500 awards [55][60][65] |

Both resellers have physical offices in the Phoenix metro area (Maricopa County), are authorized for all three major workstation brands, and have the scale to support enterprise healthcare deployments. **iT1 Source** has a dedicated healthcare division and is specifically noted for serving the healthcare sector [54].

---

## 8. 5-Year Total Cost of Ownership (TCO) Analysis for 12 Workstations

### 8.1 Assumptions and Methodology

- **Deployment:** 12 workstations
- **Operating period:** 16 hours/day, 365 days/year (conservative – also provided for 312-day scenario)
- **Electricity rate:** $0.13/kWh (blended including demand charges)
- **PUE:** 1.8 (Phoenix IT room with standard HVAC)
- **Workstation recommendation:** Mid-range configuration (Intel Xeon W7-2495X or AMD Threadripper PRO 7975WX, 64GB ECC, professional GPU)
- **Display cost:** Dual 5MP FDA-approved mammography monitors
- **Support:** 5-year extended warranty/support included

### 8.2 Hardware Acquisition Cost Estimates

| Workstation Model | Recommended Configuration | Estimated Cost per Unit | 12-Unit Fleet Cost |
|-------------------|--------------------------|------------------------|-------------------|
| **Dell Precision 5860 Tower** | Xeon W7-2495X (24C), 64GB DDR5 ECC, RTX A4500 20GB, 1TB NVMe, 10GbE | **~$9,500** | **$114,000** |
| **Dell Precision 7960 Tower** | Xeon W9-3495X (56C), 64GB DDR5 ECC, RTX A4500 20GB | **~$14,000** | **$168,000** |
| **Dell Precision 7875 Tower** (AMD) | Threadripper PRO 7975WX (32C), 64GB DDR5 ECC, RTX A4500 20GB | **~$13,000** | **$156,000** |
| **HP Z4 G5** | Xeon W7-2495X (24C), 64GB DDR5 ECC, RTX A4500 20GB | **~$9,000** | **$108,000** |
| **HP Z6 G5 A** (AMD) | Threadripper PRO 7975WX (32C), 64GB DDR5 ECC, RTX A4500 20GB | **~$12,500** | **$150,000** |
| **Lenovo ThinkStation P5** | Xeon W7-2495X (24C), 64GB DDR5 ECC, RTX A4500 20GB, 10GbE add-in | **~$8,500** | **$102,000** |
| **Lenovo ThinkStation P7** | Xeon W9-3495X (56C), 64GB DDR5 ECC, RTX A4500 20GB | **~$13,500** | **$162,000** |

*Pricing estimates are based on CDW, monitors.com, and manufacturer retail pricing as of May 2026 [66][67][68]. Actual enterprise pricing through authorized resellers may be 10-20% lower.*

### 8.3 5-Year TCO Breakdown (Per Workstation, Mid-Range Configuration)

| Cost Category | Dell Precision 5860 | HP Z4 G5 | Lenovo ThinkStation P5 |
|--------------|--------------------|----------|----------------------|
| **Hardware** | $9,500 | $9,000 | $8,500 |
| **5-Year Support (ProSupport Plus / Premium+ / Premier Plus)** | $2,500 | $2,500 | $2,500 |
| **IT Power (5-yr, 400W, 365 days)** | $1,518 | $1,518 | $1,518 |
| **Cooling Overhead (5-yr)** | $607 | $607 | $607 |
| **GPU Mid-Cycle Replacement (year 3)** | $2,200 | $2,200 | $2,200 |
| **Networking (10GbE NIC + switch share)** | $250 | $250 | $400 |
| **Displays (dual 5MP FDA-approved)** | $10,000 | $10,000 | $10,000 |
| **UPS/Backup Power** | $450 | $450 | $450 |
| **IT Staff Time (configuration, deployment, mgmt)** | $1,000 | $1,000 | $1,000 |
| **Total Per Workstation (5-Year)** | **$28,025** | **$27,525** | **$27,175** |
| **Total Fleet: 12 Units (5-Year)** | **$336,300** | **$330,300** | **$326,100** |

### 8.4 Additional TCO Factors

**Storage Infrastructure:**
- A shared PACS server or NAS with redundant SSDs (20-40 TB usable) for image caching: approximately **$15,000-25,000** for the fleet
- Hologic SecurView DX Manager server (if needed): **$20,000-35,000** for a dedicated server meeting requirements (Xeon Gold, 64GB RAM, 15-64TB storage)

**Network Infrastructure:**
- 10GbE managed switch (12-24 port): **$300-3,500**
- Cat6a cabling installation: **$200-350 per drop**, 12+ drops: **$2,400-4,200**

**UPS/Backup Power:**
- Per-workstation UPS (APC Smart-UPS 1500VA): **~$900 each**, 12 units: **$10,800** [69]
- Facility-level UPS for server room: **$3,000-5,000**

**IT Staff Time:**
- Configuration, deployment, and ongoing management: approximately 20-40 hours per workstation over 5 years for imaging-specific software setup
- Healthcare IT staffing benchmark: approximately 1 IT staff per 50-100 devices [70]
- For a 12-workstation deployment: ~0.12-0.24 FTE, valued at **$5,000-15,000** over 5 years

**Software Licensing:**
- Windows 11 Pro for Workstations (often included in hardware cost)
- PACS client licenses (often per-seat, may be included in Philips agreement)

**Compliance Costs (HIPAA/FDA):**
- HIPAA compliance costs for mid-size organizations: $80,000-450,000 initial, $90,000-500,000 per year ongoing [71]
- For a small imaging center, allocated workstation compliance costs: approximately **$3,000-8,000** over 5 years

### 8.5 Fully Loaded 5-Year TCO Estimate (12 Workstations, Mid-Range Configuration)

| Category | Low Estimate | High Estimate | Notes |
|----------|-------------|---------------|-------|
| Workstations (12 units) | $102,000 | $114,000 | Lenovo P5 / Dell 5860 range |
| 5-Year Support | $24,000 | $30,000 | Premium tier, 4-hour onsite |
| Power + Cooling (5 years, 365 days) | $25,509 | $25,509 | 12 units at 400W, PUE 1.8 |
| Displays (12 × dual 5MP FDA-approved) | $72,000 | $144,000 | Barco MDNC-6121 at $13,999/unit dual config vs Eizo RX560 |
| GPU Mid-Cycle Replacement (year 3) | $26,400 | $26,400 | 12 × RTX A4500 at ~$2,200 |
| Storage Infrastructure | $15,000 | $35,000 | PACS server/NAS |
| Network Infrastructure | $3,000 | $8,000 | Switch, cabling, NICs |
| UPS/Battery Backup | $10,800 | $16,000 | Per-workstation + facility UPS |
| IT Staff Time | $5,000 | $15,000 | Configuration, deployment, mgmt |
| Compliance/Security (HIPAA allocation) | $3,000 | $8,000 | Per-workstation share |
| **Total Fleet TCO (5-Year)** | **$286,709** | **$422,418** | |
| **Per Workstation per Year** | **$4,778** | **$7,040** | |

### 8.6 Display Cost Detail – Dual 5MP FDA-Approved Mammography Monitors

**Barco Nio MDNC-6121 (5.8MP Color Digital Mammography) [72]:**
- Starting at **$13,999.99** per unit from monitors.com
- Dual configuration: approximately **$28,000** per workstation
- Features I-Guard™ sensor for continuous DICOM calibration, Uniform Luminance Technology (ULT), SpotView™, 3-year premium warranty
- 5.8 megapixels (2100 × 2800 pixels), calibrated luminance of 600 cd/m² (max 1300 cd/m²)
- Consumes 60W power

**Eizo RadiForce RX560 (5MP Mammography) [73]:**
- Estimated pricing: approximately **$8,000-10,000** per unit (exact MSRP not publicly listed)
- Dual configuration: approximately **$16,000-20,000** per workstation
- Features LTPS LCD technology, 1100 cd/m² brightness, 1500:1 contrast ratio, 7.5mm bezel
- FDA 510(k) clearance for breast tomosynthesis and mammography
- Hybrid Gamma PXL function for pixel-by-pixel grayscale optimization
- 10-bit input support for over 1 billion colors
- Five-year warranty

For cost-optimized deployments, the **Eizo RX560** at approximately $16,000-20,000 per workstation (dual) offers substantial savings over the Barco MDNC-6121 at approximately $28,000 per workstation (dual).

---

## 9. Recommended Configurations

### 9.1 Primary Recommendation: Dell Precision 5860 Tower or Lenovo ThinkStation P5

For a healthcare imaging center running **Philips IntelliSpace PACS and Hologic SecurView DX 12.0+** serving 200 patients/day:

**Recommended Configuration (Primary – Intel Xeon W-2400 based):**

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Workstation** | Dell Precision 5860 Tower or Lenovo ThinkStation P5 | Excellent balance of performance, expandability, and cost |
| **CPU** | Intel Xeon W7-2495X (24 cores, 48 threads, up to 4.8 GHz) | Strong single-threaded for PACS UI, good multi-threaded for 3D reconstruction, 225W TDP for lower power/cooling costs |
| **RAM** | 64 GB DDR5-4800 ECC (4×16 GB or 2×32 GB) | Meets Hologic recommended 64GB, provides headroom |
| **GPU** | NVIDIA RTX A4500 20GB | Meets Hologic 8-16GB VRAM requirement, supports 10-bit color, cost-effective at ~$2,200 |
| **Storage** | 1 TB NVMe PCIe Gen 4 (OS/apps) + 4 TB SSD (local image cache) | Fast boot and application loading with sufficient local cache |
| **Network** | Built-in 10GbE (Dell Precision 5860) or add-in Intel X550-T2 (Lenovo P5) | 10GbE for fast DICOM transfer |
| **Displays** | Dual Eizo RadiForce RX560 5MP or Barco MDNC-6121 | FDA-approved for mammography |
| **OS** | Windows 10 Enterprise 64-bit | Compatible with both Philips and Hologic requirements |

**Total Estimated Cost per Workstation:** ~$9,500-10,500 (excluding displays)

### 9.2 High-Performance Alternative: Dell Precision 7875 or HP Z6 G5 A (AMD Threadripper PRO)

For maximum 3D reconstruction performance and future AI workloads:

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Workstation** | Dell Precision 7875 Tower or HP Z6 G5 A | AMD Threadripper PRO platform with 8-channel memory |
| **CPU** | AMD Ryzen Threadripper PRO 7975WX (32 cores, 64 threads, up to 5.3 GHz) | Best single-threaded performance (3,983) + excellent multi-threaded (95,489) at a reasonable $3,800 |
| **RAM** | 64 GB DDR5-5200 ECC (8×8 GB to utilize 8 channels) | 8-channel memory provides ~73% more bandwidth than quad-channel |
| **GPU** | NVIDIA RTX A5000 24GB | Additional VRAM for 3D reconstruction and future AI |
| **Storage** | 1 TB NVMe PCIe Gen 5 + 4 TB NVMe PCIe Gen 4 | Maximum storage performance |
| **Network** | Built-in 10GbE | Both models include 10GbE |
| **Displays** | Dual Eizo RadiForce RX560 5MP | Cost-effective FDA-approved option |

**Total Estimated Cost per Workstation:** ~$15,000-16,500 (excluding displays)

### 9.3 Cost-Optimized Alternative: Lenovo ThinkStation P5

For facilities where budget is a primary concern:

| Component | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Workstation** | Lenovo ThinkStation P5 | Starting at ~$3,289 base, excellent price/performance |
| **CPU** | Intel Xeon W7-2495X (24 core) or W5-2455X (12 core) | W7-2495X for balanced performance, W5-2455X for lower cost |
| **RAM** | 64 GB DDR5-4800 ECC | 8 DIMM slots, expandable to 512 GB |
| **GPU** | NVIDIA RTX A4000 16GB | Meets minimum Hologic 8GB requirement at ~$1,290 |
| **Storage** | 1 TB NVMe + 4 TB SATA SSD | Cost-effective storage with good performance |
| **Network** | Intel X550-T2 10GbE add-in card (~$475) | Single PCIe slot for 10GbE capability |
| **Displays** | Dual Eizo RadiForce RX560 5MP | Most cost-effective FDA-approved option |

**Total Estimated Cost per Workstation:** ~$7,500-8,500 (excluding displays)

### 9.4 Additional Infrastructure Recommendations

**Hologic SecurView DX Manager Server (for multiworkstation clusters) [5]:**
- Windows Server 2019 Standard Edition 64-bit
- Intel Xeon Gold 5120T (14 cores/28 threads) or equivalent
- 64 GB RAM
- 15-64 TB storage at 700 MB/s combined read/write
- 10 GbE networking (dual NICs recommended)

**PACS Server / Storage:**
- NAS or SAN with 20-40 TB usable capacity
- SSD tier for hot data (active studies) + HDD tier for archive
- RAID 10 or RAID 6 with hot spare
- 10 GbE connectivity

**Network:**
- Managed 10GbE switch (12-24 port SFP+/RJ45)
- Cat6a cabling to all workstations
- Dual 10GbE NICs for servers

---

## 10. Phoenix-Specific Considerations

### 10.1 Local Utility Context

**APS Commercial Rate Structure (Rate Schedule E-32 M) [29]:**
- Energy charges: approximately $0.11530/kWh summer (May-October), lower in winter
- Demand charges: $14.690/kW for first 100 kW, $8.068/kW for additional kW
- Additional adjustments: Renewable Energy Adjustment, Power Supply Adjustment, Transmission Cost Adjustment, Demand Side Management Adjustment
- For a 12-workstation fleet drawing ~4.8 kW IT load, the demand charges add significantly to the bill

**SRP General Service (E-36) [30]:**
- Energy charges: approximately $0.0759 to $0.1405 per kWh depending on season
- Demand charges: Summer $4.73/kW, Summer Peak (July-August) $7.13/kW, Winter $4.37/kW
- SRP is a not-for-profit utility with lower rates than APS
- Temporary price reduction of $0.0038/kWh for May-October 2026 billing cycles

**2026 Rate Increase Impact [31][32]:**
- APS proposed 14% increase being heard May-June 2026
- If approved, blended rates would rise from ~$0.115/kWh to ~$0.131/kWh
- For business customers, increases could range from 6.5% to 30%
- For the calculated 5-year TCO, a 14% increase would add approximately $3,500 to $4,400 to total fleet power costs over 5 years

### 10.2 Cooling Challenges in Desert Climate

Phoenix's desert climate presents unique cooling challenges [38]:
- Average highs exceed 100°F from June through September
- Average first 100°F day occurs around May 2
- Temperatures remain above 100°F until late September
- Extreme high: 122°F
- Monsoon season: June 15 to September 30 with high humidity

**Cooling Recommendations for Phoenix:**
- **Hot/cold aisle containment** is critical to prevent mixing of hot and cold air streams [39]
- **High-SEER rating cooling units** are recommended for dedicated server rooms
- **Air-side economizers** have limited viability (only 35% of the year usable in hot-arid climates) [40]
- **Waterside economizers** have even less viability (12% of the year)
- **Mechanical compression cooling** must be relied upon for the majority of annual cooling
- **Liquid cooling** offers 60-80% energy savings over traditional air cooling for high-density configurations
- Consider a dedicated mini-split AC system for the IT room rather than relying on building HVAC

**PUE Target for Phoenix IT Room:**
- With hot/cold aisle containment and efficient cooling: 1.5-1.6
- With standard HVAC without containment: 1.8-2.0
- The Edged Phoenix Data Center achieves PUE of 1.15 with waterless cooling, but this is a hyperscale facility [74]

### 10.3 Regulatory Environment

- Arizona eliminated the Renewable Energy Standard and Tariff (REST) in 2025, but no state-level healthcare IT infrastructure mandates currently affect workstation specifications [27]
- HIPAA compliance is federal and applies regardless of state
- The 21st Century Cures Act requires interoperability and information blocking prevention, which may affect PACS/workstation integration choices
- Arizona does not have specific state-level data privacy laws beyond HIPAA for healthcare

### 10.4 Local Support Ecosystem

The Phoenix metro area has a robust enterprise IT support ecosystem:
- **iT1 Source** (Tempe, AZ): Established 2003, global technology solutions provider, serves 3,000+ active accounts, healthcare division, PACE Purchasing Cooperative contract through December 31, 2026 [54][59][64]
- **Quadbridge** (Avondale, AZ): Established 2007, Dell Platinum Partner, Lenovo Platinum Partner, HP authorized reseller, recognized four times in the Growth 500 list [55][60][65]
- Both resellers have physical presence in Maricopa County and can provide onsite support within 4-hour response windows

---

## Sources

[1] Philips IntelliSpace PACS 4.4 Spec Sheet: https://images.philips.com/is/content/PhilipsConsumer/Campaigns/HC20140401_DG/Documents/452299109121_IntelliSpace_PACSR44_SpecSheet_FNL_LR2.pdf

[2] Philips IntelliSpace PACS Client Specs 4.4.551.0 (eHealth Saskatchewan): https://www.ehealthsask.ca/services/PACS/Documents/IntelliSpace%20PACS_Client%20Specs_4.4.551.0%202018.pdf

[3] Philips IntelliSpace Radiology 4.7 Delivery Notes: https://www.documents.philips.com/assets/Instruction%20for%20Use/20251128/ca90732d7c72423493f3b3a3005c6c76.pdf

[4] Philips IntelliSpace Radiology 4.7 Client Specs: https://www.ehealthsask.ca/services/PACS/Documents/EI_IntelliSpace%20Radiology%20%204.7%20Client%20Specs_FNL_.pdf

[5] Hologic SecurView DX Minimum System Requirements v12.0+: https://www.hologic.com/file/477126/download?token=3OBC61Ej

[6] Hologic SecurView DX Minimum System Requirements v12.0+ (alternate): https://www.hologic.com/file/405841/download?token=LHEqk8Qa

[7] NVIDIA 10 and 12-bit Grayscale Technology Technical Brief: https://www.nvidia.com/docs/IO/40049/Grayscale10bit_v03.pdf

[8] NVIDIA Healthcare Graphics Configurations: https://www.nvidia.com/content/quadro_oem/presentations/Vertical_Industry_-_Healthcare.pdf

[9] Hologic 3DQuorum Imaging Technology: https://www.hologic.com/hologic-products/breast-health/3dquorum-imaging-technology

[10] Dell Precision 7960 Tower Setup and Specifications: https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/processor

[11] Dell Precision 7875 Tower Workstation: https://www.dell.com/en-us/shop/desktop-computers/precision-7875-tower-workstation/spd/precision-t7875-workstation

[12] Dell Precision 5860 Tower Specifications: https://www.dell.com/en-us/shop/desktop-computers/precision-5860-tower/spd/precision-5860-workstation

[13] HP Z8 Fury G5 Workstation Specifications: https://h20195.www2.hp.com/v2/GetDocument.aspx?docname=c08481500

[14] HP Z6 G5 A Desktop Workstation: https://www.hp.com/us-en/workstations/z6-a.html

[15] HP Z4 G5 Workstation Specifications: https://support.hp.com/us-en/document/ish_7759887-7759932-16

[16] Lenovo ThinkStation PX PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_PX/ThinkStation_PX_Spec.pdf

[17] Lenovo ThinkStation P7 Specifications: https://thinkstation-specs.com/thinkstation/p7

[18] Lenovo ThinkStation P8 Datasheet: https://techtoday.lenovo.com/sites/default/files/2025-05/thinkstation-p8-datasheet-ww-en.pdf

[19] Lenovo ThinkStation P5 PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P5/ThinkStation_P5_Spec.pdf

[20] AMD Ryzen Threadripper PRO 7995WX PassMark: https://www.cpubenchmark.net/cpu.php?id=5726

[21] AMD Ryzen Threadripper PRO 7975WX PassMark: https://www.cpubenchmark.net/cpu.php?id=5729

[22] Intel Xeon W9-3495X PassMark: https://www.cpubenchmark.net/cpu.php?id=5480

[23] Intel Xeon W7-2495X PassMark: https://www.cpubenchmark.net/cpu.php?id=5326

[24] SPECworkstation 3.1 Results Summary: https://www.spec.org/gwpg/wpc.data/specworkstation31_summary.html

[25] SPECviewperf 2020 V3 Results: https://www.spec.org/gwpg/gpc.data/viewperf2020v3/summary.html

[26] Lenovo ThinkStation P3 Tower PSREF: https://psref.lenovo.com/syspool/Sys/PDF/ThinkStation/ThinkStation_P3_Tower/ThinkStation_P3_Tower_Spec.pdf

[27] Arizona Electricity Prices 2026 - ElectricChoice: https://www.electricchoice.com/electricity-prices-by-state/arizona

[28] Commercial Electricity Costs Arizona - Paradise Solar Energy: https://www.paradisesolarenergy.com/blog/commercial-electricity-costs

[29] APS Rate Schedule E-32 M Medium General Service: https://www.aps.com/-/media/APS/APSCOM-PDFs/Utility/Regulatory-and-Legal/Regulatory-Plan-Details-Tariffs/Business/Business-NonResidential-Plans/e32_Medium.pdf

[30] SRP General Service Price Plan E-36: https://www.srpnet.com/price-plans/business-electric/general-service

[31] APS Rate Case 2025 Update: https://www.aps.com/en/Utility/Regulatory-and-Legal/Rate-case

[32] Arizona Capitol Times - APS Rate Case Hearings: https://azcapitoltimes.com/news/2026/05/19/aps-rate-case-kicks-off-with-hours-of-protest-over-14-rate-increase

[33] Intel Xeon W9-3495X TechPowerUp: https://www.techpowerup.com/cpu-specs/xeon-w9-3495x.c2964

[34] Intel Xeon W7-2495X TechPowerUp: https://www.techpowerup.com/cpu-specs/xeon-w7-2495x.c2933

[35] AMD Ryzen Threadripper PRO 7975WX TechPowerUp: https://www.techpowerup.com/cpu-specs/ryzen-threadripper-pro-7975wx.c3298

[36] NVIDIA RTX A4000 TechPowerUp: https://www.techpowerup.com/gpu-specs/rtx-a4000.c3728

[37] NVIDIA RTX 6000 Ada Generation TechPowerUp: https://www.techpowerup.com/gpu-specs/rtx-6000-ada-generation.c3949

[38] US Climate Data - Phoenix: https://www.usclimatedata.com/climate/phoenix/arizona/united-states/usaz0166

[39] Uptime Institute Global Data Center Survey: https://uptimeinstitute.com/global-data-center-survey

[40] ScienceDirect - Hot-Arid Climate Data Center Cooling: https://www.sciencedirect.com/topics/engineering/data-center-cooling

[41] Dell Precision 7960 Tower Ports and Connectors: https://www.dell.com/support/manuals/en-us/precision-t7960-workstation/precision-t7960-setup-and-specifications/external-ports-and-slots

[42] Dell Precision 5860 Tower - ZWorkstations: https://zworkstations.com/products/dell-precision-5860

[43] HP Z8 Fury G5 Workstation - HP Store: https://www.hp.com/us-en/shop/pdp/hp-z8-fury-g5-tower-workstation-customizable-3f0p6av-mb

[44] Intel X550-T2 Dual Port 10GbE NIC - OnLogic: https://www.onlogic.com/product/ethernet-intel-x550-t2-dual-port-10gbe-nic

[45] YuLinca 12-Port 10G Managed Switch - Amazon: https://www.amazon.com/YuLinca-Managed-Switching-Support-Configuration/dp/B0CQ8H5P1J

[46] Cat6a Cabling Cost Guide - ICTAlly: https://www.ictally.com/blog/cat6a-cabling-cost

[47] NVIDIA RTX A4500 20GB - Newegg: https://www.newegg.com/p/1FT-003F-00010

[48] NVIDIA RTX A5000 24GB Pricing - ThunderCompute: https://thundercompute.com/gpu-pricing

[49] AMD Radeon Pro W7900 - B&H Photo: https://www.bhphotovideo.com/c/product/1760605-REG/amd_100_300001_01_radeon_pro_w7900_48gb.html

[50] Barco MXRT Graphics Cards: https://www.monitors.com/blogs/radiology-display-news/advantages-of-using-barco-mxrt-graphics-cards-with-barco-medical-displays

[51] Dell ProSupport for Client Products Service Description: https://i.dell.com/sites/csdocuments/legal_docs/en/us/prosupport-for-client-sd-en.pdf

[52] Dell ProSupport Plus FAQ: https://www.dell.com/support/kbdoc/en-us/000200482/prosupport-plus-and-prosupport-flex-subscription-services-frequently-asked-questions

[53] Dell ProSupport Plus for Infrastructure Service Description: https://i.dell.com/sites/csdocuments/Legal_Docs/en/us/dell-prosupport-plus-for-infrastructure-sd-en.pdf

[54] iT1 Source - CloudTango Dell Partner Profile: https://www.cloudtango.net/dell/arizona

[55] Quadbridge Dell Reseller Page: https://www.quadbridge.com/dell-reseller

[56] HP Premium Plus Support: https://www.hp.com/us-en/services/workforce-solutions/workforce-computing/support/premium-plus.html

[57] HP Active Care / Premium+ Support: https://www.hp.com/ca-en/services/workforce-solutions/workforce-computing/support/premium-plus.html

[58] HP Care Pack Services: https://www.hp.com/us-en/services/consumer/carepack-pc.html

[59] iT1 Source - HP Inc. Partner: https://it1.com/company/partner-hp-inc

[60] Quadbridge HP Authorized Reseller: https://www.quadbridge.com/partners/hp

[61] Lenovo Premier Support Plus Brochure: https://www.lenovo.com/content/dam/lenovo/ssg/global/english/services/support/support/premier-support-plus/customer/brochure/premier-support-plus_brochure_ww_en.pdf

[62] Lenovo Premier Support: https://www.lenovo.com/us/en/services/support-services/premier-support-for-data-centers

[63] Lenovo Premier Support Plus - SHI: https://www.shi.com/product/46227501/Lenovo-Premier-Support-Plus

[64] iT1 Source - Lenovo Partner: https://it1.com/company/partner-lenovo

[65] Quadbridge Lenovo Partner: https://www.quadbridge.com/partners/lenovo

[66] Dell Precision 5860 Tower - CDW: https://www.cdw.com/search/?q=precision+5860

[67] HP Z4 G5 Workstation - CDW: https://www.cdw.com/product/hp-z4-g5-workstation/45993379

[68] Lenovo ThinkStation P5 - CDW: https://www.cdw.com/product/lenovo-thinkstation-p5/7437185

[69] APC Smart-UPS 1500VA - Staples: https://www.staples.com/APC-Smart-UPS-1500VA-LCD-USB-Serial-230V/product_24581640

[70] IT Staffing Ratios in Healthcare: https://www.gartner.com/en/documents/3986243

[71] HIPAA Compliance Cost Guide: https://www.accountablehq.com/post/hipaa-compliance-cost

[72] Barco Nio MDNC-6121 5MP Mammography Monitor: https://www.monitors.com/products/barco-mdnc-6121-5mp-color-digital-mammography-monitor

[73] Eizo RadiForce RX560 5MP Mammography Monitor: https://www.eizoglobal.com/products/radiforce/rx560/index.html

[74] Edged Phoenix Data Center: https://edged.us/phoenix