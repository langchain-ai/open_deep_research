# Medium Format Camera Comparison for Commercial Fashion Photography: Fujifilm GFX100 II vs Hasselblad X2D II 100C vs Phase One XF IQ4 150MP

**Revised and Updated Report — May 2026**

This report provides a comprehensive, corrected comparison of three medium format camera systems for professional commercial fashion photography in New York City. It addresses studio strobe sync reliability, tethered shooting performance, color science accuracy, file workflow speed, lens ecosystem costs, and total system investment over three years. All corrections requested in the research brief have been incorporated, with new product releases and updated market data as of May 31, 2026.

---

## 1. Studio Strobes Sync Reliability

### 1.1 Fujifilm GFX100 II

The GFX100 II uses a focal-plane shutter with a **native flash sync speed of 1/125 second** on the mechanical shutter. This is confirmed across official Fujifilm specifications and multiple review sources. [1][2]

**Electronic Shutter Limitations**: The electronic shutter is not suitable for flash photography. In 16-bit capture mode, the sensor readout transit time is approximately 358 milliseconds (over 1/3 second), and in 14-bit mode it is approximately 167 milliseconds. This rolling shutter readout prevents simultaneous sensor exposure required for flash. [3][4]

**High-Speed Sync (HSS)**: The camera does support HSS when used with compatible triggers and strobes. HSS enables flash use above 1/125s by pulsing the flash throughout the shutter curtain travel, but this comes with a **loss of approximately three stops of flash power** compared to standard sync. Fujifilm's own HSS tutorial confirms this power penalty. [5]

**Leaf Shutter Lens Adapter**: There are no native Fujifilm GF mount leaf shutter lenses. However, Fujifilm offers a **GF-to-Hasselblad H adapter** that accepts leaf shutter lenses from the Hasselblad H system. When set to lens shutter mode, this adapter permits sync speeds of approximately **1/800–1/1000 second** without power loss. The adapter does not support autofocus, adding workflow complexity. The Hasselblad H system is effectively discontinued, meaning its lenses have depreciated substantially (20–30% of new price), making them cost-effective. [6]

**Trigger Compatibility**:
- **Profoto**: Fully compatible. The Profoto Air Remote TTL-F provides TTL and HSS functionality. Fstoppers demonstrated using the GFX100 II with Profoto B10 strobes for commercial fashion work. [7]
- **Godox**: A major known issue with Godox triggers causing EVF blackout was **fixed by Fujifilm firmware update v1.10**. After this update, Godox triggers work flawlessly with remote strobes. However, some reports indicate Godox triggers may limit burst frame rates and cannot achieve 8 fps continuous shooting when attached. [8][9]
- **Elinchrom**: The Elinchrom Transmitter PRO for Fujifilm cameras provides full compatibility with TTL and HSS support. [10]

### 1.2 Hasselblad X2D II 100C and X2D 100C

Both Hasselblad X2D models rely entirely on **leaf shutters built into each XCD lens**. There is no focal-plane shutter in the body. The electronic shutter cannot fire flash at all — this is a physical limitation of the rolling readout, not a firmware issue. [11][12]

**Flash Sync Speeds by Lens**:
- **XCD V-Series Lenses** (25V, 38V, 55V, 90V): The upgraded compact leaf shutters capture at up to **1/4000s** with full flash synchronization at all shutter speeds. The XCD 25V "adopts a large-diameter leaf shutter module with a shutter speed of up to 1/4000s and supports flash synchronization at all shutter speeds." [13][14]
- **XCD 80mm f/1.9**: Features an integral central shutter offering exposure times from 60 minutes to **1/2000s** with full flash synchronization. [15]
- **Original XCD Lenses**: Most have built-in leaf shutters, with maximum sync speeds varying between 1/2000s and 1/4000s depending on the lens.

**No HSS Penalty**: Because leaf shutters sync at full power at every speed up to their mechanical limit, there is no power loss. This is ideal for fashion photographers who need to overpower ambient light at wide apertures. [11]

**Profoto B4 Firmware Issue — Unresolved**:
**As of May 2026, Profoto firmware version B4 (released March 11, 2026) breaks TTL on Hasselblad cameras.** The only confirmed workaround is staying on pre-B4 firmware. The X2D II's hot shoe uses the Nikon i-TTL protocol, and Profoto historically supported Hasselblad via Nikon-compatible units. Users are advised not to upgrade their Profoto gear to B4 firmware if intending to use TTL with Hasselblad. Profoto has not released a fix as of this research. [11][16]

**Trigger Compatibility**:
- **Nikon Protocol**: The hot shoe uses Nikon i-TTL protocol. Officially compatible flashes include Nikon SB-300, SB-500, SB-700, SB-5000, and Profoto A10, A1, and triggers (pre-B4 firmware). [11]
- **Godox**: Community reports indicate Godox Nikon-version flashes can provide TTL on-camera when mounted directly. However, **Godox wireless TTL triggers do not support Hasselblad**. Godox flashes and triggers only support manual flash (or TTL if mounted directly via hot shoe). [11]
- **PocketWizard and Studio Strobes**: Manual triggering via the center sync pin works, benefiting from the leaf shutter's high flash sync speeds of 1/2000s to 1/4000s. [11][12]

### 1.3 Phase One XF IQ4 150MP

The Phase One XF camera body has a focal-plane shutter with a maximum flash sync speed of **1/125 second**. However, the system's primary strength comes from **Schneider Kreuznach "Blue Ring" leaf shutter lenses** that provide flash sync at shutter speeds from 1/125 up to **1/1600 seconds** at every speed increment. This enables precise ambient light control, particularly for balancing strobes against strong sunlight. [17][18]

**The X-Shutter Backup**: The X-Shutter is an electromagnetic leaf shutter with carbon fiber blades, offering shutter speeds from 1/1000 second to 60 minutes, durability tested to over 500,000 captures. While primarily featured in the XT and XC camera bodies (not the XF), the X-Shutter can sync strobe at every possible speed increment. [19]

**Electronic Shutter Flash Sync Limitations**: The purely electronic shutter on the IQ4 digital back cannot sync strobes at normal shutter speeds. Specific sync times for the IQ4-150 are:
- **L16EX (16-bit Extended)**: 1 second sync
- **L16 (16-bit)**: 1/2 second sync
- **L14 (14-bit)**: 1/4 second sync

These are not practical for fashion work. [20]

**Flash Analysis Tool**: A unique built-in tool that provides real-time data on sync timing, flash duration, and flash pulse intensity during exposure. This is invaluable for troubleshooting sync issues in mission-critical commercial shoots. Additional XF tools allow review of flash duration, output power, and sync performance. [21][22]

**Trigger Compatibility**:
- **Profoto AirTTL**: The XF system includes integrated Profoto Air wireless flash triggering. The internal flash transmitter syncs using the FAST Profoto triggering protocol by default. Tests reveal reliable performance up to approximately 230 feet, comparable to Profoto Air Remotes at moderate distances. [17]
- **Profoto B1 Strobes**: Flash duration (T.05) around 1/1000 second at full power. Noticeable exposure decline begins at shutter speeds faster than 1/320 second, with about a 1/2 stop loss by 1/500 second. Using a Profoto B4 pack with Pro-7 heads provides 1/5000 second flash duration and minimal exposure loss up to 1/1600 second. [17]
- **Broncolor Senso**: Compatible but slower flash durations (Pulso G heads) result in greater exposure falloff at high shutter speeds. [17]

### 1.4 Sync Reliability Summary

| System | Native Sync Speed | HSS Capability | Key Issues |
|--------|------------------|----------------|------------|
| **Fujifilm GFX100 II** | 1/125s (mechanical) | HSS up to 1/4000s with ~3 stop loss; leaf adapter 1/800-1/1000s (manual focus only) | Godox limits burst rate; no native leaf shutter lenses |
| **Hasselblad X2D II** | 1/2000-1/4000s (leaf shutter, lens-dependent) | Not needed (leaf shutter syncs at all speeds) | Profoto B4 firmware broke TTL (unfixed); Godox wireless TTL unsupported |
| **Phase One XF IQ4** | 1/1600s (leaf shutter lenses) | Not needed (leaf shutter syncs at all speeds) | No HSS if faster sync needed; large system; high cost |

---

## 2. Tethered Shooting Performance

### 2.1 Fujifilm GFX100 II

**Capture One Pro Support**: The GFX100 II is fully supported in Capture One Pro with official RAW support. Capture One is widely recommended by users as a stable tethering solution for Fujifilm cameras. Tether Tools explicitly lists Capture One Pro as compatible software for the GFX100 II. [23][24]

**Connection Options**: USB-C 3.2 Gen 2 (10 Gbps), Gigabit Ethernet via built-in LAN port, and wireless (WiFi) tethering. For stable tethering over longer distances, Tether Tools recommends using the TetherPro USB 3.0 SuperSpeed Active Extension Cable and the TetherBoost Pro Core Controller for cables over 15 feet. [24]

**Stability Reports**: Experiences vary significantly. Some users report excellent stability with Capture One Pro 22 (perpetual license), noting it "saves to both Capture One and the camera's SD card." However, other users report constant and irregular connection drops with the GFX100 II and Capture One 23 on Macs. GFX100S II users have reported the tethering "drops constantly. Sometimes after 20 minutes, sometimes after the first shot." [25][26]

**Fujifilm Tether Plugin**: Fujifilm provides the "FUJIFILM Tether Plugin PRO for GFX" for Adobe Photoshop Lightroom Classic (version 1.20.0 adds GFX100 II compatibility). A key limitation is that Lightroom Classic does not currently save images to both the camera's memory card and the computer simultaneously during tethering. [27][28]

**Recommended Setup**: For the most reliable tethering, users recommend using Capture One Pro over a wired Ethernet connection (which provides greater stability than USB-C), with the camera set to USB mode "COMM only" rather than "CARD READER." Mac users should ensure explicit USB permissions are granted to avoid connection conflicts. [25]

### 2.2 Hasselblad X2D II 100C and X2D 100C

**Critical Limitation — Capture One Does NOT Support Hasselblad**:
**As of May 2026, Capture One still does not support any Hasselblad camera.** This has not changed, and the situation appears permanent. The Capture One ideas portal feature request for Hasselblad X2D II 100C support has been marked **"Unlikely to implement"** and merged into another existing idea. [29][30]

Capture One's official support article states: "Support for the Hasselblad X2D 100C is **not currently planned** in Capture One. This status reflects both technical considerations and the nature of..." The lack of support stems from historical and corporate reasons involving prior associations between Phase One and Hasselblad. Users have expressed strong frustration on the ideas portal: [29][31]

- "Dear Mr. Rafael Orta, it's time for a truce – the war should be over. You could make many customers happy if you finally offered support for Hasselblad cameras in Capture One." [30]
- "Today Hasselblad is owned by DJI and C1 is owned by Axcel... The days of the 'walled garden' approach is dead." [30]

**Available Tethering Solutions**:
- **Hasselblad Phocus** (PC/Mac): Free software that fully supports tethered shooting for both X2D models. Phocus applies HNCS color processing and supports full RAW conversion. Version 4.1.2 includes adjustment layers, Keystone Perspective Correction, and focus bracketing support for tethered capture. [32][33]
- **Phocus Mobile 2** (iOS): Supports wireless transfer, remote activation, and RAW HDR image editing. Firmware update 3.1.0 for the X2D 100C added live view tethered shooting with Phocus apps. [34]
- **Adobe Lightroom**: Supports Hasselblad RAW files (Adobe has no special partnership restrictions).

**Tethering Reliability**: Mixed reports. Some users find Phocus "just as reliable as using my Sony and Capture One (once you've gone through the silly connection steps)." Others document "unreliable connections, slow buffers, and cables that don't fit." Known issues include white balance resetting to manual based on the last imported image. A community-tested solution is the Lolo Boo Tethering cable (5-meter, 10 Gbps), priced at approximately $69, which consistently works well with large 100MP files. [35][36]

### 2.3 Phase One XF IQ4 150MP

**Best-in-Class — Capture One Inside**: The IQ4 features "Capture One Inside" — the core imaging processor from Capture One is embedded directly into the digital back. This enables onboard RAW file processing, real-time application of Capture One Styles, and in-camera preview of user-defined looks. This integration provides the **most efficient and controlled capture workflow** available in any medium format system. [37][38][39]

**Three Tethering Options**:
- **USB-C (USB 3.1 Gen 2, 10 Gbps)**: High speed but recommended cable length limited to approximately 3 meters. Can be powered by external USB-C power banks in the field. [40]
- **Gigabit Ethernet with Power over Ethernet (PoE)**: The most stable option. Supports cable lengths over 100 feet with continuous power delivery over a single CAT6 cable. PoE is particularly valuable for all-day studio sessions — the camera stays powered and connected. [38][40]
- **WiFi 802.11ac**: Supports ad-hoc mode eliminating the need for a router, though with slower transfer speeds and some Live View limitations during file transfer. [40]

**Stability and Reliability**: Industry standard for reliability. The IQ4 platform is built on a Linux-based operating system with 10x the processing power of the IQ3 series. The camera comes with a 5-year warranty and uptime guarantee. Facebook community reports consistently note: "Phase One cables work better for me. I never have any connection issues." The dual storage (XQD + SD) provides automatic backup. [37][40][41]

### 2.4 Tethered Shooting Summary

| System | Capture One Support | Tethering Methods | Stability Rating |
|--------|-------------------|-------------------|-----------------|
| **Fujifilm GFX100 II** | Supported (official RAW support) | USB-C, Ethernet, WiFi | Good (variable — some users report drops) |
| **Hasselblad X2D II** | **NOT supported** — must use Phocus | USB-C, WiFi (Phocus), USB-C (Phocus Mobile 2) | Good (Phocus) but no Capture One integration |
| **Phase One XF IQ4** | Native integration (Capture One Inside) | USB-C, Ethernet (PoE), WiFi | Industry standard — most reliable |

---

## 3. Color Science Accuracy for Skin Tones

### 3.1 Fujifilm GFX100 II

**Film Simulations**: The GFX100 II features Fujifilm's renowned film simulation technology, including the **Reala Ace** simulation introduced with this camera. Reala Ace is widely praised as "one of the best Fujifilm Film Simulators so far." It combines faithful color reproduction with hard tonality, sitting tonally between Provia and Pro Neg Hi with a significant gain in overall contrast. For skin tones, Reala Ace produces **more realistic and pleasing results with reduced red saturation**, eliminating the need for additional color adjustments in post-processing. [42][43][44]

Justin Myers, a photographer who has used Fujifilm films since the 1990s, states: "Reala Ace combined with the GFX100 II is one of the most film-like experiences I've found in digital. It has been hitting it out of the park on every test/photo project so far." He praises the color accuracy, especially in rendering blues and skin tones. [42]

**Real-World Skin Tone Quality**: The Phoblographer's review states photos "look perfect straight from the camera without editing" and highlights the GFX100 II as "a worthwhile investment for high-end portraits, commercial work." A Reddit user notes: "The skin tones roll off into the shadows more like film than digital. It's made my retouching time drop by half." [45][46]

**Known Color Challenges**: Some users report issues "when shooting indoor and with different color sources. The images don't look good. The colors skins are..." suggesting challenges with mixed lighting. The system may require more post-processing than Hasselblad or Phase One to achieve optimal skin tone reproduction across a wide range of ethnicities. Professional photographer Jose Gerardo Palma (who uses GFX systems) notes that while built-in profiles are appealing, he "prefers to have a perfect starting point (like you get with Hasselblad or Phase One)." He warns that using X-Rite ColorChecker in Capture One can introduce unacceptable contrast and black level issues. [47][48]

**Bayer Sensor Advantage**: The GFX100 II uses a Bayer sensor (rather than X-Trans), which contributes to better compatibility with third-party RAW processing software and more predictable color rendering in commercial workflows. [44]

### 3.2 Hasselblad X2D II 100C and X2D 100C

**Hasselblad Natural Colour Solution (HNCS)** is widely regarded as the **industry leader for color science**. Digital Camera World states: "Hasselblad Natural Colour Solution is the best I've ever seen. Surpasses other brands known for their color accuracy." [49][50]

HNCS is a color management system independently developed by Hasselblad that "empowers cameras to render genuine, true-to-life colours with smooth, detailed transitions." Every Hasselblad medium-format camera undergoes stringent pixel-level calibration combined with high-quality lenses. HNCS includes an independently developed look-up-table (LUT), Film Curve, and unique color processing that "adapts to any illumination to maintain true contrast, rich saturation, and smooth tonal transitions, **especially in skin tones**." It removes the need for choosing specific color profiles, providing consistent, accurate colors across all lighting conditions. [49]

**16-Bit Color Depth**: The X2D delivers true 16-bit color depth capable of representing "over 281 trillion colors," enabling natural tonal transitions and exceptionally smooth gradations in skin tones. The 16-bit pipeline feeds into "tonality, transition and contrast — which imbues images with depth, dimension and 'pop'." [50][51]

**Real-World Comparison to Fujifilm**: A photographer who used both systems for a year summarizes: "The Hasselblad X2D is my heart. The Fujifilm GFX 100 II is my hustle." He describes the GFX100 II files as "more clinical and requiring more post-processing work to achieve the same emotional impact as the Hasselblad." The Hasselblad files are so forgiving that "editing a RAW file isn't correction; it's revelation." [52]

**X2D II HDR Enhancement**: The X2D II 100C introduces the **industry's first end-to-end HDR photography workflow** built on the HNCS HDR system, allowing photographers to capture single-exposure HDR images compatible with both SDR and HDR displays. This provides even greater tonal range for skin tones. [53]

**The "Hasselblad Look"**: Characterized by rich colours, excellent contrast, and exceptional clarity. "Where most of us, using most other cameras, need to crank clarity and slide S-curves and fiddle and faff with our files to turn them into something interesting, Hasselblad bodies have that cinematic look straight out of camera." Some users note the system has a "definitive muted look" — this is by design, as HNCS prioritizes accuracy over immediate visual pop. [50][54]

### 3.3 Phase One XF IQ4 150MP

**Reference Standard for Color Accuracy**: Phase One's color science is built around Capture One's processing engine. The IQ4's 151MP BSI sensor with 16-bit files and 15 stops of dynamic range provides the maximum color information available in any digital camera system. [55][56]

The **IQ3 100MP Trichromatic Digital Back** (and the IQ4 100MP Trichromatic) is specifically noted for exceptional color science: "The IQ3 100MP Trichromatic truly renders colors in a far more natural way, straight out of the camera" and "renders color in astonishing brilliance, capturing the most color nuances in the most natural way possible." Photographer Ausra Babiedaite notes: "Capture One is extremely powerful when it comes to color, the Trichromatic gave me the best possible starting point and the Skin Tone tool saved me many hours in Photoshop." [57]

**Capture One Inside for Color**: A unique advantage is the ability to **load user-defined Capture One Styles into the camera** for in-camera previews at capture time. This means fashion photographers can apply consistent branded looks and skin tone treatments immediately at capture time, ensuring the tethered preview matches the final output. This is unique to Phase One and enables personalized color workflows starting from the moment of exposure. [37][38]

**BSI Sensor Color Advantages**: The Back-side Illuminated (BSI) sensor design with ultra-efficient pixels results in more accurate color, detail, and noise handling compared to previous generations. The sensor delivers 16-bit Opticolor+ image processing. [55][58]

**Professional Retoucher Perspective**: Jose Palma (who uses GFX systems) explicitly notes that Hasselblad and Phase One provide "a perfect starting point" out of camera, implying Phase One color science is the reference standard for accuracy. The Phase One system is designed for photographers who need "a camera that was custom-built for me" — commercial photographers who cannot afford color inaccuracies. [48][59]

### 3.4 Color Science Summary

| System | Color Philosophy | Best for Skin Tones | Custom Profile Needed? |
|--------|------------------|--------------------|------------------------|
| **Fujifilm GFX100 II** | Film simulation-based, pleasing out of camera | Good, especially with Reala Ace / Astia; may need more post for diverse skin tones | Recommended for critical work; X-Rite ColorChecker can introduce issues in Capture One |
| **Hasselblad X2D II** | HNCS — accuracy and natural rendering, industry best | Excellent across all skin tones with minimal adjustment; 16-bit pipeline | Usually not needed — HNCS provides reference-quality color out of camera |
| **Phase One XF IQ4** | Reference standard with Capture One Inside | Maximum data and dynamic range; fully customizable with styles | Optional — can load custom Capture One styles into camera for instant preview |

---

## 4. File Workflow Speed

### 4.1 Fujifilm GFX100 II

**RAW File Sizes — 16-bit vs 14-bit**: The GFX100 II offers both **true 16-bit and 14-bit lossless compressed RAW capture** options. File sizes from real-world testing on the GFX100/GFX100S (same sensor generation) are: [60][61]

- **14-bit lossless compressed RAW**: Average **97.0 MB** (~54% storage savings vs uncompressed)
- **16-bit lossless compressed RAW**: Average **124.15 MB** (~41% storage savings vs uncompressed)
- These figures come from 4,811 real-world images captured over 2.5 weeks of field shooting.

**Continuous Shooting**: Up to **8 fps** with the mechanical shutter. In CH (continuous high) mode, the camera captures 8 fps but the mechanical shutter scan rate achieves speeds faster than 1/120 second with 12-bit ADC precision behavior. In blackout-free electronic shutter mode, the camera captures at **5.3 fps**. In cropped 35mm mode, the camera captures at up to 8.7 fps. [62][63]

**Practical Recommendation**: Use **14-bit lossless compressed** for fashion sessions. This provides lossless quality at approximately 100MB per file, with no quality difference from 16-bit in most real-world scenarios. Jim Kasson's testing showed that on-sensor phase-detection autofocus (OSPDAF) banding makes any subtle differences between 14-bit and 16-bit read noise quality inconsequential for most shooting. [61]

**Buffer Depth**: The camera can capture **"over 300 RAW images at full speed"** (8 fps) before buffering — described as "a significant achievement for a 102MP sensor." [63]

**Card Support**: Dual card slots — one CFexpress Type B (compatible with CFexpress 4.0) and one UHS-II SD. Two card slots provide backup capability.

**Import Workflow**: With 100+ files at approximately 100MB each, expect approximately 10+ GB of data per session. Capture One performance depends on hardware — recommended specs include 32GB RAM, a high-core CPU, and a fast SSD. The camera also supports tethered shooting via Capture One Pro, which can provide direct-to-hard-drive saving.

### 4.2 Hasselblad X2D II 100C and X2D 100C

**Unique Advantage — Built-in 1TB SSD**: Both X2D models contain a **built-in 1TB SSD with write speeds up to 2370 MB/s and read speeds up to 2850 MB/s**. This is first in its class for medium format and effectively eliminates buffer clearing as a bottleneck — the camera can write 16-bit RAW images continuously without slowdown for a typical session. The 1TB SSD stores approximately 4,600 RAW photos or 13,800 JPEGs. [64][65]

**RAW File Sizes**: Native 3FR files (16-bit) are approximately **200 MB each**. When imported into Phocus for processing, files convert to 3FFF format that can reach significantly larger sizes. The 16-bit files offer 281 trillion colors. [65]

**Continuous Shooting**:
- **X2D 100C**: **3.3 frames per second** (some sources state 3.0 fps depending on configuration). In continuous drive mode, the bit depth automatically drops to **14-bit** — the camera provides no warning and no option to override. In single-shot mode, 16-bit is available. [66][67]
- **X2D II 100C**: **3 frames per second** in mechanical and electronic shutter modes. [68]

**Card Support**: CFexpress Type B slot (additional to internal SSD). High-speed cards recommended for optimal performance. The internal SSD makes "memory cards are optional but not necessary." [64]

**Workflow Limitation**: The camera does not support Capture One tethered live view, requiring Phocus for tethered RAW conversion. This adds a workflow step for photographers who prefer Capture One's interface. However, Hasselblad RAW files are fully supported in Adobe Lightroom and Photoshop.

### 4.3 Phase One XF IQ4 150MP

**RAW File Sizes**: The 151MP BSI CMOS sensor produces 16-bit RAW files approximately **240 MB each** in IIQ L 16 format. Active pixels are 14204 x 10652. Various IIQ formats are supported to balance quality and file size: IIQ L 16 Extended, IIQ L 16, IIQ L, IIQ S, IIQ S 14+. [55][56]

**Continuous Shooting**:
- **14-bit IIQ compressed**: **1.4 fps** (can maintain for up to 44 shots before buffer fills)
- **16-bit IIQ L (highest quality)**: **0.7 fps**
- Various IIQ file formats affect shooting speeds: 0.7 to 1.3 fps depending on compression format selected. [40]

This is the slowest shooting speed of the three systems — the XF IQ4 is designed for deliberate, single-shot capture, not rapid-fire sequences.

**Buffer Depth**: The IQ4 system has **8GB of RAM** (compared to 2GB in the IQ3), allowing up to 44 shots at 14-bit compressed before the buffer fills. At 16-bit L16 quality, the camera captures at 0.7 fps with no buffer limitation. [69][70]

**Storage**: Dual card slots — **XQD** (with CFexpress support via Feature Update #8) and **SD** cards. Configurable for primary/backup/overflow modes. [71]

**Workflow Advantage — Capture One Inside**: Despite the slow capture speed, each file goes directly into the Capture One RAW processing pipeline with no conversion delay. Files are immediately ready for editing with full camera-specific color profiles. The camera can also apply user-defined Capture One Styles at capture time, reducing post-processing time. The IQ4 supports Automated Frame Averaging (long exposure without filters), Dual Exposure+ (capturing two images at different ISOs in one shutter press), and ETTR (Expose to the Right). [37][38]

### 4.4 File Workflow Speed Summary

| System | Max FPS | File Size (per RAW) | Buffer | Best For |
|--------|---------|---------------------|--------|----------|
| **Fujifilm GFX100 II** | 8 fps (mechanical), 5.3 fps (electronic, 16-bit) | ~100 MB (14-bit lossless compressed), ~124 MB (16-bit lossless compressed) | 300+ frames at 8 fps | High-volume sessions, rapid sequences |
| **Hasselblad X2D II** | 3 fps (drops to 14-bit in continuous) | ~200 MB (3FR, 16-bit) | Unlimited (1TB internal SSD at 2370 MB/s) | Deliberate, quality-focused capture |
| **Phase One XF IQ4** | 1.4 fps (14-bit compressed), 0.7 fps (16-bit) | ~240 MB (IIQ L 16) | 44 shots at 14-bit; limited by processing speed | Maximum quality, single-shot capture |

---

## 5. Lens Ecosystem Costs

### 5.1 Sensor Sizes and Crop Factors

Before comparing lenses, the correct crop factors must be established:

| System | Sensor Dimensions | Diagonal | Crop Factor | Area vs FF |
|--------|------------------|----------|-------------|------------|
| **Full-Frame 35mm** | 36 × 24 mm | 43.27 mm | 1.0x | — |
| **Fujifilm GFX / Hasselblad XCD** | 43.8 × 32.9 mm | 54.78 mm | **0.79x** | ~1.7x FF |
| **Phase One 645** | 53.4 × 40.0 mm | 66.72 mm | **0.65x** | ~2.5x FF |

**Phase One sensor dimensions corrected**: The official Phase One specifications confirm the sensor measures **53.4 × 40.0 mm**, NOT 53.7 × 40.2 mm as previously reported. This is documented in the Phase One IQ Camera System Overview Flyer and XC Camera Tech Spec PDF. [72][73]

**Crop Factor Calculations**: 
- GFX/Hasselblad: 43.27 / 54.78 = **0.79x** (confirmed by multiple sources: CameraDecision, Tonal Photo, Shutter Muse) [74][75][76]
- Phase One: 43.27 / 66.72 = **0.65x** (confirmed by Phase One's own specification that the 80mm LS f/2.8 MkII is equivalent to 50mm on full-frame, giving a crop factor of 50/80 = 0.625x; the exact mathematical crop factor is 0.6485x, rounded to 0.65x) [77]

### 5.2 Corrected Focal Length Equivalents

| Camera System | Lens | Calculation | Full-Frame Equivalent |
|---------------|------|-------------|-----------------------|
| **Fujifilm GFX** | 30mm × 0.79 | 30 × 0.79 = 23.7mm | **~24mm** |
| | 55mm × 0.79 | 55 × 0.79 = 43.5mm | **~44mm** |
| | 63mm × 0.79 | 63 × 0.79 = 49.8mm | **~50mm** |
| | 110mm × 0.79 | 110 × 0.79 = 86.9mm | **~87mm** |
| **Hasselblad XCD** | 25V × 0.79 | 25 × 0.79 = 19.75mm | **~20mm** |
| | 55V × 0.79 | 55 × 0.79 = 43.5mm | **~44mm** |
| | 90V × 0.79 | 90 × 0.79 = 71.1mm | **~71mm** |
| | 80mm f/1.9 × 0.79 | 80 × 0.79 = 63.2mm | **~63mm** |
| **Phase One 645** | 35mm × 0.65 | 35 × 0.65 = 22.75mm | **~23mm** |
| | 55mm × 0.65 | 55 × 0.65 = 35.75mm | **~36mm** |
| | 80mm × 0.65 | 80 × 0.65 = 52mm | **50mm (official)** |
| | 110mm × 0.65 | 110 × 0.65 = 71.5mm | **~72mm** |

**Correction note**: The previous report listed the GFX 30mm as ≈28mm FF equivalent. The correct equivalent is **~24mm**. The XCD 25V was listed as ≈20mm FF equivalent, which is approximately correct (19.75mm). The Phase One 35mm was listed as ≈22mm FF equivalent, which is approximately correct (22.75mm).

### 5.3 Fujifilm GFX Lenses (Current US Prices as of May 2026)

| Lens | Price | Full-Frame Equivalent | Notes |
|------|-------|----------------------|-------|
| **GF 30mm f/3.5 R WR** | **$1,699.00** [78] | ~24mm | Wide-angle, 510g, weather-sealed |
| **GF 55mm f/1.7 R WR** | **$2,099.00** (sale, regular $2,599.00) [78][79] | ~44mm | Fast standard prime, $500 savings promotion |
| **GF 63mm f/2.8 R WR** | **$1,699.00** [80] | ~50mm | Lighter alternative, 405g |
| **GF 110mm f/2 R LM WR** | **$2,699.00** (sale, regular $2,954.95) [78][81] | ~87mm | Portrait telephoto, widely reviewed as "the sharpest lens" ever tested |

**Total for three-lens kit:**
- 30mm + 55mm + 110mm = **$6,497**
- 30mm + 63mm + 110mm = **$5,997**

**Newer Lens Options**: Fujifilm has released the **GF32-90mm T3.5 PZ OIS WR** — the first power zoom lens in the GF range, originally developed for the GFX ETERNA 55 cinema camera. Firmware version 2.50 for the GFX100 II (released March 26, 2026) adds full compatibility. The **GFX100RF** (fixed-lens compact) launched in March 2025 at $4,899. Fujifilm currently has 23 native GF lenses available, and the 2026 system guide notes, "GF lenses are not about collecting focal lengths. They're about choosing a small number of lenses that you'll trust for a decade." [82][83]

### 5.4 Hasselblad XCD Lenses (Current US Prices as of May 2026)

| Lens | Price | Full-Frame Equivalent | Notes |
|------|-------|----------------------|-------|
| **XCD 25V f/2.5** | **$3,699.00** [13][14] | ~20mm | Widest V series lens, leaf shutter 1/4000s, 592g |
| **XCD 28mm f/4 P** | **$1,679.00** [84] | ~22mm | More affordable compact alternative |
| **XCD 55V f/2.5** | **$3,699.00** [85][86] | ~44mm | 372g, very compact, linear stepping motor |
| **XCD 90V f/2.5** | **$4,299.00** [87][88] | ~71mm | Photography Life: "One of the best lenses I've ever tested" |
| **XCD 80mm f/1.9** | **$4,845.00** [15] | ~63mm | Fastest Hasselblad lens at f/1.9, 1044g |

**Total for three-lens kit:**
- 25V + 55V + 90V = **$11,697**
- 25V + 55V + 80mm f/1.9 = **$12,243**

**Newer Lens Releases**: The **XCD 2.5/25V** was announced May 7, 2026, as the widest V series lens (20mm FF equivalent). The **XCD 2.8–4/35–100E** (approximately 28–76mm FF equivalent, f/2.8–f/4) was announced alongside the X2D II 100C at $4,599. All four XCD V lenses (25V, 38V, 55V, 90V) share a standardized 72mm filter thread. Hasselblad currently has 18 XCD lenses across V, P, and E series. [14][89][90]

### 5.5 Phase One Schneider Kreuznach Blue Ring Lenses (Current US Prices as of May 2026)

| Lens | Price | Full-Frame Equivalent | Notes |
|------|-------|----------------------|-------|
| **35mm LS f/3.5 Blue Ring** | **$6,790.00** [91] | ~23mm | 89° angle of view, 1370g, leaf shutter 1/1600s |
| **55mm LS f/2.8 AF** | **$4,790.00** [91] | ~36mm | 660g, 72mm filter, leaf shutter 1/1600s |
| **80mm LS f/2.8** (original) | **$2,990.00** [91] | 50mm | Lighter, smaller option |
| **80mm LS f/2.8 MkII Blue Ring** | **$5,990.00** [77][92] | 50mm | Updated optics with aspherical element, 765g |
| **110mm LS f/2.8 Blue Ring** | **$4,990.00** [91] | ~72mm | "A longer focal length with just enough optical compression for full-length fashion, beauty and portraiture" |

**Total for three-lens kit (using MkII for 80mm):**
- 35mm + 55mm + 80mm MkII = **$17,570**
- 35mm + 55mm + 110mm = **$16,570**

**Total for three-lens kit (using original 80mm):**
- 35mm + 55mm + 80mm original = **$14,570**

**Lens Ecosystem Note**: Phase One lenses are Danish-designed, German-engineered (by Schneider Kreuznach), and Japanese-manufactured. Each lens is hand-assembled by a single expert craftsman. The system includes 11 prime lenses and 2 zoom lenses, ranging from 28mm to 240mm. The Blue Ring designation indicates the highest optical quality tier. [77][91]

### 5.6 Lens Cost Comparison

| System | Wide | Standard | Telephoto | Total (3 Lenses) | Number of Native Lenses |
|--------|------|----------|-----------|------------------|------------------------|
| **Fujifilm GFX** | $1,699 (30mm f/3.5) | $2,099 (55mm f/1.7) | $2,699 (110mm f/2) | **$6,497** | 23 |
| **Hasselblad XCD** | $3,699 (25V f/2.5) | $3,699 (55V f/2.5) | $4,299 (90V f/2.5) | **$11,697** | 18 |
| **Phase One 645** | $6,790 (35mm f/3.5) | $4,790 (55mm f/2.8) | $5,990 (80mm MkII) | **$17,570** | 13 |

---

## 6. Total System Investment Over 3 Years

### 6.1 Body Pricing (Current as of May 2026)

| System | Current New Price | Notes |
|--------|-------------------|-------|
| **Fujifilm GFX100 II** | **$7,999.95** [93][94] | Reduced from $8,499.95; $500 promotional price drop valid May 4–June 28, 2026 |
| **Hasselblad X2D 100C** | Discontinued (was $8,199 at launch Sept 2022) | No longer available new from major retailers |
| **Hasselblad X2D II 100C** | **$7,399.00** (increasing to **$7,799** on June 30, 2026) [95][96] | $800 less than X2D 100C original launch price; price increase due to supply chain fluctuations |
| **Phase One XF IQ4 150MP** | **$45,990–$53,990** (system varies by configuration) [97][98] | Launch MSRP was $51,990 (2018); current system pricing varies |

### 6.2 Used Market Pricing

| System | Used Price Range | Source |
|--------|-----------------|--------|
| **Fujifilm GFX100 II** | **$7,649–$7,699** (dealer) | B&H Used, MPB [99][100] |
| **Hasselblad X2D 100C** | **$4,079–$4,926** | MPB ($4,079–$4,349), KEH ($4,529–$4,926) [101][102] |
| **Hasselblad X2D II 100C** | Too new for significant used market | Available new at $7,399 |
| **Phase One IQ4 150MP** | **$22,490+** (certified pre-owned upgrade) | Capture Integration CPO [103] |
| **Phase One XF IQ4 150MP system** | **$35,000–$40,000** (pre-owned system with ~3,625 shots) | Various dealers [104] |

### 6.3 Depreciation Estimates

**Fujifilm GFX100 II**: Used pricing at major dealers shows minimal depreciation — approximately **4–5% savings** from new (B&H used at $7,649 vs new at $7,999). Private party pricing is lower. For comparison, the GFX100S retains strong value, selling for over $4,000 used. The $500 price drop on new units may increase downward pressure on used values. **Estimated 3-year depreciation: 25–30%** for a body with a current new price of $7,999. [99][100]

**Hasselblad X2D 100C**: The release of the X2D II 100C at $7,399 (lower than the X2D 100C's $8,199) has significantly impacted used X2D 100C values. Current used prices of $4,079–$4,926 represent approximately **40–50% depreciation** from original MSRP over 2.5 years. **Estimated 3-year depreciation for X2D II: 30–40%** given the new lower launch price. [101][102]

**Phase One XF IQ4 150MP**: Ultra-high-end gear depreciates differently. The 5-year warranty, unlimited shutter actuations, and Capture One Inside integration support value retention. However, the rumored 247MP sensor (Sony IMX811, announced March 2024) could introduce obsolescence risk for the 150MP back. Current certified pre-owned pricing of approximately $22,490 (upgrade) and $35,000–$40,000 (complete system) represents **30–50% depreciation** from new over 6–7 years. **Estimated 3-year depreciation: 20–30%** for a system purchased new in 2026. [103][105]

### 6.4 Software Costs

**Capture One Pro**: Current pricing before a 6% increase taking effect July 6, 2026 [106][107]:
- **Pro (annual subscription)**: $17/month (~$204/year) — will rise to ~$18/month
- **Pro (monthly)**: $26/month — will rise to ~$27.56/month
- **Pro (perpetual license)**: $317 (one-time) — will rise to ~$336
- **All-in-One (annual)**: $23.25/month (~$279/year)
- **Studio (annual)**: $45.75/month (~$549/year)

**3-Year Software Cost Estimates** (if switching to annual before June 2 to lock in current rates):
- Fujifilm GFX100 II: **$612** (Capture One Pro annual, 3 years at $17/month)
- Hasselblad X2D II 100C: **$0–$612** (Phocus is free; optional Capture One for post-processing)
- Phase One XF IQ4 150MP: **$0** (Capture One is bundled/integrated with the IQ4 via Capture One Inside)

### 6.5 NYC Rental House Availability

**Fujifilm GFX100 II — Most Widely Available**:
- **ROOT NYC**: GFX100 II Complete at $375/day, replacement value $8,800 [108]
- **Greenwood Cinema Rentals**: GFX100 II kit at $500/day (filmmaker-focused with cage) [109]
- **FotoCare Rental** (43 W 22nd St, NYC): GFX100 II available [110]
- **ShareGrid NYC**: Peer-to-peer, replacement value $9,000 [111]
- **Adorama Rentals**: GFX100 II available [112]

**Hasselblad X2D Systems — Good Availability**:
- **JMR Equipment Rentals** (Brooklyn, 168 53rd St): X2D 100C available, XCD 21mm, XCD 65mm lenses [113]
- **LensRentals** (national, ships to NYC): X2D 100C Lightweight Portrait Kit from $454 for 3 days [114]
- **ShareGrid NYC**: Both X2D 100C and X2D II 100C available [115]
- **K&M Camera** (Brooklyn and Manhattan): Hasselblad lenses available [116]

**Phase One Systems — Limited Specialty Availability**:
- **ROOT NYC**: IQ4 150MP Digital Back at **$975/day**, replacement value $54,721.49 [108]
- **Pro Photo Rental** (NY): Phase One XF body with IQ3 100MP at $630/day [117]
- **FutureCapture NYC**: Phase One IQ3 100MP digital back at $700/day [118]
- **K&M Camera**: Phase One 110mm LS Blue Ring at $95/day [116]
- **FotoCare Rental**: Phase One equipment available [110]

**B&H Photo does not do equipment rentals** — confirmed by multiple sources. [119]

### 6.6 3-Year Total Cost of Ownership

| Cost Component | Fujifilm GFX100 II | Hasselblad X2D II 100C | Phase One XF IQ4 150MP |
|----------------|-------------------|-----------------------|------------------------|
| **Body** | $7,999 | $7,399 (until June 30, then $7,799) | $51,990 (system with 80mm) |
| **Three Lenses** | $6,497 | $11,697 | $17,570 |
| **Total Initial Investment** | **$14,496** | **$19,096** (at $7,399) / **$19,496** (at $7,799) | **$69,560** |
| **Est. Resale Value (3 years)** | ~$5,600 | ~$4,800 | ~$45,000 |
| **Depreciation Loss** | **~$5,896** | **~$14,296** | **~$24,560** |
| **Software (3 years)** | $612 | $0–$612 | $0 (bundled) |
| **Rental Backup (NYC)** | Excellent — 5+ sources, $375/day | Good — 4+ sources | Limited specialty — $975/day |
| **Total 3-Year Cost** | **~$6,508** | **~$14,296–$14,908** | **~$24,560** |

**Note**: "Total 3-Year Cost" = estimated out-of-pocket cost after factoring depreciation (buying new and selling after 3 years) plus software costs. The Phase One figure does not include potential Capture One Pro subscription costs for desktop editing if needed beyond the bundled Capture One Inside integration. Rental costs are not included in these calculations as they vary by frequency of backup body usage.

---

## 7. New Information and Corrections

### 7.1 Profoto B4 Firmware TTL Compatibility

**Critical update**: Profoto firmware version B4, released March 11, 2026, breaks TTL on Hasselblad cameras. As of May 2026, **no fix has been released**. The only workaround is staying on pre-B4 firmware. Users are advised not to upgrade their Profoto gear to B4 firmware if using Hasselblad. [11][16]

### 7.2 Capture One Hasselblad Support Policy

**Unchanged and definitive**: Capture One still does not support Hasselblad cameras. The official support article remains active, stating support is "not currently planned." The feature request on Capture One's ideas portal (CLR-I-402) has been marked **"Unlikely to implement"** and merged into another existing idea. The latest Capture One releases — version 16.7.7 (April 16, 2026) and 16.8 Beta (May 20, 2026) — do not include any Hasselblad support. [29][30][120][121]

### 7.3 Hasselblad X2D II 100C Specifications and Comparison

The X2D II 100C was announced in late August 2025 and released in September 2025 at a launch price of **$7,399**. A price increase to **$7,799** takes effect June 30, 2026, due to supply chain fluctuations. [95][96]

**Key Improvements Over X2D 100C**:

| Feature | X2D 100C | X2D II 100C |
|---------|----------|-------------|
| **Base ISO** | ISO 64 | ISO 50 |
| **Dynamic Range** | 15 stops | 15.3 stops |
| **IBIS** | 5-axis, 7-stop | 5-axis, 10-stop (8x better) |
| **Autofocus** | PDAF, 294 zones, 97% coverage | LiDAR-assisted, 425 PDAF zones, AF-C with deep learning, subject detection |
| **Rear Display** | 3.6" TFT tilting (up only) | 3.6" OLED, 100% P3, 1,400 nits, tilts 90° up/42.7° down |
| **Ergonomics** | No joystick | 5D joystick with haptic feedback, textured grip |
| **HDR Workflow** | Standard HNCS | Industry-first end-to-end HDR (HNCS HDR) |
| **Weight** | 790–895g | 730–840g (7.5% lighter) |
| **Launch Price** | $8,199 | $7,399 (rising to $7,799) |

Sources: [53][68][95]

### 7.4 Fujifilm GFX100 II Firmware Updates

**Latest Firmware: Version 2.50** (released March 26, 2026) [82][122]:
- Enables display of aperture units as T-stop values for filmmakers
- Adds compatibility with the GF32-90mm T3.5 PZ OIS WR power zoom lens
- Requires lens firmware version 1.01 for full functionality
- Digital Camera World notes: "Firmware version 2.50 doesn't bring any benefits for still photography images on the GFX100 II"

**Version 2.40**: Enhanced wireless communication security
**Version 2.30**: Added F-Log2 C mode, timecode sync improvements, AF algorithm improvements
**Version 1.10**: Fixed the critical Godox trigger EVF blackout issue

### 7.5 Phase One XF IQ4 Firmware Updates

**Current Firmware**: XF Camera Body firmware version **5.00.5** (Feature Update #8 for IQ4). [123]

**Feature Update #8** (December 2020) added [71][124]:
- CFexpress storage support
- Enhanced WiFi functionality (direct access point)
- External USB-C power integration
- 4K JPEG processing for faster previews
- ETTR (Expose to the Right)
- Customizable live view grids

**Rumored Future Development**: Phase One is rumored to be developing a new medium-format camera featuring a **247MP sensor** (likely Sony's IMX811, announced March 2024) with a 3:2 aspect ratio, 19,200 x 12,800 pixels, and 16-bit output at up to 5.3 fps. Expected launch potentially in 2025+ — this could introduce obsolescence considerations for the IQ4 150MP. [105]

### 7.6 Market Trends for Medium Format in Commercial Fashion (2026)

The 2026 medium format market shows several relevant trends [125][126][127]:

- **Authenticity and "Emotion Over Perfect"**: A shift away from overly controlled imagery toward raw, intimate moments. Medium format's ability to capture subtle tonal transitions makes it ideal for this trend.
- **Cinematic Storytelling**: Editorial fashion is embracing narrative language with shadow, movement, and atmosphere.
- **Price Competition**: Fujifilm dropped the GFX100 II by $500; Hasselblad launched the X2D II below the GFX100 II; Phase One is offering $4,990 instant savings with lens purchase in Q2 2026. This makes 2026 a competitive time to enter medium format.
- **Hybrid Production Models**: Brands are shifting toward continuous "content systems" with projected $37 billion in US creator ad spend. Medium format's file quality and tethering workflow are assets in these production environments.
- **Restrained Retouching**: Expression, age, and character are now embraced. Medium format's superior color and tonality reduces the need for heavy post-processing.

---

## 8. Overall Summary and Recommendations

### 8.1 Fujifilm GFX100 II — Best for Value, Speed, and Familiar Transition

At **$14,496 for body + 3 lenses** (using the current $7,999 body price with $500 lens promotions), the GFX100 II is the most practical choice for a professional transitioning from Canon EOS R5. It is roughly one-fifth the cost of the Phase One system with three lenses.

**Strengths**:
- **8 fps continuous shooting** — significantly faster than both competitors, making it the best choice for high-volume fashion sessions
- **Smallest lens ecosystem costs** at $6,497 for three lenses
- **Full Capture One Pro support** — most familiar workflow for photographers moving from Canon/Sony
- **Extensive NYC rental availability** — 5+ sources for backup bodies
- **Reala Ace film simulation** — excellent out-of-camera color for commercial work

**Weaknesses**:
- **1/125s flash sync** — most restrictive of the three systems; requires HSS (with 3-stop power loss) or leaf shutter adapter (manual focus only)
- **Tethered shooting reliability** — variable reports; Ethernet connection recommended for stability
- **Color science** — good but requires more post-processing than Hasselblad or Phase One for skin tone accuracy across diverse ethnicities

**Best for**: Photographers who need the fastest workflow, want the most affordable entry into medium format, and plan to use Capture One Pro as their primary software.

### 8.2 Hasselblad X2D II 100C — Best for Color Science and Sync Speed

At **$19,096 for body + 3 lenses** (at the $7,399 launch price, before the June 30 increase to $7,799), the X2D II 100C is a premium tool with the best color science and native leaf shutter sync.

**Strengths**:
- **Best-in-class color science** — HNCS is the industry reference standard for skin tone accuracy across all ethnicities
- **Native leaf shutter sync up to 1/4000s** — no HSS power loss, ideal for overpowering ambient light
- **Built-in 1TB SSD at 2370 MB/s** — unlimited buffer, no memory cards required
- **X2D II Improvements** — 10-stop IBIS, LiDAR-assisted AF, HDR workflow, lighter body
- **Phocus software is free** — no mandatory software subscription costs

**Weaknesses**:
- **Capture One does NOT support Hasselblad** — this is the system's biggest limitation for studio professionals who require Capture One tethering
- **3 fps continuous shooting** — drops to 14-bit in continuous mode
- **Lens costs are 1.8x Fujifilm** — three-lens kit costs $11,697 vs $6,497 for Fujifilm
- **Profoto B4 firmware broke TTL** — and remains unfixed as of May 2026

**Best for**: Photographers who prioritize color science and leaf shutter sync above all else, and who can work with Phocus or adapt their workflow to Capture One post-processing only.

### 8.3 Phase One XF IQ4 150MP — Best for Absolute Quality and Reliability

At **$69,560 for system + 3 lenses**, the Phase One system is an order-of-magnitude investment. It offers the highest image quality, most reliable tethering, and largest sensor.

**Strengths**:
- **Largest sensor at 53.4 × 40.0 mm** — 2.5x the area of full-frame, 150MP resolution
- **Best tethered shooting** — Capture One Inside integration, Power over Ethernet, industry-leading stability
- **Flash Analysis Tool** — unique built-in sync troubleshooting capability for mission-critical shoots
- **5-year warranty and uptime guarantee** — unmatched in the industry
- **Leaf shutter sync at 1/1600s** — ample for most studio and outdoor scenarios

**Weaknesses**:
- **Highest cost by far** — $24,560 estimated 3-year depreciation loss is more than the entire Fujifilm system
- **Slowest continuous shooting** — 0.7–1.3 fps; designed for deliberate single-shot capture
- **Limited NYC rental availability** — scarce and expensive ($975/day for digital back only)
- **Largest and heaviest system** — not ideal for location or travel work
- **17-second startup time** — requires advance preparation

**Best for**: Photographers whose commercial clients specifically request Phase One capture, who need the absolute largest file size for high-end campaigns, or who operate in studios where tethering reliability and warranty coverage justify the investment.

### 8.4 Choice Framework

| Priority | Recommended System |
|----------|-------------------|
| **Fastest workflow / highest volume** | Fujifilm GFX100 II |
| **Best color / skin tones out of camera** | Hasselblad X2D II 100C |
| **Best tethered reliability / maximum quality** | Phase One XF IQ4 150MP |
| **Lowest total cost of ownership** | Fujifilm GFX100 II |
| **Highest resale value retention** | Fujifilm GFX100 II (lowest depreciation % of purchase price) |
| **Most available rental backup in NYC** | Fujifilm GFX100 II |
| **Best flash sync flexibility** | Hasselblad X2D II 100C |

### 8.5 Final Recommendation

For a professional photographer transitioning from Canon EOS R5 to medium format for commercial fashion work in New York, the **Fujifilm GFX100 II** offers the most natural upgrade path with the least workflow disruption. At $14,496 for the complete system, it provides 8 fps continuous shooting, full Capture One Pro support, and the most extensive rental ecosystem in NYC — all for roughly one-fifth the cost of the Phase One system.

The **Hasselblad X2D II 100C** is the choice if color science and leaf shutter sync are paramount and the Capture One tethering limitation can be worked around. At $19,096 with three lenses, it costs approximately $4,600 more than the Fujifilm but offers the best skin tone accuracy in the industry and native 1/4000s flash sync.

The **Phase One XF IQ4 150MP** is the choice for photographers who need the absolute best of everything — maximum resolution, most reliable tethering, and the largest sensor — and have the budget and workflow to support it. At $69,560, it costs nearly 5x the Fujifilm system and 3.5x the Hasselblad system, but offers a level of quality, reliability, and professional support unmatched by any other medium format system.

The 2026 market is uniquely competitive for medium format — with Fujifilm offering $500 discounts on the GFX100 II and GF lenses, Hasselblad launching the X2D II below the GFX100 II's price, and Phase One offering instant savings promotions. This makes now an excellent time to invest in medium format for commercial fashion work.

---

## Sources

[1] Fujifilm GFX100 II Specifications: https://www.fujifilm-x.com/en-us/products/cameras/gfx100-ii/specifications

[2] Fujifilm GFX100 II Full Specifications: https://www.dpreview.com/products/fujifilm/slrs/fujifilm_gfx100ii/specifications

[3] Fujifilm GFX100 II Electronic Shutter Smearing: https://diglloyd.com/blog/2024/20240608_1518-FujifilmGFX100_II-electronic-shutter-smearing.html

[4] More on GFX100 II Electronic Shutter Speeds: https://blog.kasson.com/gfx-100-ii/more-on-gfx-100-ii-electronic-shutter-speeds

[5] Get the Look: High Speed Sync: https://www.fujifilm-x.com/en-us/series/lighting-masterclass/get-the-look-high-speed-sync

[6] Fuji GFX Tip: Using Leaf Shutter Lenses: https://www.captureintegration.com/fuji-gfx-tip-using-leaf-shutter-lenses

[7] Free Fashion Photography Tutorial Using Fujifilm GFX 100: https://fstoppers.com/fashion/free-fashion-photography-tutorial-using-fujifilm-gfx-100-446463

[8] Godox and Flash Issues with GFX100 II: https://diglloyd.com/blog/2023/20231106_1900-FujifilmGFX100_II-ReaderComment-flash.html

[9] GFX100 II and Godox Trigger Help: https://www.dpreview.com/forums/threads/gfx-100ii-and-godox-trigger-help-please.4754337

[10] Elinchrom Transmitter PRO for Fujifilm: https://www.facebook.com/ElinchromLTD/posts/icymi-the-transmitter-pro-for-fujifilm-cameras-is-now-available-it-provides-full/2281692108536044

[11] X2D II Flash & TTL Guide: What Works in 2026: https://blog.tonalphoto.com/flash-and-ttl-on-the-x2d-ii

[12] Hasselblad X2D 100C FAQs: https://www.captureintegration.com/new-hasselblad-x2d-100c-faqs

[13] Hasselblad XCD 25V Official Store: https://store-na.hasselblad.com/products/xcd-2-5-25v

[14] Hasselblad XCD 25V Price and Specs: https://sheclicks.net/hasselblad-xcd-25-25v-price

[15] Hasselblad XCD 80mm f/1.9: https://store-na.hasselblad.com/products/xcd-1-9-80

[16] Hasselblad X2D II 100C Price Increase: https://www.dpreview.com/forums/threads/hasselblad-x2d-ii-100c-price-increase-6-30-2026.4836658

[17] High Speed Flash Sync with Phase One XF: https://www.captureintegration.com/high-speed-flash-triggering-using-the-phase-one-xf-and-leaf-shutter-lenses-part-1

[18] Phase One IQ Camera System Overview Flyer: https://www.phaseone.com/wp-content/uploads/2021/12/IQ_Camera_System_Overview_Flyer.pdf

[19] Phase One XC Camera Tech Spec: https://www.phaseone.com/wp-content/uploads/2024/01/XC_CameraSystem_TechSpec_Overview_1Pager_Jun23.pdf

[20] Electronic Shutter Flash Sync on IQ4: https://www.captureintegration.com/electronic-shutter-flash-sync-on-phase-one-iq3-100-and-iq4-series-digital-backs

[21] Phase One Flash Analysis: https://www.phaseone.com/inspiration/flash-analysis

[22] XF Camera System Studio Photography: https://www.phaseone.com/applications/bespoke-photography/studio-photography-xf-camera-system

[23] Capture One Fuji GFX II Support: https://support.captureone.com/hc/en-us/community/posts/14760091594653-Fuji-GFX-II

[24] Tether Tools Fujifilm GFX100 II: https://tethertools.com/camera/fujifilm-gfx-100-ii

[25] GFX100 II Tether Shooting Lightroom Classic: https://www.fredmiranda.com/forum/topic/1930605

[26] GFX100S II Capture One iPad Unstable: https://www.dpreview.com/forums/threads/fuji-gfx-100s-ii-capture-one-ipad-unstable-crashes-all-the-time.4822880

[27] FUJIFILM Tether Plugin PRO for GFX: https://www.fujifilm-x.com/en-us/support/download/software/tether-plugin-pro-for-gfx

[28] Fujifilm GFX100 II Compatibility: https://www.fujifilm-x.com/en-us/support/compatibility/cameras/gfx100-ii

[29] Capture One — Why No Hasselblad Support: https://support.captureone.com/hc/en-us/articles/27689567504413-Why-Capture-One-does-not-currently-support-Hasselblad-cameras-e-g-X2D

[30] Capture One Ideas Portal — Hasselblad X2D II Support: https://captureone.ideas.aha.io/ideas/CLR-I-402

[31] Upgrading to Hasselblad X2D: https://www.captureintegration.com/upgrading-to-hasselblad-x2d-what-you-need-to-know

[32] Hasselblad Phocus Software: https://www.hasselblad.com/learn/phocus

[33] Hasselblad Phocus 4.1.2: https://www.hasselblad.com/learn/phocus/phocus-4-1-2

[34] Hasselblad X2D 100C Firmware 3.1.0: https://www.hasselblad.com/press/press-releases/2024/hasselblad-x2d-100c-firmware-3-1-0

[35] Reddit — Phocus X2D Tether: https://www.reddit.com/r/hasselblad/comments/13joax5/phocus_x2d_100c_tether

[36] Hasselblad X2D Tethering Problems: https://www.youtube.com/watch?v=xju-8jAxbw4

[37] Phase One XF IQ4 — Capture One Inside: https://www.phaseone.com/2018/08/28/phase-ones-new-xf-iq4-camera-systems-introduce-capture-one-inside-and-enable-unmatched-workflow-flexibility-and-resolution

[38] Phase One IQ4 Digital Back Features: https://www.phaseone.com/iq4-digital-backs

[39] Hands On With Phase One IQ4 150MP: https://fstoppers.com/landscapes/hands-phase-one-iq4-150mp-can-you-shoot-long-exposures-1125s-403751

[40] Phase One IQ4-150 Technical Basics: https://www.captureintegration.com/phase-one-iq4-150-technical-basics

[41] Facebook — Phase One Tethering Cables: https://www.facebook.com/groups/783465558517931/posts/3008473776017087

[42] Review of Fujifilm Reala Ace: https://www.myersphoto.com/blog/review-fujifilm-reala-ace

[43] REALA ACE Best Film Simulator: https://alikgriffin.com/reala-ace-film-simulator

[44] Fujifilm GFX100 II Review — The Phoblographer: https://www.thephoblographer.com/2023/10/16/fujifilm-gfx100-ii-review

[45] Fujifilm GFX100 II Review — Galaxus: https://www.galaxus.at/en/page/fujifilm-gfx-100-ii-review-the-ultimate-tool-29494

[46] Fujifilm GFX100 II Reddit: https://www.reddit.com/r/FujiGFX/comments/1gxaps1/new_firmware_230_released_for_gfx_100_ii_no_other

[47] Facebook — GFX Color Issues: https://www.facebook.com/groups/761523304051245/posts/3040393106164242

[48] Capture Integration Color Management: https://www.captureintegration.com/color-management-for-fujifilm-gfx

[49] Hasselblad Natural Colour Solution: https://www.hasselblad.com/learn/hasselblad-natural-colour-solution

[50] Hasselblad X2D Best Color Science: https://www.digitalcameraworld.com/features/the-hasselblad-x2d-has-the-best-color-science-in-the-business

[51] Hasselblad X2D 100C Review — Fstoppers: https://fstoppers.com/reviews/perfect-choice-perfectionist-review-hasselblad-x2d-100c-662172

[52] Hasselblad X2D vs Fuji GFX 100 II: https://medium.com/@photographer/hasselblad-x2d-vs-fujifilm-gfx-100-ii

[53] Hasselblad X2D II 100C — PetaPixel: https://petapixel.com/2026/01/08/the-hasselblad-x2d-ii-100c-honors-its-legacy-while-embracing-the-future

[54] DPReview — Hasselblad Images Muted Look: https://www.dpreview.com/forums/threads/thoughs-about-provia-and-astia-film-simulations.4726699

[55] Hands On With Phase One IQ4 150MP: https://fstoppers.com/landscapes/hands-phase-one-iq4-150mp-can-you-shoot-long-exposures-1125s-403751

[56] PCMag — Phase One IQ4 150MP Review: https://www.pcmag.com/reviews/phase-one-iq4-150mp

[57] Phase One — Beauty of Skin Tones: https://www.phaseone.com/inspiration/the-beauty-of-skin-tones-captured-in-pure-color

[58] DT Photo — Phase One IQ4 150MP: https://www.photo-digitaltransitions.com/homepage/phase-one-iq4-150-mp

[59] Paul Reiffer — Phase One XF IQ4 Review: https://www.paulreiffer.com/2018/08/hands-on-review-launching-phase-one-iq4-150mp-infinity-platform-camera-system

[60] Fujifilm GFX File Sizes 16-bit vs 14-bit: https://diglloyd.com/prem/s/MF/FujifilmGFX/FujifilmGFX100-FileSize.html

[61] Visual Comparisons 14 and 16 bit RAW Precision: https://blog.kasson.com/gfx-100/visual-comparisons-of-fuji-gfx-100-14-and-16-bit-raw-precision

[62] Fujifilm GFX100 II Key Features: https://www.fujifilm-x.com/en-us/series/fujifilm-gfx100-ii/gfx100-ii-key-features-explained

[63] Equipment Preview — GFX100 II: https://blog.michaelclarkphoto.com?p=11987

[64] Hasselblad X2D 100C Memory Cards Guide: https://www.memorywolf.com/collections/hasselblad-x2d-100c-memory-cards

[65] Hasselblad X2D 100C Specifications: https://photorumors.com/2022/09/02/hasselblad-x2d-100c-medium-format-camera-full-specifications

[66] X2D II Continuous Shooting 14-Bit RAW: https://blog.tonalphoto.com/hasselblad-x2d-ii-why-your-raw-files-drop-to-14-bit-in-continuous-mode-and-when-it-matters

[67] Hasselblad X2D 100C Review — The Phoblographer: https://www.thephoblographer.com/2022/10/04/slow-beautiful-hasselblad-x2d-100c-review

[68] Hasselblad X2D II 100C Review — Camera Duel: https://camera-duel.com/en/test/hasselblad-x2d-ii-100c

[69] Phase One IQ4 150MP Technical Basics: https://www.captureintegration.com/phase-one-iq4-150-technical-basics

[70] Phase One IQ4 150MP PCMag: https://www.pcmag.com/reviews/phase-one-iq4-150mp

[71] Phase One Feature Updates: https://www.phaseone.com/resources-support-3/feature-updates

[72] Phase One IQ Camera System Overview Flyer: https://www.phaseone.com/wp-content/uploads/2021/12/IQ_Camera_System_Overview_Flyer.pdf

[73] Phase One XC Camera Tech Spec: https://www.phaseone.com/wp-content/uploads/2024/01/XC_CameraSystem_TechSpec_Overview_1Pager_Jun23.pdf

[74] Fujifilm GFX Crop Factor: https://shuttermuse.com/fujifilm-gfx-crop-factor-and-gf-lens-35mm-full-frame-equivalent-focal-lengths

[75] Fujifilm GFX100RF Crop Factor: https://apotelyt.com/camera-specs/fujifilm-gfx-100rf-sensor-crop

[76] Hasselblad XCD Lens Guide: https://blog.tonalphoto.com/hasselblad-xcd-lens-guide

[77] New Schneider Kreuznach 80mm LS MkII: https://www.captureintegration.com/new-schneider-kreuznach-80mm-ls-f-2-8-mark-ii-lens

[78] Fujifilm GF Lenses — Houston Camera Exchange: https://www.houstoncameraexchange.com/photography/cameras/medium-format/fujifilm/medium-format-lenses.html

[79] Fujifilm GF 55mm f/1.7 — KEH: https://www.keh.com/shop/28430069.html

[80] Fujifilm GF 63mm f/2.8 — Adorama: https://www.adorama.com/l/GF63mmf28lens

[81] Fujifilm GF 110mm f/2 — Best Buy: https://www.bestbuy.com/product/fujinon-gf-110mm-f2-r-lm-wr-standard-zoom-lens-for-g-mount-cameras-black/J7929FH8C6

[82] Fujifilm GFX100 II Firmware 2.50: https://www.topteks.com/blog/firmware-version-250-just-launched-for-fujifilms-gfx100-ii

[83] Fujifilm GFX Medium Format System Guide 2026: https://www.dailycameranews.com/2025/12/fujifilm-gfx-medium-format-system-guide

[84] Hasselblad XCD 28mm f/4 P: https://store-na.hasselblad.com/products/xcd-28p

[85] Hasselblad XCD 55V: https://store-na.hasselblad.com/products/xcd-55

[86] Hasselblad XCD 55V f/2.5 Specifications: https://www.hasselblad.com/x-system/xcd-55v/specifications

[87] Hasselblad XCD 90V: https://store-na.hasselblad.com/products/xcd-90

[88] Hasselblad XCD 90V Review — Photography Life: https://photographylife.com/reviews/hasselblad-xcd-90v-f2-5

[89] Hasselblad X2D II 100C — Imaging Resource: https://www.imaging-resource.com/news/hasselblad-x2d-ii-100c-price-preorder-details-and-new-35-100mm-zoom-lens

[90] Hasselblad XCD 35-100E Official Store: https://store-na.hasselblad.com/products/xcd-35-100e

[91] FotoCare — Phase One Lenses: https://www.fotocare.com/category_s/2151.htm

[92] DT Photo — 80mm LS MkII: https://www.photo-digitaltransitions.com/product/80mm-ls-f-2-8-mark-ii

[93] Fujifilm USA Shop — GFX100 II: https://shopusa.fujifilm-x.com/gfx100-ii-body-600023590

[94] Fujifilm GFX100 II — CineD Price Drop: https://www.cined.com/fujifilm-gfx100-ii-drops-500-plus-500-off-across-most-of-the-gf-lens-lineup

[95] Hasselblad X2D II 100C — PCMag: https://me.pcmag.com/en/cameras-1/34064/hasselblad-x2d-ii-100c

[96] Hasselblad X2D II 100C — Newsshooter: https://www.newsshooter.com/2025/08/26/hasselblad-x2d-ii-100c

[97] DT Photo — Phase One IQ4 150MP System: https://www.photo-digitaltransitions.com/product/phase-one-iq4-150mp-system

[98] Phase One XF IQ4 150MP — DPReview: https://www.dpreview.com/news/4927169563/phase-one-xf-iq4-digital-backs-offer-up-to-150mp-and-capture-one-inside

[99] MPB — Used Fujifilm GFX100 II: https://www.mpb.com/en-us/product/fujifilm-gfx-100-ii

[100] B&H Used — GFX100 II: https://www.bhphotovideo.com/c/product/803408846-USE/fujifilm_600023590_gfx100_ii_medium_format.html/qa

[101] MPB — Used Hasselblad X2D 100C: https://www.mpb.com/en-us/product/hasselblad-x2d-100c

[102] KEH — Hasselblad X2D 100C: https://www.keh.com/shop/hasselblad-x2d-100c

[103] Capture Integration — IQ4 150MP Upgrade: https://www.captureintegration.com/phase-one-iq4-150mp-upgrade-as-low-as

[104] eBay — Phase One IQ4 150: https://www.ebay.com/itm/405355359889

[105] TechRadar — 247MP Sensor Rumors: https://www.techradar.com/cameras/phase-one-rumors-suggest-a-record-breaking-247mp-medium-format-camera-is-on-the-way

[106] PetaPixel — Capture One Price Increase: https://petapixel.com/2026/05/27/capture-one-to-increase-all-product-prices-by-6

[107] Capture One Pricing: https://www.captureone.com/en/pricing/capture-one-pro

[108] ROOT NYC Rentals: https://rentals.rootnyc.com/

[109] Greenwood Cinema Rentals: https://www.greenwoodcine.com/rentals/p/fujifilm-gfx100-ii-medium-format-mirrorless-g-pl-lpl-camera

[110] FotoCare Rental: https://fotocarerentals.com/

[111] ShareGrid NYC — GFX100 II: https://www.sharegrid.com/newyork/l/402177

[112] Adorama Rentals: https://www.adoramarentals.com/

[113] JMR Equipment Rentals: https://www.jmrny.com/hasselblad

[114] LensRentals — Hasselblad X2D: https://www.lensrentals.com/rent/hasselblad-x2d-100c-medium-format-mirrorless

[115] ShareGrid NYC — Hasselblad X2D II: https://www.sharegrid.com/newyork/l/327112

[116] K&M Camera Rentals: https://www.kmcamera.com/rentals

[117] Pro Photo Rental: https://prophotorental.com/

[118] FutureCapture NYC: https://www.futurecapture.nyc/

[119] Reddit — B&H Does Not Rent: https://www.reddit.com/r/AskNYC/comments/1p3ua6w/one_day_lens_rental_in_new_york

[120] Capture One 16.7.7 Release Notes: https://support.captureone.com/hc/en-us/articles/35279291925277-Capture-One-16-7-7-release-notes

[121] Capture One 16.8 Beta: https://support.captureone.com/hc/en-us/articles/35747427882653-Capture-One-16-8-Beta-release-notes

[122] Firmware 2.50 — Digital Camera World: https://www.digitalcameraworld.com/cameras/digital-cameras/filmmakers-get-some-love-from-fujifilm-with-the-latest-gfx100-ii-firmware-update-but-what-about-still-photographers

[123] Phase One Support — Latest Firmware: https://support.phaseone.com/knowledgebase/article/KA-01254/en-us

[124] Phase One Feature Update #8: https://www.imaging-resource.com/news/phase-one-announces-feature-update-8

[125] Fashion Photography Trends 2026: https://artvisionawards.com/fashion-photography-trends-to-watch-in-2026

[126] Top Five Photography Trends of 2026: https://petapixel.com/2026/01/22/the-top-five-photography-trends-of-2026

[127] Trends That Will Shape Photography in 2026: https://en.labkorner.com/blog/1682-the-trends-that-will-shape-photography-in-2026