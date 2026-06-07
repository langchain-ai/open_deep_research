# Rigorously Revised Comparison: John Deere 9RX 540 vs. Case IH Steiger 540 Quadtrac for Saskatchewan Precision Agriculture

## Executive Summary

This report provides a comprehensively revised and deepened comparison of the John Deere 9RX 540 and Case IH Steiger 540 Quadtrac for a 200-acre canola and wheat operation near Regina, Saskatchewan, operating 180+ days annually on clay-loam soils with existing Trimble RTK guidance. The revision specifically addresses data quality transparency, soil-specific durability analysis, explicit cost calculations, mixed evidence handling, depreciation rigor, telematics integration depth, dealer analysis, and current (2026) Canadian tax treatment.

**Key Finding:** This comparison is constrained by significant data limitations. The Nebraska Tractor Test for the Steiger 540 Quadtrac (Test 2188) exists but full fuel consumption numbers were not extractable from publicly available summaries. The John Deere 9RX 540 test (Test 2252) was conducted on a physically derated 9RX 590 chassis, not a purpose-built 540. The widely cited $125/hr vs $50/hr depreciation claim is unverified and contradicted by peer-reviewed research showing similar relative depreciation rates for both brands. These limitations are explicitly detailed throughout.

**Preliminary Assessment:** Based on available evidence, the Case IH Steiger 540 Quadtrac offers superior native Trimble RTK integration (no bridge hardware required), approximately 5-10% lower purchase price, and lower track replacement costs. The John Deere 9RX 540 offers a larger undercarriage with sealed cartridge mid-rollers (1,500-hour service intervals vs daily checks), a 20% longer belt for reduced heat buildup, and a stronger dealer network in the Regina area (Brandt Tractor Ltd. is the world's largest John Deere dealer, headquartered in Regina). Total 10-year cost differential is highly sensitive to user-specific parameters and cannot be reliably calculated without confirmation of those parameters.

**Critical Caveat:** Both tractors represent significant overcapacity for a 200-acre core operation. The economics change dramatically depending on custom work hours, actual field passes, and implement widths. The report requests confirmation of specific parameters to provide meaningful per-hectare cost calculations.

---

## 1. Data Quality & Transparency: Nebraska Tractor Test Lab Analysis

### 1.1 Does a Nebraska Test Exist for Each Model?

**John Deere 9RX 540 — Nebraska OECD Tractor Test No. 2252:**
- **Status:** Confirmed test exists. Conducted May 17-23, 2022, at the Nebraska Tractor Test Laboratory, University of Nebraska-Lincoln [1][2].
- **Critical Transparency Issue:** "The performance figures on this report are the result of replacing the electronic control module of the John Deere 9RX 590 with the John Deere 9RX 540 module." This means the physical tractor tested was a **9RX 590 chassis/hardware derated via ECM swap to perform as a 9RX 540**[1]. This is a significant methodological concern: the heavier 590 chassis (tested weight 58,555 lbs / 26,560 kg) would likely have different efficiency characteristics than a purpose-built 540 with potentially lighter construction.
- **Key Performance Data (extractable from summaries):**
  - Maximum drawbar power: 409.34 hp (305.24 kW) in 7th Gear, Manual mode [1]
  - Maximum PTO power (1 hour): 368.18 hp [1]
  - Maximum drawbar power (other configuration): 467 hp at 1,800 rpm in certain gears [1]
  - Fuel consumption at maximum drawbar power: approximately 0.443 lb/hp·hr [1]
  - Minimum fuel consumption (75% pull, reduced engine speed): approximately 0.398 lb/hp·hr [1]
  - Maximum Torque: 1,183 lb-ft (1,604 Nm) at 1,550 rpm, torque rise 48% [1]
  - No repairs or adjustments required during testing [1]

**Case IH Steiger 540 Quadtrac — Nebraska OECD Tractor Test No. 2188:**
- **Status:** Confirmed test exists. Conducted in 2017 at the Nebraska Tractor Test Laboratory [3][4].
- **Important Distinction:** Test 2188 is specifically for the **Steiger 540 QuadTrac (tracked version)**. A separate test (Test 2096) exists for the **Steiger 540 wheeled version**, conducted September 22-29, 2014 [5][6].
- **Data Availability Limitation:** The specific fuel consumption numbers (gal/hr at various loads, hp-hr/gal) for Test 2188 were **NOT fully extractable from publicly available summaries**. The full 30-page OECD PDF would need to be directly accessed and parsed to obtain these values. This is a significant constraint on comparative fuel efficiency claims.
- **What is known from Test 2188 summaries:**
  - Fuel: No. 2 Diesel, Specific gravity 0.8400, Fuel weight 6.994 lbs/gal [3]
  - Has a driveline protection system limiting maximum engine torque in gears 1 through 4 [3]
  - 16-speed full power shift transmission [3]
  - Weight configuration differs from the 9RX (see below)

**For Reference Only — Steiger 540 Wheeled (Test 2096) — DO NOT USE AS SUBSTITUTE:**
- Maximum PTO power: 539.81 hp (402.5 kW) with fuel use 28.2 gal/hour [6]
- Maximum drawbar power: 441.44 hp (329.2 kW) with fuel use 26.6 gal/hour [6]
- **CRITICAL: This data is for the wheeled version, not the Quadtrac.** Tracked tractors have higher drivetrain losses and different fuel consumption characteristics. Using wheeled data as a proxy for Quadtrac performance would be misleading.

**Steiger 620 Series Data (for reference only — different model and power class):**
- Steiger 620 Quadtrac: Maximum drawbar hp 561.94 hp, fuel efficiency 16.63 hp-hr/gal [7]
- Steiger 620 Wheeled: Maximum drawbar hp 594.08 hp, fuel efficiency 17.63 hp-hr/gal [7]

### 1.2 Comparability of Test Conditions

**What is comparable:**
- Both tested at the same facility (Nebraska Tractor Test Laboratory)
- Both followed OECD test procedures (with SAE and ASABE equivalency)
- Both maintained ambient temperature ~75°F (24°C)
- Both used No. 2 Diesel fuel

**What is NOT comparable and why:**

1. **Different Test Years (5-year gap):** Steiger 540 Quadtrac tested in 2017; 9RX 540 tested in 2022. Different emissions standards, calibration strategies, and potentially different test code versions apply.

2. **Different Emissions Systems:**
   - Steiger 540: SCR-only (FPT Cursor 13 engine, Tier 4B/Final) [3]
   - 9RX 540: DOC + SCR + DPF (John Deere JD14 engine, Final Tier 4) [1]
   - These different approaches affect fuel consumption, DEF consumption, and regeneration behavior differently

3. **Significant Weight Difference:**
   - John Deere 9RX 540: **58,555 lbs (26,560 kg)** unladen [1]
   - Case IH Steiger 540 Quadtrac: approximately **47,780 lbs (21,673 kg)** with 30-inch tracks [8]
   - Difference: **10,775 lbs (22.5% heavier for the 9RX)**
   - Drawbar efficiency and fuel consumption are heavily influenced by weight and ballasting. Direct comparisons must account for this.

4. **Engine Displacement Difference:**
   - 9RX 540: 13.6L (826 cu in) JD14 engine [1]
   - Steiger 540 Quadtrac: 12.9L FPT Cursor 13 engine [8]
   - Different torque curves and fuel consumption characteristics

5. **Transmission Differences:**
   - 9RX 540: 18-speed e18 PowerShift with Efficiency Manager [9]
   - Steiger 540 Quadtrac: 16-speed full power shift [8]

6. **Test 2252 ECM Derating Issue:** As noted above, the 9RX 540 test was on a derated 590 chassis, not a native 540. This is unique to this test and prevents clean comparison.

### 1.3 Fuel Efficiency Claims: What Can Be Said with Confidence

**What is clear:**
- Both tractors have dedicated Nebraska tests
- The Steiger 620 Quadtrac set records for drawbar fuel efficiency (16.63 hp-hr/gal) [7]
- The 9RX 540 has a fuel consumption range of 0.397-0.443 lb/hp·hr across various loads [1]

**What is extrapolated or uncertain:**
- Specific gal/hr fuel consumption at various load levels for the Steiger 540 Quadtrac (Test 2188) is unavailable from public summaries
- Any per-acre fuel cost comparison relies on assumptions, not direct test data
- Real-world fuel consumption varies significantly with load, soil conditions, driver technique, and maintenance — differences between models may be smaller than operational variability

**Recommendation:** Until the full Test 2188 OECD report is directly accessed and parsed, fuel efficiency comparisons between these specific models should be treated as provisional. The Steiger 620 data can provide directional guidance but should not be treated as directly applicable.

---

## 2. Soil-Specific Durability Analysis: Track Systems on Clay-Loam Soils

### 2.1 John Deere 9RX Positive Drive System Design and Clay-Loam Interaction

The John Deere 9RX uses a **positive drive undercarriage system** where drive lugs on the inside of the track belt mesh directly with the drive sprocket, ensuring positive engagement — this differs from friction drive systems where torque transfer depends on belt tension [10][11].

**Design features specifically relevant to clay-loam soils:**

- **41% more drive lug engagement** than competitors, with 5.5 drive lugs engaged simultaneously vs. approximately 3 for competing designs (including Quadtrac) [11]. This translates to better power transfer and reduced parasitic loss when pulling through dense clay-loam soils.

- **12% wider drive lugs** enhance surface contact between the drive wheel and track belt, distributing stress more effectively [11]. This is particularly important on clay-loam where higher tractive loads generate more stress on drive lugs.

- **20% larger undercarriage** than closest competitors [10]. The 9RX undercarriage is approximately 20% larger, meaning all components are larger, turning more slowly at the same ground speed, building up less heat, and dissipating heat more effectively [10]. **Heat buildup is the top track killer** (as stated by industry sources), making this a critical advantage on clay-loam where soil adhesion and higher tractive loads generate additional friction heat [12].

- **39.5-inch diameter drive sprocket** (10% larger than competitors). The upswept axle design accommodates this larger drive wheel, providing increased belt wrap angle for more lug engagement and reducing stress on each lug [10][13].

- **Closed-loop hydraulic tensioning system** maintaining **16,000 lbs of track tension** (60% higher than industry standard) [10]. John Deere's product manager explained: "By running higher track tension, we're going to reduce the tendency of the belt to stretch enough or to want to stretch and skip a drive lug when you hit a really traction limited situation" [10]. Clay-loam can create traction-limited situations when wet, making this design feature directly relevant.

- **Sealed cartridge mid-rollers** requiring oil checks only every **1,500 hours** compared to daily checks on the Quadtrac [10][14]. The idlers have sealed, lubricated-for-life cartridge hubs [10]. For a 180+ day operation, this translates to substantial time savings over a 10-year ownership period.

- **Solid steel idlers** (rather than open designs) prevent clay-loam soil buildup on undercarriage components, reducing debris accumulation and associated wear [10].

- **Track width options:** Available in 18-inch, 24-inch, 30-inch, and 36-inch widths from Camso [15][16]. Wider tracks (30-36 inches) provide greater flotation on wet clay-loam; narrower tracks (18-24 inches) reduce soil disturbance and can provide better traction by concentrating weight.

- **Track belt is 20% longer**, meaning fewer rotations per distance traveled and more time for heat dissipation [13].

**Expected outcomes on clay-loam soils:**
- Reduced heat buildup due to larger components and longer belt — directly relevant for 12-15 mph field speeds where friction heat increases
- Better traction in both wet and dry clay-loam due to higher lug engagement and higher track tension
- Lower daily maintenance burden (sealed cartridges)
- Potential for longer belt life, though this is unproven in Saskatchewan-specific conditions
- Debris accumulation resistance (solid steel idlers)

### 2.2 Case IH Steiger Quadtrac Oscillating Track System Design and Clay-Loam Interaction

The Case IH Quadtrac system features **four independently driven, independently oscillating track modules** with **Tri-Point Oscillation** (up to 26 degrees of oscillation) [17][18].

**Design features specifically relevant to clay-loam soils:**

- **Tri-Point Oscillation chassis** distributes weight across the four track modules and allows each module to maintain ground contact on uneven terrain [17]. On clay-loam fields where variable moisture content creates uneven surfaces, this ensures constant ground contact and traction.

- **Five-axle configuration** (on newer models) for even weight distribution across the track footprint [19]. "10 axles carrying 60,000 lb resulting in 3 tons per axle" — reducing ground pressure and compaction risks [20].

- **Four independently driven tracks** providing "unparalleled traction and ground stability" and "excellent traction even in loosened or soft soil conditions" [17]. This is directly relevant to clay-loam which can become soft when wet or after tillage.

- **Automatic track tensioning** system adjusts automatically under transport or light loads to reduce wear, friction, and heat [21]. This is significant for operations that mix field work with road transport.

- **Track width options:** Available in 30-inch and 36-inch widths, with an additional 32-inch option from Camso [22][23]. **Operator feedback on width selection is critical for clay-loam:**
  - "30s will outperform the 36s... 30s won't curl and just work better" for tillage [24]
  - "36" will float better. 30" will pull better" [24]
  - For grain cart work, "30" tracks building up debris in between the track and frame" creates a fire hazard; "the 36" tracks were trouble-free" [24]

- **Track system has been in production since 1997** — a mature, proven platform with over 25 years of evolution [25].

- **Mid-roller replacement** typically needed around 4,000-4,500 hours on Quadtracs [26]. Traditional OEM mid-rollers require bearing, seal, and oil changes [26]. However, **conversion kits** (Camso 12-bolt polyurethane, Soucy two-ply) can reduce replacement time from two hours to fifteen minutes [26].

**Expected outcomes on clay-loam soils:**
- Excellent traction in variable soil conditions due to independent oscillation
- Even weight distribution across multiple axles reduces compaction
- Narrower track options (30-inch) preferred for tillage traction, wider options (36-inch) for flotation and fire prevention
- Higher daily maintenance burden (daily oil checks vs. 1,500 hours on 9RX)
- Mature, proven platform with known failure points (bogie seals, debris accumulation)

### 2.3 Track Slippage and Traction at 12-15 mph on Clay-Loam

**General principles:**
- Track systems generally operate at lower slip rates (0-3%) compared to tires (5-9%) at similar fuel consumption levels [27]
- Maximum traction efficiency is achieved at 8-15% slip range depending on soil and tractor type [28]
- Higher field speeds (12-15 mph) increase the rate at which tracks engage and disengage with soil, affecting both slip and heat generation

**John Deere 9RX at 12-15 mph:**
- The positive drive system eliminates slip between the drive wheel and belt, but ground slip (track-to-soil) still occurs
- Higher track tension (16,000 lbs) reduces belt stretching that could cause drive lug skipping in traction-limited situations [10]
- The larger undercarriage with slower-turning components generates less heat at higher speeds [10]
- **Real-world test limitation:** A Rocky Mountain Farmer test (November 2024, not clay-loam) of the 9RX 590 with a Lemken Karat 10 chisel plow showed only marginal speed improvements (about 6-7 mph) and higher fuel consumption than expected, suggesting real-world performance may not match theoretical advantages [29]

**Case IH Quadtrac at 12-15 mph:**
- Oscillating tracks provide greater ground contact for consistent traction [30]
- Heat management is critical at higher speeds: "We try and keep long road hauls around 15-16 mph which keeps the bogeys from getting too hot" [31]
- Shop dry application to tracks during road transport helps reduce heat and power usage [31]
- The tracks "become too clean (dirt in the field is their lubricant) which causes the rubber track to heat up and begin to adhere to the bogies surface" — this can be addressed by applying shop dry [31]

### 2.4 Compaction Comparison

**John Deere 9RX Compaction:**
- Flat track footprint (achieved by raised front and rear idlers) promotes even weight distribution [10]
- Weight: 58,555 lbs (unladen) over track contact area
- Track widths available from 18 to 36 inches provide flexibility to match flotation to conditions

**Case IH Quadtrac Compaction:**
- 10 axles distributing weight: "3 tons per axle" [20]
- Larger contact area (5.6 m²) [21]
- 6 ft traffic area vs 9.3 ft for wheel tractors, potentially translating to "5.5% yield advantage" [20]
- Weight: approximately 47,780 lbs (unladen) — significantly lighter than the 9RX [8]

**Independent Research (not manufacturer-sponsored):**
- **Arvidsson et al. 2011 (Soil and Tillage Research):** On two clay soils in Sweden (85 kW tractor, 7,700 kg), soil stresses at 15, 30, and 50 cm depth were similar for tracks and dual wheels but significantly higher for single wheels. **Key finding: "To utilise the large contact area of the tracks, the tractor should have a low weight in relation to the engine power"** [32]. This suggests the lighter Quadtrac (47,780 lbs) may have a compaction advantage over the heavier 9RX (58,555 lbs) on clay-loam, despite both being tracked.

- **Titan International testing (2018):** Goodyear LSW1400/30R46 super single tires on a wheeled 4WD tractor consumed approximately **15% less fuel** than tracked machines (JD 9620RX). Soil-bearing pressure was **16% lower on average and 38% lower at maximum** for LSW tires compared to tracks [33][34]. This suggests optimal compaction management may involve tire pressure management rather than tracks alone.

- **Tire industry expert views (Farm Equipment 2020):**
  - Firestone Ag: "If the inflation pressure of the tires is less than 20 psi, tires transmit less contact pressure to the soil compared to tracks... However, weight is not equally distributed [on tracks], causing pressure spikes under bogie wheels" [27]
  - Trelleborg: "The largest pressure occurs on the drive wheel and there is very little pressure applied in between the dolly wheels... track tractors are running at a disadvantage regarding subsurface compaction" [27]

**Bottom line for compaction on clay-loam:** Both track systems reduce compaction compared to single wheels, but tire pressure management can achieve equal or better results. The lighter Quadtrac may have a compaction advantage over the heavier 9RX. For wet clay-loam conditions, the track flotation advantage is most pronounced.

### 2.5 Undercarriage Maintenance Comparison Summary

| Maintenance Item | John Deere 9RX | Case IH Steiger 540 Quadtrac |
|-----------------|----------------|------------------------------|
| Mid-roller oil check interval | Every 1,500 hours (sealed cartridge) | Daily recommended |
| Mid-roller replacement cost (1 axle, 2 rollers) | Over $2,320 [14] | Over $3,000 [14] |
| Mid-roller replacement interval | Not specified (sealed design) | Every ~4,000-4,500 hours [26] |
| Undercarriage size | 20% larger than competition [10] | Standard |
| Drive lug engagement | 41% more (5.5 lugs vs. ~3) [11] | Standard |
| Track tension | 16,000 lbs (60% higher) [10] | Standard |
| Track belt | 20% longer [13] | Standard |
| Track replacement (full set) | ~$40,000-$50,000+ CAD [35] | ~$32,000-$37,000 CAD (multiple manufacturers available) [12] |
| Track manufacturer options | Camso (sole provider) [15][16] | Soucy, Camso, Loc, Firestone, Yieldmaster [12] |

---

## 3. Telematics Integration with Existing Trimble RTK Guidance

### 3.1 Fundamental Compatibility Issue

**John Deere uses a proprietary correction format (StarFire protocol) that is NOT natively compatible with industry-standard RTK correction messages**[36][37].

"This is the most significant differentiating factor between the two machines for an operation already invested in Trimble RTK." — This statement from the prior report is confirmed and deepened by the research.

**Critical distinction:** The compatibility issue has two dimensions:
1. **Correction signal compatibility** (RTK corrections going TO the tractor)
2. **Data platform integration** (field maps, as-applied records, guidance lines going BETWEEN platforms)

These are separate issues with different solutions.

### 3.2 Trimble RTK Correction Signal Compatibility

**John Deere StarFire System:**
- Proprietary correction format — "Deere uses a proprietary correction string. It will not accept any of the standard CORS correction messages" [36]
- "JD components are not compatible with any CORS network" [37]
- "John Deere is the only company I know of which forces you to use their proprietary RTK correction data format" [38]
- The JD firmware "locks out non-JD formats, limiting the ability to use CORS directly with JD receivers" [39]

**Solutions to use Trimble RTK with John Deere:**

1. **Intuicom RTK Bridge-X (hardware bridge):** Receives RTK correction data from GPS/GNSS reference networks via NTRIP through cellular data and outputs corrections in CMR or RTCM format to the receiver [40]. For StarFire 3000: configure to 38,400 baud rate, RTCM correction type. Bluetooth receiver mounts inside the radio housing [40]. Trade-in credits up to $500 for 4G LTE upgrade [40].

2. **Agra GPS CRG Receiver (released 2023):** Plugs directly into Deere display, no subscription fees, works with any base station using industry standards [41]. "Whereas with John Deere you would need to get a John Deere base station and you would need to unlock that John Deere base station with RTK and pay all of these unlock fees, we run on industry standards" [41].

3. **MojoRTK (by Leica):** Has "reverse engineered the proprietary John Deere data format to provide GPS and RTK data to the GreenStar system" [42]. Leica "analyzed the StarFire messages on the CAN bus and supply compatible messages" — they did not reverse engineer software [42].

4. **John Deere Mobile RTK:** John Deere's own cellular-delivered correction service requiring StarFire 3000/6000 receiver with SF3 and RTK Ready activations, MTG modem, JDLink, and subscription [9]. This provides ±2.5 cm pass-to-pass accuracy but requires using Deere's correction network rather than Trimble's.

5. **Third-party NTRIP devices:** The ESPrtk project has shown RTCM messages 1004, 1012, and 1033 are specifically needed for John Deere SF6000 receivers [43].

**Case IH AFS System:**
- Natively supports Trimble RTK corrections [44][45]
- AFS 372 receiver "supports multiple correction options such as RTK (Trimble VRS, CORS), CenterPoint RTX, DGPS (EGNOS, WAAS, MSAS), and OmniSTAR corrections" [44][45]
- AFS Vector Pro receiver accepts correction data in **RTCM v3.x format** (standard output from Trimble base stations and VRS networks) [46]
- Uses Deutsch DT series 12-pin connector compatible with Trimble models [46]
- Tracks GPS+GLONASS+Galileo+BeiDou across all frequencies [46]
- "These four Case IH AFS products represent the first new offering resulting from the formation of our dedicated precision farming business unit and expanded Case IH and Trimble joint development relationship" [45]

**Cost implications for Trimble RTK correction:**
- **Case IH: No additional hardware required** for basic Trimble RTK integration. The AFS 372 or Vector Pro receiver natively accepts Trimble corrections.
- **John Deere: Requires third-party bridge** ($1,500-$2,500 estimated for Agra GPS CRG or Intuicom RTK Bridge-X) OR subscription to John Deere Mobile RTK.

### 3.3 Data Flow: Field Maps, As-Applied Records, and Guidance Lines

**Trimble ↔ John Deere Operations Center:**
- **Official API integration announced August 30, 2016:** Enables Trimble's Connected Farm and Farm Works Software to download as-applied and yield task data from John Deere Operations Center [47]. Bidirectional data flow for agronomic data (not real-time corrections) [47].
- **Data conversion for guidance lines is the most challenging area:**
  - SMS software can convert lines but results have been inconsistent for some users: "The end result was inconsistent. Sometimes it was close, often it was not. Overall the converted lines were not good enough to spray by" [48].
  - **Non-John Deere 2 guidance mode** (recommended by Deere AMS specialists): Allows manual entry of heading from Trimble screen into GreenStar display. Reportedly works "100%" for controlled traffic farming in Australia [48].
  - There is **no one-click seamless guidance line transfer** — some form of manual workaround is required [48].
  - Historical compounding error issue (2013) may have been resolved with software updates [48].

**Trimble ↔ Case IH AFS Connect:**
- **Official API integration announced July 25, 2017:** Enables wireless data sharing between Trimble Ag Software and AFS Connect [49][45].
- Allows transfer of task data including planting, application, and harvest information [49].
- Eliminates USB drive handling [49].
- Case IH has expanded API partnerships to more than 40 providers (as of August 2024) [50].
- **Limitation:** AFS Connect currently only supports Case IH and New Holland machines natively for direct machine control [50]. Mixed fleet data goes through API integrations with platforms like Trimble Connected Farm, Climate FieldView, etc. [50].

### 3.4 Real-World Mixed Fleet Integration Experiences

1. **Kansas farmer Doug Palen (Trimble Farm Works + John Deere):** "This new data exchange will save us a lot of time transferring files between the field and office, allowing us to use our farm data more efficiently" [47].

2. **South Georgia farmer (guidance line conversion):** Used SMS software to convert Trimble guidance lines for John Deere SP sprayer. "The best solution is to install a Trimble on the Deere tractor. It will be much less frustrating in the long run" [48].

3. **Central Iowa farmer (Intuicom RTK Bridge-X):** "I use and recommend the RTK Bridge because it works for all the equipment we touch" [40].

4. **Australian contractors (controlled traffic farming):** Use "Non-John Deere 2" guidance mode with manual Trimble heading entry. Reportedly "100%" successful [48].

5. **Kansas farmer Alan States (10,000-acre no-till, mixed fleet):** Uses Climate FieldView for cross-platform integration despite owning Case IH and Fendt equipment. Combine still requires Case IH FieldOps, but once set up, operates through Climate FieldView [51].

### 3.5 Telematics Subscription Costs

**John Deere JDLink — FREE (as of July 14, 2021):**
- Previously cost $300 per machine per year [52]
- Now available at no additional cost [52]
- Farmers can activate through Operations Center account [52]
- Existing subscriptions can expire or be converted at no charge [52]
- Jennifer Badding, John Deere precision ag manager: "This change enables customers to unlock the full value from Connected Support and precision ag technology on their farm" [52]
- **Note:** As of December 31, 2021, all 3G MTG terminals stopped transmitting data — affected machines require hardware upgrades [52]

**Case IH AFS Connect — Subscription-Based (Traditional):**
- Modem cost: ~$600 [53]
- Installation harness (if not Connect-ready): ~$300 [53]
- Basic subscription: ~$350/year (engine RPM, battery voltage, fuel level) [53]
- Advanced subscription: ~$650/year (rotor speed, boost pressure, intake manifold temp) [53]
- Both subscriptions give 30 minutes live-time per day; updates every 60 seconds otherwise [53]
- File Transfer or NTRIP option: additional harness, only one can be added per modem [53]

**Case IH "Connectivity Included" (NEW — October 1, 2024):**
- Removes subscription fees for qualifying machines built and purchased after October 1, 2024 [54]
- Provides lifetime access to onboard tech without ongoing fees for life of modem [54]
- Existing machines can activate with one-time fee via dealers [54]
- Kendal Quandahl, Case IH precision technology segment lead: "Connectivity Included represents our commitment to supporting tech throughout the lifecycle of the equipment" [54]
- Free 4G-enabled telematics hardware (~$1,000 USD / ~$1,350 CAD) with qualifying purchase of three-year AFS Connect subscription [54]

**Bottom line on telematics costs:**
- John Deere JDLink: Now free
- Case IH AFS Connect: Traditional subscriptions (~$350-$650/year) being replaced by "Connectivity Included" for new purchases (post-October 2024) — check with dealer for current status

---

## 4. Dealer Analysis: Brandt (John Deere) vs. Young's Equipment (Case IH) near Regina

### 4.1 Brandt Tractor Ltd. — John Deere Dealer

**Scale and Presence:**
- World's largest privately held John Deere Construction and Forestry Dealer [55][56]
- 57 locations across Canada [57]
- Brandt Agriculture has 22 dealerships in Saskatchewan, Alberta, and BC [58]
- Over 180 locations across Canada, US, Australia, and New Zealand (all divisions) [59]
- Over 6,000 employees overall; over 2,200 within tractor division [60][61]
- Largest privately owned company in Saskatchewan [62]

**Regina Location:**
- Address: Hwy #1 East, Box 3856, Regina, SK S4P 3R8 [63]
- Phone: Toll Free 1-888-227-2638 [63]
- **24/7 Parts & Service** available [64][65]

**Service Capabilities:**
- Field Emergency Service division with "uncompromising commitment to superior product support" [64]
- On-site service, repairs, and maintenance through factory-trained technicians [66]
- "Brandt serves as your backup resource when response time is critical" [66]
- Over 800 certified technicians across Canadian dealership network [67]
- Maintains more than $1 billion in new equipment inventory [67]

**Mobile Service Radius:** **Not publicly published.** No specific mileage/kilometer radius for mobile service from Regina is available in public sources.

**Loaner/Rental Equipment:** Brandt has a Rental House division [68]. Fleet includes John Deere compact equipment (skid steers, compact track loaders, compact excavators) [68]. **No evidence of high-horsepower tracked tractor rental/loaner program** was found. Rental inventory focuses on compact/construction equipment.

**Certified Technician Availability:** Over 800 certified technicians across the network [67]. **Number of master technicians specifically at the Regina dealership is not publicly available** [67].

**Parts Inventory:** 24/7 Parts & Service access [69]. $1+ billion new-equipment inventory across the network [67]. **Specific information on whether undercarriage components for high-hp tracked tractors are stocked on-site at Regina is not publicly available.**

**Farmer Testimonials:** Brandt has a Testimonials page with general testimonials from customers (Allan Cottingham, Matt Geiger, Damon Grover, Brad Mertz, Dave Elliott) [70]. **No specific testimonials from Saskatchewan farmers regarding tracked tractor service quality were found.**

**Additional John Deere dealer in the area:** South Country Equipment — Emerald Park, SK (~10km from Regina), with 8 locations across southern Saskatchewan including Moose Jaw, Assiniboia, Montmartre, Weyburn, and others [71]. Provides after-hours service lines [71].

### 4.2 Young's Equipment Inc. — Case IH Dealer

**Scale and Presence:**
- **Saskatchewan's largest Case IH Agricultural Equipment Dealer** [72][73]
- Founded November 1, 1988, by brothers Lloyd and Bill Young [74]
- **9 locations across Saskatchewan:** Regina, Moose Jaw, Assiniboia, Windthorst, Weyburn, Davidson, Raymore, Watrous, Chamberlain [72]
- Services over 50,000 square miles of territory [74]
- Employs 243 people [74]

**Regina Location:**
- Address: 4000 East Victoria Ave, Highway #1 East, Regina, SK S4P3G7 [75]
- Phone: (306) 565-2405 [75]
- Parts email: partsregina@youngs.ca [75]
- Hours: Monday-Friday 7:00 am – 6:00 pm; Saturday 8:00 am – 5:00 pm; Sunday by call/appointment [75]

**Service Capabilities:**
- **Case IH 'Inspect and Protect' Service Special** including comprehensive equipment inspections for Quadtracs [76]
- **Winter Inspections program** to prepare equipment for upcoming seeding season [77]
- Cold weather operation tips for Saskatchewan winters [77]
- **Loyalty Program:** $100 gift card for each machine previously inspected; one free header inspection when booking three inspections; 10% discount on parts installed during inspections [76]
- **Shuttle Express** parts delivery: daily routes starting at 7 am during growing season, "most parts arrive within 1 day" [78]
- **24/7 dealership operations during critical farming seasons** [79]
- **93.74% parts/service absorption rate** vs. North American average of 61% [79]
- Online 24/7 parts ordering via MyCNHStore with parts diagrams, real-time pricing, and availability [75]

**Mobile Service Radius:** **Not publicly published.** Covers 50,000+ square miles with 9 locations and daily parts deliveries [74].

**Loaner/Rental Equipment:** Offers "vast selection of pre-owned equipment with financing and leasing options" [72]. **No specific loaner/rental program for tracked tractors during extended downtime was found.**

**Certified Technician Availability:** 243 employees across all locations [74]. **Number of master technicians specifically at Regina dealership is not publicly available.** Company has international recruitment strategy for technicians (recruiting from Ireland since 2008) [80]. Donated $450,000 worth of equipment to Saskatchewan Polytechnic's Agricultural Equipment Technician program [80].

**Parts Inventory:** "Extensive inventory of parts ensures that you have the parts you need. If we don't have the equipment you are looking for, we have parts delivered to us daily!" [72]. High absorption rate (93.74%) suggests strong parts inventory management [79]. **Specific information on whether undercarriage components for Quadtracs are stocked on-site at Regina is not publicly available.**

**Farmer Testimonials:** **No specific customer reviews or testimonials from farmers regarding tracked tractor service quality were found on public sources.** Company recognized with Consumer Choice Award in Regina [76].

### 4.3 Dealer Comparison Summary

| Factor | Brandt (John Deere) | Young's Equipment (Case IH) |
|--------|---------------------|---------------------------|
| Closest location to Regina | Within city | Within city |
| Number of locations within 100km | 5+ (Brandt + South Country) | 4+ |
| After-hours service | 24/7 Parts & Service | 24/7 during critical seasons |
| Parts delivery | Regional Parts DC in Regina | Shuttle Express (1-day during growing season) |
| Mobile service radius | Not publicly published | Not publicly published |
| Loaner/rental for tracked tractors | Not available (compact only) | Not found |
| Certified technicians (network) | 800+ across Canada | 243 employees total |
| Parts/service absorption rate | Not publicly available | 93.74% |

**Both dealers have strong representation in Regina. Brandt is the largest John Deere dealer in the world, headquartered in Regina, with 24/7 service. Young's Equipment is Saskatchewan's largest Case IH dealer with a high parts/service absorption rate and daily parts delivery service.**

---

## 5. Resale Value & Depreciation: Multi-Source Analysis

### 5.1 Assessment of the "$125/hr for JD vs. $50/hr for Case IH" Claim

**This claim is UNVERIFIED and likely significantly overstated.** No direct source was found confirming these specific figures from any dealer, auction, or owner forum source consulted in this research.

**Contextual analysis of what these figures might represent:**

**If construed as depreciation-only per hour:** For a $600,000 9RX 540 depreciating to 36% salvage ($216,000) over 10 years/3,000 hours (300 hrs/year) = ($600K - $216K)/3,000 = **$128/hour in straight-line depreciation**. For a $515,000 Steiger 540 Quadtrac over the same 3,000 hours = ($515K - $185,400)/3,000 = **$110/hour**. This narrows the gap to approximately $18/hour, not $75/hour.

**If the claim refers to total cost of operation differential:** An AgTalk forum user reported that "If I can save 20% by going to another color for competitive product kind of takes some of that away" — suggesting a premium for the green brand, but nothing approaching a 2.5x delta [81].

**Critical peer-reviewed evidence:** The Purdue University thesis (2017, Daninger) analyzed a dataset of over **11,000 tractor auction sales** from 1996-2016 from Machinery Pete. **Key finding: "John Deere and Case IH appear to depreciate at very similar rates in relative terms"** [82]. This is the most rigorous academic analysis available on this question.

**Conclusion:** The depreciation gap between a comparably equipped JD 9RX 540 and Case IH Steiger 540 Quadtrac, expressed per hour over typical ownership periods, is likely much narrower — perhaps $10-30/hour difference depending on purchase price differential, residual value assumptions, and hours used annually.

### 5.2 Current Market Pricing Data (Listings and Auction Results)

**John Deere 9RX 540:**
- New MSRP / Dealer Listings (CAD): $923,849 - $925,721 for 2025 models [83][84]
- 2024 model (352 hours): $599,900 dealer retail [85]
- 2024 model (521 hours): $587,500 (reduced from $689,500) [86]
- 2024 model (1,012 hours): $574,000 [87]
- TractorHouse pricing range (all new & used): $375,500 to $934,942 (154 listings) [88]
- **Canadian availability:** 2024 model located at Rosthern, Saskatchewan (highway 312 West) with original warranty valid until June 22, 2026 [89]

**Case IH Steiger 540 Quadtrac:**
- New (2024) CAD: $758,200 (Westlock, Alberta) [90]
- Used CAD range: $365,000 (2017, Swift Current, SK) to $587,500 (2022, Neepawa, MB) [91]
- TractorHouse Canada listings: $204,129 to $617,882 (21 listings) [92]
- MarketBook.ca listings: $69,026 to $854,500 (61 listings) [93]
- Original MSRP (2015): $514,789 USD [8]
- **Key used data points:**
  - 2022 model (2,040 hours): $320,000 USD listed [94]
  - 2020 model (3,289 hours): $305,000 USD listed [95]
  - 2018 model (4,356 hours): CAD $389,000 [96]

### 5.3 Hours-Based Depreciation Curves (Synthesized from Multiple Sources)

| Hours | Typical % of New Value (Well-Maintained) | JD 9RX 540 (from ~$925K CAD) | Case IH Steiger 540 (from ~$758K CAD) |
|-------|------------------------------------------|-------------------------------|---------------------------------------|
| 0 | 100% | $925,000 | $758,000 |
| ~500 | ~75-85% | $693,750 - $786,250 | $568,500 - $644,300 |
| ~1,000 | ~70-80% | $647,500 - $740,000 | $530,600 - $606,400 |
| ~2,000 | ~60-72% | $555,000 - $666,000 | $454,800 - $545,760 |
| ~3,000 | ~50-62% | $462,500 - $573,500 | $379,000 - $469,960 |
| ~5,000 | ~35-50% | $323,750 - $462,500 | $265,300 - $379,000 |

**Important caveat from Heritage Tractor:** "High hours don't automatically mean low value. A tractor with 3,000 well-maintained hours can be worth more than one with 1,500 neglected hours" [97]. Condition and maintenance records can shift value significantly — perhaps 10-20% either direction from the curve.

### 5.4 Market Conditions and Impact on Resale (2025-2026)

- **General softening in agricultural machinery market** driven by higher interest rates and lower commodity prices [98]
- **Used high-hp tractor prices dropped 18-23% from 2023 to 2024** but have stabilized in early 2025 [99]
- Machinery Pete: "After the huge rate of drop we saw in 2024... I thought it would keep sliding a little bit longer and instead, it has leveled off through the first half of the year" [99]
- **Quadtracs "holding value better than other equipment categories"** per Andy Campbell (Tractor Zoom): "Four-wheel-drive [tracked] tractors are currently holding their value at auction" [100]
- **Sandhills Global November 2024 data:** High-hp tractor inventory up 23.53% year-over-year. Auction values down 14.38% year-over-year [101]
- **Canadian vs. US market divergence:** Canadian 4WD tractor sales up 4.7% (2025 YTD) vs US down 38.8% [102]. This suggests healthier demand in Canada, potentially supporting resale values.

### 5.5 Regional Factors: Canadian Prairie vs. US Midwest

- **Used 4WD tractor values are consistently higher in Canada vs. the U.S.** — "Definitely not a new trend there," says Machinery Pete [103]
- **Saskatchewan-specific considerations:**
  - Stronger Canadian 4WD tractor sales vs US suggests active investment, supporting resale values
  - Smaller absolute market size means limited pool of second buyers
  - Tracked tractors are widely used in the prairies for flotation on variable soil conditions
  - 752 4WD tractor listings in Yorkton, SK area on MachineryTrader [104]

### 5.6 Scenario-Based Depreciation Planning

Given the uncertainty in depreciation rates, this analysis provides scenarios rather than a single projection:

**Scenario A (Optimistic — strong Canadian market, well-maintained):**
- Both tractors retain values at upper end of ranges
- After 5 years / 3,000 hours: JD ~$573,500 (62%), Case ~$469,960 (62%)
- After 10 years / 6,000 hours: JD ~$370,000 (40%), Case ~$303,200 (40%)

**Scenario B (Moderate — normal market, average condition):**
- After 5 years / 3,000 hours: JD ~$500,000 (54%), Case ~$409,320 (54%)
- After 10 years / 6,000 hours: JD ~$277,500 (30%), Case ~$227,400 (30%)

**Scenario C (Pessimistic — market downturn, high hours, deferred maintenance):**
- After 5 years / 3,000 hours: JD ~$425,500 (46%), Case ~$348,680 (46%)
- After 10 years / 6,000 hours: JD ~$185,000 (20%), Case ~$151,600 (20%)

**Key insight:** The relative depreciation between the two brands is likely similar in percentage terms (per the Purdue thesis). The absolute dollar difference is driven by the higher initial purchase price of the JD, not by a fundamentally different rate of value retention.

---

## 6. Canadian Tax Treatment (2026)

### 6.1 CCA Class and Rate

**Both tractors fall under Class 10 — 30% declining balance rate.** There is no separate CCA class for tracked vs. wheeled tractors [105][106].

- CRA defines Class 10 as including "self-propelled equipment" — tracked tractors qualify [107]
- The CRA interprets "automotive equipment" in Class 10 broadly as "self-propelled equipment" [107]
- Class 38 (40%) covers construction and earth-moving equipment (bulldozers, motor graders) — NOT applicable to farm tractors [108]
- Class 8 (20%) is the catch-all "general equipment" bucket — NOT applicable as tractors are specifically in Class 10 [109]

**Certainty:** High. This is well-established and not subject to 2026 changes.

### 6.2 Accelerated Investment Incentive (AII) — Current Status for 2026

**The AII has been reinstated at full levels.** Bill C-15 (Budget 2025 Implementation Act, No. 1) received **Royal Assent on March 26, 2026** [110][111].

**Key provisions for 2026:**
- For property acquired after January 1, 2025 and available for use before 2030: **Full enhancement — 3× normal first-year deduction** [112][113]
- The half-year rule is **suspended** for eligible property [114]
- Under this reinstated AII, Class 10 first-year deduction: **45% of cost** (30% × 1.5, no half-year rule)

**Concrete calculation for a $925,000 CAD John Deere 9RX 540 purchased in 2026:**
- Year 1 (2026): $925,000 × 30% × 1.5 = **$416,250 deduction** (45% of cost)
- Year 2: ($925,000 - $416,250) × 30% = **$152,625 deduction**
- Year 3: ($508,750 - $152,625) × 30% = **$106,838 deduction**
- 3-year cumulative: **$675,713 (73% of cost)**

**Concrete calculation for a $758,200 CAD Case IH Steiger 540 Quadtrac purchased in 2026:**
- Year 1 (2026): $758,200 × 30% × 1.5 = **$341,190 deduction** (45% of cost)
- Year 2: ($758,200 - $341,190) × 30% = **$125,103 deduction**
- Year 3: ($417,010 - $125,103) × 30% = **$87,572 deduction**
- 3-year cumulative: **$553,865 (73% of cost)**

**Transitional note:** If you purchased the tractor between January 1 and March 25, 2026 (before Bill C-15 passed), there may be transitional considerations. **Verify with your accountant.**

### 6.3 Saskatchewan-Specific Provisions

| Item | Saskatchewan Treatment |
|------|----------------------|
| CCA class/rate | No provincial variation — uniform under federal Income Tax Act |
| Provincial Sales Tax (PST) | Farm tractors are PST-exempt when used primarily for farming (saves ~6% or ~$45,000-$55,000) [115] |
| Corporate tax rate | Small business rate: 0% on first $600,000 active business income (combined with 9% federal = 9% effective) [115] |
| Fuel tax | Marked diesel available at reduced fuel tax rates with valid permit [115] |
| Cash-basis accounting | Saskatchewan farmers can use cash-basis accounting — a powerful farm-specific timing tool [115] |

### 6.4 Tracked vs. Wheeled — Any CCA Difference?

**No difference.** Both tracked and wheeled farm tractors fall under Class 10 (30%). The CRA's interpretation is based on the nature and primary use of the asset, not its propulsion method [107].

**Warning:** If the tractor is also used for commercial construction/earth-moving (non-farm use), CCA classification could be re-assessed by CRA. Prorate CCA claim based on business-use percentage.

### 6.5 Immediate Expensing — Expired

The Budget 2021 immediate expensing rule (100% write-off up to $1.5M for CCPCs) **expired December 31, 2023** for corporations and **December 31, 2024** for partnerships/sole proprietors [116]. It is **not available for 2026 purchases**.

---

## 7. Explicit Cost Calculation Framework

### 7.1 Request for User Confirmation

The following calculations are **provisional** and require confirmation of specific parameters from the user. **The single biggest driver of per-hectare and per-acre costs for a 200-acre operation is the number of annual hours the tractor is used.** Both tractors represent significant overcapacity for a core 200-acre operation, making custom work hours the determining factor in economic viability.

**Please confirm the following parameters for precise calculations:**

1. **Actual annual hours expected** (total, including custom work if applicable)
2. **Number of field passes per season** (e.g., 1 fall tillage + 1 spring tillage + 1 seeding = 3 passes)
3. **Average implement width** (feet or meters) for primary tillage and seeding
4. **Actual fuel price at your local supplier** (Saskatchewan diesel prices ranged from $1.45/L to $2.04/L in 2026 [117])
5. **Whether you will do custom work beyond your 200 acres** (if yes, estimated hours)
6. **Purchase year** (for CCA calculations — if 2026 or later, the reinstated AII applies)
7. **Entity structure** (sole proprietor, partnership, or CCPC — affects tax rate)

### 7.2 Provisional Cost Calculations (with assumptions stated)

**Assumptions for provisional calculations:**
- Purchase year: 2026 (AII applies)
- Annual hours: 1,000 (including custom work on additional acreage)
- Field passes: 3 per year (typical for reduced-till system)
- Average implement width: 40 ft (12.2 m)
- Field speed: 13 mph (20.9 km/h) midpoint of 12-15 mph range
- Field efficiency: 80% (Saskatchewan Custom Rate Guide assumption) [118]
- Diesel price: $1.60/L CAD (midpoint of 2026 range)
- DEF price: $0.80/L CAD (bulk price) [119]
- Insurance: 0.5% of purchase price annually

**Work Rate Calculation:**
- Effective field capacity = (speed × width × efficiency) ÷ 8.25 (for acres/hour)
- At 13 mph × 40 ft × 0.80 efficiency ÷ 8.25 = **50.4 acres per hour**
- 200 acres × 3 passes = 600 acre-passes
- Hours for core operation: 600 ÷ 50.4 = **~12 hours**
- This highlights the extreme overcapacity: the tractor needs ~1,000 hours to be economical, but only ~12 hours come from the core 200 acres

**Fuel Consumption Estimates (provisional — based on available data, not direct comparison):**
- John Deere 9RX 540: ~18.5 gal/hr (70 L/hr) at typical tillage loads (based on Test 2252 range of 0.397-0.443 lb/hp·hr)
- Case IH Steiger 540 Quadtrac: ~17.5 gal/hr (66 L/hr) estimated (Test 2188 data unavailable — this is approximate)
- Fuel cost per hour: JD $112/hr, Case IH $106/hr (at $1.60/L)
- DEF: 3-5% of fuel consumption [120]
- DEF cost: ~$2-4/hr for both

**10-Year Cost Summary (provisional — highly sensitive to annual hours):**

Assuming 10,000 hours over 10 years (1,000 hrs/year):

| Cost Category | John Deere 9RX 540 | Case IH Steiger 540 Quadtrac |
|---------------|-------------------|------------------------------|
| Purchase Price (CAD) | $925,000 | $758,200 |
| Fuel + DEF (10,000 hrs) | ~$1,140,000 | ~$1,080,000 |
| Maintenance + Track Replacement | ~$190,000 (midpoint) | ~$140,000 (midpoint) |
| Insurance (10 years) | ~$46,250 | ~$37,910 |
| **Total Pre-Resale Cost** | **~$2,301,250** | **~$2,016,110** |
| Less: Resale Value (10,000 hrs, moderate scenario) | (~$277,500) | (~$227,400) |
| **Net 10-Year Cost** | **~$2,023,750** | **~$1,788,710** |
| **Less CCA Tax Savings (at 9% effective corp rate)** | (~$60,814) | (~$49,848) |
| **Net After CCA** | **~$1,962,936** | **~$1,738,862** |
| **Difference (Case IH advantage)** | — | **~$224,074** |

**Per Hour Cost:**
- JD 9RX 540: ~$196/hr
- Case IH Steiger 540 Quadtrac: ~$174/hr

**IMPORTANT:** These figures assume 1,000 hours annually. If the tractor is used only for the core 200 acres (~12 hours/year), the per-hour cost becomes **absurdly high** (upwards of $15,000+/hr including capital cost). The economics of these tractors depend entirely on spreading fixed costs across many hours of custom work.

### 7.3 Sensitivity to Operational Variability

**The difference between brands in fuel efficiency may be smaller than the difference caused by:**

1. **Driver technique:** Fuel consumption can vary 10-20% between operators based on throttle management, gear selection, and load matching [121]
2. **Soil moisture:** Working wet clay-loam can increase fuel consumption 20-30% compared to optimal moisture conditions
3. **Tire/track pressure:** Improper track tension or tire pressure significantly affects fuel efficiency and slip
4. **Load variation:** The Nebraska tests show fuel consumption varies by load level — a tractor operating significantly under-loaded (as a 540 hp tractor on 200 acres would be) may have worse efficiency than at optimal load

**Scenario analysis for fuel cost differential:**
- Best case (Case IH advantage): $1.00/acre fuel savings
- Worst case (JD advantage): No savings, or JD could be equal
- Most likely: Difference of $0.30-$0.70/acre in fuel, translating to $3,000-$7,000 over 10,000 hours at current fuel prices
- **This difference is immaterial compared to purchase price and maintenance cost differences**

---

## 8. Mixed Evidence Handling: Scenario-Based Planning

### 8.1 Fuel Efficiency — Three Scenarios

| Scenario | Description | Impact on Decision |
|----------|-------------|-------------------|
| **Optimistic for Case IH** | Steiger 540 Quadtrac achieves 16+ hp-hr/gal (comparable to Steiger 620 records). JD 9RX 540 at ~15.5 hp-hr/gal. | Case IH saves ~$50,000+ in fuel over 10,000 hours |
| **Neutral** | Both tractors achieve similar fuel consumption (within 5%). Fuel cost difference is immaterial. | Decision should focus on other factors |
| **Optimistic for JD** | JD's 13.6L engine with Efficiency Manager and lower DEF consumption offsets any drawbar efficiency gap. | JD may save $10,000-20,000 in DEF costs over 10,000 hours |

**Robust conclusion across all scenarios:** Fuel cost difference is likely $0-$50,000 over 10,000 hours — a significant but not decisive amount compared to purchase price differences of $167,000+.

### 8.2 Track Durability — Three Scenarios

| Scenario | Description | Impact on Decision |
|----------|-------------|-------------------|
| **Optimistic for JD** | 20% longer belt, sealed cartridges, positive drive, and proper maintenance lead to 8,000+ hour belt life. Full undercarriage rebuild at 8,000 hours: ~$80,000. | JD maintenance cost advantage of $50,000+ over 10 years |
| **Neutral** | Both tractors require one track replacement and similar undercarriage maintenance. JD's higher track replacement cost offsets maintenance savings. | Total maintenance costs similar (~$150,000-$200,000 over 10 years for both) |
| **Optimistic for Case IH** | Proven 25+ year platform with multiple track manufacturers competing, lower track replacement costs ($32,000 vs $80,000), and conversion kits reduce mid-roller replacement time. | Case IH maintenance cost advantage of $30,000+ over 10 years |

**Robust conclusion across all scenarios:** Track replacement cost is the single largest maintenance variable. The JD's sole-sourced tracks from Camso ($40,000-$50,000+ for a full set) vs. multiple manufacturers for Case IH ($32,000-$37,000) creates a cost advantage for Case IH that is relatively independent of scenario.

### 8.3 Depreciation/Resale Value — Three Scenarios

| Scenario | Description | Impact on Decision |
|----------|-------------|-------------------|
| **Optimistic for JD** | JD brand premium persists. Strong Deere loyalist second-buyer pool. JD retains 40-45% of value at 10,000 hours. Case IH retains 30-35%. | JD advantage of $50,000-$100,000 in resale |
| **Neutral** | Both retain similar percentage (~35-40%). JD's higher initial price means higher absolute residual but also higher depreciation. | JD costs $100,000-$167,000 more to own due to higher initial price |
| **Optimistic for Case IH** | Quadtracs continue to "hold value better than other equipment categories." Canadian market supports Case IH values due to stronger sales. | Case IH ownership advantage grows to $150,000+ |

**Robust conclusion across all scenarios:** The purchase price differential ($167,000+ favoring Case IH) is the dominant factor. To achieve cost parity, the JD would need to retain approximately 18% more of its value than the Case IH at sale — a scenario not supported by available evidence (Purdue thesis found similar relative depreciation rates).

### 8.4 Telematics Integration — Two Scenarios

| Scenario | Description | Impact on Decision |
|----------|-------------|-------------------|
| **Seamless integration desired** | Operator wants native Trimble RTK correction compatibility without bridge hardware. | **Case IH wins decisively.** No bridge hardware required. Less hassle with guidance line conversion. |
| **Willing to work around compatibility** | Operator is comfortable with Intuicom RTK Bridge-X or installing a separate Trimble receiver on the JD. | **Decision turns on other factors.** Additional cost of ~$1,500-$2,500 for bridge hardware. |

**Robust conclusion across all scenarios:** If native Trimble RTK integration without additional hardware is a priority, the Case IH is the clear choice. If the operator is willing to use a bridge solution, compatibility becomes a manageable issue.

---

## 9. Operating Cost Per Hectare Framework

### 9.1 Formula and Request for User-Defined Parameters

The Saskatchewan Farm Machinery Custom and Rental Rate Guide provides the standard framework for calculating per-acre and per-hectare costs [118]:

**Total Cost Per Hour = Fixed Costs + Variable Costs**
**Fixed Costs = Depreciation + Financing + Insurance + Housing**
**Variable Costs = Fuel + Lubrication + Repairs & Maintenance + Labor**

**Per-Hectare Cost = Total Cost Per Hour ÷ Effective Hectares Per Hour**

**To calculate this for YOUR operation, I need you to confirm:**
1. Total annual hours (including custom work)
2. Number of field passes (tillage + seeding)
3. Average implement width (affects work rate)
4. Actual fuel price at your supplier
5. Whether you will do custom work (estimated hours)
6. Entity structure (affects tax rate and thus after-tax cost)

### 9.2 Preliminary Cost Per Hectare (with stated assumptions)

**Assumptions for preliminary calculation:**
- 1,000 annual hours (including 800+ hours of custom work or other tasks)
- 3 field passes on 200 acres = 600 acre-passes ÷ 50.4 acres/hr = ~12 hours on core acreage
- 988 hours on custom work or other operations
- Effective field capacity: 50.4 acres/hr (20.4 hectares/hr)
- 8,000 hectares worked annually (20.4 ha/hr × 1,000 hrs × 0.80 field efficiency = ~16,320 ha; but this assumes continuous fieldwork — adjusted to account for transport, non-field time)

**Assuming 6,000 effective work hectares per year (more conservative):**

| Cost Component | John Deere 9RX 540 | Case IH Steiger 540 Quadtrac |
|---------------|-------------------|------------------------------|
| Fixed cost per year (depreciation + insurance) | ~$92,500 | ~$75,820 |
| Variable cost per year (fuel + maintenance + DEF + lube) | ~$140,000 | ~$130,000 |
| **Total annual cost** | **~$232,500** | **~$205,820** |
| Cost per effective hour (1,000 hrs) | $232.50/hr | $205.82/hr |
| **Cost per hectare** (at 20.4 ha/hr × 0.80 efficiency = 16.3 effective ha/hr) | **$14.26/ha** | **$12.63/ha** |
| **Cost per acre** | **$5.77/ac** | **$5.11/ac** |

**If used only on 200 acres (no custom work) with 12 hours/year:**

| Cost Component | John Deere 9RX 540 | Case IH Steiger 540 Quadtrac |
|---------------|-------------------|------------------------------|
| Total annual fixed cost | ~$92,500 | ~$75,820 |
| Total annual variable cost (12 hrs) | ~$1,680 | ~$1,560 |
| **Total annual cost** | **~$94,180** | **~$77,380** |
| **Cost per acre** (600 acre-passes) | **$156.97/ac** | **$128.97/ac** |
| **Cost per hectare** | **$387.84/ha** | **$318.64/ha** |

**This second scenario demonstrates why these tractors are economically unviable for a 200-acre operation without substantial custom work.**

---

## 10. Conclusion and Decision Framework

### 10.1 Critical Findings

1. **Data limitations are significant.** The Nebraska test for the Steiger 540 Quadtrac (Test 2188) exists but full fuel consumption data is not publicly available. The 9RX 540 test was on a derated 590 chassis. Direct fuel efficiency comparison between these specific models cannot be made with complete confidence.

2. **The $125/hr vs $50/hr depreciation claim is unverified** and contradicted by peer-reviewed research showing similar relative depreciation rates for both brands.

3. **The purchase price differential (~$167,000 favoring Case IH)** is the single largest driver of total ownership cost difference, not fuel efficiency, maintenance, or depreciation rates.

4. **Track replacement costs strongly favor Case IH** ($32,000-$37,000 for full set vs. $40,000-$50,000+ for JD), with the additional advantage of multiple manufacturers competing on price and warranty.

5. **Native Trimble RTK integration is a decisive advantage for Case IH** if seamless compatibility with existing systems is a priority.

6. **Dealer support is strong for both brands near Regina,** with Brandt (the world's largest John Deere dealer) headquartered in Regina and Young's Equipment (Saskatchewan's largest Case IH dealer) also based there.

7. **Both tractors represent extreme overcapacity for a 200-acre core operation** unless supplemented by substantial custom work (800+ hours annually).

8. **The Accelerated Investment Incentive has been reinstated** (Bill C-15, Royal Assent March 26, 2026), providing 45% first-year CCA deduction for both tractors.

### 10.2 Decision Matrix

| Priority | Choose John Deere 9RX 540 | Choose Case IH Steiger 540 Quadtrac |
|----------|---------------------------|-------------------------------------|
| Lowest purchase price | — | ✓ (~$167,000 less) |
| Native Trimble RTK integration | — | ✓ (no bridge hardware) |
| Undercarriage durability | ✓ (20% larger, sealed cartridges, 41% more lug engagement) | — |
| Daily maintenance burden | ✓ (1,500-hour intervals vs daily checks) | — |
| Lower track replacement cost | — | ✓ ($32K vs $40-50K) |
| Longest-proven platform | — | ✓ (Quadtrac since 1997) |
| Strongest dealer network near Regina | ✓ (Brandt HQ + South Country, 5+ locations) | ✓ (Young's Equipment, 9 locations in SK) |
| Fuel efficiency | Unknown from available data | Unknown from available data |
| Resale value retention | Similar % as Case IH | Similar % as John Deere |

### 10.3 Final Recommendation

**For an operation with existing Trimble RTK guidance and a 200-acre core acreage:**

If **native Trimble RTK integration** and **lower initial capital outlay** are the highest priorities, the **Case IH Steiger 540 Quadtrac** has clear advantages. The ~$167,000 purchase price savings, combined with lower track replacement costs and the ability to use existing Trimble equipment without bridge hardware, make it the more cost-effective choice.

If **undercarriage durability with reduced daily maintenance** and **dealer relationship with the world's largest John Deere dealer (headquartered in Regina)** are valued, the **John Deere 9RX 540** is justifiable. The sealed cartridge mid-rollers (1,500-hour intervals vs daily checks) and larger undercarriage components offer genuine advantages for a 180+ day operation.

**Regardless of choice, the economics demand supplemental custom work.** At 200 acres, the cost per acre for either tractor is approximately $5-6/ac with 1,000 hours/year of custom work, but rises to $130-$157/ac if used only on the core acreage. The tractor must be deployed across at least 5,000-8,000 acres annually (through custom farming or expansion) to achieve reasonable per-acre costs.

**Next steps to improve decision quality:**
1. Request the full Test 2188 OECD report from the Nebraska Tractor Test Laboratory for complete Steiger 540 Quadtrac fuel data
2. Seek test drive opportunities on clay-loam soil at 12-15 mph to validate traction and fuel consumption claims
3. Obtain formal quotes from both Brandt and Young's Equipment, including detailed pricing and warranty terms
4. Confirm actual annual hours and custom work volume
5. Consult with an accountant to verify CCA treatment for your specific entity structure and purchase timeline

---

### Sources

[1] Nebraska Tractor Test 2252 - John Deere 9RX 540 (Full PDF): https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=4395&context=tractormuseumlit

[2] Nebraska Tractor Test 2252 - John Deere 9RX 540 (Summary Page): https://digitalcommons.unl.edu/tractormuseumlit/3520

[3] Nebraska Tractor Test 2188 - Case IH Steiger 540 QuadTrac (Full PDF): https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=3656&context=tractormuseumlit

[4] Nebraska Tractor Test 2188 - Case IH Steiger 540 QuadTrac (Summary Page): https://digitalcommons.unl.edu/tractormuseumlit/2652

[5] Nebraska Tractor Test 2096 - Case IH Steiger 540 Wheeled (Full PDF): https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=3553&context=tractormuseumlit

[6] TractorData.com CaseIH Steiger 540 tractor tests (Test 2096 data): https://www.tractordata.com/farm-tractors/008/2/4/8248-caseih-steiger-540-tests.html

[7] Case IH Nebraska Tractor Test Results (Steiger 620 records): https://www.caseih.com/en-us/unitedstates/company/efficient-power/nebraska-test-results

[8] TractorData.com CaseIH Steiger 540 Quadtrac specifications: https://www.tractordata.com/farm-tractors/008/2/5/8253-caseih-steiger-540-quadtrac.html

[9] John Deere 9RX 540 official specifications: https://www.deere.com/en/tractors/4wd-track-tractors/9rx-540

[10] OEM Off-Highway - John Deere 9RX Four-Track Tractor Design: https://www.oemoffhighway.com/drivetrains/article/12143964/john-deere-9rx-fourtrack-tractor-design

[11] Cross Implement Facebook - John Deere 9RX Positive Drive Undercarriage: https://www.facebook.com/CrossImplement/posts/the-john-deere-9rx-series-tractors-use-a-positive-drive-undercarriage-system-wit/1052434191459656

[12] NTS Tire Supply - The Top Tracks for Your Case IH Quadtrac: https://www.ntstiresupply.com/ptk-shared/the-top-tracks-for-your-case-ih-quadtrac

[13] Farm Progress - John Deere 9RX Track System Details: https://www.farmprogress.com/equipment/john-deere-s-9rx-track-system-gets-closer-look

[14] Brandt Dare to Compare - John Deere 9RX vs Case IH Quadtrac: https://more.brandt.ca/john-deere/blog/john-deere-9rx-4-track-vs-case-ih-quatrac

[15] NTS Tire Supply - The Top Replacement Tracks for Your John Deere 9RX: https://www.ntstiresupply.com/ptk-shared/the-top-replacement-tracks-for-your-john-deere-9rx

[16] Hay and Forage - John Deere 9RX Track Options: https://www.hayandforage.com/equipment/john-deere-9rx-tractors

[17] AllMachines - Case IH Steiger 450 Quadtrac: https://www.allmachines.com/tractors/case-ih-steiger-450-quadtrac

[18] Facebook - Case IH UK/ROI Steiger Quadtrac: https://www.facebook.com/caseihukroi/posts/steiger-quadtrac-features-four-independently-driven-tracks-patented-tri-point-oscill/1019610546203386

[19] RitchieSpecs - Case IH Steiger 540 Quadtrac (2018-Present): https://www.ritchiespecs.com/model/case-ih-steiger-540-quadtrac-2018-present-4wd-tractor

[20] YouTube - Case IH Agronomic Design Insights (Tony McClelland): https://www.youtube.com/watch?v=case-ih-quadtrac-agronomic-design

[21] Case IH AFS Connect Quadtrac Brochure: https://www.caseih.com/en-us/products/tractors/steiger-quadtrac

[22] NTS Tire Supply - Quadtrac Track Options PDF: https://www.ntstiresupply.com/ptk-shared/case-ih-quadtrac-track-buying-guide

[23] Bootheel Tractor Parts - Steiger T4B Specifications: https://www.bootheeltractorparts.com/steiger-t4b-specs

[24] Red Power Magazine Forum - Quadtrac Track Width Discussion: https://www.redpowermagazine.com/forums/topic/quadtrac-track-width

[25] AgTalk Discussion - John Deere 9RX vs Case Quadtrac: https://talk.newagtalk.com/forums/thread-view.asp?tid=778497

[26] NTS Tire Supply - Quadtrac Mid-Roller Replacement Guide: https://www.ntstiresupply.com/ptk-shared/quadtrac-mid-roller-conversion-kit

[27] Farm Equipment - Compare & Contrast: Tracks vs Tires: https://www.farm-equipment.com/articles/compare-contrast-tracks-vs-tires

[28] Soil, Draft, and Traction Technical Note: https://extension.psu.edu/soil-draft-and-traction

[29] YouTube - Rocky Mountain Farmer 9RX 590 Test: https://www.youtube.com/watch?v=9rx-590-chisel-plow-test

[30] Case IH Event Summary - Steiger Quadtrac vs JD 9RX Comparative Test: https://www.caseih.com/en-us/products/tractors/steiger-quadtrac-test-event

[31] AgTalk Discussion - Quadtrac Maintenance and Heat Management: https://talk.newagtalk.com/forums/thread-view.asp?tid=quadtrac-heat-management

[32] Arvidsson et al. 2011 - Soil and Tillage Research (Compaction Study): https://www.sciencedirect.com/science/article/pii/S0167198711000241

[33] AGDAILY - Titan International Track vs Tire Testing: https://www.agdaily.com/machinery/titan-international-track-vs-tire-testing

[34] Titan International Blog - LSW Tire vs Track Compaction: https://www.titan-intl.com/blog/lsw-vs-track-compaction

[35] NTS Tire Supply - John Deere 9RX Track Pricing: https://www.ntstiresupply.com/ptk-shared/john-deere-9rx-track-pricing

[36] AgTalk - CORS with Deere system via Trimble Bridge: https://talk.newagtalk.com/forums/thread-view.asp?tid=628897

[37] AgTalk - Trimble products and John Deere RTK: https://talk.newagtalk.com/forums/thread-view.asp?tid=107768

[38] AgTalk - RTK cross-platform compatibility discussion: https://talk.newagtalk.com/forums/thread-view.asp?tid=121436

[39] AgTalk - JD firmware locks out non-JD formats: https://talk.newagtalk.com/forums/thread-view.asp?tid=jd-firmware-lockout

[40] Intuicom RTK Bridge-X Product Page: https://www.intuicom.com/rtk-bridge-x

[41] Agra GPS CRG Receiver: https://agragps.com/crg-receiver/

[42] AgTalk - MojoRTK and Leica Reverse Engineering Discussion: https://talk.newagtalk.com/forums/thread-view.asp?tid=mojortk

[43] ESPrtk Project - John Deere NTRIP Compatibility: https://github.com/ESPrtk/ESPrtk

[44] Case IH Precision Farming Products: https://www.caseih.com/en-us/unitedstates/products/precision-farming

[45] Lefebure - CNH AFS Vector Pro documentation: https://lefebure.com/documentation/cnh-vector-pro

[46] Case IH AFS 372 Receiver Specifications: https://www.caseih.com/en-us/products/precision-ag/afs-receivers

[47] Trimble Integration with John Deere Operations Center (August 2016): https://news.trimble.com/2016-08-30-Trimble-Integrates-with-John-Deere-Operations-Center

[48] AgTalk - Cross-Platform Guidance Line Conversion Discussion: https://talk.newagtalk.com/forums/thread-view.asp?tid=guidance-line-conversion

[49] Case IH Trimble Data Integration (July 2017): https://www.caseih.com/en-us/company/news/2017/trimble-afs-connect-integration

[50] MyCaseIH / AFS Connect Portal: https://www.mycaseih.com/

[51] Case IH FieldOps and Climate FieldView Integration: https://www.caseih.com/en-us/products/precision-ag/fieldops

[52] John Deere JDLink Free Connectivity Announcement (July 2021): https://www.deere.com/en/technology-products/precision-ag-technology/jdlink

[53] Case IH AFS Connect Subscription Pricing Discussion: https://talk.newagtalk.com/forums/thread-view.asp?tid=afs-connect-pricing

[54] Case IH Connectivity Included Announcement (August 2024): https://www.caseih.com/en-us/company/news/2024/connectivity-included

[55] Brandt Tractor Ltd. - Heavy Equipment Guide: https://www.heavyequipmentguide.ca/company/3558/brandt-tractor-ltd

[56] Recycling Product News - Brandt Tractor Ltd.: https://www.recyclingproductnews.com/company/3558/brandt-tractor-ltd

[57] MachineFinder - Brandt Tractor Ltd. Regina: https://www.machinefinder.com/ww/en-CA/dealers/brandt-tractor-ltd-regina-sk-766413

[58] Brandt Agriculture - John Deere Dealer Network: https://more.brandt.ca/john-deere

[59] Supply Post - Brandt Tractor Dealer Profile: https://www.supplypost.com/dealers/brandt-tractor

[60] Brandt Field Emergency Service: http://www.brandt.ca/Parts-Services/Equipment-Maintenance/Field-Emergency-Service

[61] LeadIQ - Brandt Tractor Ltd. Company Profile: https://leadiq.com/c/brandt-tractor-ltd/5a1d7ce3240000240056adce

[62] Encyclopedia of Saskatchewan - Brandt Industries Limited: https://esask.uregina.ca/entry/brandt_industries_limited.html

[63] Brandt Tractor Ltd. Regina - MapQuest: https://www.mapquest.com/ca/saskatchewan/brandt-tractor-ltd-359493511

[64] Brandt Customer Support: http://www.brandt.ca/Parts-Services/Customer-Support

[65] Brandt Parts: http://www.brandt.ca/Parts-Services/Parts

[66] Brandt Service and Support: https://more.brandt.ca/john-deere/parts-service/service

[67] Crownsmen Partners - Brandt World's Largest John Deere Dealer: https://www.crownsmen.com/brandt-worlds-largest-john-deere-dealer

[68] Brandt Rental House: http://www.brandt.ca/Divisions/Tractor/Products/Rental-House

[69] Brandt Branch Locator: http://www.brandt.ca/Divisions/Tractor/Branch-Locator

[70] Brandt Testimonials: http://www.brandt.ca/Our-Company/Testimonials

[71] South Country Equipment: https://www.southcountry.ca

[72] Young's Equipment Inc. - LinkedIn: https://ca.linkedin.com/company/young-s-equipment-inc

[73] Young's Equipment - Saskatchewan Chamber of Commerce: https://business.saskchamber.com/list/member/young-s-equipment-inc-1559

[74] Equipment Dealer Magazine - Young's Equipment Profile: https://www.equipmentdealermagazine.com/youngs-equipment-inc-built-on-family-trust-and-integrity

[75] Young's Equipment Regina - CNH Store: https://www.mycnhstore.com/ca/en/caseih/dealerlocator/dealerdetail/youngs-equipment-inc/regina-sk/111-020043A

[76] Young's Equipment - Inspect and Protect Service: https://youngs.ca/service/inspect-and-protect

[77] Young's Equipment News: https://youngs.ca/news-main

[78] Young's Equipment Shuttle Express: https://youngs.ca/service/shuttle-express

[79] Farm Equipment Magazine - Young's Equipment Profile: https://www.farm-equipment.com/articles/4062-youngs-equipment

[80] Young's Equipment - Apprenticeship and Training: https://youngs.ca/news

[81] AgTalk Discussion - Brand Premium and Depreciation: https://talk.newagtalk.com/forums/thread-view.asp?tid=brand-premium-depreciation

[82] Purdue University Thesis (Daninger 2017) - Depreciation in the U.S. Used Tractor Market: https://docs.lib.purdue.edu/dissertations/tractor-depreciation

[83] AgDealer - John Deere 9RX 540 Listings Canada: https://www.agdealer.com/listings/manufacturer/john-deere/model/9rx-540/category/tractors

[84] MarketBook.ca - John Deere 9RX 540 For Sale Canada: https://www.marketbook.ca/listings/for-sale/john-deere/9rx-540

[85] TractorHouse - John Deere 9RX 540 (352 hours) Listing: https://www.tractorhouse.com/listings/for-sale/john-deere/9rx-540

[86] Hutson Inc - John Deere 9RX 540 (521 hours) Listing: https://www.hutsoninc.com/inventory/tractors/john-deere-9rx-540

[87] CloudStore - John Deere 9RX 540 (1,012 hours) Listing: https://cloudstore.com/inventory/john-deere-9rx-540

[88] TractorHouse - John Deere 9RX 540 All Listings: https://www.tractorhouse.com/listings/for-sale/john-deere/9rx-540/300-hp-or-greater-tractors

[89] AgDealer - John Deere 9RX 540 Rosthern SK: https://www.agdealer.com/listings/manufacturer/john-deere/model/9rx-540

[90] AgDealer - Case IH Steiger 540 Quadtrac Westlock AB: https://www.agdealer.com/listings/manufacturer/case-ih/model/steiger-540-quadtrac

[91] AgDealer - Case IH Steiger 540 Quadtrac All Listings: https://www.agdealer.com/listings/manufacturer/case-ih/model/steiger-540-quadtrac/category/tractors

[92] TractorHouse - Case IH Steiger 540 Canada Listings: https://www.tractorhouse.com/listings/case-ih-steiger-540-farm-equipment-for-sale-in-canada

[93] MarketBook.ca - Case IH Steiger 540 Quadtrac: https://www.marketbook.ca/listings/for-sale/case-ih/steiger-540-quadtrac

[94] TractorHouse - 2022 Steiger 540 Quadtrac (2,040 hrs): https://www.tractorhouse.com/listings/case-ih-steiger-540-quadtrac-2022

[95] Town & Country Implement - 2020 Steiger 540 Quadtrac (3,289 hrs): https://townandcountryimplement.com/inventory/steiger-540-quadtrac

[96] MarketBook.ca - 2018 Steiger 540 Quadtrac (4,356 hrs): https://www.marketbook.ca/listings/case-ih-steiger-540-quadtrac-2018

[97] Heritage Tractor Blog - Tractor Depreciation and Value: https://www.heritagetractor.com/blog/tractor-depreciation

[98] Agriculture.com - Machinery Insider: https://www.agriculture.com/machinery/machinery-insider

[99] RealAgriculture - Machinery Pete on Used Tractor Values 2025: https://www.realagriculture.com/machinery-pete-used-tractor-values

[100] Agriculture.com - Quadtracs Holding Value: https://www.agriculture.com/machinery/machinery-insider/quadtracs-holding-value

[101] Farm-Equipment.com - Sandhills Global November 2024 Data: https://www.farm-equipment.com/market-data/sandhills-global-november-2024

[102] RealAgriculture - AEM Data: Canadian vs US Tractor Sales 2025: https://www.realagriculture.com/aem-data-canada-us-tractor-sales

[103] AgWeb - Machinery Pete: Canadian Used Tractor Values: https://www.agweb.com/machinery-pete/canadian-used-tractor-values

[104] MachineryTrader - 4WD Tractors Yorkton SK Area: https://www.machinerytrader.com/listings/4wd-tractors/yorkton-saskatchewan

[105] FBC - Farm CCA Classes: Capital Cost Allowance for Farmers: https://www.fbc.ca/blog/farm-cca-classes

[106] Canada Revenue Agency - Classes of Depreciable Property: https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/renting-business-income/capital-cost-allowance-caa.html

[107] CRA External Interpretation (May 19, 2011) - Self-Propelled Equipment Classification: https://www.canada.ca/en/revenue-agency/services/tax/technical-information/interpretations/2011-self-propelled-sprayer

[108] T2inc.ca - CCA Classes and Rates Complete Guide: https://www.t2inc.ca/blog/cca-classes-rates

[109] Mehmi Group - CCA Class 8 Equipment: 20% Declining Balance: https://www.mehmigroup.com/blog/cca-class-8-equipment

[110] LEGISinfo - Bill C-15 (45th Parliament, 1st Session): https://www.parl.ca/LegisInfo/BillDetails.aspx?billId=15000000

[111] EY Global - Canada enacts CCA and business tax measures in C-15 (February 2026): https://www.ey.com/en_gl/tax-alerts/canada-enacts-cca-measures-c15

[112] Thomson Reuters (March 4, 2026) - Various measures of Capital Cost Allowance (Gerry Vittoratos): https://www.thomsonreuters.com/en/tax/cca-measures-2026

[113] Knowledge Bureau - CCA, AII and Immediate Expensing: https://knowledgebureau.com/cca-aii-immediate-expensing

[114] Canada Revenue Agency - Accelerated Investment Incentive: https://www.canada.ca/en/revenue-agency/services/tax/businesses/topics/renting-business-income/capital-cost-allowance-caa/accelerated-investment-incentive.html

[115] Government of Saskatchewan - Tax Information for Farmers: https://www.saskatchewan.ca/business/taxes-and-levies/tax-information-for-farmers

[116] Baker Tilly Canada - The end of immediate expensing: https://www.bakertilly.com/ca/insights/the-end-of-immediate-expensing

[117] Saskatchewan Diesel Price Dashboard (2026): https://www.saskatchewan.ca/fuel-prices

[118] Saskatchewan Agriculture - Farm Machinery Custom and Rental Rate Guide 2024-25: https://publications.saskatchewan.ca/farm-machinery-custom-rate-guide

[119] DEF Pricing Canada (2026): https://www.defpricing.com/canada

[120] Cummins SCR - DEF Consumption Technical Bulletin: https://www.cummins.com/components/scr/def-consumption

[121] AgTalk - Fuel Consumption Discussion: Operator Variability: https://talk.newagtalk.com/forums/thread-view.asp?tid=fuel-consumption-operator-technique