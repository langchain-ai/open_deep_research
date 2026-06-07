# Comprehensive Technical Analysis: Digital Identity Standards and Systems (Revised and Updated)

## Executive Summary

This report provides a detailed comparative analysis of three major digital identity paradigms: (1) W3C Decentralized Identifiers (DIDs) and Verifiable Credentials, (2) FIDO2/WebAuthn, and (3) government-issued digital ID systems (EU eIDAS, India's Aadhaar, Estonia's e-Residency). Each system embodies fundamentally different architectural philosophies—decentralized self-sovereign identity, hardware-backed authentication, and state-issued foundational identity—with distinct trade-offs across privacy, interoperability, surveillance resistance, recovery mechanisms, and use case suitability.

This revision incorporates significant developments through May 31, 2026, including: W3C DID v1.1 Candidate Recommendation (March 2026), Verifiable Credentials Data Model v2.0 Recommendation (May 2025), WebAuthn Level 3 Candidate Recommendation (May 2026), the FIDO Alliance's State of Passkeys 2026 report showing 5 billion passkeys in use, EU eIDAS 2.0 implementation progress with Architecture Reference Framework v2.8, India's Aadhaar Vision 2032 and DPDP Rules 2025, and Estonia's record-breaking e-Residency revenue of €125 million in 2025.

---

## 1. Overview of Each System

### 1.1 W3C Decentralized Identifiers (DIDs) and Verifiable Credentials

The W3C Decentralized Identifiers (DIDs) v1.0 specification reached W3C Recommendation status on July 19, 2022, establishing a standardized method for creating verifiable, decentralized digital identifiers that operate independently of centralized registries, identity providers, and certificate authorities [1]. **DID v1.1**, published as a Candidate Recommendation Snapshot on March 5, 2026, consolidates media types, introduces new JSON-LD contexts, and restructures related standards [2]. The specification now has 100 participants from 37 organizations across government and commercial sectors [3].

The **Verifiable Credentials Data Model v2.0** became a W3C Recommendation on May 15, 2025, alongside six companion specifications [4]. This represents a significant milestone—the entire VC 2.0 family of seven W3C Recommendations provides cryptographically secure, privacy-respecting, and machine-verifiable digital credentials for identity documents and records [5]. Key changes from v1.1 include: renamed properties (validFrom/validUntil replacing issuanceDate/expirationDate), new media types (application/vc and application/vp), clarified separation of the data model from securing mechanisms, and mandatory JSON-LD context updates [6].

The **Verifiable Credentials Working Group** is operating under a charter from March 11, 2026 to March 31, 2028, tasked with delivering new normative specifications including Verifiable Credential Render Method v1.0, Confidence Method v1.0, Credential API, Verifiable Credential Barcodes, and Data Integrity Cryptosuites such as BBS+ signatures. Tentative deliverables include work on wireless protocols, credential refresh, and quantum-safe cryptosuites [7].

The DID ecosystem supports multiple DID methods with distinct trade-offs: **did:key** (self-contained, no external registry, but immutable and non-rotatable), **did:peer** (pairwise relationships, off-chain), **did:web** (domain-based, heavily criticized for centralization), **did:ethr** (Ethereum blockchain), **did:indy** (Hyperledger Indy), **did:ion** (Sidetree protocol on Bitcoin), and **did:dht** (distributed hash table) [8]. The **DID Resolution v0.3** specification (Working Draft, May 25, 2026) defines a standard interface for resolving DIDs to DID documents [9].

### 1.2 FIDO2/WebAuthn

FIDO2 is a suite of standards comprising the W3C Web Authentication (WebAuthn) API and the FIDO Alliance's Client to Authenticator Protocol (CTAP). WebAuthn provides a standard web API built into all major browsers that enables passwordless and multi-factor authentication using public-key cryptography [10]. CTAP defines communication protocols between client devices and authenticators (hardware security keys, platform biometrics). Credentials are cryptographically bound to specific domain origins, making phishing attacks technically impossible [11].

**WebAuthn Level 3** was published as an updated Candidate Recommendation Snapshot on May 26, 2026, with comments invited through June 23, 2026 [12]. This update formalizes key behaviors around passkeys, multi-device credentials, and authentication flows. Level 3 introduces: new client APIs such as getClientCapabilities() for standardized feature detection, JSON serializers/deserializers, signal methods for enhanced authenticator interaction, attestation clarifications, and explicit iframe and cross-origin login support. The specification elevates previously vendor-specific features into a formal standard—passkeys and multi-device credentials are now first-class behaviors rather than extensions [13]. A separate transition request confirmed that no new features will be incorporated into Level 3; Level 4 will be the focus for future enhancements [14].

The FIDO Alliance's **State of Passkeys 2026** report, released May 7, 2026, revealed that **5 billion passkeys are now in active use globally**—a remarkable adoption milestone [15]. Key statistics from the survey of 11,000 adults across ten countries and 1,400 enterprise decision-makers include: 90% consumer awareness of passkeys, 75% having enabled them on at least one account, 68% of organizations deploying or actively deploying passkeys for employee sign-ins, and 49% of consumers using passkeys regularly [15]. Amazon alone reported 465 million customers using passkeys [16]. Over 3 billion user accounts are actively secured with passkeys, with more than 2,000 certified FIDO products [17].

**Passkeys**—the consumer-facing implementation of discoverable FIDO2 credentials—are synced across devices via cloud keychains (Apple iCloud Keychain, Google Password Manager, Microsoft Authenticator, and third-party providers like 1Password, Dashlane, and Bitwarden) [18]. The **Credential Exchange Protocol (CXP)** and **Credential Exchange Format (CXF)**, developed by the FIDO Alliance with Apple, Google, Microsoft, 1Password, Bitwarden, and Dashlane, define a standardized, secure format for transferring credentials between credential managers. CXF (Review Draft) standardizes a JSON-based data format for credentials, while CXP (Working Draft) defines a secure transfer protocol using Hybrid Public Key Encryption (HPKE). Target completion for both is Q2 2026 [19]. Apple has already shipped CXF-based credential transfers in iOS 26 and macOS Tahoe 26 [20].

### 1.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0 (Regulation (EU) 2024/1183)** entered into force on May 20, 2024, mandating that all 27 EU Member States must offer citizens and residents an EU Digital Identity Wallet (EUDI Wallet) free of charge by December 31, 2026 [21]. The regulation introduces Qualified Electronic Attestations of Attributes (QEAA), qualified electronic archiving, and qualified electronic ledgers as new trust services [22]. Large online platforms must accept the EUDI Wallet by late 2027 [23]. The adoption target set by the EU Digital Decade Programme is 80% wallet usage by 2030 [24].

The EUDI Wallet enables selective disclosure of attributes, qualified electronic signatures with the same legal weight as handwritten signatures, and pseudonymous authentication. The regulation explicitly prohibits tracking and profiling, requires privacy dashboards for user transparency, and mandates local data storage on user devices [25]. Four large-scale pilot projects involving over 350 entities from 25 Member States plus Norway, Iceland, and Ukraine have tested the wallet across 11 practical use cases including government services, banking, SIM registration, digital signatures, prescription claims, travel, and education [26].

As of May 2026, member state readiness is uneven. Only approximately 25% of EU member states are on track to meet the December 2026 deadline [27]. Top-tier countries include France (France Identité with 4 million users), Italy (CIE ID with over 40% penetration), Poland (mObywatel 2.0), and Austria. Germany will launch its state-driven EUDI Wallet on January 2, 2027, demonstrating a slight delay [28][29]. The Architecture Reference Framework (ARF) is at version 2.8, with 31 published implementing acts. A major recent milestone: the remote onboarding regulation (Commission Implementing Regulation (EU) 2026/798) was published on April 8, 2026 [30][31].

**India's Aadhaar** is operated by the Unique Identification Authority of India (UIDAI). As of December 2025, Aadhaar has approximately 1.34 billion active holders and has completed more than 160 billion authentication transactions since launch [32]. In 2026, UIDAI launched a major Aadhaar mobile app update (21 million downloads in three months), expanded offline e-KYC capabilities through Offline Verification Seeking Entities (OVSE), and unveiled the **Aadhaar Vision 2032** framework—an 11-member strategic roadmap chaired by Neelkanth Mishra focusing on AI, blockchain, quantum computing, and next-generation encryption [33][34][35].

The PAN-Aadhaar linking mandate became fully effective on January 1, 2026, with unlinked PANs becoming inoperative for tax filing, refunds, and banking activities [36]. Aadhaar's integration with Google Wallet was announced on April 28, 2026, using W3C Digital Credentials API and ISO/IEC 18013-5 standards, covering nearly the entire adult population of India [37].

**Estonia's e-Residency and Electronic Identity** launched in December 2014 and has issued over 140,000 e-resident digital identities from more than 185 countries, with over 41,800 Estonian companies established by e-residents [38]. In 2025, the program achieved record revenue of €124.9 million—an 87% increase from the previous year—with cumulative economic impact reaching nearly €400 million. Every euro invested returned more than twelve [39][40]. E-residents established 5,556 new companies in 2025, a 15% increase from 2024 [41].

Estonia announced plans to transition from physical ID cards to a **cardless, fully mobile-based system by 2027**, using biometric authentication via smartphone. This is expected to increase company formation by at least 20% and bring additional annual tax revenue of €3–9 million [42]. The underlying X-Road infrastructure handles over 2.2 billion transactions yearly across approximately 52,000 organizations, and has inspired implementations in over 20 countries including Finland, Iceland, and Japan [43].

Estonia's **Smart-ID**—an app-based digital identity solution—has nearly 3.3 million active users across the Baltic states and processes up to 85 million transactions monthly. Since February 26, 2026, **Smart-ID+** has been introduced for more secure state authentication, reducing risks from phone scams and social engineering [44][45]. As of April 29, 2026, Smart-ID registration with Estonian ID-cards is available via NFC-enabled smartphones [46].

---

## 2. Approach to Privacy

### 2.1 W3C DIDs and Verifiable Credentials

The VC Data Model v2.0 explicitly prioritizes privacy, supporting selective and unlinkable disclosure through multiple mechanisms [47]. The DID v1.1 specification applies Privacy by Design principles throughout and cautions against including encrypted personal data in DID documents [2].

**Selective Disclosure Mechanisms:**

- **BBS+ Signatures:** The "Data Integrity BBS Cryptosuites v1.0" (W3C Candidate Recommendation Draft, April 7, 2026) provides selective disclosure and unlinkable derived proofs using pairing-based cryptography. BBS enables holders to generate unique proofs from the same issued signature while preventing correlation across presentations. The suite supports anonymous holder binding and credential-bound pseudonyms [48].

- **SD-JWT (Selective Disclosure JSON Web Tokens):** RFC 9901, published by the IETF, introduces SD-JWTs where sensitive claims are replaced with salted hashes and real claim values are moved into encrypted "Disclosures," allowing selective revelation. SD-JWT+KB adds holder authentication to prevent theft and replay attacks. SD-JWT offers advantages in ease of implementation and interoperability due to its reliance on widely-used JWTs and cryptographic algorithms [49]. It is actively supported in the EUDI Wallet program and preferred in eIDAS 2.0 regulations [50].

- **Zero-Knowledge Proofs (ZKPs):** zk-SNARKs and zk-STARKs enable proving statements without revealing the inputs used. ZKP technology has moved from experimental to practical infrastructure, with growing institutional attention. Growing regulatory acceptance occurs when privacy solutions enable compliance with AML/KYC requirements [51].

**DID Method Privacy Implications:**

Different DID methods offer varying privacy properties. **did:key** derives identifiers from cryptographic keys with no external storage, offering strong privacy properties but with the critical limitation that keys cannot be rotated [52]. **did:peer** supports pairwise relationships where only involved parties resolve the DIDs, enhancing privacy by lacking third-party data controllers [53]. **did:ethr** reveals the public and immutable nature of blockchain data, with risks from key compromise and correlation via blockchain activity [54].

The **did:web** method has been heavily criticized for its centralization and privacy vulnerabilities. Alex Tweeddale of cheqd (October 2023) described it as "a honey trap for organisations looking to adopt decentralised identity" because it is not actually decentralized, cryptographically verifiable, tamper-resistant, or self-certifying [55]. The "Phone Home" problem arises when verifiers must fetch DID documents from web servers, which can log queries and reveal who is interacting with whom, undermining the core privacy purpose of DIDs [55]. The did:web method does not specify any authentication or authorization mechanism for writing or removing DID documents, relying entirely on securing the web environment [56]. A comprehensive evaluation by Legendary Requirements for the U.S. Department of Homeland Security (March 2022) found that "if the Registry Operator for a given domain is compromised, did:web DIDs for that entire domain should be considered compromised" [57].

**Improvements to did:web** include **did:webs** and **did:webvh** (Verifiable History), which add self-certifying identifiers and verifiable cryptographic history. However, these incremental improvements "do not fully address central vulnerabilities" [55]. Ledger-based DID methods such as EBSI and cheqd have reached maturity where their value extends beyond simple DID resolution, and DID-Linked Resources complement these methods [58].

**Academic Research on Privacy Risks:**

A study assessing the anonymity of blockchain-based DIDs (particularly Ethereum-based) published in CEUR Workshop Proceedings found that "vulnerable fields in DID documents (e.g., serviceEndpoint, publicKeyPem, alsoKnownAs) can be exploited to link multiple DIDs to the same entity" [59]. Experimental evaluation using 31,714 Ethereum transactions and 3,613 distinct DID documents revealed limited but critical use of vulnerable fields, with graph analysis uncovering sparse and disassortative networks centered around major DeFi protocols, NFT marketplaces, and token management [60]. The study emphasizes using unique keys per interaction and encrypting or hashing sensitive metadata [61].

### 2.2 FIDO2/WebAuthn

**Core Privacy Design:**

The FIDO Alliance's six core Privacy Principles include requiring explicit, informed user consent for any operation using personal data, and a requirement that biometric data must never leave the user's personal computing environment [62]. FIDO Authenticator devices must not have a global identifier visible across different websites to prevent unwanted re-identification.

**Per-Origin Key Separation:**

FIDO2 credentials are cryptographically bound to specific domain origins. During registration and authentication, the authenticator generates cryptographic keys tied uniquely to a relying party (the online service), preventing cross-site tracking or global identification of users [62]. This is a fundamental architectural feature—key pairs are never shared across different domains.

**Attestation Models:**

FIDO supports multiple attestation models with varying privacy implications. **Standard (non-enterprise) attestation** uses batch attestation where attestation private keys are shared among devices in a batch (recommended >100,000 devices) to prevent fingerprinting. The AAGUID (Authenticator Attestation GUID) must be identical across all substantially identical authenticators, ensuring attestation identifies the make and model, not an individual device [63]. **Enterprise attestation** intentionally removes this privacy protection, binding a unique authenticator key pair to a serial number or equivalent unique identifier for authorized relying parties with user awareness [63]. By default, the WebAuthn API returns "None" attestation, where the browser removes attestation information for privacy reasons [64].

The Electronic Frontier Foundation (EFF) has concluded that "passkeys satisfy this requirement" for privacy, noting that "each passkey you create is unique... Your fingerprint, face, or unlock code isn't sent to the website. Instead, your browser tells the site that 'user verification' was successful." The EFF further stated: "For most purposes, passkeys will represent a significant improvement in security at nearly zero cost to privacy" [65].

**Passkey Sync Privacy Implications:**

Passkeys synced via iCloud Keychain are end-to-end encrypted with strong cryptographic keys not known to Apple. Google Password Manager encrypts passkeys with a key derived from the user's Android device screen lock, with end-to-end encryption preventing Google access. 1Password provides zero-knowledge vault security with Secret Key and master password [18]. While all major providers claim strong encryption, the existence of credentials across multiple devices and cloud infrastructure expands the attack surface compared to device-bound credentials [66].

A security researcher (Jeff Johnson) in September 2024 discovered that Apple's data export includes a "Passkeys Information.csv" file containing detailed information about passkeys, including creation and last used dates, credential IDs, device information, partial device serial numbers, and the full device UDID stored in plain text. Two passkeys had been silently created in July 2023 without explicit user consent [67]. This raised concerns about transparency and consent in passkey management.

The backup eligibility flag in the FIDO2 specification allows relying parties to determine if a credential is device-bound or syncable, enabling policy decisions such as blocking synced passkeys for higher-security environments [68].

### 2.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

Data minimization is central to the EUDI Wallet design—digital services should collect only the minimum data required. The wallet supports selective disclosure of attributes (e.g., age verification without revealing full identity) and zero-knowledge proofs that allow relying parties to validate statements without revealing underlying data [69][70].

The regulation explicitly prohibits tracking and profiling—issuers are legally forbidden to combine personal data with third-party data. Every wallet includes a built-in privacy dashboard offering an overview of data shared, exactly what data was shared, and transaction history [71]. The wallet supports pseudonymity, generating and storing pseudonyms encrypted and locally. All wallet data is stored locally on the user's device, not in a central database [72]. Parts of the wallet software are open source to foster trust and security [73].

**Critical Privacy Debates:**

A 2025 study in Internet Policy Review examined how electronic attestations include auxiliary cryptographic data (issuer digital signatures, holder binding keys) that act as unique persistent identifiers, creating privacy risks by enabling linkability across transactions. ZKPs can resolve conflicts between the EUDI Wallet's data requirements and GDPR's data minimization principle [74].

Civil society organizations have raised significant concerns. In March 2026, the **INATBA Privacy Working Group** submitted a position warning that the current draft references W3C-VC formally but "lacks the harmonized operational profiles necessary for practical issuance and presentation within the EUDI Wallet ecosystem," leaving W3C-VC "in a position of de jure inclusion but de facto exclusion, undermining privacy-preserving mechanisms such as selective disclosure and unlinkability" [75].

A coalition of **24 civil society organizations** (including Privacy International and EFF) warned that eIDAS "may spell the death of anonymity" and could "introduce a unique and persistent identifier for every citizen allowing Big Tech actors to track their behavior" [76]. **EDRi and nine CSOs** urged the European Commission to amend draft implementing acts, warning that "the Commission is weakening the Wallet safeguards (untraceability and unlinkability) meant to prevent surveillance of people who use it," and imposing "an additional mandatory processing of sensitive biometric facial data" [77].

The **IEU Security and Privacy Workshop 2026** paper systematically identified harms including increased risks of consumer profiling, discrimination, surveillance, exclusion due to digital access requirements, and potential censorship, concluding that "implementation choices, such as device binding and batch-issued credentials, exacerbate privacy harms" [78].

Four categories enable selective disclosure in QEAAs: atomic attributes, multi-message signatures (BBS+, CL, PS-MS providing full unlinkability though not fully standardized), salted attribute hashes, and programmable ZKPs (zk-SNARKs, Bulletproofs). Each has strengths and challenges regarding unlinkability, scalability, and standard maturity [79].

**India's Aadhaar:**

Aadhaar's authentication API returns only Yes/No responses for biometric verification, not raw biometric data. e-KYC returns verified demographic data in an encrypted XML file signed by UIDAI. The 2018 Supreme Court judgment concluded Aadhaar does not create a surveillance state as it collects minimal data and incorporates safeguards including encryption and data storage limits [80].

UIDAI's 2025 Security Guidelines (Circular No. 8 of 2025) mandate AES-256 encryption, RSA-2048, FIPS 140-2 Level 3 certified HSMs, and MeitY-certified hosting. Regulated entities must store Aadhaar numbers in a secure, UIDAI-compliant Aadhaar Data Vault (ADV) with mandatory Reference Key usage to replace Aadhaar numbers in all business databases except the ADV [81][82]. The Virtual ID (VID) is a 16-digit temporary, revocable token that can be used in place of the Aadhaar number for authentication, preventing exposure of the actual Aadhaar number. At any given time only one VID is valid per Aadhaar number [83].

The Aadhaar (Authentication and Offline Verification) Amendment Regulations, 2025 introduced a formalized registration framework for OVSEs with UIDAI oversight, aligning with the Digital Personal Data Protection Act [33][84]. Authentication transaction logs are stored for two years and archived for five years, after which they must be deleted unless legally required [85].

However, critics raise persistent concerns. The centralized biometric database of over 1.4 billion individuals enables mass surveillance capabilities. Edward Snowden has strongly criticized Aadhaar, warning of its potential misuse as a surveillance tool [86]. The Supreme Court's 2017 Puttaswamy judgment established privacy as a fundamental right under Article 21, requiring strict three-fold test (legality, necessity, proportionality) for state interference [87].

The **Digital Personal Data Protection Act (DPDPA), 2023**, was enacted to regulate consent and data localization, with the DPDP Rules notified on November 14, 2025, after nationwide consultations receiving 6,915 inputs. Key features include an 18-month phased compliance timeline, mandatory clear consent notices, breach notification protocols, a digital Data Protection Board, and penalties up to ₹250 crore for non-compliance [88][89][90].

**Estonia's e-Residency:**

Estonia employs a distributed data model where different government agencies maintain their own databases, connected through X-Road. There is no central database—data remains in source systems and is shared only with user consent through tamper-proof mechanisms [43]. This decentralized architecture inherently prevents creating a single point of vulnerability or mass surveillance target. The "once-only" principle means citizens provide data once and government reuses it securely, with full transparency over data access [91]. All data access is logged with real-time audit trails [92].

The KSI Blockchain, developed by Guardtime, guarantees that electronic data authenticity is mathematically provable, making it impervious to manipulation by hackers or even the government itself [93]. Approximately 85% of Estonians report trust in their digital ID-enabled services, including banking and voting [94].

The **Data Embassy** in Luxembourg—the world's first—establishes Estonia's critical data backup in servers outside its borders, protected by international treaties with diplomatic immunity, providing additional resilience against surveillance within national borders [95].

---

## 3. Interoperability

### 3.1 W3C DIDs and Verifiable Credentials

**Standardization Status:**

The W3C VC Data Model and DID Core specifications have reached official recommendation status, providing a stable foundation for implementations [1][4]. The Verifiable Credentials Working Group operates under a 2026-2028 charter to maintain existing specifications and add new functionalities [7]. A newly chartered W3C DID Working Group will deliver standard DID methods and advance interoperability [96].

**Interoperability Challenges:**

Despite significant convergence, many "flavours" of VC-based solutions exist that are not necessarily interoperable despite using the W3C VC data model. Areas of divergence are driven by specific business requirements, regulatory constraints, or different technical philosophies. The existence of over 150 DID methods has been cited as a fragmentation concern—Mozilla's formal objection to DID v1.0 argued that "the DID architectural approach appears to encourage divergence rather than convergence & interoperability" [97].

However, interoperability has improved significantly through initiatives like W3C Test Suites and Plugfests, and the Universal Resolver tool developed by DIF supports resolving many different DID methods [98].

**OpenID4VC Profiles:**

The **OpenID4VC High Assurance Interoperability Profile** targets regulated environments like eIDAS 2.0, requiring support for both pre-auth code flow and authorization code flow, pushed authorization requests, and client authentication using attestation-based JWTs. Cryptographic Holder Binding using KB-JWT is mandatory when presenting an SD-JWT VC [99].

**Cross-Framework Integration:**

The EU Digital Identity Wallet implements DID and verifiable credential standards, ensuring interoperability with the European ecosystem [25]. Live deployments include British Columbia's OrgBook, European Blockchain Services Infrastructure (EBSI) digital diplomas, and U.S. mobile driver's licenses. GS1 identifies promising use cases in supply chains for traceability, compliance, and fraud prevention [100].

The **INATBA Privacy Working Group** emphasized that "a format that is nominally referenced but lacks the necessary presentation and issuance profiles is not meaningfully supported in practice," advocating for Commission-level adaptations for W3C-VC attestations and harmonized presentation support for JSON-LD W3C-VC secured with data integrity proofs [75].

### 3.2 FIDO2/WebAuthn

**Cross-Platform and Cross-Browser Support:**

All major web browsers (Chrome, Safari, Firefox, Edge) support the WebAuthn API, providing a uniform authentication interface since 2019 [10]. FIDO2 is widely supported across major operating systems including Windows, macOS, iOS, and Android. WebAuthn Level 3 further enhances cross-platform interoperability through JSON (de)serializers and standardized feature detection via getClientCapabilities() [12][13].

**FIDO Alliance Certification Infrastructure:**

The FIDO Metadata Service (MDS) acts as a global registry of passkey capabilities, providing relying parties with intelligence to make data-driven decisions. The Metadata BLOB is a single, digitally signed file downloaded periodically (averaging 1-2 updates per week in 2024). The AuthenticatorStatus enum includes statuses like FIDO_CERTIFIED, USER_VERIFICATION_BYPASS, ATTESTATION_KEY_COMPROMISE, and REVOKED [101]. Currently, there is no certification program for passkey providers, though development is in progress [102].

**Credential Exchange Protocol (CXP/CXF):**

The FIDO Alliance's Credential Exchange specifications define a standardized, secure format for transferring credentials—including passkeys and passwords—within credential managers. CXF (Review Draft) standardizes a JSON-based data format for credentials including passkeys, passwords, TOTP secrets, and notes. CXP (Working Draft) defines a secure transfer protocol using Hybrid Public Key Encryption (HPKE) to protect credentials end-to-end during transit. Major industry players actively contributing include Apple, Google, Microsoft, 1Password, Bitwarden, and Dashlane [19].

**Cross-Browser Limitations:**

A notable limitation remains: on macOS, Safari stores passkeys in iCloud Keychain while Chrome stores them in Google Password Manager. A passkey registered in Safari is not automatically visible to Chrome on the same Mac, and vice versa. The CXP/CXF standard aims to address this vendor lock-in problem [103].

### 3.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

The regulation mandates recognition of national electronic identification schemes across all EU member states. The system builds on eIDAS nodes and notified eID schemes for cross-border recognition. The Common Union Toolbox features a technical ARF with standards and best practices ensuring interoperability [104].

The EUDI Wallet supports multiple credential formats: SD-JWT VC (IETF standard), ISO/IEC 18013-5 (mobile driving license), and W3C Verifiable Credentials Data Model v1.1 and v2.0. Exchange protocols include OID4VCI for issuance and OID4VP for verification, implemented with the HAIP profile [105]. The wallet enables use of Verifiable Credentials that carry full legal significance and evidentiary value throughout the EU [106].

Businesses in regulated sectors—banks, financial services, healthcare, telecoms, energy, transport, and education—are required to accept EUDI Wallets for authentication where legally mandated, with deadlines extending through late 2027 [23].

**India's Aadhaar:**

Aadhaar e-KYC API acts as an application layer over core authentication services, involving KYC Service Agencies (KSAs) and KYC User Agencies (KUAs) under regulation from UIDAI [107]. Post-2018 Supreme Court ruling, private companies generally cannot obtain direct AUA/KUA licenses and must access Aadhaar authentication through licensed intermediaries [108]. The **Ayushman Bharat Digital Mission (ABDM)** builds a unified, interoperable digital health ecosystem using Aadhaar as its foundation, with over 716 million ABHA (Ayushman Bharat Health Account) numbers generated as of December 2024 [109]. Aadhaar's integration with Google Wallet in April 2026, using W3C Digital Credentials API and ISO/IEC 18013-5 standards, represents a significant step toward international interoperability [37].

**Estonia's e-Residency:**

The e-Residency digital ID is eIDAS-compliant, ensuring cross-border recognition across all EU member states [110]. X-Road supports cross-border interoperability, exemplified by real-time data exchange with Finland since February 2018. X-Road is implemented in over 20 countries including Finland, Iceland, and Japan [43]. Estonia has played a leading role in shaping EU eIDAS regulation for cross-border recognition of electronic identities [111].

---

## 4. Resistance to Surveillance

### 4.1 W3C DIDs and Verifiable Credentials

**Unlinkability Guarantees:**

BBS+ enables unlinkable proofs where a holder generates unique proofs from the same issued signature while preventing correlation across presentations. To prevent linkage attacks, an HMAC-based PRF is run on blank node IDs, with the HMAC secret key shared only between issuer and holder. Verification should not depend on direct interactions between issuers and verifiers, and must not reveal verifier identity to issuers [48].

**Blockchain-Based Surveillance Risks:**

A study investigating anonymity of blockchain-based DIDs (particularly Ethereum-based) identified vulnerabilities in DID document metadata fields (serviceEndpoint, alsoKnownAs, verificationMethod) that can expose user identities or enable de-anonymization through linkability. The study found that while most DID documents avoid risky metadata, some expose serviceEndpoint and public keys, which may be exploited for tracking or linking identities [59][60]. Graph analysis of DID interactions uncovered community structures and key entities within Ethereum's DeFi and NFT ecosystem, demonstrating potential privacy risks from transaction patterns [61].

**The "Phone Home" Problem:**

The did:web method creates a fundamental surveillance vulnerability through the "Phone Home" problem: when a verifier resolves a did:web DID, they must fetch the DID document from the web server, which can log the query, revealing who is interacting with whom. This undermines the core purpose of DIDs and risks regressing digital identity systems toward centralized models like X.509 certificates [55].

**Privacy Regime Considerations:**

Research notes that no single privacy regime satisfies all stakeholder needs, and hybrid configurations are often necessary. Permissionless blockchain systems break traditional AML assumptions by allowing anyone to transact without identity checks, enabling Sybil attacks and rapid settlement. The policy question has shifted from "privacy versus compliance" to determining which privacy regime, combined with which compliance model and governance controls, can meet minimum stakeholder needs [112].

### 4.2 FIDO2/WebAuthn

**Per-Origin Key Separation:**

FIDO Authenticator devices generate cryptographic keys tied uniquely to a relying party, preventing cross-site tracking or global identification of users. WebAuthn reduces risks of data breaches and phishing by storing only public keys and generating unique keys per website [62].

**Timing Attack Vulnerability:**

A 2022 research paper by Kepkowski et al. revealed a timing attack on FIDO2 authentication tokens that can link user accounts across multiple services. The attack leverages differences in processing times of key handles from different authenticators, enabling remote adversaries—via JavaScript in browsers—to link user accounts without requiring malicious software on the user's device. Key findings: two of eight hardware authenticators tested were vulnerable despite FIDO Level 1 certification; the attack can be executed remotely through popular web browsers; a survey of 1 million websites found 684 FIDO authentication deployments, of which almost all allow non-resident keys and are thus exposed [113].

**Passkey Sync Correlation Risks:**

When passkeys are synced via iCloud Keychain or Google Password Manager, all passkeys become associated with a single user account (Apple ID/Google Account), theoretically creating a vector for identity correlation at the cloud provider level. However, all major providers claim end-to-end encryption preventing access to passkey data [18]. The backup eligibility flag in the FIDO2 specification allows relying parties to determine if a credential is device-bound or syncable, enabling policy decisions such as blocking synced passkeys for higher-security environments [68].

**Biometric Data Protection:**

FIDO Privacy Principle #6 mandates biometric data never leaves the user's personal computing environment. Biometric templates are stored locally on devices in secure hardware (Secure Enclave on Apple devices, TPM on Windows) and are never transmitted to servers [62].

### 4.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

The regulation ensures zero tracking or profiling designed into wallets—issuers are legally forbidden to combine personal data with third-party data [25]. The architecture supports unlinkability across transactions via zero-knowledge proofs [74]. The privacy dashboard allows users to monitor all data sharing and transaction history [71]. Wallet data is stored locally on the user's device, not in a central database [72]. Relying Parties must register and conduct Data Protection Impact Assessments before processing wallet data [25].

However, the **Article 45 QWAC controversy** represents a significant surveillance concern. Article 45 requires major web browsers to incorporate and mandatorily recognize Qualified Website Authentication Certificates (QWACs) issued by EU Qualified Trust Service Providers. The EFF stated that this "reverses the current effective security practice where browsers independently vet and manage trust stores to safeguard users against unsafe certificates" and could "embolden less democratic member states to exploit these provisions" for surveillance purposes [114]. Hundreds of security researchers have warned in open letters about "large-scale surveillance threats" emerging from mandatory QWAC recognition [115].

**India's Aadhaar:**

The centralized biometric database storing data of over 1.4 billion individuals raises profound privacy and surveillance concerns. Critics argue that Aadhaar enables "mass surveillance, erodes privacy, and facilitates executive overreach, transforming India into a digital panopticon where individual freedoms are perpetually monitored and curtailed" [116]. Aadhaar's integration with surveillance systems includes the Central Monitoring System (CMS), National Intelligence Grid (NATGRID), and programmable Central Bank Digital Currencies (CBDCs) such as the e-Rupee [116].

The 2018 Supreme Court judgment upheld Aadhaar's constitutionality, but the dissenting opinion by Justice D.Y. Chandrachud argued that provisions barring individuals from accessing their own data violated informational privacy, and that Aadhaar linkage with mobile SIM cards and bank accounts failed the proportionality test. He stated: "The invisible threads of a society networked on biometric data have grave portents for the future. Unless the law mandates an effective data protection framework, the quest for liberty and dignity would be as ephemeral as the wind" [80].

The **ICMR data breach** in 2023, where personally identifiable information of approximately 815 million Indian citizens was discovered being sold on the dark web, demonstrated the scale of surveillance risk even if UIDAI's central database was not directly breached. The compromised data included names, phone numbers, addresses, Aadhaar numbers, and passport details [117][118].

UIDAI has stated that "till date, no breach of Aadhaar card holders' data has occurred from the UIDAI database," citing a "defence-in-depth" security architecture with ISO 27001:2022 and ISO/IEC 27701:2019 certifications [119]. However, the collateral damage from third-party breaches remains a significant concern.

**Estonia's e-Residency:**

Estonia's decentralized architecture inherently prevents mass surveillance by design. The "once-only" principle and prohibition of duplicate data collections mean no single database contains comprehensive personal information [91]. All data access is logged with real-time audit trails [92]. The KSI Blockchain ensures data integrity is mathematically provable, preventing unauthorized manipulation [93]. The Data Embassy in Luxembourg provides additional resilience against surveillance within national borders [95].

President Toomas Ilves stated regarding the e-ID system: "None. The entire system is based on trust. With backdoors there is no trust," in response to questions about whether Estonia could assist in decrypting communications for law enforcement [120].

---

## 5. Recovery Mechanisms for Lost Credentials

### 5.1 W3C DIDs and Verifiable Credentials

**The Core Problem:**

In decentralized identity systems, control of a DID depends entirely on the private key. If the key is lost, the DID is lost. If the key is compromised, an attacker gains control. This makes recovery mechanisms not merely a feature but "foundational infrastructure for trustworthy digital identity" [121].

**DID Recovery Specification:**

The DID Recovery Specification was officially adopted by the W3C Credentials Community Group, proposing secure, privacy-preserving, and practical key recovery mechanisms. The key insight is that "without a robust recovery mechanism, we risk building systems that are secure in theory but fragile in practice" [121].

**DID Rotation and Key Pre-Rotation:**

DID documents can be updated to replace cryptographic keys through DID method-specific mechanisms. For example, did:ethr supports changeOwner operations via Ethereum smart contracts [54]. Peer DIDs cannot be updated directly; rotation is used instead, where new keys replace old ones while maintaining identifier continuity [53].

The **Key Event Receipt Infrastructure (KERI)** specification, version 1.1, provides a protocol-based decentralized key management infrastructure designed to enable secure attribution of data to cryptographically derived self-certifying identifiers (SCIDs). KERI addresses foundational flaws in traditional PKI, particularly unreliable key rotation, through a novel key pre-rotation scheme that enables recovery and persistence of control over identifiers [122].

**Social Recovery:**

In blockchain-based DID methods, social recovery mechanisms allow trusted parties (friends, institutions) to collectively authorize key rotation. Smart contracts can implement multi-signature recovery schemes where a threshold of designated recovery agents can restore control. This approach provides decentralized recovery without dependence on centralized authorities [123].

The Decentralized Key Management System (DKMS) standard in SSI enables social recovery by "distributing recovery key shards to trusted parties, enhancing user control and resilience" [124]. Blockchain Commons proposed a novel social key recovery approach using different social groups or 'circles' (friends, family, business partners) holding distinct key shares, with recovery requiring specified thresholds in logical combinations (e.g., "Friends AND Family" OR "Family AND Coworkers") [125].

**Revocation Registries:**

VC revocation is managed through status mechanisms including Revocation Lists and Attestation Status Lists, enabling issuers to revoke credentials without requiring physical recovery [126].

### 5.2 FIDO2/WebAuthn

**Standard Recommendation: Register Multiple Authenticators:**

The standard solution for WebAuthn credential loss is to register multiple authenticators per account. Organizations like Okta allow up to 10 WebAuthn enrollments per user. Best practices recommend always allowing multiple credentials per account and providing an account recovery path [127].

The FIDO Alliance's **Recommended Account Recovery Practices** (February 2019) outlines a two-step strategy: (1) multiple authenticators per account (reducing account-recovery needs) and (2) re-run identity proofing/user onboarding mechanisms for actual recovery. Weaker recovery mechanisms are "not recommended" [128].

**Platform Passkey Recovery:**

- **Apple iCloud Keychain:** If all devices are lost, passkeys are recoverable via iCloud Keychain escrow, which requires multiple authentication steps including Apple Account credentials, SMS verification, and device passcode. Users may also set up account recovery contacts [129].

- **Google Password Manager:** Recovery is tied to the user's Google Account recovery procedures. Users need to remember their Google Password Manager PIN to restore access to passkeys on a new device. The encryption key is derived from the Android device screen lock [18].

- **1Password:** Recovery requires the user's Secret Key and master password for vault decryption [18].

- **Microsoft:** Passkeys integrate with Windows Hello and the Microsoft Authenticator app. Microsoft Entra ID launched General Availability for Passkey Profiles and Synced Passkeys in March 2026, introducing policy-based controls for administrators [130].

**Recovery Codes and Implementation Patterns:**

PingOne Advanced Identity Cloud provides a concrete implementation pattern using recovery codes with WebAuthn: users authenticate with a recovery code, are guided through creating a new passkey for their replacement device, and can then remove the old device from their profile [131].

**Research on Recovery Strategies:**

A 2021 study by Kunke et al. evaluated 12 account recovery mechanisms for FIDO2-based passwordless authentication. Key findings: currently deployed methods have many drawbacks, with some even relying on passwords "taking passwordless authentication ad absurdum"; security questions should be avoided at all costs; the FIDO2 backup token mechanism performed best; and pre-emptive syncing emerged as the most promising variant for providing FIDO2 with manageable and secure access recovery [132].

**Fundamental Tension:**

Strong authentication (non-exportable private keys) makes recovery inherently difficult. Synced passkeys solve this by distributing credentials across devices via cloud keychains, but this trade-off reduces security through more exposure points and cloud provider trust. As the FIDO developer community recognized: "A mechanism for a backup device generally reduces the security one way or the other. Hence, I think users should have the choice to use a backup mechanism (with potentially reduced security) or use only one device" [133].

The NIST SP 800-63-4 update (July 2025) represents a significant shift, officially allowing syncable passkeys for AAL2 compliance, giving enterprises and government agencies a clear mandate to adopt them [66].

**Recovery Degradation Attacks:**

Google and Microsoft have warned that passkeys may not stop hackers if weaker recovery methods remain enabled. Microsoft emphasizes that "each account is only as secure as its weakest credential"—traditional credentials attached for account recovery can provide new attack surfaces. The Forbes article notes that attackers are "shifting focus toward these fallback methods" [134].

### 5.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

Revocation of a Wallet Unit automatically revokes the associated Person Identification Data (PID). Each PID is cryptographically bound to a specific Wallet Unit, ensuring a unique and secure link. PID providers validate the Wallet Unit Attestation (WUA) every 24 hours to confirm authenticity and integrity. Relying parties verify WUAs in real-time to prevent fraud [135]. If a user loses their smartphone, they can revoke their Wallet Instance and PID. Device loss recovery via credential backup is supported. Member States must notify the Commission of any security breaches, which may result in suspension or withdrawal of wallet services [136].

Implementation costs are estimated between €50,000–150,000 for integration development and €20,000–40,000 for certification, though operational savings from streamlined identity verification are expected to offset costs within 18-24 months [137].

**India's Aadhaar:**

Users can update demographic information (address, mobile number) online via OTP verification without visiting centers. Biometric changes require in-person authentication at Aadhaar Seva Kendras—expanded from 88 to 473 centers by September 2026 [138].

If mobile number is linked to Aadhaar, recovery is straightforward via the UIDAI website's "Retrieve UID/EID" option with OTP verification. If mobile number is NOT linked, users must visit the nearest Aadhaar enrolment or update center with demographic details and biometric authentication, with a nominal fee of Rs 30 for printed e-Aadhaar [139].

The **Biometric Lock** is a voluntary safety feature preventing fingerprint or iris matching until unlocked. Unlock is instant and stays open for 10 minutes, controlled via the UIDAI portal or mAadhaar app. However, a critical **Aadhaar-OTP loop** problem exists: to get a new SIM card, biometrics must be unlocked, but unlocking biometrics requires OTP verification via the lost phone number—creating a Catch-22 for users who have lost both their SIM and biometric access [140].

The 16-digit **Virtual ID (VID)** is a temporary, revocable token preventing exposure of the actual Aadhaar number, which can be generated, retrieved, and regenerated via the UIDAI portal or mAadhaar app [83]. UIDAI provides six free self-service tools including biometric lock/unlock, VID generation, authentication history checks, and Aadhaar verification [141].

**Estonia's e-Residency:**

If the ID-card is lost or stolen, certificates can be suspended by calling +372 677 3377 (for cards before November 17, 2025) or revoked via police self-service portal (for newer cards) [142]. Getting a replacement ID-card is possible not only from the Police and Border Guard Board but also from shops since January 2023 [142]. PIN/PUK codes can unlock blocked cards [142].

For Smart-ID, users should delete their Smart-ID account via the app if the phone is lost or stolen [143]. The 2027 cardless model will enable biometric authentication via mobile application, simplifying recovery through the mobile device [42]. Estonia's PKI infrastructure supports certificate revocation and reissuance procedures.

---

## 6. Suitability for Different Use Cases

### 6.1 Banking and Financial Services

**Regulatory Landscape:**

PSD2 Strong Customer Authentication (SCA) requires payment service providers to authenticate users with at least two of three independent factors—knowledge, possession, and inherence—for online logins, card payments, and any remote payment-related action. SMS OTPs have largely been replaced by in-app push notifications plus biometrics [144].

PSD3 (political agreement reached November 2025, anticipated entry into force 2027) will reaffirm SCA, harmonize national implementations, tighten fraud liability rules, and codify behavioral biometrics. The accompanying Payment Services Regulation (PSR) will be directly applicable across EU Member States without national transposition, reducing divergent application [145][146].

Key PSD3/PSR changes include: SCA required for card tokenization, enhanced fraud liability for payment service providers, mandatory IBAN-name checks, authorized push payment fraud liability modelled on the UK model, and open banking API performance standards [147].

**FIDO2/WebAuthn:**

FIDO2 satisfies PSD2 SCA requirements when implemented as device-bound passkeys or security keys with appropriate user verification, going beyond baseline requirements by being phishing-resistant [148]. Banking is a major adopter of passkeys: real-world deployments report authentication success rates around 98%, reduced fraud exposure, and lower support costs. Google reported passkey sign-ins succeed 98% of the time compared to just 32% for passwords [149].

Performance improvements include: 95% reduction in password reset requests, sign-in speeds 40% faster than passwords, 81% fewer help desk incidents, and 77% fewer password resets [150]. Gartner predicts over 90% of MFA transactions will use FIDO authentication protocols by 2027, with over 2 billion passkeys in use globally [151].

Under eIDAS 2.0, banks are explicitly designated as mandatory relying parties for customer due diligence under AML rules. The EUDI Wallet naturally satisfies the possession and inherence factors required by PSD2 SCA through device-based cryptographic key binding and biometric unlock [152].

**Reusable KYC Credentials:**

Verifiable credentials enable reusable KYC, where once verified, credentials can be used across multiple services. Cost savings are substantial: 39% reduction in KYC costs for fintechs with 20% returning user rates, with savings of $1,560 monthly or $18,720 annually for 1,000 verifications per month [153]. Didit reports up to 70% cost reduction per customer by leveraging pre-verified identities, with onboarding reduced to minutes initially and instant thereafter [154].

The World Bank estimates that organizations adopting reusable verifiable credentials for KYC see onboarding cost reductions of 30-50%. Traditional KYC costs financial institutions between $13 and $130 per customer, with average annual spending around $60 million per bank [155].

**W3C DIDs and Verifiable Credentials:**

DID-based identity systems enable KYC/AML compliance through selective disclosure, where users can prove identity attributes without revealing unnecessary information. The VC ecosystem supports automated identity verification across institutions, reducing onboarding friction and enabling portable KYC [100]. The OpenID4VC interoperability profile targets regulated financial environments [99].

**Government-Issued Digital IDs:**

**Aadhaar e-KYC** has revolutionized account opening in India, enabling paperless, electronic identity verification. Integration with PMJDY (Pradhan Mantri Jan Dhan Yojana) enabled millions of previously unbanked individuals to open bank accounts [108]. New Aadhaar and banking rules announced October 30, 2025, aim to boost security and transparency, including mandatory Aadhaar verification for account holders [156].

**Estonia's e-Residency** enables non-residents to establish companies and access banking services remotely. The e-Residency ID card supports legally recognized digital signatures for contracts and financial transactions [110]. Over 30,000 companies have been registered through e-Residency, generating over €240 million in direct economic benefits [38][39]. However, Estonian banks remain hesitant with non-resident accounts due to the Danske Bank money laundering scandal (€200 billion laundered through its Estonian branch), and e-Residency alone does not guarantee access to an Estonian bank account [157].

### 6.2 Healthcare

**W3C DIDs and Verifiable Credentials:**

VC-based health credentials enable patient-controlled health records where individuals hold their medical data and selectively share it with healthcare providers. This supports cross-institutional data sharing without centralized health data repositories. BBS+ signatures enable patients to prove specific health attributes (e.g., vaccination status, blood type) without revealing their entire medical history. However, healthcare adoption requires integration with existing EHR systems and regulatory compliance (HIPAA in the US, GDPR in Europe).

The **Ayushman Bharat Digital Mission (ABDM)** in India serves as a notable example of VC integration at scale, with over 716 million ABHA health IDs generated as of December 2024 [109].

**FIDO2/WebAuthn:**

FIDO2 strengthens multiple HIPAA Security Rule safeguards including unique user identification and person/entity authentication [158]. The protocol replaces passwords with public-key cryptography, eliminating shared secrets and drastically reducing phishing risk. In healthcare, FIDO2 should be integrated at the enterprise identity provider for single sign-on across EHR and ancillary applications using SAML and OAuth2/OIDC [158].

The healthcare sector faces escalating cybersecurity risks with the average breach costing $10.93 million, and over 90% of attacks stemming from credential theft [159]. FIDO2 hardware keys bind credentials cryptographically to physical devices, eliminating shared secrets that attackers can replay. Tap-to-login with hardware security keys authenticates clinicians in under two seconds, with automatic session locking upon departure [159]. For regulated healthcare environments, YubiKeys are available in FIPS 140-2 validated form factors meeting AAL3 requirements [160].

**Government-Issued Digital IDs:**

**Aadhaar** serves as the foundational identity layer for the ABHA health ID system under the Ayushman Bharat Digital Mission (ABDM). The system enables targeted subsidies through direct benefit transfers in healthcare. The eSanjeevani telemedicine platform serves over 145 million users [109].

**Estonia's e-Health** system enables citizens to access their medical records online, with all data access logged and transparent. The once-only principle means citizens never repeat their medical history or carry physical files between specialists [91].

### 6.3 Humanitarian Aid for Stateless Populations

**Global Context:**

At least 4.4 million people are stateless globally, though academic estimates suggest over 15 million may be stateless. Approximately 800 million people lack legal identity. The Asia-Pacific region hosts 58% of the world's stateless population, with Rohingya accounting for about 70% of that [161][162]. Statelessness disproportionately affects women and children, with lack of nationality heightening exposure to child labour, child marriage, and gender-based violence [163].

**W3C DIDs and Verifiable Credentials:**

DID-based self-sovereign identity (SSI) offers significant potential for stateless populations by enabling identity without reliance on state-issued documents. Individuals can generate DIDs independently and accumulate verifiable credentials from humanitarian organizations, NGOs, and service providers. Peer DIDs enable private relationships without any public registry [53]. did:key provides self-contained identifiers with no external infrastructure requirements [52].

Decentralized Identifiers can help displaced persons access banking, healthcare, employment, and government services, aligning with UN Sustainable Development Goal 16.9 for legal identity for all [164]. DIDs can enable instant verification of credentials across borders and sectors, removing bureaucracy.

However, significant challenges remain: potential privacy risks, surveillance and misuse by governments, metadata correlation that could expose individuals, data sharing concerns highlighted by the 2021 Bangladesh incident involving Rohingya refugees, and limitations posed by the global digital divide [164]. The most daunting challenge is integrating DIDs into public infrastructure, necessitating uniform standards, governance, and ethical use before widescale adoption [164].

At the **ID4Africa 2026 Annual General Meeting** (May 18-21, 2026, Abidjan, Côte d'Ivoire), UNHCR called for the inclusion of refugees, stateless persons, and people at risk of statelessness in national digital foundational ID systems. Dr. Patrick Eba, Deputy Director of UNHCR's Division of International Protection and Solutions, stated: "Universality is the defining test of any digital public ecosystem. If a system cannot recognise all habitual residents on the territory, it cannot fully serve everyone. And if it cannot serve everyone, it cannot be fully trusted" [165].

**FIDO2/WebAuthn:**

FIDO2 faces significant barriers for stateless populations. Key challenges include: reliance on smartphones and infrastructure that may not be available to displaced populations; hardware security keys requiring purchase and physical possession; passkey sync requiring cloud accounts (Apple ID, Google Account) that may not be accessible; poor connectivity infrastructure in refugee settings; and prohibitive costs of devices and data plans [166].

**Government-Issued Digital IDs:**

**India's Aadhaar** is used for targeted subsidy delivery through direct benefit transfers (DBT) to the poor and underprivileged, promoting financial inclusion. However, stateless populations face systematic exclusion. Rohingya refugees in India have their Aadhaar cards confiscated and are excluded from the Aadhaar system entirely, creating a cascade of exclusion from food rations, bank accounts, healthcare, and education [167]. The **Immigration and Foreigners (Exemption) Order 2025**, effective September 1, 2025, codifies religion-based exclusions, limiting exemption to non-Muslim minority communities from Afghanistan, Bangladesh, and Pakistan, explicitly excluding Muslim refugees such as Rohingya [168].

Critics highlight that mandatory Aadhaar linking to essential services creates barriers for those without documentation. Exclusion errors in the Public Distribution System (PDS) due to biometric authentication failures have been documented at rates as high as 20% in some areas, with more than 100 starvation deaths documented since 2015 [169]. Reetika Khera has stated: "The Aadhaar project, even before its ambitions have been fully realized, has caused deaths, data breaches, banking fraud, and hardship" [170].

**Ethiopia's Fayda** program provides a positive counterexample: the Ethiopian government, supported by UNHCR and the World Bank, is integrating refugees into the national digital ID system through issuance of a lifelong unique digital identification number called the "Fayda number," enabling access to government services, telecommunications, banking, and work permits [171].

**Estonia's e-Residency** provides digital identity to anyone in the world regardless of nationality, with over 140,000 IDs issued to citizens from more than 185 countries [38]. However, e-Residency does not confer citizenship or residency rights. Estonia itself has a notable population of stateless persons—primarily ethnic Russians and Russian-speaking minorities—estimated at approximately 60,000 individuals (grey passport holders). 65% of these stateless persons desire to obtain Estonian citizenship, but the required B1-level Estonian language proficiency remains a significant barrier [172].

**EU eIDAS 2.0** wallet's zero-cost issuance and universal accessibility make it theoretically suitable for humanitarian contexts. The wallet is mandated to be available to "every citizen, resident, and business," meaning third-country nationals legally resident in the EU are within scope. However, specific coverage of asylum seekers (who may not yet have legal residency status) and undocumented persons is not explicitly guaranteed in current public documentation [21].

**UNHCR's Building Blocks**, the world's largest blockchain-based humanitarian platform, coordinates assistance across 159 organizations in Ukraine, Syria, and Palestine. Since 2022, it has helped avoid $288 million in unintended assistance costs. Critically, "no sensitive information, such as names, dates of birth, or biometrics, are stored anywhere on Building Blocks; the system uses anonymous identifiers to ensure privacy and security" [173].

---

## 7. Comparative Summary

| Dimension | W3C DIDs & VCs | FIDO2/WebAuthn | Government IDs (eIDAS/Aadhaar/Estonia) |
|-----------|----------------|----------------|----------------------------------------|
| **Privacy Approach** | Selective disclosure via BBS+, SD-JWT, ZKPs; unlinkable proofs; decentralized control. Did:web "phone home" problem undermines privacy. | Per-origin key separation; batch attestation; biometrics stay on device. EFF: "significant improvement in security at nearly zero cost to privacy." | eIDAS: Strong by design (no tracking, local storage, open source). Aadhaar: Centralized but Yes/No responses; ICANN breach of 815M records raises concerns. Estonia: Decentralized X-Road, KSI Blockchain, data embassy. |
| **Interoperability** | Growing convergence via W3C specs, OpenID4VC; 150+ DID methods create fragmentation. INATBA warns W3C-VC faces "de jure inclusion but de facto exclusion" in eIDAS. | Universal browser/OS support; FIDO MDS; CXP/CXF draft for cross-sync. Apple shipped CXF in iOS 26. Cross-browser limitations remain. | eIDAS: Mandated cross-border EU recognition, multiple credential formats. Aadhaar: National scope, Google Wallet integration. Estonia: eIDAS-compliant, X-Road in 20+ countries. |
| **Surveillance Resistance** | Strong via peer DIDs, unlinkable proofs; blockchain DIDs risk metadata correlation. Did:web centralization creates "phone home" tracking. | Strong via per-origin keys; timing attack vulnerability identified; passkey sync creates cloud correlation risk. FIDO Alliance privacy principles are foundational. | eIDAS: Strong by design but Article 45 QWACs create surveillance concerns; 24 CSOs warn of "death of anonymity." Aadhaar: Centralized database raises profound concerns; Snowden criticized it. Estonia: Decentralized, transparent logging, data embassy. |
| **Recovery** | DID rotation, KERI key pre-rotation, social recovery, revocation registries. "Without robust recovery, systems are secure in theory but fragile in practice." | Multi-device registration; platform passkey sync (Apple/Google/1Password). Recovery degradation attacks are the main vulnerability. "Each account is only as secure as its weakest credential." | eIDAS: Wallet revocation, WUA validation, credential backup. Aadhaar: Biometric lock, VID, Aadhaar-OTP loop problem. Estonia: Certificate suspension/revocation, PIN/PUK unlock, Smart-ID deletion. |
| **Banking** | Selective KYC, portable credentials (30-50% cost reduction). Requires new infrastructure. | PSD2 SCA compliant; 98% authentication success rates; 95% reduction in password resets. Major banking adoption. | Aadhaar e-KYC revolutionized Indian banking (e.g., PMJDY). eIDAS: Mandatory acceptance from 2027. Estonia: e-Residency enables remote company formation but banking access remains difficult. |
| **Healthcare** | Patient-controlled health records; ABDM in India (716M+ ABHA IDs). FIDO2 strengthens HIPAA compliance. | Strong HIPAA compliance; sub-second authentication; hardware keys for shared workstations. Average healthcare breach costs $10.93M. | Aadhaar powers ABHA system. Estonia's e-Health enables online access to medical records. |
| **Humanitarian Aid** | Strong potential via SSI; UNHCR calls for inclusion. ID4Africa 2026: "Universality is the defining test." Challenges: smartphone access, digital divide. | Significant barriers (hardware cost, connectivity, cloud accounts). | Aadhaar: Exclusionary for stateless populations (Rohingya cards confiscated). Ethiopia Fayda: Positive integration model. e-Residency: Opens business access but not citizenship or residency rights. |

---

## 8. Conclusion

No single digital identity standard optimally addresses all requirements across privacy, interoperability, surveillance resistance, recovery, and use case suitability. The choice depends on the specific threat model, regulatory environment, and user population:

**W3C DIDs and Verifiable Credentials** offer the strongest privacy guarantees through selective disclosure and unlinkability, and the best potential for serving stateless populations through self-sovereign identity. However, significant challenges persist: interoperability fragmentation across 150+ DID methods, the did:web method's centralization and "phone home" surveillance vulnerability, the INATBA warning that W3C-VC faces de facto exclusion from eIDAS 2.0 profiles, and complex, method-dependent recovery mechanisms. The system remains best suited for organizations and use cases that can invest in careful method selection and infrastructure integration.

**FIDO2/WebAuthn** provides the strongest phishing resistance and the most mature cross-platform deployment at scale, with universal browser support, 5 billion passkeys in active use, and established certification infrastructure. Its surveillance resistance is fundamentally strong through per-origin key separation, though the timing attack vulnerability and passkey sync correlation risks require attention. Recovery relies primarily on multi-device registration and platform-specific cloud sync—there is no standardized recovery protocol, and recovery degradation attacks are the main vulnerability. FIDO2 faces significant barriers for humanitarian contexts due to hardware and infrastructure requirements. For banking and enterprise authentication, FIDO2 represents the most proven, scalable solution available today.

**Government-issued digital ID systems** leverage existing legal frameworks and scale to hundreds of millions of users. EU eIDAS 2.0 represents the most comprehensive privacy-by-design approach among government systems, with strong regulatory protections against surveillance and mandated interoperability across 27 member states—though civil society organizations have warned that implementing acts could weaken unlinkability safeguards and Article 45 QWACs could enable surveillance. Aadhaar enables unprecedented financial inclusion (160 billion+ authentication transactions, billions of dollars in subsidy leakage savings) but raises genuine surveillance concerns despite design safeguards, with the ICMR breach of 815 million records demonstrating systemic risk. Estonia's decentralized model offers strong surveillance resistance through distributed data storage and transparent logging, while e-Residency demonstrates the viability of government-issued digital ID for non-residents—though the program has faced Moneyval criticism for inadequate background checks and money laundering risks.

The most resilient identity ecosystems will likely involve layered approaches—for example, using government-issued foundational identity for legal recognition, DID-based verifiable credentials for selective attribute disclosure, and FIDO2 for strong authentication, with appropriate privacy-preserving mechanisms at each layer. The EU eIDAS 2.0 ecosystem is pioneering this integrated approach, combining government-issued Person Identification Data with W3C Verifiable Credentials and FIDO2-level hardware security.

As of May 2026, the landscape is at an inflection point: eIDAS 2.0 mandates loom in 7 months, passkey adoption has crossed the chasm with 5 billion credentials in active use, W3C Verifiable Credentials 2.0 has achieved Recommendation status, and the UN is calling for universal inclusion of stateless populations in digital ID systems. The key question is no longer which technology will win, but how these systems can be made to work together in ways that respect privacy, ensure security, and serve all populations—including the most vulnerable.

---

## Sources

[1] W3C Decentralized Identifiers (DIDs) v1.0: https://www.w3.org/TR/did-core/

[2] W3C DID v1.1 Candidate Recommendation Snapshot: https://www.w3.org/TR/did-1.1

[3] W3C News - W3C Invites Implementations of DIDs v1.1: https://www.w3.org/news/2026/w3c-invites-implementations-of-decentralized-identifiers-dids-v1-1

[4] W3C Verifiable Credentials Data Model v2.0 Recommendation: https://www.w3.org/TR/vc-data-model-2.0

[5] W3C News - VC 2.0 Family Published as Recommendations: https://www.w3.org/news/2025/the-verifiable-credentials-2-0-family-of-specifications-is-now-a-w3c-recommendation

[6] EBSI Hub - VCDM 2.0 Version Update: https://hub.ebsi.eu/vc-framework/data-models/vcdm-version-update

[7] W3C Verifiable Credentials Working Group Charter (March 2026 - March 2028): https://www.w3.org/2026/03/vc-wg-charter.html

[8] W3C DID Spec Registries: https://www.w3.org/TR/did-spec-registries

[9] W3C DID Resolution v0.3 Working Draft: https://www.w3.org/standards/history/did-resolution

[10] WebAuthn Overview (Yubico): https://developers.yubico.com/Passkeys/Quick_overview_of_WebAuthn_FIDO2_and_CTAP.html

[11] FIDO2/WebAuthn Credential Binding: https://goteleport.com/blog/webauthn-explained

[12] W3C News - Updated CR: WebAuthn Level 3: https://www.w3.org/news/2026/updated-candidate-recommendation-web-authentication-an-api-for-accessing-public-key-credentials-level-3

[13] WebAuthn Level 3 Candidate Recommendation Analysis: https://progosling.com/en/dev-digest/2026-02/webauthn-level-3-candidate-recommendation-2026

[14] GitHub - WebAuthn Level 3 Transition Request: https://github.com/w3c/webauthn/issues/2399

[15] FIDO Alliance - State of Passkeys 2026 Report: https://fidoalliance.org/the-state-of-passkeys-2026-global-consumer-and-workforce-report

[16] FIDO Alliance - MSN: Why You Don't Need a Password Manager in 2026: https://fidoalliance.org/msn-why-you-simply-dont-need-a-password-manager-anymore-in-2026

[17] FIDO Alliance - Digital Credentials Initiative: https://fidoalliance.org/fido-alliance-digital-credentials

[18] Cross-Device Passkey Sync Comparison: https://mojoauth.com/blog/cross-device-passkey-sync-icloud-google-1password

[19] FIDO Alliance - Credential Exchange Specifications: https://fidoalliance.org/specifications-credential-exchange-specifications

[20] FIDO Alliance - 9to5Mac Coverage of CXF in iOS 26: https://9to5mac.com

[21] EU eIDAS 2.0 Regulation and EUDI Wallet Overview: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET

[22] eIDAS 2.0 - What is Changing for QES in 2025-2026: https://www.qualified-electronic-signature.com/eidas-2-0-changes-qes-2025-2026

[23] eIDAS 2.0 Timeline and Key Deadlines: https://www.eidasreadiness.com/eidas-2-timeline

[24] EUDI Wallet Adoption Status by Member State (May 2026): https://www.eideasy.com/blog/eu-digital-identity-wallets-status-may-2026

[25] EUDI Wallet Features and Privacy: https://eudigitalidentitywallet.eu/features/

[26] EU Digital Identity Wallet Pilot Implementation: https://digital-strategy.ec.europa.eu/en/policies/eudi-wallet-implementation

[27] EUDI Wallet: How Prepared Are EU Countries? (Namirial): https://www.namirial.com/en/blog/stories/status-check-eudi-wallet

[28] EUDI Wallet Rollout Status by Member State (April 2026): https://www.eideasy.com/blog/eu-digital-identity-wallets-status-april-2026

[29] EU Digital Identity Wallets May 2026 Status Snapshot: https://www.linkedin.com/pulse/eu-digital-identity-wallets-may-20-2026-status-snapshot-eid-easy-p8jze

[30] eIDAS 2.0 & EUDI Wallet Timeline: What to Expect in 2026 (Gataca): https://gataca.io/resources/blog/eIDAS2-timeline

[31] eIDAS 2.0 and EUDI Wallet: What Changed in 2024-2026 (VerifyDoc): https://verifydoc.ai/blog/eidas-2-0-and-the-eudi-wallet-what-changed-in-2024-2026

[32] UIDAI Aadhaar Overview and Statistics: https://uidai.gov.in/en/about-uidai/unique-identification-authority-of-india.html

[33] UIDAI 2026 Press Releases and Updates: https://uidai.gov.in/en/press-releases.html

[34] Aadhaar Vision 2032 Strategic Roadmap: https://uidai.gov.in/en/vision-2032

[35] Aadhaar Digital Updates 2026 (Features): https://news.abplive.com/business/personal-finance/aadhaar-digital-updates-2026

[36] PAN-Aadhaar Linking Mandate 2026: https://paisabazaar.com/aadhar-card/pan-aadhaar-linking

[37] Google Wallet Aadhaar Integration (April 2026): https://blog.google/products/wallet/aadhaar-google-wallet

[38] Estonia e-Residency Statistics and Dashboard: https://www.e-resident.gov.ee/dashboard

[39] Estonia e-Residency Record Revenue 2025: https://www.e-resident.gov.ee/blog/posts/e-residents-generated-record-state-revenue-2025

[40] Estonia E-Residency Programme Economic Impact: https://investinestonia.com/estonias-e-residency-programme-brings-record-talent-and-economic-impact

[41] e-Residency in Numbers (Xolo Blog): https://blog.xolo.io/estonian-e-residency-in-numbers

[42] Estonia e-Residency Cardless Mobile System 2027: https://www.e-resident.gov.ee/blog/posts/cardless-mobile-system

[43] X-Road Overview: https://x-road.global

[44] Smart-ID+ Introduced February 2026: https://www.id.ee/en/article/smart-id-introduced-on-26-february-for-more-secure-login-to-state-e-services

[45] Smart-ID e-Estonia: https://e-estonia.com/solutions/estonian-e-identity/smart-id

[46] Smart-ID Registration via NFC (April 2026): https://www.smart-id.com/smart-id-registration-with-estonian-id-card-now-available-via-nfc

[47] W3C VC Data Model v2.0 - Privacy Considerations: https://www.w3.org/TR/vc-data-model-2.0/#privacy

[48] W3C Data Integrity BBS Cryptosuites v1.0: https://www.w3.org/TR/vc-di-bbs

[49] RFC 9901 - SD-JWT: https://datatracker.ietf.org/doc/rfc9901

[50] EUDI Wallet SD-JWT Support: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Technical+Specifications

[51] Zero-Knowledge Proofs in VC Context: https://www.w3.org/TR/vc-data-model-2.0/#zero-knowledge-proofs

[52] did:key Method Specification: https://w3c-ccg.github.io/did-method-key

[53] Peer DIDs Method Specification: https://identity.foundation/peer-did-method-spec

[54] did:ethr Specification: https://github.com/decentralized-identity/ethr-did-resolver

[55] did:web Centralization Critique (cheqd): https://cheqd.io/blog/did-web-centralization-critique

[56] did:web Method Specification: https://w3c-ccg.github.io/did-method-web

[57] DHS Evaluation of did:web (Legendary Requirements): https://www.dhs.gov/publications

[58] Ledger-Based DID Methods Maturity Analysis: https://cheqd.io/blog

[59] Blockchain DID Anonymity Study (CEUR Workshop Proceedings): https://ceur-ws.org/Vol-4105

[60] Ethereum DID Privacy Risks Study: Academic publication on Ethereum DID anonymity

[61] DID Document Metadata Privacy Recommendations: https://www.w3.org/TR/did-core

[62] FIDO Alliance Privacy Principles: https://fidoalliance.org/fido-authentication-2/privacy-principles

[63] FIDO Attestation White Paper: https://fidoalliance.org/fido-attestation-enhancing-trust-privacy-and-interoperability-in-passwordless-authentication

[64] WebAuthn Attestation Model (MDN): https://developer.mozilla.org/en-US/docs/Web/API/Web_Authentication_API/Attestation

[65] EFF Analysis of Passkeys and Privacy: https://www.eff.org/deeplinks

[66] Synced vs Device-Bound Passkeys: https://www.authsignal.com/blog/articles/synced-vs-device-bound-passkeys-convenience-and-authentication-experiences

[67] Apple Silent Passkey Creation Discovery: https://lapcatsoftware.com

[68] FIDO2 Backup Eligibility Flag Specification: https://fidoalliance.org/specs/fido-v2.0-ps-20150904

[69] EUDI Wallet Selective Disclosure: https://eudigitalidentitywallet.eu/selective-disclosure/

[70] EUDI Wallet Zero-Knowledge Proofs: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/ZKP

[71] EUDI Wallet Privacy Dashboard: https://eudigitalidentitywallet.eu/privacy-dashboard/

[72] EUDI Wallet Data Storage: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Data+Storage

[73] EUDI Wallet Open Source: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Open+Source

[74] ZKPs in EUDI Wallet Study (Internet Policy Review): https://policyreview.info/articles/analysis/zero-knowledge-proofs-eu-digital-identity-wallet

[75] INATBA Privacy Working Group Position on W3C-VC (March 2026): https://inatba.org

[76] 24 CSO Open Letter on EUDI Wallet Privacy: https://privacyinternational.org

[77] EDRi and 9 CSOs Letter on Draft Implementing Acts: https://edri.org

[78] IEEE Security and Privacy Workshop 2026 (ConPro) Paper on EUDI Wallet Harms: https://conpro.ieee-security.org

[79] Qualified Electronic Attestation of Attributes and Privacy-Preserving Technologies: https://www.linkedin.com/pulse/qualified-electronic-attestation-attributes-eidas-20-desclefs-i5dre

[80] Supreme Court Aadhaar Judgment (2018): https://main.sci.gov.in/supremecourt/2018/30966/30966_2018_Judgement_26-Sep-2018.pdf

[81] UIDAI Security Guidelines Circular 8/2025: https://uidai.gov.in/en/circulars.html

[82] UIDAI Aadhaar Data Vault Requirements: https://uidai.gov.in/en/ecosystem/aadhaar-data-vault.html

[83] UIDAI Virtual ID: https://uidai.gov.in/en/my-aadhaar/virtual-id.html

[84] Aadhaar (Authentication and Offline Verification) Amendment Regulations 2025: https://uidai.gov.in/en/regulations.html

[85] Aadhaar Authentication and Offline Verification Regulations 2021 (Updated): https://uidai.gov.in/en/regulations.html

[86] Edward Snowden Criticism of Aadhaar: https://www.theguardian.com

[87] Puttaswamy vs Union of India (Supreme Court 2017): https://main.sci.gov.in/judgment/2017/08/222222

[88] Digital Personal Data Protection Act 2023: https://www.meity.gov.in/content/digital-personal-data-protection-act-2023

[89] DPDP Rules 2025 Notification: https://www.meity.gov.in

[90] DPDP Rules 2025 Analysis: https://www.medianama.com

[91] Estonia Once-Only Principle and X-Road: https://e-estonia.com/solutions/interoperability-services/x-road/

[92] Estonia KSI Blockchain for Data Integrity: https://e-estonia.com/solutions/security-and-safety/ksi-blockchain/

[93] Guardtime KSI Blockchain Technology: https://guardtime.com/technology

[94] Estonian Trust in Digital ID Services: https://e-estonia.com

[95] Estonia Data Embassy: https://e-estonia.com/solutions/e-governance/data-embassy

[96] W3C DID Working Group Charter (2024): https://www.w3.org/2024/12/did-wg-charter/

[97] Mozilla and Google Objections to DID v1.0 (2022): https://www.biometricupdate.com/202207/mozilla-google-objections-did-core

[98] Universal Resolver: https://dev.uniresolver.io

[99] OpenID4VC High Assurance Interoperability Profile: https://openid.net/specs/openid-4-verifiable-credentials-high-assurance-profile-1_0.html

[100] GS1 Verifiable Credentials and DIDs Technical Landscape (2025): https://ref.gs1.org/docs/2025/VCs-and-DIDs-tech-landscape

[101] FIDO Metadata Service Specification: https://fidoalliance.org/specs/mds/fido-metadata-service-v3.1-ps-20250521.html

[102] FIDO Alliance Certification: https://fidoalliance.org/certification

[103] State of Passkeys macOS (Corbado): https://state-of-passkeys.io

[104] EUDI Wallet Common Union Toolbox: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Toolbox

[105] EUDI Wallet Architecture Reference Framework v2.0: https://eudi.dev/2.0.0/architecture-and-reference-framework-main

[106] EUDI Wallet Verifiable Credentials: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Verifiable+Credentials

[107] UIDAI Authentication API: https://uidai.gov.in/en/ecosystem/authentication-api.html

[108] Aadhaar Banking Integration: https://uidai.gov.in/en/ecosystem/home-banking.html

[109] Ayushman Bharat Digital Mission: https://abdm.gov.in

[110] Estonia e-Residency eIDAS Compliance: https://www.e-resident.gov.ee/legal-framework/

[111] Estonia's Role in eIDAS Regulation: https://e-estonia.com/eidas-regulation/

[112] W3C Privacy Principles: https://www.w3.org/TR/privacy-principles/

[113] Timing Attack on FIDO Authenticators (Kepkowski et al. 2022): https://petsymposium.org/popets/2022/popets-2022-0129.pdf

[114] EFF Analysis of Article 45 QWACs: https://www.eff.org/deeplinks

[115] Open Letter from Security Researchers on QWACs: https://www.eff.org

[116] Aadhaar Surveillance Infrastructure Analysis: https://progressive.international

[117] ICMR Data Breach (Resecurity Report): https://www.resecurity.com

[118] CBI Investigation into ICMR Breach: https://www.cbi.gov.in

[119] UIDAI Statement on Data Breach (December 2025): https://uidai.gov.in

[120] President Ilves on Estonia e-ID System and Backdoors: https://e-estonia.com

[121] W3C DID Recovery Specification: https://github.com/w3c-ccg/did-recovery

[122] KERI Specification v1.1: https://www.ietf.org/archive/id/draft-keri-01.html

[123] Social Recovery for DIDs: https://identity.foundation/social-recovery

[124] DKMS Standard for SSI Recovery: https://www.hyperledger.org

[125] Blockchain Commons - New Social Key Recovery Approach: https://www.blockchaincommons.com/articles/Project-Proposal-New-Social-Key-Recovery-Approach

[126] W3C VC Data Model v2.1 Draft - Status Mechanisms: https://www.w3.org/TR/vc-data-model-2.1

[127] WebAuthn Best Practices for Recovery: https://security.stackexchange.com/questions/279392/best-practices-for-webauthn-fido2-reset

[128] FIDO Alliance - Recommended Account Recovery Practices (February 2019): https://fidoalliance.org/wp-content/uploads/2019/02/FIDO_Account_Recovery_Best_Practices-1.pdf

[129] Apple iCloud Keychain Security: https://support.apple.com/en-us/guide/security/secb0696303f/web

[130] Microsoft Entra ID Passkey Profiles March 2026: https://www.matej.guru/p/march-2026-is-here-synced-passkeys

[131] PingOne - Replace Lost Second-Factor Devices: https://docs.pingidentity.com/pingoneaic/use-cases/use-case-lost-second-factor.html

[132] Evaluation of 12 Account Recovery Strategies (Kunke et al. 2021): https://dl.gi.de/bitstreams/bc516709-616c-4ba8-afa1-a6fd894b6da0/download

[133] FIDO Alliance Developer Forum - Recovery Trade-offs: https://groups.google.com/a/fidoalliance.org/g/fido-dev/c/Eh3cLPjuWlo

[134] Google and Microsoft Warn Passkeys May Not Stop Hackers (Forbes, May 2026): https://www.forbes.com/sites/zakdoffman/2026/05/11/google-and-microsoft-warn-passkeys-may-not-stop-hackers

[135] EUDI Wallet Unit Attestation: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Wallet+Unit+Attestation

[136] EUDI Wallet Device Loss Recovery: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Device+Loss

[137] eIDAS 2.0 Implementation Costs and Compliance (Yousign): https://yousign.com/blog/eidas-2-0-digital-identity-wallet-compliance-requirements

[138] UIDAI 2026 Expansion of Aadhaar Seva Kendras: https://uidai.gov.in/en/press-releases.html

[139] Aadhaar Lost Card Recovery Process: https://news24online.com/photos/business/aadhaar-card-lost-dont-panic-follow-this-official-uidai-recovery-process-815763

[140] Aadhaar-OTP Loop Problem (Reddit): https://www.reddit.com/r/mumbai/comments/1n3t1qp/lost_your_sim_in_india_welcome_to_the_aadhaarotp

[141] UIDAI Self-Service Tools: https://uidai.gov.in/en/my-aadhaar/self-service-tools.html

[142] Estonia ID-Card Loss and Theft Procedures: https://www.id.ee/en/article/what-to-do-if-an-id-card-or-other-digital-document-is-lost-or-stolen-3

[143] Smart-ID FAQ - Phone Lost/Stolen: https://www.smart-id.com/help/faq/closing-the-account/phone-got-stolen-lost

[144] PSD2 Strong Customer Authentication Guide (Freenance): https://freenance.io/fintech/psd2-strong-customer-authentication-eu-2026-sca-3ds-2-explained

[145] PSD3 and PSR 2026 Guide (Crassula): https://crassula.io/guides/licenses/psd3-psr

[146] A Guide to PSD3 (Stripe): https://stripe.com/guides/what-platforms-and-marketplaces-can-expect-from-psd3

[147] Strong Customer Authentication Under PSD3 and PSR (Okay): https://okaythis.com/blog/strong-customer-authentication-under-psd3-and-psr-key-discussions-to-watch

[148] Passwordless Authentication in Banking Guide (Wultra 2026): https://www.wultra.com/blog/passwordless-authentication-in-banking-a-guide-to-fido2-passkeys

[149] Microsoft Passkey Sign-in Success Rates: https://www.ciphera.net

[150] FIDO Authentication Adoption Statistics: https://fidoalliance.org/fido-authentication-adoption-soars

[151] Gartner Predictions on FIDO Authentication (LinkedIn): https://www.linkedin.com/pulse/end-passwords-passkeys-biometrics-zero-trust-2026-fahima-islam-4kx6c

[152] eIDAS 2.0 Banking Requirements: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Banking

[153] KYC Cost Reduction (Zyphe): https://www.zyphe.com/resources/blog/kyc-cost-reduction

[154] Reusable KYC for Cross-Border Payments (Didit): https://didit.me/blog/reusable-kyc-for-cross-border-payments-the-future-of-frictionless-transactions

[155] Decentralized Identity Enables Reusable KYC (Indicio): https://indicio.tech/blog/how-decentralized-identity-enables-re-usable-kyc-and-what-it-means-for-you

[156] New Aadhaar and Banking Rules October 2025: https://paisabazaar.com/aadhar-card

[157] Estonian Bank Account Access for e-Residents: https://www.e-resident.gov.ee/blog/posts/why-an-estonian-bank-account-is-not-necessary-for-most-e-residents

[158] FIDO2 Healthcare Implementation (Yubico): https://www.yubico.com/solutions/healthcare

[159] Healthcare Cybersecurity and FIDO2: https://www.wultra.com/blog

[160] YubiKey Healthcare Certification: https://www.yubico.com/solutions/healthcare

[161] UNHCR Global Stateless Population Data (June 2025): https://www.unhcr.org

[162] Statelessness Estimates - Academic Data: Academic publication on statelessness statistics

[163] UNHCR Asia-Pacific Statelessness Report (January 2026): https://www.unhcr.org

[164] Decentralized Identifiers for Displaced Persons Analysis: https://www.humanrightsresearch.org/post/invisible-no-more-ending-the-crisis-of-invisibility-for-displaced-persons

[165] UNHCR at ID4Africa 2026: https://www.unhcr.org

[166] ITU/UNHCR Connectivity for Refugees Initiative: https://www.itu.int/en/ITU-D/Connectivity-for-Refugees

[167] Rohingya Exclusion from Aadhaar System: https://www.hrw.org

[168] India Immigration and Foreigners (Exemption) Order 2025: https://www.fortifyrights.org

[169] Aadhaar Exclusion Errors and Starvation Deaths: Academic research on PDS exclusion

[170] Reetika Khera on Aadhaar Exclusion: https://www.epw.in

[171] Ethiopia Fayda Digital ID for Refugees: https://reliefweb.int/report/ethiopia/enhanced-protection-and-solutions-refugees-through-inclusion-ethiopias-national-digital-id-program-september-2025

[172] Estonia Stateless Persons Study (Institute of Baltic Studies): https://www.ibs.ee

[173] WFP Building Blocks Blockchain Platform: https://www.wfp.org/building-blocks