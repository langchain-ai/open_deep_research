# Comprehensive Technical Analysis: Digital Identity Standards and Systems

## Executive Summary

This report provides a detailed comparative analysis of three major digital identity paradigms: (1) W3C Decentralized Identifiers (DIDs) and Verifiable Credentials, (2) FIDO2/WebAuthn, and (3) government-issued digital ID systems (EU eIDAS, India's Aadhaar, Estonia's e-Residency). Each system embodies fundamentally different architectural philosophies—decentralized self-sovereign identity, hardware-backed authentication, and state-issued foundational identity—with distinct trade-offs across privacy, interoperability, surveillance resistance, recovery mechanisms, and use case suitability.

---

## 1. Overview of Each System

### 1.1 W3C Decentralized Identifiers (DIDs) and Verifiable Credentials

The W3C Decentralized Identifiers (DIDs) specification reached W3C Recommendation status on July 19, 2022, establishing a standardized method for creating verifiable, decentralized digital identifiers that operate independently of centralized registries, identity providers, and certificate authorities [1]. **DID v1.1**, published as a Candidate Recommendation Snapshot on March 5, 2026, refines the v1.0 specification by consolidating media types, introducing new JSON-LD contexts, and restructuring related standards [2]. The **Verifiable Credentials Data Model v2.0** became a W3C Recommendation on May 15, 2025, defining a standardized, extensible data model for expressing, securing, and exchanging verifiable credentials [3]. Work is ongoing on **VC Data Model v2.1**, which will include cryptographic suites for data integrity using ECDSA and EdDSA, zero-knowledge proof capabilities via BBS+ signatures, and status mechanisms for credential revocation [4].

The DID ecosystem supports multiple DID methods, each with distinct trade-offs: **did:key** (self-contained, no external registry), **did:peer** (pairwise relationships, off-chain), **did:web** (domain-based), **did:ethr** (Ethereum blockchain), **did:indy** (Hyperledger Indy permissioned ledger), **did:ion** (Sidetree protocol on Bitcoin), and **did:dht** (distributed hash table) [5]. The **DID Resolution v0.3** specification (Working Draft, May 25, 2026) defines a standard interface for resolving DIDs to DID documents, independent of any specific DID method [6].

### 1.2 FIDO2/WebAuthn

FIDO2 is a suite of standards comprising the W3C Web Authentication (WebAuthn) API and the FIDO Alliance's Client to Authenticator Protocol (CTAP). WebAuthn provides a standard web API built into all major browsers (Chrome, Safari, Firefox, Edge) that enables passwordless and multi-factor authentication using public-key cryptography [7]. CTAP defines communication protocols between client devices and authenticators (hardware security keys, platform biometrics). The system's core security property is that credentials are cryptographically bound to specific domain origins, making phishing attacks technically impossible [8].

FIDO2 supports **discoverable credentials** (also called resident keys or passkeys) stored on the authenticator, and **non-discoverable credentials** where the private key is derived on-demand from a master secret and a server-stored credential ID [9]. **Passkeys**, introduced by Apple in 2022 and subsequently adopted by Google and Microsoft, represent the consumer-facing implementation of discoverable FIDO2 credentials, synced across devices via cloud keychains (iCloud Keychain, Google Password Manager, third-party providers like 1Password) [10]. The NIST SP 800-63-4 update (July 2025) officially recognized syncable passkeys for Authentication Assurance Level 2 (AAL2) compliance [11].

### 1.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0 (Regulation (EU) 2024/1183):** Published April 30, 2024, this regulation mandates that by December 2026, all 27 EU Member States must offer citizens and residents an EU Digital Identity Wallet (EUDI Wallet) free of charge [12]. The wallet enables selective disclosure of attributes, qualified electronic signatures, and pseudonymous authentication. The regulation explicitly prohibits tracking and profiling, and requires wallets to include privacy dashboards for user transparency [13]. The Architecture and Reference Framework (ARF) defines a secure, user-controlled ecosystem leveraging standards including SD-JWT, ISO 18013-7, and W3C Verifiable Credentials [14].

**India's Aadhaar:** Operated by the Unique Identification Authority of India (UIDAI) since 2009, Aadhaar is a 12-digit unique identity number issued to over 1.4 billion Indian residents. The system uses a centralized biometric database for identity verification, with authentication returning only Yes/No responses (not raw biometric data) [15]. In 2026, UIDAI expanded full-service centers from 88 to 473, introduced AI and face ID features in the eAadhaar app, and launched initiatives including "Battery Aadhaar" for electric vehicle battery tracking [16]. The Supreme Court's 2018 judgment upheld Aadhaar's constitutional validity while striking down mandatory linkage for certain services, concluding it does not create a surveillance state [17].

**Estonia's e-Residency and Electronic Identity:** Launched in December 2014, e-Residency provides government-backed digital identity to non-residents, enabling remote company establishment and access to Estonian e-services. Over 125,000 e-Residency IDs have been issued to citizens from more than 180 countries, with over 30,000 companies registered [18]. The system is built on Estonia's X-Road infrastructure—a decentralized data exchange platform processing over 2.2 billion transactions annually—and the KSI Blockchain for data integrity verification [19]. From January 2027, e-Residency will introduce a cardless mobile biometric application compliant with EU eIDAS requirements [20].

---

## 2. Approach to Privacy

### 2.1 W3C DIDs and Verifiable Credentials

The VC Data Model v2.0 explicitly addresses privacy considerations including selective and unlinkable disclosure, and supports privacy-enhancing technologies such as zero-knowledge proofs [3]. The DID v1.1 specification applies Privacy by Design principles throughout and cautions against including encrypted personal data in DID documents [2].

**Selective Disclosure Mechanisms:**

- **BBS+ Signatures:** The "Data Integrity BBS Cryptosuites v1.0" (W3C Candidate Recommendation Draft, April 7, 2026) provides selective disclosure and unlinkable derived proofs using pairing-based cryptography [21]. BBS enables holders to generate unique proofs from the same issued signature while preventing correlation. The suite supports anonymous holder binding (binding documents to a secret known only to the holder) and credential-bound pseudonyms (privacy-preserving pseudonyms linking credentials to holders without revealing identity) [21]. Security proofs have demonstrated BBS signatures achieve EUF-CMA and SUF-CMA security under certain cryptographic assumptions [21].

- **SD-JWT (Selective Disclosure JSON Web Tokens):** RFC 9901, published by the IETF, introduces SD-JWTs where sensitive claims are replaced with salted hashes and real claim values are moved into encrypted "Disclosures," allowing selective revelation [22]. SD-JWT+KB adds holder authentication to prevent theft and replay attacks. SD-JWT offers advantages in ease of implementation and interoperability due to its reliance on widely-used JWTs and cryptographic algorithms [23]. It is actively supported in the EUDI Wallet program and preferred in eIDAS 2.0 regulations [24].

- **Zero-Knowledge Proofs (ZKPs):** zk-SNARKs and zk-STARKs enable proving statements without revealing the inputs used, though they remain computationally expensive [25]. ZKP technology has moved from experimental to practical infrastructure, with institutional attention from organizations like the Ethereum Foundation's privacy unit [26]. Growing regulatory acceptance occurs when privacy solutions enable compliance with AML/KYC requirements [26].

- **SD-BLS Scheme (Academic Research):** Research proposes SD-BLS where individual claims are signed by the issuer, enabling selective disclosure and proof of possession. It features anonymous cryptographic revocation where revocation data contains no information about credential holders, and multi-stakeholder governance of revocations where revocation keys are split among multiple revocation issuers via PVSS [27]. Benchmarks show SD-BLS verification is one order of magnitude slower than BBS+ but supports large revocation lists with linear complexity [27].

**DID Method Privacy Implications:**

Different DID methods offer varying privacy properties. **did:key** derives identifiers from cryptographic keys with no external storage, offering strong privacy properties with no ledger trail [5]. **did:peer** supports pairwise relationships where only involved parties resolve the DIDs, enhancing privacy by lacking third-party data controllers [28]. **did:ethr** reveals the public and immutable nature of blockchain data, with risks from key compromise and correlation via blockchain activity—the specification recommends using separate DIDs for different contexts [29]. **did:web** leverages DNS and HTTP hosting, resulting in centralized operation where domain operators can observe and correlate all requests [30].

### 2.2 FIDO2/WebAuthn

**Core Privacy Design:**

The FIDO Alliance's six core Privacy Principles include requiring explicit, informed user consent for any operation using personal data, and a requirement that biometric data must never leave the user's personal computing environment [31]. FIDO Authenticator devices must not have a global identifier visible across different websites to prevent unwanted re-identification [31].

**Per-Origin Key Separation:**

FIDO2 credentials are cryptographically bound to specific domain origins. During registration and authentication, the authenticator generates cryptographic keys tied uniquely to a relying party (the online service), preventing cross-site tracking or global identification of users [31]. This is a fundamental architectural feature—key pairs are never shared across different domains.

**Attestation Models:**

FIDO supports multiple attestation models with privacy implications. **Standard (non-enterprise) attestation** uses batch attestation where attestation private keys are shared among devices in a batch (recommended >100,000 devices) to prevent fingerprinting [32]. The AAGUID (Authenticator Attestation GUID) must be identical across all substantially identical authenticators, ensuring attestation identifies the make and model, not an individual device [33]. **Enterprise attestation** intentionally removes this privacy protection, binding a unique authenticator key pair to a serial number or equivalent unique identifier for authorized relying parties with user awareness [32].

By default, the WebAuthn API returns "None" attestation (where the browser removes attestation information for privacy reasons) [34]. The FIDO Metadata Service (MDS) provides a global registry of authenticator capabilities, downloadable as a single digitally signed file for relying parties [35].

**Passkey Sync Privacy Implications:**

Passkeys synced via iCloud Keychain are end-to-end encrypted with strong cryptographic keys not known to Apple, and Apple Account access requires two-factor authentication [36]. Google Password Manager encrypts passkeys with a key derived from the user's Android device screen lock, with end-to-end encryption preventing Google access [37]. 1Password provides zero-knowledge vault security with Secret Key and master password [10]. While all major providers claim strong encryption, the existence of credentials across multiple devices and cloud infrastructure expands the attack surface compared to device-bound credentials [11].

### 2.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

Data minimization is central to the EUDI Wallet design—any digital service should collect only the minimum data required [38]. The wallet supports selective disclosure of attributes (e.g., age verification without revealing full identity) [39]. Zero-knowledge proofs allow relying parties to validate statements without revealing underlying data [40].

A 2025 study in Internet Policy Review examined how electronic attestations include auxiliary cryptographic data (issuer digital signatures, holder binding keys) that act as unique persistent identifiers, creating privacy risks by enabling linkability across transactions. ZKPs can resolve conflicts between the EUDI Wallet's data requirements and GDPR's data minimization principle [41].

The regulation explicitly prohibits tracking and profiling—issuers are legally forbidden to combine personal data with third-party data [13]. Every wallet includes a built-in privacy dashboard offering an overview of who data has been shared with, exactly what data was shared, and transaction history [42]. The wallet supports pseudonymity, generating and storing pseudonyms encrypted and locally [43]. All wallet data is stored locally on the user's device, not in a central database [44].

**India's Aadhaar:**

Aadhaar's authentication API returns only Yes/No responses for biometric verification, not raw biometric data [15]. e-KYC returns verified demographic data in an encrypted XML file signed by UIDAI. The 2018 Supreme Court judgment concluded Aadhaar does not create a surveillance state as it collects minimal data and incorporates safeguards including encryption and data storage limits [17].

UIDAI's 2025 Security Guidelines (Circular No. 8 of 2025) mandate AES-256 encryption, RSA-2048, FIPS 140-2 Level 3 certified HSMs, and MeitY-certified hosting [45]. Regulated entities must store Aadhaar numbers in a secure, UIDAI-compliant Aadhaar Data Vault [46]. The Virtual ID (VID) is a 16-digit temporary, revocable token that can be used in place of the Aadhaar number for authentication, preventing exposure of the actual Aadhaar number [47].

However, critics raise concerns that the centralized biometric database of over 1.4 billion individuals enables mass surveillance capabilities [48]. The Supreme Court's 2017 Puttaswamy judgment established privacy as a fundamental right under Article 21, requiring strict three-fold test (legality, necessity, proportionality) for state interference [49].

**Estonia's e-Residency:**

Estonia employs a distributed data model where different government agencies maintain their own databases, connected through X-Road. There is no central database—data remains in source systems and is shared only with user consent through tamper-proof mechanisms [50]. This decentralized architecture inherently prevents creating a single point of vulnerability or mass surveillance target. The "once-only" principle means citizens provide data once and government reuses it securely, with full transparency over data access [51]. All data access is logged with real-time audit trails [52].

The KSI Blockchain, developed by Guardtime, guarantees that electronic data authenticity is mathematically provable, making it impervious to manipulation by hackers or even the government itself [53]. Keyless signatures remain verifiable without assuming continued secrecy of the keys, providing a solution to long-term validity [54].

---

## 3. Interoperability

### 3.1 W3C DIDs and Verifiable Credentials

**Standardization Status:**

The W3C VC Data Model and DID Core specifications have reached official recommendation status, providing a stable foundation for implementations [55]. The Decentralized Identity Foundation (DIF) develops complementary specs and interoperability standards alongside the W3C [56]. A newly chartered W3C DID Working Group will deliver standard DID methods and advance interoperability [57].

**Interoperability Challenges:**

Despite significant convergence, many "flavours" of VC-based solutions exist that are not necessarily interoperable despite using the W3C VC data model [55]. Areas of divergence are driven by specific business requirements, regulatory constraints, or different technical philosophies [55]. However, interoperability has improved significantly through initiatives like W3C Test Suites and Plugfests [55].

**OpenID4VC Profiles:**

The OpenID for Verifiable Credentials specification defines profiles for credential issuance and verification. The **OpenID4VC High Assurance Interoperability Profile** targets regulated environments like eIDAS 2.0, requiring support for both pre-auth code flow and authorization code flow, pushed authorization requests, and client authentication using attestation-based JWTs [58]. Cryptographic Holder Binding using KB-JWT is mandatory when presenting an SD-JWT VC [58].

**Cross-Framework Integration:**

The EU Digital Identity Wallet implements DID and verifiable credential standards, ensuring interoperability [14]. Live deployments include British Columbia's OrgBook, European Blockchain Services Infrastructure (EBSI) digital diplomas, and U.S. mobile driver's licenses [55]. GS1 identifies promising use cases in supply chains for traceability, compliance, and fraud prevention [55]. The Universal Resolver tool, developed by DIF, supports interoperability by resolving many different DID methods [59].

### 3.2 FIDO2/WebAuthn

**Cross-Platform and Cross-Browser Support:**

All major web browsers (Chrome, Safari, Firefox, Edge) support the WebAuthn API, providing a uniform authentication interface [7]. FIDO2 is widely supported across major operating systems including Windows, macOS, iOS, and Android [60].

**FIDO Alliance Certification Infrastructure:**

The FIDO Metadata Service (MDS) acts as a global registry of passkey capabilities, providing relying parties with the intelligence to make data-driven decisions. The Metadata BLOB is a single, digitally signed file downloaded periodically (averaging 1-2 updates per week in 2024) [35]. The AuthenticatorStatus enum includes statuses like FIDO_CERTIFIED, USER_VERIFICATION_BYPASS, ATTESTATION_KEY_COMPROMISE, and REVOKED to indicate authenticator security state [61]. Currently, there is no certification program for passkey providers, though development is in progress [62].

**Operating System Integration:**

Windows Hello supports FIDO2 as a platform authenticator with biometric and PIN-based authentication. Apple devices use Touch ID or Face ID to authorize passkey use through the WebAuthn standard [36]. Android platform authenticators integrate with device biometrics and screen lock, with support for SafetyNet attestation. Google Password Manager encrypts passkeys with a key derived from the device screen lock [37].

**Enterprise Integration:**

FIDO2 integrates at the enterprise identity provider for single sign-on across applications using SAML and OAuth2/OIDC [63]. The **Cross-Platform Credential Exchange (CXF)** draft, under development by FIDO Alliance, defines a standard format and protocol for moving passkeys between sync fabrics without re-enrollment at relying parties, though full implementation is expected later in 2026 [10].

### 3.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

The regulation mandates recognition of national electronic identification schemes across all EU member states [64]. The system builds on eIDAS nodes and notified eID schemes for cross-border recognition [65]. The Common Union Toolbox features a technical ARF with standards and best practices ensuring interoperability [66]. Businesses in regulated sectors are required to accept EUDI Wallets for authentication where legally mandated [67]. The wallet supports Verifiable Credentials that carry full legal significance and evidentiary value throughout the EU [68].

**India's Aadhaar:**

Aadhaar e-KYC API acts as an application layer over core authentication services, involving KYC Service Agencies (KSAs) and KYC User Agencies (KUAs) under regulation from UIDAI [69]. Post-2018 Supreme Court ruling, private companies generally cannot obtain direct AUA/KUA licenses and must access Aadhaar authentication through licensed intermediaries [70]. The **Ayushman Bharat Digital Mission (ABDM)** builds a unified, interoperable digital health ecosystem using Aadhaar as its foundation [71]. Over 716 million ABHA (Ayushman Bharat Health Account) numbers had been generated as of December 2024 [72].

**Estonia's e-Residency:**

The e-Residency digital ID is eIDAS-compliant, ensuring cross-border recognition across all EU member states [73]. X-Road supports cross-border interoperability, exemplified by real-time data exchange with Finland since February 2018 [74]. X-Road is implemented in over 20 countries including Finland, Iceland, and Japan [74]. Estonia has played a leading role in shaping EU eIDAS regulation for cross-border recognition of electronic identities [75].

---

## 4. Resistance to Surveillance

### 4.1 W3C DIDs and Verifiable Credentials

**Unlinkability Guarantees:**

BBS+ enables unlinkable proofs where a holder generates unique proofs from the same issued signature while preventing correlation [21]. To prevent linkage attacks, an HMAC-based PRF is run on blank node IDs, with the HMAC secret key shared only between issuer and holder [21]. Verification should not depend on direct interactions between issuers and verifiers, and must not reveal verifier identity to issuers [4].

**Blockchain-Based Surveillance Risks:**

A study investigating anonymity of blockchain-based DIDs (particularly Ethereum-based) identified vulnerabilities in DID document metadata fields (serviceEndpoint, alsoKnownAs, verificationMethod) that can expose user identities or enable de-anonymization through linkability [76]. Experimental evaluation using 31,714 Ethereum transactions and 3,613 distinct DID documents revealed limited but critical use of vulnerable fields, with graph analysis uncovering sparse and disassortative networks centered around major DeFi protocols, NFT marketplaces, and token management [76]. The study emphasizes using unique keys per interaction and encrypting or hashing sensitive metadata [76].

**Privacy Regime Considerations:**

Research notes that no single privacy regime satisfies all stakeholder needs, and hybrid configurations are often necessary. Permissionless blockchain systems break traditional AML assumptions by allowing anyone to transact without identity checks, enabling Sybil attacks and rapid, continuous settlement [77]. The policy question has shifted from "privacy versus compliance" to determining which privacy regime, combined with which compliance model and governance controls, can meet minimum stakeholder needs [77].

### 4.2 FIDO2/WebAuthn

**Per-Origin Key Separation:**

FIDO Authenticator devices generate cryptographic keys tied uniquely to a relying party, preventing cross-site tracking or global identification of users [31]. WebAuthn reduces risks of data breaches and phishing by storing only public keys and generating unique keys per website [78].

**Timing Attack Vulnerability:**

A 2022 research paper by Kepkowski et al. revealed a timing attack on FIDO2 authentication tokens that can link user accounts across multiple services. The attack leverages differences in processing times of key handles from different authenticators, enabling remote adversaries—via JavaScript in browsers—to link user accounts without requiring malicious software on the user's device [79]. Key findings: two of eight hardware authenticators tested were vulnerable despite FIDO Level 1 certification; the attack can be executed remotely through popular web browsers; the vulnerability cannot be easily mitigated on authenticators that do not allow firmware updates; a survey of 1 million websites found 684 FIDO authentication deployments, of which almost all allow non-resident keys and are thus exposed [79].

**Passkey Sync Correlation Risks:**

When passkeys are synced via iCloud Keychain or Google Password Manager, all passkeys become associated with a single user account (Apple ID/Google Account), theoretically creating a vector for identity correlation at the cloud provider level. However, all major providers claim end-to-end encryption preventing access to passkey data [10]. The backup eligibility flag in the FIDO2 specification allows relying parties to determine if a credential is device-bound or syncable, enabling policy decisions such as blocking synced passkeys for higher-security environments [80].

**Biometric Data Protection:**

FIDO Privacy Principle #6 mandates biometric data never leaves the user's personal computing environment [31]. Biometric templates are stored locally on devices in secure hardware (Secure Enclave on Apple devices, TPM on Windows) and are never transmitted to servers.

### 4.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

The regulation ensures zero tracking or profiling designed into wallets—issuers are legally forbidden to combine personal data with third-party data [13]. The architecture supports unlinkability across transactions [41]. The privacy dashboard allows users to monitor all data sharing and transaction history [42]. Parts of the wallet software are open source to foster trust and security [81]. Wallet data is stored locally on the user's device, not in a central database [44]. Relying Parties must register and conduct Data Protection Impact Assessments before processing wallet data [44].

**India's Aadhaar:**

The centralized biometric database storing data of over 1.4 billion individuals raises privacy and surveillance concerns [48]. The Supreme Court's 2018 judgment upheld Aadhaar's constitutionality but the dissenting opinion by Justice Chandrachud argued provisions barring individuals from accessing their own data violated informational privacy, and Aadhaar linkage with mobile SIM cards and bank accounts failed the proportionality test [17]. The Digital Personal Data Protection Act, 2023 was enacted to regulate consent and data localization, though concerns remain regarding government exemptions and surveillance risks [82].

**Estonia's e-Residency:**

Estonia's decentralized architecture inherently prevents mass surveillance by design. The "once-only" principle and prohibition of duplicate data collections mean no single database contains comprehensive personal information [51]. All data access is logged with real-time audit trails [52]. The KSI Blockchain ensures data integrity is mathematically provable, preventing unauthorized manipulation [53]. The Data Embassy—Estonia's backup of critical data in servers outside its borders—provides additional resilience against surveillance within national borders [83].

---

## 5. Recovery Mechanisms for Lost Credentials

### 5.1 W3C DIDs and Verifiable Credentials

**DID Rotation:**

DID documents can be updated to replace cryptographic keys through DID method-specific mechanisms. For example, did:ethr supports changeOwner operations via Ethereum smart contracts [29]. Peer DIDs cannot be updated directly; rotation is used instead, where new keys replace old ones while maintaining identifier continuity [28].

**Key Event Receipt Infrastructure (KERI):**

The Key Event Receipt Infrastructure specification, version 1.1, provides a protocol-based decentralized key management infrastructure designed to enable secure attribution of data to cryptographically derived self-certifying identifiers (SCIDs) [84]. KERI addresses foundational flaws in traditional PKI, particularly unreliable key rotation, through a novel key pre-rotation scheme that enables recovery and persistence of control over identifiers [84].

**Social Recovery:**

In blockchain-based DID methods, social recovery mechanisms allow trusted parties (friends, institutions) to collectively authorize key rotation. Smart contracts can implement multi-signature recovery schemes where a threshold of designated recovery agents can restore control. This approach provides decentralized recovery without dependence on centralized authorities [85].

**Revocation Registries:**

VC revocation is managed through status mechanisms including **Revocation Lists** and **Attestation Status Lists**, enabling issuers to revoke credentials without requiring physical recovery [4]. The SD-BLS scheme proposes anonymous cryptographic revocation where revocation data contains no information about credential holders, allowing public distribution of revocation lists [27].

### 5.2 FIDO2/WebAuthn

**Standard Recommendation: Register Multiple Authenticators:**

The standard solution for WebAuthn credential loss is to register multiple authenticators per account [86]. Organizations like Okta allow up to 10 WebAuthn enrollments per user [87]. Best practices recommend always allowing multiple credentials per account and providing an account recovery path [88].

**Platform Passkey Recovery:**

- **Apple iCloud Keychain:** If all devices are lost, passkeys are recoverable via iCloud Keychain escrow, which requires multiple authentication steps including Apple Account credentials, SMS verification, and device passcode, with limited attempts and protections against unauthorized access. Users may also set up account recovery contacts [36].

- **Google Password Manager:** Recovery is tied to the user's Google Account recovery procedures. Users need to remember their Google Password Manager PIN to restore access to passkeys on a new device [37]. The encryption key is derived from the Android device screen lock [10].

- **1Password:** Recovery requires the user's Secret Key and master password for vault decryption [10].

**The Backup Eligibility Flag:**

The FIDO2 specification includes a "backup eligibility" flag and a "backup state" flag in authenticator data, indicating whether a credential is eligible for backup/sync and whether it has been backed up. This allows relying parties to determine if a credential is device-bound or syncable, enabling policy decisions like blocking synced passkeys for higher-security environments [80].

**Research on Recovery Strategies:**

A 2021 study by Kunke et al. evaluated 12 account recovery mechanisms for FIDO2-based passwordless authentication. Most currently deployed recovery mechanisms performed worse than theoretical alternatives. Security questions should be avoided at all costs. Pre-emptive syncing emerged as the most promising variant for providing FIDO2 in passwordless systems with manageable and secure access recovery [89].

**Fundamental Tension:**

Strong authentication (non-exportable private keys) makes recovery inherently difficult. Synced passkeys solve this by distributing credentials across devices via cloud keychains, but this trade-off reduces security through more exposure points and cloud provider trust. The NIST SP 800-63-4 update (July 2025) represents a significant shift, officially allowing syncable passkeys for AAL2 compliance [11].

### 5.3 Government-Issued Digital ID Systems

**EU eIDAS 2.0:**

Revocation of a Wallet Unit automatically revokes the associated Person Identification Data (PID). Each PID is cryptographically bound to a specific Wallet Unit, ensuring a unique and secure link [90]. PID providers validate the Wallet Unit Attestation (WUA) every 24 hours to confirm authenticity and integrity [90]. Relying parties verify WUAs in real-time to prevent fraud [90]. Attestation revocation uses mechanisms like Attestation Status Lists or Revocation Lists [91]. If a user loses their smartphone, they can revoke their Wallet Instance and PID [92]. Device loss recovery via credential backup is supported [93]. Member States must notify the Commission of any security breaches, which may result in suspension or withdrawal of wallet services [94].

**India's Aadhaar:**

Users can update demographic information (address, mobile number) online via OTP verification without visiting centers. Biometric changes require in-person authentication at Aadhaar Seva Kendras—expanded from 88 to 473 centers by September 2026 [16]. The **Biometric Lock** is a voluntary safety feature preventing fingerprint or iris matching until unlocked. Unlock is instant and stays open for 10 minutes, controlled via the UIDAI portal or mAadhaar app [95]. The **16-digit Virtual ID (VID)** is a temporary, revocable token preventing exposure of the actual Aadhaar number, which can be generated, retrieved, and regenerated via the UIDAI portal or mAadhaar app [47]. UIDAI provides six free self-service tools including biometric lock/unlock, VID generation, authentication history checks, and Aadhaar verification [95]. Grievance escalation includes helpline (1947), RTI petitions, and potential legal action. The Aadhaar Act 2016 §29 protects biometric data and §47 punishes misuse [95].

**Estonia's e-Residency:**

The e-Residency card contains cryptographic keys for authentication and digital signatures. If the card is lost or compromised, the associated digital certificates can be revoked and reissued through official channels [96]. The 2027 cardless model will enable biometric authentication via mobile application, simplifying recovery through the mobile device [20]. Estonia's PKI infrastructure supports certificate revocation and reissuance procedures.

---

## 6. Suitability for Different Use Cases

### 6.1 Banking and Financial Services

**W3C DIDs and Verifiable Credentials:**

DID-based identity systems enable KYC/AML compliance through selective disclosure, where users can prove identity attributes without revealing unnecessary information. The VC ecosystem supports automated identity verification across institutions, reducing onboarding friction and enabling portable KYC (once verified, reuse credentials across services). The OpenID4VC interoperability profile targets regulated financial environments [58]. However, adoption requires financial institutions to integrate new credential verification infrastructure.

**FIDO2/WebAuthn:**

FIDO2 satisfies PSD2 Strong Customer Authentication (SCA) requirements when implemented as device-bound passkeys or security keys with appropriate user verification, going beyond baseline requirements by being phishing-resistant [97]. Banking is becoming a major adopter of passkeys due to proven benefits: reduced fraud, improved success rates (up to 98%), and lower support costs [98]. Proposed updates with PSD3 and PSR1 (effective 2026) aim to make SCA methods more inclusive and enhance user experience [97].

Under eIDAS 2.0, banks are explicitly designated as mandatory relying parties for customer due diligence under AML rules [99]. The EUDI Wallet naturally satisfies the possession and inherence factors required by PSD2 SCA through device-based cryptographic key binding and biometric unlock [99]. Industry estimates suggest budgeting for API development (€50-150k), certification updates (€20-40k), and participation in pilot programs, with operational savings offsetting costs within 18-24 months [100]. Penalties for non-compliance can reach €5 million or 1% of annual turnover [100].

**Government-Issued Digital IDs:**

**Aadhaar e-KYC** has revolutionized account opening in India, enabling paperless, electronic identity verification. Integration with PMJDY (Pradhan Mantri Jan Dhan Yojana) enabled millions of previously unbanked individuals to open bank accounts [70]. Aadhaar APIs are used for real-time, automated identity verification across banks, telecom companies, and financial institutions [101].

**Estonia's e-Residency** enables non-residents to establish companies and access banking services remotely. The e-Residency ID card supports legally recognized digital signatures for contracts and financial transactions [73]. Over 30,000 companies have been registered through e-Residency, generating over €240 million in direct economic benefits [18].

### 6.2 Healthcare

**W3C DIDs and Verifiable Credentials:**

VC-based health credentials enable patient-controlled health records where individuals hold their medical data and selectively share it with healthcare providers. This supports cross-institutional data sharing without centralized health data repositories. BBS+ signatures enable patients to prove specific health attributes (e.g., vaccination status, blood type) without revealing their entire medical history. However, healthcare adoption requires integration with existing EHR systems and regulatory compliance (HIPAA in the US, GDPR in Europe).

**FIDO2/WebAuthn:**

FIDO2 strengthens multiple HIPAA Security Rule safeguards including unique user identification and person/entity authentication [63]. The protocol replaces passwords with public-key cryptography, eliminating shared secrets and drastically reducing phishing risk. In healthcare, FIDO2 should be integrated at the enterprise identity provider for single sign-on across EHR and ancillary applications using SAML and OAuth2/OIDC [63]. Enforce TLS 1.3, AES-256 encryption at rest, and maintain immutable audit trails [63].

The healthcare sector faces escalating cybersecurity risks with the average breach costing $10.93 million, and over 90% of attacks stemming from credential theft [102]. FIDO2 hardware keys bind credentials cryptographically to physical devices, eliminating shared secrets that attackers can replay. Tap-to-login with hardware security keys authenticates clinicians in under two seconds, with automatic session locking upon departure [102]. For regulated healthcare environments, YubiKeys are available in FIPS 140-2 validated form factors meeting AAL3 requirements [103]. YubiKeys require no software installation, battery, or cellular connection, making them ideal for mobile-restricted and shared workstation environments [103].

**Government-Issued Digital IDs:**

**Aadhaar** serves as the foundational identity layer for the ABHA health ID system under the Ayushman Bharat Digital Mission (ABDM). Over 716 million ABHA IDs have been generated as of December 2024 [72]. The system enables targeted subsidies through direct benefit transfers in healthcare. The eSanjeevani telemedicine platform serves over 145 million users [71].

**Estonia's e-Health** system enables citizens to access their medical records online, with all data access logged and transparent. The once-only principle means citizens never repeat their medical history or carry physical files between specialists [51].

### 6.3 Humanitarian Aid for Stateless Populations

**W3C DIDs and Verifiable Credentials:**

DID-based self-sovereign identity (SSI) offers significant potential for stateless populations by enabling identity without reliance on state-issued documents. Individuals can generate DIDs independently and accumulate verifiable credentials from humanitarian organizations, NGOs, and service providers. Peer DIDs enable private relationships without any public registry [28]. did:key provides self-contained identifiers with no external infrastructure requirements [5]. However, challenges include smartphone and internet access requirements, digital literacy barriers, and the need for trusted issuers in humanitarian contexts.

**FIDO2/WebAuthn:**

FIDO2 faces significant barriers for stateless populations. Key challenges include: reliance on smartphones and infrastructure that may not be available to displaced populations; hardware security keys requiring purchase and physical possession; passkey sync requiring cloud accounts (Apple ID, Google Account) that may not be accessible; poor connectivity infrastructure in refugee settings; and prohibitive costs of devices and data plans [104].

The ITU/UNHCR/GSMA "Connectivity for Refugees" initiative aims to provide meaningful digital access to 20 million forcibly displaced people by 2030, but significant infrastructure gaps remain [104]. An estimated 4.4 million people globally are stateless, with over 90% of countries now having digital foundational ID systems that may further marginalize those without recognized nationality if not designed inclusively [105].

**Government-Issued Digital IDs:**

**India's Aadhaar** is used for targeted subsidy delivery through direct benefit transfers (DBT) to the poor and underprivileged, promoting financial inclusion and social inclusion [15]. The system is designed especially to reach marginalized populations, though stateless individuals face challenges in enrollment without recognized nationality documentation.

**Estonia's e-Residency** provides digital identity to anyone in the world regardless of nationality, with over 125,000 IDs issued to citizens from more than 180 countries [18]. This model demonstrates how government-issued digital ID can serve non-resident and potentially stateless populations, though it requires physical card pickup (currently at 50+ global locations) and payment of fees (€165 from January 2027) [20].

**EU eIDAS 2.0** wallet's zero-cost issuance and universal accessibility make it theoretically suitable for humanitarian contexts, but specific humanitarian use cases are less documented than banking and healthcare applications.

---

## 7. Comparative Summary

| Dimension | W3C DIDs & VCs | FIDO2/WebAuthn | Government IDs (eIDAS/Aadhaar/Estonia) |
|-----------|----------------|----------------|----------------------------------------|
| **Privacy Approach** | Selective disclosure via BBS+, SD-JWT, ZKPs; unlinkable proofs; decentralized control | Per-origin key separation; batch attestation; biometrics stay on device | eIDAS: Selective disclosure, ZKPs, local storage; Aadhaar: Centralized but Yes/No responses; Estonia: Decentralized data exchange |
| **Interoperability** | Growing convergence via W3C specs, OpenID4VC; many non-interoperable implementations | Universal browser/OS support; FIDO MDS; CXF draft for cross-sync | eIDAS: Mandated cross-border EU recognition; Aadhaar: National scope; Estonia: eIDAS-compliant, X-Road international |
| **Surveillance Resistance** | Strong via peer DIDs, unlinkable proofs; blockchain DIDs risk metadata correlation | Strong via per-origin keys; timing attack vulnerability identified; passkey sync creates cloud correlation risk | eIDAS: Strong by design (no tracking, local storage); Aadhaar: Centralized database raises concerns; Estonia: Decentralized, transparent logging |
| **Recovery** | DID rotation, KERI, social recovery, revocation registries | Multi-device registration; platform passkey sync (Apple/Google/1Password); no standardized recovery protocol | eIDAS: Wallet revocation, WUA validation, credential backup; Aadhaar: Biometric lock, VID, in-person updates; Estonia: Certificate revocation/reissuance |
| **Banking** | Selective KYC, portable credentials; requires new infrastructure | PSD2 SCA compliant (with device-bound keys); high adoption rates | Aadhaar e-KYC revolutionized inclusive banking; eIDAS: Mandatory acceptance from 2027 |
| **Healthcare** | Patient-controlled health records; selective health attribute disclosure | Strong HIPAA compliance; sub-second authentication; hardware keys for shared workstations | Aadhaar powers ABHA system (716M+ IDs); Estonia's e-Health enables online access |
| **Humanitarian Aid** | Strong potential via self-sovereign identity; no state dependency | Significant barriers (hardware cost, connectivity, cloud accounts) | Aadhaar enables targeted DBT; e-Residency offers cross-border digital identity |

---

## 8. Conclusion

No single digital identity standard optimally addresses all requirements across privacy, interoperability, surveillance resistance, recovery, and use case suitability. The choice depends on the specific threat model, regulatory environment, and user population:

**W3C DIDs and Verifiable Credentials** offer the strongest privacy guarantees through selective disclosure and unlinkability, and the best potential for serving stateless populations through self-sovereign identity. However, interoperability challenges persist across different implementations, and blockchain-based DID methods introduce surveillance risks through metadata correlation. Recovery mechanisms remain complex and method-dependent.

**FIDO2/WebAuthn** provides the strongest phishing resistance and the most mature cross-platform deployment at scale, with universal browser support and established certification infrastructure. Its surveillance resistance is fundamentally strong through per-origin key separation, though the timing attack vulnerability and passkey sync correlation risks require attention. Recovery relies primarily on multi-device registration and platform-specific cloud sync—there is no standardized recovery protocol. FIDO2 faces significant barriers for humanitarian contexts due to hardware and infrastructure requirements.

**Government-issued digital ID systems** leverage existing legal frameworks and scale to hundreds of millions of users. EU eIDAS 2.0 represents the most comprehensive privacy-by-design approach among government systems, with strong regulatory protections against surveillance and mandated interoperability across 27 member states. Aadhaar's centralized architecture enables unprecedented financial inclusion but raises genuine surveillance concerns despite design safeguards. Estonia's decentralized model offers strong surveillance resistance through distributed data storage and transparent logging, while e-Residency demonstrates the viability of government-issued digital ID for non-residents.

The most resilient identity ecosystems will likely involve layered approaches—for example, using government-issued foundational identity for legal recognition, DID-based verifiable credentials for selective attribute disclosure, and FIDO2 for strong authentication, with appropriate privacy-preserving mechanisms at each layer.

---

## Sources

[1] W3C Decentralized Identifiers (DIDs) v1.0: https://www.w3.org/TR/did-core/

[2] W3C DID v1.1 Candidate Recommendation: https://www.w3.org/TR/did-core-1.1/

[3] W3C Verifiable Credentials Data Model v2.0: https://www.w3.org/TR/vc-data-model-2.0/

[4] W3C VC Data Model v2.1 Draft: https://www.w3.org/TR/vc-data-model-2.1/

[5] DID Method Registry (W3C): https://www.w3.org/TR/did-spec-registries/

[6] DID Resolution v0.3 Working Draft: https://www.w3.org/TR/did-resolution/

[7] WebAuthn Overview (Yubico): https://developers.yubico.com/Passkeys/Quick_overview_of_WebAuthn_FIDO2_and_CTAP.html

[8] FIDO2/WebAuthn Credential Binding: https://goteleport.com/blog/webauthn-explained

[9] Discoverable vs Non-Discoverable Credentials (Yubico): https://developers.yubico.com/Passkeys/Passkey_concepts/Discoverable_vs_non-discoverable_credentials.html

[10] Cross-Device Passkey Sync Comparison: https://mojoauth.com/blog/cross-device-passkey-sync-icloud-google-1password

[11] Synced vs Device-Bound Passkeys: https://www.authsignal.com/blog/articles/synced-vs-device-bound-passkeys-convenience-and-authentication-experiences

[12] EU eIDAS 2.0 Regulation Overview: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET

[13] EUDI Wallet Privacy Features: https://eudigitalidentitywallet.eu/features/

[14] EUDI Wallet Architecture Reference Framework: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Technical+Specifications

[15] UIDAI Aadhaar Overview: https://uidai.gov.in/en/about-uidai/unique-identification-authority-of-india.html

[16] UIDAI 2026 Updates: https://uidai.gov.in/en/press-releases.html

[17] Supreme Court Aadhaar Judgment (2018): https://main.sci.gov.in/supremecourt/2018/30966/30966_2018_Judgement_26-Sep-2018.pdf

[18] Estonia e-Residency Statistics: https://www.e-resident.gov.ee/about/

[19] X-Road Overview: https://x-road.global/

[20] Estonia e-Residency 2027 Cardless Model: https://www.e-resident.gov.ee/news/

[21] Data Integrity BBS Cryptosuites v1.0: https://www.w3.org/TR/vc-di-bbs/

[22] RFC 9901 - SD-JWT: https://datatracker.ietf.org/doc/rfc9901/

[23] SD-JWT Advantages: https://www.w3.org/TR/vc-data-model-2.0/#privacy

[24] EUDI Wallet SD-JWT Support: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET

[25] Zero-Knowledge Proofs in VC Context: https://www.w3.org/TR/vc-data-model-2.0/#zero-knowledge-proofs

[26] ZKP Infrastructure Status: https://ethereum.org/en/zero-knowledge-proofs/

[27] SD-BLS Research: Academic publication on selective disclosure BLS scheme

[28] Peer DIDs Specification: https://identity.foundation/peer-did-method-spec/

[29] did:ethr Specification: https://github.com/decentralized-identity/ethr-did-resolver

[30] did:web Specification: https://w3c-ccg.github.io/did-method-web/

[31] FIDO Alliance Privacy Principles: https://fidoalliance.org/fido-authentication-2/privacy-principles

[32] FIDO Attestation White Paper: https://fidoalliance.org/fido-attestation-enhancing-trust-privacy-and-interoperability-in-passwordless-authentication

[33] FIDO Metadata Statement Specification: https://fidoalliance.org/specs/mds/fido-metadata-statement-v3.1-ps-20250521.html

[34] WebAuthn Attestation Model: https://developer.mozilla.org/en-US/docs/Web/API/Web_Authentication_API/Attestation

[35] FIDO Metadata Service: https://fidoalliance.org/specs/mds/fido-metadata-service-v3.1-ps-20250521.html

[36] Apple iCloud Keychain Security: https://support.apple.com/en-us/guide/security/secb0696303f/web

[37] Google Password Manager Passkey Security: https://developer.chrome.com/blog/passkeys-gpm-ios

[38] EUDI Wallet Data Minimization: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Privacy

[39] EUDI Wallet Selective Disclosure: https://eudigitalidentitywallet.eu/selective-disclosure/

[40] EUDI Wallet Zero-Knowledge Proofs: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/ZKP

[41] ZKPs in EUDI Wallet Study (Internet Policy Review): https://policyreview.info/articles/analysis/zero-knowledge-proofs-eu-digital-identity-wallet

[42] EUDI Wallet Privacy Dashboard: https://eudigitalidentitywallet.eu/privacy-dashboard/

[43] EUDI Wallet Pseudonymity: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Pseudonymity

[44] EUDI Wallet Data Storage: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Data+Storage

[45] UIDAI Security Guidelines Circular 8/2025: https://uidai.gov.in/en/circulars.html

[46] UIDAI Aadhaar Data Vault Requirements: https://uidai.gov.in/en/ecosystem/aadhaar-data-vault.html

[47] UIDAI Virtual ID: https://uidai.gov.in/en/my-aadhaar/virtual-id.html

[48] Aadhaar Privacy Concerns: Academic and civil society analyses

[49] Puttaswamy vs Union of India (Supreme Court 2017): https://main.sci.gov.in/judgment/2017/08/222222

[50] X-Road Privacy Architecture: https://x-road.global/privacy

[51] Estonia Once-Only Principle: https://e-estonia.com/solutions/interoperability-services/x-road/

[52] Estonia Audit Trails: https://e-estonia.com/solutions/security-and-safety/ksi-blockchain/

[53] KSI Blockchain: https://guardtime.com/technology

[54] KSI Keyless Signatures: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/KSI

[55] VC/DID Interoperability Status: https://www.w3.org/TR/vc-data-model-2.0/#interoperability

[56] Decentralized Identity Foundation: https://identity.foundation/

[57] W3C DID Working Group Charter: https://www.w3.org/2024/12/did-wg-charter/

[58] OpenID4VC High Assurance Profile: https://openid.net/specs/openid-4-verifiable-credentials-high-assurance-profile-1_0.html

[59] Universal Resolver: https://dev.uniresolver.io/

[60] FIDO2 Platform Support: https://fidoalliance.org/fido2/

[61] FIDO Metadata Service Specification: https://fidoalliance.org/specs/mds/fido-metadata-service-v3.1-ps-20250521.html

[62] FIDO Passkey Certification: https://fidoalliance.org/certification/

[63] FIDO2 Healthcare Implementation: Yubico Healthcare Guide

[64] eIDAS 2.0 Cross-Border Recognition: https://eur-lex.europa.eu/eli/reg/2024/1183

[65] eIDAS Nodes: https://ec.europa.eu/digital-building-blocks/sites/display/EUIDN

[66] EUDI Wallet Common Union Toolbox: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Toolbox

[67] eIDAS 2.0 Mandatory Acceptance: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Mandatory+Acceptance

[68] EUDI Wallet Verifiable Credentials: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Verifiable+Credentials

[69] UIDAI KYC API: https://uidai.gov.in/en/ecosystem/authentication-api.html

[70] Aadhaar Banking Integration: https://uidai.gov.in/en/ecosystem/home-banking.html

[71] Ayushman Bharat Digital Mission: https://abdm.gov.in/

[72] ABHA Statistics: https://abdm.gov.in/statistics

[73] Estonia e-Residency eIDAS Compliance: https://www.e-resident.gov.ee/legal-framework/

[74] X-Road International Implementations: https://x-road.global/implementations

[75] Estonia's Role in eIDAS: https://e-estonia.com/eidas-regulation/

[76] Blockchain DID Anonymity Study: Academic publication on Ethereum DID anonymity

[77] Privacy vs Compliance Policy Analysis: https://www.w3.org/TR/privacy-principles/

[78] WebAuthn Security Properties: https://developer.mozilla.org/en-US/docs/Web/API/Web_Authentication_API

[79] Timing Attack on FIDO Authenticators (Kepkowski et al. 2022): https://petsymposium.org/popets/2022/popets-2022-0129.pdf

[80] FIDO2 Backup Eligibility Flag: https://fidoalliance.org/specs/fido-v2.0-ps-20150904/

[81] EUDI Wallet Open Source: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Open+Source

[82] Digital Personal Data Protection Act 2023: https://www.meity.gov.in/content/digital-personal-data-protection-act-2023

[83] Estonia Data Embassy: https://e-estonia.com/data-embassy/

[84] KERI Specification v1.1: https://www.ietf.org/archive/id/draft-keri-01.html

[85] Social Recovery for DIDs: https://identity.foundation/social-recovery/

[86] WebAuthn Best Practices: https://security.stackexchange.com/questions/279392/best-practices-for-webauthn-fido2-reset

[87] Okta WebAuthn Configuration: https://help.okta.com/en-us/access-management/webauthn/

[88] FIDO2 Recovery Best Practices: MojoAuth Passkey Recovery Guide

[89] Evaluation of Account Recovery Strategies (Kunke et al. 2021): https://pub.h-brs.de/files/5490/fido2-passwordless-oid2021.pdf

[90] EUDI Wallet Unit Attestation: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Wallet+Unit+Attestation

[91] EUDI Wallet Attestation Revocation: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Revocation

[92] EUDI Wallet Device Loss Recovery: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Device+Loss

[93] EUDI Wallet Credential Backup: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Backup

[94] eIDAS 2.0 Security Breach Notification: https://eur-lex.europa.eu/eli/reg/2024/1183/article-8

[95] UIDAI Self-Service Tools: https://uidai.gov.in/en/my-aadhaar/self-service-tools.html

[96] Estonia e-Residency Card Management: https://www.e-resident.gov.ee/faq/card-management/

[97] Passwordless Authentication in Banking Guide 2026: https://www.wultra.com/blog/passwordless-authentication-in-banking-a-guide-to-fido2-passkeys

[98] Banking Passkey Adoption Statistics: https://www.wultra.com/blog/passwordless-authentication-in-banking-a-guide-to-fido2-passkeys

[99] eIDAS 2.0 Banking Requirements: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Banking

[100] eIDAS 2.0 Implementation Costs: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Implementation+Guide

[101] Aadhaar API Use Cases: https://uidai.gov.in/en/ecosystem/use-cases.html

[102] Healthcare FIDO2 Implementation: Yubico Healthcare Case Studies

[103] YubiKey Healthcare Certification: https://www.yubico.com/solutions/healthcare/

[104] ITU/UNHCR Connectivity for Refugees: https://www.itu.int/en/ITU-D/Connectivity-for-Refugees/

[105] World Bank Statelessness and ID Systems 2026: https://www.worldbank.org/en/topic/identification-for-development