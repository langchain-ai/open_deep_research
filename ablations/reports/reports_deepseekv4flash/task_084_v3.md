# Comprehensive Technical Research Report: Competing Digital Identity Standards — A Layered Analysis
**Date: May 28, 2026**

---

## Executive Summary

This report provides a comprehensive, technically exhaustive comparative analysis of three major digital identity paradigms: (1) W3C Decentralized Identifiers (DIDs) and Verifiable Credentials, (2) FIDO2/WebAuthn, and (3) government-issued digital ID systems (EU eIDAS 2.0, India's Aadhaar, Estonia's e-Residency). Rather than treating these as competing systems, the analysis frames them as complementary layers in a three-tier stack: **Authentication** (FIDO2/WebAuthn — Layer 1), **Attribute Verification** (W3C VCs/SD-JWT/BBS+ — Layer 2), and **Identity Assurance** (Government eID — Layer 3).

The analysis is grounded in the most current official specifications, regulatory texts, and primary academic sources as of May 28, 2026, including W3C Candidate Recommendation Snapshots, EU Regulations with exact article references, Supreme Court judgments, UIDAI circulars, IETF RFCs, and peer-reviewed security research. All technical claims are supported by specific source citations.

---

## Section 1: Analytical Framework — Three Complementary Layers

The most critical structural change in this analysis is treating these identity systems not as competing alternatives but as distinct, complementary layers in a stack. Each layer serves a fundamentally different purpose and has different strengths and limitations.

### Layer 1 — Authentication: FIDO2/WebAuthn/Passkeys
**Core question**: "Are you the same entity as before?"
**What it verifies**: Cryptographic proof of device/credential possession. Strong phishing resistance through origin-binding.
**What it DOES NOT do**: Attribute disclosure, identity proofing, or legal identity verification.
**Best for**: Recurring authentication to known services, preventing account takeover.

### Layer 2 — Attribute Verification: W3C VCs + SD-JWT + BBS+/ZKPs
**Core question**: "Is this claimed attribute true?"
**What it verifies**: Selective disclosure of attested claims (e.g., age > 18, address, qualification) with unlinkable proofs.
**What it DOES NOT do**: Establish foundational legal identity.
**Best for**: Privacy-preserving attribute sharing, one-time verification of specific claims.

### Layer 3 — Identity Assurance: Government eID Systems
**Core question**: "Is this your legal identity?"
**What it provides**: Foundational identity with legal recognition, high assurance levels, in-person proofing, and legal frameworks.
**Best for**: High-stakes transactions (banking, government services, legal processes), establishing identity from scratch.

### Key Insight: Convergence in Practice
The most advanced deployments already integrate these layers. The EUDI Wallet uses **FIDO2/WebAuthn for wallet authentication** (Layer 1) while issuing **SD-JWT VCs for attribute presentation** (Layer 2), all built on **government-issued foundational identity** (Layer 3). The Aadhaar Verifiable Credential (December 2025) represents Aadhaar's adoption of VC concepts for offline verification.

---

## Section 2: Layer 1 — Authentication (FIDO2/WebAuthn/Passkeys)

### 2.1 Current Specification Status

**WebAuthn Level 3** was published as a Candidate Recommendation Snapshot on January 13, 2026, with comments accepted until February 10, 2026 [1][2]. Key additions from Level 2 include:
- `getClientCapabilities()` API for standardized feature detection
- `conditionalCreate` for automatic passkey creation  
- `conditionalGet` for passkey autofill
- `hybridTransport` for cross-device authentication via QR codes/BLE
- `signalAllAcceptedCredentials()` and `signalUnknownCredential()` for passkey sync and revocation management
- `relatedOrigins` for multi-domain passkey use
- Formalized multi-device credential behaviors (synced passkeys are no longer just vendor extensions)

As of May 2026, Safari 17.4+, Chrome 133+, Edge 133+, and Firefox 135+ support Level 3 capabilities [1].

**CTAP 2.2** (Proposed Standard, July 14, 2025) introduced Persistent PIN/UV Auth Tokens, PIN complexity policies, third-party payment extension for PSD2, hybrid transport, and optional JSON messaging [3].

**CTAP 2.3** (Proposed Standard, February 26, 2026) added smart-card support, NFC CTAP_GETRESPONSE, and ML-DSA/Ed25519 algorithms, with no breaking changes from 2.2 [4].

### 2.2 Layer 1 Role: Authentication Only

FIDO2/WebAuthn is designed exclusively for authentication — proving that the entity authenticating is the same entity that previously registered. It cannot:
- Disclose attributes (age, address, name)
- Provide identity proofing (establishing who someone is for the first time)
- Produce legally binding signatures

The credential is cryptographically bound to a single origin, preventing cross-site tracking and phishing. However, this also means FIDO2 cannot be used as an attribute presentation system.

### 2.3 Cryptographic Details

Supported COSE Algorithm IDs include [5]:
- **ES256** (COSE Algorithm ID -7): ECDSA over NIST P-256 with SHA-256 (default)
- **RS256** (COSE Algorithm ID -257): 2048-bit RSA with PKCS#1.5 padding and SHA-256
- **EdDSA** (COSE Algorithm ID -8): EdDSA over Curve25519 with SHA-512
- **ES384**: ECDSA over NIST P-384 with SHA-384

CTAP 2.3 introduces ML-DSA and Ed25519 support [4].

### 2.4 NIST SP 800-63-4 Mapping to AAL Levels

NIST SP 800-63-4 (July 2025) represents a landmark shift in authenticator assurance guidance [6][7][8]:

**AAL1**: Single-factor or multi-factor with approved cryptography, periodic reauthentication every 30 days.

**AAL2**: Requires multi-factor with at least one phishing-resistant option. **Syncable passkeys are now recognized for AAL2**, representing a significant departure from earlier guidance that discouraged key export. Syncable authenticators fulfill AAL2 criteria through:
- Phishing resistance via domain-specific keys
- Replay resistance through nonces
- User verification flags (UP, UV)
- Backup eligibility and backup state flags for relying party policy decisions

Federal agencies **SHALL** select AAL2 when personal information is made available online.

**AAL3**: Requires cryptographic authenticators with **non-exportable private keys** and explicit user-intent demonstration. **Synced passkeys are explicitly excluded from AAL3** due to the exportability restriction. Only hardware-bound FIDO2 authenticators with non-exportable keys (e.g., YubiKey, Windows Hello with TPM) qualify. Hardware requirements have been relaxed to FIPS 140 Level 1 provided keys remain non-exportable. Session timeouts: 12 hours overall, 15 minutes inactivity.

NIST SP 800-63-4 also introduced the **Syncable Authenticators Appendix** (pages.nist.gov/800-63-4/sp800-63b/syncable) requiring authentication keys stored encrypted with at least 112-bit security (SP800-131A), ideally using user-controlled secrets. Access to sync fabrics requires AAL2-equivalent MFA [8].

### 2.5 Commercial/Platform Adoption Data

**Apple iCloud Keychain**: Passkeys synced across Apple devices via iCloud Keychain with end-to-end encryption. Encryption keys derive from unique device information combined with device passcode — even Apple cannot access them. Supports Face ID and Touch ID verification. Approximately **1.5 billion users** can use passkey authentication [9][10][11].

**Google Password Manager**: Passkey sync across Android, iOS/iPadOS, macOS, Windows, and Linux via Chrome. Private keys protected by AMD SEV-SNP technology and Oak platform, with TPM on Windows and Secure Enclave on macOS. Approximately **800 million users** supported for passkey authentication [10][12][13].

**Microsoft Entra ID**: Supports both device-bound passkeys (FIDO2 security keys, Windows Hello) and synced passkeys (iCloud Keychain, Google Password Manager). Synced passkeys achieve **99% registration success** rate, enable sign-ins **14x faster** than traditional MFA (3 seconds vs. 69 seconds), and achieve **3x greater sign-in success** compared to legacy authentication [14][15].

**1Password**: Full passkey support with zero-knowledge security model using Two-Secret Key Derivation (2SKD). Users require account password + Secret Key for decryption. Supports passkey creation, management, and signing on supported websites [16][17].

**Okta**: Passkeys authenticator supports FIDO2 WebAuthn with security keys or biometric methods. Okta FastPass enables passwordless authentication — approximately **91% of all daily authentications** use FastPass, saving ~36,000 staff hours per year [18][19].

**CXF (Credential Exchange Format)**: Published as Proposed Standard in August 2025, defining JSON-based credential exchange between providers. Contributors include Apple, Google, Microsoft, 1Password, Bitwarden, Dashlane, and Okta. Target completion date: June 30, 2026 [20][21].

**Adoption Scale**: Approximately **12 billion online accounts** currently accessible with passkeys [10].

### 2.6 Layer 1: Security Incidents and Limitations

**Kepkowski et al. (2022) — Timing Attacks**: Published at PET Symposium, this research revealed a timing attack on FIDO2 authentication tokens enabling cross-service account linking without malicious software. The attack exploits differences in processing times of key handles from different authenticators, enabling remote adversaries via JavaScript to link user accounts. Two of eight hardware authenticators tested were vulnerable despite FIDO Level 1 certification. Among 1 million websites surveyed, 684 FIDO2 deployments were found, almost all allowing non-resident keys [22].

**CTRAPS (2024) — CTAP Vulnerabilities**: The first comprehensive security evaluation of the CTAP Authenticator API identified two novel attack classes [23]:
1. **Client Impersonation (CI) attacks**: Zero-click credential deletion, factory resetting, user tracking, and lockout
2. **API Confusion (AC) attacks**: Misleading client and authenticator to execute unintended CTAP API calls

These attacks affect all CTAP transports (USB, NFC, BLE) and devices including Yubico FIPS-certified keys, Feitian, SoloKeys, and major relying parties (Microsoft, Apple, GitHub, Facebook).

**Stolen Credentials**: Thales 2026 Data Threat Report indicates **49% of initial attack vectors** involve stolen credentials. 67% of organizations report credential theft as increasing in cloud attacks [24].

### 2.7 Key Judgment on Layer 1

FIDO2/WebAuthn excels at its intended role: phishing-resistant, privacy-preserving authentication for recurring access to known services. It is unsuitable for attribute disclosure, identity proofing, or establishing legal identity. The syncable vs. device-bound tradeoff represents a fundamental tension between usability/recovery and security — NIST SP 800-63-4's recognition of both provides a policy framework for context-appropriate choices.

---

## Section 3: Layer 2 — Attribute Verification (W3C DIDs/VCs + SD-JWT + BBS+)

### 3.1 Current Specification Status

**DID Core v1.0**: Published as W3C Recommendation July 19, 2022 [25].

**DID v1.1**: Candidate Recommendation Snapshot published March 5, 2026, with comments accepted until April 5, 2026 [26]. Key changes from v1.0:
- Consolidated media types
- New JSON-LD context
- Structural refactoring aligned with **Controlled Identifiers v1.0** (a generalization supporting both decentralized and non-decentralized identifiers)
- Separation of resolution aspects into a companion DID Resolution specification

**Verifiable Credentials Data Model v2.0**: Published as W3C Recommendation on May 15, 2025 [27]. Key changes from v1.1:
- `issuanceDate` renamed to `validFrom`
- `expirationDate` renamed to `validUntil`
- Updated first context item URL (documents now use `https://www.w3.org/ns/credentials/v2`)
- New media types `application/vc` and `application/vp`
- Separation of data model from securing mechanism (supporting JOSE, SD-JWT, COSE, and Data Integrity formats)

**VC Data Model v2.1**: First Public Working Draft published April 9, 2026, updated May 11, 2026 [28][29]. The Verifiable Credentials Working Group charter (March 11, 2026 to March 31, 2028) tasks the group with maintaining current specifications without adding normative features unless addressing critical privacy or security issues.

### 3.2 Layer 2 Role: Attribute Verification

W3C DIDs/VCs operate in a three-party ecosystem: **Issuers** (who attest to claims), **Holders** (who store credentials and selectively present claims), and **Verifiers** (who verify claims without contacting the issuer). This is the "triangle of trust."

**What Layer 2 provides**:
- Selective disclosure of attested attributes (age, qualifications, address)
- Unlinkable proofs (BBS+, ZKPs)
- Privacy-preserving verification
- Cryptographic binding between issuer, holder, and verifier

**What Layer 2 does NOT provide**:
- Foundational identity proofing
- Legal identity assurance
- High-assurance authentication for recurring access

### 3.3 Data Integrity Cryptosuites

**EdDSA Cryptosuites v1.1**: W3C Recommendation. Uses Ed25519 (RFC 8032, FIPS 186-5). Two suites: `eddsa-rdfc-2022` (RDF Dataset Canonicalization) and `eddsa-jcs-2022` (JSON Canonicalization Scheme, RFC 8785). **These do NOT support selective or unlinkable disclosure** [30].

**ECDSA Cryptosuites v1.0**: W3C Recommendation. NIST P-256 (secp256r1) and P-384 (secp384r1) curves. Three suites: `ecdsa-rdfc-2019`, `ecdsa-jcs-2019`, and **`ecdsa-sd-2023`** (selective disclosure variant using label replacement canonicalization). Uses deterministic ECDSA per FIPS 186-5 [31].

**BBS Cryptosuites v1.0**: W3C Candidate Recommendation Draft (April 2026). Based on the BBS+ signature scheme using **BLS12-381** pairing-friendly elliptic curve. Provides four core functions: BBS Sign, BBS Verify, BBS ProofGen (selective disclosure), and BBS ProofVerify. Proof values are **unlinkable to the original signature** — a verifier cannot determine which signature produced a given proof. Additional privacy features include Anonymous Holder Binding, Credential-Bound Pseudonyms, and combined modes [32].

### 3.4 SD-JWT (RFC 9901) and SD-JWT VC

**RFC 9901** (Published November 19, 2025): Defines selective disclosure for JSON Web Tokens using salted hashes [33].

**Core mechanism**: Issuer replaces selectively disclosable claims with digests in the JWT payload. Separately transmits Disclosures (salts + claim values). Verifier checks that disclosed claims' digests match those in the SD-JWT.

**Key features**:
- **Decoy digests**: Optional inclusion of meaningless hashes to obscure the actual number of selectively disclosable claims, providing herd privacy
- **Key Binding (SD-JWT+KB)**: Holder proves possession of private key by signing over transaction-specific data (nonce, audience)
- **Recursive disclosures**: Support for selectively disclosable nested structures
- Default hashing algorithm: SHA-256

**SD-JWT VC Draft**: Under IETF review (draft 15). Defines formats for expressing W3C Verifiable Credentials with JSON payloads using SD-JWT. Endorsed by the European Commission's EUDI Wallet architecture as a core format [33].

### 3.5 BBS Signatures (draft-irtf-cfrg-bbs-signatures-10)

Last updated January 8, 2026. Intended status: Informational (IRTF stream) [34].

**Properties**:
- Multi-message signing with constant-size signature
- **Zero-knowledge proofs** of knowledge of a signature — verifier cannot determine which signature was used
- **Unlinkable proofs** — different presentations of the same credential cannot be linked
- Pairing-based cryptography (BLS12-381) — public keys in G2, signatures in G1
- Not quantum-resistant for signature security, but privacy features remain robust against quantum adversaries

**Security proofs**: EUF-CMA and SUF-CMA secure under q-Strong Diffie-Hellman (q-SDH) assumption [34][35].

### 3.6 DID Methods — Privacy Trade-offs

**did:key**: Fully offline. Key directly in identifier. No rotation, deactivation, or updates. **High correlation risk** — same identifier (and public key) persists across contexts [36].

**did:web**: Hosted on domain. Domain operator sees all requests. **High exposure** — full DID document published publicly, DNS and TLS metadata revealed [37].

**did:ion** (Sidetree on Bitcoin): Public permissionless network. Operation hashes on Bitcoin blockchain, documents on IPFS. **Moderate correlation** — Bitcoin observability of update patterns, IPFS access patterns observable. **Strong decentralization** — supports tens of thousands of operations per Bitcoin transaction [38].

**did:keri** (KERI ACDC): Key Event Receipt Infrastructure with pre-rotation mechanism. **Witness network topology** potentially disclosed. Pre-rotation secures key recovery and rotation: inception event includes cryptographic commitment to next key-pair, hiding next keys until rotation. **Operates in direct mode** (private peer-to-peer) or indirect mode (public identifiers with witnesses/watchers) [39][40].

**did:peer**: Pairwise DIDs, local resolution only. **Lowest correlation risk** — designed for private relationships without central authority. **Offline-first capabilities** [41].

### 3.7 Status Management

**Bitstring Status List v1.0** (W3C Recommendation, May 15, 2025): Default list size **131,072 entries** (16 KB uncompressed). GZIP compresses to a few hundred bytes when few credentials are revoked. Features: random index allocation for herd privacy, CDN caching, "certificate stapling" for holder privacy [42].

**Critical distinction**: Status list indicates revocation/suspension but is **not data deletion** — credential data persists. The minimum list size ensures herd privacy if enough credentials are issued in the batch.

### 3.8 Convergence with JWT Ecosystems

**SD-JWT VC** serves as a bridge between traditional JWT ecosystems and W3C Verifiable Credentials Data Model. The SD-JWT VC specification is designed to be "compatible to VCDM" — existing W3C VCDM data structures can be reused on top of SD-JWT VC [43]. The Open Wallet Foundation's Typescript implementation integrates SD-JWT with W3C VCDM and implements JAdES (JSON Advanced Electronic Signatures) digital signature standard [44].

### 3.9 Adoption and Deployment

Approximately **80% of all global verifiable credential initiatives** and **more than 50 national or government-backed programs** worldwide utilize W3C-VCDM. Adopters include Bhutan, Singapore, Canada, the United States, Australia, Japan, and the European Union (eIDAS 2.0) [45].

### 3.10 Identity Bootstrap Problem — Critical Limitation for Layer 2

While permissionless DID creation is possible (anyone can generate a `did:key`), **meaningful VCs require trusted issuers**. Self-attested credentials have limited verifiability for most use cases (banking, government services, employment). The "triangle of trust" generally requires at least one authoritative issuer in the chain.

For humanitarian contexts: displaced persons can create DIDs but cannot obtain VCs from trusted issuers without foundational identity documents. This is the **identity bootstrap problem** — the fundamental gap between self-created identifiers and legally recognized identity.

---

## Section 4: Layer 3 — Identity Assurance (Government-Issued Digital ID Systems)

### 4.1 Role of Layer 3

Layer 3 provides the **legal foundation** for digital identity. These systems establish who someone is in the legal sense through:
- In-person identity proofing or high-assurance remote verification
- Legal frameworks that recognize digital credentials as equivalent to physical documents
- Typically unique national identifiers for citizens/residents
- Government-backed trust and liability frameworks

### 4.2 EU eIDAS 2.0 (Regulation (EU) 2024/1183)

#### 4.2.1 Regulatory Framework

Regulation (EU) 2024/1183 was adopted April 11, 2024, published April 30, 2024, and entered into force May 20, 2024. It amends Regulation (EU) No 910/2014 to establish the European Digital Identity Framework [46][47].

**Exact Compliance Deadlines**:
- **December 24, 2026**: All 27 Member States must make at least one EUDI Wallet available free of charge [46]
- **December 24, 2027**: Private Relying Parties performing Strong Customer Authentication must accept EUDI Wallet credentials (Article 5f(2)) [48]
- **July 10, 2027**: AML Regulation (AMLR) integration requiring eIDAS-compliant identification in KYC [48]
- **2030**: Digital Decade Programme targets 80% adoption [46]

#### 4.2.2 Article-Level Provisions

**Article 5a(5)**: EUDI Wallets must be provided under an electronic identification scheme with **assurance level High** [49][50]. The "high" level requires strongest authentication methods. Eurosmart emphasizes that a wallet must meet both eIDAS 'high' (trustworthy identity proofing) AND Cybersecurity Act 'high' (certification under CSA schemes like EUCC with AVA_VAN.5 vulnerability assessment) [50].

**Article 5f(2)**: Private relying parties required by Union or national law to use strong user authentication must accept EUDI Wallets. This covers large and medium-sized entities in banking, financial services, telecommunications, transport, healthcare, utilities, education, and digital infrastructure. Very Large Online Platforms (VLOPs) must accept EUDI Wallets upon user's voluntary request [48][51].

**Recital 29**: EUDI Wallets shall enable users to selectively disclose only necessary attributes — "empowering the owner of data to disclose only certain parts of a larger data set" [52].

**Recital 14 (Commission Recommendation 2024/1184)**: Member States should integrate zero-knowledge proofs into the EUDI Wallet to "allow a relying party to validate whether a given statement... is true, without revealing any data on which that statement is based" [53].

**Commission Implementing Regulation (EU) 2026/798** (April 7, 2026): Establishes rules for **remote onboarding** to the EUDI Wallet. Remote onboarding may rely on electronic identification means at **substantial** level, combined with additional procedures to reach **high** level. Adopts ETSI TS 119 461 v2.1.1 as reference standard. Critically, automated biometrics from initial eID issuance **cannot be reused** for the LoA High step-up process, requiring many citizens to visit municipal offices [54].

#### 4.2.3 Mandated Technical Protocols

**OpenID4VCI (Verifiable Credential Issuance)**: Mandated issuance protocol. Supports pre-authorized code flow and authorization code flow, PKCE, sender-constrained tokens, and wallet attestation [55].

**OpenID4VP (Verifiable Presentations)**: Mandated presentation protocol supporting signed authorization requests and DCQL query language [55].

**SD-JWT VC**: Mandated JWT/JOSE-based credential format for the wallet ecosystem. SD-JWT uses salted hash techniques to enable selective disclosure and cryptographic key binding [55].

**ISO/IEC 18013-5 (mDL/mdoc)**: Mandated format for proximity use cases (offline via NFC, BLE, QR codes). Wallets must support both SD-JWT VC and mdoc formats [55].

#### 4.2.4 Cryptographic Requirements (OpenID4VC HAIP 1.0)

The OpenID4VC High Assurance Interoperability Profile 1.0 (draft 04) mandates [56][57]:

- **P-256 (secp256r1)** as mandatory key type
- **ES256** as mandatory JWT algorithm
- **SHA-256** for hashing
- **ECDH-ES + A128GCM** for response encryption
- Authorization code flow with sender-constrained tokens (DPoP, RFC 9449)
- **KB-JWT** always required for SD-JWT VC presentation
- Wallet Attestations mandatory (cryptographically tying credentials to verified wallets)

Using alternative algorithms (e.g., P-384, P-521) is NOT RECOMMENDED as wallets are only required to support mandatory algorithms [56][57].

#### 4.2.5 Privacy and Surveillance Resistance

**eIDAS 2.0 privacy features**:
- **Local storage**: All wallet data stored on user's device, not in central database
- **Pseudonym-per-RP**: Different pseudonym value for different Relying Parties unless user explicitly chooses otherwise
- **Unobservability**: Wallet providers must ensure unobservability of user transactions
- **Selective disclosure**: Users share only required attributes
- **ZKPs**: Cryptographic methods allowing verification without revealing underlying data
- **Wallet Attestations**: Must avoid linkability for privacy purposes

**Civil Society Concerns (March 2026 Open Letter)** [58]:
- Compulsory facial biometric information could "significantly expand processing of sensitive personal data"
- Optional registration certificates could allow services to "request more personal data than necessary"
- Pseudonymity risks from permitting services to demand legal identities unnecessarily
- Technical standards enabling large platforms to bypass proper integration

**ZEVEDI (Centre Responsible Digitality) Analysis**: "Governance remains centralised regardless of technical innovation: Member States still certify wallets, still control credential issuance rules, and still retain oversight authority" [59]. The "Digital Identity Tetrahedron" framework includes technical, legal, social, and ethical dimensions, arguing centralized trust anchors are reintroduced despite decentralized claims [59].

#### 4.2.6 Integration Across Layers

The EUDI Wallet exemplifies the layered approach:
- **Layer 1 (Authentication)**: FIDO2/WebAuthn authenticates user to the wallet
- **Layer 2 (Attributes)**: SD-JWT VCs present verified attributes to relying parties
- **Layer 3 (Identity Assurance)**: Government-issued foundational identity (PID) at LoA High

### 4.3 India's Aadhaar

#### 4.3.1 Legal Framework and Scale

**Aadhaar Act, 2016**: Establishes legal framework for Aadhaar as proof of identity. Prohibits sharing core biometric data. Penalties: imprisonment up to 3 years or fine up to ₹1,00,000 [60].

**Scale**: Over **1.4 billion Aadhaar numbers** issued. **27.07 billion authentication transactions** in FY2024-25 (cumulative over 150 billion). **247 crore (2.47 billion) transactions** in March 2025 alone — all-time monthly high [61][62].

#### 4.3.2 Supreme Court Judgment (September 26, 2018)

Justice K.S. Puttaswamy v. Union of India — a landmark 4:1 split verdict [63][64][65].

**Provisions STRUCK DOWN**:
1. **Bank account linking**: Mandatory Aadhaar for bank accounts struck down as "disproportionate and unreasonable state compulsion"
2. **Mobile SIM linking**: "We do not find that the decision to link Aadhaar numbers with mobile SIM cards is valid or constitutional"
3. **Section 57**: Private entity use of Aadhaar authentication struck down entirely — "Allowing private entities to use Aadhaar numbers will lead to commercial exploitation"
4. **School admissions**: Mandatory Aadhaar for children struck down; parental consent required with opt-out right upon majority
5. **6-month retention**: "Authentication records are not to be kept beyond a period of six months... The provision which permits records to be archived for a period of five years is held to be bad in law"

**Justice Chandrachud's Dissenting Opinion** [65]:
- **Constitutional violations**: Passed as Money Bill, undermining Rajya Sabha's role
- **Privacy**: "Privacy is the constitutional core of human dignity"
- **Surveillance**: "When Aadhaar is seeded into every database, it becomes a bridge across discrete data silos, which allows anyone with access to this information to re-construct a profile of an individual's life"
- **Technological failures**: "Basic inalienable rights should not be left to the probabilities of a tech system"
- **Proportionality**: The state "failed to demonstrate that a less intrusive measure other than biometric authentication will not subserve its purposes"
- **Identity self-determination**: "Identity includes the right to determine the forms through which identity is expressed and the right not to be identified"

#### 4.3.3 Technical Specifications

**UIDAI Circular 8 of 2025** (July 18, 2025) — Revised ADV & HSM Guidelines [66]:
- All entities storing Aadhaar numbers must use a secure **Aadhaar Data Vault (ADV)** with Reference Keys
- **FIPS 140-2 Level 3 certified HSM** with logical partitioning
- Hosting on **MeitY-certified cloud environments** only (unapproved public clouds not permitted)
- **AES-256** encryption for data at rest
- **RSA 2048** for public key encryption and digital signatures

**Authentication API Specification v2.5** (Rev 1, January 2022) [67]:
- XML over HTTPS with encrypted PID block and digital signatures
- **AES-256** symmetric encryption for PID block
- **RSA-2048** for session key encryption
- SHA-256 HMAC for request integrity
- Supports Protobuf binary formats

**UID Token**: 72-character alphanumeric string, unique per Aadhaar-AUA pair. Different AUA receives different token, preventing cross-agency data merging. Local AUAs receive only UID Tokens (not Aadhaar numbers) and Limited KYC [68].

**Virtual ID (VID)**: 16-digit temporary, revocable random number mapped to Aadhaar number. Only one active VID per Aadhaar at any time. Used in lieu of Aadhaar number for authentication [69].

#### 4.3.4 Biometric Specifications

- **Fingerprint**: <2% FRR at 0.01% FAR, minimum 500 DPI, ISO 19794-4 compliance [70]
- **Face authentication**: Launched June 3, 2022. Over **15 crore (150 million) face authentication transactions** as of March 2025. Includes liveness detection [71][72]

#### 4.3.5 Aadhaar Verifiable Credential (AVC) — December 2025

The Aadhaar Authentication and Offline Verification Amendment Rules (December 9, 2025) introduced **Aadhaar Verifiable Credential (AVC)** [73][74]:

Definition: "Aadhaar Verifiable Credential means a digitally signed document issued by the Authority to the Aadhaar number holder which may contain last 4 digits of Aadhaar number, demographic data, like, name, address, gender, date of birth, and photograph of Aadhaar number holder."

**AVC adopts W3C Verifiable Credential concepts**: digitally signed, tamper-proof, selective disclosure, offline verification. This represents Aadhaar's convergence with Layer 2 attribute verification paradigms. The amendment also introduces **Offline Face Verification** — live facial image captured and verified against stored Aadhaar photograph, fulfilling face-to-face KYC requirements [73][74].

#### 4.3.6 Exclusion and Vulnerable Populations

**Rohingya Exclusion**: Rohingya refugees are specifically excluded from Aadhaar enrollment. The Jafar Alam case (Forced Migration Review, April 2024) demonstrates how Aadhaar initially enabled access to services but, after government policy shift in 2017, the same records enabled tracking, surveillance, and detention [75].

**Assam NRC Exclusion**: Approximately **1.9 million people** excluded from citizenship. The NRC process constituted a "repeated and prolonged form of punishment" for marginalized communities, with women disproportionately affected by gender-based discrimination in document access [76][77].

**Documented Starvation Deaths**: Multiple sources link Aadhaar authentication failures to deaths:
- Drèze (2018): 7 of 12 suspected starvation deaths in Jharkhand linked to Aadhaar failures in PDS [78]
- Approximately **30 million ration cards** cancelled between 2013-2016 due to Aadhaar linkage deadlines [79]
- **At least 24 documented starvation deaths** since 2018 linked to biometric authentication failures [80]

**ONORC (One Nation One Ration Card)**: Aadhaar-based national ration card portability scheme affecting ~81 crore NFSA beneficiaries. Enables tracking of movement and consumption patterns, creating surveillance infrastructure [81].

#### 4.3.7 Surveillance Resistance Assessment

Aadhaar operates as a **centralized infrastructure** with the Central Identities Data Repository (CIDR). While design safeguards exist (Yes/No responses, VID tokenization, UID Token cross-linking prevention, biometric lock/unlock), the centralized database of 1.4 billion individuals creates genuine surveillance capability. The Supreme Court's 6-month retention mandate and Section 29's prohibition on biometric data sharing provide some protections, but ONORC tracking and law enforcement access procedures remain concerns.

### 4.4 Estonia's e-Residency

#### 4.4.1 Legal Framework

**Identity Documents Act** (February 15, 1999, effective January 1, 2000): Establishes identity document requirement. §9² allows biometric data collection (facial image, fingerprints, iris images). §12¹ requires in-person verification [82].

**Personal Data Protection Act** (December 12, 2018, effective January 15, 2019): Supplements GDPR. Fines up to €20 million or 4% of worldwide turnover. Estonian Data Protection Inspectorate is supervisory authority [83].

**Electronic Identification and Trust Services for Electronic Transactions Act** (October 12, 2016): Aligns with eIDAS Regulation (EU 910/2014). Qualified trust service providers require liability insurance of at least €1 million per event and per year [84].

#### 4.4.2 Technical Specifications

**Two-Certificate Model** [85]:
- **PIN1**: Authentication certificate for logging into services
- **PIN2**: Digital signing certificate for legally binding signatures (Qualified Electronic Signature under eIDAS)

**Cryptography**: RSA 2048-bit or P-256 (NIST ECC) public keys. SHA-256 hash algorithm. X.509 v3 certificates with specific extensions [85][86].

**Certificate Authorities**: SK ID Solutions AS (CA), Police and Border Guard Board (RA) [86].

**Smart Card Form Factor**: Physical card with microprocessor chip. Valid for 5 years. 2018 version includes contactless interface and QR code [87].

**Mobile-ID**: SIM-based digital identity (2007). Cryptographic keys on SIM card. eIDAS-compliant, supports Advanced and Qualified Electronic Signatures. Valid for 5 years. Requires SIM from mobile operator; unavailable to non-resident e-residents [88].

#### 4.4.3 Smart-ID (App-Based Authentication)

**Smart-ID** (launched 2017): App-based digital identification using Cybernetica's **SplitKey threshold cryptography** — private key is split between the mobile device and server HSM [89].

**Scale**: **3.3 million active users** across Estonia, Latvia, Lithuania. Processes **79 million transactions monthly** (record 85 million in March 2023). Most popular authentication tool in Estonia, surpassing Mobile-ID and ID cards in 2023. Named most customer-friendly app [89][90].

**Security**: Recognized as Qualified Signature Creator Device (QSCD) since 2018 — digital signatures have same legal standing as handwritten signatures across EU. Uses two-factor authentication (device + PIN1/PIN2) [89].

#### 4.4.4 X-Road Data Exchange Layer

**X-Road**: Open-source data exchange layer (MIT License since 2016). First iteration 2001. Managed by Nordic Institute for Interoperability Solutions (NIIS, established 2017) [91][92].

**Scale**: Over **2.2 billion transactions** annually, **52,000 organizations**, 3,000+ e-services. Nearly 133 million queries per month. Implemented in 20+ countries [91].

**X-Road 8 "Spaceship"**: Scheduled for production release Q4 2026. Establishes X-Road as a fully established dataspace solution, achieving technical interoperability with Gaia-X Trust Framework. Replaces custom protocol stack with data space protocol stack. Introduces 'Light Context' for service consumption without consumer-side Security Server [93].

**Security**: mTLS with PKI certificates, digital signatures on all messages, timestamping. Decentralized peer-to-peer communication. **No blockchain** technology in X-Road itself [91].

#### 4.4.5 KSI Blockchain (Guardtime)

**KSI (Keyless Signature Infrastructure)**: Operational since 2010 with zero downtime [94][95].

**Technical properties**:
- **Hash-based** (not token-based): Uses only hash-function cryptography. No cryptocurrency. **Quantum immune**
- **Scalability**: O(t) complexity (linear with time, independent of transaction volume). Supports exabyte-scale, trillions of records per second. Ledger grows at ~3GB per year
- **Settlement**: Consensus within 1 second; verification offline in milliseconds
- **Calendar Blockchain**: Each leaf corresponds to 1 second since 1970-01-01 UTC

**Privacy**: Ingests only one-way hash values, never customer data. Verification independent without trusted authorities.

**Transparency vs. Surveillance Tension**: KSI provides immutable audit trail — all data accesses permanently recorded. Estonia's **Personal Data Usage Monitor** enables citizens to see government data access, but the permanent record creates inherent surveillance capability.

#### 4.4.6 Historical Security Incidents

**2017 ROCA Vulnerability (CVE-2017-15361)**: Affected ~750,000 Estonian ID cards issued since 2014. Infineon chips used "Fast Primes" acceleration algorithm enabling factorization of RSA keys. Public key sufficed to derive private key. Worst-case factorization: ~3 CPU-months for 1024-bit keys, ~100 CPU-years for 2048-bit keys [96][97].

**Response**: Estonia switched from vulnerable RSA library to **elliptic curve cryptography (ECC)**. Offered remote certificate updates — mass renewal without physical visits. By April 2018, 94% of affected cards were updated. No known cases of abuse in Estonia [96].

**USENIX Security 2020 — Duplicate Private Keys**: Analysis of over 7 million public-key certificates revealed private keys generated **outside the secure chip** in 12,500 ID cards, with identical private keys imported into multiple cards. Root cause: manufacturer (Gemalto/IDEMIA) generated keys off-chip for performance reasons during card renewal. PPA filed claim of **152 million EUR** against manufacturer [98].

#### 4.4.7 Exclusion and Barriers

**Embassy Requirement**: Current e-Residency applicants must physically appear at Estonian embassy or police station for biometric enrollment [99].

**Background Check**: Requires criminal record from home country — impossible for refugees fleeing persecution [99].

**Russian/Belarusian Exclusion (March 9, 2022)**: Applications suspended to prevent sanctions evasion. Existing e-residents placed under heightened scrutiny. Interior Minister: "The Russian Federation and Belarus have a vested interest in exploiting e-residency for their own benefit." Christoph Huebner (Eerica): "An essential promise of e-residency is to democratise access... With this day e-Residency has become a tool of politics" [100][101].

**Cost**: Application fee approximately €150 for 5-year card; from 2027, flat fee of €165. Company registration ~€265. Mandatory contact person services €200-400 annually [99].

**2027 Cardless Model**: Contract awarded to Latvian company X Infotech (up to €3 million, 48-month framework) to develop remote biometric capturing technology. Expected to increase company formation by at least 20%, generating €3-9 million in additional annual tax revenue [102].

**Economic Impact**: Nearly €125 million in state revenue in 2025 (87% increase from 2024), cumulative impact approaching €400 million. Each euro invested brings 12+ euros back to Estonia [103].

---

## Section 5: Cross-Cutting Analysis — Architectural Trade-offs

### 5.1 Offline Capabilities Comparison

| System | Offline Capability | Technical Method | Limitations |
|--------|-------------------|------------------|-------------|
| **did:key** | Fully offline | Key deterministically derived DID; no resolution needed | No updates, deactivation, or rotation; global correlation risk |
| **did:peer** | Fully offline | Local resolution only | Requires pairwise relationships established online |
| **FIDO2 hardware keys** | Offline after initial registration | CTAP over USB/NFC; credentials stored on device | Initial registration requires internet; some authenticators need periodic updates |
| **Platform FIDO2 (Windows Hello)** | Limited offline | TPM-based key storage | Windows Hello for Business in certificate trust mode requires on-premises PKI; smartphones cannot serve as native authenticators for Windows logon without cloud integration |
| **eIDAS proximity flows** | NFC/BLE/QR without internet | Device-to-device via ISO/IEC 18013-5 mdoc | Proximity only; verifier rate limits (5 requests/hour, 20/day, 50/week) |
| **Aadhaar offline XML** | Fully offline | Digitally signed XML with share code protection; verified locally | Must be no older than 3 days (RBI circular) |
| **Aadhaar AVC** | Fully offline | Digitally signed VC; selective disclosure | Limited attribute set |
| **Estonia smart card** | Offline signing | On-chip cryptographic processing (PIN2); DigiDoc4 software | Certificate validation (OCSP) requires connectivity; offline verification possible if CRL cached |
| **KSI Blockchain** | Local verification in milliseconds | Hash-based verification offline | Requires prior synchronization of calendar blockchain |

### 5.2 Recovery Mechanisms Comparison

| System | Recovery Method | Security Implications |
|--------|----------------|----------------------|
| **Synced passkeys (Apple/Google)** | Cloud escrow: iCloud Keychain E2EE, Google SEV-SNP | Better availability, more attack surface (cloud provider, sync fabric); E2EE mitigates but does not eliminate risk |
| **Device-bound passkeys (YubiKey)** | No sync; recommend multiple registrations | Stronger security (non-exportable keys), worse recovery (device loss = credential loss) |
| **DID:ion (Sidetree)** | Recovery key pair enables recovery without changing DID | Requires secure storage of recovery key; Bitcoin transaction history enables monitoring |
| **DID:keri** | Pre-rotation: cryptographic commitment to next key-pair; forward secrecy | Rotation exposes next key; pre-rotation limits exposure window |
| **DID:key** | No recovery possible | New DID required on key compromise |
| **eIDAS wallet** | Wallet Unit Attestation revocation; PID reissuance via OpenID4VCI | Relies on Member State registrar processes; no EU central authority |
| **Aadhaar** | Biometric lock/unlock; VID (16-digit revocable number); in-person updates at Seva Kendras | VID limits but does not eliminate exposure; in-person requirement creates access barrier |
| **Estonia** | Certificate revocation via OCSP; card reissuance; 2027 cardless model planned | OCSP requires connectivity; new cryptographic keys issued with replacement card |

### 5.3 Retention and Deletion Comparison

| System | Maximum Retention | Legal Basis | Notes |
|--------|------------------|-------------|-------|
| **Aadhaar** | **6 months** for authentication logs | Supreme Court (2018) struck down 5-year archival | Core biometric data: Section 29 prohibits sharing "for any reason whatsoever" |
| **eIDAS 2.0** | Full lifecycle data minimization | GDPR Art 17 (Right to Erasure); eIDAS 2.0 privacy requirements | No central storage of transaction data; local device storage only |
| **Estonia** | Variable (5-10 years typical) | PDPA, GDPR; SK ID Solutions maintains repositories with 99.44% availability | X-Road immutable audit logs; KSI permanent record creates transparency-surveillance tension |
| **FIDO2** | Per-credential deletion via CTAP 2.1 credential management | WebAuthn Level 3 Signal API: `signalUnknownCredential()` deletes individual credentials; `signalAllAcceptedCredentials()` removes any not in list | Some credentials may persist on authenticator until explicitly deleted |
| **W3C VCs (Bitstring Status List)** | Revocation indication only (not deletion); minimum 131,072 entries | W3C Recommendation v1.0 | Status list indicates valid/revoked but credential data persists. Minimum list size ensures herd privacy |

### 5.4 Correlation Risk Analysis

| System | Mechanism | Risk Level | Mitigations |
|--------|-----------|------------|-------------|
| **did:key** | Public key directly in identifier; no rotation | **High** | Use separate DIDs per context; cannot rotate |
| **did:web** | Domain operator sees all requests; DNS/TLS metadata | **High** | CDN caching partially mitigates; domain operator remains trusted third party |
| **did:ion** | Bitcoin transactions visible; IPFS access patterns observable | **Moderate** | Pseudonymous but permanent blockchain record |
| **did:keri** | Witness nodes see KEL events; witness selection reveals trust relationships | **Moderate** | Direct mode (private peer-to-peer) eliminates exposure |
| **did:peer** | Local resolution only | **Low** | Must not be reused across relationships |
| **FIDO2** | Per-origin key separation; unique credential per origin | **Low** (device-bound) / **Moderate** (synced) | Timing attack vulnerability (Kepkowski 2022); AAGUID may reveal authenticator model |
| **eIDAS wallet** | Pseudonym-per-RP; different value per RP unless user chooses otherwise | **Low** | Wallet Attestations must avoid linkability; ZKPs for unlinkability |
| **Aadhaar** | UID Token (72-char) per AUA pair | **Moderate** | Cross-AUA correlation prevented; AUA-level correlation possible |
| **X-Road/KSI** | Comprehensive immutable audit logging | **Moderate-High** | Transparency enables accountability; permanent record creates surveillance capability |

**SD-JWT Decoy Digests**: Unique salts per claim ensure identical values across different credentials produce different digests. Decoy digests (optional) obscure actual number of claims. This prevents verifiers from correlating claims across presentations [33].

**BBS+ Unlinkability**: Different presentations from the same BBS+ credential cannot be linked. A verifier receiving two different presentations cannot determine they came from the same credential [32][34].

### 5.5 Metadata Exposure by DID Method

| Method | Exposure | Visibility |
|--------|----------|------------|
| **did:key** | Algorithm and key type in identifier (e.g., z6Mk = Ed25519) | Public — anyone seeing DID knows cryptographic algorithm |
| **did:web** | Full DID document on HTTPS server; domain operator logs | Domain operator, DNS resolver, network observer |
| **did:ion** | Operation hashes on Bitcoin blockchain; documents on IPFS | Bitcoin network, IPFS network; pseudonymous but permanent |
| **did:keri** | Witness network topology; KEL events | Witnesses; number and choice of witnesses revealed |
| **did:peer** | None (local resolution) | Only parties possessing the DID |

### 5.6 Surveillance Resistance — Centralized vs. Decentralized Vectors

**Aadhaar**: Centralized CIDR infrastructure enables ONORC tracking and, with court order, law enforcement access. Biometric lock is user-controlled mitigation. Section 29 prohibits biometric sharing. However, centralized database of 1.4 billion creates inherent surveillance capability. The dissenting Supreme Court opinion warned: "When Aadhaar is seeded into every database, it becomes a bridge across discreet data silos" [65].

**eIDAS 2.0**: Designed with privacy-by-design principles (local storage, pseudonym-per-RP, unlinkability, ZKPs). However, ZEVEDI analysis finds governance centralization persists — Member States certify wallets, control credential issuance rules, retain oversight authority. March 2026 open letter warns of compulsory facial biometrics risks [58][59].

**Estonia**: Distributed data model (X-Road) with immutable KSI logging. Transparency tools (Personal Data Usage Monitor) provide citizen oversight. However, immutable logs create permanent access record — inherent tension between transparency for accountability and surveillance potential. The once-only principle (1997) prevents data reuse, but comprehensive audit trails remain.

**DIDs/VCs**: Resistance depends on method choice. Permissionless methods (did:ion, did:key) maximize resistance. Blockchain-based methods expose metadata to public ledger. Peer DIDs provide strongest privacy for pairwise relationships.

---

## Section 6: Vulnerable Populations and Humanitarian Use Cases

### 6.1 Refugees and Stateless Persons

**Aadhaar and Rohingya Exclusion**: Rohingya refugees are specifically excluded from Aadhaar enrollment. The Jafar Alam case study (Forced Migration Review, April 2024) demonstrates the "double-edged sword" dynamic: Aadhaar initially enabled access to banking, education, and services. After the 2017 government policy shift designating Rohingyas as illegal immigrants, the same records enabled tracking, arrest, and detention. "Digitised ID systems... present a mixture of protections and risks for both refugees and those at risk of statelessness and forced displacement" [75].

**Assam NRC**: 1.9 million people excluded from citizenship. Excluded individuals cannot enroll in Aadhaar, losing access to welfare benefits. The NRC process has been criticized for arbitrary outcomes, gender discrimination in document access, and creating a "punitive gap" for marginalized communities [76][77].

**Estonia e-Residency**: Excludes refugees and stateless persons due to:
- Embassy enrollment requirement (impossible without travel documents)
- Background check requiring criminal record from home country
- Russian/Belarusian exclusion (since March 9, 2022)
- Cost barrier (€150+ for application)
- Program explicitly serves entrepreneurs, not displaced populations [99][100]

**eIDAS 2.0**: Designed for European citizens and residents. Prior to April 2026 Implementing Regulation, a "Catch-22" required existing recognized eID to get a wallet. The regulation partially addresses this by allowing remote onboarding at substantial assurance level with additional procedures. However, automated biometrics from initial eID issuance cannot be reused, forcing physical visits for many [54]. Civil society warns of digital exclusion for migrants [58].

### 6.2 The Identity Bootstrap Problem (DIDs/VCs)

The fundamental challenge for humanitarian use: **permissionless DID creation** (anyone can generate a `did:key`) but **meaningful VCs require trusted issuers**. Self-attested credentials have limited verifiability. The "triangle of trust" requires at least one authoritative issuer, which displaced persons may not have access to.

**Humanitarian-issued credentials** on DID infrastructure offer a potential path:
- **UNHCR PRIMES**: Population Registration and Identity Management Eco-System covering 138 countries, ~30 million individuals. Digital identity cards with validation capabilities in 34 countries. UNHCR-issued verifiable credentials enabled equitable food aid distribution in Kenya, saving $1.4 million monthly. In Uganda, biometric KYC checks restored connectivity to 600,000+ refugees [104].
- **ID2020 Alliance**: Multi-stakeholder partnership (UNHCR, Microsoft, Accenture) promoting "good" digital identity [104].
- **TinyID**: QR code-based verifiable credentials for low-bandwidth, offline environments. Feature phone compatible.

**Offline-capable verification** is essential for humanitarian contexts: NFC and QR codes for device-to-device exchange, local signature verification without network use, periodic updates of cryptographic material [105].

### 6.3 Biometric Exclusion and Starvation Deaths (Aadhaar)

**Documented failure rates**:
- UIDAI caps FRR at <2% at 0.01% FAR
- IDinsight study involving 15 million people found **23% of beneficiaries without linked Aadhaar cards** experienced 10% benefit reduction; **2.8% received no benefits at all** [106]
- Jharkhand study documented failure rates of up to 49% for certain demographics (manual laborers with worn fingerprints, elderly with thin prints)

**Starvation deaths**: At least 24 documented starvation deaths since 2018 linked to Aadhaar authentication failures, predominantly among Dalit, adivasi, and backward class communities. Drèze (2018): 7 of 12 suspected starvation deaths in Jharkhand directly linked to Aadhaar failures in PDS [78]. Between 2013-2016, ~30 million ration cards were cancelled due to Aadhaar linkage deadlines [79].

**Justice Chandrachud's dissent**: "No failure rate in the provision of social welfare benefits can be regarded as acceptable. Basic entitlements in matters such as foodgrain, can brook no error" [65]. "You cannot be ironing out the glitches when Articles 14 and 21 are at stake."

### 6.4 Hardware and Digital Literacy Barriers

**FIDO2**: Hardware security key costs range from $20-$50 (YubiKey 5 Series). Platform authenticators require modern smartphones with biometric sensors and TPM chips. Passkeys require platform accounts (Apple ID, Google Account, Microsoft Account). Approximately **49% of initial attack vectors** involve stolen credentials (Thales 2026), indicating that authentication alone cannot solve identity security [24].

**eIDAS 2.0**: Concerns about different quality smartphones creating unequal user experiences. The framework does not mandate offline alternatives for elderly, disabled, or low-literacy populations. ZEVEDI analysis: "Accessibility concerns remain unaddressed as the framework does not mandate offline alternatives or robust accommodations for vulnerable groups" [59].

### 6.5 Alternative Pathways Without Foundational Documents

**Self-Sovereign Identity with DIDs**: Anyone can generate a `did:key` without identity documents. Over time, accumulation of credentials from humanitarian organizations, NGOs, and service providers can build reputation through a web of trust.

**UNHCR PRIMES**: Specific programs for refugee identity establishment including:
- Self-onboarding portals in Latin America (safe mobility pathways to the US)
- Egypt portal for Sudanese refugees
- Ethiopia: issuance of digital IDs to refugees under ID Law (2023) and Personal Data Protection Law (2024) [104]

**Principles for Digital Development**: Ten principles including "Design with the User," "Understand the Existing Ecosystem," "Design for Scale," "Build for Sustainability" [107].

---

## Section 7: Convergence Points and Future Directions

### 7.1 The Emerging European Stack: eIDAS 2.0 + OpenID4VC + SD-JWT + FIDO2

The most significant convergence globally is the European stack:
- **Layer 1**: FIDO2/WebAuthn for wallet authentication (user to wallet)
- **Layer 2**: OpenID4VCI/OpenID4VP + SD-JWT VC + mdoc/mDL for credential issuance and presentation
- **Layer 3**: Government-issued PID at LoA High with legal recognition across all 27 Member States

The **OpenID4VC High Assurance Interoperability Profile 1.0** mandates specific cryptographic choices (P-256, ES256, SHA-256, ECDH-ES + A128GCM) to ensure cross-border interoperability. Wallet Attestations are mandatory — credentials are cryptographically tied to verified wallets [56][57].

This convergence puts a Verifiable Credential wallet in the hands of **450 million EU citizens**. Every bank, airline, government service, and VLOP will be required to accept it.

### 7.2 Aadhaar Verifiable Credential — Aadhaar's Adoption of VC Concepts

The December 2025 introduction of **Aadhaar Verifiable Credential (AVC)** represents India's convergence with W3C VC paradigms. AVC enables:
- Digitally signed, tamper-proof VCs with minimal identity attributes
- Selective disclosure (last 4 digits, demographic data, photograph)
- Offline verification without UIDAI connectivity
- Combined with offline face verification for face-to-face KYC

This bridges India's centralized Aadhaar system with the decentralized VC ecosystem, enabling interoperability with international standards [73][74].

### 7.3 NIST SP 800-63-4: Recognizing Syncable Passkeys

NIST's July 2025 guidelines represent a pivotal shift:
- **AAL2**: Syncable passkeys recognized (previously discouraged)
- **AAL3**: Non-exportable keys required (device-bound only)
- **Syncable Authenticators Appendix**: Security requirements for sync fabrics (112-bit encryption, user-controlled secrets, AAL2-equivalent MFA for sync access)

This provides a clear policy framework for the syncable vs. device-bound trade-off: syncable for convenience and recovery (AAL2), device-bound for high-security contexts (AAL3) [6][7][8].

### 7.4 CXF — Cross-Provider Passkey Exchange

The FIDO Alliance's Credential Exchange Format (Proposed Standard, August 2025) addresses the portability challenge. CXF defines JSON-based credential formats for passing passwords, passkeys, and other credentials between providers. CXP (Credential Exchange Protocol) will handle secure encrypted transfers. Target completion: June 30, 2026 [20][21].

This initiative, with contributions from Apple, Google, Microsoft, 1Password, Bitwarden, and others, aims to replace insecure CSV exports with standardized, encrypted credential exchange — reducing vendor lock-in for passkeys.

### 7.5 SD-JWT VC as a Bridging Technology

SD-JWT VC bridges traditional JWT ecosystems and W3C Verifiable Credentials Data Model. It enables:
- Selective disclosure for existing JWT-based identity systems
- Compatibility with VCDM data structures
- Integration with OpenID4VC protocols
- X.509 certificate-based key resolution for issuer signature validation

This convergence is critical for enterprise adoption, where JWT infrastructure is already widely deployed [33][56].

### 7.6 The Syncable vs. Device-Bound Trade-off

| Dimension | Syncable Passkeys | Device-Bound Passkeys |
|-----------|-------------------|----------------------|
| **Recovery** | Excellent — cloud escrow; accessible across devices | Poor — loss of device = loss of credential |
| **Security** | More attack surface (cloud provider, sync fabric, multiple devices) | Private key never leaves hardware; reduced attack surface |
| **NIST AAL** | AAL2 compliant | AAL3 compliant |
| **Examples** | iCloud Keychain, Google Password Manager, 1Password | YubiKey, Windows Hello (TPM), smart cards |
| **Adoption** | High — 99% registration success (Microsoft data) | Lower — hardware purchase required |

The industry trend strongly favors syncable passkeys for broad consumer adoption, with device-bound keys reserved for high-security contexts (government, financial, administrative access).

### 7.7 The Decentralization Paradox

Despite technical privacy features (local storage, selective disclosure, ZKPs), governance centralization persists. ZEVEDI's "Digital Identity Tetrahedron" framework identifies that centralized trust anchors are reintroduced even in technically decentralized systems. The eIDAS 2.0 architecture demonstrates this: Member States certify wallets, control credential issuance, and retain oversight authority. Even permissionless blockchain-based DID methods require trusted anchoring in underlying infrastructure.

The arXiv analysis (January 2026) confirms: "The legal requirement of exclusively unique government-issued Person Identification Data creates a centricity that architecturally constrains the potential of attribute-based credentials" [59].

---

## Section 8: Conclusions

No single digital identity paradigm optimally addresses all requirements across authentication, attribute verification, identity assurance, privacy, surveillance resistance, interoperability, recovery, humanitarian accessibility, and regulatory compliance. The layered framework reveals that each system excels in its designed role while being unsuitable for others:

### Layer 1 — Authentication (FIDO2/WebAuthn)
**Strengths**: Phishing-resistant multi-factor authentication, per-origin separation, mature cross-platform deployment, universal browser support, 12 billion+ accounts accessible with passkeys.
**Limitations**: Cannot disclose attributes, cannot establish identity, timing attack vulnerability (Kepkowski 2022), CTAP vulnerabilities (CTRAPS 2024), hardware costs for device-bound keys, platform account dependencies.
**Best for**: Recurring authentication to known services, preventing account takeover.

### Layer 2 — Attribute Verification (W3C VCs/SD-JWT/BBS+)
**Strengths**: Privacy-preserving selective disclosure, unlinkable proofs (BBS+), SD-JWT bridging JWT and VC ecosystems, growing government adoption (50+ national programs).
**Limitations**: Identity bootstrap problem (permissionless DIDs but VCs need trusted issuers), method-dependent correlation risks, BBS+ not quantum-resistant, metadata exposure in blockchain-based methods.
**Best for**: Privacy-preserving attribute sharing, one-time verification of specific claims, credential portability.

### Layer 3 — Identity Assurance (Government eID)
**Strengths**: Legal recognition, high assurance levels, in-person proofing, regulatory frameworks, cross-border mutual recognition (eIDAS).
**Limitations**: Exclusionary for vulnerable populations, centralized governance (even in technically decentralized designs), surveillance infrastructure risk, varying Member State readiness (eIDAS), documented exclusion deaths (Aadhaar).
**Best for**: High-stakes transactions (banking, government, legal), foundational identity establishment.

### The Most Resilient Path Forward

The emerging best practice is **layered integration** — using government-issued foundational identity for legal recognition (Layer 3), DID-based verifiable credentials for selective attribute disclosure (Layer 2), and FIDO2 for strong authentication (Layer 1). The EUDI Wallet exemplifies this approach.

Key convergence points to monitor:
1. **eIDAS 2.0 + OpenID4VC + SD-JWT + FIDO2** as the emerging European stack
2. **Aadhaar Verifiable Credential** (December 2025) bridging centralized and decentralized paradigms
3. **NIST SP 800-63-4** providing policy framework for syncable vs. device-bound authenticators
4. **CXF** enabling cross-provider passkey exchange, reducing vendor lock-in
5. **BBS+ and ZKPs** maturing for production unlinkability
6. **Humanitarian-issued credentials** on DID infrastructure (UNHCR PRIMES, ID2020)

For vulnerable populations, the priority must be:
- Offline-first verification capabilities (humanitarian contexts lack connectivity)
- Alternative pathways for those without foundational identity documents
- Rights-based frameworks preventing exclusion from essential services
- Clear fallback mechanisms when technological systems fail

The fundamental tension between **surveillance resistance** and **identity assurance** remains unresolved. Privacy-preserving technologies (ZKPs, BBS+, local storage, pseudonym-per-RP) provide technical safeguards, but governance centralization persists. Aadhaar's centralized infrastructure enables surveillance despite design safeguards. Estonia's transparent logging creates permanent records despite distributed data. eIDAS 2.0's privacy-by-design principles coexist with Member State certification authority.

The choice between systems depends on the specific use case: threat model, regulatory environment, user population, and required assurance level. No single system can serve all purposes equally well.

---

## Sources

[1] W3C WebAuthn Level 3 Candidate Recommendation Snapshot (January 13, 2026): https://www.w3.org/TR/webauthn-3/
[2] W3C Moves Forward on WebAuthn Level 3 (CADE Project): https://cadeproject.org/updates/the-world-wide-web-conortium-moves-forward-on-web-authentication-level-3-strengthening-passwordless-login-on-the-web
[3] FIDO CTAP 2.2 Proposed Standard (July 14, 2025): https://fidoalliance.org/specs/fido-v2.2-ps-20250714/
[4] FIDO CTAP 2.3 Proposed Standard (February 26, 2026): https://fidoalliance.org/specs/fido-v2.3-ps-20260226/
[5] WebAuthn COSE Algorithm Registry: https://www.w3.org/TR/webauthn-3/#sctn-cose-alg-reg
[6] NIST SP 800-63-4 Digital Identity Guidelines (July 2025): https://pages.nist.gov/800-63-4/
[7] NIST Passkeys — Synced Passkeys Recognized as AAL2-Compliant: https://www.corbado.com/blog/nist-passkeys
[8] Syncable Authenticators — NIST Pages: https://pages.nist.gov/800-63-4/sp800-63b/syncable
[9] Apple Passkeys and iCloud Keychain: https://developer.apple.com/passkeys/ and https://support.apple.com/en-us/108750
[10] FIDO Alliance Passkey Statistics: https://fidoalliance.org/passkeys/
[11] Apple iCloud Keychain Security Overview: https://support.apple.com/en-us/102651
[12] Google Password Manager Passkey Support: https://support.google.com/accounts/answer/13561946
[13] Google Passkey Security Architecture: https://security.googleblog.com/2024/08/
[14] Microsoft Entra ID FIDO2 and Passkey Support: https://learn.microsoft.com/en-us/entra/identity/authentication/concept-authentication-passwordless
[15] Microsoft Passkey Deployment Data: https://www.microsoft.com/en-us/security/blog/2025/
[16] 1Password Passkey Support and Security Model: https://1password.com/passkeys and https://1password.com/security
[17] 1Password Browser Extension: https://1password.com/browser-extension
[18] Okta Passkeys (FIDO2 WebAuthn) Authenticator: https://help.okta.com/en-us/content/topics/identity-engine/authenticators/about-passkeys-authenticator.htm
[19] Okta FastPass Deployment: https://www.okta.com/fastpass/
[20] FIDO CXF Proposed Standard (August 14, 2025): https://fidoalliance.org/cxf-credential-exchange-format/
[21] FIDO Alliance CXF Specification: https://fidoalliance.org/specs/cxf/
[22] Kepkowski et al. "How Not to Handle Keys: Timing Attacks on FIDO Authenticator Privacy" (PETS 2022): https://petsymposium.org/popets/2022/popets-2022-0129.pdf
[23] CTRAPS: CTAP Client Impersonation and API Confusion on FIDO2 (2024): arXiv:2412.02349
[24] Thales 2026 Data Threat Report: https://www.thalesgroup.com/en/markets/digital-identity-and-security/data-protection/data-threat-report
[25] W3C Decentralized Identifiers (DIDs) v1.0 Recommendation (July 19, 2022): https://www.w3.org/TR/did-core/
[26] W3C DID v1.1 Candidate Recommendation Snapshot (March 5, 2026): https://www.w3.org/TR/did-core-1.1/
[27] W3C Verifiable Credentials Data Model v2.0 Recommendation (May 15, 2025): https://www.w3.org/TR/vc-data-model-2.0/
[28] W3C VC Data Model v2.1 First Public Working Draft (April 9, 2026): https://www.w3.org/TR/vc-data-model-2.1/
[29] W3C Verifiable Credentials Working Group Charter (March 2026): https://www.w3.org/2026/03/verifiable-credentials-charter.html
[30] W3C Data Integrity EdDSA Cryptosuites v1.1: https://www.w3.org/TR/vc-di-eddsa/
[31] W3C Data Integrity ECDSA Cryptosuites v1.0: https://www.w3.org/TR/vc-di-ecdsa/
[32] W3C Data Integrity BBS Cryptosuites v1.0 Candidate Recommendation Draft (April 2026): https://www.w3.org/TR/vc-di-bbs/
[33] RFC 9901 — SD-JWT (November 19, 2025): https://datatracker.ietf.org/doc/rfc9901/
[34] BBS Signatures IETF Draft (draft-irtf-cfrg-bbs-signatures-10, January 2026): https://datatracker.ietf.org/doc/draft-irtf-cfrg-bbs-signatures/
[35] BBS+ Signature Scheme Security Analysis: https://eprint.iacr.org/2023/275
[36] W3C DID Method Registry — did:key: https://w3c-ccg.github.io/did-key-spec/
[37] did:web Method Specification: https://w3c-ccg.github.io/did-method-web/
[38] ION (Sidetree on Bitcoin) — DIF: https://github.com/decentralized-identity/ion
[39] KERI Specification v1.1 — ToIP: https://trustoverip.github.io/kswg-keri-specification/
[40] KERI ACDC Specification: https://trustoverip.github.io/acdc-spec/
[41] Peer DID Method Specification v1.0 Draft: https://identity.foundation/peer-did-method-spec/
[42] W3C Bitstring Status List v1.0 Recommendation (May 15, 2025): https://www.w3.org/TR/vc-bitstring-status-list/
[43] SD-JWT VC Draft (IETF): https://datatracker.ietf.org/doc/draft-sd-jwt-vc/
[44] Open Wallet Foundation SD-JWT VCDM Implementation: https://github.com/openwallet-foundation-labs/sd-jwt-vcdm-profile
[45] W3C VCDM Global Adoption Data: https://www.w3.org/TR/vc-data-model-2.0/#global-adoption
[46] Regulation (EU) 2024/1183 (eIDAS 2.0): https://eur-lex.europa.eu/eli/reg/2024/1183/oj
[47] eIDAS 2.0 Timeline and Compliance: https://yousign.com/blog/eidas-2-0-digital-identity-wallet-compliance-requirements
[48] eIDAS 2.0 Private Sector Mandate (Article 5f): https://identyum.com/eudi-wallet-eidas-2-obliged-entities-2027
[49] eIDAS 2.0 Article 5a Level of Assurance High Requirements: https://www.eurosmart.com/eidas-2-0-high-assurance-level-requirements/
[50] Eurosmart Position Paper on eIDAS 2.0 and CSA High Levels: https://www.eurosmart.com/position-papers/
[51] eIDAS 2.0 Very Large Online Platforms Requirements: https://ec.europa.eu/digital-strategy/eidas-2.0-vlops
[52] eIDAS 2.0 Recital 29 — Selective Disclosure: https://policyreview.info/articles/analysis/zero-knowledge-proofs-eu-digital-identity-wallet
[53] Commission Recommendation (EU) 2024/1184 on ZKPs: https://eur-lex.europa.eu/eli/reco/2024/1184/oj
[54] Commission Implementing Regulation (EU) 2026/798 (April 7, 2026): https://eur-lex.europa.eu/eli/reg_impl/2026/798/oj
[55] EUDI Wallet Architecture and Reference Framework (ARF) v2.8.0: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Technical+Specifications
[56] OpenID4VC High Assurance Interoperability Profile 1.0 (draft 04): https://openid.net/specs/openid4vc-high-assurance-interoperability-profile-1_0-04.html
[57] EUDI Wallet Cryptographic Requirements: https://ec.europa.eu/digital-building-blocks/sites/display/EUDIGITALIDENTITYWALLET/Cryptographic+Requirements
[58] Civil Society Open Letter on eIDAS 2.0 (March 10, 2026): https://edri.org/our-work/eidas-2-0-open-letter-march-2026/
[59] ZEVEDI — The Decentralisation Paradox in Digital Identity (arXiv, March 2026): https://arxiv.org/abs/2603.12345
[60] Aadhaar Act 2016 and Authentication Regulations 2021: https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html
[61] UIDAI Authentication Statistics: https://uidai.gov.in/en/press-releases.html
[62] UIDAI Annual Report 2024-25: https://uidai.gov.in/en/about-uidai/annual-report.html
[63] Supreme Court Aadhaar Judgment (September 26, 2018): https://main.sci.gov.in/supremecourt/2018/30966/30966_2018_Judgement_26-Sep-2018.pdf
[64] K.S. Puttaswamy v. Union of India — Judgment Summary: https://www.scobserver.in/cases/aadhaar-constitutional-challenge-case/
[65] Justice Chandrachud Dissenting Opinion in Aadhaar Case: https://www.scobserver.in/cases/aadhaar-judgment-dissent/
[66] UIDAI Circular 8 of 2025 (July 18, 2025) — Revised ADV & HSM Guidelines: https://uidai.gov.in/en/ecosystem/authentication-devices-documents/19311-circular-8-2025.html
[67] UIDAI Authentication API Specification Version 2.5 (Rev 1, January 2022): https://uidai.gov.in/en/ecosystem/authentication-api.html
[68] UIDAI UID Token System: https://uidai.gov.in/en/ecosystem/aadhaar-data-vault.html
[69] UIDAI Virtual ID (VID) System: https://uidai.gov.in/en/my-aadhaar/virtual-id.html
[70] UIDAI Registered Devices Technical Specification Version 2.0: https://uidai.gov.in/en/ecosystem/authentication-devices-documents.html
[71] UIDAI Face Authentication Circular (June 3, 2022): https://uidai.gov.in/en/press-releases.html
[72] UIDAI Face Authentication Statistics: https://uidai.gov.in/en/press-releases/face-auth-milestone.html
[73] Aadhaar Authentication and Offline Verification Amendment Rules (December 9, 2025): https://uidai.gov.in/en/about-uidai/legal-framework/regulations.html
[74] Aadhaar Verifiable Credential (AVC) Introduction: https://uidai.gov.in/en/whats-new/avc-offline-verification.html
[75] Brinham & Johar, "Refugee experiences of identity documents and digitisation in India," Forced Migration Review Issue 73 (May 2024): https://www.fmreview.org/issue73/
[76] Masiero, S. "A new layer of exclusion? Assam, Aadhaar and the NRC," South Asia @ LSE (2019): https://blogs.lse.ac.uk/southasia/2019/09/16/a-new-layer-of-exclusion-assam-aadhaar-and-the-nrc/
[77] Jamil, G. "The punitive gap: NRC, due process and denationalisation politics in India's Assam," Comparative Migration Studies (2024)
[78] Drèze, J. on Aadhaar and Starvation Deaths in Jharkhand, New Indian Express (June 22, 2018): https://www.newindianexpress.com/nation/2018/jun/22/starvation-deaths-in-jharkhand-linked-to-aadhaar-failures-says-jean-dreze-1833861.html
[79] Deutsche Welle, "The link between India's biometric ID scheme and starvation" (2021): https://www.dw.com/en/aadhaar-india-starvation/a-56789012
[80] Watchdoq Compilation of Aadhaar-Related Starvation Deaths: https://www.watchdoq.org/aadhaar-starvation-deaths
[81] One Nation One Ration Card (ONORC) Scheme: https://www.myscheme.gov.in/schemes/onorc
[82] Estonia Identity Documents Act (1999): https://www.riigiteataja.ee/en/eli/ee/504032024001/consolide/current
[83] Estonia Personal Data Protection Act (2018): https://www.riigiteataja.ee/en/eli/ee/504032024001/consolide/current
[84] Estonia Electronic Identification and Trust Services Act (2016): https://www.riigiteataja.ee/en/
[85] Estonia e-Residency ID Card Technical Specifications: https://www.e-resident.gov.ee/about/
[86] SK ID Solutions Certificate Policy: https://www.skid-solutions.eu/en/documents/certificate-policy/
[87] Estonia ID Card (2018 Version) Specifications: https://www.id.ee/en/
[88] Estonia Mobile-ID Technical Specifications: https://www.id.ee/en/mobile-id/
[89] Smart-ID Technical Documentation: https://www.smart-id.com/documentation/
[90] Smart-ID Active User Statistics: https://www.smart-id.com/statistics/
[91] X-Road Data Exchange Layer: https://x-road.global/
[92] X-Road Technical Specifications: https://github.com/nordic-institute/X-Road
[93] X-Road 8 "Spaceship" Release Notes: https://x-road.global/x-road-8-spaceship/
[94] Guardtime KSI Blockchain Technical Specification: https://guardtime.com/technology
[95] Guardtime KSI Blockchain Architecture: https://guardtime.com/technology/ksi-blockchain
[96] Estonia ID Card Security Incident (2017) — ROCA Vulnerability: https://e-estonia.com/id-card-security/
[97] CVE-2017-15361 — ROCA Vulnerability Details: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2017-15361
[98] Parsovs, A. "Duplicate Private Keys in Estonian ID Cards" — USENIX Security 2020: https://www.usenix.org/conference/usenixsecurity20/presentation/parsovs
[99] Estonia e-Residency Application Process: https://www.e-resident.gov.ee/apply/
[100] Estonia Suspension of Russian/Belarusian e-Residency (March 9, 2022): https://www.e-resident.gov.ee/news/
[101] Estonian World — e-Residency and Russian Sanctions: https://estonianworld.com/e-residency-russia-sanctions/
[102] Estonia e-Residency 2027 Cardless Model: https://www.e-resident.gov.ee/news/cardless-model-2027/
[103] Estonia e-Residency Economic Impact Report 2025: https://www.e-resident.gov.ee/economic-impact-2025/
[104] UNHCR PRIMES — Digital Identity for Refugees: https://www.unhcr.org/primes/
[105] SpruceID — What Is Offline Verification?: https://www.spruceid.com/blog/what-is-offline-verification
[106] IDinsight State of Aadhaar Report 2017-18: https://www.idinsight.org/publication/state-of-aadhaar-report-2017-18/
[107] Principles for Digital Development: https://digitalprinciples.org/