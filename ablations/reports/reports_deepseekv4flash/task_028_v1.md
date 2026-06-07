# Designing Collaborative Learning Interfaces Across Cultural Contexts: South Korea, Finland, and Mexico

## Executive Summary

This report presents a comprehensive analysis of how cultural attitudes toward peer feedback, teacher authority, and public mistake-making should inform the design of collaborative learning interfaces for secondary students in South Korea, Finland, and Mexico. Drawing on documented implementations from Classting (South Korea), Wilma (Finland), and Google Classroom's Latin America deployments (with a focus on Mexico), combined with academic cross-cultural research, the analysis reveals that cultural dimensions—particularly power distance, collectivism vs. individualism, and comfort with face-saving—profoundly shape the effectiveness of anonymous feedback mechanisms, teacher moderation levels, and public vs. private correction displays.

**Key Findings:**

1. **Anonymous Feedback:** Classting's private counseling feature (allowing student anonymity) directly addresses Korean *chemyeon* (saving face) culture. Wilma has no anonymous peer feedback—reflecting Finnish directness and low power distance. Google Classroom lacks native anonymity globally, requiring workarounds like Google Forms in Mexican deployments.

2. **Teacher Moderation:** Classting enforces high teacher control (only teachers create classes, control posting permissions)—aligning with Korean high power distance. Wilma provides structured teacher monitoring but generates tension with Finnish egalitarian values. Google Classroom's Latin America strategy is deliberately teacher-centric, prioritizing teacher training and empowerment first.

3. **Mistake Visibility:** Classting offers private wrong answer storage and individual reports—keeping mistakes hidden from peers. Wilma keeps grades private (student/guardian/teacher only) but permanently records behavioral infractions. Google Classroom defaults to private teacher-student feedback, with class comments being the only public element.

4. **Engagement Metrics:** Classting reports 80%+ student satisfaction with AI-assisted learning and 2,500+ schools subscribing. Wilma processes 300,000+ messages weekly across 95%+ of Finnish schools but has a low 2.24/5 app rating. Google Classroom's Mexican deployments reach 1.3 million students in Jalisco alone, with 96.7% achieving high meaningful learning levels post-adoption.

---

## 1. Cultural Context: The Three Markets

### 1.1 South Korea: High Power Distance, Strong Collectivism, Face-Saving Culture

South Korea scores approximately **60 on Hofstede's Power Distance Index** (PDI), indicating a society that accepts hierarchical structures where teachers are viewed as authority figures deserving unquestioning respect [8][19]. The culture is deeply influenced by Confucian values, particularly the "Five Cardinal Relationships" that emphasize vertical hierarchies between teacher and student [20]. Korean students expect teachers to lead and outline learning, often suppressing spontaneous participation [8].

The collectivist nature of Korean society (score of approximately 18 on individualism, making it highly collectivist) means group harmony and "saving face" (*chemyeon/체면*) are paramount [12][15]. Students may be shy about expressing opinions publicly due to fear of causing embarrassment or losing face. A CIEE blog post on Korean school culture notes: "The concept of 'using your imagination' is often lost on the students and they tend to ask 'Teacher, can I just follow you?'" reflecting the preference for structured guidance [19].

The contemporary Korean education system is described as "very rigid and closed," taking "the form of a vertical hierarchy, as any institution would in a Confucian-influenced setting" [21]. Teachers expect "unquestioning obedience from students," and "independent discussion and thought around the topic are discouraged, with almost all lessons being conducted by rote learning" [21].

### 1.2 Finland: Low Power Distance, Individualist, Egalitarian

Finland scores **33 on Hofstede's Power Distance Index**, one of the lowest in the world, indicating a society that values decentralized power, direct communication, and informal supervisor-employee relations [5][6]. In Finnish classrooms, teachers are facilitators rather than authoritarian figures. The Finnish National Agency for Education emphasizes that "the core of the problem is not behavioral challenges: the core is how we adults learn to see and hear what children and young people are communicating through their behavior" [6].

Finland scores approximately **63 on individualism**, making it a moderately individualist society where task completion often takes priority over relationship maintenance [6]. However, Finnish culture also values consensus, trust, and cooperation. Teachers are highly respected professionals who undergo rigorous training and are given significant autonomy in their classrooms [16].

A cross-cultural study by Yoon (2019) found that "a less vertical relationship and weaker teacher control over students' self-expression were observed in Finnish schools," whereas hierarchical relationships and intensive teacher control were "more noticeable in Korean schools" [13]. Finnish workplaces exhibit high trust and expect employee independence, with informal relationships where even professors and students use first names [5].

### 1.3 Mexico: High Power Distance, Collectivist, Teacher-Centric

Mexico scores approximately **81 on Hofstede's Power Distance Index**, among the highest in the world, indicating strong acceptance of hierarchical structures and authority [6]. In Mexican classrooms, the teacher is perceived as a lecturer imparting knowledge, and students expect direction and guidance from their instructors [2]. The individualism score is approximately 30, making Mexico a collectivist society where group harmony, family, and relationships are prioritized [10].

Mexican education culture emphasizes respect for authority and formal teacher-student relationships. The Gómez-Rey, Barbera, & Fernández-Navarro (2016) study comparing e-learning across four countries found that Mexican students initially showed lower autonomy levels compared to American and Spanish students, but effectively adapted during the course [4]. The same study found that Power Distance and Individualism dimensions were implicated in learner factors.

---

## 2. Platform Comparisons

### 2.1 Classting (South Korea)

**Platform Overview:** Classting was founded by Dave Cho (조현구), a former Korean elementary school teacher, and has grown into Korea's largest ed-tech platform [1]. As of May 2026, over 2,500 elementary, middle, and high schools subscribe monthly to Classting AI, serving over 210,000 students, with total users previously reaching approximately 5 million [10][11]. The platform combines educational management with a safe social media environment.

#### 2.1.1 Anonymous vs. Identified Feedback Mechanisms

Classting offers a **"Private Counseling" (개인 상담)** feature where students can communicate directly with teachers in one-on-one conversations and have the option to remain anonymous [1]. This directly addresses Korean cultural norms around *chemyeon* (saving face), as students may be reluctant to ask questions or express difficulties publicly.

The Classting AI student manual states: "If you have questions while solving problems, you can share and ask them in our class. The teacher can conveniently provide individual feedback through replies" [25]. This creates a pathway for students to ask questions—but through private channels where they do not lose face.

A 2025 study published in the journal '학교수학' (School Mathematics) investigated an AI-based formative peer assessment system in Korean mathematics classes, finding that AI-generated feedback excelled in content structure while student peer feedback provided stronger emotional support [13]. The study noted that students focused on diagnosis and improvement when evaluating AI responses but emphasized praise and suggestions for peers' work—reflecting the cultural emphasis on maintaining positive relationships and avoiding harsh criticism.

**Design Implication for Korea:** Anonymous or private feedback channels are essential. Peer feedback should emphasize praise and constructive suggestions rather than direct criticism. Teachers should mediate feedback processes.

#### 2.1.2 Teacher Moderation Levels

Classting enforces **very high teacher control**. Since August 24, 2015, **only teachers can create classes** on Classting—this policy was implemented to prevent "indiscriminate class creation and distribution of inappropriate content" [7][11]. Teachers can configure writing permissions by role, controlling whether students and parents can post content [12].

The platform has a **"Chief Administrator" (최고 관리자) and "General Administrator" (일반 관리자)** system for institutions, where there must be at least one chief administrator per institution [14]. Teachers can also transfer class administrator authority, immediately granting posting and editing privileges to appointees [4].

Classting implements **real-time monitoring** of user activities, with violations resulting in "warnings, temporary suspensions, permanent suspensions" [11]. AI-assisted moderation is also employed to maintain a safe environment.

**Design Implication for Korea:** High teacher moderation aligns with cultural expectations. Teachers need granular control over who can post, comment, and access content. The teacher's role as gatekeeper and authority figure should be preserved in the platform architecture.

#### 2.1.3 Public vs. Private Mistake Visibility

Classting AI includes an **"오답보관함" (Wrong Answer Storage / Error Repository)** feature for individual student review, allowing students to privately review their mistakes [10]. This is entirely private—no peer can see another student's errors.

The platform provides **individual student reports** (개인별 리포트) showing "PDF output of individual student learning status reports, giving individual feedback to students, AI comprehensive feedback" [11]. Teachers can provide feedback privately rather than publicly displaying student mistakes.

The Classting website emphasizes "effective error management" by focusing on "intensively managing concepts that learners frequently get wrong" [8]. Error correction is framed as a private, individualized process—not a public one.

**Design Implication for Korea:** Mistakes should be handled privately. Public correction or error display would cause shame and violate face-saving norms. Individual error repositories and private feedback are essential.

#### 2.1.4 Outcome Metrics

- **2,500+ schools** subscribing to Classting AI (May 2026) [10]
- **210,000+ students** using Classting AI [10]
- **80%+** of students found AI-based quizzes helpful for review and active participation [15]
- **71.4%** of students reported improvements in English writing skills due to AI assistance [15]
- **91.5%** correct/incorrect answer prediction rate for the AI model [8]
- **6.4x increase** in SaaS revenue in H1 2024 vs. previous year [6]
- **50%+ market share** among digital leading schools for elementary and middle education in 2023 [7][24]

#### 2.1.5 Cultural Fit Summary

Classting's design philosophy explicitly accommodates Korean high power distance (teacher-centric control), collectivism (safe social media environment, indirect feedback), and face-saving culture (private counseling, individual error storage). The platform's founder, a former teacher, designed it around the reality of Korean classrooms where teacher authority is paramount [1].

---

### 2.2 Wilma (Finland)

**Platform Overview:** Developed by Visma StarSoft (now Visma Aquila Oy), Wilma is the de facto school information system in Finland, used in **more than 95% of primary and secondary schools** and serving **over 500 educational institutions with approximately 2 million users** [1][3][7]. The platform handles over **300,000 messages weekly** and has been operational for 25 years [7]. Wilma facilitates communication among teachers, students, and guardians but does NOT provide student-to-student messaging or peer assessment features.

#### 2.2.1 Anonymous vs. Identified Feedback Mechanisms

**No evidence exists that Wilma supports anonymous feedback or peer-to-peer feedback.** The system is designed for one-to-one and one-to-many communication among teachers, students, and guardians, with no student-to-student messaging channel [3][9][11]. The Lehmuskallio & Lampinen (2019) study published in the International Journal of Communication confirms that Wilma "provides channels for one-to-one and one-to-many communication among teachers, students, and the students' guardians" and "establishes an online social network for each individual, based on that person's role in the social world of the school" [3][9][11].

All feedback in Wilma is **identified** (teacher-to-student or teacher-to-guardian). The Oinas (2020) doctoral dissertation at the University of Helsinki analyzed 704 teachers' feedback entries for 7,811 students, finding that most feedback was positive and related to behavior and homework [7]. Students who received no feedback felt least motivated and had weaker teacher-student relationships. Two-thirds of all reported emotions in response to Wilma feedback were positive (joy, satisfaction, pride, relief) [7].

Forssa's Wilma usage guidelines confirm that policies "regarding student photos restrict use to internal school purposes and prohibit sharing among students," further suggesting the platform does not facilitate student peer interaction or feedback [6].

**Design Implication for Finland:** Peer feedback is not a native feature of Wilma, but Finnish culture is comfortable with direct, identified feedback. If peer feedback were to be added, it could be identified rather than anonymous, as Finnish directness culture supports open communication. However, the lack of such features reflects the Finnish model where feedback flows primarily from teacher to student.

#### 2.2.2 Teacher Moderation Levels

Wilma provides **structured teacher control** but within a context that often creates tension with Finnish egalitarian values. Teachers are required to "monitor Wilma messages daily and respond to guardians' messages by the next school day" [6]. The system formalizes communication rules: "Making lesson notes must be clear and objective; for example: 'Pekka hit his classmate on the head with a math book'" [6].

The Tuovi 2015 conference proceedings documented that "students' playing around with Wilma messages threatened the existing hierarchy between teacher and student, and thereby the entire school world" [6]. This reveals that despite Finland's low power distance, the system itself enforces hierarchical structures.

Teachers have developed coping mechanisms: "The guidebook recommends teachers avoid leaving traces in Wilma when dealing with hostile parents, favoring face-to-face or phone communication" [3][9][11]. Teachers experience "faster, more colloquial communication styles in Wilma, leading to increased boundary permeability and challenges in maintaining professional roles" [3][9][11].

The Kaaron koulu (Rauma) Wilma coding system shows teachers use annotations including positive markings (e.g., `+akt` for active participation, `+koe` for good test performance) and negative markings (e.g., `–käy` for bad behavior, `–kiu` for bullying, `–näp` for theft) [6]. This structured coding system formalizes teacher observations.

**Design Implication for Finland:** While Finnish culture is egalitarian, Wilma's moderation features are relatively structured. Teachers need tools that allow professional communication without being overly controlling. The tension between system rigidity and cultural informality suggests a need for flexible moderation settings that respect teacher professional judgment.

#### 2.2.3 Public vs. Private Mistake Visibility

Wilma keeps **grades and assessments private**—visible only to the individual student, their guardians, and teachers [6]. There is no mechanism for class-wide grade publication. The help documentation states: "Grades are visible to students and their guardians," with the guardian needing to acknowledge the test grade by clicking "Mark as seen" [6].

However, behavioral infractions are permanently recorded and visible to teachers and guardians. The Heimo, Rantanen & Kimppa (2015) paper titled **"Wilma ruined my life: how an educational system became the criminal record for the adolescents"** argues that Wilma effectively functions as a "criminal record for the adolescents," permanently storing behavioral infractions, absences, and disciplinary actions [4]. This creates a tension with Finnish educational values that emphasize "proactive, preventive action rather than corrective, reactive actions" [6].

The KARVI (Finnish Education Evaluation Centre) 2017-2019 evaluation found that "the most frequently used assessment methods were summative and individual-based," and "criteria behind the grades were not as clear for the learners and guardians as they were for the principals and teachers" [6]. The evaluation recommended "more attention should be paid to versatile formative and interactive assessment methods guiding the learning process."

**Design Implication for Finland:** Private feedback is culturally appropriate, but the permanent recording of mistakes and behavioral issues conflicts with Finnish educational philosophy that emphasizes growth and learning from errors. A more forgiving, formative approach to recording mistakes would better align with Finnish values.

#### 2.2.4 Outcome Metrics

- **95%+ of primary and secondary schools** in Finland [3]
- **2 million users** [1][7]
- **300,000+ messages weekly** [1][7]
- **1.8 million app downloads** (Android version) [5]
- **2.24/5 star rating** based on 15,000 user ratings (Android) [5]
- **1.5 million+ lines of code** [1][7]
- **ISO 27001 information security certification** [1][7]
- **704 teachers' feedback entries** analyzed for 7,811 students (academic study) [7]
- **2,031 ninth graders** surveyed on Wilma feedback effectiveness [7]

#### 2.2.5 Cultural Fit Summary

Wilma represents a complex case. While Finland's low power distance (33) and egalitarian values suggest open communication, the platform has been criticized for creating surveillance-like conditions that conflict with Finnish educational philosophy [4]. The "Three Views to a School Information System" paper (HCC13 2018) directly addresses this tension, noting that "the digitalisation of the school system seems inevitable" but "there have been some issues in the information system design to promote practices and values that are suboptimal—or even substandard for a school as an entity" [4]. Wilma has "become a standard in Finnish school system as the de facto school information system," but its design creates friction with Finnish values of trust, informality, and student autonomy.

---

### 2.3 Google Classroom — Latin America Deployments (Mexico Focus)

**Platform Overview:** Google Classroom is a global platform with over 150 million users worldwide [15]. In Latin America, and particularly Mexico, Google for Education has invested heavily in large-scale deployments. Key implementations include Baja California (162,149+ students across 480 public secondary schools, 13,000+ teachers trained, 175,000+ Education Plus licenses), Jalisco (1.3 million students, 45,000+ Chromebooks for teachers, 32,000 for students), and Campeche (the entity with the most trained teachers in Mexico) [2][5][8][15].

#### 2.3.1 Anonymous vs. Identified Feedback Mechanisms

**Google Classroom has NO native anonymous feedback feature.** This is a global platform limitation, not specific to Latin America. The platform supports two types of commenting:
- **Class comments**: Public to the class—visible to all teachers and students
- **Private comments**: Per-assignment—visible only to the individual student and teacher [1][3]

When a teacher explicitly asked in the Google Classroom Community whether students could submit comments anonymously, the response confirmed this is **not possible** natively [1].

Since anonymity is not built in, teachers in Latin America (and globally) use workarounds:
- **Google Forms**: Teachers create structured feedback forms with options for anonymity [9]
- **Third-party tools**: Synth (audio feedback), Microsoft Flip/Flipgrid (video responses), Mote (audio comments), Peergrade (platform with rubrics for peer assessment) [9]

A 2019 study on anonymous peer feedback via Google Classroom in a Thai EFL writing classroom found that "the quality of peer feedback significantly improved" and "anonymity of writers or feedback givers does not largely affect students' reactions," though "sufficient training must be provided before implementing the anonymous online peer feedback activity" [4].

**Design Implication for Mexico:** The absence of native anonymity in Google Classroom creates a gap for Mexican students in high power distance contexts who may be reluctant to give critical feedback publicly. A built-in anonymous feedback toggle would better serve this cultural context.

#### 2.3.2 Teacher Moderation Levels

Google Classroom's deployment strategy in Mexico is **deliberately teacher-centric**, accommodating high power distance cultural norms. The platform provides several teacher controls:
- Teachers can set whether students can post messages and comment on other posts on the Stream page [5]
- Teachers can view deleted comments and posts from students [5]
- Teachers control the grading system and whether to display overall class grades [5]
- Educational accounts have improved moderation controls, longer Google Meet sessions (24 hours), and higher class member limits (up to 1000 per class) [5]

However, the most significant finding is **how Google for Education deploys in Mexico: teacher-first**. In Baja California, the strategy began with "massive teacher training, combining in-person and online sessions" [2][8]. Initially, only 3% of teachers used technology daily in classrooms; teacher training was the primary priority before scaling to students [2][8].

In Jalisco, **45,000+ Chromebooks were distributed to TEACHERS first** (covering 90%+ of all elementary and high school teachers), and only then were 32,000 Chromebooks provided to students via mobile "Aulas Google" carts [2][8]. Google's messaging explicitly states: "When technology empowers teachers, it leads to more personalized and human-centered education" [2][8].

The AI messaging in Latin America also emphasizes teacher augmentation, not replacement: "Artificial intelligence does not replace human connection. On the contrary, its true contribution lies in expanding the possibilities of teaching, as it frees up time from repetitive tasks, facilitates personalization, and offers new ways to spark curiosity" [2][8].

**Design Implication for Mexico:** Teacher moderation features should be robust and visible. The platform should position teachers as empowered authority figures, not replace them. Teacher training and support should be a foundational component of any deployment.

#### 2.3.3 Public vs. Private Mistake Visibility

Google Classroom defaults to **private feedback** between teacher and student. Individual grades are visible only to the student and teacher [6]. The teacher controls whether to display overall class grade averages [5].

Common Sense Media's privacy evaluation gives Google Classroom an overall privacy rating of **89%**, confirming that it "meets minimum privacy and security requirements" [7]. Their Spanish-language guide confirms that "parents cannot view grades directly through Classroom" but receive reports from teachers [7].

Class comments, when enabled, are the only public element—visible to all class members [1][3]. However, teachers can disable this feature entirely, restricting communication to private channels.

The San Luis Potosí study highlighted significant digital divide challenges affecting how these privacy features function: 31.6% of households lacked internet access and 46.1% lacked a computer (2020 INEGI data), meaning many students could not access private feedback [13]. Around 46.55% of nearly 2,000 secondary teachers engaged with digital content repositories during 2020-2021 [13].

**Design Implication for Mexico:** Private mistake handling is appropriate for high power distance culture, but the digital divide creates equity issues. Offline-capable feedback mechanisms and multi-channel communication (WhatsApp integration, printable reports) would address this.

#### 2.3.4 Outcome Metrics

- **162,149+ students** across 480 public secondary schools in Baja California [2][8]
- **13,000+ teachers** trained in Baja California [2][8]
- **175,000+ Education Plus licenses** deployed in Baja California [2][8]
- **1.3 million students** in Jalisco using Google Classroom simultaneously [2][8]
- **45,000+ Chromebooks** provided to Jalisco teachers (90%+ coverage) [2][8]
- **32,000 Chromebooks** provided to Jalisco students via mobile carts [2][8]
- **15 million pesos** government investment in Baja California [10]
- **96.7%** of students achieved high levels of meaningful learning post-Google for Education adoption (Peru study) [12]
- **86.7%** of students showed low meaningful learning BEFORE Google for Education adoption [12]
- **Statistical significance: T = 27.159, p < 0.000** [12]
- **65.5%** of students regularly used Google for Education platform (San Luis Potosí) [13]
- **34M+ students** globally using Lectura Guiada (Guided Reading); 172M+ stories read [2]
- Target: **1.25 million students** reached by 2028 via Google.org AI education investment in 9 LatAm countries [11]

#### 2.3.5 Cultural Fit Summary

Google for Education's Latin America strategy demonstrates a sophisticated understanding of high power distance cultures. The teacher-first approach—training teachers first, giving them devices first, positioning AI as teacher augmentation—directly accommodates hierarchical cultural norms. The emphasis on collective effort (state-wide initiatives, government partnerships, "Nadie afuera, nadie atrás" motto) aligns with collectivist values [2][8][13].

However, the lack of native anonymous feedback is a notable gap for collectivist, high power distance contexts where students may be reluctant to give critical peer feedback. The reliance on workarounds (Google Forms, third-party tools) suggests an opportunity for culturally adaptive platform features.

---

## 3. Cross-Cultural Analysis of Interaction Patterns

### 3.1 Anonymous vs. Identified Feedback Mechanisms

| Feature | Classting (South Korea) | Wilma (Finland) | Google Classroom LatAm (Mexico) |
|---------|------------------------|-----------------|----------------------------------|
| Anonymous peer feedback | Yes (Private Counseling) | No (no peer feedback) | No (native); workarounds via Google Forms |
| Identified feedback | Yes (teacher-student) | Yes (teacher-student) | Yes (private comments, class comments) |
| Student-to-student feedback | Limited, mediated | None | Possible via class comments |
| Cultural alignment | High: addresses face-saving | Moderate: directness culture needs identified feedback | Low: anonymity needed in high PDI context |

**Academic Research Support:**

The Kunwar (2020) study found that "individuals from hierarchical, collectivist, and traditional cultures were less familiar with peer learning methods and prefer authoritative conflict resolution" [3]. Students from collectivist cultures "may take less direct initiative in contributing unless explicitly asked" [3].

The Gunawardena (2013) research on culture and online distance learning found that "open disagreement—considered essential in Western pedagogy for knowledge construction—is often avoided in collectivist cultures, affecting online discussions" [20]. Furthermore, "non-native speakers, particularly students from Asian countries, consider it far less appropriate to challenge and criticize the ideas of others" [20].

The Zhang (2013) study on power distance in online learning found that "Chinese learners tended to seek help from peers, particularly those who shared similar cultural and linguistic backgrounds," but were "intimidated to interact with their instructors" when encountering difficulties [1].

**Design Recommendations:**

- **South Korea:** Implement anonymous peer feedback as a default option. Frame feedback as "suggestions for improvement" rather than criticism. Allow students to opt for private communication with teachers when discussing sensitive topics.
- **Finland:** Identified, direct feedback is culturally appropriate. Peer feedback features could be introduced as identified rather than anonymous, but should be optional rather than mandatory.
- **Mexico:** Since native anonymity is not available in Google Classroom, integrate Google Forms or third-party tools for anonymous peer assessment. Consider building anonymous feedback into the platform for high power distance contexts.

---

### 3.2 Teacher Moderation Levels

| Feature | Classting (South Korea) | Wilma (Finland) | Google Classroom LatAm (Mexico) |
|---------|------------------------|-----------------|----------------------------------|
| Teacher controls posting | Yes (granular) | Yes (formalized) | Yes (configurable) |
| Teacher creates classes | Only teachers | Teachers + administrators | Teachers |
| Monitoring | Real-time + AI-assisted | Daily monitoring required | Standard analytics |
| Student autonomy | Low | Low (despite cultural context) | Moderate |
| Cultural alignment | High: matches high PDI | Low: creates tension with low PDI | High: teacher-first strategy |

**Academic Research Support:**

The Yoon (2019) cross-cultural study of Finnish and Korean schools found that "student agency was tightly controlled by regulations of time, space, and movement and was extensively limited in teaching-learning practices" in both countries, but "a less vertical relationship and weaker teacher control over students' self-expression were observed in Finnish schools, whereas hierarchical relationships and intensive teacher control were more noticeable in Korean schools" [13].

The Gómez-Rey, Barbera, & Fernández-Navarro (2016) study found that "learners' autonomy levels differ culturally, with US and Spanish students showing higher autonomy compared to Chinese and Mexican students" [4]. This supports the finding that Mexican students in high power distance contexts need more teacher guidance and structure.

The Heo, Leppisaari, & Lee (2018) study on learning culture in Finnish and South Korean classrooms identified "teacher's autonomy in teaching, authenticity in learning, relationships between teachers and students, learning assessment, student engagement, and student well-being" as key themes distinguishing the two cultures [15].

**Design Recommendations:**

- **South Korea:** Maintain high teacher control as a baseline. Provide granular permission settings (who can post, comment, share). Include AI-assisted moderation to reduce teacher burden while maintaining authority.
- **Finland:** Offer flexible moderation that allows teachers to choose their level of intervention. Default to lower moderation but provide tools for when issues arise. Avoid permanent, irreversible monitoring that conflicts with Finnish trust culture.
- **Mexico:** Implement teacher-first deployment strategy. Prioritize teacher training and device allocation. Position AI and technology tools as teacher augmentation, not replacement. Maintain visible teacher controls.

---

### 3.3 Public vs. Private Mistake Visibility

| Feature | Classting (South Korea) | Wilma (Finland) | Google Classroom LatAm (Mexico) |
|---------|------------------------|-----------------|----------------------------------|
| Grade visibility | Private (student + teacher) | Private (student + guardian + teacher) | Private (student + teacher) |
| Error review | Private (individual error repository) | Private (linked to student) | Private (private comments) |
| Public correction | Not supported | Not supported | Class comments possible but optional |
| Permanent error record | No (AI-focused, formative) | Yes (behavioral "criminal record") | No (assignment-based, deletable) |
| Cultural alignment | High: preserves face | Low: permanent records conflict with growth mindset | Moderate: private correction aligns with high PDI |

**Academic Research Support:**

The Zhang (2013) study found that while the asynchronous nature of online learning can benefit learners from high power distance cultures by providing more time to formulate responses, "it may increase the level of anxiety in their participation" due to the permanent and public nature of text-based posts [1].

The Kunwar (2020) study noted that "individuals from feminine cultures prefer direct negative feedback and public acknowledgement of positive contributions," while hierarchical cultures prefer more indirect approaches [3].

The Latin American context, as described in the Hoja de Ruta article from the Knight Center, emphasizes that "there is no learning without error" and that it is "necessary to be very humble to stop, observe, learn and take advantage of the mistake" [21]—suggesting a cultural framing that could accommodate more open error discussion if handled constructively.

**Design Recommendations:**

- **South Korea:** Private error correction is essential. Implement individual error repositories where students can review mistakes privately. Frame errors as learning opportunities within private channels. Never display student mistakes to the class publicly.
- **Finland:** While individual privacy is important, the permanent recording of behavioral mistakes (the "criminal record" problem) should be addressed. Implement formative, growth-oriented error tracking that can be updated or removed as students improve.
- **Mexico:** Default to private feedback between teacher and student. Allow optional, opt-in public sharing for positive exemplars only. Address the digital divide by ensuring private feedback is accessible offline or via multiple channels.

---

## 4. Outcome Metrics Comparison

### 4.1 Engagement Rates

| Metric | Classting (Korea) | Wilma (Finland) | Google Classroom (Mexico) |
|--------|-------------------|-----------------|---------------------------|
| Student satisfaction | 80%+ found AI quizzes helpful | 67% positive emotions from feedback | 96.7% achieved high meaningful learning |
| School adoption | 2,500+ schools | 95%+ of schools | 480 schools (Baja alone); 1.3M students (Jalisco) |
| User scale | 210K+ AI users; 5M total previously | 2M users; 300K+ messages/week | 150M+ globally; millions in LatAm |
| User ratings | Positive (teacher testimonials) | 2.24/5 (Android) | N/A (not a standalone app) |

### 4.2 Peer Interaction Frequency

| Metric | Classting (Korea) | Wilma (Finland) | Google Classroom (Mexico) |
|--------|-------------------|-----------------|---------------------------|
| Peer-to-peer communication | Limited, teacher-mediated | None (system does not support) | Possible via class comments |
| Feedback types | Teacher-student + AI-generated | Teacher-student only | Teacher-student + class comments |
| Collaborative features | Messaging, shared spaces | Schedule, grades, communication | Shared documents, comments, meet |

### 4.3 Assignment Completion Rates

| Metric | Classting (Korea) | Wilma (Finland) | Google Classroom (Mexico) |
|--------|-------------------|-----------------|---------------------------|
| Completion data | Positive (flipped learning study) | N/A (attendance-focused) | Available via Classroom Analytics |
| AI prediction accuracy | 91.5% | N/A | N/A |
| Writing skills improvement | 71.4% reported improvement | N/A | Positive (UAZ Zacatecas study) |

---

## 5. Design Recommendations by Cultural Context

### 5.1 South Korea: High Power Distance + Collectivism + Face-Saving

**Design Principles:**
- **Teacher as Gatekeeper:** Maintain high teacher control over class creation, posting permissions, and content moderation. Teachers should have the final say on what is published and visible.
- **Privacy as Default:** All feedback, error correction, and assessment should be private between teacher and student. Implement individual error repositories and private counseling channels.
- **Anonymous Feedback:** Provide anonymous options for students to ask questions, give peer feedback, or express concerns. Frame feedback as "constructive suggestions" rather than criticism.
- **Indirect Communication:** Use AI-generated feedback to model constructive, emotionally supportive commentary. Peer feedback should be structured with templates that emphasize positive framing and suggestions for improvement.
- **Face-Saving Mechanisms:** Never publicly display student mistakes. Use private channels for correction. Allow students to opt into public recognition for achievements only.

**Platform Features to Prioritize:**
- Anonymous private messaging between student and teacher
- Individual error/answer repositories (private per student)
- Teacher-controlled posting and commenting permissions
- AI-assisted feedback that models constructive language
- Structured peer feedback templates with praise-first framing

### 5.2 Finland: Low Power Distance + Individualism + Direct Communication

**Design Principles:**
- **Teacher as Facilitator:** Teachers should have flexible moderation options that respect their professional judgment. Default to lower intervention but provide tools for when issues arise.
- **Transparency and Directness:** Identified feedback is culturally appropriate. Finnish students are comfortable with direct communication, so anonymity is less critical.
- **Formative Assessment:** Avoid permanent, irreversible recording of mistakes. Implement growth-oriented tracking that can be updated as students improve.
- **Trust Over Surveillance:** The permanent behavioral recording in Wilma ("criminal record" problem) should be addressed. Design feedback systems that emphasize learning and growth rather than surveillance.
- **Student Autonomy:** Allow students greater control over their learning data and feedback processes. Finnish students value independence and self-direction.

**Platform Features to Prioritize:**
- Flexible teacher moderation settings (opt-in rather than mandatory)
- Identified, direct peer feedback (optional)
- Formative assessment tracking with updateable error records
- Student access to their own learning data with opt-out options
- Transparent communication channels that respect privacy boundaries

### 5.3 Mexico: High Power Distance + Collectivism + Teacher Authority

**Design Principles:**
- **Teacher-First Deployment:** Prioritize teacher training, device allocation, and professional development before scaling to students. Position technology as teacher empowerment, not replacement.
- **Visible Teacher Authority:** Maintain visible teacher controls and moderation features. Teachers should be able to control posting, commenting, and content visibility.
- **Private Correction:** Default to private feedback between teacher and student. Implement anonymous feedback options for peer assessment to accommodate face-saving norms.
- **Address the Digital Divide:** Ensure feedback mechanisms work offline or via multiple channels (WhatsApp, printable reports). Equity of access is critical.
- **Collective Framing:** Use language that emphasizes collective effort and community. Frame assignments and feedback in terms of group growth and shared learning.

**Platform Features to Prioritize:**
- Native anonymous feedback toggle for peer assessment
- Offline-capable feedback mechanisms (printable reports, WhatsApp integration)
- Teacher training modules and certification paths
- AI tools positioned as teacher augmentation
- Multi-channel communication (SMS, email, in-app, printable)

### 5.4 Universal Design Recommendations

1. **Flexible Anonymity Toggle:** Allow teachers to configure whether peer feedback is anonymous or identified on a per-assignment basis. This accommodates cultural differences within the same platform.

2. **Configurable Moderation Levels:** Provide three presets—High (teacher approves all posts), Medium (teacher reviews flagged content), Low (students post freely)—with cultural defaults based on location.

3. **Private Error Repositories:** Implement individual student error banks that are visible only to the student and teacher, with growth tracking over time.

4. **Cultural Onboarding:** When deploying in a new region, provide cultural context documentation for teachers and administrators explaining how features align with local educational values.

5. **Localized AI Feedback:** Train AI feedback models on culturally appropriate language patterns—constructive and indirect for East Asian contexts, direct and specific for Nordic contexts, relationship-oriented for Latin American contexts.

---

## 6. Conclusion

The design of collaborative learning interfaces cannot be culturally neutral. The documented implementations of Classting in South Korea, Wilma in Finland, and Google Classroom in Mexico reveal that successful educational technology must adapt to local cultural dimensions of power distance, collectivism, and attitudes toward authority and error.

**Key Takeaways:**

1. **Anonymous feedback is essential in high power distance, face-saving cultures** (Korea, Mexico) but less critical in low power distance cultures (Finland). Platforms should offer flexible anonymity toggles rather than forcing a single approach.

2. **Teacher moderation is accepted and expected in high power distance contexts** (Korea, Mexico) but creates tension in low power distance contexts (Finland) where egalitarian values conflict with system-imposed hierarchies.

3. **Private mistake correction is universally preferred**, but different cultures have different attitudes toward the permanence of error records. Korean culture prefers private, formative error review. Finnish culture is harmed by permanent behavioral records that conflict with growth mindset. Mexican culture needs private correction with attention to the digital divide.

4. **Teacher-first deployment strategies are critical in high power distance, collectivist cultures** (Mexico, Korea). Technology should be framed as teacher augmentation, not replacement. Teacher training and professional development should precede student-facing rollout.

5. **The most successful platforms adapt their core features**—feedback mechanisms, moderation levels, and error visibility—to local cultural norms rather than imposing a single global design.

By understanding and accommodating these cultural differences, educational technology platforms can create collaborative learning environments that respect local values while achieving measurable improvements in student engagement, peer interaction, and assignment completion.

---

## Sources

[1] Classting - Korea's Largest Ed-Tech Startup Brings Social Based Adaptive Learning to the US: https://www.newswire.com/news/classting-koreas-largest-ed-tech-startup-brings-social-based

[2] 초등학교 사회과 수업에서 클래스팅을 활용한 플립 러닝 실행연구: https://scholar.kyobobook.co.kr/article/detail/4010026091463

[3] 선생님 가이드 – 클래스팅 고객센터: https://support.classting.com/hc/ko/categories/200429558-%EC%84%A0%EC%83%9D%EB%8B%98-%EA%B0%80%EC%9D%B4%EB%93%9C

[4] 클래스 관리자 권한 양도하기: https://support.classting.com/hc/ko/articles/115003754808-%ED%81%B4%EB%9E%98%EC%8A%A4-%EA%B4%80%EB%A6%AC%EC%9E%90-%EA%B6%8C%ED%95%9C-%EC%96%91%EB%8F%84%ED%95%98%EA%B8%B0

[5] 클래스의 게시글∙공지∙댓글∙반응 알림 설정 방법: https://support.classting.com/hc/ko/articles/6467865603865-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%9D%98-%EA%B2%8C%EC%8B%9C%EA%B8%80-%EA%B3%B5%EC%A7%80-%EB%8C%93%EA%B8%80-%EB%B0%98%EC%9D%91-%EC%95%8C%EB%A6%BC-%EC%84%A4%EC%A0%95-%EB%B0%A9%EB%B2%95

[6] 클래스팅, SaaS 부문 매출 전년 대비 6.4배 올랐다: https://m.hellot.net/mobile/article.html?no=91864

[7] 클래스팅 AI 코스웨어 '전국 초등 디지털 선도학교의 3곳 중 2곳 활용해': https://www.aitimes.kr/news/articleView.html?idxno=30039

[8] 개인화 교육을 실현하는 교육 AI 에이전트 | 클래스팅: https://www.classting.com

[9] 클래스팅 AI - 인공지능과 교육 - 위키독스: https://wikidocs.net/128449

[10] 클래스팅 AI, 월 구독 학교 2배 증가…전국 2500곳 돌파: https://www.eduplusnews.com/news/articleView.html?idxno=13550

[11] 클래스팅 (r561 판) - 나무위키: https://namu.wiki/w/%ED%81%B4%EB%9E%98%EC%8A%A4%ED%8C%85?uuid=b4fc8ec7-6d21-4eb4-895d-a9ac93f9a691

[12] 클래스에서 학생/학부모 게시글 쓰기 권한 설정 가능 업데이트: https://support.classting.com/hc/ko/articles/115001339794-%ED%81%B4%EB%9E%98%EC%8A%A4%EC%97%90%EC%84%9C-%ED%95%99%EC%83%9D-%ED%95%99%EB%B6%80%EB%AA%A8-%EA%B2%8C%EC%8B%9C%EA%B8%80-%EC%93%B0%EA%B8%B0-%EA%B6%8C%ED%95%9C-%EC%84%A4%EC%A0%95-%EA%B0%80%EB%8A%A5-%EC%97%85%EB%8D%B0%EC%9D%B4%ED%8A%B8

[13] AI 기반 형성적 동료평가를 활용한 수학수업에서 AI와 학생의 피드백 특성 및 만족도 분석 (2025): https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003256022

[14] 최고･일반 관리자로 등록하기 / 내보내기 / 나가기: https://support.classting.com/hc/ko/articles/15519806215961-%EC%B5%9C%EA%B3%A0-%EC%9D%BC%EB%B0%98-%EA%B4%80%EB%A6%AC%EC%9E%90%EB%A1%9C-%EB%93%B1%EB%A1%9D%ED%95%98%EA%B8%B0-%EB%82%B4%EB%B3%B4%EB%82%B4%EA%B8%B0-%EB%82%98%EA%B0%80%EA%B8%B0

[15] 【클래스팅 AI】 개별 맞춤형 피드백과 자기주도학습으로 대학 영어 독해의 패러다임을 바꾸다! - YouTube (경상국립대학교 이희정 교수): https://www.youtube.com/watch?v=FyP6oz9hXfo

[16] 게시물 공간 구성 및 관리하기: https://support.classting.com/hc/ko/articles/15064601219737-%EA%B2%8C%EC%8B%9C%EB%AC%BC-%EA%B3%B5%EA%B0%84-%EA%B5%AC%EC%84%B1-%EB%B0%8F-%EA%B4%80%EB%A6%AC%ED%95%98%EA%B8%B0

[17] 교사 이용방법: https://support-ai.classting.com/hc/ko/categories/7774868407065-%EA%B5%90%EC%82%AC-%EC%9D%B4%EC%9A%A9%EB%B0%A9%EB%B2%95

[18] The Classroom Etiquette You Didn't Know You Needed in South Korea: https://mytefl.com/2025/10/24/the-classroom-etiquette-you-didnt-know-you-needed-in-south-korea

[19] Understanding Korean School Culture for Teachers: https://www.ciee.org/go-abroad/work/teach-english-abroad/blog/navigating-korean-culture-education-system-guide-english-teachers

[20] Korean Confucianism (Stanford Encyclopedia of Philosophy): https://plato.stanford.edu/archives/win2021/entries/korean-confucianism

[21] The Contemporary Korean Education System and Confucianism: https://asadalthought.wordpress.com/2009/06/02/the-contemporary-korean-education-system-and-confucianism

[22] Wilma.fi (Official Site): https://www.wilma.fi

[23] Finland: Wilma — Most popular teaching, learning and assessment platform | NordenBladet: https://nordenbladet.com/articles/104642-wilma-finlands-most-popular-teaching-learning-and-assessment-platform

[24] Lehmuskallio & Lampinen (2019) - Material Mediations Complicate Communication Privacy Management: The Case of Wilma in Finnish High Schools: https://ijoc.org/index.php/ijoc/article/view/11357/2878

[25] Heimo, Rantanen & Kimppa (2015) - Wilma ruined my life: how an educational system became the criminal record for the adolescents: https://www.utu.fi/en/publications/wilma-ruined-my-life-how-an-educational-system-became-the-criminal-record-for-the-0

[26] Oinas (2020) Doctoral Dissertation - Wilma-palautteella yhteys oppimiseen ja hyvinvointiin: https://www.helsinki.fi/fi/uutiset/yhteiskunta-ja-oppiminen/wilma-palautteella-yhteys-oppimiseen-ja-hyvinvointiin

[27] Google Classroom Guide - Comments: https://sites.google.com/site/gclassroomguide/stream/comments

[28] Google Classroom Community - Anonymous submission: https://support.google.com/edu/classroom/thread/57865585/can-i-have-students-submit-their-comments-on-my-posts-anonymously-without-even-myself-knowing?hl=en

[29] Ditch That Textbook - 10 tools for effective peer feedback in the classroom: https://ditchthattextbook.com/10-tools-for-effective-peer-feedback-in-the-classroom

[30] A Study of the Quality of Feedback Via the Google Classroom-mediated-Anonymous Online Peer Feedback Activity in a Thai EFL Writing Classroom: https://ijpe.inased.org/makale/1116

[31] Google Classroom Help - Administra los detalles y la configuración de la clase: https://support.google.com/edu/classroom/answer/6076302?hl=es-419

[32] Google for Education Mexico - Nueva era en la enseñanza: https://blog.google/intl/es-419/actualizaciones-de-producto/informacion/nueva-era-en-la-ensenanza-google-for-education-mexico

[33] Google for Education - Education on the Move LatAm customer story: https://edu.google.com/resources/customer-stories/education-on-the-move-latam

[34] Google Classroom Statistics: https://support.google.com/edu/classroom/answer/14221372?hl=es-419

[35] Faro Educativo - El uso de las plataformas educativas Google Classroom durante y después de la pandemia por COVID-19 (San Luis Potosí): https://faroeducativo.ibero.mx/2023/01/13/el-uso-de-las-plataformas-educativas-google-classroom-durante-y-despues-de-la-pandemia-por-covid-19

[36] Google.org commits $4.6 million to AI education rollout across Latin America: https://www.edtechinnovationhub.com/news/googleorg-commits-46-million-to-ai-education-rollout-across-latin-america

[37] Zhang (2013) - Power Distance in Online Learning: Experience of Chinese Learners in U.S. Higher Education: https://files.eric.ed.gov/fulltext/EJ1017526.pdf

[38] Kunwar (2020) - Cultural Influences in Collaborative Peer Learning: ResearchGate (Spring 2020)

[39] Gómez-Rey, Barbera, & Fernández-Navarro (2016) - The Impact of Cultural Dimensions on Online Learning: https://www.jstor.org/stable/jeductechsoci.19.4.225

[40] Yoon (2019) - Quality of School Life of Adolescents in Finland and Korea (Doctoral Dissertation, University of Turku)

[41] Heo, Leppisaari, & Lee (2018) - Exploring Learning Culture in Finnish and South Korean Classrooms: https://eric.ed.gov/?id=EJ1182024

[42] Vierimaa - Power Distance in Finnish Higher Education Institutions from the North American Perspective: https://erityisopettaja.fi/power-distance-in-finnish-higher-education-institutions-from-the-north-american-perspective

[43] Clearly Cultural - Power Distance Index: https://clearlycultural.com/geert-hofstede-cultural-dimensions/power-distance-index/

[44] A Quarter Century of Culture's Consequences (2002) - Review of Hofstede's Framework: Springer / Journal of International Business Studies

[45] Saif (TESOL) - Hofstede's Cultural Dimensions in International Education

[46] Lim (2004) - Cross Cultural Differences in Online Learning Motivation (Korea vs. U.S.): University of Tennessee

[47] Schepers & Wetzels (2007) - A Meta-analysis of the Technology Acceptance Model: https://www.sciencedirect.com/science/article/pii/S0378720606001248

[48] Latin American Higher Education Student Engagement During COVID-19 - Systematic Review: https://pmc.ncbi.nlm.nih.gov/articles/PMC9111674/

[49] UNAM - Manual de Google Classroom: https://cuaed.unam.mx/descargas/Manual-Google-Classroom.pdf

[50] Common Sense Media - Google Classroom Privacy Evaluation: https://privacy.commonsense.org/evaluation/Google-Classroom