Triagegeist: AI-Powered Emergency Triage Acuity Prediction
Subtitle
Combining clinical decision science, NLP-derived chief complaint features, and gradient boosting to predict ESI triage levels with 98.6% accuracy and minimal demographic bias.

1. Clinical Problem Statement
Emergency department (ED) triage is the critical first point of clinical decision-making. Triage nurses must rapidly assign an acuity level — typically using the Emergency Severity Index (ESI) — that determines how quickly a patient receives care. This decision relies on integrating vital signs, chief complaints, mental status, and clinical judgment, often under extreme time pressure with incomplete information.
Inter-rater variability in triage scoring is well-documented in the literature. Studies have shown kappa values of 0.60–0.80 between experienced triage nurses, meaning meaningful disagreement occurs in 20–40% of cases. Undertriage — assigning a lower acuity than clinically warranted — poses a direct patient safety risk, as it delays care for patients who may be deteriorating. Systematic undertriage of certain populations has also been identified as an equity concern.
This project addresses the question: Can a machine learning model, trained on structured clinical data and free-text chief complaints, accurately predict triage acuity to serve as a real-time clinical decision support tool?
We focus specifically on:

Predicting ESI levels 1–5 from data available at the point of triage
Identifying features most predictive of acuity to support clinical transparency
Analyzing potential demographic biases in model predictions
Assessing clinical safety through undertriage pattern analysis


2. Methodology
2.1 Data Preparation
The competition provides four linked datasets covering 100,000 Finnish ED visits (80,000 train / 20,000 test):

Structured clinical data: Demographics, vital signs (BP, HR, RR, temperature, SpO2), GCS, pain score, NEWS2 score, shock index, arrival information
Chief complaints: Free-text presenting complaints with system classification
Patient history: 25 binary comorbidity flags (hypertension, diabetes, heart failure, etc.)

Data leakage prevention: We identified that disposition and ed_los_hours appear in the training set but not the test set. These are post-triage outcomes that would not be available at the time of triage. We explicitly excluded them from all feature engineering.
Missing values: Vital sign missingness (5.2% for blood pressure, 3.8% for respiratory rate, 0.7% for temperature) was preserved as-is, as missingness in ED vitals is clinically meaningful — a missing vital sign may indicate the patient was too unstable to measure, or that the clinical team prioritized immediate intervention.
2.2 Feature Engineering
We created 30+ clinically-motivated features organized into six categories:

Hemodynamic instability indicators: Hypotension (SBP<90), tachycardia (HR>100), bradycardia (HR<60), elevated shock index (SI≥1.0)
Respiratory distress indicators: Tachypnea (RR>20), hypoxia (SpO2<94%), severe hypoxia (SpO2<88%)
Temperature abnormalities: Fever (≥38°C), high fever (≥39°C), hypothermia (<36°C)
Neurological status: GCS-based severity categories (severe ≤8, moderate 9-12, normal 15), NEWS2 risk stratification (high ≥7)
NLP keyword extraction: 11 clinical keyword features from chief complaints (severe, acute, mild, chronic, cardiac, neurological, trauma, bleeding, psychiatric, respiratory, word count)
Comorbidity burden: Total comorbidity count, cardiac risk composite, high-risk combinations (elderly + cardiac disease)

Additionally, we built 100 TF-IDF features from chief complaint bigrams to capture semantic patterns beyond simple keyword matching.
2.3 Model Architecture
We trained a LightGBM gradient boosting classifier using 5-fold stratified cross-validation. Key design choices:

Algorithm selection: LightGBM was chosen for its handling of categorical features, robustness to missing values, and computational efficiency on 80K samples with 186 features
Stratified CV: Essential given class imbalance (ESI-1 is only 4% of data)
Early stopping: 20 rounds patience to prevent overfitting
Ensemble: Final predictions are the mean of 5 fold-level probability estimates

2.4 Evaluation Metrics
We report accuracy, quadratic weighted kappa (QWK), per-class precision/recall/F1, and clinically-specific metrics including undertriage rates by ESI level and demographic group.

3. Results
3.1 Model Performance
MetricScoreOverall Accuracy98.59%Quadratic Weighted Kappa0.9932ESI-1 Recall94.4%ESI-2 Recall98.5%ESI-3 Recall99.0%ESI-4 Recall98.9%ESI-5 Recall98.2%
Cross-validation was highly stable, with fold-level accuracy ranging from 98.52% to 98.65% and QWK from 0.9929 to 0.9936.
3.2 Feature Importance
The top predictive features align with established clinical triage criteria:

NEWS2 score — The single most important feature, confirming that standardized early warning scores capture the core dimensions of clinical acuity
Respiratory rate — A key vital sign in deterioration detection
Chief complaint system — The body system category provides strong baseline signal
SpO2 — Oxygen saturation is critical for respiratory and cardiac emergencies
Temperature — Fever is a key discriminator between urgent and non-urgent presentations

NLP features contributed meaningfully: chief complaint word count, and TF-IDF features capturing specific clinical phrases, all appeared in the top 25 features.
3.3 Clinical Safety Analysis
ESI LevelUndertriage RatePrimary ErrorESI-15.6% (182/3222)→ ESI-2 (99.5% of errors)ESI-21.0% (128/13439)→ ESI-3ESI-30.8% (242/28921)→ ESI-4ESI-40.5% (121/23020)→ ESI-5
The 5.6% undertriage rate for ESI-1 patients is the primary safety concern. However, nearly all ESI-1 errors are classified as ESI-2 (still emergent priority), not as ESI-3 or lower. No ESI-1 patient was predicted as ESI-4 or ESI-5. This "adjacent-class" error pattern is clinically less dangerous than skip-level undertriage.
3.4 Bias Analysis
We found minimal demographic bias across all examined dimensions:
DimensionAccuracy RangeMax DisparitySex (M/F/Other)98.51% – 98.70%0.19%Age Group98.49% – 98.79%0.30%Language98.33% – 98.65%0.32%Insurance Type98.31% – 98.65%0.34%
No subgroup showed statistically or clinically meaningful undertriage disparities. The model performs equitably across the Finnish multilingual ED population.

4. Findings and Clinical Implications
Key Insight 1: NEWS2 score alone explains much of triage acuity, but the model's additional features improve discrimination for ESI-3/4 borderline cases where NEWS2 alone is insufficient.
Key Insight 2: NLP features from chief complaints are the model's "clinical judgment layer." Keywords like "severe," "acute," and "mild" mirror the severity language that experienced triage nurses use to calibrate their assessments. The model effectively learns this clinical vocabulary.
Key Insight 3: GCS creates a near-perfect boundary for ESI-1 identification. All patients with GCS ≤ 8 were ESI-1, reflecting the clinical reality that comatose patients require immediate airway management.
Key Insight 4: The model's equitable performance across demographics is encouraging for potential deployment in diverse clinical settings. However, this finding needs validation on real clinical data, as synthetic data may not capture subtle bias patterns.

5. Limitations and Reproducibility
Limitations:

Data is synthetic, which may not capture all real-world clinical complexity (rare presentations, atypical symptoms, concurrent conditions)
The model uses keyword-based NLP rather than contextual embeddings; transformer-based models may improve chief complaint understanding
ESI-1 undertriage rate of 5.6% would need targeted improvement before clinical deployment
No prospective validation on live clinical data has been performed
The model does not account for ED crowding, resource availability, or real-time patient flow

Reproducibility: The notebook runs end-to-end in approximately 5 minutes on Kaggle's CPU environment. All random seeds are fixed. No external datasets or APIs are required.

6. Novelty and Impact Potential
This work contributes a clinically-grounded, interpretable triage prediction system that:

Bridges clinical and ML methodologies by engineering features that map directly to emergency medicine decision criteria, making the model transparent to clinicians
Integrates NLP with structured data to capture both quantitative vital signs and qualitative chief complaint semantics
Provides systematic bias auditing across demographics — essential for responsible clinical AI deployment
Identifies specific safety concerns (ESI-1 undertriage patterns) with clinically meaningful interpretation, not just statistical error rates

The model could serve as the foundation for a real-time clinical decision support system that provides triage nurses with a "second opinion" — particularly valuable during high-volume shifts, overnight hours, and in settings with less experienced triage staff. The Laitinen-Fredriksson Foundation's pilot studies in Northern European hospital networks could directly benefit from this approach.
