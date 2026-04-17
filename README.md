# Triagegeist: AI-Powered Emergency Triage Acuity Prediction

> Combining clinical decision science, NLP-derived chief complaint features, and gradient boosting to predict ESI triage levels with **98.6% accuracy** and **minimal demographic bias**.

![Cover](cover_image.png)

## Overview

This project is a submission to the [Triagegeist Kaggle Competition](https://kaggle.com/competitions/triagegeist) hosted by the Laitinen-Fredriksson Foundation. It builds an AI-powered triage acuity prediction system for emergency departments.

### Key Results

| Metric | Score |
|--------|-------|
| Overall Accuracy | **98.59%** |
| Quadratic Weighted Kappa | **0.9932** |
| ESI-1 (Critical) Recall | 94.4% |
| Max Demographic Bias Gap | <0.35% |
| Features Used | 186 |


## Methodology

### Data Pipeline
1. **Data Integration**: Merged structured vitals, free-text chief complaints, and patient history across 100,000 Finnish ED visits
2. **Leakage Prevention**: Excluded post-triage outcome variables (`disposition`, `ed_los_hours`)
3. **Clinical Feature Engineering**: 30+ features grounded in emergency medicine literature
4. **NLP Pipeline**: TF-IDF bigrams + keyword extraction from chief complaints
5. **Model**: LightGBM gradient boosting with 5-fold stratified cross-validation

### Feature Categories
- **Hemodynamic instability**: Hypotension, tachycardia, shock index
- **Respiratory distress**: Tachypnea, hypoxia thresholds
- **Neurological status**: GCS severity categories, NEWS2 risk bands
- **Clinical NLP**: Severity keywords, complaint system classification
- **Comorbidity burden**: Composite risk scores from 25 medical history flags

### Clinical Safety Analysis
- ESI-1 undertriage rate: 5.6% (nearly all errors are adjacent-class → ESI-2)
- No skip-level undertriage (ESI-1 never predicted as ESI-4/5)
- Equitable performance across sex, age, language, and insurance groups

## Interactive Demo

The `triagegeist_demo.jsx` file contains a React-based clinical decision support interface that:
- Accepts patient vitals, mental status, and chief complaint
- Predicts ESI triage level with confidence score
- Shows clinical reasoning and feature contributions
- Includes appropriate clinical disclaimers

## How to Run

### Kaggle Notebook
1. Go to [Triagegeist Competition](https://kaggle.com/competitions/triagegeist)
2. Create a new notebook
3. Copy contents of `triagegeist_notebook.py`
4. Run all cells

### Local
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
python triagegeist_notebook.py
```

## Clinical Context

The Emergency Severity Index (ESI) is a 5-level triage system:
- **ESI-1**: Resuscitation (immediate life-saving intervention)
- **ESI-2**: Emergent (high-risk, altered consciousness)
- **ESI-3**: Urgent (stable, 2+ resources needed)
- **ESI-4**: Less urgent (1 resource needed)
- **ESI-5**: Non-urgent (no resources needed)

## Limitations

- Trained on synthetic data; requires prospective clinical validation
- Keyword-based NLP; transformer models may improve performance
- ESI-1 undertriage rate needs targeted improvement before deployment
- Model should augment, not replace, clinical judgment

## Citation

```
@misc{triagegeist2026,
  title={Triagegeist: AI-Powered Emergency Triage Acuity Prediction},
  year={2026},
  note={Kaggle Competition Submission}
}
```

## License

This project is submitted under the Triagegeist competition rules. Code is available for research and educational purposes.
