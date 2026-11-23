# Presentation Outline: State-Wise Annual Rainfall Analysis & Prediction in India
**Civil Engineering Undergraduate Project Presentation**
**Time Allocation:** 10 to 15 Minutes

---

## Slide 1: Title & Author
- **Title:** State-Wise Annual Rainfall Analysis and Prediction in India Using Machine Learning
- **Discipline:** Department of Civil Engineering (Water Resources & Hydrology)
- **Student Name & Roll Number:** LALIT CHOUDHARY
- **Guide / Mentor:** Dr. Kesheo Prasad

---

## Slide 2: Motivation & Problem Statement
- **Civil Engineering Relevance:** Water resources infrastructure (dams, canals, storm drainage) relies on reliable precipitation data.
- **Key Question:** Can basic machine learning algorithms forecast future annual rainfall using historical observation records?

---

## Slide 3: Hydrology Concepts
- **Precipitation & Rainfall:** Definitions and physical units ($1\,\text{mm} = 1\,\text{L/m}^2$).
- **Rainfall Normal:** IMD/WMO standard 30-year climatological baseline.
- **Southwest Monsoon:** June–September delivers $>75\%$ of India's annual water supply.

---

## Slide 4: Datasets Inspected & Acquired
- **Dataset 1 (`district wise rainfall normal.csv`):** 641 districts; static normals; no `YEAR` column (cannot train time series alone).
- **Dataset 2 (`rainfall in india 1901-2015.csv`):** IMD official historical record; 115 continuous years; 36 meteorological subdivisions.

---

## Slide 5: Data Cleaning & Preprocessing
- **Missing Value Imputation:** Subdivision-wise monthly median imputation.
- **Physical Sum Check:** $\text{ANNUAL} = \sum \text{JAN} \dots \text{DEC}$.
- **Subdivision-to-State Mapping:** Aggregated 36 IMD subdivisions into Indian States.

---

## Slide 6: Exploratory Data Analysis (EDA) — Key Insights
- **Spatial Disparity:** Arunachal Pradesh ($3,435\,\text{mm}$) vs. Rajasthan ($474\,\text{mm}$) — a $7\times$ difference justifying inter-basin water transfer.
- **Temporal Extremes:** Historic national flood year ($1961: 1,824.5\,\text{mm}$) vs. severe drought year ($2009: 1,258.9\,\text{mm}$).
- **Seasonality Heatmap:** Monsoon concentration between June and September.

---

## Slide 7: Machine Learning Methodology
- **Input Feature:** `YEAR` | **Target:** `ANNUAL` rainfall ($\text{mm}$).
- **Train/Test Splitting:** Strict Chronological Split (Train: 1901–1992, Test: 1993–2015). No random shuffle to eliminate lookahead bias.
- **Algorithms Evaluated:**
  1. Historical Mean Baseline (Climatological Normal)
  2. Linear Regression (Trend line $y = mx + c$)
  3. Decision Tree Regressor (Recursive interval partitioning)
  4. Random Forest Regressor (Ensemble of 100 decorrelated trees)

---

## Slide 8: Results & Model Comparison
- **Evaluation Metrics Table (Maharashtra Test Data):**
  - Baseline: $\text{RMSE} = 131.90\,\text{mm}$, $\text{MAE} = 114.52\,\text{mm}$ (Rank 1).
  - Random Forest: $\text{RMSE} = 132.92\,\text{mm}$ (Rank 2).
  - Linear Regression: $\text{RMSE} = 139.00\,\text{mm}$ (Rank 3).
  - Decision Tree: $\text{RMSE} = 157.54\,\text{mm}$ (Rank 4).
- **Why Decision Tree Failed:** Cannot extrapolate trends outside the training interval.

---

## Slide 9: Interactive Decision Support System
- Built inside Jupyter Notebook using **`ipywidgets`**.
- Features: State dropdown, year input, model selection, live plot generation, automated horizon extrapolation warnings ($>2025$), and empirical $\pm 1\sigma$ uncertainty bounds.

---

## Slide 10: Civil Engineering Applications
- **Reservoir Sizing:** Active storage design via Rippl's Mass Curve method.
- **Spillway Safety:** Maximum probable flood discharge calculations based on historical extremes.
- **Irrigation Scheduling:** Canal delivery allocations based on monsoon deficit forecasts.

---

## Slide 11: Limitations & Future Scope
- **Limitation:** Temporal index (`YEAR`) alone lacks physical ocean-atmospheric variables.
- **Future Scope:** Integrating sea-surface temperatures (ENSO/IOD), pre-monsoon pressure gradients, and regional climate model downscaling.

---

## Slide 12: Conclusion & References
- Summary of findings.
- Acknowledgments and official citations (IMD, CWC, WMO).
- **Q&A / Viva Discussion.**

