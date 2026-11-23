# State-Wise Annual Rainfall Analysis and Prediction in India Using Machine Learning
**A Civil Engineering Undergraduate Project Report**

---

## 1. Introduction
Water is the lifeblood of civil engineering. Whether designing a concrete gravity dam, sizing a storm-water drainage network, planning an irrigation canal, or managing flood risk in urban centers, the primary input parameter is **precipitation**. 

In India, rainfall is heavily dominated by the **Southwest Monsoon (June–September)**, which exhibits tremendous spatial and temporal variability. With growing population pressure and climate change, understanding long-term rainfall trends and developing predictive capabilities is critical for sustainable water resources management.

This project integrates classical **Engineering Hydrology** with **Machine Learning (ML)** to evaluate 115 years ($1901–2015$) of historical rainfall records from the **India Meteorological Department (IMD)**.

---

## 2. Problem Statement
Civil engineers frequently face the challenge of sizing water storage structures for future decades without knowing whether future annual rainfall will increase, decrease, or remain stable around historical norms.

The core research question of this study is:
> **"Can historical rainfall data be used to estimate the annual rainfall of an Indian state in a future year using basic Machine Learning algorithms?"**

---

## 3. Objectives
1. Inspect, clean, and structure historical rainfall data across Indian states without introducing lookahead bias or data leakage.
2. Conduct comprehensive **Exploratory Data Analysis (EDA)** to characterize long-term rainfall trends, spatial disparity, seasonal concentration, and historical climatic extremes (e.g., 1961 flood peak and 2009 drought dip).
3. Implement and benchmark three standard regression algorithms (**Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**) against a **Historical Mean Baseline** using strict chronological train/test splitting.
4. Evaluate model performance using standard hydrological metrics: **MAE**, **MSE**, **RMSE**, and **$R^2$ Score**.
5. Develop an interactive decision-support prediction tool inside Jupyter Notebook using **`ipywidgets`** with built-in forecast horizon warnings and empirical fluctuation bounds.

---

## 4. Literature Review & Hydrological Foundation
- **Rainfall Normal:** Defined by the World Meteorological Organization (WMO) and India Meteorological Department (IMD) as the arithmetic mean of rainfall over a standard 30-year or 50-year benchmark period (e.g., 1951–2000). It serves as the baseline to categorize rainfall as *excess* ($>+19\%$), *normal* ($-19\%$ to $+19\%$), *deficient* ($-20\%$ to $-59\%$), or *scanty* ($\le -60\%$).
- **Orographic Lift & Spatial Disparity:** Moister-laden winds striking physical barriers (such as the Western Ghats or the Eastern Himalayas) rise, cool adiabatically, and precipitate heavily, creating rainforest climates on the windward side and semi-arid rain-shadow zones on the leeward side.
- **Atmospheric Stochasticity:** Indian monsoon rainfall is driven by coupled ocean-atmosphere interactions, notably the **El Niño Southern Oscillation (ENSO)** in the Pacific Ocean and the **Indian Ocean Dipole (IOD)** in the Indian Ocean. Because these phenomena are governed by complex fluid dynamics and thermodynamics, calendar year alone carries high unexplained variance.

---

## 5. Dataset Description & Limitations

### 5.1 District-Wise Rainfall Normals (`district wise rainfall normal.csv`)
- **Structure:** 641 rows (districts), 19 columns.
- **Key Observation:** Contains 30-year static baseline averages (`ANNUAL`, monthly, and seasonal normals).
- **Limitation:** Contains **no `YEAR` column**. It represents a single static spatial snapshot and cannot be used alone for time-series forecasting.

### 5.2 IMD 115-Year Historical Time Series (`rainfall in india 1901-2015.csv`)
- **Source:** India Meteorological Department (IMD), Ministry of Earth Sciences, Government of India.
- **Time Horizon:** 115 continuous years (1901 to 2015).
- **Scope:** 36 meteorological subdivisions spanning 4,116 annual records.
- **Units:** Millimetres ($\text{mm}$).
- **Limitations:** Subdivisions group several smaller states or split large states; remote rain gauges in early 20th-century records have occasional missing values.

---

## 6. Methodology

```
+-----------------------------------+
| Raw IMD Historical Data (1901-2015)|
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
|  Phase 2: Hydrological Cleaning   |
| - Subdivision-wise Median Impute  |
| - Physical Sum: Annual = Sum(M)   |
| - State Aggregation Mapping       |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
|  Phase 3: Exploratory Data Analysis|
| - Spatial Bar Charts & Heatmaps   |
| - 5-Year Moving Average Trends    |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
|   Phase 4: Machine Learning       |
| - Chronological Split (1901-1992) |
| - Unseen Test Set (1993-2015)     |
| - Baseline, Linear, Tree, Forest  |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
|  Phase 5 & 6: Evaluation & UI     |
| - Metrics: MAE, MSE, RMSE, R²     |
| - Interactive ipywidgets System   |
+-----------------+-----------------+
```

---

## 7. Results and Discussion

### Model Evaluation on Unseen Test Horizon (1993–2015) — State: Maharashtra

| Model Name | MAE ($\text{mm}$) | MSE ($\text{mm}^2$) | RMSE ($\text{mm}$) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Historical Mean Baseline** | **$114.52$** | **$17,397.61$** | **$131.90$** | **$-0.0020$** | **1 (Best Baseline)** |
| **Random Forest Regressor** | $115.98$ | $17,667.72$ | $132.92$ | $-0.0177$ | **2 (Best ML Model)** |
| **Linear Regression** | $121.16$ | $19,321.00$ | $139.00$ | $-0.1127$ | **3** |
| **Decision Tree Regressor** | $140.59$ | $24,818.84$ | $157.54$ | $-0.4296$ | **4 (Worst)** |

### Key Scientific Insights:
1. **The Historical Mean Baseline achieved the lowest MAE and RMSE.** This validates the classical civil engineering practice of relying on long-term climatological normals for water resource planning.
2. **Decision Tree Regressor failed on future years** ($\text{RMSE} = 157.54\text{ mm}$, $R^2 = -0.4296$) because decision trees partition training space and cannot extrapolate beyond observed feature boundaries.
3. **Random Forest performed best among ML models** ($\text{RMSE} = 132.92\text{ mm}$) because ensembling 100 decorrelated trees prevented erratic step predictions.
4. **$R^2$ scores were near zero**, proving that calendar year alone lacks atmospheric physics (ENSO, sea-surface temperatures) and cannot deterministically predict individual year rainfall peaks.

---

## 8. Civil Engineering Applications

1. **Dam & Reservoir Capacity Design:** Knowing the mean annual runoff and empirical standard deviation ($\pm 1\sigma$) allows engineers to compute **Mass Inflow Curves (Rippl's Method)** to size reservoir storage required to meet dry-season water demands.
2. **Spillway Design & Flood Management:** Identifying historic rainfall peaks (such as 1961) informs maximum probable flood calculations to prevent catastrophic dam overtopping.
3. **Irrigation Planning & Canal Networks:** Quantifying seasonal deficits in states like Rajasthan ($474\text{ mm}$) and Punjab ($594\text{ mm}$) dictates canal delivery scheduling and groundwater recharge requirements.
4. **Inter-Basin Water Transfers (National River Linking Project):** The 7-fold disparity between wet northeastern basins ($>3400\text{ mm}$) and dry western basins ($<500\text{ mm}$) provides the physical justification for inter-basin water transfers.

---

## 9. Conclusion
- Historical rainfall data demonstrates strong seasonal concentration ($>75\%$ during monsoon) and vast spatial disparity across India.
- Basic machine learning regression algorithms cannot reliably forecast exact rainfall amounts decades in advance using calendar year alone due to high natural stochasticity.
- For near-term civil engineering planning, the **long-term historical normal (baseline)** and **Random Forest ensemble** provide the most mathematically stable, reliable estimates.
- Long-term multi-decadal infrastructure planning requires dynamic, physically coupled climate models (GCMs) rather than purely statistical regression.

---

## 10. References
1. India Meteorological Department (IMD), Ministry of Earth Sciences, Government of India. *Customized Rainfall Information System (CRIS) & Historical Archives (1901–2015)*.
2. Subramanya, K. (2013). *Engineering Hydrology* (4th ed.). McGraw Hill Education.
3. World Meteorological Organization (WMO). (2017). *WMO Guidelines on the Calculation of Climate Normals* (WMO-No. 1203).
4. Central Water Commission (CWC), Government of India. *National Water Development Agency Inter-Basin Water Transfer Studies*.

