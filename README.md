# Monetary Policy Transmission Analytics Platform - Nigeria

> **A production-quality econometric platform analyzing how Central Bank of Nigeria (CBN) monetary policy shocks transmit to inflation through exchange rate and money supply channels.**

---

## 🎯 Project Overview

This platform models the **transmission mechanism** of monetary policy in Nigeria using advanced time series econometrics. It answers the research question:

**"How do changes in the Monetary Policy Rate (MPR) affect inflation in Nigeria, and through which channels?"**

### Key Features

- **ARDL (Autoregressive Distributed Lag)** modeling for short-run and long-run dynamics
- **VAR (Vector Autoregression)** with structural identification via Cholesky decomposition
- **Impulse Response Functions (IRF)** showing dynamic effects of policy shocks
- **Forecast Error Variance Decomposition (FEVD)** quantifying channel contributions
- **Policy simulation** framework for scenario analysis (+100bps MPR shock)
- **REST API** (Spring Boot) for programmatic access to results

### Academic & Professional Applications

✅ **MSc Thesis-Ready**: Complete methodology, interpretation, and literature grounding
✅ **Job Interview Portfolio**: Demonstrates econometrics, Python, Java, and system design
✅ **Policy Research**: Actionable insights for central bank decision-making

---

## 📊 Methodology

### Variables (Monthly Frequency, 2010-2025)

| Variable | Description | Source | Unit |
|----------|-------------|--------|------|
| **MPR** | Monetary Policy Rate | CBN | % per annum |
| **Inflation** | Headline CPI (YoY) | NBS/CBN | % |
| **Exchange Rate** | NGN/USD (NAFEX) | CBN | Naira per Dollar |
| **M2** | Broad Money Supply | CBN | ₦ Billions |

### Econometric Models

#### 1. ARDL (Bounds Testing Approach)
- Tests for long-run cointegration relationships
- Estimates short-run error correction dynamics
- Robust to mixed I(0)/I(1) variables

#### 2. VAR (Vector Autoregression)
- **Identification**: Cholesky decomposition with ordering:
  ```
  MPR → Exchange Rate → Money Supply (M2) → Inflation
  ```
- **Rationale**:
  - MPR is exogenous (CBN policy decision)
  - Exchange rate responds immediately via interest rate parity
  - M2 adjusts through banking system (lagged)
  - Inflation responds slowest (pass-through lags)

#### 3. Impulse Response Functions
- Track dynamic response of inflation to 1% MPR shock over 24 months
- Confidence intervals via bootstrap (500 replications)

#### 4. Policy Simulations
- Counterfactual scenarios: +100bps MPR increase
- Quantifies expected inflation impact

---

## 🗂️ Project Structure

```
monetary-policy-platform/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Version control exclusions
│
├── data/
│   ├── raw/                          # Original CBN/NBS data (CSV/Excel)
│   ├── processed/                    # Cleaned, analysis-ready datasets
│   └── metadata.json                 # Data dictionary & sources
│
├── src/
│   ├── data_ingestion/
│   │   ├── data_loader.py           # Load & preprocess raw data
│   │   └── data_validator.py        # Quality checks & validation
│   ├── econometrics/
│   │   ├── stationarity_tests.py    # ADF, PP, KPSS tests
│   │   ├── ardl_model.py            # ARDL estimation
│   │   ├── var_model.py             # VAR estimation & diagnostics
│   │   └── simulations.py           # Policy shock scenarios
│   ├── visualization/
│   │   └── plots.py                 # Time series, IRF, FEVD plots
│   └── utils/
│       └── config.py                # Global settings
│
├── api/                              # Java Spring Boot REST API
│   └── (Spring Boot project structure)
│
├── results/
│   ├── stationarity/                # Unit root test outputs
│   ├── ardl/                        # ARDL estimation results
│   ├── var/                         # VAR, IRF, FEVD outputs
│   └── simulations/                 # Policy scenario results
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Initial EDA
│
└── docs/
    ├── methodology.md               # Detailed econometric theory
    ├── identification_strategy.md   # VAR ordering justification
    └── interpretation.md            # Nigerian economic context
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd Monetary-Policy-Transmission-Analytics-Platform

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition

**Option A: Download from CBN**
- Visit: https://statistics.cbn.gov.ng/
- Download monthly data for MPR, Inflation, Exchange Rate, M2 (2010-2025)
- Place CSV in `data/raw/nigeria_macro_data.csv`

**Option B: Use Template Data**
- Template data already included in `data/raw/`
- Based on actual CBN trends (for demonstration/testing)

### 3. Load and Validate Data

```bash
cd src/data_ingestion
python data_loader.py
```

Expected output:
```
✓ Loaded 181 observations from 2010-01 to 2025-01
✓ No missing values detected
✓ Saved processed data to data/processed/
```

### 4. Exploratory Analysis

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 📈 Expected Results (To Be Generated)

### Week 2 Deliverables
- ✅ Stationarity test results (ADF, PP, KPSS)
- ✅ ARDL model estimation with bounds test
- ✅ Long-run coefficients and ECM term

### Week 3 Deliverables
- ✅ VAR model estimation (optimal lag selection via AIC/BIC)
- ✅ Impulse Response Functions (12-24 month horizon)
- ✅ Forecast Error Variance Decomposition

### Week 4 Deliverables
- ✅ Policy shock simulations (+100bps MPR)
- ✅ Interpretation of results in Nigerian context
- ✅ REST API for model access

---

## 🎓 Academic Rigor

### Literature Foundation
- **Sims (1980)**: Macroeconomics and Reality (VAR methodology)
- **Pesaran et al. (2001)**: Bounds testing approach to ARDL
- **Christiano, Eichenbaum, Evans (1999)**: Monetary policy shocks identification
- **Nigerian Context**: CBN Working Papers on monetary policy transmission

### Thesis Defense Points
1. **Why ARDL?** → Handles mixed integration orders, small sample performance
2. **Why VAR ordering?** → Based on economic theory (policy exogeneity, pass-through lags)
3. **Why monthly data?** → Captures transmission lags without high-frequency noise
4. **Why 2010-2025?** → Covers multiple policy cycles, structural breaks (2016, 2020, 2023)

---

## 🛠️ Tech Stack

| Component | Technology | Justification |
|-----------|-----------|---------------|
| **Econometrics** | Python + statsmodels | Industry standard for time series analysis |
| **Data Processing** | Pandas, NumPy | Efficient, widely adopted |
| **Visualization** | Matplotlib, Seaborn | Publication-quality plots |
| **API** | Java 17 + Spring Boot | Enterprise-grade, scalable |
| **Database** | PostgreSQL | Relational storage for time series |
| **Version Control** | Git | Reproducibility |

---

## 📅 Development Timeline

- **Week 1** (Days 1-5): Data acquisition, validation, stationarity tests
- **Week 2** (Days 6-10): ARDL modeling and cointegration analysis
- **Week 3** (Days 11-17): VAR estimation, IRF, FEVD
- **Week 4** (Days 18-24): Policy simulations and interpretation
- **Week 5** (Days 25-30): API development, documentation, finalization

---

## 🤝 Contributing

This is an academic/portfolio project. For questions or collaboration:
- Open an issue for bugs/suggestions
- Follow academic citation standards when using results

---

## 📜 License

This project is for educational and research purposes. Data sources retain their original licenses (CBN, NBS).

---

## 📧 Contact

**Author**: [Your Name]
**Institution**: [University Name]
**Purpose**: MSc Thesis - Monetary Policy Transmission in Nigeria
**Date**: February 2025

---

## 🔗 References

- Central Bank of Nigeria (2024). *Statistical Bulletin*. Abuja: CBN.
- National Bureau of Statistics (2024). *Consumer Price Index Reports*. Abuja: NBS.
- Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*, 16(3), 289-326.
- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 1-48.

---

**Status**: Week 1 - Day 1 ✅ Complete
**Next**: Day 2 - Data cleaning and exploratory time series analysis
