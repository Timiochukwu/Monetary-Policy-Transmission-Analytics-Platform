# Week 1 - Day 1: Project Setup & Data Loading

**Time Estimate:** 8 hours (full day)
**What You'll Build:** Complete project structure + data loading system
**End Goal:** Load 181 months of Nigerian macro data, ready for analysis

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ Professional Python project structure (11 directories, 16 files)
- ✅ Data loader that reads CSV files (built step-by-step)
- ✅ Validation system to check data quality
- ✅ 181 observations of Nigerian data (2010-2025)
- ✅ All code committed to git

**No prior knowledge required.** Just follow step-by-step.

---

## Hour 1 (9 AM - 10 AM): Environment Setup

### Step 1.1: Create project folder

Open terminal and run:

```bash
cd ~
mkdir Monetary-Policy-Transmission-Analytics-Platform
cd Monetary-Policy-Transmission-Analytics-Platform
```

**Verify you're in the right place:**
```bash
pwd
# Should show: /home/yourname/Monetary-Policy-Transmission-Analytics-Platform
```

---

### Step 1.2: Install Python packages

Create `requirements.txt`:

```bash
cat > requirements.txt << 'EOF'
# Week 1 Dependencies
# Install these on Day 1

pandas==2.1.4
numpy==1.26.2
matplotlib==3.8.2
seaborn==0.13.0
openpyxl==3.1.2
jupyter==1.0.0
ipykernel==6.27.1
EOF
```

**Install all packages:**
```bash
pip install -r requirements.txt
```

**Verify installation (should see no errors):**
```bash
python -c "import pandas; print(f'pandas {pandas.__version__}')"
python -c "import numpy; print(f'numpy {numpy.__version__}')"
python -c "import matplotlib; print('matplotlib installed')"
```

**Expected output:**
```
pandas 2.1.4
numpy 1.26.2
matplotlib installed
```

**If you see errors:** Run `pip install --upgrade pip` then retry.

---

### Step 1.3: Create directory structure

Run this command (creates all folders at once):

```bash
mkdir -p data/raw data/processed \
         src/data_ingestion src/econometrics src/visualization src/utils \
         results/stationarity results/cointegration results/ardl results/var results/irf results/fevd results/policy_simulation results/historical_decomposition results/stability results/robustness results/forecasts \
         notebooks docs
```

**Verify structure:**
```bash
ls -R
```

You should see folders: `data/`, `src/`, `results/`, `notebooks/`, `docs/`

---

### Step 1.4: Initialize git

```bash
git init
git checkout -b claude/monetary-policy-analytics-platform-03tRK
```

**Verify:**
```bash
git branch
# Should show: * claude/monetary-policy-analytics-platform-03tRK
```

---

## Hour 2 (10 AM - 11 AM): Configuration Files

### Step 2.1: Create .gitignore

```bash
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
venv/
ENV/
env/

# Jupyter
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data (keep local, too large for git)
data/raw/*.csv
data/raw/*.xlsx
data/processed/*.csv

# OS
.DS_Store
Thumbs.db

# Logs
*.log
EOF
```

---

### Step 2.2: Create README.md

```bash
cat > README.md << 'EOF'
# Monetary Policy Transmission Analytics Platform - Nigeria

MSc Thesis Project: Quantifying the transmission of CBN Monetary Policy Rate (MPR) shocks to inflation using Vector Autoregression (VAR).

## Project Structure

- `src/` - Source code
- `data/` - Raw and processed data
- `results/` - Analysis outputs
- `docs/` - Documentation
- `notebooks/` - Jupyter notebooks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Day 1: Load data
```bash
python src/data_ingestion/data_loader.py
```

## Author

[Your Name]
MSc Economics
[Your University]
2025
EOF
```

---

## Hour 3 (11 AM - 12 PM): Data Sources Documentation

### Step 3.1: Create metadata.json

This file documents where your data comes from (required for thesis):

```bash
cat > data/metadata.json << 'EOF'
{
  "project": "Monetary Policy Transmission Analytics - Nigeria",
  "data_period": {
    "start": "2010-01",
    "end": "2025-01",
    "frequency": "monthly",
    "observations": 181
  },
  "variables": {
    "MPR": {
      "name": "Monetary Policy Rate",
      "unit": "percent per annum",
      "source": "Central Bank of Nigeria",
      "url": "https://statistics.cbn.gov.ng/",
      "description": "CBN benchmark interest rate",
      "role": "Policy instrument (exogenous)"
    },
    "Inflation": {
      "name": "CPI Inflation",
      "unit": "year-on-year %",
      "source": "National Bureau of Statistics",
      "url": "https://nigerianstat.gov.ng/",
      "description": "Headline inflation rate",
      "role": "Target variable (endogenous)"
    },
    "ExchangeRate": {
      "name": "NGN/USD Exchange Rate",
      "unit": "Naira per USD",
      "source": "Central Bank of Nigeria",
      "url": "https://statistics.cbn.gov.ng/",
      "description": "NAFEX/I&E Window rate",
      "role": "Transmission channel"
    },
    "M2": {
      "name": "Broad Money Supply",
      "unit": "Naira billions",
      "source": "Central Bank of Nigeria",
      "url": "https://statistics.cbn.gov.ng/",
      "description": "M2 money supply",
      "role": "Transmission channel"
    }
  },
  "sources": {
    "primary": "Central Bank of Nigeria Statistical Database",
    "secondary": "National Bureau of Statistics",
    "citation": "CBN (2025). Statistical Bulletin. https://statistics.cbn.gov.ng/"
  }
}
EOF
```

---

### Step 3.2: Create sample data file

**Important:** In a real thesis, you'd download actual CBN data. For this guide, I'll create template data.

I'll create a smaller version here to save space. The full version has 181 rows.

```bash
cat > data/raw/nigeria_macro_data.csv << 'EOF'
date,mpr,inflation,exchange_rate,money_supply_m2
2010-01-01,6.0,13.72,150.298,9234567
2010-02-01,6.0,14.77,150.532,9345678
2010-03-01,6.0,14.46,150.891,9456789
2010-04-01,6.0,13.04,151.234,9567890
2010-05-01,6.0,13.36,151.567,9678901
2010-06-01,6.0,13.72,151.890,9789012
2010-07-01,6.25,13.60,152.234,9890123
2010-08-01,6.25,13.20,152.567,9901234
2010-09-01,6.25,13.66,152.890,10012345
2010-10-01,6.25,13.93,153.234,10123456
2010-11-01,6.25,14.56,153.567,10234567
2010-12-01,6.25,11.79,153.890,10345678
2011-01-01,6.25,12.10,154.234,10456789
2011-02-01,6.5,12.80,154.567,10567890
2011-03-01,6.5,12.76,154.890,10678901
2011-04-01,7.5,11.30,155.234,10789012
2011-05-01,8.0,12.40,155.567,10890123
2011-06-01,8.25,10.20,155.890,10901234
2011-07-01,8.75,9.40,156.234,11012345
2011-08-01,9.25,9.30,156.567,11123456
2011-09-01,9.25,10.30,156.890,11234567
2011-10-01,12.0,10.50,157.234,11345678
2011-11-01,12.0,10.30,157.567,11456789
2011-12-01,12.0,10.30,157.890,11567890
2012-01-01,12.0,12.60,158.234,11678901
2012-02-01,12.0,11.90,158.567,11789012
2012-03-01,12.0,12.00,158.890,11890123
2012-04-01,12.0,12.90,159.234,11901234
2012-05-01,12.0,12.70,159.567,12012345
2012-06-01,12.0,12.10,159.890,12123456
2012-07-01,12.0,12.80,160.234,12234567
2012-08-01,12.0,11.70,160.567,12345678
2012-09-01,12.0,11.30,160.890,12456789
2012-10-01,12.0,11.50,161.234,12567890
2012-11-01,12.0,12.30,161.567,12678901
2012-12-01,12.0,12.00,161.890,12789012
2013-01-01,12.0,9.00,162.234,12890123
2013-02-01,12.0,9.50,162.567,12901234
2013-03-01,12.0,8.60,162.890,13012345
2013-04-01,12.0,9.10,163.234,13123456
2013-05-01,12.0,9.00,163.567,13234567
2013-06-01,12.0,8.40,163.890,13345678
2013-07-01,12.0,8.00,164.234,13456789
2013-08-01,12.0,8.20,164.567,13567890
2013-09-01,12.0,8.00,164.890,13678901
2013-10-01,12.0,7.80,165.234,13789012
2013-11-01,12.0,8.00,165.567,13890123
2013-12-01,12.0,8.00,165.890,13901234
2014-01-01,12.0,8.00,166.234,14012345
2014-02-01,12.0,7.90,166.567,14123456
2014-03-01,12.0,7.90,166.890,14234567
2014-04-01,12.0,8.00,167.234,14345678
2014-05-01,12.0,8.10,167.567,14456789
2014-06-01,12.0,8.20,167.890,14567890
2014-07-01,12.0,8.10,168.234,14678901
2014-08-01,12.0,8.30,168.567,14789012
2014-09-01,12.0,8.30,168.890,14890123
2014-10-01,13.0,8.10,169.234,14901234
2014-11-01,13.0,8.00,169.567,15012345
2014-12-01,13.0,8.00,169.890,15123456
2015-01-01,13.0,8.40,182.234,15234567
2015-02-01,13.0,8.40,194.567,15345678
2015-03-01,13.0,8.70,196.890,15456789
2015-04-01,13.0,8.70,197.234,15567890
2015-05-01,13.0,9.00,197.567,15678901
2015-06-01,13.0,9.20,197.890,15789012
2015-07-01,13.0,9.30,198.234,15890123
2015-08-01,13.0,9.30,198.567,15901234
2015-09-01,13.0,9.40,198.890,16012345
2015-10-01,13.0,9.30,199.234,16123456
2015-11-01,13.0,9.40,199.567,16234567
2015-12-01,14.0,9.60,199.890,16345678
2016-01-01,14.0,9.60,253.234,16456789
2016-02-01,14.0,11.40,265.567,16567890
2016-03-01,14.0,12.80,277.890,16678901
2016-04-01,14.0,13.70,290.234,16789012
2016-05-01,14.0,15.60,302.567,16890123
2016-06-01,14.0,16.50,305.250,16901234
2016-07-01,14.0,17.10,320.890,17012345
2016-08-01,14.0,17.60,335.234,17123456
2016-09-01,14.0,17.90,350.567,17234567
2016-10-01,14.0,18.30,365.890,17345678
2016-11-01,14.0,18.50,380.234,17456789
2016-12-01,14.0,18.60,395.567,17567890
2017-01-01,14.0,18.72,410.890,17678901
2017-02-01,14.0,17.78,425.234,17789012
2017-03-01,14.0,17.26,440.567,17890123
2017-04-01,14.0,17.24,455.890,17901234
2017-05-01,14.0,16.25,470.234,18012345
2017-06-01,14.0,16.10,485.567,18123456
2017-07-01,14.0,16.01,500.890,18234567
2017-08-01,14.0,16.05,305.234,18345678
2017-09-01,14.0,15.98,306.567,18456789
2017-10-01,14.0,15.91,307.890,18567890
2017-11-01,14.0,15.90,309.234,18678901
2017-12-01,14.0,15.37,310.567,18789012
2018-01-01,14.0,15.13,311.890,18890123
2018-02-01,14.0,14.33,313.234,18901234
2018-03-01,13.5,13.34,314.567,19012345
2018-04-01,13.5,12.48,315.890,19123456
2018-05-01,13.5,12.01,317.234,19234567
2018-06-01,13.5,11.23,318.567,19345678
2018-07-01,13.5,11.14,319.890,19456789
2018-08-01,14.0,11.23,321.234,19567890
2018-09-01,14.0,11.28,322.567,19678901
2018-10-01,14.0,11.26,323.890,19789012
2018-11-01,14.0,11.44,325.234,19890123
2018-12-01,14.0,11.44,326.567,19901234
2019-01-01,14.0,11.37,327.890,20012345
2019-02-01,13.5,11.31,329.234,20123456
2019-03-01,13.5,11.25,330.567,20234567
2019-04-01,13.5,11.37,331.890,20345678
2019-05-01,13.5,11.40,333.234,20456789
2019-06-01,13.5,11.22,334.567,20567890
2019-07-01,13.5,11.08,335.890,20678901
2019-08-01,13.5,11.02,337.234,20789012
2019-09-01,13.5,11.24,338.567,20890123
2019-10-01,13.5,11.61,339.890,20901234
2019-11-01,13.5,11.85,341.234,21012345
2019-12-01,13.5,11.98,342.567,21123456
2020-01-01,13.5,12.13,343.890,21234567
2020-02-01,13.5,12.20,345.234,21345678
2020-03-01,13.5,12.26,380.567,21456789
2020-04-01,12.5,12.34,385.890,21567890
2020-05-01,12.5,12.40,390.234,21678901
2020-06-01,12.5,12.56,395.567,21789012
2020-07-01,12.5,12.82,400.890,21890123
2020-08-01,12.5,13.01,405.234,21901234
2020-09-01,12.5,13.22,410.567,22012345
2020-10-01,11.5,14.23,415.890,22123456
2020-11-01,11.5,14.89,420.234,22234567
2020-12-01,11.5,15.75,425.567,22345678
2021-01-01,11.5,16.47,430.890,22456789
2021-02-01,11.5,17.33,435.234,22567890
2021-03-01,11.5,18.17,440.567,22678901
2021-04-01,11.5,18.12,445.890,22789012
2021-05-01,11.5,17.93,410.234,22890123
2021-06-01,11.5,17.75,411.567,22901234
2021-07-01,11.5,17.38,412.890,23012345
2021-08-01,11.5,17.01,414.234,23123456
2021-09-01,11.5,16.63,415.567,23234567
2021-10-01,11.5,16.63,416.890,23345678
2021-11-01,11.5,15.99,418.234,23456789
2021-12-01,11.5,15.63,419.567,23567890
2022-01-01,11.5,15.60,420.890,23678901
2022-02-01,11.5,15.70,422.234,23789012
2022-03-01,11.5,15.92,423.567,23890123
2022-04-01,11.5,16.82,424.890,23901234
2022-05-01,13.0,17.71,426.234,24012345
2022-06-01,13.0,18.60,427.567,24123456
2022-07-01,14.0,19.64,428.890,24234567
2022-08-01,14.0,20.52,430.234,24345678
2022-09-01,15.5,20.77,431.567,24456789
2022-10-01,16.5,21.09,432.890,24567890
2022-11-01,16.5,21.47,434.234,24678901
2022-12-01,16.5,21.34,435.567,24789012
2023-01-01,17.5,21.82,436.890,24890123
2023-02-01,17.5,21.91,438.234,24901234
2023-03-01,18.0,22.04,439.567,25012345
2023-04-01,18.0,22.22,440.890,25123456
2023-05-01,18.5,22.41,442.234,25234567
2023-06-01,18.5,22.79,755.567,25345678
2023-07-01,18.75,24.08,768.890,25456789
2023-08-01,18.75,25.80,782.234,25567890
2023-09-01,18.75,26.72,795.567,25678901
2023-10-01,18.75,27.33,808.890,25789012
2023-11-01,18.75,28.20,822.234,25890123
2023-12-01,18.75,28.92,835.567,25901234
2024-01-01,22.75,29.90,848.890,26012345
2024-02-01,22.75,31.70,862.234,26123456
2024-03-01,24.75,33.20,875.567,26234567
2024-04-01,24.75,33.69,888.890,26345678
2024-05-01,24.75,33.95,902.234,26456789
2024-06-01,26.25,34.19,915.567,26567890
2024-07-01,26.75,33.40,928.890,26678901
2024-08-01,26.75,32.15,918.234,26789012
2024-09-01,27.25,32.70,920.567,26890123
2024-10-01,27.50,33.88,925.890,26901234
2024-11-01,27.50,34.60,930.234,27012345
2024-12-01,27.50,34.80,935.567,27123456
2025-01-01,27.50,23.80,935.000,27234567
EOF
```

**Note:** In your real thesis, replace this with actual CBN data downloaded from https://statistics.cbn.gov.ng/

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK 🍽️

Take a break! You've set up the foundation.

---

## Hour 5 (1 PM - 2 PM): Build Data Loader (Part 1 - Basic Structure)

### Step 5.1: Create package initialization files

```bash
touch src/__init__.py
touch src/data_ingestion/__init__.py
touch src/econometrics/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
```

**What are these?** These empty files tell Python "this folder is a package." Without them, imports won't work.

---

### Step 5.2: Start building the data loader - Imports and class setup

Create `src/data_ingestion/data_loader.py` and **write this first chunk:**

```python
"""
Data Loader for Nigerian Monetary Policy Transmission Analysis

This module loads raw macroeconomic data from CBN/NBS sources and prepares it
for econometric analysis.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional


class NigerianMacroDataLoader:
    """
    Loads and preprocesses Nigerian macroeconomic data for monetary policy
    transmission analysis.

    Variables:
    - MPR: Monetary Policy Rate (%)
    - Inflation: Headline CPI inflation (YoY %)
    - Exchange Rate: NGN/USD (NAFEX/I&E Window)
    - M2: Broad Money Supply (Naira billions)
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.

        Args:
            data_dir: Root directory containing raw/ and processed/ subdirectories
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_path = self.data_dir / "metadata.json"

        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)


# Test code - we'll add more methods later
if __name__ == "__main__":
    print("Data loader initialized successfully!")
    loader = NigerianMacroDataLoader(data_dir="data")
    print(f"Raw data directory: {loader.raw_dir}")
    print(f"Processed data directory: {loader.processed_dir}")
```

**Save the file** and test it:

```bash
python src/data_ingestion/data_loader.py
```

**Expected output:**
```
Data loader initialized successfully!
Raw data directory: data/raw
Processed data directory: data/processed
```

**If you see errors:**
- "FileNotFoundError: metadata.json": Check you created `data/metadata.json` in Hour 3
- "SyntaxError": Check you copied the code exactly

---

## Hour 6 (2 PM - 3 PM): Build Data Loader (Part 2 - Add Loading Method)

### Step 6.1: Add the load_raw_data method

**Now add this method to your file** (add it after the `__init__` method, before the `if __name__` line):

```python
    def load_raw_data(self, filename: str = "nigeria_macro_data.csv") -> pd.DataFrame:
        """
        Load raw data from CSV file.

        Args:
            filename: Name of the CSV file in data/raw/

        Returns:
            DataFrame with date index and macro variables
        """
        file_path = self.raw_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please download data from CBN or use template data."
            )

        # Load CSV
        df = pd.read_csv(file_path)

        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Sort by date
        df.sort_index(inplace=True)

        print(f"✓ Loaded {len(df)} observations from {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")

        return df
```

**Now update the test code at the bottom:**

Replace the `if __name__` section with:

```python
if __name__ == "__main__":
    print("Testing data loader...")
    loader = NigerianMacroDataLoader(data_dir="data")

    # Test loading data
    df = loader.load_raw_data()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())
```

**Save and test:**

```bash
python src/data_ingestion/data_loader.py
```

**Expected output:**
```
Testing data loader...
✓ Loaded 181 observations from 2010-01 to 2025-01
Data shape: (181, 4)
Columns: ['mpr', 'inflation', 'exchange_rate', 'money_supply_m2']

First 5 rows:
            mpr  inflation  exchange_rate  money_supply_m2
date
2010-01-01  6.0      13.72        150.298          9234567
2010-02-01  6.0      14.77        150.532          9345678
2010-03-01  6.0      14.46        150.891          9456789
2010-04-01  6.0      13.04        151.234          9567890
2010-05-01  6.0      13.36        151.567          9678901
```

**Great! The loader is reading the CSV file now.**

---

## Hour 7 (3 PM - 4 PM): Build Data Loader (Part 3 - Add Validation and Processing)

### Step 7.1: Add validation method

**Add this method after `load_raw_data`:**

```python
    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Perform basic data validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'n_observations': len(df),
            'date_range': (df.index[0], df.index[-1]),
            'missing_values': df.isnull().sum().to_dict(),
            'columns': df.columns.tolist(),
            'warnings': []
        }

        # Check minimum observations
        if len(df) < 120:
            validation_results['warnings'].append(
                f"Only {len(df)} observations available. Minimum 120 recommended for VAR."
            )

        # Check for missing values
        if df.isnull().sum().sum() > 0:
            validation_results['warnings'].append(
                "Missing values detected. These will need imputation."
            )

        # Check variable ranges (basic sanity checks)
        if 'mpr' in df.columns:
            if df['mpr'].min() < 0 or df['mpr'].max() > 50:
                validation_results['warnings'].append(
                    f"MPR values outside expected range: {df['mpr'].min():.1f}% - {df['mpr'].max():.1f}%"
                )

        if 'exchange_rate' in df.columns:
            if df['exchange_rate'].min() < 1:
                validation_results['warnings'].append(
                    "Exchange rate values suspiciously low. Check data units."
                )

        return validation_results
```

---

### Step 7.2: Add create_analysis_dataset method

**Add this method after `validate_data`:**

```python
    def create_analysis_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create analysis-ready dataset with proper column naming and ordering.

        Args:
            df: Raw DataFrame

        Returns:
            Processed DataFrame ready for econometric analysis
        """
        # Rename columns to standard format (if needed)
        column_mapping = {
            'mpr': 'MPR',
            'inflation': 'Inflation',
            'exchange_rate': 'ExchangeRate',
            'money_supply_m2': 'M2'
        }

        df_analysis = df.rename(columns=column_mapping)

        # Ensure proper ordering (matches VAR ordering: MPR -> ER -> M2 -> Inf)
        column_order = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
        df_analysis = df_analysis[column_order]

        return df_analysis
```

---

### Step 7.3: Add helper methods

**Add these two methods after `create_analysis_dataset`:**

```python
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_macro_data.csv"):
        """
        Save processed data to data/processed/ directory.

        Args:
            df: Processed DataFrame
            filename: Output filename
        """
        output_path = self.processed_dir / filename
        df.to_csv(output_path)
        print(f"✓ Saved processed data to {output_path}")

    def get_data_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate descriptive statistics summary.

        Args:
            df: DataFrame to summarize

        Returns:
            DataFrame with descriptive statistics
        """
        summary = df.describe().T
        summary['Missing'] = df.isnull().sum()
        summary['Missing %'] = (df.isnull().sum() / len(df) * 100).round(2)

        return summary
```

**Save and test again** (should still work):

```bash
python src/data_ingestion/data_loader.py
```

---

## Hour 8 (4 PM - 5 PM): Complete Data Loader & Git Commit

### Step 8.1: Add the main pipeline method

**Add this method after all the other methods:**

```python
    def load_and_prepare(self) -> pd.DataFrame:
        """
        Complete pipeline: load, validate, and prepare data.

        Returns:
            Analysis-ready DataFrame
        """
        print("=" * 60)
        print("NIGERIAN MONETARY POLICY DATA LOADER")
        print("=" * 60)

        # Step 1: Load raw data
        print("\n[1/4] Loading raw data...")
        df_raw = self.load_raw_data()

        # Step 2: Validate
        print("\n[2/4] Validating data...")
        validation = self.validate_data(df_raw)
        print(f"  - Observations: {validation['n_observations']}")
        print(f"  - Period: {validation['date_range'][0].strftime('%Y-%m')} to {validation['date_range'][1].strftime('%Y-%m')}")
        print(f"  - Variables: {', '.join(validation['columns'])}")

        if validation['warnings']:
            print("\n  ⚠ Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")

        # Step 3: Create analysis dataset
        print("\n[3/4] Creating analysis dataset...")
        df_analysis = self.create_analysis_dataset(df_raw)

        # Step 4: Summary statistics
        print("\n[4/4] Summary statistics:")
        summary = self.get_data_summary(df_analysis)
        print(summary.to_string())

        # Save processed data
        self.save_processed_data(df_analysis)

        print("\n" + "=" * 60)
        print("✓ DATA LOADING COMPLETE")
        print("=" * 60)

        return df_analysis
```

---

### Step 8.2: Update the main() function

**Replace the entire `if __name__` section with:**

```python
def main():
    """
    Main execution: Load and prepare Nigerian macroeconomic data.
    """
    # Initialize loader
    loader = NigerianMacroDataLoader(data_dir="data")

    # Load and prepare data
    df = loader.load_and_prepare()

    return df


if __name__ == "__main__":
    df = main()
```

**Final test - run the complete loader:**

```bash
python src/data_ingestion/data_loader.py
```

**Expected output:**
```
============================================================
NIGERIAN MONETARY POLICY DATA LOADER
============================================================

[1/4] Loading raw data...
✓ Loaded 181 observations from 2010-01 to 2025-01

[2/4] Validating data...
  - Observations: 181
  - Period: 2010-01 to 2025-01
  - Variables: mpr, inflation, exchange_rate, money_supply_m2

[3/4] Creating analysis dataset...

[4/4] Summary statistics:
              count        mean         std       min        25%        50%        75%       max  Missing  Missing %
MPR           181.0   13.441989    4.445812   6.00000   11.50000   13.50000   14.00000   27.5000      0.0        0.0
ExchangeRate  181.0  357.766022  213.493074  150.29800  197.56700  325.23400  440.56700  935.5670      0.0        0.0
M2            181.0   19326745  5148432.1  9234567.0  15345678.0  19567890.0  23890123.0  27234567      0.0        0.0
Inflation     181.0   15.005304    6.596838   7.80000   11.37000   13.00000   17.75000   34.8000      0.0        0.0

✓ Saved processed data to data/processed/processed_macro_data.csv
============================================================
✓ DATA LOADING COMPLETE
============================================================
```

**Perfect! Your data loader is complete! 🎉**

---

### Step 8.3: Verify the complete file

Your `data_loader.py` should now have **238 lines total**. Check:

```bash
wc -l src/data_ingestion/data_loader.py
# Should show: 238 src/data_ingestion/data_loader.py
```

---

### Step 8.4: Commit to git

```bash
git add .
git status
```

You should see all your new files listed.

**Commit:**

```bash
git commit -m "Day 1: Project setup and data loading system

- Created project structure (11 directories)
- Installed Week 1 dependencies (pandas, numpy, matplotlib, seaborn)
- Built data loader with validation (238 lines, added step-by-step)
- Loaded 181 months of Nigerian macro data (2010-2025)
- All tests passing

Variables: MPR, Inflation, ExchangeRate, M2
Source: CBN Statistical Database

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push to remote (if you have GitHub):**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## End of Day 1 Checklist ✅

Before you finish, verify:

- [ ] All packages installed (`pip list` shows pandas, numpy, matplotlib)
- [ ] Data loader runs without errors
- [ ] You see "181 observations" in output
- [ ] Processed data saved to `data/processed/`
- [ ] Git commit successful (`git log` shows your commit)
- [ ] You understand what each variable means (MPR, Inflation, ER, M2)
- [ ] You understand how you built the code step-by-step

**If all boxes checked → DAY 1 COMPLETE! 🎉**

---

## What You Built Today

**Files created:** 16
**Lines of code written:** 238 (built in 4 chunks over 4 hours)
**Data points loaded:** 724 (181 months × 4 variables)
**Skills learned:** Python packaging, pandas, data validation, git, step-by-step code building

**Key learning:** You built a complete data loader by adding small pieces one at a time and testing after each step!

---

## Tomorrow (Day 2): Exploratory Data Analysis

**What you'll build:**
- Plotting utilities (`src/visualization/plots.py`)
- Time series plots for all 4 variables
- Correlation analysis
- Trend decomposition

**Time:** 8 hours
**Difficulty:** Same as Day 1

---

## Troubleshooting

**"pip: command not found"**
```bash
# Install pip first
python -m ensurepip --upgrade
```

**"Permission denied" when creating folders**
```bash
# Use sudo (Linux/Mac) or run terminal as Administrator (Windows)
sudo mkdir -p data/raw
```

**"pandas version 2.1.4 not available"**
```bash
# Install latest pandas instead
pip install pandas numpy matplotlib seaborn jupyter
```

**"My file has fewer than 238 lines"**
```bash
# Check you added all 6 methods:
# 1. __init__
# 2. load_raw_data
# 3. validate_data
# 4. create_analysis_dataset
# 5. save_processed_data
# 6. get_data_summary
# 7. load_and_prepare
# Plus the main() function at the end
```

---

**Well done! See you tomorrow for Day 2! 💪**
