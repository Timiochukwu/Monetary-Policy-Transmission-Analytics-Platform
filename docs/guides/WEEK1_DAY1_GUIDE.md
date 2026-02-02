# Week 1 - Day 1: Project Foundation & Data Acquisition
## Complete Build Guide (Learn by Doing)

**Duration**: 4-6 hours
**Difficulty**: Beginner-Intermediate
**Goal**: Set up complete project infrastructure and acquire Nigerian macroeconomic data

---

## 📚 What You'll Learn

By the end of Day 1, you will understand:

1. How to structure a professional data science project
2. Python package organization with `__init__.py`
3. Building ETL pipelines for economic data
4. Data validation and quality assurance
5. Academic documentation standards
6. Git version control best practices

---

## 🎯 Learning Objectives

### Technical Skills
- ✅ Create modular Python packages
- ✅ Implement object-oriented data loaders
- ✅ Use pandas for time series data
- ✅ Build validation frameworks
- ✅ Configure Jupyter notebooks

### Domain Knowledge
- ✅ Understand Nigerian monetary policy variables
- ✅ Learn CBN data sources
- ✅ Grasp VAR model variable ordering
- ✅ Academic citation standards

---

## 📋 Prerequisites

**Required Knowledge**:
- Basic Python (functions, classes, imports)
- pandas fundamentals (DataFrames, indexing)
- Command line basics (cd, mkdir, ls)
- Git basics (add, commit, push)

**Software Needed**:
- Python 3.9+ installed
- Git installed
- Text editor (VS Code, PyCharm, or any)
- Terminal/command prompt

**Time Estimate**: 4-6 hours (don't rush, understand each step)

---

## 🏗️ Part 1: Project Structure Setup (30 minutes)

### Step 1.1: Create Root Directory

**Why?** Proper organization prevents chaos as project grows.

```bash
# Navigate to where you want the project
cd ~/projects  # or C:\Users\YourName\projects on Windows

# Create project root
mkdir Monetary-Policy-Transmission-Analytics-Platform
cd Monetary-Policy-Transmission-Analytics-Platform
```

**Understanding**: The project name is descriptive - it tells anyone what this does.

---

### Step 1.2: Initialize Git Repository

**Why?** Version control from Day 1 prevents lost work and enables collaboration.

```bash
# Initialize git
git init

# Create the feature branch
git checkout -b claude/monetary-policy-analytics-platform-03tRK
```

**Understanding**:
- `git init` creates `.git/` folder for tracking changes
- Branch naming convention: `claude/` prefix + descriptive name + session ID

---

### Step 1.3: Create Directory Structure

**Why?** Separation of concerns - each folder has a clear purpose.

```bash
# Create all directories at once
mkdir -p data/raw data/processed \
         src/data_ingestion src/econometrics src/visualization src/utils \
         api \
         results/stationarity results/ardl results/var results/simulations \
         notebooks \
         docs/guides

# Verify structure
ls -R
```

**Understanding Each Directory**:

| Directory | Purpose | Example Contents |
|-----------|---------|------------------|
| `data/raw/` | Original, untouched data | CBN downloaded CSVs |
| `data/processed/` | Cleaned, analysis-ready data | Processed CSVs |
| `src/data_ingestion/` | Data loading code | `data_loader.py` |
| `src/econometrics/` | Statistical models | `ardl_model.py`, `var_model.py` |
| `src/visualization/` | Plotting utilities | `plots.py` |
| `src/utils/` | Helper functions | `config.py` |
| `api/` | Java Spring Boot API | (Week 5) |
| `results/` | Model outputs | Saved plots, tables |
| `notebooks/` | Jupyter analysis | `exploratory_analysis.ipynb` |
| `docs/` | Documentation | Methodology, guides |

**Key Principle**: **Never mix code with data**. Never mix raw data with processed data.

---

### Step 1.4: Create Python Package Structure

**Why?** `__init__.py` makes directories importable as Python modules.

```bash
# Create __init__.py files
touch src/__init__.py
touch src/data_ingestion/__init__.py
touch src/econometrics/__init__.py
touch src/visualization/__init__.py
touch src/utils/__init__.py
```

**Understanding**:
- Empty `__init__.py` = "This directory is a Python package"
- Allows: `from src.data_ingestion import data_loader`
- Without it: `ModuleNotFoundError`

**Test It**:
```bash
python3 -c "import sys; sys.path.append('.'); import src"
# No error = success!
```

---

## 🛡️ Part 2: Version Control Configuration (15 minutes)

### Step 2.1: Create .gitignore

**Why?** Prevents committing large data files, secrets, and system files.

**Create file**: `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.venv/
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# PyCharm
.idea/

# VS Code
.vscode/

# Data files (keep raw data local, not in git)
data/raw/*.csv
data/raw/*.xlsx
data/raw/*.xls
data/processed/*.csv
data/processed/*.parquet

# Sensitive files
*.env
.env
.env.local
config/secrets.yml
credentials.json

# Java/Spring Boot
*.class
*.jar
*.war
*.ear
target/
.gradle/
build/
.gradle
gradle/
gradlew
gradlew.bat

# PostgreSQL
*.sql.backup
*.dump

# OS
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# Results (optional - keep generated outputs in git for thesis)
# results/**/*.png
# results/**/*.csv

# Logs
*.log
logs/
```

**Understanding Key Sections**:

1. **Python bytecode** (`__pycache__/`): Auto-generated, don't need in git
2. **Virtual environments** (`.venv/`): Each developer creates their own
3. **Data files**: Too large for git, use data sources documentation instead
4. **Secrets**: Never commit API keys, passwords, credentials
5. **IDE files**: Personal preferences, not project files

**Pro Tip**: Notice `data/raw/*.csv` is ignored, but we'll force-add a template later for demonstration.

---

### Step 2.2: Create requirements.txt

**Why?** Reproducibility - anyone can recreate your environment.

**Create file**: `requirements.txt`

```txt
# Monetary Policy Transmission Analytics Platform
# Day 1 Dependencies - Data Acquisition & EDA

# Core Data Manipulation
pandas==2.1.4
numpy==1.26.2

# Data Visualization
matplotlib==3.8.2
seaborn==0.13.0

# Excel/CSV handling
openpyxl==3.1.2
xlrd==2.0.1

# Jupyter for exploratory analysis
jupyter==1.0.0
ipykernel==6.27.1

# Note: Econometric libraries will be added in Week 2
# statsmodels, arch, scipy will be installed when needed
```

**Understanding**:
- **Version pinning** (`==2.1.4`): Ensures everyone uses same versions
- **Comments**: Explain why each dependency exists
- **Phased installation**: Don't install everything at once (bloat)

**Install Dependencies**:
```bash
# Create virtual environment (RECOMMENDED)
python3 -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

**Expected output**:
```
Package         Version
--------------- -------
pandas          2.1.4
numpy           1.26.2
matplotlib      3.8.2
...
```

---

## 📊 Part 3: Data Acquisition & Documentation (45 minutes)

### Step 3.1: Understand the Data

**Variables We Need** (Monthly frequency, 2010-2025):

| Variable | Description | Source | Why Needed? |
|----------|-------------|--------|-------------|
| **MPR** | Monetary Policy Rate | CBN | Policy instrument (independent variable) |
| **Inflation** | CPI Year-over-Year | NBS/CBN | Target variable (dependent) |
| **Exchange Rate** | NGN/USD | CBN | Transmission channel (pass-through) |
| **M2** | Broad Money Supply | CBN | Transmission channel (liquidity) |

**Economic Story**:
```
CBN raises MPR → Banks increase lending rates →
  ├─ Exchange rate appreciates (capital inflows)
  ├─ M2 growth slows (credit tightening)
  └─ Inflation decreases (reduced demand + lower import costs)
```

**Understanding**: This is the **monetary policy transmission mechanism** we're modeling.

---

### Step 3.2: Create Data Sources Documentation

**Why?** Academic integrity - cite your sources. Thesis examiners will ask "where's the data from?"

**Create file**: `data/metadata.json`

```json
{
  "project": "Monetary Policy Transmission Analytics Platform - Nigeria",
  "data_period": {
    "start": "2010-01",
    "end": "2025-01",
    "frequency": "monthly",
    "observations_target": 180
  },
  "variables": {
    "mpr": {
      "name": "Monetary Policy Rate",
      "unit": "percent per annum",
      "source": "Central Bank of Nigeria (CBN)",
      "source_url": "https://statistics.cbn.gov.ng/",
      "description": "Official benchmark interest rate set by CBN Monetary Policy Committee",
      "transformation": "none (already in percentage)",
      "role": "Policy variable (exogenous in VAR ordering)",
      "academic_reference": "CBN Communiqués of Monetary Policy Committee Meetings"
    },
    "inflation": {
      "name": "Headline Consumer Price Index Inflation",
      "unit": "year-on-year percentage change",
      "source": "National Bureau of Statistics (NBS) / CBN",
      "source_url": "https://nigerianstat.gov.ng/",
      "description": "All-items CPI inflation rate (YoY growth)",
      "transformation": "YoY percentage change from CPI index",
      "role": "Target variable (most endogenous in VAR)",
      "academic_reference": "NBS CPI Reports (monthly)"
    },
    "exchange_rate": {
      "name": "Official Exchange Rate (Naira per USD)",
      "unit": "NGN/USD",
      "source": "Central Bank of Nigeria",
      "source_url": "https://statistics.cbn.gov.ng/",
      "description": "NAFEX (Investors & Exporters Window) rate - market-determined rate",
      "transformation": "End-of-month or monthly average",
      "role": "Transmission channel (responds to MPR via interest parity)",
      "academic_reference": "CBN Statistical Bulletin, Exchange Rate Section",
      "note": "Pre-2017 data uses interbank rate; post-2017 uses I&E window"
    },
    "money_supply": {
      "name": "Broad Money Supply (M2)",
      "unit": "Naira billions",
      "source": "Central Bank of Nigeria",
      "source_url": "https://statistics.cbn.gov.ng/",
      "description": "M2 = Currency in Circulation + Demand Deposits + Savings & Time Deposits",
      "transformation": "Log-level or growth rate",
      "role": "Monetary aggregate (credit channel indicator)",
      "academic_reference": "CBN Statistical Bulletin, Money & Credit Section"
    }
  },
  "data_sources_priority": [
    {
      "rank": 1,
      "source": "CBN Statistical Bulletin",
      "url": "https://www.cbn.gov.ng/documents/Statbulletin.asp",
      "coverage": "All variables, official government data",
      "reliability": "Highest - used in academic publications"
    },
    {
      "rank": 2,
      "source": "CBN Statistics Database",
      "url": "https://statistics.cbn.gov.ng/",
      "coverage": "Interactive query system for all CBN data",
      "reliability": "High - same source as bulletin, more up-to-date"
    },
    {
      "rank": 3,
      "source": "IMF International Financial Statistics",
      "url": "https://data.imf.org/",
      "coverage": "Cross-country validated data for Nigeria",
      "reliability": "High - good for missing CBN data points"
    },
    {
      "rank": 4,
      "source": "FRED (Federal Reserve Bank of St. Louis)",
      "url": "https://fred.stlouisfed.org/",
      "coverage": "Limited Nigeria series (inflation, exchange rate)",
      "reliability": "Medium-High - aggregated from official sources"
    }
  ],
  "data_collection_notes": {
    "missing_data_strategy": "Linear interpolation for isolated missing months; forward fill for end-of-sample",
    "outlier_handling": "Flag values >3 SD from mean; investigate against CBN policy events",
    "seasonal_adjustment": "Not applied - monthly data retains seasonal patterns for VAR",
    "citation_format": "Central Bank of Nigeria (2024). Statistical Bulletin. Abuja: CBN."
  },
  "academic_justification": {
    "sample_period": "2010-2025 covers multiple MPR cycles, naira devaluation episodes (2015-2016, 2020, 2023), and inflation surge periods, providing sufficient variation for identification.",
    "variable_selection": "Four-variable VAR follows Sims (1980) parsimony principle while capturing key transmission channels: interest rate, exchange rate, and monetary aggregate effects on prices.",
    "data_frequency": "Monthly frequency balances transmission lag dynamics (policy takes 3-12 months to affect inflation) with sufficient observations for estimation (T=180)."
  }
}
```

**Understanding**:
- **JSON format**: Easy to parse programmatically
- **Comprehensive**: Every variable documented with source, unit, role
- **Academic**: References and justifications included
- **Practical**: URLs to actual data sources

**Pro Tip**: This file is gold during thesis defense. Examiner asks "why monthly?" - you have the answer documented.

---

### Step 3.3: Create Template Data (FOR LEARNING PURPOSES)

**Important**: In real research, you'd download actual CBN data. For this tutorial, we'll create realistic template data.

**Create file**: `data/raw/nigeria_macro_data.csv`

I'll show you the first 24 rows - the full file has 181 rows (2010-01 to 2025-01):

```csv
date,mpr,inflation,exchange_rate,money_supply_m2
2010-01,6.00,12.8,150.3,9234.5
2010-02,6.00,13.0,150.5,9312.3
2010-03,6.00,13.6,150.8,9401.2
2010-04,6.00,12.5,151.0,9489.7
2010-05,6.00,13.2,151.2,9578.9
2010-06,6.00,13.7,151.5,9668.4
2010-07,6.25,13.2,151.8,9758.2
2010-08,6.25,13.6,152.0,9848.6
2010-09,6.25,13.9,152.3,9939.1
2010-10,6.25,13.4,152.5,10030.8
2010-11,6.25,12.8,152.8,10123.4
2010-12,6.25,11.8,153.0,10216.9
2011-01,6.50,12.1,153.3,10311.2
... (continues to 2025-01)
```

**Understanding the Data Pattern**:
1. **MPR**: Steps up from 6% (2010) → 27.5% (2024) reflecting CBN tightening cycles
2. **Inflation**: Volatile, with surges in 2016 and 2023 (devaluation periods)
3. **Exchange Rate**: Depreciates from ₦150/$ → ₦935/$ (reflects naira weakness)
4. **M2**: Smooth upward trend (money supply growth)

**Key Events Embedded**:
- 2016: Naira devaluation (₦197 → ₦305)
- 2020: COVID-19 shock
- 2023: FX unification shock (₦460 → ₦750)

**How to Get Real Data** (your homework):
1. Visit https://statistics.cbn.gov.ng/
2. Navigate to:
   - Monetary Policy → MPR
   - Prices → CPI
   - Exchange Rates → NAFEX
   - Money & Credit → M2
3. Download monthly data (2010-2025)
4. Replace `nigeria_macro_data.csv`

---

## 🐍 Part 4: Build Data Loader (90 minutes)

This is where you learn **object-oriented programming** for data pipelines.

### Step 4.1: Create Configuration Module First

**Why?** Centralized settings prevent hardcoded values scattered everywhere.

**Create file**: `src/utils/config.py`

```python
"""
Global Configuration Settings for Monetary Policy Transmission Platform

Centralized configuration for paths, parameters, and settings used across modules.
"""

from pathlib import Path

# =============================================================================
# PROJECT STRUCTURE
# =============================================================================

# Root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Variable names (standardized)
VARIABLE_NAMES = {
    'mpr': 'MPR',
    'inflation': 'Inflation',
    'exchange_rate': 'ExchangeRate',
    'money_supply_m2': 'M2'
}

# VAR ordering (Cholesky decomposition)
VAR_ORDERING = ['MPR', 'ExchangeRate', 'M2', 'Inflation']

# ... (see full file in repository for complete config)
```

**Understanding Key Concepts**:

1. **`Path(__file__)`**: Gets current file's location
2. **`.parent.parent.parent`**: Goes up 3 levels (config.py → utils → src → project root)
3. **`/` operator**: Pathlib's way to join paths (works on Windows/Mac/Linux)

**Test It**:
```bash
python3 -c "from src.utils.config import PROJECT_ROOT; print(PROJECT_ROOT)"
# Should print your project directory path
```

---

### Step 4.2: Build Data Loader Class

**Why?** Classes encapsulate related functionality - all data operations in one place.

**Create file**: `src/data_ingestion/data_loader.py`

Let me break this down section by section:

#### Section 1: Imports and Class Definition

```python
"""
Data Loader for Nigerian Monetary Policy Transmission Analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional


class NigerianMacroDataLoader:
    """
    Loads and preprocesses Nigerian macroeconomic data.

    Variables:
    - MPR: Monetary Policy Rate (%)
    - Inflation: Headline CPI inflation (YoY %)
    - Exchange Rate: NGN/USD
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
```

**Understanding**:
- **Docstrings** (`""" ... """`): Documentation for functions/classes
- **Type hints** (`data_dir: str`): Makes code self-documenting
- **`self.`**: Instance variables - each loader object has its own paths
- **Pathlib**: Modern way to handle file paths

**Test It**:
```python
from src.data_ingestion.data_loader import NigerianMacroDataLoader
loader = NigerianMacroDataLoader()
print(loader.raw_dir)  # Should show data/raw path
```

#### Section 2: Load Raw Data Method

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

**Understanding**:
- **Error handling**: `if not exists()` prevents cryptic pandas errors
- **`pd.to_datetime()`**: Converts string dates to datetime objects
- **`.set_index('date')`**: Makes date the row index (time series convention)
- **`.strftime('%Y-%m')`**: Formats datetime as "2010-01"

**Common Pitfall**: Forgetting `set_index()` means you can't use time-based operations later.

#### Section 3: Validation Method

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

        return validation_results
```

**Understanding**:
- **`.isnull().sum()`**: Counts missing values per column
- **`.sum().sum()`**: Total missing values (sum of column sums)
- **Returning dict**: Flexible - can add more checks without changing function signature

**Why 120 observations?**: VAR models need sufficient data. Rule of thumb: `observations > (variables × lags) × 10`

#### Section 4: Create Analysis Dataset

```python
    def create_analysis_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create analysis-ready dataset with proper column naming and ordering.

        Args:
            df: Raw DataFrame

        Returns:
            Processed DataFrame ready for econometric analysis
        """
        # Rename columns to standard format
        column_mapping = {
            'mpr': 'MPR',
            'inflation': 'Inflation',
            'exchange_rate': 'ExchangeRate',
            'money_supply_m2': 'M2'
        }

        df_analysis = df.rename(columns=column_mapping)

        # Ensure proper ordering (matches VAR ordering)
        column_order = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
        df_analysis = df_analysis[column_order]

        return df_analysis
```

**Understanding**:
- **Standardization**: Raw data has lowercase names, analysis uses PascalCase
- **Ordering matters**: VAR models are sensitive to variable order (more in Week 3)
- **`df[column_order]`**: Reorders columns

**Pro Tip**: Always reorder columns to match your econometric model's expectations.

#### Section 5: Main Pipeline

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

        if validation['warnings']:
            print("\n  ⚠ Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")

        # Step 3: Create analysis dataset
        print("\n[3/4] Creating analysis dataset...")
        df_analysis = self.create_analysis_dataset(df_raw)

        # Step 4: Summary
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

**Understanding**:
- **Pipeline pattern**: Step-by-step with clear outputs
- **User feedback**: Progress messages help debugging
- **Return value**: Other code can use the loaded data

**Test the Complete Loader**:
```bash
cd Monetary-Policy-Transmission-Analytics-Platform
python src/data_ingestion/data_loader.py
```

Expected output:
```
============================================================
NIGERIAN MONETARY POLICY DATA LOADER
============================================================

[1/4] Loading raw data...
✓ Loaded 181 observations from 2010-01 to 2025-01

[2/4] Validating data...
  - Observations: 181
  - Period: 2010-01 to 2025-01

[3/4] Creating analysis dataset...

[4/4] Summary statistics:
              count   mean    std     min     max  Missing
MPR           181.0  13.44   4.45    6.0    27.5        0
ExchangeRate  181.0  357.8  213.5  150.3   935.0        0
M2            181.0  26474  14890   9234   63665        0
Inflation     181.0  15.00   6.60    7.8    34.8        0

✓ Saved processed data to data/processed/processed_macro_data.csv
============================================================
✓ DATA LOADING COMPLETE
============================================================
```

---

### Step 4.3: Build Data Validator

**Why?** Separate validation logic from loading - single responsibility principle.

**Create file**: `src/data_ingestion/data_validator.py`

This is a more advanced class. I'll highlight key methods:

```python
"""
Data Validation Module for Nigerian Monetary Policy Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataValidator:
    """
    Validates macroeconomic data quality for econometric analysis.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.validation_report = {}

    def check_missing_values(self) -> Dict[str, int]:
        """Check for missing values in each variable."""
        missing_counts = self.df.isnull().sum().to_dict()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict()

        self.validation_report['missing_values'] = {
            'counts': missing_counts,
            'percentages': missing_pct
        }

        return missing_counts

    def detect_outliers(self, threshold: float = 3.0) -> Dict[str, List[Tuple]]:
        """
        Detect outliers using z-score method.

        Args:
            threshold: Number of standard deviations (default: 3.0)

        Returns:
            Dictionary with outliers per variable
        """
        outliers = {}

        for col in self.df.columns:
            # Calculate z-scores
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())

            # Find outliers
            outlier_indices = z_scores[z_scores > threshold].index
            outlier_values = [(idx.strftime('%Y-%m'), self.df.loc[idx, col])
                             for idx in outlier_indices]

            if outlier_values:
                outliers[col] = outlier_values

        return outliers
```

**Understanding Z-Score Outlier Detection**:
```
z-score = (value - mean) / std_dev

If |z-score| > 3:
  → Value is more than 3 standard deviations from mean
  → Likely an outlier (or structural break!)
```

**Example**:
- Inflation mean = 15%
- Std dev = 6%
- Observation = 35%
- Z-score = (35 - 15) / 6 = 3.33 → **Outlier!**

But wait - is this a data error or the 2023 inflation surge? **You must investigate!**

**Full validator code**: See `data_validator.py` in repository.

---

## 📓 Part 5: Jupyter Notebook for Exploration (30 minutes)

### Step 5.1: Create Exploratory Analysis Notebook

**Why?** Jupyter notebooks are perfect for iterative data exploration and visualization.

**Create file**: `notebooks/exploratory_analysis.ipynb`

I'll show you the key cells:

**Cell 1: Setup**
```python
# Import libraries
import sys
sys.path.append('../src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')
%matplotlib inline

print("✓ Libraries loaded")
```

**Understanding**:
- **`sys.path.append('../src')`**: Adds src/ to Python path so we can import our modules
- **`%matplotlib inline`**: Shows plots in notebook

**Cell 2: Load Data**
```python
from data_ingestion.data_loader import NigerianMacroDataLoader

loader = NigerianMacroDataLoader(data_dir='../data')
df = loader.load_and_prepare()
```

**Cell 3: Time Series Plot**
```python
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# MPR
axes[0].plot(df.index, df['MPR'], color='#2E86AB', linewidth=2)
axes[0].set_title('Monetary Policy Rate (MPR)', fontweight='bold')
axes[0].set_ylabel('% per annum')
axes[0].grid(True, alpha=0.3)

# Inflation
axes[1].plot(df.index, df['Inflation'], color='#A23B72', linewidth=2)
axes[1].set_title('Headline Inflation', fontweight='bold')
axes[1].set_ylabel('% YoY')
axes[1].grid(True, alpha=0.3)

# Exchange Rate
axes[2].plot(df.index, df['ExchangeRate'], color='#F18F01', linewidth=2)
axes[2].set_title('Exchange Rate (NGN/USD)', fontweight='bold')
axes[2].set_ylabel('Naira per Dollar')
axes[2].grid(True, alpha=0.3)

# M2
axes[3].plot(df.index, df['M2'], color='#6A994E', linewidth=2)
axes[3].set_title('Broad Money Supply (M2)', fontweight='bold')
axes[3].set_ylabel('Naira Billions')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Understanding**:
- **`subplots(4, 1)`**: 4 rows, 1 column of plots
- **Color codes**: Professional color palette (not default matplotlib)
- **`tight_layout()`**: Prevents labels from overlapping

**Launch Jupyter**:
```bash
cd notebooks
jupyter notebook exploratory_analysis.ipynb
```

---

## 📝 Part 6: Documentation (60 minutes)

### Step 6.1: Create Professional README

**Why?** README is the first thing anyone sees - make it count!

**Create file**: `README.md`

Structure:
1. **Project title and tagline**
2. **Overview** (what problem does this solve?)
3. **Methodology** (high-level econometrics)
4. **Project structure** (directory tree)
5. **Quick start** (how to run it)
6. **Academic rigor** (literature references)
7. **Timeline** (30-day plan)

**Key Sections**:

```markdown
# Monetary Policy Transmission Analytics Platform - Nigeria

> A production-quality econometric platform analyzing how Central Bank of Nigeria (CBN) monetary policy shocks transmit to inflation.

## 🎯 Project Overview

This platform models the **transmission mechanism** of monetary policy in Nigeria using advanced time series econometrics.

**Research Question**: "How do changes in the Monetary Policy Rate (MPR) affect inflation in Nigeria, and through which channels?"

### Key Features

- ARDL (Autoregressive Distributed Lag) modeling
- VAR (Vector Autoregression) with Cholesky identification
- Impulse Response Functions (IRF)
- Forecast Error Variance Decomposition (FEVD)
- Policy simulation framework

### Applications

✅ MSc Thesis-Ready
✅ Job Interview Portfolio
✅ Policy Research
```

**Understanding**:
- **Tagline**: One sentence that explains everything
- **Bullet points**: Easy scanning (recruiters spend 30 seconds)
- **Checkmarks** (`✅`): Visual appeal

**Full README**: See repository for complete version (346 lines).

---

### Step 6.2: Academic Methodology Documentation

**Create file**: `docs/methodology.md`

This is your **thesis methodology chapter** in markdown form.

**Structure**:
1. Unit root and stationarity testing
2. ARDL model specification
3. VAR model structure
4. Impulse response functions
5. Variance decomposition
6. Policy simulations

**Example Section** (Stationarity Tests):

```markdown
## 1. Unit Root and Stationarity Testing

### Why Test for Stationarity?

Non-stationary variables can lead to **spurious regressions** (Granger & Newbold, 1974).

### Augmented Dickey-Fuller (ADF) Test

**Null Hypothesis**: Variable has a unit root (non-stationary)

Test equation:
\`\`\`
Δy_t = α + βt + γy_{t-1} + Σδ_i Δy_{t-i} + ε_t
\`\`\`

- **Test statistic**: t-statistic on γ
- **Rejection**: If t-stat < critical value → stationary
- **Lag selection**: Schwarz Information Criterion
```

**Understanding**:
- **Mathematical notation**: Shows you understand the theory
- **Decision rules**: Clear interpretation guidelines
- **Citations**: Academic credibility

**Pro Tip**: Write methodology BEFORE coding - it guides implementation.

---

### Step 6.3: VAR Identification Strategy Document

**Create file**: `docs/identification_strategy.md`

This answers: **"Why did you order variables this way in the VAR?"**

Thesis examiners WILL ask this. Having a 300+ line document ready is impressive.

**Structure**:
1. The identification problem
2. Your chosen ordering with justification
3. Alternative orderings (and why rejected)
4. Literature support
5. Nigerian context
6. Expected thesis defense questions

**Key Argument**:

```markdown
## Cholesky Ordering for Nigeria

\`\`\`
MPR → Exchange Rate → M2 → Inflation
\`\`\`

### Position 1: MPR (Most Exogenous)

**Why first?**

1. CBN MPC sets MPR independently based on past data
2. Not affected contemporaneously by other variables
3. Policy rate is "sticky" (changed only at MPC meetings)

**Assumption**:
\`\`\`
MPR_t = f(Ω_{t-1})   [past information only]
\`\`\`

### Position 2: Exchange Rate

**Why second?**

1. Uncovered Interest Parity (UIP):
   \`\`\`
   E_t[Δs_{t+1}] = i_NGN - i_USD
   \`\`\`
2. FX markets react within 24-48 hours to MPR changes
3. Transmission via capital flows (fast)

**Assumption**:
\`\`\`
ExchangeRate_t = g(MPR_t, Ω_{t-1})
\`\`\`
```

**Understanding**:
- **Economic theory**: UIP, capital flows, price stickiness
- **Institutional**: CBN meeting schedule, FX market structure
- **Empirical**: Literature from similar countries

**Full document**: See `identification_strategy.md` (366 lines).

---

## 🔧 Part 7: Testing & Verification (30 minutes)

### Step 7.1: Test Data Loader

```bash
# Test 1: Run data loader standalone
python src/data_ingestion/data_loader.py

# Expected: Success message + 181 observations loaded
```

### Step 7.2: Test from Python REPL

```python
# Test 2: Import as module
from src.data_ingestion.data_loader import NigerianMacroDataLoader

loader = NigerianMacroDataLoader()
df = loader.load_and_prepare()

# Verify
print(df.shape)  # Should be (181, 4)
print(df.columns.tolist())  # ['MPR', 'ExchangeRate', 'M2', 'Inflation']
print(df.isnull().sum())  # Should all be 0
```

### Step 7.3: Test Data Validator

```python
from src.data_ingestion.data_validator import DataValidator

validator = DataValidator(df)
report = validator.validate_all()

# Should print comprehensive validation report
```

### Step 7.4: Verify File Structure

```bash
find . -type f -name "*.py" -o -name "*.md" -o -name "*.json" | grep -v .git
```

Expected output:
```
./README.md
./data/metadata.json
./data/raw/nigeria_macro_data.csv
./docs/methodology.md
./docs/identification_strategy.md
./requirements.txt
./src/__init__.py
./src/data_ingestion/data_loader.py
./src/data_ingestion/data_validator.py
./src/utils/config.py
```

---

## 📦 Part 8: Git Commit & Push (20 minutes)

### Step 8.1: Review Changes

```bash
git status
```

Should show all new files as untracked.

### Step 8.2: Stage Files

```bash
# Stage everything
git add -A

# Force-add template data (normally gitignored)
git add -f data/raw/nigeria_macro_data.csv

# Verify staged files
git status
```

### Step 8.3: Create Meaningful Commit

```bash
git commit -m "$(cat <<'EOF'
Day 1: Project foundation and data acquisition infrastructure

Established complete project architecture for Nigerian Monetary Policy
Transmission Analytics Platform (MSc thesis / portfolio project).

Key Deliverables:
✅ Project structure with separation of concerns
✅ Data ingestion framework (loader + validator)
✅ Template macroeconomic data (2010-2025, 181 observations)
✅ Comprehensive documentation
✅ Exploratory analysis Jupyter notebook
✅ Configuration management system

Technical Stack:
- Python 3.9+ with pandas, numpy, matplotlib
- Object-oriented data pipeline
- JSON-based metadata
- Academic documentation standards

Next: Day 2 - Stationarity tests (ADF, PP, KPSS)

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs
EOF
)"
```

**Understanding**:
- **Multi-line commit**: Detailed context for future you
- **Structured format**: What/Why/How
- **Session link**: Traceability

### Step 8.4: Push to Remote

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

**Verify**:
```bash
git log --oneline -1
# Should show your commit

git remote -v
# Should show GitHub URL
```

---

## ✅ Day 1 Completion Checklist

Use this to verify you've completed everything:

### Files Created (16 total)

- [ ] `.gitignore`
- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `data/metadata.json`
- [ ] `data/raw/nigeria_macro_data.csv`
- [ ] `docs/methodology.md`
- [ ] `docs/identification_strategy.md`
- [ ] `src/__init__.py`
- [ ] `src/data_ingestion/__init__.py`
- [ ] `src/data_ingestion/data_loader.py`
- [ ] `src/data_ingestion/data_validator.py`
- [ ] `src/econometrics/__init__.py`
- [ ] `src/utils/__init__.py`
- [ ] `src/utils/config.py`
- [ ] `src/visualization/__init__.py`
- [ ] `notebooks/exploratory_analysis.ipynb`

### Directories Created (11 total)

- [ ] `data/raw/`
- [ ] `data/processed/`
- [ ] `src/data_ingestion/`
- [ ] `src/econometrics/`
- [ ] `src/visualization/`
- [ ] `src/utils/`
- [ ] `results/stationarity/`
- [ ] `results/ardl/`
- [ ] `results/var/`
- [ ] `results/simulations/`
- [ ] `notebooks/`
- [ ] `docs/`

### Functionality Tests

- [ ] Data loader runs without errors: `python src/data_ingestion/data_loader.py`
- [ ] 181 observations loaded (2010-01 to 2025-01)
- [ ] No missing values in any variable
- [ ] Processed data saved to `data/processed/`
- [ ] Jupyter notebook opens: `jupyter notebook notebooks/exploratory_analysis.ipynb`

### Git Verification

- [ ] All files committed
- [ ] Pushed to remote repository
- [ ] Commit message is descriptive

### Knowledge Check

Can you explain:
- [ ] Why we use Pathlib instead of string paths?
- [ ] What `__init__.py` does?
- [ ] Why MPR is ordered first in the VAR?
- [ ] What a z-score outlier is?
- [ ] How `pd.to_datetime()` helps with time series?

If you checked all boxes → **Day 1 COMPLETE!** 🎉

---

## 🎓 Key Concepts Learned

### Programming Concepts

1. **Object-Oriented Programming**: Classes encapsulate related functionality
2. **Type Hints**: Make code self-documenting
3. **Pathlib**: Cross-platform file path handling
4. **Docstrings**: In-code documentation
5. **Error Handling**: `if not exists()` prevents crashes

### Data Science Concepts

1. **ETL Pipeline**: Extract (load CSV) → Transform (clean) → Load (save processed)
2. **Data Validation**: Check quality before analysis
3. **Time Series Index**: `set_index('date')` for temporal operations
4. **Outlier Detection**: Z-score method
5. **Exploratory Data Analysis**: Visualize before modeling

### Econometric Concepts

1. **Monetary Policy Transmission**: MPR → ER/M2 → Inflation
2. **VAR Variable Ordering**: Cholesky decomposition requires ordering
3. **Data Frequency**: Monthly balances lags and observations
4. **Sample Size**: Need >120 observations for reliable VAR

### Professional Practices

1. **Version Control**: Commit early, commit often
2. **Documentation**: Write for your future self (and examiners)
3. **Project Structure**: Organize by function, not file type
4. **Reproducibility**: `requirements.txt`, clear instructions

---

## 🚀 Next Steps: Day 2 Preview

**Day 2 Objective**: Data cleaning, validation, and exploratory time series analysis

**What You'll Build**:
1. Plotting utilities (`src/visualization/plots.py`)
2. Trend decomposition analysis
3. Growth rate calculations
4. Visual identification of structural breaks

**What You'll Learn**:
- Advanced matplotlib customization
- Time series decomposition (trend/seasonal/residual)
- Rolling statistics
- Structural break detection (visual)

**Time Estimate**: 4-5 hours

---

## 💡 Pro Tips for Success

1. **Don't rush**: Understanding > speed
2. **Type the code**: Don't copy-paste (muscle memory helps)
3. **Experiment**: Change parameters, see what breaks
4. **Read error messages**: They're helpful 90% of the time
5. **Git commit frequently**: Every working state
6. **Ask "why?"**: Every design decision has a reason
7. **Document as you go**: Don't leave it for later

---

## 📚 Further Reading

### Python Best Practices
- PEP 8 Style Guide: https://pep8.org/
- Real Python Tutorials: https://realpython.com/

### Pandas for Time Series
- Pandas Time Series: https://pandas.pydata.org/docs/user_guide/timeseries.html

### Econometrics Theory
- Sims (1980): "Macroeconomics and Reality"
- Hamilton (1994): "Time Series Analysis" (textbook)

### Nigerian Economic Context
- CBN Statistical Bulletin: https://www.cbn.gov.ng/
- CBN Monetary Policy: https://www.cbn.gov.ng/MonetaryPolicy/

---

## 🆘 Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

**Solution**:
```python
import sys
sys.path.append('.')  # Add current directory to path
```

### "FileNotFoundError: data/metadata.json"

**Solution**: You're running from wrong directory
```bash
cd Monetary-Policy-Transmission-Analytics-Platform  # Go to project root
python src/data_ingestion/data_loader.py
```

### Pandas version conflicts

**Solution**:
```bash
pip install --upgrade pandas==2.1.4
```

### Jupyter kernel not found

**Solution**:
```bash
python -m ipykernel install --user --name=monetary-policy
```

---

## ✨ You Did It!

If you followed this guide completely, you've built a **production-quality data science project foundation** that demonstrates:

✅ Professional software engineering skills
✅ Academic research standards
✅ Econometric domain knowledge
✅ Nigerian economic context understanding

**This is portfolio-worthy right now, and we've only finished Day 1!**

---

**Guide Version**: 1.0
**Date**: 2025-02-02
**Estimated Completion Time**: 4-6 hours
**Difficulty**: ⭐⭐⭐ (Intermediate)

---

**Ready for Day 2?** See `WEEK1_DAY2_GUIDE.md` (coming next!)
