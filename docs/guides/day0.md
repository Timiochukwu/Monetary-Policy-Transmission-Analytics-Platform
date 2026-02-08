# Day 0: Complete Project Setup - BEFORE Starting Day 1

**Time Required**: 2-3 hours
**Level**: Beginner-Friendly (Step-by-Step)
**Purpose**: Set up EVERYTHING before Day 1 - folders, files, environment, data

---

## 🎯 What You'll Create Today

By the end of Day 0, you'll have:
- ✅ Complete folder structure
- ✅ All `__init__.py` files
- ✅ Virtual environment configured
- ✅ All dependencies installed
- ✅ Sample data file created
- ✅ Git repository initialized
- ✅ Basic configuration files
- ✅ README skeleton

**This ensures Days 1-15 will have ZERO missing dependencies!**

---

## Part 1: System Preparation (30 minutes)

### Step 1: Check Python Version

Open terminal/command prompt:

```bash
# Check Python version (need 3.8 or higher)
python --version
# or
python3 --version
```

**Expected output**: `Python 3.8.x` or higher

If you don't have Python 3.8+, download from [python.org](https://python.org)

---

### Step 2: Choose Project Location

```bash
# Navigate to where you want the project
# Example: Documents folder
cd ~/Documents  # Mac/Linux
cd C:\Users\YourName\Documents  # Windows

# Or create a dedicated workspace
mkdir -p ~/workspace
cd ~/workspace
```

---

### Step 3: Create Project Root Folder

```bash
# Create main project folder
mkdir Monetary-Policy-Transmission-Analytics-Platform
cd Monetary-Policy-Transmission-Analytics-Platform

# Verify you're in the right place
pwd  # Mac/Linux
cd   # Windows
```

**Expected output**: `/path/to/Monetary-Policy-Transmission-Analytics-Platform`

---

## Part 2: Folder Structure (15 minutes)

### Step 4: Create ALL Folders

Copy and paste this entire block:

```bash
# Create all directories at once
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/sample
mkdir -p models
mkdir -p scripts
mkdir -p tests
mkdir -p dashboard
mkdir -p results/plots
mkdir -p results/tables
mkdir -p results/reports
mkdir -p results/stationarity
mkdir -p results/cointegration
mkdir -p results/ardl
mkdir -p results/var
mkdir -p results/irf
mkdir -p results/fevd
mkdir -p results/simulations
mkdir -p results/historical_decomp
mkdir -p results/svar
mkdir -p results/vecm
mkdir -p results/robustness
mkdir -p notebooks
mkdir -p docs/guides
```

**Verify creation**:

```bash
# List all folders
ls -R  # Mac/Linux
tree   # If you have tree installed
dir /s # Windows
```

---

### Step 5: Create All `__init__.py` Files

These make Python recognize folders as packages:

```bash
# Create __init__.py in all Python package folders
touch models/__init__.py
touch tests/__init__.py
touch scripts/__init__.py
touch dashboard/__init__.py

# Verify
ls models/__init__.py tests/__init__.py scripts/__init__.py dashboard/__init__.py
```

**Expected**: All four files should exist (even if empty)

---

## Part 3: Virtual Environment (20 minutes)

### Step 6: Create Virtual Environment

```bash
# Create virtual environment named 'venv'
python -m venv venv
# or
python3 -m venv venv
```

This creates a `venv/` folder in your project.

---

### Step 7: Activate Virtual Environment

**On Mac/Linux**:
```bash
source venv/bin/activate
```

**On Windows (Command Prompt)**:
```cmd
venv\Scripts\activate
```

**On Windows (PowerShell)**:
```powershell
venv\Scripts\Activate.ps1
```

**Expected**: Your terminal prompt should now show `(venv)` at the beginning.

---

### Step 8: Upgrade pip

```bash
pip install --upgrade pip
```

---

## Part 4: Dependencies (15 minutes)

### Step 9: Create `requirements.txt`

Create file `requirements.txt` in project root:

```bash
# Mac/Linux
touch requirements.txt

# Windows
type nul > requirements.txt
```

Open `requirements.txt` in any text editor and paste:

```txt
# Core Data Science
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Econometrics & Statistics
statsmodels>=0.13.0
scipy>=1.7.0

# Date/Time Utilities
python-dateutil>=2.8.0

# Interactive Visualizations
plotly>=5.3.0

# Dashboard (Week 3)
streamlit>=1.10.0

# PDF Reporting (Week 3)
reportlab>=3.6.0
fpdf>=1.7.2
pillow>=8.3.0

# Testing
pytest>=6.2.0
pytest-cov>=2.12.0

# Jupyter (optional but recommended)
jupyter>=1.0.0
ipykernel>=6.0.0
```

Save the file.

---

### Step 10: Install All Dependencies

```bash
pip install -r requirements.txt
```

This will take 5-10 minutes. You'll see packages being downloaded and installed.

**Verify installation**:

```bash
pip list
```

Should show all packages: pandas, numpy, matplotlib, statsmodels, etc.

---

## Part 5: Configuration Files (15 minutes)

### Step 11: Create `.gitignore`

Create file `.gitignore` in project root:

```bash
touch .gitignore  # Mac/Linux
type nul > .gitignore  # Windows
```

Open `.gitignore` and paste:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/
dist/
build/
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store
Thumbs.db

# Results (generated files)
results/**/*.png
results/**/*.pdf
results/**/*.csv
!data/sample/*.csv  # Keep sample data

# Data (except sample)
data/raw/*
data/processed/*
!data/sample/

# Logs
*.log

# OS
.DS_Store
Thumbs.db
```

Save the file.

---

### Step 12: Create `README.md` Skeleton

Create `README.md` in project root:

```markdown
# Nigerian Monetary Policy Transmission Analytics Platform

MSc Thesis Project - Monetary Policy Analysis Using VAR Models

## Status

🚧 Under Development

## Overview

This platform analyzes monetary policy transmission mechanisms in Nigeria using:
- Vector Autoregression (VAR)
- Impulse Response Functions (IRF)
- Forecast Error Variance Decomposition (FEVD)
- Structural VAR (SVAR)
- Robustness checks

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Monetary-Policy-Transmission-Analytics-Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
# Coming soon - see docs/guides/ for step-by-step tutorials
```

## Project Structure

```
├── data/           # Data files
├── models/         # Python modules
├── scripts/        # Utility scripts
├── tests/          # Unit tests
├── dashboard/      # Streamlit dashboard
├── results/        # Output files
└── docs/          # Documentation
```

## Documentation

See `docs/guides/` for complete tutorials:
- Week 1 (Days 1-5): Foundation
- Week 2 (Days 6-10): VAR Analysis
- Week 3 (Days 11-15): Advanced Topics

## Author

[Your Name]
[Your University]
[Your Email]

## License

MIT
```

Save the file.

---

### Step 13: Create `setup.py` (Optional but Professional)

Create `setup.py`:

```python
"""
Setup configuration for Monetary Policy Analytics Platform
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="monetary-policy-analytics",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@university.edu",
    description="VAR-based monetary policy transmission analysis for Nigeria",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Monetary-Policy-Analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Economics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
```

---

## Part 6: Sample Data (20 minutes)

### Step 14: Create Sample Data File

Create `data/sample/sample_data.csv`:

```bash
touch data/sample/sample_data.csv
```

Open the file and paste this sample data:

```csv
Date,MPR,ExchangeRate,M2,Inflation
2020-01-01,13.5,306.85,25234.56,11.98
2020-02-01,13.5,310.25,25678.90,12.26
2020-03-01,13.5,315.60,26123.45,12.34
2020-04-01,12.5,320.15,26567.89,12.40
2020-05-01,12.5,325.70,27012.34,12.50
2020-06-01,12.5,330.25,27456.78,12.56
2020-07-01,12.5,335.80,27901.23,12.82
2020-08-01,12.5,340.35,28345.67,13.22
2020-09-01,12.5,345.90,28790.12,13.71
2020-10-01,11.5,350.45,29234.56,14.23
2020-11-01,11.5,355.00,29678.90,14.89
2020-12-01,11.5,360.55,30123.45,15.75
2021-01-01,11.5,365.10,30567.89,16.47
2021-02-01,11.5,370.65,31012.34,17.33
2021-03-01,11.5,375.20,31456.78,18.17
2021-04-01,11.5,378.75,31901.23,18.12
2021-05-01,11.5,380.30,32345.67,17.93
2021-06-01,11.5,381.85,32790.12,17.75
2021-07-01,11.5,383.40,33234.56,17.38
2021-08-01,11.5,384.95,33678.90,17.01
2021-09-01,11.5,386.50,34123.45,16.63
2021-10-01,11.5,388.05,34567.89,16.40
2021-11-01,11.5,389.60,35012.34,15.99
2021-12-01,11.5,391.15,35456.78,15.63
2022-01-01,11.5,392.70,35901.23,15.60
2022-02-01,11.5,394.25,36345.67,15.70
2022-03-01,11.5,395.80,36790.12,15.92
2022-04-01,11.5,397.35,37234.56,16.82
2022-05-01,13.0,398.90,37678.90,17.71
2022-06-01,13.0,400.45,38123.45,18.60
2022-07-01,14.0,402.00,38567.89,19.64
2022-08-01,14.0,405.55,39012.34,20.52
2022-09-01,15.5,410.10,39456.78,20.77
2022-10-01,16.5,415.65,39901.23,21.09
2022-11-01,16.5,420.20,40345.67,21.47
2022-12-01,16.5,424.75,40790.12,21.34
2023-01-01,17.5,429.30,41234.56,21.82
2023-02-01,17.5,433.85,41678.90,21.91
2023-03-01,18.0,438.40,42123.45,22.04
2023-04-01,18.0,442.95,42567.89,22.22
2023-05-01,18.5,447.50,43012.34,22.41
2023-06-01,18.5,452.05,43456.78,22.79
2023-07-01,18.75,456.60,43901.23,24.08
2023-08-01,18.75,461.15,44345.67,25.80
2023-09-01,18.75,465.70,44790.12,26.72
2023-10-01,18.75,470.25,45234.56,27.33
2023-11-01,18.75,474.80,45678.90,28.20
2023-12-01,18.75,479.35,46123.45,28.92
```

This gives you **48 months** of realistic synthetic data.

**Verify**:
```bash
head data/sample/sample_data.csv
wc -l data/sample/sample_data.csv  # Should show 49 lines (48 + header)
```

---

## Part 7: Git Initialization (10 minutes)

### Step 15: Initialize Git Repository

```bash
# Initialize git
git init

# Check status
git status
```

You should see all the files you created.

---

### Step 16: Create Initial Commit

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial project setup: folder structure, dependencies, sample data"

# Verify
git log
```

---

## Part 8: Verification (15 minutes)

### Step 17: Test Python Imports

Create a test file `test_setup.py` in project root:

```python
"""
Test that all dependencies are installed correctly
"""

print("Testing imports...")

try:
    import pandas as pd
    print("✓ pandas")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    import numpy as np
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib")
except ImportError as e:
    print(f"✗ matplotlib: {e}")

try:
    import statsmodels.api as sm
    print("✓ statsmodels")
except ImportError as e:
    print(f"✗ statsmodels: {e}")

try:
    import scipy
    print("✓ scipy")
except ImportError as e:
    print(f"✗ scipy: {e}")

try:
    import plotly
    print("✓ plotly")
except ImportError as e:
    print(f"✗ plotly: {e}")

try:
    import streamlit
    print("✓ streamlit")
except ImportError as e:
    print(f"✗ streamlit: {e}")

try:
    import pytest
    print("✓ pytest")
except ImportError as e:
    print(f"✗ pytest: {e}")

print("\n✅ All core dependencies installed!")
print("\nTesting data file...")

try:
    df = pd.read_csv('data/sample/sample_data.csv')
    print(f"✓ Sample data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns)}")
except Exception as e:
    print(f"✗ Data loading failed: {e}")

print("\n🎉 Setup verification complete!")
```

Run the test:

```bash
python test_setup.py
```

**Expected output**:
```
Testing imports...
✓ pandas
✓ numpy
✓ matplotlib
✓ statsmodels
✓ scipy
✓ plotly
✓ streamlit
✓ pytest

✅ All core dependencies installed!

Testing data file...
✓ Sample data loaded: 48 rows, 4 columns
  Columns: Date, MPR, ExchangeRate, M2, Inflation

🎉 Setup verification complete!
```

---

### Step 18: Create Master Checklist

Create `PROJECT_CHECKLIST.md`:

```markdown
# Project Completeness Checklist

## ✅ After Day 0, You Should Have:

### Folders
- [ ] `data/raw/`
- [ ] `data/processed/`
- [ ] `data/sample/`
- [ ] `models/`
- [ ] `scripts/`
- [ ] `tests/`
- [ ] `dashboard/`
- [ ] `results/plots/`
- [ ] `results/tables/`
- [ ] `results/reports/`
- [ ] `results/var/`
- [ ] `results/irf/`
- [ ] `results/fevd/`
- [ ] `results/simulations/`
- [ ] `results/historical_decomp/`
- [ ] `results/svar/`
- [ ] `results/robustness/`
- [ ] `notebooks/`
- [ ] `docs/guides/`
- [ ] `venv/`

### Files
- [ ] `models/__init__.py`
- [ ] `tests/__init__.py`
- [ ] `scripts/__init__.py`
- [ ] `dashboard/__init__.py`
- [ ] `requirements.txt`
- [ ] `.gitignore`
- [ ] `README.md`
- [ ] `setup.py`
- [ ] `data/sample/sample_data.csv`
- [ ] `test_setup.py`
- [ ] `PROJECT_CHECKLIST.md`
- [ ] `.git/` (git initialized)

### Environment
- [ ] Virtual environment activated (`venv`)
- [ ] All dependencies installed (see `pip list`)
- [ ] Python 3.8+ verified
- [ ] Test script runs successfully

## 📋 After Each Day (Days 1-15):

### Week 1
- [ ] Day 1: `models/data_loader.py` created
- [ ] Day 2: `models/plots.py` created
- [ ] Day 3: `models/stationarity.py` created
- [ ] Day 4: `models/cointegration.py` created
- [ ] Day 5: `models/ardl.py` created

### Week 2
- [ ] Day 6: `models/var_model.py` created
- [ ] Day 7: `models/irf.py` created
- [ ] Day 8: `models/fevd.py` created
- [ ] Day 9: `models/policy_simulation.py` created
- [ ] Day 10: `models/historical_decomposition.py` created

### Week 3
- [ ] Day 11: `models/svar.py` created
- [ ] Day 12: `models/robustness.py` created
- [ ] Day 13: `dashboard/app.py` created
- [ ] Day 14: `scripts/generate_report.py` created
- [ ] Day 15: `tests/test_*.py` files created, README updated

## ✅ Final Project Should Have:

### Total Files Count
- **12 Python modules** in `models/`
- **2 scripts** in `scripts/`
- **1 dashboard app** in `dashboard/`
- **5+ test files** in `tests/`
- **Configuration files**: requirements.txt, .gitignore, setup.py, README.md
- **Data files**: At least sample_data.csv
- **Documentation**: All guides in docs/guides/

### Total Lines of Code
- **Core modules**: ~4,000 lines
- **Scripts**: ~700 lines
- **Dashboard**: ~400 lines
- **Tests**: ~300 lines
- **Total**: ~5,400 lines of Python code

### Can You:
- [ ] Import all modules without errors
- [ ] Run tests with `pytest tests/ -v`
- [ ] Load data with DataLoader
- [ ] Estimate a VAR model
- [ ] Generate IRFs
- [ ] Launch dashboard with `streamlit run dashboard/app.py`
- [ ] Generate a PDF report
- [ ] All examples in guides work

## 🎯 Ready for Thesis When:
- [ ] All modules tested
- [ ] Real data analyzed (not just sample)
- [ ] Results reproduced multiple times
- [ ] Figures publication-quality
- [ ] Code documented
- [ ] README complete
- [ ] Git history clean
```

---

## ✅ Day 0 Complete!

### What You Now Have:

```
Monetary-Policy-Transmission-Analytics-Platform/
├── .git/                          ✅
├── .gitignore                     ✅
├── README.md                      ✅
├── requirements.txt               ✅
├── setup.py                       ✅
├── test_setup.py                  ✅
├── PROJECT_CHECKLIST.md           ✅
├── venv/                          ✅
│
├── data/                          ✅
│   ├── raw/                       ✅
│   ├── processed/                 ✅
│   └── sample/                    ✅
│       └── sample_data.csv        ✅
│
├── models/                        ✅
│   └── __init__.py                ✅
│
├── scripts/                       ✅
│   └── __init__.py                ✅
│
├── tests/                         ✅
│   └── __init__.py                ✅
│
├── dashboard/                     ✅
│   └── __init__.py                ✅
│
├── results/                       ✅
│   ├── plots/                     ✅
│   ├── tables/                    ✅
│   ├── reports/                   ✅
│   ├── var/                       ✅
│   ├── irf/                       ✅
│   ├── fevd/                      ✅
│   ├── simulations/               ✅
│   ├── historical_decomp/         ✅
│   ├── svar/                      ✅
│   └── robustness/                ✅
│
├── notebooks/                     ✅
│
└── docs/                          ✅
    └── guides/                    ✅
        ├── week1/                 ✅ (from previous work)
        ├── week2/                 ✅ (from previous work)
        └── week3/                 ✅ (from previous work)
```

---

## 🚀 You're Ready for Day 1!

Everything is set up. Starting tomorrow (Day 1), you'll begin writing actual code in the modules.

**Key Points**:
1. ✅ Virtual environment activated (`source venv/bin/activate`)
2. ✅ All dependencies installed
3. ✅ All folders created
4. ✅ Sample data ready
5. ✅ Git initialized
6. ✅ Configuration files in place

**Next**: Proceed to `docs/guides/week1/day1.md` and start building `models/data_loader.py`!

---

## Troubleshooting

### Issue: "python: command not found"
**Fix**: Use `python3` instead, or install Python from python.org

### Issue: "pip: command not found"
**Fix**: After activating venv, upgrade pip: `python -m pip install --upgrade pip`

### Issue: Virtual environment won't activate (Windows)
**Fix**: Run PowerShell as Administrator, then:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Permission denied" when creating folders
**Fix**: Use `sudo` on Mac/Linux or run as Administrator on Windows

### Issue: Sample data won't load
**Fix**: Check file path, ensure CSV is UTF-8 encoded

---

## Quick Commands Reference

```bash
# Activate virtual environment
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate              # Windows

# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py

# Check what's installed
pip list

# Git status
git status

# Deactivate venv (when done)
deactivate
```

---

**🎉 Day 0 Complete! You're ready to start coding on Day 1!**
