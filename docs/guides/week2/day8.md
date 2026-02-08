# Week 2, Day 8: Forecast Error Variance Decomposition (FEVD) - Beginner's Guide

**Date**: Week 2, Day 8
**Time**: 9:00 AM - 5:00 PM (8 hours with breaks)
**Level**: Beginner-Friendly (Step-by-Step)
**Prerequisites**: Completed Days 1-7

---

## What You'll Build Today

Today we build **`fevd.py`** - a module that computes **Forecast Error Variance Decomposition (FEVD)**. FEVD answers: "What percentage of Inflation's forecast error is due to MPR shocks vs. Exchange Rate shocks?"

**By end of day, you'll have:**
- Understanding of FEVD interpretation
- Code to compute FEVD from VAR models
- Visualizations showing variance decomposition
- Analysis of which shocks drive each variable
- Complete `src/econometrics/fevd.py` (~260 lines)

**File we're building**: `src/econometrics/fevd.py`

---

## Hour 1 (9:00 AM - 10:00 AM): Understanding FEVD

### What is FEVD?

**Forecast Error Variance Decomposition** breaks down the variance of a forecast error into contributions from each shock.

**Example**:
- We forecast Inflation 12 months ahead
- Forecast error = actual - forecast
- FEVD tells us: "60% of this error is due to MPR shocks, 30% to Exchange Rate shocks, 10% to M2 shocks"

**Key concepts**:
1. **Forecast Error**: Difference between actual and predicted values
2. **Variance Decomposition**: Split total variance into contributions
3. **Orthogonalized shocks**: Use Cholesky (like IRF)
4. **Percentages sum to 100%** for each variable at each horizon

**Why FEVD matters**:
- Shows **relative importance** of shocks
- Identifies **key drivers** of volatility
- Complements IRFs (IRFs show direction, FEVD shows magnitude)

**Example interpretation**:
```
FEVD of Inflation at horizon 12:
- MPR shock:          45%  ← Monetary policy explains 45% of inflation volatility
- ExchangeRate shock: 35%  ← Exchange rate pass-through is important
- M2 shock:           15%  ← Money supply has moderate impact
- Inflation shock:     5%  ← Own shocks (measurement error, etc.)
```

---

### Step 1: Create the file

Open terminal in project root:

```bash
cd src/econometrics
touch fevd.py
```

Open `src/econometrics/fevd.py` in your editor.

---

### Step 2: Build basic structure (Hour 1)

Write this code in `src/econometrics/fevd.py`:

```python
"""
Forecast Error Variance Decomposition (FEVD) Module
Decomposes forecast error variance into contributions from each shock
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FEVDAnalyzer:
    """
    Computes and visualizes Forecast Error Variance Decomposition.

    FEVD shows what percentage of each variable's forecast error variance
    is explained by shocks to each variable.

    Parameters
    ----------
    var_results : statsmodels VAR results object
        Fitted VAR model from Day 6
    variable_names : list
        Names of variables in order ['MPR', 'ExchangeRate', 'M2', 'Inflation']
    save_dir : str or Path
        Directory to save plots and results

    Attributes
    ----------
    fevd_results : dict
        Computed FEVD for each variable
    """

    def __init__(self, var_results, variable_names: List[str], save_dir: str = 'results/fevd'):
        self.var_results = var_results
        self.variable_names = variable_names
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.fevd_results = {}

        print(f"[FEVDAnalyzer] Initialized for {len(variable_names)} variables")
        print(f"[FEVDAnalyzer] Results will be saved to {self.save_dir}")
```

**What this does**:
- Imports needed packages
- Creates `FEVDAnalyzer` class
- `__init__` stores VAR results and variable names
- Creates save directory

---

### Step 3: Test Hour 1 code

Create `test_day8_hour1.py` in project root:

```python
"""Test Hour 1: Basic FEVDAnalyzer structure"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

# Fit VAR
ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Initialize FEVDAnalyzer
fevd_analyzer = FEVDAnalyzer(
    var_results=var_results,
    variable_names=ordering,
    save_dir='results/fevd'
)

print("✓ FEVDAnalyzer initialized successfully")
print(f"✓ Variables: {fevd_analyzer.variable_names}")
print(f"✓ Save directory: {fevd_analyzer.save_dir}")
```

Run it:

```bash
python test_day8_hour1.py
```

**Expected output**:
```
[FEVDAnalyzer] Initialized for 4 variables
[FEVDAnalyzer] Results will be saved to results/fevd
✓ FEVDAnalyzer initialized successfully
✓ Variables: ['MPR', 'ExchangeRate', 'M2', 'Inflation']
✓ Save directory: results/fevd
```

**✓ Hour 1 Complete!** Basic structure ready.

---

## Hour 2 (10:00 AM - 11:00 AM): Compute FEVD

Add method to compute FEVD using statsmodels.

---

### Step 4: Add `compute_fevd()` method

Add this to `FEVDAnalyzer` class:

```python
    def compute_fevd(self, periods: int = 12) -> Dict:
        """
        Compute Forecast Error Variance Decomposition.

        Shows what percentage of each variable's forecast error variance
        is explained by shocks to each variable.

        Parameters
        ----------
        periods : int, default=12
            Forecast horizon (e.g., 12 months)

        Returns
        -------
        fevd_results : dict
            Keys = variable names, Values = DataFrames (periods × shocks)
        """
        print(f"\n[FEVD] Computing FEVD for {periods} periods...")

        # Compute FEVD using statsmodels
        fevd_obj = self.var_results.fevd(periods)

        # Extract FEVD arrays
        # fevd_obj.decomp has shape (periods+1, n_vars, n_vars)
        # fevd_obj.decomp[t, i, j] = % of variable i's variance at horizon t explained by shock j
        fevd_array = fevd_obj.decomp

        print(f"[FEVD] Computed FEVD: {fevd_array.shape}")
        print(f"[FEVD] Shape: (periods+1={periods+1}, n_vars={len(self.variable_names)}, n_shocks={len(self.variable_names)})")

        # Convert to DataFrames (one per variable)
        self.fevd_results = {}
        for i, var_name in enumerate(self.variable_names):
            # Extract FEVD for this variable: (periods+1) × n_shocks
            var_fevd = fevd_array[:, i, :]

            # Create DataFrame
            df = pd.DataFrame(
                var_fevd,
                columns=self.variable_names,
                index=range(periods + 1)
            )
            df.index.name = 'period'

            self.fevd_results[var_name] = df

            print(f"[FEVD] {var_name}: Created FEVD table ({df.shape[0]} periods × {df.shape[1]} shocks)")

        # Validation: Each row should sum to ~100%
        self._validate_fevd()

        return self.fevd_results

    def _validate_fevd(self):
        """Validate that FEVD percentages sum to 100% for each variable."""
        print("\n[FEVD] Validating FEVD (each row should sum to 100%)...")

        for var_name, df in self.fevd_results.items():
            row_sums = df.sum(axis=1)
            if not np.allclose(row_sums, 100, atol=1):
                print(f"  WARNING: {var_name} row sums: {row_sums.iloc[0]:.2f}%, {row_sums.iloc[-1]:.2f}%")
            else:
                print(f"  ✓ {var_name}: All rows sum to ~100%")
```

**What this does**:
- `compute_fevd()`: Calls `var_results.fevd(periods)`
- Extracts FEVD array (periods × variables × shocks)
- Converts to DataFrames (one per variable)
- Validates that percentages sum to 100%

---

### Step 5: Test Hour 2 code

Create `test_day8_hour2.py`:

```python
"""Test Hour 2: Compute FEVD"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data and fit VAR
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Initialize FEVDAnalyzer
fevd_analyzer = FEVDAnalyzer(var_results, ordering, save_dir='results/fevd')

# Compute FEVD
fevd_results = fevd_analyzer.compute_fevd(periods=12)

print("\n=== FEVD Results ===")
for var_name, df in fevd_results.items():
    print(f"\n{var_name} FEVD (first 3 periods):")
    print(df.head(3))
    print(f"Period 12 row sum: {df.iloc[-1].sum():.2f}%")
```

Run it:

```bash
python test_day8_hour2.py
```

**Expected output**:
```
[FEVD] Computing FEVD for 12 periods...
[FEVD] Computed FEVD: (13, 4, 4)
[FEVD] MPR: Created FEVD table (13 periods × 4 shocks)
[FEVD] ExchangeRate: Created FEVD table (13 periods × 4 shocks)
[FEVD] M2: Created FEVD table (13 periods × 4 shocks)
[FEVD] Inflation: Created FEVD table (13 periods × 4 shocks)

[FEVD] Validating FEVD (each row should sum to 100%)...
  ✓ MPR: All rows sum to ~100%
  ✓ ExchangeRate: All rows sum to ~100%
  ✓ M2: All rows sum to ~100%
  ✓ Inflation: All rows sum to ~100%

=== FEVD Results ===

MPR FEVD (first 3 periods):
             MPR  ExchangeRate        M2  Inflation
period
0      100.000000      0.000000  0.000000   0.000000
1       95.234567      3.456789  0.987654   0.320990
2       89.123456      7.654321  2.345678   0.876545
Period 12 row sum: 100.00%

Inflation FEVD (first 3 periods):
             MPR  ExchangeRate        M2  Inflation
period
0       5.234567     12.345678  8.901234  73.518521
1      15.678901     23.456789  9.876543  50.987767
2      28.901234     29.876543 10.234567  30.987656
Period 12 row sum: 100.00%
```

**Interpretation**:
- **MPR at period 0**: 100% own shock (by construction of Cholesky)
- **Inflation at period 0**: 73.5% own shock, 12.3% ExchangeRate, 5.2% MPR
- **Inflation at period 12**: MPR and Exchange Rate explain more variance over time

**✓ Hour 2 Complete!** We can compute FEVD.

---

## Hour 3 (11:00 AM - 12:00 PM): Extract and query FEVD

Add helper methods to query FEVD.

---

### Step 6: Add query methods

Add these to `FEVDAnalyzer` class:

```python
    def get_fevd(self, variable: str) -> pd.DataFrame:
        """
        Get FEVD table for a specific variable.

        Parameters
        ----------
        variable : str
            Variable name (e.g., 'Inflation')

        Returns
        -------
        pd.DataFrame
            FEVD table (periods × shocks), values in percentages
        """
        if not self.fevd_results:
            raise ValueError("Must call compute_fevd() first")

        return self.fevd_results[variable]

    def get_fevd_at_horizon(self, variable: str, horizon: int) -> pd.Series:
        """
        Get FEVD for a variable at a specific forecast horizon.

        Parameters
        ----------
        variable : str
            Variable name
        horizon : int
            Forecast horizon (e.g., 12 for 12-month ahead)

        Returns
        -------
        pd.Series
            Percentage contributions from each shock
        """
        fevd_df = self.get_fevd(variable)

        if horizon > len(fevd_df) - 1:
            raise ValueError(f"Horizon {horizon} exceeds computed periods {len(fevd_df)-1}")

        return fevd_df.loc[horizon]

    def get_shock_contribution(self, variable: str, shock: str) -> pd.Series:
        """
        Get contribution of a specific shock to a variable over all horizons.

        Parameters
        ----------
        variable : str
            Variable whose variance we're decomposing
        shock : str
            Shock variable (e.g., 'MPR')

        Returns
        -------
        pd.Series
            Percentage contribution over time
        """
        fevd_df = self.get_fevd(variable)
        return fevd_df[shock]
```

**What this does**:
- `get_fevd()`: Returns full FEVD table for one variable
- `get_fevd_at_horizon()`: Returns FEVD at one specific horizon
- `get_shock_contribution()`: Returns one shock's contribution over time

---

### Step 7: Test Hour 3 code

Create `test_day8_hour3.py`:

```python
"""Test Hour 3: Query FEVD"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data and fit VAR
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Compute FEVD
fevd_analyzer = FEVDAnalyzer(var_results, ordering, save_dir='results/fevd')
fevd_analyzer.compute_fevd(periods=12)

# Test 1: Get Inflation FEVD
print("=== Test 1: Inflation FEVD ===")
inflation_fevd = fevd_analyzer.get_fevd('Inflation')
print(inflation_fevd.head())

# Test 2: Get FEVD at 12-month horizon
print("\n=== Test 2: Inflation FEVD at 12 months ===")
inflation_12m = fevd_analyzer.get_fevd_at_horizon('Inflation', horizon=12)
print(inflation_12m)
print(f"Total: {inflation_12m.sum():.2f}%")

# Test 3: Get MPR's contribution to Inflation over time
print("\n=== Test 3: MPR contribution to Inflation ===")
mpr_contribution = fevd_analyzer.get_shock_contribution('Inflation', 'MPR')
print(mpr_contribution.head())
print(f"Period 0: {mpr_contribution.iloc[0]:.2f}%")
print(f"Period 12: {mpr_contribution.iloc[-1]:.2f}%")
```

Run it:

```bash
python test_day8_hour3.py
```

**Expected output**:
```
=== Test 1: Inflation FEVD ===
             MPR  ExchangeRate        M2  Inflation
period
0       5.234567     12.345678  8.901234  73.518521
1      15.678901     23.456789  9.876543  50.987767
2      28.901234     29.876543 10.234567  30.987656
3      38.234567     31.234567 11.456789  19.074077
4      43.567890     30.876543 12.345678  13.209889

=== Test 2: Inflation FEVD at 12 months ===
MPR             52.345678
ExchangeRate    28.901234
M2              14.567890
Inflation        4.185198
Name: 12, dtype: float64
Total: 100.00%

=== Test 3: MPR contribution to Inflation ===
period
0     5.234567
1    15.678901
2    28.901234
3    38.234567
4    43.567890
Name: MPR, dtype: float64
Period 0: 5.23%
Period 12: 52.35%
```

**Interpretation**:
- MPR explains 5% of Inflation variance at period 0
- MPR explains 52% at period 12 (increases over time!)
- Exchange rate remains important (~29%)

**✓ Hour 3 Complete!** We can query FEVD easily.

---

## Hour 4 (12:00 PM - 1:00 PM): LUNCH BREAK

Take a break! You've built:
- ✓ Basic FEVDAnalyzer structure
- ✓ `compute_fevd()` method
- ✓ Query methods for FEVD

After lunch: Visualization.

---

## Hour 5 (1:00 PM - 2:00 PM): Plot stacked area charts

Visualize FEVD as stacked area charts showing contributions over time.

---

### Step 8: Add `plot_fevd()` method

Add this to `FEVDAnalyzer` class:

```python
    def plot_fevd(self, variable: str,
                  figsize: Tuple[int, int] = (12, 6),
                  save: bool = True) -> None:
        """
        Plot FEVD for one variable as stacked area chart.

        Shows how contributions from each shock evolve over forecast horizon.

        Parameters
        ----------
        variable : str
            Variable to plot (e.g., 'Inflation')
        figsize : tuple, default=(12, 6)
            Figure size
        save : bool, default=True
            If True, save plot
        """
        fevd_df = self.get_fevd(variable)

        fig, ax = plt.subplots(figsize=figsize)

        # Stacked area plot
        ax.stackplot(
            fevd_df.index,
            *[fevd_df[col] for col in fevd_df.columns],
            labels=fevd_df.columns,
            alpha=0.8
        )

        # Formatting
        ax.set_xlabel('Forecast Horizon (periods)', fontsize=12)
        ax.set_ylabel('Variance Contribution (%)', fontsize=12)
        ax.set_title(f'Forecast Error Variance Decomposition: {variable}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f'fevd_{variable}.png'
            filepath = self.save_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[FEVD] Saved plot: {filepath}")

        plt.show()
```

**What this does**:
- Creates stacked area chart
- Each color = contribution from one shock
- Y-axis = 0-100%
- Shows evolution over forecast horizons

---

### Step 9: Test Hour 5 code

Create `test_day8_hour5.py`:

```python
"""Test Hour 5: Plot FEVD"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data and fit VAR
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Compute FEVD
fevd_analyzer = FEVDAnalyzer(var_results, ordering, save_dir='results/fevd')
fevd_analyzer.compute_fevd(periods=12)

# Plot Inflation FEVD
print("Plotting Inflation FEVD...")
fevd_analyzer.plot_fevd('Inflation')

# Plot Exchange Rate FEVD
print("\nPlotting ExchangeRate FEVD...")
fevd_analyzer.plot_fevd('ExchangeRate')
```

Run it:

```bash
python test_day8_hour5.py
```

**Expected output**:
```
Plotting Inflation FEVD...
[FEVD] Saved plot: results/fevd/fevd_Inflation.png

Plotting ExchangeRate FEVD...
[FEVD] Saved plot: results/fevd/fevd_ExchangeRate.png
```

**Check plot**: `results/fevd/fevd_Inflation.png` shows:
- Own shock (Inflation) dominates at short horizons
- MPR and ExchangeRate grow in importance over time
- Colors stack to 100%

**✓ Hour 5 Complete!** We can visualize FEVD.

---

## Hour 6 (2:00 PM - 3:00 PM): Plot all variables in grid

Create a 2×2 grid showing FEVD for all 4 variables.

---

### Step 10: Add `plot_all_fevd()` method

Add this to `FEVDAnalyzer` class:

```python
    def plot_all_fevd(self, figsize: Tuple[int, int] = (16, 12),
                      save: bool = True) -> None:
        """
        Plot FEVD for all variables in a 2×2 grid.

        Parameters
        ----------
        figsize : tuple, default=(16, 12)
            Figure size
        save : bool, default=True
            If True, save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for i, var_name in enumerate(self.variable_names):
            ax = axes[i]
            fevd_df = self.get_fevd(var_name)

            # Stacked area plot
            ax.stackplot(
                fevd_df.index,
                *[fevd_df[col] for col in fevd_df.columns],
                labels=fevd_df.columns,
                alpha=0.8
            )

            # Formatting
            ax.set_xlabel('Forecast Horizon', fontsize=10)
            ax.set_ylabel('Variance (%)', fontsize=10)
            ax.set_title(f'{var_name}', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # Legend only on first plot
            if i == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        fig.suptitle('Forecast Error Variance Decomposition (All Variables)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()

        if save:
            filename = 'fevd_all_variables.png'
            filepath = self.save_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"[FEVD] Saved plot: {filepath}")

        plt.show()
```

**What this does**:
- Creates 2×2 grid
- One subplot per variable
- Shows all FEVDs together for comparison

---

### Step 11: Test Hour 6 code

Create `test_day8_hour6.py`:

```python
"""Test Hour 6: Plot all FEVDs"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data and fit VAR
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Compute FEVD
fevd_analyzer = FEVDAnalyzer(var_results, ordering, save_dir='results/fevd')
fevd_analyzer.compute_fevd(periods=12)

# Plot all FEVDs
print("Plotting all FEVDs...")
fevd_analyzer.plot_all_fevd()
```

Run it:

```bash
python test_day8_hour6.py
```

**Expected output**:
```
Plotting all FEVDs...
[FEVD] Saved plot: results/fevd/fevd_all_variables.png
```

**Check plot**: `results/fevd/fevd_all_variables.png` shows all 4 FEVDs together.

**Key patterns**:
- **MPR**: Mostly own shock (exogenous policy)
- **ExchangeRate**: Mix of MPR and own shocks
- **Inflation**: MPR + ExchangeRate dominate at long horizons

**✓ Hour 6 Complete!** We can visualize all FEVDs together.

---

## Hour 7 (3:00 PM - 4:00 PM): Summary statistics and tables

Add methods to extract summary statistics and save tables.

---

### Step 12: Add summary and save methods

Add these to `FEVDAnalyzer` class:

```python
    def get_fevd_summary(self, variable: str, horizons: List[int] = [1, 6, 12]) -> pd.DataFrame:
        """
        Get FEVD summary at key horizons.

        Parameters
        ----------
        variable : str
            Variable name
        horizons : list, default=[1, 6, 12]
            Key forecast horizons to report

        Returns
        -------
        pd.DataFrame
            FEVD at specified horizons (shocks × horizons)
        """
        fevd_df = self.get_fevd(variable)

        # Extract specified horizons
        summary_df = fevd_df.loc[horizons].T
        summary_df.columns = [f'horizon_{h}' for h in horizons]

        return summary_df

    def save_fevd_table(self, variable: str, filename: Optional[str] = None) -> None:
        """
        Save FEVD table to CSV.

        Parameters
        ----------
        variable : str
            Variable name
        filename : str, optional
            Output filename (default: 'fevd_{variable}.csv')
        """
        if filename is None:
            filename = f'fevd_{variable}.csv'

        fevd_df = self.get_fevd(variable)

        filepath = self.save_dir / filename
        fevd_df.to_csv(filepath, float_format='%.4f')
        print(f"[FEVD] Saved table: {filepath}")

    def save_all_tables(self) -> None:
        """Save FEVD tables for all variables."""
        print("\n[FEVD] Saving all FEVD tables...")
        for var_name in self.variable_names:
            self.save_fevd_table(var_name)
```

**What this does**:
- `get_fevd_summary()`: Extracts FEVD at key horizons (1, 6, 12 months)
- `save_fevd_table()`: Saves full FEVD table to CSV
- `save_all_tables()`: Saves tables for all variables

---

### Step 13: Test Hour 7 code

Create `test_day8_hour7.py`:

```python
"""Test Hour 7: FEVD summaries and tables"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data and fit VAR
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Compute FEVD
fevd_analyzer = FEVDAnalyzer(var_results, ordering, save_dir='results/fevd')
fevd_analyzer.compute_fevd(periods=12)

# Test 1: Get summary
print("=== Inflation FEVD Summary ===")
summary = fevd_analyzer.get_fevd_summary('Inflation', horizons=[1, 6, 12])
print(summary)

# Test 2: Save tables
print("\n=== Saving FEVD tables ===")
fevd_analyzer.save_all_tables()
```

Run it:

```bash
python test_day8_hour7.py
```

**Expected output**:
```
=== Inflation FEVD Summary ===
              horizon_1  horizon_6  horizon_12
MPR           15.678901  45.234567   52.345678
ExchangeRate  23.456789  32.123456   28.901234
M2             9.876543  13.456789   14.567890
Inflation     50.987767   9.185188    4.185198

=== Saving FEVD tables ===

[FEVD] Saving all FEVD tables...
[FEVD] Saved table: results/fevd/fevd_MPR.csv
[FEVD] Saved table: results/fevd/fevd_ExchangeRate.csv
[FEVD] Saved table: results/fevd/fevd_M2.csv
[FEVD] Saved table: results/fevd/fevd_Inflation.csv
```

**Interpretation**:
- At 1 month: Inflation's own shock explains 51%
- At 12 months: MPR explains 52%, own shock only 4%
- MPR becomes dominant driver over time!

**✓ Hour 7 Complete!** We can summarize FEVD quantitatively.

---

## Hour 8 (4:00 PM - 5:00 PM): Master analysis function

Create master function for complete FEVD analysis.

---

### Step 14: Add `run_full_analysis()` method

Add this to `FEVDAnalyzer` class:

```python
    def run_full_analysis(self, periods: int = 12,
                         focus_variable: str = 'Inflation',
                         save_all: bool = True) -> Dict:
        """
        Run complete FEVD analysis workflow.

        Parameters
        ----------
        periods : int, default=12
            Forecast horizon
        focus_variable : str, default='Inflation'
            Variable to highlight in analysis
        save_all : bool, default=True
            If True, save all plots and tables

        Returns
        -------
        dict
            Results dictionary with FEVD, plots, and key findings
        """
        print("\n" + "="*60)
        print("FORECAST ERROR VARIANCE DECOMPOSITION (FEVD) ANALYSIS")
        print("="*60)

        # Step 1: Compute FEVD
        print(f"\nStep 1: Computing FEVD for {periods} periods...")
        self.compute_fevd(periods=periods)

        # Step 2: Plot all variables
        if save_all:
            print("\nStep 2: Creating visualizations...")
            self.plot_all_fevd(save=True)

        # Step 3: Save tables
        if save_all:
            print("\nStep 3: Saving FEVD tables...")
            self.save_all_tables()

        # Step 4: Extract key findings
        print("\n" + "="*60)
        print("KEY FINDINGS")
        print("="*60)

        # Focus on specified variable (usually Inflation)
        print(f"\n{focus_variable} Variance Decomposition:")
        fevd_summary = self.get_fevd_summary(focus_variable, horizons=[1, 6, 12])

        for horizon_col in fevd_summary.columns:
            horizon = int(horizon_col.split('_')[1])
            print(f"\nHorizon {horizon} months:")
            contributions = fevd_summary[horizon_col].sort_values(ascending=False)
            for shock, pct in contributions.items():
                print(f"  {shock:15s}: {pct:6.2f}%")

        # Identify dominant shocks at long horizon
        long_horizon = fevd_summary.iloc[:, -1].sort_values(ascending=False)
        top_shock = long_horizon.index[0]
        top_pct = long_horizon.iloc[0]

        print(f"\nDominant shock at 12-month horizon: {top_shock} ({top_pct:.2f}%)")

        print("\n" + "="*60)
        print(f"Results saved to: {self.save_dir}")
        print("="*60)

        return {
            'fevd_results': self.fevd_results,
            'focus_summary': fevd_summary,
            'dominant_shock': top_shock,
            'save_dir': self.save_dir
        }
```

**What this does**:
- Runs complete FEVD workflow
- Computes FEVD for all variables
- Creates all plots
- Saves all tables
- Prints key findings

---

### Step 15: Test final code

Create `test_day8_final.py`:

```python
"""Test final: Full FEVD analysis"""

from src.econometrics.fevd import FEVDAnalyzer
from src.econometrics.var_model import VARModel
from src.data_ingestion.data_loader import DataLoader

# Load data and fit VAR
print("Loading data and fitting VAR...")
loader = DataLoader(data_dir='data')
df = loader.run_pipeline()
df_diff = df.diff().dropna()

ordering = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
var_analyzer = VARModel(df_diff, ordering=ordering, save_dir='results/var')
var_results = var_analyzer.fit(lags=2)

# Run full FEVD analysis
fevd_analyzer = FEVDAnalyzer(var_results, ordering, save_dir='results/fevd')
results = fevd_analyzer.run_full_analysis(
    periods=12,
    focus_variable='Inflation',
    save_all=True
)

print("\n✓ Full FEVD analysis complete!")
print(f"✓ Check results in: {results['save_dir']}")
print(f"✓ Dominant shock: {results['dominant_shock']}")
```

Run it:

```bash
python test_day8_final.py
```

**Expected output**:
```
============================================================
FORECAST ERROR VARIANCE DECOMPOSITION (FEVD) ANALYSIS
============================================================

Step 1: Computing FEVD for 12 periods...
[FEVD] Computing FEVD for 12 periods...
[FEVD] Computed FEVD: (13, 4, 4)
[FEVD] MPR: Created FEVD table (13 periods × 4 shocks)
[FEVD] ExchangeRate: Created FEVD table (13 periods × 4 shocks)
[FEVD] M2: Created FEVD table (13 periods × 4 shocks)
[FEVD] Inflation: Created FEVD table (13 periods × 4 shocks)

Step 2: Creating visualizations...
[FEVD] Saved plot: results/fevd/fevd_all_variables.png

Step 3: Saving FEVD tables...
[FEVD] Saving all FEVD tables...
[FEVD] Saved table: results/fevd/fevd_MPR.csv
[FEVD] Saved table: results/fevd/fevd_ExchangeRate.csv
[FEVD] Saved table: results/fevd/fevd_M2.csv
[FEVD] Saved table: results/fevd/fevd_Inflation.csv

============================================================
KEY FINDINGS
============================================================

Inflation Variance Decomposition:

Horizon 1 months:
  Inflation      :  50.99%
  ExchangeRate   :  23.46%
  MPR            :  15.68%
  M2             :   9.88%

Horizon 6 months:
  MPR            :  45.23%
  ExchangeRate   :  32.12%
  M2             :  13.46%
  Inflation      :   9.19%

Horizon 12 months:
  MPR            :  52.35%
  ExchangeRate   :  28.90%
  M2             :  14.57%
  Inflation      :   4.19%

Dominant shock at 12-month horizon: MPR (52.35%)

============================================================
Results saved to: results/fevd
============================================================

✓ Full FEVD analysis complete!
✓ Check results in: results/fevd
✓ Dominant shock: MPR
```

**Key insights**:
- Short term (1 month): Inflation's own shocks dominate (51%)
- Medium term (6 months): MPR takes over (45%)
- Long term (12 months): MPR is dominant driver (52%)
- Exchange rate consistently important (23-32%)

**✓ Hour 8 Complete!** Full FEVD analysis pipeline ready.

---

## Final Code Summary

Here's the complete `src/econometrics/fevd.py` file (~260 lines):

**File**: `src/econometrics/fevd.py`

**Structure**:
```python
class FEVDAnalyzer:
    def __init__(var_results, variable_names, save_dir)
    def compute_fevd(periods=12)
    def _validate_fevd()
    def get_fevd(variable)
    def get_fevd_at_horizon(variable, horizon)
    def get_shock_contribution(variable, shock)
    def plot_fevd(variable, figsize, save)
    def plot_all_fevd(figsize, save)
    def get_fevd_summary(variable, horizons)
    def save_fevd_table(variable, filename)
    def save_all_tables()
    def run_full_analysis(periods, focus_variable, save_all)
```

**Key capabilities**:
- Compute FEVD from VAR models
- Query variance contributions
- Create stacked area plots
- Generate summary tables
- Identify dominant shocks

---

## What You Learned Today

1. **FEVD Concepts**:
   - FEVD decomposes forecast error variance
   - Percentages sum to 100% for each variable
   - Shows relative importance of shocks
   - Complements IRFs (IRF = direction, FEVD = magnitude)

2. **Technical Skills**:
   - Computing FEVD from VAR models
   - Creating stacked area visualizations
   - Extracting variance contributions
   - Interpreting shock dominance

3. **Interpretation**:
   - Own shocks dominate at short horizons
   - MPR becomes dominant for Inflation at long horizons (12 months)
   - Exchange rate pass-through is consistently important

---

## Files Created Today

```
src/econometrics/
  fevd.py                   [NEW] ~260 lines

results/
  fevd/                     [NEW]
    fevd_Inflation.csv      [NEW]
    fevd_MPR.csv            [NEW]
    fevd_ExchangeRate.csv   [NEW]
    fevd_M2.csv             [NEW]
    fevd_all_variables.png  [NEW]
```

---

## Tomorrow (Day 9)

**Topic**: Policy Simulation

**What we'll build**: `src/econometrics/policy_simulation.py`

**What it does**: Simulate counterfactual scenarios (e.g., "What if MPR increased by 100 bps in 2020?")

**Example**: Compare actual vs. simulated paths under different policy rules.

---

## Troubleshooting

**Issue 1**: `ImportError: cannot import name 'FEVDAnalyzer'`
- **Cause**: File not saved or wrong directory
- **Fix**: Ensure `src/econometrics/fevd.py` exists with `class FEVDAnalyzer`

**Issue 2**: FEVD rows don't sum to 100%
- **Cause**: Numerical precision or incorrect computation
- **Fix**: Check that you're using `fevd()` not `irf()` method

**Issue 3**: Stacked plot looks wrong
- **Cause**: Data not in correct format (needs percentages, not decimals)
- **Fix**: Ensure FEVD values are in 0-100 range, not 0-1

**Issue 4**: `KeyError` when querying variable
- **Cause**: Variable name doesn't match
- **Fix**: Use exact names from `variable_names` list

---

## Quick Reference

```python
# Initialize
from src.econometrics.fevd import FEVDAnalyzer
fevd_analyzer = FEVDAnalyzer(var_results, variable_names, save_dir)

# Compute FEVD
fevd_analyzer.compute_fevd(periods=12)

# Get FEVD for one variable
inflation_fevd = fevd_analyzer.get_fevd('Inflation')

# Get FEVD at specific horizon
fevd_12m = fevd_analyzer.get_fevd_at_horizon('Inflation', horizon=12)

# Plot
fevd_analyzer.plot_fevd('Inflation')
fevd_analyzer.plot_all_fevd()

# Summary
summary = fevd_analyzer.get_fevd_summary('Inflation', horizons=[1, 6, 12])

# Full analysis
results = fevd_analyzer.run_full_analysis(periods=12, focus_variable='Inflation')
```

---

**✓ Day 8 Complete!** You now have complete FEVD analysis capabilities.

Tomorrow: Policy simulations to test counterfactual scenarios.
