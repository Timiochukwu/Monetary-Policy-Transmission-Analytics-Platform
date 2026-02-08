# Week 1 - Day 2: Exploratory Data Analysis & Visualization

**Time Estimate:** 8 hours (full day)
**What You'll Build:** Plotting utilities + time series visualizations
**End Goal:** Understand your data through plots and statistical summaries

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ Visualization utilities (`models/plots.py` - 180 lines)
- ✅ Time series plots for all 4 variables
- ✅ Correlation heatmap
- ✅ Statistical summaries and trend analysis
- ✅ Jupyter notebook with all visualizations

**Pre-requisite:** Day 1 completed (data loader working)

---

## Hour 1 (9 AM - 10 AM): Verify Day 1 & Create Basic Plots

### Step 1.1: Test that Day 1 still works

```bash
cd ~/Monetary-Policy-Transmission-Analytics-Platform
python models/data_loader.py
```

**Expected output:** Should load 181 observations successfully.

**If errors:** Go back to Day 1 and fix before proceeding.

---

### Step 1.2: Create visualization module - Basic structure

Create `models/plots.py` and **write this first chunk:**

```python
"""
Visualization Utilities for Nigerian Monetary Policy Analysis

This module provides plotting functions for exploratory data analysis.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


# Set default plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class TimeSeriesPlotter:
    """
    Create time series visualizations for macro data.
    """

    def __init__(self, save_dir: str = "results/plots"):
        """
        Initialize the plotter.

        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Plots will be saved to: {self.save_dir}")


# Test code
if __name__ == "__main__":
    print("Testing plotter initialization...")
    plotter = TimeSeriesPlotter()
    print("✓ Plotter initialized successfully!")
```

**Save and test:**

```bash
python models/plots.py
```

**Expected output:**
```
Testing plotter initialization...
✓ Plots will be saved to: results/plots
✓ Plotter initialized successfully!
```

---

### Step 1.3: Add single variable plot method

**Add this method to the class** (after `__init__`):

```python
    def plot_single_series(self, df: pd.DataFrame, column: str,
                          title: Optional[str] = None,
                          ylabel: Optional[str] = None,
                          save_name: Optional[str] = None):
        """
        Plot a single time series.

        Args:
            df: DataFrame with DatetimeIndex
            column: Column name to plot
            title: Plot title (optional)
            ylabel: Y-axis label (optional)
            save_name: Filename to save (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the series
        ax.plot(df.index, df[column], linewidth=2, color='steelblue')

        # Labels
        ax.set_title(title or f"{column} Over Time", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(ylabel or column, fontsize=12)

        # Grid
        ax.grid(True, alpha=0.3, linestyle='--')

        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save if requested
        if save_name:
            save_path = self.save_dir / save_name
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {save_path}")

        plt.close(fig)
```

**Update the test code at the bottom:**

```python
if __name__ == "__main__":
    print("Testing plotter...")

    # Load data using Day 1's loader
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    # Test single plot
    print("\nTesting single series plot...")
    plotter = TimeSeriesPlotter()
    plotter.plot_single_series(df, 'MPR',
                              title='Monetary Policy Rate (2010-2025)',
                              ylabel='Rate (%)',
                              save_name='mpr_single.png')

    print("\n✓ Test complete! Check results/plots/ folder")
```

**Save and test:**

```bash
python models/plots.py
```

**Expected output:**
```
[Data loader output...]
Testing single series plot...
✓ Plots will be saved to: results/plots
  ✓ Saved: results/plots/mpr_single.png

✓ Test complete! Check results/plots/ folder
```

**Verify the plot exists:**
```bash
ls -lh results/plots/
# Should show: mpr_single.png
```

**Great! Your first plot is created!**

---

## Hour 2 (10 AM - 11 AM): Add Multi-Variable Plots

### Step 2.1: Add method to plot all variables

**Add this method after `plot_single_series`:**

```python
    def plot_all_variables(self, df: pd.DataFrame, save_name: str = "all_variables.png"):
        """
        Create a 2x2 grid of plots for all 4 variables.

        Args:
            df: DataFrame with MPR, Inflation, ExchangeRate, M2
            save_name: Filename to save
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Nigerian Macro Variables (2010-2025)',
                    fontsize=16, fontweight='bold', y=0.995)

        # Define variables and their properties
        variables = [
            ('MPR', 'Monetary Policy Rate (%)', 'blue'),
            ('Inflation', 'Inflation Rate (% YoY)', 'red'),
            ('ExchangeRate', 'Exchange Rate (NGN/USD)', 'green'),
            ('M2', 'Money Supply M2 (Billions NGN)', 'purple')
        ]

        # Plot each variable
        for idx, (var, ylabel, color) in enumerate(variables):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            ax.plot(df.index, df[var], linewidth=2, color=color)
            ax.set_title(ylabel, fontsize=12, fontweight='bold')
            ax.set_xlabel('Date', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # Save
        save_path = self.save_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

        plt.close(fig)
```

**Update the test code:**

```python
if __name__ == "__main__":
    print("Testing plotter...")

    # Load data
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    plotter = TimeSeriesPlotter()

    # Test single plot
    print("\n[1/2] Testing single series plot...")
    plotter.plot_single_series(df, 'MPR',
                              title='Monetary Policy Rate (2010-2025)',
                              ylabel='Rate (%)',
                              save_name='mpr_single.png')

    # Test multi-variable plot
    print("\n[2/2] Testing all variables plot...")
    plotter.plot_all_variables(df, save_name='all_variables.png')

    print("\n✓ All tests complete! Check results/plots/ folder")
```

**Save and test:**

```bash
python models/plots.py
```

**Verify plots:**
```bash
ls -lh results/plots/
# Should show: mpr_single.png, all_variables.png
```

---

## Hour 3 (11 AM - 12 PM): Add Correlation Analysis

### Step 3.1: Add correlation heatmap method

**Add this method after `plot_all_variables`:**

```python
    def plot_correlation_heatmap(self, df: pd.DataFrame,
                                save_name: str = "correlation_heatmap.png"):
        """
        Create correlation matrix heatmap.

        Args:
            df: DataFrame with variables
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate correlation
        corr = df.corr()

        # Create heatmap
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, ax=ax)

        ax.set_title('Correlation Matrix - Nigerian Macro Variables',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save
        save_path = self.save_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

        plt.close(fig)
```

**Update test code to include correlation plot:**

```python
if __name__ == "__main__":
    print("Testing plotter...")

    # Load data
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    plotter = TimeSeriesPlotter()

    print("\n[1/3] Testing single series plot...")
    plotter.plot_single_series(df, 'MPR',
                              title='Monetary Policy Rate (2010-2025)',
                              ylabel='Rate (%)',
                              save_name='mpr_single.png')

    print("\n[2/3] Testing all variables plot...")
    plotter.plot_all_variables(df, save_name='all_variables.png')

    print("\n[3/3] Testing correlation heatmap...")
    plotter.plot_correlation_heatmap(df, save_name='correlation_heatmap.png')

    print("\n✓ All tests complete! Check results/plots/ folder")
```

**Save and test:**

```bash
python models/plots.py
```

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK 🍽️

Take a break! You've created 3 types of plots.

---

## Hour 5 (1 PM - 2 PM): Add Statistical Summary

### Step 5.1: Add summary statistics method

**Add this method after `plot_correlation_heatmap`:**

```python
    def generate_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics.

        Args:
            df: DataFrame with variables

        Returns:
            DataFrame with summary statistics
        """
        # Basic statistics
        summary = df.describe().T

        # Add additional statistics
        summary['Skewness'] = df.skew()
        summary['Kurtosis'] = df.kurtosis()
        summary['CV (%)'] = (df.std() / df.mean() * 100).round(2)

        # Add missing values
        summary['Missing'] = df.isnull().sum()
        summary['Missing %'] = (df.isnull().sum() / len(df) * 100).round(2)

        return summary

    def print_summary_stats(self, df: pd.DataFrame):
        """
        Print formatted summary statistics.

        Args:
            df: DataFrame with variables
        """
        summary = self.generate_summary_stats(df)

        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS - Nigerian Macro Variables")
        print("=" * 80)
        print(summary.to_string())
        print("=" * 80)

        # Interpretations
        print("\nKEY INSIGHTS:")
        print(f"  • MPR Range: {df['MPR'].min():.1f}% to {df['MPR'].max():.1f}%")
        print(f"  • Inflation Range: {df['Inflation'].min():.1f}% to {df['Inflation'].max():.1f}%")
        print(f"  • Exchange Rate Range: {df['ExchangeRate'].min():.1f} to {df['ExchangeRate'].max():.1f} NGN/USD")
        print(f"  • M2 Range: {df['M2'].min()/1e9:.2f}B to {df['M2'].max()/1e9:.2f}B NGN")
        print()
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing plotter...")

    # Load data
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    plotter = TimeSeriesPlotter()

    print("\n[1/4] Testing single series plot...")
    plotter.plot_single_series(df, 'MPR', save_name='mpr_single.png')

    print("\n[2/4] Testing all variables plot...")
    plotter.plot_all_variables(df, save_name='all_variables.png')

    print("\n[3/4] Testing correlation heatmap...")
    plotter.plot_correlation_heatmap(df, save_name='correlation_heatmap.png')

    print("\n[4/4] Testing summary statistics...")
    plotter.print_summary_stats(df)

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python models/plots.py
```

---

## Hour 6 (2 PM - 3 PM): Add Structural Break Detection (Visual)

### Step 6.1: Add method to highlight key events

**Add this method after `print_summary_stats`:**

```python
    def plot_with_events(self, df: pd.DataFrame, column: str,
                        save_name: str = "events_plot.png"):
        """
        Plot time series with key economic events marked.

        Args:
            df: DataFrame with DatetimeIndex
            column: Column to plot
            save_name: Filename to save
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot the main series
        ax.plot(df.index, df[column], linewidth=2, color='steelblue', label=column)

        # Key events in Nigerian economic history
        events = {
            '2016-06': ('Naira\nDevaluation', 'red'),
            '2020-03': ('COVID-19\nPandemic', 'orange'),
            '2023-06': ('FX\nUnification', 'purple')
        }

        for date_str, (label, color) in events.items():
            event_date = pd.to_datetime(date_str)
            if event_date in df.index:
                ax.axvline(event_date, color=color, linestyle='--',
                          linewidth=2, alpha=0.7, label=label)

                # Add text label
                y_pos = ax.get_ylim()[1] * 0.9
                ax.text(event_date, y_pos, label,
                       rotation=0, ha='center', fontsize=9,
                       color=color, weight='bold')

        ax.set_title(f'{column} with Key Economic Events',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(column, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save
        save_path = self.save_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {save_path}")

        plt.close(fig)
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing plotter...")

    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    plotter = TimeSeriesPlotter()

    print("\n[1/5] Single series plot...")
    plotter.plot_single_series(df, 'MPR', save_name='mpr_single.png')

    print("\n[2/5] All variables plot...")
    plotter.plot_all_variables(df, save_name='all_variables.png')

    print("\n[3/5] Correlation heatmap...")
    plotter.plot_correlation_heatmap(df, save_name='correlation_heatmap.png')

    print("\n[4/5] Summary statistics...")
    plotter.print_summary_stats(df)

    print("\n[5/5] Events plot...")
    plotter.plot_with_events(df, 'ExchangeRate', save_name='exchange_rate_events.png')

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python models/plots.py
```

**You should now see the Exchange Rate plot with vertical lines marking 2016, 2020, 2023!**

---

## Hour 7 (3 PM - 4 PM): Create Master Visualization Function

### Step 7.1: Add comprehensive plotting method

**Add this method after `plot_with_events`:**

```python
    def create_full_report(self, df: pd.DataFrame):
        """
        Generate all plots and statistics in one go.

        Args:
            df: DataFrame with macro variables
        """
        print("\n" + "=" * 80)
        print("GENERATING FULL EXPLORATORY DATA ANALYSIS REPORT")
        print("=" * 80)

        print("\n[1/6] Creating individual time series plots...")
        for col in df.columns:
            self.plot_single_series(df, col,
                                   title=f'{col} (2010-2025)',
                                   save_name=f'{col.lower()}_timeseries.png')

        print("\n[2/6] Creating combined plot...")
        self.plot_all_variables(df, save_name='all_variables.png')

        print("\n[3/6] Creating correlation heatmap...")
        self.plot_correlation_heatmap(df, save_name='correlation_heatmap.png')

        print("\n[4/6] Creating events plot for key variables...")
        self.plot_with_events(df, 'Inflation', save_name='inflation_events.png')
        self.plot_with_events(df, 'ExchangeRate', save_name='exchange_rate_events.png')

        print("\n[5/6] Generating summary statistics...")
        summary = self.generate_summary_stats(df)

        # Save summary to CSV
        summary_path = self.save_dir / 'summary_statistics.csv'
        summary.to_csv(summary_path)
        print(f"  ✓ Saved: {summary_path}")

        print("\n[6/6] Printing summary...")
        self.print_summary_stats(df)

        print("\n" + "=" * 80)
        print("✓ FULL REPORT COMPLETE!")
        print(f"✓ All outputs saved to: {self.save_dir}")
        print("=" * 80)
```

**Update the test code to use the master function:**

```python
def main():
    """
    Main execution: Generate full EDA report.
    """
    print("=" * 80)
    print("NIGERIAN MONETARY POLICY - EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Load data
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    print("\nLoading data...")
    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    # Create plotter
    plotter = TimeSeriesPlotter(save_dir="results/plots")

    # Generate full report
    plotter.create_full_report(df)


if __name__ == "__main__":
    main()
```

**Save and run:**

```bash
python models/plots.py
```

**Expected output:**
```
[Loading output...]
GENERATING FULL EXPLORATORY DATA ANALYSIS REPORT
[1/6] Creating individual time series plots...
  ✓ Saved: results/plots/mpr_timeseries.png
  ✓ Saved: results/plots/exchangerate_timeseries.png
  ✓ Saved: results/plots/m2_timeseries.png
  ✓ Saved: results/plots/inflation_timeseries.png
[2/6] Creating combined plot...
  ✓ Saved: results/plots/all_variables.png
[3/6] Creating correlation heatmap...
  ✓ Saved: results/plots/correlation_heatmap.png
[4/6] Creating events plot...
  ✓ Saved: results/plots/inflation_events.png
  ✓ Saved: results/plots/exchange_rate_events.png
[5/6] Generating summary statistics...
  ✓ Saved: results/plots/summary_statistics.csv
[6/6] Printing summary...
[Statistics output...]
✓ FULL REPORT COMPLETE!
```

**Check your plots:**
```bash
ls -lh results/plots/
# Should show ~10 PNG files + 1 CSV
```

**Excellent! Your visualization module is complete!**

---

## Hour 8 (4 PM - 5 PM): Create Jupyter Notebook & Git Commit

### Step 8.1: Create exploratory notebook

Create `notebooks/day2_exploratory_analysis.ipynb`:

```bash
jupyter notebook notebooks/
```

**In the browser, create a new notebook with these cells:**

**Cell 1 - Setup:**
```python
# Day 2: Exploratory Data Analysis
# Nigerian Monetary Policy Transmission

import sys
sys.path.append('..')

from models.data_loader import NigerianMacroDataLoader
from models.plots import TimeSeriesPlotter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

%matplotlib inline

print("✓ Libraries loaded")
```

**Cell 2 - Load Data:**
```python
# Load processed data
loader = NigerianMacroDataLoader(data_dir="../data")
df = loader.load_and_prepare()

print(f"✓ Loaded {len(df)} observations")
print(f"✓ Variables: {df.columns.tolist()}")
```

**Cell 3 - Quick Summary:**
```python
# Display first few rows
df.head(10)
```

**Cell 4 - Summary Statistics:**
```python
# Summary statistics
plotter = TimeSeriesPlotter(save_dir="../results/plots")
plotter.print_summary_stats(df)
```

**Cell 5 - Visualizations:**
```python
# Generate all plots
plotter.create_full_report(df)
```

**Cell 6 - Correlation Analysis:**
```python
# Display correlation matrix
corr = df.corr()
print("Correlation Matrix:")
print(corr.round(3))

# Key observations
print("\nKey Observations:")
print(f"  • MPR vs Inflation correlation: {corr.loc['MPR', 'Inflation']:.3f}")
print(f"  • ExchangeRate vs Inflation correlation: {corr.loc['ExchangeRate', 'Inflation']:.3f}")
print(f"  • M2 vs Inflation correlation: {corr.loc['M2', 'Inflation']:.3f}")
```

**Save the notebook** (Ctrl+S in browser)

---

### Step 8.2: Verify complete file

Check your `plots.py` has ~180 lines:

```bash
wc -l models/plots.py
# Should show: ~180 models/plots.py
```

---

### Step 8.3: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Day 2: Exploratory Data Analysis & Visualization

- Created visualization utilities (plots.py, 180 lines, built in chunks)
- Implemented 6 plot types: single series, multi-variable, correlation, events
- Generated full EDA report with 10+ plots
- Created Jupyter notebook for interactive analysis

Key visualizations:
  • Time series plots for all 4 variables
  • Correlation heatmap (MPR-Inflation: weak negative)
  • Events plot (2016 devaluation, 2020 COVID, 2023 FX unification)
  • Summary statistics (181 observations, no missing values)

Outputs: results/plots/ (10 PNG files + 1 CSV)

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## Final Code Summary

Here's the complete `models/plots.py` file (~180 lines):

**File**: `models/plots.py`

**Structure**:
```python
class TimeSeriesPlotter:
    def __init__(save_dir="results/plots")
    def plot_single_series(df, column, title, save, filename)
    def plot_all_variables(df, save_name)
    def plot_correlation_heatmap(df, save, filename)
    def generate_summary_stats(df)
    def print_summary_stats(df)
    def plot_with_events(df, column, events, title, save, filename)
    def create_full_report(df)

def main()
```

**Key capabilities**:
- Plot individual time series with customization
- Create 4-panel plot of all variables
- Correlation heatmap with annotations
- Summary statistics generation
- Event annotation (CBN rate changes)
- Full automated report generation

**Verify your file is complete:**
```bash
python -c "
from models.plots import TimeSeriesPlotter
import inspect
methods = [m for m in dir(TimeSeriesPlotter) if not m.startswith('_')]
print('Methods:', methods)
"
```

---

## End of Day 2 Checklist ✅

Before you finish, verify:

- [ ] `plots.py` runs without errors
- [ ] You see ~10 PNG files in `results/plots/`
- [ ] Correlation heatmap shows values (MPR vs Inflation = -0.3 to -0.5)
- [ ] Events plot shows vertical lines at 2016, 2020, 2023
- [ ] Jupyter notebook opens and all cells run
- [ ] Git commit successful
- [ ] You understand each plot type created

**If all boxes checked → DAY 2 COMPLETE! 🎉**

---

## What You Built Today

**Files created:** 2 (`plots.py`, Jupyter notebook)
**Lines of code:** ~180 (built in 6 chunks over 6 hours)
**Plots generated:** 10 PNG files + 1 CSV
**Skills learned:** matplotlib, seaborn, correlation analysis, event marking

**Key learning:** Visual inspection reveals:
- 2016: Exchange rate spike (devaluation)
- 2020: Inflation rise (COVID supply shocks)
- 2023: Exchange rate jump (FX unification)
- MPR and Inflation have weak negative correlation (surprising? Will investigate with VAR later)

---

## Tomorrow (Day 3): Stationarity Testing

**What you'll build:**
- Stationarity tests module (`models/stationarity.py`)
- ADF, PP, KPSS tests
- Integration order classification (I(0) vs I(1))

**New package to install:**
```bash
pip install statsmodels==0.14.1 scipy==1.11.4
```

**Time:** 8 hours
**Difficulty:** Same as Days 1-2

---

## Troubleshooting

**"Matplotlib not showing plots in Jupyter"**
```python
# Add this to first cell:
%matplotlib inline
```

**"Plots look ugly"**
```python
# Increase DPI in your plots.py:
plt.rcParams['figure.dpi'] = 150
```

**"Events not showing on plot"**
```python
# Check your data covers those dates:
print(df.index.min(), df.index.max())
# Should span 2010-01 to 2025-01
```

**"Correlation heatmap values look wrong"**
```python
# Verify data loaded correctly:
print(df.describe())
# All variables should have non-zero std dev
```

---

**Excellent work! See you tomorrow for Day 3! 💪**
