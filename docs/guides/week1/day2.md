# Week 1 - Day 2: Data Validation & Exploratory Data Analysis

**Time Estimate:** 8 hours (full day)
**What You'll Build:** Data validator + 9 publication-quality visualizations
**End Goal:** Verify data quality and understand every variable visually

**Files you'll create today:**
- `src/data_ingestion/data_validator.py` (~270 lines)
- `src/visualization/plots.py` (~380 lines)

**Pre-requisite:** Day 1 completed (`src/data_ingestion/data_loader.py` working)

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ Data validator (`src/data_ingestion/data_validator.py` - 270 lines)
- ✅ 9 professional time-series plots for your thesis (`src/visualization/plots.py` - 380 lines)
- ✅ Structural-break annotations (2016 devaluation, 2020 COVID, 2023 FX unification)
- ✅ Publication-quality PNG files saved at 300 DPI

---

## Hour 1 (9 AM - 10 AM): Verify Day 1 & Build DataValidator Structure

### Step 1.1: Test that Day 1 still works

```bash
cd ~/Monetary-Policy-Transmission-Analytics-Platform
python src/data_ingestion/data_loader.py
```

**Expected output:** Should load observations successfully.

**If errors:** Go back to Day 1 and fix before proceeding.

---

### Step 1.2: Create data validator — Basic structure

Create `src/data_ingestion/data_validator.py` and **write this first chunk:**

```python
"""
Data Validation Module for Nigerian Monetary Policy Analysis

Performs quality checks, outlier detection, and data integrity validation
for macroeconomic time series data.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings


class DataValidator:
    """
    Validates macroeconomic data quality for econometric analysis.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with DataFrame.

        Args:
            df: DataFrame with macroeconomic variables
        """
        self.df = df
        self.validation_report = {}
        print(f"✓ Validator initialized with {len(df)} observations")


# Test code
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    validator = DataValidator(df)
    print("✓ DataValidator initialized successfully!")
```

**Save and test:**

```bash
python src/data_ingestion/data_validator.py
```

---

## Hour 2 (10 AM - 11 AM): Add Missing Values & Outlier Checks

### Step 2.1: Add missing values check method

**Add this method to the class** (after `__init__`):

```python
    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in each variable.

        Returns:
            Dictionary with count of missing values per variable
        """
        missing_counts = self.df.isnull().sum().to_dict()
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100).round(2).to_dict()

        self.validation_report['missing_values'] = {
            'counts': missing_counts,
            'percentages': missing_pct
        }

        return missing_counts
```

### Step 2.2: Add outlier detection method

**Add this method after `check_missing_values`:**

```python
    def detect_outliers(self, threshold: float = 3.0) -> Dict[str, List[Tuple]]:
        """
        Detect outliers using z-score method (|z| > threshold).

        Args:
            threshold: Number of standard deviations for outlier detection

        Returns:
            Dictionary with outliers per variable
        """
        outliers = {}

        for col in self.df.columns:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outlier_indices = z_scores[z_scores > threshold].index
            outlier_values = [(idx.strftime('%Y-%m'), self.df.loc[idx, col])
                             for idx in outlier_indices]

            if outlier_values:
                outliers[col] = outlier_values

        self.validation_report['outliers'] = outliers
        return outliers
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    validator = DataValidator(df)

    print("\n[1/2] Checking missing values...")
    missing = validator.check_missing_values()
    for var, count in missing.items():
        print(f"  {var}: {count} missing")

    print("\n[2/2] Detecting outliers (3 sigma)...")
    outliers = validator.detect_outliers()
    if outliers:
        for var, vals in outliers.items():
            print(f"  {var}: {len(vals)} outliers detected")
    else:
        print("  No outliers detected")
```

**Save and test:**

```bash
python src/data_ingestion/data_validator.py
```

---

## Hour 3 (11 AM - 12 PM): Add Range, Temporal & Monotonicity Checks

### Step 3.1: Add value range, temporal, and monotonicity methods

**Add these three methods after `detect_outliers`:**

```python
    def check_value_ranges(self) -> Dict[str, Dict]:
        """
        Check if values are within expected economic ranges.
        """
        # Expected ranges (based on Nigerian economic history)
        expected_ranges = {
            'MPR': (0, 50),
            'Inflation': (-5, 100),
            'ExchangeRate': (1, 2000),
            'M2': (0, 1e10)
        }

        range_checks = {}
        for col in self.df.columns:
            if col in expected_ranges:
                min_val, max_val = expected_ranges[col]
                actual_min = self.df[col].min()
                actual_max = self.df[col].max()
                in_range = (actual_min >= min_val) and (actual_max <= max_val)
                range_checks[col] = {
                    'expected': expected_ranges[col],
                    'actual': (actual_min, actual_max),
                    'valid': in_range
                }

        self.validation_report['range_checks'] = range_checks
        return range_checks

    def check_temporal_consistency(self) -> Dict[str, bool]:
        """
        Check that dates are properly ordered and no duplicates exist.
        """
        temporal_checks = {}
        temporal_checks['no_duplicate_dates'] = not self.df.index.duplicated().any()
        temporal_checks['dates_sorted'] = self.df.index.is_monotonic_increasing
        freq = pd.infer_freq(self.df.index)
        temporal_checks['inferred_frequency'] = freq if freq else "Irregular"

        self.validation_report['temporal_consistency'] = temporal_checks
        return temporal_checks

    def check_monotonicity(self, variables: List[str] = None) -> Dict[str, str]:
        """
        Check if M2 is mostly increasing (economic expectation).
        """
        if variables is None:
            variables = ['M2'] if 'M2' in self.df.columns else []

        monotonicity = {}
        for var in variables:
            if var not in self.df.columns:
                continue
            diff = self.df[var].diff()
            increasing_pct = (diff > 0).sum() / len(diff) * 100
            decreasing_pct = (diff < 0).sum() / len(diff) * 100

            if increasing_pct > 80:
                monotonicity[var] = "Mostly increasing (expected for M2)"
            elif decreasing_pct > 80:
                monotonicity[var] = "Mostly decreasing (unusual for M2)"
            else:
                monotonicity[var] = f"Mixed ({increasing_pct:.1f}% increasing)"

        self.validation_report['monotonicity'] = monotonicity
        return monotonicity
```

**Save and test:**

```bash
python src/data_ingestion/data_validator.py
```

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK

Take a break! You've completed the data validator checks.

---

## Hour 5 (1 PM - 2 PM): Add Master Validation & Start Visualization Module

### Step 5.1: Add validate_all() and get_summary() to DataValidator

**Add these final methods to the class:**

```python
    def validate_all(self) -> Dict:
        """
        Run all validation checks and print formatted report.
        """
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)

        print("\n[1/5] Checking missing values...")
        missing = self.check_missing_values()
        if sum(missing.values()) == 0:
            print("  No missing values detected")
        else:
            for var, count in missing.items():
                if count > 0:
                    pct = self.validation_report['missing_values']['percentages'][var]
                    print(f"    - {var}: {count} ({pct}%)")

        print("\n[2/5] Detecting outliers (3 sigma threshold)...")
        outliers = self.detect_outliers()
        if not outliers:
            print("  No significant outliers detected")
        else:
            for var, outlier_list in outliers.items():
                print(f"    - {var}: {len(outlier_list)} outliers")
                for date, value in outlier_list[:3]:
                    print(f"      {date}: {value:.2f}")

        print("\n[3/5] Checking value ranges...")
        ranges = self.check_value_ranges()
        for var, check in ranges.items():
            status = "OK" if check['valid'] else "WARNING"
            print(f"  [{status}] {var}: {check['actual'][0]:.2f} to {check['actual'][1]:.2f}")

        print("\n[4/5] Checking temporal consistency...")
        temporal = self.check_temporal_consistency()
        for check, result in temporal.items():
            print(f"  {check}: {result}")

        print("\n[5/5] Checking monotonicity...")
        monotonic = self.check_monotonicity()
        for var, status in monotonic.items():
            print(f"  - {var}: {status}")

        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        return self.validation_report

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of key statistics.
        """
        summary_data = []
        for col in self.df.columns:
            row = {
                'Variable': col,
                'N_Obs': len(self.df[col]),
                'Missing': self.df[col].isnull().sum(),
                'Min': round(self.df[col].min(), 2),
                'Max': round(self.df[col].max(), 2),
                'Mean': round(self.df[col].mean(), 2),
                'Std': round(self.df[col].std(), 2)
            }
            summary_data.append(row)
        return pd.DataFrame(summary_data)


def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    validator = DataValidator(df)
    validator.validate_all()

    print("\nSummary table:")
    print(validator.get_summary().to_string(index=False))


if __name__ == "__main__":
    main()
```

**Save and run final validator test:**

```bash
python src/data_ingestion/data_validator.py
```

**DataValidator is complete. Now move to visualizations.**

---

### Step 5.2: Create visualization module — Structure & constants

Create `src/visualization/plots.py` and **write this first chunk:**

```python
"""
Visualization Module — Nigerian Monetary Policy Transmission Analysis

Generates publication-quality plots for EDA and thesis presentation.
All figures save at 300 DPI into the results/ tree.

Day 2 deliverable.  Dependencies: pandas, numpy, matplotlib, seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# PROJECT-WIDE CONSTANTS  (single source of truth for every plot)
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    "MPR":            "#2E86AB",
    "Inflation":      "#A23B72",
    "ExchangeRate":   "#F18F01",
    "M2":             "#6A994E",
}

LABELS = {                          # (title, y-axis unit)
    "MPR":            ("Monetary Policy Rate",         "% per annum"),
    "Inflation":      ("Headline Inflation (CPI)",     "% YoY"),
    "ExchangeRate":   ("Exchange Rate",                "NGN / USD"),
    "M2":             ("Broad Money Supply (M2)",      "N Billions"),
}

# Nigerian structural-break episodes
EVENTS = {
    "2016-06-01": ("2016 Devaluation",    "#E74C3C"),
    "2020-03-01": ("COVID-19",            "#E67E22"),
    "2023-06-01": ("2023 FX Unification", "#8E44AD"),
}


def _style_ax(ax, title, ylabel, xlabel=None):
    """Apply a consistent look to every Axes object."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


class MacroPlotter:
    """
    Publication-quality plotting for Nigerian macro data.

    Parameters
    ----------
    df       : pd.DataFrame  - columns MPR | ExchangeRate | M2 | Inflation
    save_dir : str | Path    - root folder for saved images (e.g. "results")
    """

    def __init__(self, df: pd.DataFrame, save_dir: str = "results"):
        self.df = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update({"font.size": 11, "figure.dpi": 100})
        print(f"Plots will be saved to: {self.save_dir}")

    def _save(self, fig, filename: str):
        path = self.save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved  {path}")

    def _draw_events(self, ax, label_top=True):
        """Overlay vertical lines for structural-break episodes."""
        ymin, ymax = ax.get_ylim()
        for date_str, (label, color) in EVENTS.items():
            ts = pd.Timestamp(date_str)
            if ts < self.df.index[0] or ts > self.df.index[-1]:
                continue
            ax.axvline(ts, color=color, linestyle="--", alpha=0.6, linewidth=1.4)
            if label_top:
                ax.text(
                    ts, ymax * 0.97, label,
                    fontsize=8, color=color, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
                )


# Test code
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    plotter = MacroPlotter(df, save_dir="results")
    print("MacroPlotter initialized successfully!")
```

**Save and test:**

```bash
python src/visualization/plots.py
```

---

## Hour 6 (2 PM - 3 PM): Add Time-Series & Key-Relationship Plots

### Step 6.1: Add the first four plot methods

**Add these methods inside the class** (after `_draw_events`):

```python
    def plot_time_series(self) -> plt.Figure:
        """Individual time-series panel for every variable."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            "Nigerian Macroeconomic Variables (2010-2025)",
            fontsize=16, fontweight="bold", y=1.01,
        )

        for ax, col in zip(axes, self.df.columns):
            title, ylabel = LABELS.get(col, (col, ""))
            ax.plot(self.df.index, self.df[col],
                    color=COLORS.get(col, "#333"), linewidth=2)
            ax.axhline(self.df[col].mean(),
                       color="red", linestyle="--", alpha=0.4, linewidth=1,
                       label=f"Mean {self.df[col].mean():.1f}")
            _style_ax(ax, title, ylabel)
            ax.legend(loc="upper left", fontsize=9)
            self._draw_events(ax, label_top=(ax is axes[0]))

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "time_series_levels.png")
        return fig

    def plot_mpr_vs_inflation(self) -> plt.Figure:
        """Key-relationship overlay: MPR vs Inflation."""
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(self.df.index, self.df["MPR"],
                label="Monetary Policy Rate", color=COLORS["MPR"], linewidth=2.5)
        ax.plot(self.df.index, self.df["Inflation"],
                label="Inflation", color=COLORS["Inflation"], linewidth=2.5)

        self._draw_events(ax)
        ax.set_title("MPR vs Inflation — Transmission Dynamics",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Percent (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        self._save(fig, "mpr_vs_inflation.png")
        return fig

    def plot_rolling_statistics(self, window: int = 12) -> plt.Figure:
        """12-month rolling mean +/- 1 sigma band for every variable."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        fig.suptitle(f"Rolling Statistics ({window}-Month Window)",
                     fontsize=15, fontweight="bold", y=1.01)

        for ax, col in zip(axes, self.df.columns):
            title, ylabel = LABELS.get(col, (col, ""))
            raw  = self.df[col]
            mean = raw.rolling(window).mean()
            std  = raw.rolling(window).std()

            ax.plot(self.df.index, raw,  color=COLORS[col], alpha=0.25, linewidth=1, label="Raw")
            ax.plot(self.df.index, mean, color=COLORS[col], linewidth=2.5,
                    label=f"{window}-mo mean")
            ax.fill_between(self.df.index, mean - std, mean + std,
                            color=COLORS[col], alpha=0.12, label="+/- 1 sigma")

            _style_ax(ax, f"{title} - rolling", ylabel)
            ax.legend(loc="upper left", fontsize=9)

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "rolling_statistics.png")
        return fig

    def plot_first_differences(self) -> plt.Figure:
        """Month-on-month change bar chart."""
        df_d = self.df.diff().dropna()
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        fig.suptitle("First Differences (Month-on-Month Change)",
                     fontsize=15, fontweight="bold", y=1.01)

        for ax, col in zip(axes, df_d.columns):
            title, ylabel = LABELS.get(col, (col, ""))
            vals = df_d[col].values
            colours = [COLORS.get(col, "#333") if v >= 0 else "#BDBDBD" for v in vals]

            ax.bar(df_d.index, vals, width=18, color=colours, edgecolor="none")
            ax.axhline(0, color="black", linewidth=0.7)
            _style_ax(ax, f"Delta {title}", f"Delta {ylabel}")

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "first_differences.png")
        return fig
```

**Update test code:**

```python
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    plotter = MacroPlotter(df, save_dir="results")

    print("\n[1/4] Time series levels...")
    plotter.plot_time_series()
    plt.close("all")

    print("\n[2/4] MPR vs Inflation...")
    plotter.plot_mpr_vs_inflation()
    plt.close("all")

    print("\n[3/4] Rolling statistics...")
    plotter.plot_rolling_statistics()
    plt.close("all")

    print("\n[4/4] First differences...")
    plotter.plot_first_differences()
    plt.close("all")

    print("\n4 plots saved!")
```

**Save and test:**

```bash
python src/visualization/plots.py
ls results/*.png
```

---

## Hour 7 (3 PM - 4 PM): Add Remaining 5 Plots

### Step 7.1: Add distributions, correlation, scatter, normalised & volatility

**Add these methods after `plot_first_differences`:**

```python
    def plot_distributions(self) -> plt.Figure:
        """Histogram + mean/median lines for each variable."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Variable Distributions", fontsize=15, fontweight="bold")

        for idx, col in enumerate(self.df.columns):
            ax = axes[idx // 2, idx % 2]
            title, xlabel = LABELS.get(col, (col, ""))

            ax.hist(self.df[col], bins=28, color=COLORS.get(col, "#333"),
                    edgecolor="black", alpha=0.55, linewidth=0.6)

            mn, md = self.df[col].mean(), self.df[col].median()
            ax.axvline(mn, color="red",       linestyle="--", linewidth=1.8,
                       label=f"Mean   {mn:.1f}")
            ax.axvline(md, color="darkgreen", linestyle=":",  linewidth=1.8,
                       label=f"Median {md:.1f}")

            _style_ax(ax, title, "Frequency", xlabel)
            ax.legend(fontsize=9)

        fig.tight_layout()
        self._save(fig, "distributions.png")
        return fig

    def plot_correlation_heatmap(self) -> plt.Figure:
        """Annotated Pearson correlation matrix."""
        corr = self.df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, linewidths=1.5, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "correlation_heatmap.png")
        return fig

    def plot_scatter_pairs(self) -> plt.Figure:
        """Four economically-motivated scatter plots with Pearson r."""
        pairs = [
            ("MPR",          "Inflation",    "MPR -> Inflation (target)"),
            ("ExchangeRate", "Inflation",    "Exchange Rate -> Inflation (pass-through)"),
            ("MPR",          "ExchangeRate", "MPR -> Exchange Rate (interest parity)"),
            ("M2",           "Inflation",    "M2 -> Inflation (quantity theory)"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("Key Pairwise Relationships", fontsize=15, fontweight="bold")

        for ax, (x, y, title) in zip(axes.flat, pairs):
            ax.scatter(self.df[x], self.df[y],
                       color="#2C3E50", alpha=0.55, edgecolors="white", s=55, zorder=3)

            # OLS trend line
            coeffs = np.polyfit(self.df[x], self.df[y], 1)
            x_line = np.linspace(self.df[x].min(), self.df[x].max(), 120)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    "r--", linewidth=2, alpha=0.7, zorder=2)

            # Pearson r badge
            r = self.df[x].corr(self.df[y])
            ax.text(0.04, 0.94, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

            _style_ax(ax, title,
                      LABELS.get(y, (y, ""))[0],
                      LABELS.get(x, (x, ""))[0])

        fig.tight_layout()
        self._save(fig, "scatter_pairs.png")
        return fig

    def plot_normalised_overlay(self) -> plt.Figure:
        """All four variables rescaled to [0, 1] — reveals relative moves."""
        normed = (self.df - self.df.min()) / (self.df.max() - self.df.min())

        fig, ax = plt.subplots(figsize=(14, 6))
        for col in normed.columns:
            ax.plot(normed.index, normed[col],
                    color=COLORS[col], linewidth=1.8,
                    label=LABELS.get(col, (col, ""))[0])

        self._draw_events(ax)
        ax.set_title("Normalised Variables (0-1 scale)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalised value", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        self._save(fig, "normalised_overlay.png")
        return fig

    def plot_volatility(self, window: int = 12) -> plt.Figure:
        """Rolling std of monthly % change — measures uncertainty."""
        pct = self.df.pct_change().dropna() * 100
        vol = pct.rolling(window).std()

        fig, ax = plt.subplots(figsize=(14, 5))
        for col in vol.columns:
            ax.plot(vol.index, vol[col],
                    color=COLORS[col], linewidth=1.8,
                    label=LABELS.get(col, (col, ""))[0])

        self._draw_events(ax)
        ax.set_title(f"Rolling Volatility ({window}-Month Std Dev of Monthly % Change)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Volatility (% pts)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        self._save(fig, "volatility.png")
        return fig

    def generate_all(self):
        """Run every plot method in sequence."""
        print("=" * 60)
        print("  GENERATING ALL EDA VISUALISATIONS")
        print("=" * 60)

        steps = [
            ("1/9  Time series (levels)",   self.plot_time_series),
            ("2/9  MPR vs Inflation",       self.plot_mpr_vs_inflation),
            ("3/9  Rolling statistics",     self.plot_rolling_statistics),
            ("4/9  First differences",      self.plot_first_differences),
            ("5/9  Distributions",          self.plot_distributions),
            ("6/9  Correlation heatmap",    self.plot_correlation_heatmap),
            ("7/9  Scatter pairs",          self.plot_scatter_pairs),
            ("8/9  Normalised overlay",     self.plot_normalised_overlay),
            ("9/9  Volatility",             self.plot_volatility),
        ]

        for label, fn in steps:
            print(f"\n  [{label}]")
            fn()
            plt.close("all")

        print("\n" + "=" * 60)
        print("  ALL 9 PLOTS SAVED")
        print("=" * 60)


def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    plotter = MacroPlotter(df, save_dir="results")
    plotter.generate_all()


if __name__ == "__main__":
    main()
```

**Save and run the master function:**

```bash
python src/visualization/plots.py
```

**Expected output:**
```
GENERATING ALL EDA VISUALISATIONS
  [1/9  Time series (levels)]
  saved  results/time_series_levels.png
  [2/9  MPR vs Inflation]
  saved  results/mpr_vs_inflation.png
  ...
  [9/9  Volatility]
  saved  results/volatility.png
  ALL 9 PLOTS SAVED
```

---

## Hour 8 (4 PM - 5 PM): Verify & Git Commit

### Step 8.1: Verify file sizes

```bash
wc -l src/data_ingestion/data_validator.py
# Should show: ~270 lines

wc -l src/visualization/plots.py
# Should show: ~380 lines

ls -lh results/*.png
# Should show: 9 PNG files
```

---

### Step 8.2: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Day 2: Data validation + 9 publication-quality EDA plots

- src/data_ingestion/data_validator.py (270 lines)
  - check_missing_values, detect_outliers, check_value_ranges
  - check_temporal_consistency, check_monotonicity
  - validate_all() master pipeline
- src/visualization/plots.py (380 lines)
  - MacroPlotter class with 9 plot methods
  - Structural-break annotations (2016, 2020, 2023)
  - 300-DPI PNG output

Key findings:
  - No missing values in dataset
  - M2 monotonically increasing (as expected)
  - MPR vs Inflation weak negative correlation
  - 2016 devaluation and 2023 FX unification clearly visible

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## Final Code Summary

### File 1: `src/data_ingestion/data_validator.py`

```python
class DataValidator:
    def __init__(df)
    def check_missing_values()
    def detect_outliers(threshold=3.0)
    def check_value_ranges()
    def check_temporal_consistency()
    def check_monotonicity(variables=None)
    def validate_all()
    def get_summary()

def main()
```

### File 2: `src/visualization/plots.py`

```python
# Module-level: COLORS, LABELS, EVENTS, _style_ax()

class MacroPlotter:
    def __init__(df, save_dir="results")
    def _save(fig, filename)
    def _draw_events(ax, label_top=True)
    def plot_time_series()
    def plot_mpr_vs_inflation()
    def plot_rolling_statistics(window=12)
    def plot_first_differences()
    def plot_distributions()
    def plot_correlation_heatmap()
    def plot_scatter_pairs()
    def plot_normalised_overlay()
    def plot_volatility(window=12)
    def generate_all()

def main()
```

**Verify both files are complete:**
```bash
python -c "
from src.data_ingestion.data_validator import DataValidator
from src.visualization.plots import MacroPlotter
print('DataValidator methods:', [m for m in dir(DataValidator) if not m.startswith('_')])
print('MacroPlotter methods:', [m for m in dir(MacroPlotter) if not m.startswith('_')])
"
```

---

## End of Day 2 Checklist

Before you finish, verify:

- [ ] `data_validator.py` runs without errors
- [ ] `plots.py` runs without errors
- [ ] You see 9 PNG files in `results/`
- [ ] Correlation heatmap shows MPR vs Inflation value
- [ ] Normalised overlay shows 2016, 2020, 2023 event lines
- [ ] Git commit successful

**If all boxes checked -> DAY 2 COMPLETE!**

---

## What You Built Today

**Files created:** 2 (`data_validator.py`, `plots.py`)
**Lines of code:** ~650
**Plots generated:** 9 publication-ready PNG files at 300 DPI
**Skills learned:** Data validation, matplotlib, seaborn, structural-break annotation

**Key insight from plots:**
- 2016: Exchange rate spike (Naira devaluation)
- 2020: Inflation rise (COVID-19 supply shocks)
- 2023: Massive exchange rate jump (FX unification)
- MPR and Inflation have weak/negative correlation (VAR will reveal the lag structure)

---

## Tomorrow (Day 3): Stationarity Testing

**What you'll build:**
- `src/econometrics/stationarity_tests.py`
- ADF + Phillips-Perron + KPSS tests on levels AND first differences
- Integration order classification: I(0) vs I(1) for each variable

**New package to install:**
```bash
pip install statsmodels==0.14.1 scipy==1.11.4
```

---

## Troubleshooting

**"seaborn-v0_8-darkgrid style not found"**
```bash
# Check matplotlib version
python -c "import matplotlib; print(matplotlib.__version__)"
# If older, replace 'seaborn-v0_8-darkgrid' with 'seaborn-darkgrid'
```

**"Plots not saving"**
```python
# Ensure results/ directory exists
import os
os.makedirs("results", exist_ok=True)
```

**"ImportError for seaborn"**
```bash
pip install seaborn --upgrade
```

---

**Well done! See you tomorrow for Day 3!**
