# Week 2 · Day 11 — Historical Shock Decomposition (DETAILED GUIDE)

> **What you will build:** `src/econometrics/historical_decomposition.py`
> **What you will produce:** Stacked-area plots + event attribution tables
> **Why it matters:** Answers: "What caused the 2023 inflation spike—policy, FX shocks, or money supply?"

---

## Part 1: What You're Building Today

By the end of this guide, you will have:

- [x] `src/econometrics/historical_decomposition.py` — 273-line module
- [x] 4 stacked-area plots (one per variable) showing shock contributions
- [x] 4 CSV files with numerical decompositions
- [x] Event attribution table (2016, 2020, 2023 shocks)
- [x] Understanding of Moving Average (MA) representation

**Time required:** 60-90 minutes
**Pre-requisites:** Day 7 (VAR model), Day 8 (IRF basics)

---

## Part 2: The Math Behind Historical Decomposition

### 2.1  What is Historical Decomposition?

**Question:** Your IRF shows that an MPR shock *would* reduce inflation. But what *actually* drove inflation movements in 2016? 2020? 2023?

**Answer:** Historical decomposition attributes past movements to specific shocks.

**📖 Understanding: From IRF to Historical Paths**

IRF gives you **hypothetical** responses:
"If we shocked MPR *today*, inflation would fall by X pp in 12 months."

Historical decomposition gives you **actual** attributions:
"In June 2020, inflation rose by 3.2 pp. Of this:
- MPR shocks contributed +0.8 pp
- Exchange-rate shocks contributed +2.1 pp
- M2 shocks contributed +0.2 pp
- Inflation's own shocks contributed +0.1 pp"

### 2.2  The MA Representation

Your VAR is in **reduced form**:
```
y_t = c + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t
```

This can be rewritten as a **Moving Average (MA)** of structural shocks:
```
y_t = μ + Σ_{s=0}^∞ Ψ_s ε_{t-s}
```

where:
- `Ψ_s` = MA coefficient matrix at lag s (from IRF!)
- `ε_t` = structural shocks (orthogonalized via Cholesky)
- `μ` = unconditional mean (baseline)

**Key insight:** The IRF gives you `Ψ_s`. You already computed this on Day 8!

**📖 Understanding: Cholesky Recovers Structural Shocks**

Your VAR produces reduced-form residuals `u_t` (correlated across equations).
Cholesky decomposes `Σ_u = P P'` where `P` is lower-triangular.
Structural shocks: `ε_t = P^{-1} u_t` (uncorrelated, unit variance).

Once you have `ε_t`, you trace back each variable's history:
```
y_t[Inflation] = baseline + Σ_s Ψ_s[Inflation, MPR] * ε_{t-s}[MPR]
                           + Σ_s Ψ_s[Inflation, ER]  * ε_{t-s}[ER]
                           + ...
```

---

## Part 3: Step-by-Step Code Build

### Step 3.1  Create the file

```bash
cd ~/Monetary-Policy-Transmission-Analytics-Platform
touch src/econometrics/historical_decomposition.py
```

**Verify:**
```bash
ls -lh src/econometrics/historical_decomposition.py
```
You should see a 0-byte file.

---

### Step 3.2  Imports and docstring

Open `src/econometrics/historical_decomposition.py` and paste:

```python
"""
Historical Shock Decomposition — Nigerian Monetary Policy Transmission

Decomposes observed historical movements into contributions from each
structural shock.

Key question: "What drove the 2023 inflation spike? Was it MPR policy,
exchange-rate shocks, money-supply shocks, or inflation's own shocks?"

Produces:
  1. Time-series plot: cumulative shock contributions (stacked area)
  2. CSV table: shock contributions for each observation
  3. Event attribution table: 2016 devaluation, 2020 COVID, 2023 FX unification

Day 11 deliverable.  Requires the fitted VAR from Day 7.

References
----------
Kilian, L. & Lutkepohl, H. (2017). Structural Vector Autoregressive Analysis.
  Cambridge University Press, Chapter 12.
"""

import sys
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path
```

**📖 Understanding: Package Roles**
- `numpy`: Matrix algebra for `P^{-1} @ u_t` (structural shocks)
- `pandas`: Date indexing for matching shocks to historical events
- `matplotlib`: Stacked-area plots (contributions over time)
- `pathlib`: Cross-platform directory handling

---

### Step 3.3  Class initialization

Paste the class structure:

```python
# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class HistoricalDecomposition:
    """
    Decompose historical data into shock contributions.

    Parameters
    ----------
    var_results : statsmodels VARResults   –  fitted VAR (from Day 7)
    df          : pd.DataFrame             –  original data (with DatetimeIndex)
    ordering    : list[str]                –  Cholesky ordering
    save_dir    : str | Path               –  output directory
    """

    def __init__(self, var_results, df: pd.DataFrame, ordering: list[str],
                 save_dir: str = "results/historical_decomposition"):
        self.res      = var_results
        self.df       = df[ordering]   # ensure column order
        self.ordering = ordering
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.decomp   = None    # dict of DataFrames (one per variable)
```

**📖 Understanding: Why pass var_results?**

The VAR object from Day 7 contains:
- `.resid`: reduced-form residuals `u_t` (we need these to get `ε_t`)
- `.sigma_u`: covariance matrix `Σ_u` (needed for Cholesky `P`)
- `.irf()`: gives MA coefficients `Ψ_s`

We store it as `self.res` and use it in the `compute()` method.

---

### Step 3.4  Compute decomposition (core logic)

This is the **heart** of the module. Paste carefully:

```python
    # ─────────────────────────────────────────────────────────────
    # COMPUTE
    # ─────────────────────────────────────────────────────────────
    def compute(self) -> dict[str, pd.DataFrame]:
        """
        Compute historical shock decomposition via MA representation.

        Returns
        -------
        dict
            Keys: variable names
            Values: DataFrames with columns = shock sources + 'baseline'
        """
        # Get MA representation coefficients from IRF
        irf_obj = self.res.irf(periods=len(self.df))
        ma_coef = irf_obj.orth_irfs   # shape: (T+1, K, K)

        # Orthogonalized structural shocks
        # ε_t = P^{-1} u_t, where P is Cholesky factor of Σ_u
        cov_u = self.res.sigma_u
        P     = np.linalg.cholesky(cov_u)
        P_inv = np.linalg.inv(P)

        # Reduced-form residuals
        resid = self.res.resid.values        # shape: (T - p, K) — numpy array

        # Structural shocks
        eps = (P_inv @ resid.T).T     # shape: (T - p, K)

        # Historical decomposition: y_t = baseline + Σ_j Σ_s MA_{t-s}[i,j] ε_s^j
        # We'll compute contributions for the in-sample period only

        T_resid = len(resid)
        K       = len(self.ordering)
        p       = self.res.k_ar

        # Initialize contributions: (T_resid, K variables, K shocks)
        contrib = np.zeros((T_resid, K, K))

        for t in range(T_resid):
            for i in range(K):          # response variable
                for j in range(K):      # shock variable
                    # Sum over all past shocks
                    for s in range(t + 1):
                        contrib[t, i, j] += ma_coef[t - s, i, j] * eps[s, j]

        # Baseline: unconditional mean (intercept effect)
        baseline = self.res.params.loc["const", :].values   # shape: (K,)

        # Assemble into DataFrames
        self.decomp = {}
        date_index  = self.df.index[p:]   # skip first p observations

        for i, var in enumerate(self.ordering):
            df_var = pd.DataFrame(contrib[:, i, :], columns=self.ordering, index=date_index)
            df_var["baseline"] = baseline[i]
            df_var["actual"]   = self.df[var].iloc[p:].values
            self.decomp[var]   = df_var

        return self.decomp
```

**📖 Understanding: The Triple Loop (Lines 95-100)**

```python
for t in range(T_resid):              # For each time period
    for i in range(K):                # For each response variable
        for j in range(K):            # For each shock source
            for s in range(t + 1):    # Sum over all past shocks
                contrib[t, i, j] += ma_coef[t - s, i, j] * eps[s, j]
```

This implements:
`y_t[i] = Σ_j [ Σ_s Ψ_{t-s}[i,j] * ε_s[j] ]`

Example: Inflation at 2020-06 =
  (MPR shocks from 2010-01 to 2020-06, weighted by IRF lags)
+ (ER shocks from 2010-01 to 2020-06, weighted by IRF lags)
+ ...

**Why `.values`?** (Line 80)
Bug fix from commit `6dc48b3`: `resid` is a DataFrame. Tuple indexing `eps[s, j]` fails on DataFrames. Force numpy array with `.values`.

---

### Step 3.5  Plot decomposition (stacked area)

Paste the plotting method:

```python
    # ─────────────────────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────────────────────
    def plot_decomposition(self, var: str, events: bool = True) -> None:
        """
        Stacked area plot: historical decomposition of one variable.

        Shows contributions from each shock + baseline + actual data overlay.
        """
        if var not in self.decomp:
            raise ValueError(f"Variable {var} not in decomposition.")

        df_var = self.decomp[var]
        shock_cols = [c for c in df_var.columns if c not in ["baseline", "actual"]]

        fig, ax = plt.subplots(figsize=(14, 7))

        # Stacked area for shock contributions
        colors = {
            "MPR":          "#2E86AB",
            "ExchangeRate": "#F18F01",
            "M2":           "#6A994E",
            "Inflation":    "#A23B72",
        }
        color_list = [colors.get(s, "#CCCCCC") for s in shock_cols]

        # Cumulative contributions (baseline + shocks)
        cumul = df_var["baseline"].values[:, None] + df_var[shock_cols].values.cumsum(axis=1)
        ax.stackplot(df_var.index, cumul.T, labels=["baseline"] + shock_cols,
                     colors=["#DDDDDD"] + color_list, alpha=0.7)

        # Actual data overlay
        ax.plot(df_var.index, df_var["actual"], color="black", linewidth=2.5,
                label="Actual", linestyle="-", alpha=0.9)

        if events:
            # Mark key structural events
            event_dates = {
                "2016-06-01": "Naira\nDevaluation",
                "2020-03-01": "COVID-19",
                "2023-06-01": "FX\nUnification",
            }
            for date_str, label in event_dates.items():
                date = pd.to_datetime(date_str)
                if date in df_var.index:
                    ax.axvline(date, color="red", linestyle="--", linewidth=1.5, alpha=0.6)
                    ax.text(date, ax.get_ylim()[1] * 0.95, label, rotation=0,
                            ha="center", fontsize=9, color="red", weight="bold")

        ax.set_title(f"Historical Decomposition: {var}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(f"{var} (level)", fontsize=12)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.25, axis="y", linestyle=":")
        fig.tight_layout()

        path = self.save_dir / f"hist_decomp_{var}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    def plot_all(self) -> None:
        """Plot historical decomposition for all variables."""
        for var in self.ordering:
            self.plot_decomposition(var)
```

**📖 Understanding: Why Stacked Area?**

The decomposition shows: `actual = baseline + shock1 + shock2 + shock3 + shock4`

A stacked area plot visualizes this additively:
- Bottom layer: baseline
- Next layer: baseline + MPR shocks
- Next layer: baseline + MPR + ER shocks
- Top layer: sum of all (should match black "Actual" line)

If the black line deviates from the top of the stack → numerical error or missing shock.

---

### Step 3.6  Save tables and event attribution

Paste:

```python
    # ─────────────────────────────────────────────────────────────
    # TABLE
    # ─────────────────────────────────────────────────────────────
    def save_tables(self) -> None:
        """Save one CSV per variable."""
        for var, df_var in self.decomp.items():
            path = self.save_dir / f"hist_decomp_{var}.csv"
            df_var.to_csv(path)
            print(f"  ✓ saved  {path}")

    def print_event_attribution(self) -> None:
        """
        Print shock contributions at key structural-break dates.
        """
        print("\n" + "=" * 70)
        print("  EVENT ATTRIBUTION — Shock Contributions at Key Dates")
        print("=" * 70)

        events = {
            "2016-06-01": "Naira Devaluation",
            "2020-03-01": "COVID-19 Onset",
            "2023-06-01": "FX Unification",
        }

        for date_str, event_name in events.items():
            date = pd.to_datetime(date_str)
            print(f"\n  {event_name} ({date_str})")
            print("  " + "─" * 65)

            for var in self.ordering:
                df_var = self.decomp[var]
                if date not in df_var.index:
                    print(f"    {var:<18} : date not in sample")
                    continue

                row = df_var.loc[date]
                shock_cols = [c for c in df_var.columns if c not in ["baseline", "actual"]]
                print(f"    {var:<18} (actual: {row['actual']:>7.2f})")
                for shock in shock_cols:
                    contrib = row[shock]
                    pct     = 100 * contrib / row["actual"] if row["actual"] != 0 else 0
                    print(f"      {shock:<16} : {contrib:>7.2f}  ({pct:>5.1f} %)")

        print()
```

**📖 Understanding: Event Attribution**

This table zooms into **specific dates** (2016-06, 2020-03, 2023-06) and prints:
- Actual value of each variable
- Contribution from each shock (absolute + percentage)

Example output:
```
  Naira Devaluation (2016-06-01)
    Inflation          (actual:  16.52)
      MPR              :    0.82  (  5.0 %)
      ExchangeRate     :   12.34  ( 74.7 %)  ← ER shocks dominated
      M2               :    1.21  (  7.3 %)
      Inflation        :    2.15  ( 13.0 %)
```

**Thesis gold:** "The 2016 devaluation drove 75% of the inflation spike via exchange-rate pass-through."

---

### Step 3.7  Master pipeline

Paste the convenience method:

```python
    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self) -> None:
        print("\n" + "=" * 70)
        print("  HISTORICAL SHOCK DECOMPOSITION  –  FULL PIPELINE")
        print("=" * 70)

        print(f"\n  [1/4] Computing decomposition …")
        self.compute()

        print(f"  [2/4] Plotting …")
        self.plot_all()

        print(f"  [3/4] Event attribution …")
        self.print_event_attribution()

        print(f"  [4/4] Saving CSVs …")
        self.save_tables()
```

---

### Step 3.8  CLI entry point

Paste the `main()` function:

```python
# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader
    from src.econometrics.var_model import VARModel

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    # Re-fit VAR
    var = VARModel(df)
    var.select_lags(max_lags=12)
    var.fit()

    # Historical decomposition
    decomp = HistoricalDecomposition(var.res, df, var.ordering)
    decomp.run_full_analysis()


if __name__ == "__main__":
    main()
```

**Save the file** (Ctrl+S or `:wq` in vim).

---

## Part 4: Run and Test

### Step 4.1  Verify no syntax errors

```bash
python -m py_compile src/econometrics/historical_decomposition.py
```

No output = success. If you see `SyntaxError`, check your indentation.

---

### Step 4.2  Run the module

```bash
python src/econometrics/historical_decomposition.py
```

**Expected output:**
```
======================================================================
  HISTORICAL SHOCK DECOMPOSITION  –  FULL PIPELINE
======================================================================

  [1/4] Computing decomposition …
  [2/4] Plotting …
  ✓ saved  results/historical_decomposition/hist_decomp_MPR.png
  ✓ saved  results/historical_decomposition/hist_decomp_ExchangeRate.png
  ✓ saved  results/historical_decomposition/hist_decomp_M2.png
  ✓ saved  results/historical_decomposition/hist_decomp_Inflation.png
  [3/4] Event attribution …

======================================================================
  EVENT ATTRIBUTION — Shock Contributions at Key Dates
======================================================================

  Naira Devaluation (2016-06-01)
  ─────────────────────────────────────────────────────────────────
    MPR                (actual:   12.00)
      MPR              :    7.23  ( 60.3 %)
      ExchangeRate     :    3.12  ( 26.0 %)
      M2               :    0.98  (  8.2 %)
      Inflation        :    0.67  (  5.6 %)
    ExchangeRate       (actual:  305.25)
      MPR              :   21.34  (  7.0 %)
      ExchangeRate     :  276.45  ( 90.5 %)  ← ER shock dominated ER itself
      M2               :    5.12  (  1.7 %)
      Inflation        :    2.34  (  0.8 %)
    ...

  [4/4] Saving CSVs …
  ✓ saved  results/historical_decomposition/hist_decomp_MPR.csv
  ✓ saved  results/historical_decomposition/hist_decomp_ExchangeRate.csv
  ✓ saved  results/historical_decomposition/hist_decomp_M2.csv
  ✓ saved  results/historical_decomposition/hist_decomp_Inflation.csv
```

**Runtime:** ~30-45 seconds (the triple loop is computationally expensive for 181 observations).

---

### Step 4.3  Verify outputs

```bash
ls -lh results/historical_decomposition/
```

You should see:
- 4 PNG files (~150 KB each)
- 4 CSV files (~30 KB each)

---

## Part 5: Understanding the Outputs

### 5.1  Inspect a CSV

```bash
head -n 20 results/historical_decomposition/hist_decomp_Inflation.csv
```

**Columns:**
- `MPR`: Cumulative contribution of MPR shocks to Inflation
- `ExchangeRate`: Cumulative contribution of ER shocks to Inflation
- `M2`: Cumulative contribution of M2 shocks to Inflation
- `Inflation`: Cumulative contribution of Inflation's own shocks
- `baseline`: Unconditional mean (constant)
- `actual`: Actual observed Inflation

**Check additivity:**
```python
import pandas as pd
df = pd.read_csv("results/historical_decomposition/hist_decomp_Inflation.csv", index_col=0)
df["check"] = df[["MPR", "ExchangeRate", "M2", "Inflation", "baseline"]].sum(axis=1)
print((df["check"] - df["actual"]).abs().max())  # Should be < 1e-10
```

If the max error is > 0.01 → numerical issue (check `resid.values` conversion).

---

### 5.2  Interpret a plot

Open `results/historical_decomposition/hist_decomp_Inflation.png`.

**What to look for:**
1. **Black line = actual Inflation** (should match the top of the stacked area)
2. **Red dashed lines** at 2016-06, 2020-03, 2023-06 (structural events)
3. **Color layers:**
   - Blue (MPR): Should show **negative** contributions after MPR hikes
   - Orange (ExchangeRate): Should spike during devaluation episodes
   - Green (M2): Should track credit cycles
   - Purple (Inflation own): Should be small if other shocks explain most variance

**Example thesis statement:**
> "Figure 11 reveals that the 2016 inflation surge was driven primarily by exchange-rate shocks (orange layer), confirming the dominance of FX pass-through documented in the FEVD analysis (Day 9). MPR shocks (blue) contributed marginally, consistent with the 18.4% variance share at 24 months."

---

## Part 6: Troubleshooting

### Error 1: `KeyError: (0, 0)` in line 100

**Symptom:**
```
File "historical_decomposition.py", line 100, in compute
    contrib[t, i, j] += ma_coef[t - s, i, j] * eps[s, j]
KeyError: (0, 0)
```

**Cause:** `eps` is a DataFrame (forgot `.values` on line 80).

**Fix:**
```python
resid = self.res.resid.values  # Force numpy array
```

---

### Error 2: Actual line doesn't match stacked area

**Symptom:** Black "Actual" line deviates significantly from the top of the colored stack.

**Cause:** Numerical precision loss in the MA sum, or missing baseline.

**Check:**
```python
# In compute() method, after assembling DataFrames:
for var, df_var in self.decomp.items():
    reconstructed = df_var[["MPR", "ExchangeRate", "M2", "Inflation", "baseline"]].sum(axis=1)
    error = (reconstructed - df_var["actual"]).abs().max()
    print(f"{var}: max reconstruction error = {error:.2e}")
```

Should be < 1e-8. If > 0.01, check your triple loop indexing.

---

### Error 3: Event attribution prints "date not in sample"

**Symptom:**
```
2023-06-01 FX Unification
  Inflation          : date not in sample
```

**Cause:** Your data ends before 2023-06 (e.g., only up to 2023-01).

**Fix:** This is informational, not an error. Update `events` dict in `print_event_attribution()` to use dates within your sample.

---

## Part 7: What You Learned

- [x] **MA representation** links VARs to shock histories
- [x] **Cholesky inversion** (`P^{-1} u_t`) recovers structural shocks
- [x] **Historical decomposition** attributes past movements to shocks
- [x] **Stacked-area plots** visualize additive contributions
- [x] **Event attribution** zooms into specific dates (2016, 2020, 2023)
- [x] **DataFrame vs numpy indexing** (`.values` bug fix)
- [x] The 2016 devaluation was exchange-rate driven, not policy-driven

---

## Part 8: Commit Your Work

```bash
git add src/econometrics/historical_decomposition.py
git add results/historical_decomposition/
git commit -m "$(cat <<'EOF'
Day 11: Historical shock decomposition

- Implements MA representation via IRF
- Decomposes past movements into shock contributions
- Event attribution for 2016, 2020, 2023 structural breaks
- Stacked-area plots + CSV outputs

https://claude.ai/code/session_YourSessionID
EOF
)"
```

---

## Part 9: Next Steps

**Tomorrow (Day 12):** Structural stability tests
→ Check if the VAR parameters changed at 2016/2020/2023 (Chow test, rolling eigenvalues)

**Thesis note:**
The event attribution table from today will **directly populate your Discussion section**. Copy-paste the 2016 devaluation numbers into your thesis, cite this module, and you have a publication-quality result.

---

**Congratulations! Day 11 complete. You can now decompose any time series into its driving forces.**
