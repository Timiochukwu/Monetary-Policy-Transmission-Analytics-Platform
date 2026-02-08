# Week 2 - Day 6B (Supplementary): Vector Error Correction Model (VECM)

**Time Estimate:** 5 hours
**What You'll Build:** VECM module as a robustness check alongside VAR
**End Goal:** Model long-run equilibrium AND short-run dynamics jointly
**File we're building:** `src/econometrics/vecm.py`

> **When to use this guide:**
> - You completed Day 4 (cointegration) and Johansen test found rank ≥ 1
> - Your variables include at least 2 that are I(1) (MPR, ExchangeRate)
> - Your examiner asks: "Did you consider VECM given cointegration?"
>
> **Relationship to VAR (Day 6):**
> VECM does NOT replace VAR. It supplements it.
> VECM = VAR in differences + error correction term (speed of adjustment to equilibrium)

---

## What You're Building Today

By end of session, you will have:

- ✅ VECM estimation module (`src/econometrics/vecm.py` — ~280 lines)
- ✅ Johansen rank selection (how many cointegrating relationships)
- ✅ Cointegrating vectors (the long-run equation: what equilibrium looks like)
- ✅ Adjustment speeds (how fast variables return to equilibrium)
- ✅ Short-run dynamics from the ECM representation
- ✅ Results saved to `results/vecm/`

---

## Hour 1 (9 AM - 10 AM): Understand VECM Theory

### Step 1.1: What problem does VECM solve?

**The VAR dilemma with I(1) variables:**

- Variables like MPR and ExchangeRate are I(1) — they wander randomly
- VAR on levels with I(1) → spurious regression risk
- VAR on differences → loses long-run information
- **VECM solves this:** use differences BUT add an error correction term

**The VECM equation:**

```
ΔMPR(t)         = α₁·ECT(t-1) + Σ Γ₁ᵢ·ΔX(t-i) + ε₁(t)
ΔExchangeRate(t) = α₂·ECT(t-1) + Σ Γ₂ᵢ·ΔX(t-i) + ε₂(t)
```

Where:
- **ECT** = Error Correction Term = β'X(t-1) = the long-run equilibrium residual
- **α** = adjustment speed — how fast each variable corrects deviations
- **Γ** = short-run coefficients (same as VAR in differences)

**Intuition:**
- If MPR is too high relative to its long-run relationship with ExchangeRate (ECT > 0)
- Then α pulls MPR back down toward equilibrium next period
- The **sign and size of α** tells you WHO adjusts and HOW FAST

---

### Step 1.2: When is VECM valid?

| Condition | Check |
|-----------|-------|
| Variables must be I(1) | ✓ MPR, ExchangeRate are I(1) (Day 3) |
| Cointegration must exist | ✓ Johansen found rank ≥ 1 (Day 4) |
| Same integration order | ⚠️ M2, Inflation are I(0) — problem for full system |

**For this project:**
- Run VECM on the **I(1) subset**: `['MPR', 'ExchangeRate']` (bivariate)
- Or include all 4 variables and note the I(0) limitation
- **Primary analysis**: ARDL (Day 5) — handles mixed orders correctly
- **VECM**: robustness check and additional insight on adjustment dynamics

---

### Step 1.3: Create the module skeleton

Create `src/econometrics/vecm.py` and write this first chunk:

```python
"""
Vector Error Correction Model (VECM) for Nigerian Monetary Policy Analysis

This module estimates VECM for cointegrated I(1) variables to capture
both short-run dynamics and long-run equilibrium adjustment.

Use Case:
    - Day 4 confirmed cointegration among I(1) variables (MPR, ExchangeRate)
    - VECM captures how fast variables return to long-run equilibrium
    - Supplements VAR (Day 6) with explicit long-run structure

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank


class VECMAnalyzer:
    """
    Vector Error Correction Model for cointegrated systems.

    VECM = VAR in differences + error correction terms.
    Models both short-run dynamics and long-run equilibrium.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with I(1) variables (date index)
    variable_names : list of str
        Variables to include (should all be I(1))
    coint_rank : int, optional
        Number of cointegrating relationships. If None, determined by Johansen.
    save_dir : str
        Directory to save results
    """

    def __init__(self, df: pd.DataFrame, variable_names: List[str],
                 coint_rank: Optional[int] = None,
                 save_dir: str = "results/vecm"):
        self.df = df[variable_names].copy()
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
        self.coint_rank = coint_rank
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = None
        self._k_ar_diff = 1


# Test skeleton
if __name__ == "__main__":
    print("VECM module loaded successfully.")
    print("Run after Day 4 cointegration confirms I(1) variables are cointegrated.")
```

**Save and test:**
```bash
python src/econometrics/vecm.py
```

**Expected:**
```
VECM module loaded successfully.
Run after Day 4 cointegration confirms I(1) variables are cointegrated.
```

---

## Hour 2 (10 AM - 11 AM): Rank Selection and Model Estimation

### Step 2.1: Add `select_rank()` method

**Add this method inside the class, after `__init__`:**

```python
    def select_rank(self, k_ar_diff: int = 1,
                    det_order: int = -1) -> pd.DataFrame:
        """
        Use Johansen trace test to determine cointegration rank.

        Parameters
        ----------
        k_ar_diff : int
            Lags in the differenced VAR (VAR lag order minus 1)
        det_order : int
            Deterministic terms: -1 (none), 0 (restricted constant), 1 (unrestricted)

        Returns
        -------
        pd.DataFrame with rank test results
        """
        print(f"\n[Rank Selection] Testing cointegration rank...")
        print(f"  Variables: {self.variable_names}")
        print(f"  Lags (k_ar_diff): {k_ar_diff}")

        rank_result = select_coint_rank(
            self.df,
            det_order=det_order,
            k_ar_diff=k_ar_diff,
            method='trace',
            signif=0.05
        )

        # Build results table
        rows = []
        for i in range(self.n_vars):
            rows.append({
                'Null Hypothesis': f'rank ≤ {i}',
                'Trace Statistic': round(rank_result.test_stats[i], 4),
                'Critical Value (5%)': round(rank_result.crit_vals[i], 4),
                'Reject H0': rank_result.test_stats[i] > rank_result.crit_vals[i]
            })

        results_df = pd.DataFrame(rows)

        # Determine rank: highest rank where H0 is rejected
        rejected = results_df[results_df['Reject H0'] == True]
        if len(rejected) > 0:
            selected_rank = len(rejected)
        else:
            selected_rank = 0

        print(f"\n  Trace test results:")
        print(results_df.to_string(index=False))
        print(f"\n  Selected rank: {selected_rank}")

        if selected_rank == 0:
            print("  ⚠ No cointegration found at 5% level.")
            print("    Use VAR in differences instead of VECM.")
        elif selected_rank == self.n_vars:
            print("  ⚠ Full rank — all variables stationary? Check I(1) assumption.")
        else:
            print(f"  ✓ {selected_rank} cointegrating relationship(s) found.")
            print(f"    Proceeding with VECM(rank={selected_rank})")

        # Store rank if not already set
        if self.coint_rank is None:
            self.coint_rank = selected_rank
            print(f"  → coint_rank set to {selected_rank}")

        return results_df
```

**Update the test block at the bottom:**
```python
if __name__ == "__main__":
    print("Testing VECM rank selection...")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    # Use only I(1) variables for VECM
    i1_vars = ['MPR', 'ExchangeRate']
    analyzer = VECMAnalyzer(df, i1_vars)
    rank_df = analyzer.select_rank(k_ar_diff=1)
    print("\nRank selection complete.")
```

**Save and test:**
```bash
python src/econometrics/vecm.py
```

**Expected output (approximate):**
```
Testing VECM rank selection...
[Data loader output...]

[Rank Selection] Testing cointegration rank...
  Variables: ['MPR', 'ExchangeRate']
  Lags (k_ar_diff): 1

  Trace test results:
  Null Hypothesis  Trace Statistic  Critical Value (5%)  Reject H0
        rank ≤ 0          XX.XXXX              15.4947       True
        rank ≤ 1           X.XXXX               3.8415      False

  Selected rank: 1
  ✓ 1 cointegrating relationship(s) found.
  → coint_rank set to 1
```

---

### Step 2.2: Add `estimate()` method

**Add this method after `select_rank()`:**

```python
    def estimate(self, k_ar_diff: int = 1,
                 det_order: int = -1) -> None:
        """
        Estimate the VECM model.

        Parameters
        ----------
        k_ar_diff : int
            Number of lagged differences in the model.
            Equivalent to (VAR lag order - 1).
            Set to optimal VAR lag - 1 from Day 6's select_lags().
        det_order : int
            Deterministic terms specification:
            -1 = no constant or trend
             0 = restricted constant (most common for macro)
             1 = unrestricted constant
        """
        if self.coint_rank is None:
            print("⚠ coint_rank not set. Running select_rank() first...")
            self.select_rank(k_ar_diff=k_ar_diff, det_order=det_order)

        if self.coint_rank == 0:
            raise ValueError(
                "Cannot estimate VECM with rank=0 (no cointegration). "
                "Use VAR in differences (Day 6) instead."
            )

        print(f"\n[Estimation] Fitting VECM...")
        print(f"  Variables: {self.variable_names}")
        print(f"  Cointegration rank: {self.coint_rank}")
        print(f"  Lagged differences: {k_ar_diff}")

        # Deterministic term string for statsmodels
        det_map = {-1: 'nc', 0: 'ci', 1: 'co'}
        det_string = det_map.get(det_order, 'ci')

        model = VECM(
            self.df,
            k_ar_diff=k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic=det_string
        )
        self.results = model.fit()
        self._k_ar_diff = k_ar_diff

        print(f"  ✓ VECM estimated successfully.")
        print(f"  Log-likelihood: {self.results.llf:.4f}")
```

**Update the test block:**
```python
if __name__ == "__main__":
    print("Testing VECM estimation...")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    i1_vars = ['MPR', 'ExchangeRate']
    analyzer = VECMAnalyzer(df, i1_vars)
    analyzer.select_rank(k_ar_diff=1)
    analyzer.estimate(k_ar_diff=1)
    print("\nEstimation complete.")
```

**Save and test:**
```bash
python src/econometrics/vecm.py
```

---

## Hour 3 (11 AM - 12 PM): Extract and Interpret Results

### Step 3.1: Add `get_cointegrating_vectors()` method

This is the **long-run equilibrium equation** — the most important output for your thesis.

**Add after `estimate()`:**

```python
    def get_cointegrating_vectors(self) -> pd.DataFrame:
        """
        Extract normalized cointegrating vectors (beta matrix).

        The cointegrating vector β defines the long-run equilibrium:
            β' * X(t) = 0  (in equilibrium)

        For example, with MPR and ExchangeRate:
            β₁*MPR + β₂*ExchangeRate = 0  →  MPR = -β₂/β₁ * ExchangeRate

        Returns
        -------
        pd.DataFrame : beta matrix (variables × cointegrating equations)
        """
        if self.results is None:
            raise RuntimeError("Run estimate() first.")

        beta = self.results.beta
        df_beta = pd.DataFrame(
            beta,
            index=self.variable_names,
            columns=[f'CE{i+1}' for i in range(self.coint_rank)]
        )

        print("\n[Cointegrating Vectors (β)]")
        print("These define the long-run equilibrium relationships:")
        print(df_beta.round(4).to_string())
        print("\nInterpretation:")
        for j in range(self.coint_rank):
            print(f"  CE{j+1}: ", end="")
            terms = []
            for i, var in enumerate(self.variable_names):
                coef = df_beta.iloc[i, j]
                if abs(coef) > 1e-10:
                    terms.append(f"{coef:.4f}×{var}")
            print(" + ".join(terms) + " = 0")

        return df_beta
```

---

### Step 3.2: Add `get_adjustment_speeds()` method

This tells you **which variable adjusts** when equilibrium is disturbed.

**Add after `get_cointegrating_vectors()`:**

```python
    def get_adjustment_speeds(self) -> pd.DataFrame:
        """
        Extract adjustment speed coefficients (alpha matrix).

        Alpha (α) measures how fast each variable corrects deviations
        from long-run equilibrium.

        Rules of thumb:
        - α = -0.1 means 10% of deviation corrected per month
        - α should be negative for the 'adjusting' variable
        - α ≈ 0 means the variable does NOT adjust (exogenous)

        Returns
        -------
        pd.DataFrame : alpha matrix (variables × cointegrating equations)
        """
        if self.results is None:
            raise RuntimeError("Run estimate() first.")

        alpha = self.results.alpha
        df_alpha = pd.DataFrame(
            alpha,
            index=self.variable_names,
            columns=[f'CE{i+1}' for i in range(self.coint_rank)]
        )

        print("\n[Adjustment Speeds (α)]")
        print("How fast each variable corrects deviations from equilibrium:")
        print(df_alpha.round(4).to_string())
        print("\nInterpretation:")
        for i, var in enumerate(self.variable_names):
            for j in range(self.coint_rank):
                a = df_alpha.iloc[i, j]
                half_life = abs(np.log(0.5) / np.log(1 + a)) if abs(a) > 0.001 else np.inf
                direction = "corrects toward equilibrium" if a < 0 else "moves away (unstable)"
                print(f"  {var} (CE{j+1}): α={a:.4f} → {direction}, "
                      f"half-life ≈ {half_life:.1f} months")

        return df_alpha
```

---

### Step 3.3: Add `print_summary()` method

```python
    def print_summary(self) -> None:
        """Print full VECM estimation summary."""
        if self.results is None:
            raise RuntimeError("Run estimate() first.")

        print("\n" + "=" * 70)
        print("VECM ESTIMATION RESULTS")
        print("=" * 70)
        print(f"Variables:          {', '.join(self.variable_names)}")
        print(f"Observations:       {len(self.df)}")
        print(f"Cointegration rank: {self.coint_rank}")
        print(f"Lagged differences: {self._k_ar_diff}")
        print(f"Log-likelihood:     {self.results.llf:.4f}")
        print("=" * 70)

        # Cointegrating vectors
        self.get_cointegrating_vectors()

        # Adjustment speeds
        self.get_adjustment_speeds()

        print("\n" + "=" * 70)
        print("FULL STATSMODELS SUMMARY:")
        print("=" * 70)
        print(self.results.summary())
```

**Update test block:**
```python
if __name__ == "__main__":
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    i1_vars = ['MPR', 'ExchangeRate']
    analyzer = VECMAnalyzer(df, i1_vars)
    analyzer.select_rank(k_ar_diff=1)
    analyzer.estimate(k_ar_diff=1)
    analyzer.print_summary()
```

**Save and test — this is the main economics output:**
```bash
python src/econometrics/vecm.py
```

**What to look for in the output:**
- **Cointegrating vector**: Shows the long-run relationship (e.g., MPR = 0.03 × ExchangeRate)
- **α for MPR**: Should be negative if MPR adjusts to equilibrium
- **α for ExchangeRate**: Sign tells you if exchange rate is the "adjusting" variable
- **Half-life**: How many months to correct 50% of a deviation

---

## Hour 4 (12 PM - 2 PM): Plotting and Comparison

### Step 4.1: LUNCH BREAK (12-1 PM)

---

### Step 4.2: Add `plot_cointegrating_relation()` method

**Add this after `print_summary()`:**

```python
    def plot_cointegrating_relation(self, ce_index: int = 0,
                                    save: bool = True) -> None:
        """
        Plot the cointegrating relation (ECT) over time.

        If ECT oscillates around zero → cointegration confirmed visually.
        Persistent deviation → equilibrium being restored.

        Parameters
        ----------
        ce_index : int
            Which cointegrating equation to plot (0-indexed)
        save : bool
            Whether to save figure to results/vecm/
        """
        if self.results is None:
            raise RuntimeError("Run estimate() first.")

        # Compute ECT = β'X
        beta = self.results.beta[:, ce_index]
        ect = self.df.values @ beta

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Cointegrating Relation CE{ce_index + 1}\n'
                     f'Variables: {", ".join(self.variable_names)}',
                     fontsize=13, fontweight='bold')

        # ECT over time
        axes[0].plot(self.df.index, ect, color='navy', linewidth=1.5)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, label='Equilibrium')
        axes[0].fill_between(self.df.index, ect, 0,
                              where=(ect > 0), alpha=0.3, color='red', label='Above equilibrium')
        axes[0].fill_between(self.df.index, ect, 0,
                              where=(ect < 0), alpha=0.3, color='blue', label='Below equilibrium')
        axes[0].set_title('Error Correction Term (ECT) — should oscillate around zero')
        axes[0].set_ylabel('ECT value')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Individual variables (normalized)
        for var in self.variable_names:
            normalized = (self.df[var] - self.df[var].mean()) / self.df[var].std()
            axes[1].plot(self.df.index, normalized, linewidth=1.2, label=var)
        axes[1].set_title('Variables (normalized) — cointegrated series move together')
        axes[1].set_ylabel('Standardized value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.save_dir / f'cointegrating_relation_CE{ce_index + 1}.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {path}")

        plt.show()
```

---

### Step 4.3: Add `compare_with_var()` method

This is critical for your thesis — show that VECM and VAR give consistent short-run dynamics.

```python
    def compare_with_var(self, var_results=None) -> pd.DataFrame:
        """
        Compare VECM short-run coefficients with VAR results.

        Short-run dynamics in VECM should be similar to VAR in differences.
        This validates both models.

        Returns
        -------
        pd.DataFrame with comparison summary
        """
        if self.results is None:
            raise RuntimeError("Run estimate() first.")

        print("\n[VECM vs VAR Comparison]")
        print("Short-run VECM coefficients (Γ matrices):")

        # VECM short-run coefficients (gamma)
        # gamma is stored as blocks of lags
        comparison_rows = []

        for lag in range(self._k_ar_diff):
            print(f"\n  Lag {lag + 1} coefficients:")
            if hasattr(self.results, 'gamma') and self.results.gamma is not None:
                gamma = self.results.gamma
                # gamma shape: (n_vars, n_vars * k_ar_diff)
                n = self.n_vars
                gamma_lag = gamma[:, lag * n:(lag + 1) * n]
                df_gamma = pd.DataFrame(
                    gamma_lag,
                    index=[f'Δ{v}' for v in self.variable_names],
                    columns=[f'Δ{v}(t-{lag+1})' for v in self.variable_names]
                )
                print(df_gamma.round(4).to_string())
                comparison_rows.append(df_gamma)

        print("\n  Key check: VECM short-run ≈ VAR in differences coefficients")
        print("  → If similar, both models are consistent ✓")

        return pd.concat(comparison_rows) if comparison_rows else pd.DataFrame()
```

---

## Hour 5 (2 PM - 3 PM): Complete Pipeline and Save Results

### Step 5.1: Add `save_results()` method

```python
    def save_results(self) -> None:
        """Save all VECM results to results/vecm/ directory."""
        if self.results is None:
            raise RuntimeError("Run estimate() first.")

        print(f"\n[Saving Results] to {self.save_dir}/")

        # 1. Cointegrating vectors
        beta_df = self.get_cointegrating_vectors()
        beta_df.to_csv(self.save_dir / 'cointegrating_vectors.csv')
        print(f"  ✓ cointegrating_vectors.csv")

        # 2. Adjustment speeds
        alpha_df = self.get_adjustment_speeds()
        alpha_df.to_csv(self.save_dir / 'adjustment_speeds.csv')
        print(f"  ✓ adjustment_speeds.csv")

        # 3. Model summary stats
        summary_stats = {
            'n_observations': len(self.df),
            'n_variables': self.n_vars,
            'variables': ', '.join(self.variable_names),
            'coint_rank': self.coint_rank,
            'k_ar_diff': self._k_ar_diff,
            'log_likelihood': round(self.results.llf, 4),
            'aic': round(self.results.aic, 4) if hasattr(self.results, 'aic') else 'N/A',
        }
        pd.Series(summary_stats).to_csv(self.save_dir / 'vecm_summary.csv',
                                         header=['Value'])
        print(f"  ✓ vecm_summary.csv")

        # 4. Error correction term series
        for j in range(self.coint_rank):
            beta_j = self.results.beta[:, j]
            ect = pd.Series(
                self.df.values @ beta_j,
                index=self.df.index,
                name=f'ECT_{j+1}'
            )
            ect.to_csv(self.save_dir / f'ect_CE{j+1}.csv', header=True)
            print(f"  ✓ ect_CE{j+1}.csv")

        print(f"\n  All results saved to {self.save_dir}/")
```

---

### Step 5.2: Add `run_full_analysis()` master method

```python
    def run_full_analysis(self, k_ar_diff: int = 1,
                          det_order: int = -1) -> None:
        """
        Complete VECM pipeline:
        1. Select cointegration rank (Johansen trace)
        2. Estimate VECM
        3. Print summary (cointegrating vectors + adjustment speeds)
        4. Plot cointegrating relations
        5. Save all results

        Parameters
        ----------
        k_ar_diff : int
            Lagged differences — set to (optimal VAR lags - 1)
        det_order : int
            Deterministic terms: -1, 0, or 1
        """
        print("\n" + "=" * 70)
        print("VECM FULL ANALYSIS: Nigerian Monetary Policy")
        print("=" * 70)
        print(f"Variables: {self.variable_names}")
        print(f"Note: VECM supplements VAR — captures long-run equilibrium")

        # Step 1: Rank selection
        print("\n[Step 1/5] Cointegration rank selection...")
        rank_df = self.select_rank(k_ar_diff=k_ar_diff, det_order=det_order)
        rank_df.to_csv(self.save_dir / 'rank_selection.csv', index=False)

        if self.coint_rank == 0:
            print("\n⚠ VECM not applicable (no cointegration).")
            print("  → Your primary model (VAR, Day 6) is appropriate.")
            return

        # Step 2: Estimate
        print("\n[Step 2/5] Estimating VECM...")
        self.estimate(k_ar_diff=k_ar_diff, det_order=det_order)

        # Step 3: Print summary
        print("\n[Step 3/5] Results summary...")
        self.print_summary()

        # Step 4: Plot
        print("\n[Step 4/5] Plotting cointegrating relations...")
        for j in range(self.coint_rank):
            self.plot_cointegrating_relation(ce_index=j, save=True)

        # Step 5: Compare with VAR and save
        print("\n[Step 5/5] Saving results...")
        self.compare_with_var()
        self.save_results()

        print("\n" + "=" * 70)
        print("✓ VECM ANALYSIS COMPLETE")
        print("=" * 70)
        print("\nThesis interpretation guide:")
        print("  1. Cointegrating vector → 'Long-run equilibrium equation'")
        print("  2. Adjustment speed (α) → 'Speed of mean reversion'")
        print("  3. Half-life → 'Months to correct 50% of a shock'")
        print("  4. Compare α signs: who adjusts? (CBN policy or market?)")
        print(f"\n  Results saved to: {self.save_dir}/")
```

---

### Step 5.3: Update the `main()` function

**Replace the entire test block at the bottom with:**

```python
def main():
    """
    Run VECM analysis on Nigerian monetary policy data.
    Uses I(1) variables only: MPR and ExchangeRate.
    """
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    # Load data
    loader = NigerianMacroDataLoader(data_dir="data")
    df = loader.load_and_prepare()

    # VECM on I(1) variables only
    # Note: M2 and Inflation are I(0) so excluded from pure VECM
    i1_variables = ['MPR', 'ExchangeRate']

    print(f"\nRunning VECM on I(1) variables: {i1_variables}")
    print("(M2 and Inflation excluded — they are I(0))")
    print("Use ARDL (Day 5) for the full 4-variable system")

    analyzer = VECMAnalyzer(
        df=df,
        variable_names=i1_variables,
        save_dir="results/vecm"
    )

    analyzer.run_full_analysis(k_ar_diff=1)

    return analyzer


if __name__ == "__main__":
    analyzer = main()
```

---

### Step 5.4: Final test

```bash
python src/econometrics/vecm.py
```

**Expected output structure:**
```
======================================================================
VECM FULL ANALYSIS: Nigerian Monetary Policy
======================================================================
Variables: ['MPR', 'ExchangeRate']
Note: VECM supplements VAR — captures long-run equilibrium

[Step 1/5] Cointegration rank selection...
  ✓ 1 cointegrating relationship(s) found.
  → coint_rank set to 1

[Step 2/5] Estimating VECM...
  ✓ VECM estimated successfully.

[Step 3/5] Results summary...
[Cointegrating Vectors (β)]
              CE1
MPR       1.0000
Exchange  -X.XXXX

[Adjustment Speeds (α)]
                 CE1
MPR          -X.XXXX
ExchangeRate  X.XXXX

[Step 4/5] Plotting...
  ✓ Saved: results/vecm/cointegrating_relation_CE1.png

[Step 5/5] Saving results...
  ✓ cointegrating_vectors.csv
  ✓ adjustment_speeds.csv
  ✓ vecm_summary.csv
  ✓ ect_CE1.csv

======================================================================
✓ VECM ANALYSIS COMPLETE
======================================================================

Thesis interpretation guide:
  1. Cointegrating vector → 'Long-run equilibrium equation'
  2. Adjustment speed (α) → 'Speed of mean reversion'
  3. Half-life → 'Months to correct 50% of a shock'
  4. Compare α signs: who adjusts? (CBN policy or market?)
```

---

## Final Code Summary

Here's the complete `src/econometrics/vecm.py` file (~280 lines):

**File**: `src/econometrics/vecm.py`

**Structure**:
```python
class VECMAnalyzer:
    def __init__(df, variable_names, coint_rank=None, save_dir="results/vecm")
    def select_rank(k_ar_diff=1, det_order=-1)
    def estimate(k_ar_diff=1, det_order=-1)
    def get_cointegrating_vectors()
    def get_adjustment_speeds()
    def print_summary()
    def plot_cointegrating_relation(ce_index=0, save=True)
    def compare_with_var(var_results=None)
    def save_results()
    def run_full_analysis(k_ar_diff=1, det_order=-1)

def main()
```

**Key capabilities**:
- Johansen trace test for rank selection
- VECM estimation via statsmodels
- Cointegrating vectors (long-run equilibrium)
- Adjustment speeds with half-life calculation
- ECT plot (visual cointegration confirmation)
- Comparison with VAR short-run dynamics
- Full results saved to `results/vecm/`

**Verify your file is complete:**
```bash
python -c "
from src.econometrics.vecm import VECMAnalyzer
import inspect
methods = [m for m in dir(VECMAnalyzer) if not m.startswith('_')]
print('Methods:', methods)
print('Expected: compare_with_var, estimate, get_adjustment_speeds,')
print('          get_cointegrating_vectors, plot_cointegrating_relation,')
print('          print_summary, run_full_analysis, save_results, select_rank')
"
```

---

## Where VECM Fits in Your Thesis

```
Chapter 4: Methodology
├── 4.1 Unit Root Tests (Day 3 → stationarity.py)
├── 4.2 Cointegration Tests (Day 4 → cointegration.py)
├── 4.3 ARDL Bounds Test (Day 5 → ardl.py)      ← primary long-run model
├── 4.4 VAR Model (Day 6 → var_model.py)        ← primary short-run model
└── 4.5 VECM (Day 6B → vecm.py)                ← robustness/supplement

Chapter 5: Results
├── 5.3 Long-run relationships: ARDL and VECM consistent
└── 5.5 Speed of adjustment: α shows CBN MPR adjusts faster than ExchangeRate
```

**Key thesis sentences you can now write:**
> "As a robustness check, we estimate a bivariate VECM for the I(1) subset
> {MPR, ExchangeRate}. The adjustment coefficient α = -0.XX indicates that
> XX% of any deviation from long-run equilibrium is corrected within one month,
> consistent with active CBN monetary policy management."

---

## Git Commit

```bash
git add src/econometrics/vecm.py
git status
```

```bash
git commit -m "Day 6B: Add VECM module as robustness supplement to VAR

- Implements VECMAnalyzer class (~280 lines)
- Johansen rank selection (trace test)
- VECM estimation via statsmodels
- Cointegrating vectors (long-run equilibrium)
- Adjustment speeds (α) with half-life calculation
- ECT plots (visual cointegration check)
- Compare short-run dynamics with VAR
- Results saved to results/vecm/

Note: Applied to I(1) subset (MPR, ExchangeRate) only.
Full 4-variable system uses ARDL (Day 5) for mixed I(0)/I(1).

Thesis use: Section 4.5 robustness check + Section 5.5 interpretation

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

---

## End of Day 6B Checklist ✅

- [ ] `vecm.py` runs without errors
- [ ] Rank selection prints a table with trace statistics
- [ ] VECM estimates successfully (rank ≥ 1 required)
- [ ] Cointegrating vector printed with interpretation
- [ ] Adjustment speeds printed with half-life
- [ ] ECT plot saved to `results/vecm/`
- [ ] 4 CSV files in `results/vecm/`
- [ ] You understand the difference: ARDL = primary, VECM = supplement
- [ ] Git committed

**If all boxes checked → DAY 6B COMPLETE! 🎉**

---

## Tomorrow: Day 7 — Impulse Response Functions (IRF)

Now that both VAR (Day 6) and VECM (Day 6B) are done, proceed to IRF analysis
using the VAR results (the primary dynamic model).
