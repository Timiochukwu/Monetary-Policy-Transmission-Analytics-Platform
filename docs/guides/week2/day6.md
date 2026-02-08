# Week 2 - Day 6: VAR Model Estimation

**Time Estimate:** 8 hours (full day)
**What You'll Build:** Vector Autoregression (VAR) model estimation module
**End Goal:** Estimate VAR model and perform Granger causality tests

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ VAR model estimation module (`src/econometrics/var_model.py` - 290 lines)
- ✅ Automatic lag selection (AIC, BIC, HQ criteria)
- ✅ VAR model estimation with proper ordering
- ✅ Granger causality tests (all variable pairs)
- ✅ Results saved to `results/var/`
- ✅ **Week 2 Started!**

**Pre-requisite:** Week 1 completed (know your data's properties)

---

## Hour 1 (9 AM - 10 AM): Understand VAR Models

### Step 1.1: What is a VAR model?

**VAR = Vector Autoregression**

**The idea:**
- Instead of modeling one variable (like ARDL), model ALL variables together
- Each variable is regressed on its own past values AND past values of ALL other variables
- Captures dynamic relationships among variables

**VAR(p) model:**
```
y_t = c + A₁ y_{t-1} + A₂ y_{t-2} + ... + Aₚ y_{t-p} + ε_t
```

Where:
- `y_t` = [MPR_t, ExchangeRate_t, M2_t, Inflation_t]' (4×1 vector)
- `c` = constant vector (4×1)
- `Aᵢ` = coefficient matrices (4×4)
- `p` = lag order
- `ε_t` = error vector (4×1)

**Example with p=2, K=4 variables:**
```
MPR_t = c₁ + a₁₁MPR_{t-1} + a₁₂ER_{t-1} + a₁₃M2_{t-1} + a₁₄Inf_{t-1}
          + b₁₁MPR_{t-2} + b₁₂ER_{t-2} + b₁₃M2_{t-2} + b₁₄Inf_{t-2} + ε₁t

ExchangeRate_t = c₂ + a₂₁MPR_{t-1} + ... (similar)
M2_t = c₃ + ... (similar)
Inflation_t = c₄ + ... (similar)
```

**Total parameters:** K × (K×p + 1) = 4 × (4×2 + 1) = 36

**Why use VAR?**
- Captures feedback effects (MPR affects Inflation, but Inflation also affects MPR)
- No need to specify which variables are endogenous/exogenous
- Can analyze shocks (IRF) and variance decomposition (FEVD)

---

### Step 1.2: Create VAR module - Basic structure

Create `src/econometrics/var_model.py` and **write this first chunk:**

```python
"""
Vector Autoregression (VAR) Model for Nigerian Monetary Policy Analysis

Estimates VAR model, performs lag selection, and Granger causality tests.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from statsmodels.tsa.api import VAR


class VARModel:
    """
    VAR model estimation and analysis.
    """

    def __init__(self, df: pd.DataFrame,
                 ordering: List[str] = None,
                 save_dir: str = "results/var"):
        """
        Initialize VAR analyzer.

        Args:
            df: DataFrame with macro variables
            ordering: Variable ordering (important for Cholesky decomposition later)
            save_dir: Directory to save results
        """
        self.df = df
        # Default ordering: Policy → Transmission channels → Target
        self.ordering = ordering or ['MPR', 'ExchangeRate', 'M2', 'Inflation']
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Reorder DataFrame
        self.data = self.df[self.ordering]

        print(f"✓ VAR Analyzer initialized")
        print(f"  Variables: {', '.join(self.ordering)}")
        print(f"  Observations: {len(self.data)}")


# Test code
if __name__ == "__main__":
    print("Testing VAR module initialization...")

    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    var = VARModel(df)
    print("✓ Module initialized successfully!")
```

**Save and test:**

```bash
python src/econometrics/var_model.py
```

**Expected output:**
```
[Data loading messages...]
✓ VAR Analyzer initialized
  Variables: MPR, ExchangeRate, M2, Inflation
  Observations: 181
✓ Module initialized successfully!
```

---

## Hour 2 (10 AM - 11 AM): Implement Lag Selection

### Step 2.1: Add lag selection method

**Add this method to the class** (after `__init__`):

```python
    def select_lags(self, max_lags: int = 12, ic: str = 'aic') -> int:
        """
        Select optimal lag order using information criteria.

        Args:
            max_lags: Maximum lag order to test
            ic: Information criterion ('aic', 'bic', 'hqic')

        Returns:
            Optimal lag order
        """
        print("\n" + "=" * 80)
        print(f"VAR LAG SELECTION (Maximum lags: {max_lags})")
        print("=" * 80)

        # Create VAR model instance
        model = VAR(self.data)

        # Select lag order
        lag_order_results = model.select_order(maxlags=max_lags)

        print("\nInformation Criteria:")
        print(f"  AIC selects: {lag_order_results.aic} lags")
        print(f"  BIC selects: {lag_order_results.bic} lags")
        print(f"  HQIC selects: {lag_order_results.hqic} lags")

        # Get selected lag
        if ic == 'aic':
            opt_lag = lag_order_results.aic
        elif ic == 'bic':
            opt_lag = lag_order_results.bic
        else:
            opt_lag = lag_order_results.hqic

        print(f"\n✓ Selected lag order ({ic.upper()}): {opt_lag}")

        self.opt_lag = opt_lag
        self.lag_order_results = lag_order_results

        return opt_lag
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing VAR module...")

    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    var = VARModel(df)

    print("\n[1/1] Testing lag selection...")
    opt_lag = var.select_lags(max_lags=12, ic='aic')
    print(f"\nOptimal lag: {opt_lag}")

    print("\n✓ Test complete!")
```

**Save and test:**

```bash
python src/econometrics/var_model.py
```

**Expected output:**
```
[Loading messages...]
VAR LAG SELECTION (Maximum lags: 12)

Information Criteria:
  AIC selects: 11 lags
  BIC selects: 3 lags
  HQIC selects: 6 lags

✓ Selected lag order (AIC): 11

Optimal lag: 11

✓ Test complete!
```

**Note:** AIC often selects higher lags than BIC. We'll use AIC for richer dynamics.

---

## Hour 3 (11 AM - 12 PM): Estimate VAR Model

### Step 3.1: Add VAR estimation method

**Add this method after `select_lags`:**

```python
    def fit(self, lags: int = None) -> Dict:
        """
        Estimate VAR model.

        Args:
            lags: Lag order (if None, uses selected optimal lag)

        Returns:
            Dictionary with estimation results
        """
        if lags is None:
            if not hasattr(self, 'opt_lag'):
                print("  → Running lag selection first...")
                self.select_lags()
            lags = self.opt_lag

        print("\n" + "=" * 80)
        print(f"ESTIMATING VAR({lags}) MODEL")
        print("=" * 80)

        # Create and fit VAR model
        model = VAR(self.data)
        self.res = model.fit(lags)

        print(f"\n✓ VAR({lags}) model estimated successfully")
        print(f"  Total observations: {self.res.nobs}")
        print(f"  Parameters estimated: {self.res.params.size}")
        print(f"  AIC: {self.res.aic:.4f}")
        print(f"  BIC: {self.res.bic:.4f}")

        # Store results
        self.results = {
            'lag_order': lags,
            'nobs': self.res.nobs,
            'n_params': self.res.params.size,
            'aic': self.res.aic,
            'bic': self.res.bic,
            'model': self.res
        }

        return self.results

    def print_summary(self):
        """Print VAR model summary."""
        if not hasattr(self, 'res'):
            print("  ⚠ Model not fitted yet. Call fit() first.")
            return

        print("\n" + "=" * 80)
        print("VAR MODEL SUMMARY")
        print("=" * 80)
        print(self.res.summary())
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing VAR module...")

    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    var = VARModel(df)

    print("\n[1/2] Selecting lags...")
    var.select_lags(max_lags=12, ic='aic')

    print("\n[2/2] Estimating VAR model...")
    var.fit()

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python src/econometrics/var_model.py
```

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK 🍽️

Take a break! Your VAR model is now estimated.

---

## Hour 5 (1 PM - 2 PM): Implement Granger Causality Tests

### Step 5.1: Add Granger causality test method

**Add this method after `print_summary`:**

```python
    def granger_causality(self, caused: str, causing: List[str],
                         alpha: float = 0.05) -> Dict:
        """
        Test Granger causality: Does 'causing' Granger-cause 'caused'?

        H0: 'causing' does NOT Granger-cause 'caused'

        Args:
            caused: Dependent variable name
            causing: List of causing variable names
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        if not hasattr(self, 'res'):
            print("  ⚠ Model not fitted yet. Call fit() first.")
            return {}

        # Run Granger causality test
        # Note: Using 'signif' parameter (not 'alpha')
        test_result = self.res.test_causality(caused, causing=causing,
                                             kind='f', signif=alpha)

        # Extract results
        # Note: Attributes are 'test_statistic' and 'pvalue'
        f_stat = test_result.test_statistic
        p_value = test_result.pvalue

        result = {
            'caused': caused,
            'causing': ' + '.join(causing),
            'f_statistic': round(float(f_stat), 4),
            'p_value': round(float(p_value), 4),
            'significant': p_value < alpha,
            'conclusion': f"{'Reject H0' if p_value < alpha else 'Fail to reject H0'}: " +
                         f"{'Granger-causes' if p_value < alpha else 'Does not Granger-cause'}"
        }

        return result
```

---

### Step 5.2: Add method to test all variable pairs

**Add this method after `granger_causality`:**

```python
    def test_all_granger_causality(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Test Granger causality for all variable pairs.

        Tests each variable as caused by all others.

        Args:
            alpha: Significance level

        Returns:
            DataFrame with all test results
        """
        print("\n" + "=" * 80)
        print("GRANGER CAUSALITY TESTS")
        print("=" * 80)

        results = []

        for caused_var in self.ordering:
            # Test if all other variables Granger-cause this variable
            causing_vars = [v for v in self.ordering if v != caused_var]

            print(f"\nTesting: {' + '.join(causing_vars)} → {caused_var}")

            result = self.granger_causality(caused_var, causing_vars, alpha)
            results.append(result)

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print("GRANGER CAUSALITY RESULTS")
        print("=" * 80)
        print(results_df[['caused', 'causing', 'f_statistic', 'p_value', 'significant']].to_string(index=False))

        return results_df
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing VAR module...")

    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    var = VARModel(df)

    print("\n[1/3] Selecting lags...")
    var.select_lags(max_lags=12, ic='aic')

    print("\n[2/3] Estimating VAR model...")
    var.fit()

    print("\n[3/3] Testing Granger causality...")
    granger_df = var.test_all_granger_causality(alpha=0.05)

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python src/econometrics/var_model.py
```

---

## Hour 6 (2 PM - 3 PM): Add Diagnostic Tests

### Step 6.1: Add stability check method

**Add this method after `test_all_granger_causality`:**

```python
    def check_stability(self) -> Dict:
        """
        Check VAR stability (all eigenvalues inside unit circle).

        Returns:
            Dictionary with stability results
        """
        if not hasattr(self, 'res'):
            print("  ⚠ Model not fitted yet.")
            return {}

        # Get eigenvalues of companion matrix
        eigenvalues = self.res.roots

        # Check if all |eigenvalues| < 1
        max_eigenvalue = np.max(np.abs(eigenvalues))
        is_stable = max_eigenvalue < 1.0

        result = {
            'max_eigenvalue': round(float(max_eigenvalue), 4),
            'stable': is_stable,
            'conclusion': 'VAR is stable' if is_stable else 'VAR is NOT stable (unstable)'
        }

        print("\n" + "=" * 80)
        print("VAR STABILITY CHECK")
        print("=" * 80)
        print(f"  Max eigenvalue modulus: {result['max_eigenvalue']:.4f}")
        print(f"  Status: {result['conclusion']}")
        print("=" * 80)

        return result
```

---

### Step 6.2: Add save results method

**Add this method after `check_stability`:**

```python
    def save_results(self, granger_df: pd.DataFrame, stability: Dict):
        """
        Save all results to CSV files.

        Args:
            granger_df: Granger causality results
            stability: Stability check results
        """
        # Save Granger causality results
        granger_path = self.save_dir / 'var_granger_causality.csv'
        granger_df.to_csv(granger_path, index=False)
        print(f"  ✓ Saved Granger causality: {granger_path}")

        # Save model info
        model_info = {
            'lag_order': self.results['lag_order'],
            'observations': self.results['nobs'],
            'parameters': self.results['n_params'],
            'aic': self.results['aic'],
            'bic': self.results['bic'],
            'max_eigenvalue': stability['max_eigenvalue'],
            'stable': stability['stable']
        }

        model_df = pd.DataFrame([model_info])
        model_path = self.save_dir / 'var_model_info.csv'
        model_df.to_csv(model_path, index=False)
        print(f"  ✓ Saved model info: {model_path}")

        # Save parameter estimates
        params_path = self.save_dir / 'var_parameters.csv'
        self.res.params.to_csv(params_path)
        print(f"  ✓ Saved parameters: {params_path}")
```

---

## Hour 7 (3 PM - 4 PM): Create Master Analysis Function

### Step 7.1: Add comprehensive analysis method

**Add this method after `save_results`:**

```python
    def run_full_analysis(self, max_lags: int = 12,
                         ic: str = 'aic', alpha: float = 0.05):
        """
        Run complete VAR analysis pipeline.

        Args:
            max_lags: Maximum lag order for selection
            ic: Information criterion
            alpha: Significance level for tests
        """
        print("\n" + "=" * 80)
        print("VAR MODEL - FULL ANALYSIS PIPELINE")
        print("=" * 80)

        # Step 1: Lag selection
        print("\n[1/5] Selecting optimal lag order...")
        self.select_lags(max_lags=max_lags, ic=ic)

        # Step 2: Estimate VAR
        print("\n[2/5] Estimating VAR model...")
        self.fit()

        # Step 3: Stability check
        print("\n[3/5] Checking stability...")
        stability = self.check_stability()

        # Step 4: Granger causality
        print("\n[4/5] Testing Granger causality...")
        granger_df = self.test_all_granger_causality(alpha=alpha)

        # Step 5: Save results
        print("\n[5/5] Saving results...")
        self.save_results(granger_df, stability)

        # Print summary
        self._print_final_summary(granger_df, stability)

        print("\n✓ FULL ANALYSIS COMPLETE!")

        return {
            'granger': granger_df,
            'stability': stability,
            'model': self.results
        }

    def _print_final_summary(self, granger_df: pd.DataFrame, stability: Dict):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("VAR ANALYSIS SUMMARY")
        print("=" * 80)

        print(f"\n📊 Model Specification:")
        print(f"  → VAR({self.results['lag_order']}) model")
        print(f"  → Variables: {', '.join(self.ordering)}")
        print(f"  → Observations: {self.results['nobs']}")
        print(f"  → AIC: {self.results['aic']:.4f}")

        print(f"\n📊 Stability:")
        print(f"  → {stability['conclusion']}")

        print(f"\n📊 Granger Causality (significant at 5%):")
        significant = granger_df[granger_df['significant'] == True]
        if len(significant) > 0:
            for _, row in significant.iterrows():
                print(f"  → {row['causing']} ⇒ {row['caused']} (p={row['p_value']:.4f})")
        else:
            print("  → No significant Granger causality detected")

        print("\n💡 Next Steps:")
        print("  → Day 7: Impulse Response Functions (IRF)")
        print("  → Day 8: Forecast Error Variance Decomposition (FEVD)")

        print("=" * 80)
```

**Update main function:**

```python
def main():
    """
    Main execution: Run VAR analysis.
    """
    print("=" * 80)
    print("NIGERIAN MONETARY POLICY - VAR MODEL ESTIMATION")
    print("=" * 80)

    # Load data
    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    print("\nLoading data...")
    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    # Run VAR analysis
    var = VARModel(df, save_dir="results/var")
    results = var.run_full_analysis(max_lags=12, ic='aic', alpha=0.05)

    return results


if __name__ == "__main__":
    results = main()
```

**Save and run final test:**

```bash
python src/econometrics/var_model.py
```

**Expected output:**
```
[Loading messages...]
VAR MODEL - FULL ANALYSIS PIPELINE

[1/5] Selecting optimal lag order...
AIC selects: 11 lags
BIC selects: 3 lags
✓ Selected lag order (AIC): 11

[2/5] Estimating VAR model...
✓ VAR(11) model estimated successfully

[3/5] Checking stability...
Max eigenvalue modulus: 0.9876
Status: VAR is stable

[4/5] Testing Granger causality...
[Granger test results...]

[5/5] Saving results...
  ✓ Saved Granger causality: results/var/var_granger_causality.csv
  ✓ Saved model info: results/var/var_model_info.csv
  ✓ Saved parameters: results/var/var_parameters.csv

VAR ANALYSIS SUMMARY
[Summary output...]

✓ FULL ANALYSIS COMPLETE!
```

---

## Hour 8 (4 PM - 5 PM): Verify & Git Commit

### Step 8.1: Verify file completeness

```bash
wc -l src/econometrics/var_model.py
# Should show: ~290 lines
```

---

### Step 8.2: Review results

```bash
ls -lh results/var/
# Should show: 3 CSV files

head results/var/var_granger_causality.csv
```

---

### Step 8.3: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Week 2 Day 6: VAR model estimation

- Created VAR model estimation module (290 lines, built in chunks)
- Automatic lag selection (AIC, BIC, HQIC)
- VAR(11) model estimated (AIC criterion)
- Granger causality tests for all variable pairs
- Stability check (eigenvalues)

Results:
  • VAR(11) selected by AIC
  • VAR(3) selected by BIC
  • Model is stable (max eigenvalue < 1)
  • Granger causality: [Check your results]

Outputs: results/var/ (3 CSV files)

Implication: VAR model ready for IRF analysis (Day 7)

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## Final Code Summary

Here's the complete `src/econometrics/var_model.py` file (~350 lines):

**File**: `src/econometrics/var_model.py`

**Structure**:
```python
class VARModel:
    def __init__(df, variable_names, save_dir="results/var")
    def select_lags(max_lags=12, ic="aic")
    def fit(lags=None)
    def print_summary()
    def granger_causality(caused, causing, alpha)
    def test_all_granger_causality(alpha)
    def check_stability()
    def save_results(granger_df, stability)
    def run_full_analysis(max_lags, alpha, save_results)

def main()
```

**Key capabilities**:
- Lag length selection (AIC, BIC, HQIC)
- VAR model estimation with statsmodels
- Granger causality testing
- Stability check (eigenvalue modulus)
- Model summary output
- Save results for thesis tables

**Verify your file is complete:**
```bash
python -c "
from src.econometrics.var_model import VARModel
import inspect
methods = [m for m in dir(VARModel) if not m.startswith('_')]
print('Methods:', methods)
"
```

---

## End of Day 6 Checklist ✅

Before you finish, verify:

- [ ] `var_model.py` runs without errors
- [ ] You see 3 CSV files in `results/var/`
- [ ] VAR model is stable (max eigenvalue < 1)
- [ ] Granger causality results saved
- [ ] You understand VAR(11) means 11 lags
- [ ] Git commit successful

**If all boxes checked → DAY 6 COMPLETE! 🎉**

---

## What You Built Today

**Files created:** 1 (`var_model.py`)
**Lines of code:** ~290 (built in 6 chunks over 6 hours)
**Key outputs:** VAR(11) model, Granger causality tests, stability check
**Skills learned:** VAR modeling, lag selection, Granger causality, stability analysis

**Key result:** You now have an estimated VAR model that captures the dynamic relationships among MPR, ExchangeRate, M2, and Inflation!

---

## Tomorrow (Day 7): Impulse Response Functions

**What you'll build:**
- IRF module (`src/econometrics/irf.py`)
- Orthogonalized impulse responses (Cholesky decomposition)
- IRF plots for all shock-response combinations
- Peak response analysis

**Question to answer:** How does a 1 standard deviation MPR shock affect Inflation over 24 months?

**Time:** 8 hours
**Difficulty:** Same as Day 6

---

## Troubleshooting

**"VAR estimation fails with LinAlgError"**
- Check: Any constant variables? `df.std()`
- Check: Missing values? `df.isnull().sum()`
- Try: Lower lag order

**"Max eigenvalue > 1"**
- VAR is unstable (explosive)
- Check: Using levels or differences?
- Try: Lower lag order or differencing I(1) variables

**"All Granger tests non-significant"**
- This is possible if variables are unrelated
- Check: Using correct lag order?
- Note: Non-causality doesn't mean no relationship (could be contemporaneous)

**"AttributeError: 'VARResults' object has no attribute 'test_statistic'"**
- Use `test_result.test_statistic` (not `f_statistic`)
- Use `test_result.pvalue` (not `p_value`)
- Check statsmodels version: `pip install --upgrade statsmodels`

---

**Excellent work! See you tomorrow for Day 7! 💪**
