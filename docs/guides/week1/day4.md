# Week 1 - Day 4: Cointegration Testing

**Time Estimate:** 8 hours (full day)
**What You'll Build:** Cointegration testing module (Engle-Granger + Johansen)
**End Goal:** Determine if I(1) variables have long-run equilibrium relationships

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ Cointegration testing module (`models/cointegration.py` - 280 lines)
- ✅ Engle-Granger pairwise tests (all combinations)
- ✅ Johansen multivariate test
- ✅ Cointegration rank determination
- ✅ Results saved to `results/cointegration/`

**Pre-requisite:** Day 3 completed (know which variables are I(1))

---

## Hour 1 (9 AM - 10 AM): Understand Cointegration

### Step 1.1: What is cointegration?

**Cointegration:** When two or more I(1) series have a long-run equilibrium relationship.

**Example:**
- MPR and ExchangeRate are both I(1) (random walk, trending)
- But their **difference** (or linear combination) might be I(0) (stationary)
- If so, they "move together" in the long run despite short-run deviations

**Why does it matter?**
- I(1) variables in VAR → spurious regression (meaningless results)
- **BUT** if they're cointegrated → VAR in levels is valid!
- Or use VECM (Vector Error Correction Model) to model short + long run

**Two tests today:**
1. **Engle-Granger (1987):** Pairwise test for 2 variables
2. **Johansen (1988):** Multivariate test for all variables together

---

### Step 1.2: Create cointegration module - Basic structure

Create `models/cointegration.py` and **write this first chunk:**

```python
"""
Cointegration Tests for Nigerian Monetary Policy Analysis

This module implements Engle-Granger and Johansen tests to detect long-run
equilibrium relationships among I(1) variables.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations


class CointegrationTester:
    """
    Test for cointegration among I(1) time series.
    """

    def __init__(self, save_dir: str = "results/cointegration"):
        """
        Initialize the tester.

        Args:
            save_dir: Directory to save results
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        print(f"✓ Results will be saved to: {self.save_dir}")


# Test code
if __name__ == "__main__":
    print("Testing cointegration module initialization...")
    tester = CointegrationTester()
    print("✓ Module initialized successfully!")
```

**Save and test:**

```bash
python models/cointegration.py
```

**Expected output:**
```
Testing cointegration module initialization...
✓ Results will be saved to: results/cointegration
✓ Module initialized successfully!
```

---

## Hour 2 (10 AM - 11 AM): Implement Engle-Granger Test

### Step 2.1: Add Engle-Granger pairwise test

**Add this method to the class** (after `__init__`):

```python
    def engle_granger_test(self, y: pd.Series, x: pd.Series,
                          var1_name: str, var2_name: str) -> Dict:
        """
        Engle-Granger two-step cointegration test.

        Step 1: Regress y on x (both I(1))
        Step 2: Test if residuals are I(0) using ADF

        Args:
            y: Dependent variable (I(1))
            x: Independent variable (I(1))
            var1_name: Name of y
            var2_name: Name of x

        Returns:
            Dictionary with test results
        """
        # Step 1: OLS regression y = α + β*x + residuals
        # Using numpy for simple OLS
        X = np.column_stack([np.ones(len(x)), x.values])
        Y = y.values

        # OLS: β = (X'X)^(-1) X'Y
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        residuals = Y - X @ beta

        # Step 2: ADF test on residuals
        adf_result = adfuller(residuals, maxlag=12, regression='nc')
        # 'nc' = no constant (residuals already demeaned)

        adf_stat = adf_result[0]
        p_value = adf_result[1]

        # Engle-Granger critical values (MacKinnon 1991)
        # For 2 variables, 5% critical value ≈ -3.37
        critical_5pct = -3.37

        is_cointegrated = adf_stat < critical_5pct

        return {
            'pair': f'{var1_name} ~ {var2_name}',
            'adf_statistic': round(adf_stat, 4),
            'p_value': round(p_value, 4) if p_value < 0.10 else '>0.10',
            'critical_5%': critical_5pct,
            'cointegrated': is_cointegrated,
            'conclusion': 'Cointegrated' if is_cointegrated else 'Not cointegrated'
        }
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing cointegration module...")

    # Load data
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    tester = CointegrationTester()

    # Test one pair (MPR and ExchangeRate - both are I(1))
    print("\n[1/1] Testing Engle-Granger: MPR ~ ExchangeRate...")
    result = tester.engle_granger_test(df['MPR'], df['ExchangeRate'],
                                       'MPR', 'ExchangeRate')

    print("\nEngle-Granger Results:")
    print(f"  Pair: {result['pair']}")
    print(f"  ADF Statistic: {result['adf_statistic']}")
    print(f"  Critical Value (5%): {result['critical_5%']}")
    print(f"  Conclusion: {result['conclusion']}")

    print("\n✓ Test complete!")
```

**Save and test:**

```bash
python models/cointegration.py
```

---

## Hour 3 (11 AM - 12 PM): Test All Pairs

### Step 3.1: Add method to test all pairwise combinations

**Add this method after `engle_granger_test`:**

```python
    def test_all_pairs(self, df: pd.DataFrame,
                      i1_variables: List[str]) -> pd.DataFrame:
        """
        Test all pairwise combinations of I(1) variables.

        Args:
            df: DataFrame with variables
            i1_variables: List of I(1) variable names

        Returns:
            DataFrame with all pairwise test results
        """
        print("\n" + "=" * 80)
        print("ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS")
        print("=" * 80)

        results = []

        # Get all unique pairs
        pairs = list(combinations(i1_variables, 2))

        for var1, var2 in pairs:
            print(f"\nTesting: {var1} ~ {var2}...")

            # Test both directions (y~x and x~y)
            result1 = self.engle_granger_test(df[var1], df[var2], var1, var2)
            result2 = self.engle_granger_test(df[var2], df[var1], var2, var1)

            results.append(result1)
            results.append(result2)

        results_df = pd.DataFrame(results)

        print("\n" + "=" * 80)
        print(f"Tested {len(pairs)} pairs (both directions = {len(results)} tests)")
        print("=" * 80)

        return results_df
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing cointegration module...")

    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    tester = CointegrationTester()

    # I(1) variables from Day 3
    i1_vars = ['MPR', 'ExchangeRate']  # Based on Day 3 results

    print("\n[1/1] Testing all pairs of I(1) variables...")
    results_df = tester.test_all_pairs(df, i1_vars)

    print("\n" + "=" * 80)
    print("PAIRWISE RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python models/cointegration.py
```

**Expected output:**
```
[Loading messages...]
ENGLE-GRANGER PAIRWISE COINTEGRATION TESTS

Testing: MPR ~ ExchangeRate...
Testing: ExchangeRate ~ MPR...

Tested 1 pairs (both directions = 2 tests)

PAIRWISE RESULTS
pair                         adf_statistic  p_value  critical_5%  cointegrated  conclusion
MPR ~ ExchangeRate           -2.5432        >0.10    -3.37        False         Not cointegrated
ExchangeRate ~ MPR           -2.6543        >0.10    -3.37        False         Not cointegrated

✓ All tests complete!
```

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK 🍽️

Take a break! You've implemented Engle-Granger tests.

---

## Hour 5 (1 PM - 2 PM): Implement Johansen Test

### Step 5.1: Add Johansen multivariate test

**Add this method after `test_all_pairs`:**

```python
    def johansen_test(self, df: pd.DataFrame, variables: List[str]) -> Dict:
        """
        Johansen multivariate cointegration test.

        Tests for multiple cointegrating relationships among all variables.

        Args:
            df: DataFrame with variables
            variables: List of variable names to test

        Returns:
            Dictionary with test results
        """
        print("\n" + "=" * 80)
        print("JOHANSEN MULTIVARIATE COINTEGRATION TEST")
        print("=" * 80)

        # Prepare data
        data = df[variables].dropna()

        # Run Johansen test
        # det_order=0: no deterministic trend
        # k_ar_diff=11: lag order (from Day 3, or use AIC selection)
        result = coint_johansen(data, det_order=0, k_ar_diff=11)

        # Extract results
        trace_stats = result.lr1  # Trace statistics
        trace_crit = result.cvt   # Critical values (90%, 95%, 99%)

        max_eig_stats = result.lr2  # Max eigenvalue statistics
        max_eig_crit = result.cvm   # Critical values

        # Determine cointegration rank
        # Check trace statistic at 5% level (column 1)
        rank = 0
        for i in range(len(trace_stats)):
            if trace_stats[i] > trace_crit[i, 1]:  # Compare with 95% critical value
                rank = i + 1

        results = {
            'variables': variables,
            'n_variables': len(variables),
            'cointegration_rank': rank,
            'trace_statistics': trace_stats.tolist(),
            'trace_critical_5%': trace_crit[:, 1].tolist(),
            'interpretation': self._interpret_johansen_rank(rank, len(variables))
        }

        return results

    def _interpret_johansen_rank(self, rank: int, n_vars: int) -> str:
        """Interpret Johansen cointegration rank."""
        if rank == 0:
            return "No cointegration detected"
        elif rank == n_vars:
            return f"All {n_vars} variables are cointegrated (unlikely, check specification)"
        else:
            return f"{rank} cointegrating relationship(s) detected"
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing cointegration module...")

    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    tester = CointegrationTester()

    # Test all I(1) variables
    i1_vars = ['MPR', 'ExchangeRate']

    print("\n[1/2] Engle-Granger pairwise tests...")
    eg_results = tester.test_all_pairs(df, i1_vars)

    print("\n[2/2] Johansen multivariate test...")
    # Test with all 4 variables (standard approach)
    all_vars = ['MPR', 'ExchangeRate', 'M2', 'Inflation']
    johansen_results = tester.johansen_test(df, all_vars)

    print(f"\n  Variables tested: {', '.join(johansen_results['variables'])}")
    print(f"  Cointegration rank: {johansen_results['cointegration_rank']}")
    print(f"  Interpretation: {johansen_results['interpretation']}")

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python models/cointegration.py
```

---

## Hour 6 (2 PM - 3 PM): Format Results & Save

### Step 6.1: Add method to format Johansen output

**Add this method after `johansen_test`:**

```python
    def print_johansen_results(self, results: Dict):
        """
        Print formatted Johansen test results.

        Args:
            results: Dictionary from johansen_test()
        """
        print("\n" + "=" * 80)
        print("JOHANSEN TEST RESULTS")
        print("=" * 80)

        print(f"\nVariables: {', '.join(results['variables'])}")
        print(f"Number of variables: {results['n_variables']}")

        print("\nTrace Statistics:")
        print(f"{'Rank':>6} {'Trace Stat':>12} {'Critical 5%':>14} {'Reject H0?':>12}")
        print("-" * 50)

        for i, (stat, crit) in enumerate(zip(results['trace_statistics'],
                                             results['trace_critical_5%'])):
            reject = "Yes" if stat > crit else "No"
            print(f"{i:>6} {stat:>12.4f} {crit:>14.4f} {reject:>12}")

        print(f"\n✓ Cointegration Rank: {results['cointegration_rank']}")
        print(f"✓ Interpretation: {results['interpretation']}")
        print("=" * 80)
```

---

### Step 6.2: Add save method

**Add this method after `print_johansen_results`:**

```python
    def save_results(self, eg_results: pd.DataFrame, johansen_results: Dict):
        """
        Save all results to CSV files.

        Args:
            eg_results: Engle-Granger DataFrame
            johansen_results: Johansen results dictionary
        """
        # Save Engle-Granger results
        eg_path = self.save_dir / 'engle_granger_pairwise.csv'
        eg_results.to_csv(eg_path, index=False)
        print(f"  ✓ Saved Engle-Granger results: {eg_path}")

        # Save Johansen summary
        johansen_summary = {
            'variables': ', '.join(johansen_results['variables']),
            'cointegration_rank': johansen_results['cointegration_rank'],
            'interpretation': johansen_results['interpretation']
        }

        johansen_df = pd.DataFrame([johansen_summary])
        johansen_path = self.save_dir / 'johansen_summary.csv'
        johansen_df.to_csv(johansen_path, index=False)
        print(f"  ✓ Saved Johansen summary: {johansen_path}")
```

---

## Hour 7 (3 PM - 4 PM): Create Master Analysis Function

### Step 7.1: Add comprehensive analysis method

**Add this method after `save_results`:**

```python
    def run_full_analysis(self, df: pd.DataFrame, i1_variables: List[str]):
        """
        Run complete cointegration analysis.

        Args:
            df: DataFrame with macro variables
            i1_variables: List of I(1) variable names from Day 3
        """
        print("\n" + "=" * 80)
        print("FULL COINTEGRATION ANALYSIS - Nigerian Macro Variables")
        print("=" * 80)

        # Engle-Granger pairwise tests
        print("\n[1/3] Running Engle-Granger pairwise tests...")
        eg_results = self.test_all_pairs(df, i1_variables)

        # Johansen multivariate test (use all 4 variables)
        print("\n[2/3] Running Johansen multivariate test...")
        all_vars = df.columns.tolist()
        johansen_results = self.johansen_test(df, all_vars)
        self.print_johansen_results(johansen_results)

        # Save results
        print("\n[3/3] Saving results...")
        self.save_results(eg_results, johansen_results)

        # Summary
        self._print_summary(eg_results, johansen_results)

        # Store results
        self.results = {
            'engle_granger': eg_results,
            'johansen': johansen_results
        }

        print("\n✓ ANALYSIS COMPLETE!")
        return self.results

    def _print_summary(self, eg_results: pd.DataFrame, johansen_results: Dict):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("COINTEGRATION TEST SUMMARY")
        print("=" * 80)

        # Engle-Granger summary
        n_cointegrated = eg_results['cointegrated'].sum()
        print(f"\n📊 Engle-Granger (Pairwise):")
        print(f"  → {n_cointegrated} out of {len(eg_results)} pairs cointegrated")

        if n_cointegrated > 0:
            coint_pairs = eg_results[eg_results['cointegrated'] == True]
            print(f"  → Cointegrated pairs:")
            for _, row in coint_pairs.iterrows():
                print(f"      • {row['pair']}")

        # Johansen summary
        print(f"\n📊 Johansen (Multivariate):")
        print(f"  → Cointegration rank: {johansen_results['cointegration_rank']}")
        print(f"  → {johansen_results['interpretation']}")

        # Modeling implications
        print(f"\n💡 IMPLICATIONS FOR MODELING:")
        if johansen_results['cointegration_rank'] > 0:
            print(f"  → {johansen_results['cointegration_rank']} long-run relationship(s) detected")
            print("  → Can use VAR in levels (cointegration justifies this)")
            print("  → Or use VECM to model error correction explicitly")
        else:
            print("  → No cointegration detected")
            print("  → Use differenced VAR (all in first differences)")
            print("  → Or use ARDL bounds test (Day 5) for mixed orders")

        print("=" * 80)
```

**Update main function:**

```python
def main():
    """
    Main execution: Run cointegration tests.
    """
    print("=" * 80)
    print("NIGERIAN MONETARY POLICY - COINTEGRATION ANALYSIS")
    print("=" * 80)

    # Load data
    import sys
    sys.path.append('.')
    from models.data_loader import NigerianMacroDataLoader

    print("\nLoading data...")
    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    # I(1) variables from Day 3
    i1_variables = ['MPR', 'ExchangeRate']

    # Run cointegration tests
    tester = CointegrationTester(save_dir="results/cointegration")
    results = tester.run_full_analysis(df, i1_variables)

    return results


if __name__ == "__main__":
    results = main()
```

**Save and run final test:**

```bash
python models/cointegration.py
```

---

## Hour 8 (4 PM - 5 PM): Verify & Git Commit

### Step 8.1: Verify file completeness

```bash
wc -l models/cointegration.py
# Should show: ~280 lines
```

---

### Step 8.2: Review results

```bash
cat results/cointegration/johansen_summary.csv
```

**Key finding to remember:**
- Johansen rank tells you how many long-run relationships exist
- Rank 0: No cointegration
- Rank 1: One cointegrating relationship (common)
- Rank > 1: Multiple relationships (rare)

---

### Step 8.3: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Day 4: Cointegration testing (Engle-Granger + Johansen)

- Created cointegration testing module (280 lines, built in chunks)
- Implemented Engle-Granger pairwise tests
- Implemented Johansen multivariate test
- Cointegration rank determination

Results:
  • Engle-Granger: X out of Y pairs cointegrated
  • Johansen rank: Z (check your actual results)
  • Variables: MPR, ExchangeRate, M2, Inflation

Outputs: results/cointegration/ (2 CSV files)

Implication: [Fill based on your results]
  If rank > 0: Can use VAR in levels or VECM
  If rank = 0: Use differenced VAR or ARDL (Day 5)

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## Final Code Summary

Here's the complete `models/cointegration.py` file (~280 lines):

**File**: `models/cointegration.py`

**Structure**:
```python
class CointegrationTester:
    def __init__(save_dir="results/cointegration")
    def engle_granger_test(y, x, var_names)
    def test_all_pairs(df, i1_variables)
    def johansen_test(df, variables)
    def print_johansen_results(results)
    def save_results(eg_results, johansen_results)
    def run_full_analysis(df, i1_variables)

def main()
```

**Key capabilities**:
- Engle-Granger pairwise cointegration test
- Test all I(1) variable pairs automatically
- Johansen multivariate cointegration test
- Cointegration rank determination
- Save results to CSV for thesis tables

**Verify your file is complete:**
```bash
python -c "
from models.cointegration import CointegrationTester
import inspect
methods = [m for m in dir(CointegrationTester) if not m.startswith('_')]
print('Methods:', methods)
"
```

---

## End of Day 4 Checklist ✅

Before you finish, verify:

- [ ] `cointegration.py` runs without errors
- [ ] You see 2 CSV files in `results/cointegration/`
- [ ] Johansen test shows cointegration rank
- [ ] You understand what cointegration rank means
- [ ] You know the modeling implications (VAR vs VECM vs differencing)
- [ ] Git commit successful

**If all boxes checked → DAY 4 COMPLETE! 🎉**

---

## What You Built Today

**Files created:** 1 (`cointegration.py`)
**Lines of code:** ~280 (built in 6 chunks over 6 hours)
**Tests implemented:** 2 (Engle-Granger, Johansen)
**Skills learned:** Cointegration testing, long-run relationships, VECM concepts

**Key concept:** Cointegration allows you to use I(1) variables in levels (without differencing) if they share a long-run equilibrium.

---

## Tomorrow (Day 5): ARDL Bounds Testing

**What you'll build:**
- ARDL bounds testing module (`models/ardl.py`)
- Pesaran et al. (2001) approach for mixed I(0)/I(1)
- Long-run equilibrium estimation
- Alternative to Johansen for mixed orders

**Question to answer:** Is there a long-run relationship even with mixed integration orders?

**Time:** 8 hours
**Difficulty:** Same as Day 4

---

## Troubleshooting

**"ImportError: cannot import name 'coint_johansen'"**
```bash
# Check statsmodels version
python -c "import statsmodels; print(statsmodels.__version__)"
# Should be 0.14.1

# If wrong version:
pip install --upgrade statsmodels==0.14.1
```

**"Johansen returns rank = 0 for all"**
- This is possible if no cointegration exists
- Check: Are you using levels? Not differences?
- Try: Different lag orders (k_ar_diff parameter)

**"Engle-Granger ADF on residuals gives error"**
- Check: Both variables have same length?
- Check: No missing values? `df[col].isnull().sum()`
- Try: Use `dropna()` before passing to function

**"All Engle-Granger tests show 'Not cointegrated'"**
- This is valid! Not all I(1) pairs are cointegrated
- Check Johansen for multivariate relationships
- If neither works, use ARDL (Day 5) or differenced VAR

---

**Excellent work! See you tomorrow for the final day of Week 1! 💪**
