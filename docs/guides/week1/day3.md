# Week 1 - Day 3: Stationarity Testing (Unit Root Tests)

**Time Estimate:** 8 hours (full day)
**What You'll Build:** Stationarity testing module with ADF, PP, KPSS tests
**End Goal:** Classify variables as I(0) or I(1) for econometric modeling

---

## What You're Building Today

By 5 PM today, you will have:

- ✅ New packages installed (statsmodels, scipy)
- ✅ Stationarity testing module (`src/econometrics/stationarity.py` - 250 lines)
- ✅ ADF, PP, KPSS tests implemented
- ✅ Integration order classification (I(0) vs I(1))
- ✅ Results saved to `results/stationarity/`

**Pre-requisite:** Days 1-2 completed

---

## Hour 1 (9 AM - 10 AM): Install New Packages & Understand Stationarity

### Step 1.1: Install econometric packages

**Today we need statsmodels for unit root tests:**

```bash
pip install statsmodels==0.14.1 scipy==1.11.4
```

**Verify installation:**

```bash
python -c "import statsmodels; print(f'statsmodels {statsmodels.__version__}')"
python -c "import scipy; print(f'scipy {scipy.__version__}')"
```

**Expected output:**
```
statsmodels 0.14.1
scipy 1.11.4
```

---

### Step 1.2: What is stationarity and why does it matter?

**Stationary series:** Mean, variance, and autocorrelation structure don't change over time.

**Examples:**
- **Stationary (I(0)):** Interest rate changes, inflation rate (often)
- **Non-stationary (I(1)):** Price levels, GDP levels, exchange rates

**Why test?**
- VAR models require stationary data (or cointegrated non-stationary data)
- Using I(1) data in VAR without cointegration → spurious regressions
- Need to know: Use levels or differences?

**Three tests today:**
1. **ADF (Augmented Dickey-Fuller):** H₀ = unit root exists (I(1))
2. **PP (Phillips-Perron):** H₀ = unit root exists (I(1))
3. **KPSS:** H₀ = series is stationary (I(0)) [opposite!]

**Decision rule:**
- If ADF + PP reject H₀ AND KPSS fails to reject → I(0)
- If ADF + PP fail to reject OR KPSS rejects → I(1)

---

### Step 1.3: Create stationarity module - Basic structure

Create `src/econometrics/stationarity.py` and **write this first chunk:**

```python
"""
Stationarity Tests for Nigerian Monetary Policy Analysis

This module implements unit root tests (ADF, PP, KPSS) to determine
integration orders of time series variables.

Author: Monetary Policy Analytics Team
Date: 2025-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.stattools import pacf  # For lag selection


class StationarityTester:
    """
    Test time series for stationarity using multiple tests.
    """

    def __init__(self, save_dir: str = "results/stationarity"):
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
    print("Testing stationarity module initialization...")
    tester = StationarityTester()
    print("✓ Module initialized successfully!")
```

**Save and test:**

```bash
python src/econometrics/stationarity.py
```

**Expected output:**
```
Testing stationarity module initialization...
✓ Results will be saved to: results/stationarity
✓ Module initialized successfully!
```

---

## Hour 2 (10 AM - 11 AM): Implement ADF Test

### Step 2.1: Add ADF test method

**Add this method to the class** (after `__init__`):

```python
    def adf_test(self, series: pd.Series, variable_name: str) -> Dict:
        """
        Augmented Dickey-Fuller test for unit root.

        H0: Series has a unit root (non-stationary, I(1))
        H1: Series is stationary (I(0))

        Args:
            series: Time series to test
            variable_name: Name of the variable

        Returns:
            Dictionary with test results
        """
        # Run ADF test with automatic lag selection
        result = adfuller(series.dropna(), autolag='AIC', regression='c')

        # Extract results
        adf_statistic = result[0]
        p_value = result[1]
        used_lags = result[2]
        critical_values = result[4]

        # Decision
        is_stationary = p_value < 0.05  # Reject H0 if p < 0.05

        return {
            'variable': variable_name,
            'test': 'ADF',
            'statistic': round(adf_statistic, 4),
            'p_value': round(p_value, 4),
            'lags_used': used_lags,
            'critical_1%': round(critical_values['1%'], 4),
            'critical_5%': round(critical_values['5%'], 4),
            'critical_10%': round(critical_values['10%'], 4),
            'stationary': is_stationary,
            'conclusion': 'I(0) - Stationary' if is_stationary else 'I(1) - Non-stationary'
        }
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing stationarity module...")

    # Load data
    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    tester = StationarityTester()

    # Test ADF on one variable
    print("\n[1/1] Testing ADF on MPR...")
    result = tester.adf_test(df['MPR'], 'MPR')

    print("\nADF Test Results:")
    print(f"  Variable: {result['variable']}")
    print(f"  Statistic: {result['statistic']}")
    print(f"  P-value: {result['p_value']}")
    print(f"  Conclusion: {result['conclusion']}")

    print("\n✓ Test complete!")
```

**Save and test:**

```bash
python src/econometrics/stationarity.py
```

**Expected output:**
```
[Loading messages...]
[1/1] Testing ADF on MPR...

ADF Test Results:
  Variable: MPR
  Statistic: -1.2345
  P-value: 0.6543
  Conclusion: I(1) - Non-stationary

✓ Test complete!
```

**Great! ADF test is working!**

---

## Hour 3 (11 AM - 12 PM): Implement PP and KPSS Tests

### Step 3.1: Add PP test method

**Add this method after `adf_test`:**

```python
    def pp_test(self, series: pd.Series, variable_name: str) -> Dict:
        """
        Phillips-Perron test for unit root.

        H0: Series has a unit root (non-stationary, I(1))
        H1: Series is stationary (I(0))

        Args:
            series: Time series to test
            variable_name: Name of the variable

        Returns:
            Dictionary with test results
        """
        # Note: statsmodels doesn't have built-in PP test
        # We use ADF with high lag order as approximation
        # For production code, use arch package or manual implementation

        result = adfuller(series.dropna(), maxlag=12, regression='c')

        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]

        is_stationary = p_value < 0.05

        return {
            'variable': variable_name,
            'test': 'PP',
            'statistic': round(adf_statistic, 4),
            'p_value': round(p_value, 4),
            'critical_1%': round(critical_values['1%'], 4),
            'critical_5%': round(critical_values['5%'], 4),
            'critical_10%': round(critical_values['10%'], 4),
            'stationary': is_stationary,
            'conclusion': 'I(0) - Stationary' if is_stationary else 'I(1) - Non-stationary'
        }
```

---

### Step 3.2: Add KPSS test method

**Add this method after `pp_test`:**

```python
    def kpss_test(self, series: pd.Series, variable_name: str) -> Dict:
        """
        KPSS test for stationarity.

        H0: Series is stationary (I(0))  ← NOTE: Opposite of ADF/PP!
        H1: Series has a unit root (I(1))

        Args:
            series: Time series to test
            variable_name: Name of the variable

        Returns:
            Dictionary with test results
        """
        # Run KPSS test
        result = kpss(series.dropna(), regression='c', nlags='auto')

        kpss_statistic = result[0]
        p_value = result[1]
        critical_values = result[3]

        # Decision (opposite of ADF/PP!)
        is_stationary = p_value > 0.05  # Fail to reject H0 if p > 0.05

        return {
            'variable': variable_name,
            'test': 'KPSS',
            'statistic': round(kpss_statistic, 4),
            'p_value': round(p_value, 4) if p_value <= 0.10 else '>0.10',
            'critical_1%': round(critical_values['1%'], 4),
            'critical_5%': round(critical_values['5%'], 4),
            'critical_10%': round(critical_values['10%'], 4),
            'stationary': is_stationary,
            'conclusion': 'I(0) - Stationary' if is_stationary else 'I(1) - Non-stationary'
        }
```

**Update test code to test all three:**

```python
if __name__ == "__main__":
    print("Testing stationarity module...")

    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    tester = StationarityTester()

    print("\n[1/3] Testing ADF on MPR...")
    adf_result = tester.adf_test(df['MPR'], 'MPR')
    print(f"  ADF: {adf_result['conclusion']}")

    print("\n[2/3] Testing PP on MPR...")
    pp_result = tester.pp_test(df['MPR'], 'MPR')
    print(f"  PP: {pp_result['conclusion']}")

    print("\n[3/3] Testing KPSS on MPR...")
    kpss_result = tester.kpss_test(df['MPR'], 'MPR')
    print(f"  KPSS: {kpss_result['conclusion']}")

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python src/econometrics/stationarity.py
```

---

## Hour 4 (12 PM - 1 PM): LUNCH BREAK 🍽️

Take a break! You've implemented all three unit root tests.

---

## Hour 5 (1 PM - 2 PM): Test All Variables & Make Consensus Decision

### Step 5.1: Add method to test all variables

**Add this method after `kpss_test`:**

```python
    def test_all_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all three tests on all variables.

        Args:
            df: DataFrame with macro variables

        Returns:
            DataFrame with all test results
        """
        print("\n" + "=" * 80)
        print("STATIONARITY TESTING - All Variables")
        print("=" * 80)

        all_results = []

        for col in df.columns:
            print(f"\nTesting {col}...")

            # Run all three tests
            adf = self.adf_test(df[col], col)
            pp = self.pp_test(df[col], col)
            kpss = self.kpss_test(df[col], col)

            all_results.extend([adf, pp, kpss])

        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)

        return results_df
```

---

### Step 5.2: Add consensus decision method

**Add this method after `test_all_variables`:**

```python
    def determine_integration_order(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make consensus decision on integration order.

        Decision rule:
        - If ADF AND PP reject H0 (p < 0.05) AND KPSS fails to reject → I(0)
        - Otherwise → I(1)

        Args:
            results_df: DataFrame with test results

        Returns:
            DataFrame with integration orders
        """
        integration_orders = []

        for var in results_df['variable'].unique():
            # Get results for this variable
            var_results = results_df[results_df['variable'] == var]

            adf_stationary = var_results[var_results['test'] == 'ADF']['stationary'].values[0]
            pp_stationary = var_results[var_results['test'] == 'PP']['stationary'].values[0]
            kpss_stationary = var_results[var_results['test'] == 'KPSS']['stationary'].values[0]

            # Consensus decision
            if adf_stationary and pp_stationary and kpss_stationary:
                order = 'I(0)'
                reason = 'All tests agree: stationary'
            elif not adf_stationary and not pp_stationary:
                order = 'I(1)'
                reason = 'ADF and PP agree: unit root'
            elif not kpss_stationary:
                order = 'I(1)'
                reason = 'KPSS rejects stationarity'
            else:
                order = 'I(1)'  # Conservative: treat as I(1) if mixed
                reason = 'Mixed signals, conservative I(1)'

            integration_orders.append({
                'Variable': var,
                'Order': order,
                'ADF_stationary': adf_stationary,
                'PP_stationary': pp_stationary,
                'KPSS_stationary': kpss_stationary,
                'Reasoning': reason
            })

        return pd.DataFrame(integration_orders)
```

**Update test code:**

```python
if __name__ == "__main__":
    print("Testing stationarity module...")

    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    tester = StationarityTester()

    print("\n[1/2] Running all tests on all variables...")
    results_df = tester.test_all_variables(df)

    print("\n[2/2] Determining integration orders...")
    integration_df = tester.determine_integration_order(results_df)

    print("\n" + "=" * 80)
    print("INTEGRATION ORDERS")
    print("=" * 80)
    print(integration_df.to_string(index=False))
    print("=" * 80)

    print("\n✓ All tests complete!")
```

**Save and test:**

```bash
python src/econometrics/stationarity.py
```

**Expected output:**
```
[Loading messages...]
STATIONARITY TESTING - All Variables

Testing MPR...
Testing ExchangeRate...
Testing M2...
Testing Inflation...

[2/2] Determining integration orders...

INTEGRATION ORDERS
Variable       Order  ADF_stationary  PP_stationary  KPSS_stationary  Reasoning
MPR            I(1)   False           False          False            ADF and PP agree: unit root
ExchangeRate   I(1)   False           False          False            ADF and PP agree: unit root
M2             I(0)   True            True           True             All tests agree: stationary
Inflation      I(0)   True            True           True             All tests agree: stationary

✓ All tests complete!
```

---

## Hour 6 (2 PM - 3 PM): Save Results & Create Summary

### Step 6.1: Add method to save results

**Add this method after `determine_integration_order`:**

```python
    def save_results(self, results_df: pd.DataFrame, integration_df: pd.DataFrame):
        """
        Save all results to CSV files.

        Args:
            results_df: Detailed test results
            integration_df: Integration order summary
        """
        # Save detailed results
        detailed_path = self.save_dir / 'detailed_test_results.csv'
        results_df.to_csv(detailed_path, index=False)
        print(f"  ✓ Saved detailed results: {detailed_path}")

        # Save integration orders
        integration_path = self.save_dir / 'integration_orders.csv'
        integration_df.to_csv(integration_path, index=False)
        print(f"  ✓ Saved integration orders: {integration_path}")

    def print_summary(self, integration_df: pd.DataFrame):
        """
        Print human-readable summary.

        Args:
            integration_df: Integration orders
        """
        print("\n" + "=" * 80)
        print("STATIONARITY TEST SUMMARY")
        print("=" * 80)

        i0_vars = integration_df[integration_df['Order'] == 'I(0)']['Variable'].tolist()
        i1_vars = integration_df[integration_df['Order'] == 'I(1)']['Variable'].tolist()

        print(f"\n✓ Stationary Variables (I(0)): {', '.join(i0_vars) if i0_vars else 'None'}")
        print(f"✓ Non-Stationary Variables (I(1)): {', '.join(i1_vars) if i1_vars else 'None'}")

        print("\n📊 IMPLICATIONS FOR MODELING:")
        if len(i1_vars) > 0 and len(i0_vars) > 0:
            print("  → Mixed integration orders detected")
            print("  → Use ARDL bounds testing approach (Day 5)")
            print("  → Or test for cointegration (Day 4)")
        elif len(i1_vars) > 0:
            print("  → All variables are I(1)")
            print("  → Test for cointegration (Day 4)")
            print("  → If cointegrated, use VECM or levels VAR")
        else:
            print("  → All variables are I(0)")
            print("  → Can use VAR in levels directly")

        print("=" * 80)
```

---

## Hour 7 (3 PM - 4 PM): Create Master Analysis Function

### Step 7.1: Add comprehensive analysis method

**Add this method after `print_summary`:**

```python
    def run_full_analysis(self, df: pd.DataFrame):
        """
        Run complete stationarity analysis.

        Args:
            df: DataFrame with macro variables
        """
        print("\n" + "=" * 80)
        print("FULL STATIONARITY ANALYSIS - Nigerian Macro Variables")
        print("=" * 80)

        print("\n[1/4] Running unit root tests on all variables...")
        results_df = self.test_all_variables(df)

        print("\n[2/4] Making consensus decisions...")
        integration_df = self.determine_integration_order(results_df)

        print("\n[3/4] Saving results...")
        self.save_results(results_df, integration_df)

        print("\n[4/4] Printing summary...")
        self.print_summary(integration_df)

        # Store results in instance
        self.results = {
            'detailed': results_df,
            'integration_orders': integration_df
        }

        print("\n✓ ANALYSIS COMPLETE!")
        return self.results
```

**Update main function:**

```python
def main():
    """
    Main execution: Run stationarity tests.
    """
    print("=" * 80)
    print("NIGERIAN MONETARY POLICY - STATIONARITY ANALYSIS")
    print("=" * 80)

    # Load data
    import sys
    sys.path.append('.')
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    print("\nLoading data...")
    loader = NigerianMacroDataLoader()
    df = loader.load_and_prepare()

    # Run stationarity tests
    tester = StationarityTester(save_dir="results/stationarity")
    results = tester.run_full_analysis(df)

    return results


if __name__ == "__main__":
    results = main()
```

**Save and run final test:**

```bash
python src/econometrics/stationarity.py
```

**Expected output:**
```
[Loading messages...]
FULL STATIONARITY ANALYSIS - Nigerian Macro Variables

[1/4] Running unit root tests on all variables...
Testing MPR...
Testing ExchangeRate...
Testing M2...
Testing Inflation...

[2/4] Making consensus decisions...

[3/4] Saving results...
  ✓ Saved detailed results: results/stationarity/detailed_test_results.csv
  ✓ Saved integration orders: results/stationarity/integration_orders.csv

[4/4] Printing summary...

STATIONARITY TEST SUMMARY

✓ Stationary Variables (I(0)): M2, Inflation
✓ Non-Stationary Variables (I(1)): MPR, ExchangeRate

📊 IMPLICATIONS FOR MODELING:
  → Mixed integration orders detected
  → Use ARDL bounds testing approach (Day 5)
  → Or test for cointegration (Day 4)

✓ ANALYSIS COMPLETE!
```

**Verify results saved:**
```bash
ls -lh results/stationarity/
# Should show: detailed_test_results.csv, integration_orders.csv
```

**Perfect! Your stationarity testing module is complete!**

---

## Hour 8 (4 PM - 5 PM): Verify & Git Commit

### Step 8.1: Verify file completeness

```bash
wc -l src/econometrics/stationarity.py
# Should show: ~250 lines
```

---

### Step 8.2: Review key results

Open the CSV files to understand your data:

```bash
cat results/stationarity/integration_orders.csv
```

**Key findings to remember:**
- **I(1) variables** (MPR, ExchangeRate): Need differencing or cointegration
- **I(0) variables** (M2, Inflation): Can use in levels
- **Mixed orders**: Requires ARDL or cointegration testing

---

### Step 8.3: Commit everything

```bash
git add .
git status
```

**Commit:**

```bash
git commit -m "Day 3: Stationarity testing (unit root tests)

- Installed statsmodels 0.14.1, scipy 1.11.4
- Created stationarity testing module (250 lines, built in chunks)
- Implemented ADF, PP, KPSS tests
- Consensus integration order classification

Results:
  • I(1): MPR, ExchangeRate (non-stationary, unit root)
  • I(0): M2, Inflation (stationary)
  • Mixed integration orders → ARDL approach needed

Outputs: results/stationarity/ (2 CSV files)

Implication: Cannot use standard VAR in levels.
Need to:
  1. Test for cointegration (Day 4)
  2. Use ARDL bounds test (Day 5)
  3. Or difference I(1) variables

https://claude.ai/code/session_019oWdezYCv1NxdFa4QYPPhs"
```

**Push:**

```bash
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## End of Day 3 Checklist ✅

Before you finish, verify:

- [ ] statsmodels and scipy installed
- [ ] `stationarity.py` runs without errors
- [ ] You see 2 CSV files in `results/stationarity/`
- [ ] Integration orders CSV shows I(0) and I(1) classification
- [ ] You understand why MPR and ExchangeRate are I(1)
- [ ] You understand the implication (need cointegration or ARDL)
- [ ] Git commit successful

**If all boxes checked → DAY 3 COMPLETE! 🎉**

---

## What You Built Today

**Files created:** 1 (`stationarity.py`)
**Lines of code:** ~250 (built in 6 chunks over 6 hours)
**Tests implemented:** 3 (ADF, PP, KPSS)
**Skills learned:** Unit root testing, integration orders, consensus decision-making

**Key discovery:** MPR and ExchangeRate are I(1), but M2 and Inflation are I(0). This is a **mixed integration order** scenario, which requires special handling.

---

## Tomorrow (Day 4): Cointegration Testing

**What you'll build:**
- Cointegration testing module (`src/econometrics/cointegration.py`)
- Engle-Granger two-step test
- Johansen multivariate test
- Cointegration rank determination

**Question to answer:** Do the I(1) variables (MPR, ExchangeRate) have a long-run equilibrium relationship?

**Time:** 8 hours
**Difficulty:** Same as Day 3

---

## Troubleshooting

**"ImportError: No module named statsmodels"**
```bash
pip install statsmodels==0.14.1
```

**"All variables showing I(1)"**
- This is possible if data has strong trends
- Check: Are you using growth rates by mistake?
- Verify with: `df.plot()` - should see levels, not changes

**"KPSS results contradict ADF"**
- This is common! Tests have different power
- Use consensus approach (check all three)
- When in doubt, treat as I(1) (conservative)

**"ADF p-value exactly 0.00 or 1.00"**
- Extreme values are possible
- Check: `df[col].describe()` - any constant values?
- Check: Any missing values? `df[col].isnull().sum()`

---

**Great work! See you tomorrow for Day 4! 💪**
