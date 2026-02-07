"""
Robustness Checks — Nigerian Monetary Policy Transmission

Tests sensitivity of key results to modeling choices:
  1. Alternative lag orders (VAR(3) vs VAR(11))
  2. Alternative Cholesky ordering (Inflation first vs MPR first)
  3. Sub-sample analysis (pre-2020 vs post-2020)

Day 13 deliverable.  Addresses examiner questions:
  "Are your results robust to different specifications?"

References
----------
Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48.
"""

import sys
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class RobustnessChecker:
    """
    Run robustness checks on the baseline VAR.

    Parameters
    ----------
    df       : pd.DataFrame   –  macro data (DatetimeIndex)
    baseline_ordering : list[str]  –  baseline Cholesky ordering
    baseline_lag      : int        –  baseline lag order
    save_dir : str | Path          –  output directory
    """

    def __init__(self, df: pd.DataFrame, baseline_ordering: list[str],
                 baseline_lag: int, save_dir: str = "results/robustness"):
        self.df               = df
        self.baseline_ordering = baseline_ordering
        self.baseline_lag      = baseline_lag
        self.save_dir          = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Alternative lag orders
    # ─────────────────────────────────────────────────────────────
    def compare_lag_orders(self, lags: list[int]) -> pd.DataFrame:
        """
        Compare IRF peak responses under different lag orders.
        """
        from statsmodels.tsa.api import VAR

        results = []
        for lag in lags:
            try:
                var = VAR(self.df[self.baseline_ordering]).fit(lag)
                irf_obj = var.irf(24)

                mpr_idx = self.baseline_ordering.index("MPR")
                inf_idx = self.baseline_ordering.index("Inflation")

                inf_response = irf_obj.orth_irfs[:, inf_idx, mpr_idx]
                peak_response = inf_response.min()   # most negative
                peak_month    = int(inf_response.argmin())

                results.append({
                    "lag_order": lag,
                    "AIC": round(var.aic, 2),
                    "BIC": round(var.bic, 2),
                    "peak_inflation_response": round(peak_response, 4),
                    "peak_month": peak_month,
                })
            except Exception as exc:
                results.append({"lag_order": lag, "error": str(exc)})

        return pd.DataFrame(results)

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Alternative ordering
    # ─────────────────────────────────────────────────────────────
    def compare_orderings(self) -> dict:
        """
        Compare baseline ordering vs alternative (Inflation first).
        """
        from statsmodels.tsa.api import VAR

        # Baseline: MPR first
        df_baseline = self.df[self.baseline_ordering]
        var_baseline = VAR(df_baseline).fit(self.baseline_lag)
        irf_baseline = var_baseline.irf(24)

        # Alternative: Inflation first
        alt_ordering = ["Inflation", "ExchangeRate", "M2", "MPR"]
        df_alt = self.df[alt_ordering]
        var_alt = VAR(df_alt).fit(self.baseline_lag)
        irf_alt = var_alt.irf(24)

        # Extract MPR → Inflation IRFs
        mpr_idx_base = self.baseline_ordering.index("MPR")
        inf_idx_base = self.baseline_ordering.index("Inflation")
        mpr_idx_alt  = alt_ordering.index("MPR")
        inf_idx_alt  = alt_ordering.index("Inflation")

        irf_base_series = irf_baseline.orth_irfs[:, inf_idx_base, mpr_idx_base]
        irf_alt_series  = irf_alt.orth_irfs[:, inf_idx_alt, mpr_idx_alt]

        return {
            "baseline": {
                "ordering": self.baseline_ordering,
                "peak": round(float(irf_base_series.min()), 4),
                "peak_month": int(irf_base_series.argmin()),
            },
            "alternative": {
                "ordering": alt_ordering,
                "peak": round(float(irf_alt_series.min()), 4),
                "peak_month": int(irf_alt_series.argmin()),
            },
        }

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Sub-sample
    # ─────────────────────────────────────────────────────────────
    def compare_subsamples(self, split_date: str = "2020-01-01") -> dict:
        """
        Compare VAR estimates pre- and post-split.
        """
        from statsmodels.tsa.api import VAR

        split_dt = pd.to_datetime(split_date)
        df_pre   = self.df.loc[:split_dt]
        df_post  = self.df.loc[split_dt:]

        results = {}
        for label, df_sub in [("pre", df_pre), ("post", df_post)]:
            try:
                var = VAR(df_sub[self.baseline_ordering]).fit(self.baseline_lag)
                irf_obj = var.irf(24)

                mpr_idx = self.baseline_ordering.index("MPR")
                inf_idx = self.baseline_ordering.index("Inflation")
                inf_response = irf_obj.orth_irfs[:, inf_idx, mpr_idx]

                results[label] = {
                    "n_obs": len(df_sub),
                    "peak_response": round(float(inf_response.min()), 4),
                    "peak_month": int(inf_response.argmin()),
                }
            except Exception as exc:
                results[label] = {"error": str(exc)}

        return results

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self) -> None:
        print("\n" + "=" * 70)
        print("  ROBUSTNESS CHECKS")
        print("=" * 70)

        # Test 1: Lag orders
        print("\n  [1/3] Alternative lag orders …")
        lag_df = self.compare_lag_orders([3, 6, 11])
        print("\n  " + "─" * 65)
        print(f"  {'Lag':>4} {'AIC':>10} {'BIC':>10} {'Peak Inf Resp':>15} {'Peak Month':>12}")
        print("  " + "─" * 65)
        for _, row in lag_df.iterrows():
            if "error" in row:
                print(f"  {int(row['lag_order']):>4}  ERROR: {row.get('error')}")
                continue
            flag = " ◄" if int(row["lag_order"]) == self.baseline_lag else ""
            print(f"  {int(row['lag_order']):>4} {row['AIC']:>10.2f} {row['BIC']:>10.2f}"
                  f" {row['peak_inflation_response']:>15.4f} {int(row['peak_month']):>12}{flag}")

        path = self.save_dir / "lag_order_comparison.csv"
        lag_df.to_csv(path, index=False)
        print(f"\n  ✓ saved  {path}")

        # Test 2: Orderings
        print("\n  [2/3] Alternative Cholesky ordering …")
        ord_comp = self.compare_orderings()
        print(f"\n    Baseline  ({' → '.join(ord_comp['baseline']['ordering'])})")
        print(f"      Peak: {ord_comp['baseline']['peak']:.4f}  Month: {ord_comp['baseline']['peak_month']}")
        print(f"\n    Alternative  ({' → '.join(ord_comp['alternative']['ordering'])})")
        print(f"      Peak: {ord_comp['alternative']['peak']:.4f}  Month: {ord_comp['alternative']['peak_month']}")

        # Test 3: Sub-samples
        print("\n  [3/3] Sub-sample analysis (pre-2020 vs post-2020) …")
        subsample = self.compare_subsamples()
        for label, res in subsample.items():
            if "error" in res:
                print(f"\n    {label.upper()}: ERROR — {res['error']}")
                continue
            print(f"\n    {label.upper()} ({res['n_obs']} obs)")
            print(f"      Peak: {res['peak_response']:.4f}  Month: {res['peak_month']}")

        print()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    baseline_ordering = ["MPR", "ExchangeRate", "M2", "Inflation"]
    baseline_lag      = 11

    checker = RobustnessChecker(df, baseline_ordering, baseline_lag)
    checker.run_full_analysis()


if __name__ == "__main__":
    main()
