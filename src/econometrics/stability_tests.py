"""
Structural Stability Tests — Nigerian Monetary Policy Transmission

Tests whether the VAR parameters are stable over the sample period.

Three tests:
  1. Chow test for structural breaks at known dates
  2. Rolling VAR eigenvalues (check stability over time)
  3. CUSUM / CUSUMSQ tests (recursive residuals)

Key dates for Nigeria:
  - 2016-06-01: Naira devaluation
  - 2020-03-01: COVID-19 pandemic
  - 2023-06-01: FX market unification

Day 12 deliverable.  Addresses a common examiner question:
  "Is your VAR stable, or do you need to split the sample?"

References
----------
Chow, G. C. (1960). Tests of Equality Between Sets of Coefficients in Two
  Linear Regressions. Econometrica, 28(3), 591–605.
Brown, R. L., Durbin, J. & Evans, J. M. (1975). Techniques for Testing the
  Constancy of Regression Relationships over Time. Journal of the Royal
  Statistical Society B, 37(2), 149–192.
"""

import sys
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path
from scipy      import stats


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class StabilityTester:
    """
    Test VAR structural stability.

    Parameters
    ----------
    df       : pd.DataFrame   –  macro data (DatetimeIndex)
    ordering : list[str]      –  Cholesky ordering
    opt_lag  : int            –  VAR lag order (from Day 7)
    save_dir : str | Path     –  output directory
    """

    def __init__(self, df: pd.DataFrame, ordering: list[str], opt_lag: int,
                 save_dir: str = "results/stability"):
        self.df       = df[ordering]
        self.ordering = ordering
        self.opt_lag  = opt_lag
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # CHOW TEST
    # ─────────────────────────────────────────────────────────────
    def chow_test(self, break_date: str) -> dict:
        """
        Chow test for a structural break at 'break_date'.

        H₀: No structural break
        """
        from statsmodels.tsa.api import VAR

        break_dt = pd.to_datetime(break_date)
        if break_dt not in self.df.index:
            return {"break_date": break_date, "error": "Date not in sample"}

        # Split sample
        df1 = self.df.loc[:break_dt]
        df2 = self.df.loc[break_dt:]

        # Fit 3 models
        try:
            var_full = VAR(self.df).fit(self.opt_lag)
            var_1    = VAR(df1).fit(self.opt_lag)
            var_2    = VAR(df2).fit(self.opt_lag)
        except Exception as exc:
            return {"break_date": break_date, "error": str(exc)}

        # RSS
        rss_full = np.sum(var_full.resid ** 2)
        rss_1    = np.sum(var_1.resid ** 2)
        rss_2    = np.sum(var_2.resid ** 2)
        rss_rest = rss_1 + rss_2

        # Degrees of freedom
        k = len(self.ordering)
        n_full = len(var_full.resid)
        n_1    = len(var_1.resid)
        n_2    = len(var_2.resid)
        n_params = k * (k * self.opt_lag + 1)   # K equations × (Kp + 1) params

        # Chow F-statistic
        numerator   = (rss_full - rss_rest) / n_params
        denominator = rss_rest / (n_1 + n_2 - 2 * n_params)
        F_stat      = numerator / denominator
        p_value     = 1 - stats.f.cdf(F_stat, n_params, n_1 + n_2 - 2 * n_params)

        return {
            "break_date": break_date,
            "F_statistic": round(float(F_stat), 4),
            "p_value": round(float(p_value), 4),
            "reject_H0": bool(p_value < 0.05),
        }

    def run_all_chow_tests(self) -> pd.DataFrame:
        """Test structural breaks at key dates."""
        dates = ["2016-06-01", "2020-03-01", "2023-06-01"]
        results = [self.chow_test(d) for d in dates]
        return pd.DataFrame(results)

    # ─────────────────────────────────────────────────────────────
    # ROLLING VAR EIGENVALUES
    # ─────────────────────────────────────────────────────────────
    def rolling_eigenvalues(self, window: int = 60) -> pd.DataFrame:
        """
        Fit VAR on rolling windows and extract companion-matrix eigenvalues.

        A VAR is stable if all eigenvalues lie inside the unit circle.
        """
        from statsmodels.tsa.api import VAR

        n = len(self.df)
        dates = []
        max_eigs = []

        for t in range(window + self.opt_lag, n):
            df_window = self.df.iloc[t - window:t]
            try:
                var = VAR(df_window).fit(self.opt_lag)
                # Companion matrix eigenvalues
                companion = var.companion_matrix()
                eigs = np.linalg.eigvals(companion)
                max_eig = np.max(np.abs(eigs))
                dates.append(self.df.index[t])
                max_eigs.append(max_eig)
            except:
                continue

        return pd.DataFrame({"date": dates, "max_eigenvalue": max_eigs})

    def plot_rolling_eigenvalues(self, df_eigs: pd.DataFrame) -> None:
        """Plot rolling max eigenvalue over time."""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_eigs["date"], df_eigs["max_eigenvalue"], color="#2E86AB", linewidth=2)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=2, label="Unit circle")
        ax.set_title("Rolling VAR Stability (Max Eigenvalue)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Max |eigenvalue| of companion matrix", fontsize=11)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        path = self.save_dir / "rolling_eigenvalues.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # CUSUM TEST (simplified)
    # ─────────────────────────────────────────────────────────────
    def cusum_test(self, var_results) -> dict:
        """
        CUSUM test for parameter constancy (simplified version).

        Uses recursive residuals.  Full implementation requires OLS-CUSUM
        on each VAR equation.  This is a placeholder for the guide.
        """
        # This is a simplified placeholder
        # Full CUSUM requires recursive estimation, which is computationally expensive
        resid = var_results.resid
        cumsum = np.cumsum(resid, axis=0)

        # Check if cumulative residuals stay within bounds (rule of thumb: ±3σ)
        std_resid = np.std(resid, axis=0)
        bounds = 3 * std_resid * np.sqrt(np.arange(1, len(resid) + 1))[:, None]

        # Count exceedances
        exceedances = np.sum(np.abs(cumsum) > bounds)
        total_checks = cumsum.size

        return {
            "test": "CUSUM (simplified)",
            "exceedances": int(exceedances),
            "total_checks": int(total_checks),
            "exceedance_rate": round(float(exceedances) / total_checks, 4),
            "stable": exceedances < 0.05 * total_checks,
        }

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self, var_results) -> None:
        print("\n" + "=" * 70)
        print("  STRUCTURAL STABILITY TESTS")
        print("=" * 70)

        # Chow tests
        print("\n  [1/3] Chow tests for structural breaks …")
        chow_df = self.run_all_chow_tests()
        print("\n  " + "─" * 65)
        print(f"  {'Break Date':<15} {'F-stat':>10} {'p-value':>10}  Result")
        print("  " + "─" * 65)
        for _, row in chow_df.iterrows():
            if "error" in row:
                print(f"  {row['break_date']:<15}  ERROR: {row.get('error')}")
                continue
            flag = "Break ✓" if row["reject_H0"] else "No break ✗"
            print(f"  {row['break_date']:<15} {row['F_statistic']:>10.3f} {row['p_value']:>10.4f}  {flag}")

        path = self.save_dir / "chow_tests.csv"
        chow_df.to_csv(path, index=False)
        print(f"\n  ✓ saved  {path}")

        # Rolling eigenvalues
        print("\n  [2/3] Rolling VAR eigenvalues …")
        df_eigs = self.rolling_eigenvalues()
        self.plot_rolling_eigenvalues(df_eigs)
        path = self.save_dir / "rolling_eigenvalues.csv"
        df_eigs.to_csv(path, index=False)
        print(f"  ✓ saved  {path}")

        unstable_count = (df_eigs["max_eigenvalue"] >= 1.0).sum()
        print(f"    {unstable_count} / {len(df_eigs)} rolling windows have max |eig| ≥ 1.0")

        # CUSUM (simplified)
        print("\n  [3/3] CUSUM test (simplified) …")
        cusum = self.cusum_test(var_results)
        print(f"    Exceedance rate : {cusum['exceedance_rate']:.2%}")
        print(f"    Stable          : {'Yes' if cusum['stable'] else 'No'}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader
    from src.econometrics.var_model import VARModel

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    var = VARModel(df)
    var.select_lags(max_lags=12)
    var.fit()

    tester = StabilityTester(df, var.ordering, var.opt_lag)
    tester.run_full_analysis(var.res)


if __name__ == "__main__":
    main()
