"""
ARDL Bounds Testing — Nigerian Monetary Policy Transmission

Pesaran, Shin & Smith (2001) bounds-testing approach.  Handles the
mixed I(0) / I(1) integration orders produced by Day 3 naturally —
no need to difference any series.

Dependent variable  :  Inflation
Regressors          :  MPR, ExchangeRate, M2

Steps
-----
1.  Grid-search over (p, q₁, q₂, q₃) — select by AIC
2.  Re-estimate as UECM (error-correction reparameterisation)
3.  Bounds test  (F-stat vs Pesaran critical-value bands)
4.  If cointegrated → extract long-run equation + ECM short-run
5.  Residual diagnostics (serial correlation, normality)

Day 6 deliverable.  Requires statsmodels ≥ 0.14.

Reference
---------
Pesaran, M. H., Shin, Y. & Smith, R. J. (2001).
  Bounds testing approaches to the analysis of level relationships.
  Journal of Applied Econometrics, 16(3), 289–326.
"""

import sys
import itertools
import pandas          as pd
import numpy           as np
import matplotlib.pyplot as plt
from pathlib           import Path
from typing            import Dict, List

from statsmodels.tsa.ardl          import ARDL, UECM
from statsmodels.stats.stattools   import durbin_watson
from scipy.stats                   import jarque_bera


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class ARDLBoundsTest:
    """
    Full ARDL bounds-testing pipeline.

    Parameters
    ----------
    df         : pd.DataFrame   –  macro data with DatetimeIndex
    dependent  : str            –  name of the dependent variable
    regressors : list[str]      –  ordered regressor names
    save_dir   : str | Path     –  output directory
    """

    # Pesaran (2001) deterministic-term cases:
    #   3 = unrestricted constant, no trend   ← default for macro levels
    #   5 = unrestricted constant + trend
    BOUNDS_CASE = 3

    def __init__(self,
                 df: pd.DataFrame,
                 dependent:  str            = "Inflation",
                 regressors: List[str] | None = None,
                 save_dir:   str            = "results/ardl"):
        self.df         = df
        self.dependent  = dependent
        self.regressors = regressors or [c for c in df.columns if c != dependent]
        self.save_dir   = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.endog = df[self.dependent]          # Series (name preserved)
        self.exog  = df[self.regressors]         # DataFrame

        # populated by the pipeline
        self.best_order : Dict      = {}
        self.uecm_res               = None
        self.bounds_res             = None
        self.long_run   : pd.Series = pd.Series(dtype=float)
        self.full_summary : pd.DataFrame = pd.DataFrame()

    # ─────────────────────────────────────────────────────────────
    # LAG SELECTION  (exhaustive AIC grid)
    # ─────────────────────────────────────────────────────────────
    def _select_lags(self, max_p: int = 4, max_q: int = 4) -> Dict:
        """
        Exhaustive grid-search: ARDL(p, q₁, …, qₖ).
        Returns the spec with minimum AIC.

        Note: UECM requires order ≥ 1 for each regressor, so q_i
        starts at 1.
        """
        n_models = max_p * (max_q ** len(self.regressors))
        print(f"  [Lag selection] grid: p ∈ [1,{max_p}],"
              f" q_i ∈ [1,{max_q}]  ({n_models} models)")

        best_aic  = np.inf
        best_spec = {}
        count     = 0

        q_ranges = [range(1, max_q + 1) for _ in self.regressors]

        for p in range(1, max_p + 1):
            for qs in itertools.product(*q_ranges):
                order = dict(zip(self.regressors, qs))
                try:
                    model = ARDL(self.endog, lags=p,
                                 exog=self.exog, order=order, trend='c')
                    res   = model.fit()
                    if res.aic < best_aic:
                        best_aic  = res.aic
                        best_spec = {'ar': p, **order}
                except Exception:
                    pass
                count += 1

        print(f"  [Lag selection] evaluated {count} models.")
        print(f"  [Lag selection] best AIC = {best_aic:.2f}  →  {best_spec}")
        self.best_order = best_spec
        return best_spec

    # ─────────────────────────────────────────────────────────────
    # UECM FIT
    # ─────────────────────────────────────────────────────────────
    def _fit_uecm(self) -> None:
        """Fit UECM with the AIC-selected lag structure."""
        ar_lags  = self.best_order['ar']
        dl_order = {k: v for k, v in self.best_order.items() if k != 'ar'}

        model         = UECM(self.endog, lags=ar_lags,
                             exog=self.exog, order=dl_order, trend='c')
        self.uecm_res = model.fit()
        print(f"  [UECM] fitted  –  {len(self.uecm_res.params)} parameters")

    # ─────────────────────────────────────────────────────────────
    # BOUNDS TEST
    # ─────────────────────────────────────────────────────────────
    def _run_bounds_test(self) -> None:
        """Pesaran bounds test + decision rule."""
        self.bounds_res = self.uecm_res.bounds_test(case=self.BOUNDS_CASE)

        print("\n" + "=" * 70)
        print("  ARDL  –  BOUNDS TEST")
        print("=" * 70)
        print(f"  F-statistic  :  {self.bounds_res.stat:.4f}")
        print(f"  Case         :  {self.BOUNDS_CASE} (unrestricted constant)")
        print(f"  k            :  {len(self.regressors)} regressors")
        print()
        print(f"  {'Significance':<14} {'Lower I(0)':>12} {'Upper I(1)':>12}")
        print("  " + "─" * 42)
        for pct, row in self.bounds_res.crit_vals.iterrows():
            marker = ""
            if self.bounds_res.stat > row["upper"]:  marker = " ◄"
            elif self.bounds_res.stat > row["lower"]: marker = " ←"
            print(f"  {pct:>10.0f} %  {row['lower']:>12.4f} {row['upper']:>12.4f}{marker}")

        print()
        upper_p = float(self.bounds_res.p_values["upper"])
        lower_p = float(self.bounds_res.p_values["lower"])
        if upper_p < 0.05:
            print("  📌 COINTEGRATION CONFIRMED  (F > upper bound at 5 %)")
            print("     → Long-run + ECM equations are valid.")
        elif lower_p < 0.05:
            print("  📌 INCONCLUSIVE  (F between lower and upper bounds at 5 %)")
            print("     → Borderline evidence; ECM is tentatively supported.")
        else:
            print("  📌 NO COINTEGRATION  (F < lower bound at 5 %)")
            print("     → Long-run relationship not supported at 5 %.")
        print()

    # ─────────────────────────────────────────────────────────────
    # COEFFICIENT EXTRACTION
    # ─────────────────────────────────────────────────────────────
    def _extract_coefficients(self) -> None:
        """
        Long-run multipliers + full ECM parameter table.

        UECM layout (Pesaran notation):
            Δy_t = const
                 + π  y_{t-1}                  speed of adjustment
                 + θ₁ x₁_{t-1} + …            level regressors
                 + Σ φ_j  Δy_{t-j}            short-run AR
                 + Σ ψᵢⱼ  Δxᵢ_{t-j}          short-run DL

        Long-run multiplier  :  θᵢ / (−π)
        """
        params = self.uecm_res.params

        # speed of adjustment (π)
        pi_key  = f"{self.dependent}.L1"
        pi      = float(params[pi_key])

        # long-run multipliers
        lr = {"speed_of_adjustment": round(pi, 6)}
        for reg in self.regressors:
            theta = float(params[f"{reg}.L1"])
            lr[reg] = round(theta / (-pi), 6) if pi != 0 else float('inf')
        self.long_run = pd.Series(lr)

        # full parameter table with t-stats + p-values
        self.full_summary = pd.DataFrame({
            "coeff":   self.uecm_res.params.round(6),
            "std_err": self.uecm_res.bse.round(6),
            "t_stat":  self.uecm_res.tvalues.round(4),
            "p_value": self.uecm_res.pvalues.round(4),
        })

    # ─────────────────────────────────────────────────────────────
    # PRINT
    # ─────────────────────────────────────────────────────────────
    def _print_equations(self) -> None:
        print("=" * 70)
        print("  LONG-RUN EQUATION  (equilibrium relationship)")
        print("=" * 70)
        pi = self.long_run["speed_of_adjustment"]
        print(f"\n  Speed of adjustment (π)  :  {pi}")
        stability = "stable ECM" if pi < 0 else "WARNING — unstable (π > 0)"
        print(f"    ({stability})")
        print()
        print(f"  {'Regressor':<18} {'Long-run coeff':>16}")
        print("  " + "─" * 38)
        for reg in self.regressors:
            print(f"  {reg:<18} {self.long_run[reg]:>16.6f}")
        print()

        print("=" * 70)
        print("  SHORT-RUN  /  ECM EQUATION  (full UECM)")
        print("=" * 70)
        print(f"\n  {'Parameter':<22} {'Coeff':>10} {'Std Err':>10}"
              f" {'t-stat':>8} {'p-val':>8}  Sig")
        print("  " + "─" * 66)
        for idx, row in self.full_summary.iterrows():
            stars = ""
            if row["p_value"] < 0.01:   stars = "***"
            elif row["p_value"] < 0.05: stars = "** "
            elif row["p_value"] < 0.10: stars = "*  "
            print(f"  {idx:<22} {row['coeff']:>10.4f} {row['std_err']:>10.4f}"
                  f" {row['t_stat']:>8.3f} {row['p_value']:>8.4f}  {stars}")
        print("  Significance codes: *** p<0.01  ** p<0.05  * p<0.10\n")

    # ─────────────────────────────────────────────────────────────
    # DIAGNOSTICS + PLOTS
    # ─────────────────────────────────────────────────────────────
    def _diagnostics(self) -> Dict:
        """Durbin-Watson + Jarque-Bera on UECM residuals."""
        resid = self.uecm_res.resid
        dw    = float(durbin_watson(resid))
        jb_stat, jb_p = jarque_bera(resid)

        diag = {
            "durbin_watson":     round(dw, 4),
            "jarque_bera_stat":  round(float(jb_stat), 4),
            "jarque_bera_p":     round(float(jb_p), 4),
            "n_obs":             len(resid),
            "n_params":          len(self.uecm_res.params),
        }
        print("  Diagnostics")
        serial_ok = "OK" if 1.5 < dw < 2.5 else "⚠ check serial correlation"
        normal_ok = "OK" if diag["jarque_bera_p"] > 0.05 else "⚠ non-normal"
        print(f"    Durbin-Watson       :  {dw:.4f}  ({serial_ok})")
        print(f"    Jarque-Bera p       :  {diag['jarque_bera_p']:.4f}  ({normal_ok})")
        print(f"    Obs / Params        :  {diag['n_obs']} / {diag['n_params']}")
        return diag

    def _plot_residuals(self) -> None:
        """3-panel residual diagnostic chart."""
        resid = self.uecm_res.resid

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle("ARDL — Residual Diagnostics", fontsize=14, fontweight="bold")

        # 1 — time series
        axes[0].plot(resid.index, resid.values, color="#2C3E50", linewidth=0.9)
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_title("Residuals over time")
        axes[0].set_ylabel("Residual")
        axes[0].grid(True, alpha=0.2)

        # 2 — ACF
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(resid, lags=12, ax=axes[1], alpha=0.05)
        axes[1].set_title("ACF of residuals")

        # 3 — QQ
        from scipy.stats import probplot
        probplot(resid, dist="norm", plot=axes[2])
        axes[2].set_title("Normal Q-Q")
        axes[2].get_lines()[0].set_color("#2E86AB")
        axes[2].get_lines()[1].set_color("red")
        axes[2].grid(True, alpha=0.2)

        fig.tight_layout()
        path = self.save_dir / "ardl_residual_diagnostics.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────
    def _save_results(self, diag: Dict) -> None:
        # full UECM coefficient table
        self.full_summary.to_csv(self.save_dir / "ardl_uecm_coefficients.csv")
        print(f"  ✓ saved  {self.save_dir / 'ardl_uecm_coefficients.csv'}")

        # long-run equation
        pd.DataFrame([self.long_run]).to_csv(
            self.save_dir / "ardl_long_run.csv", index=False)
        print(f"  ✓ saved  {self.save_dir / 'ardl_long_run.csv'}")

        # bounds-test critical-value table
        bt = self.bounds_res
        bt_rows = []
        for pct, row in bt.crit_vals.iterrows():
            bt_rows.append({
                "significance_pct": pct,
                "lower_I0":  round(float(row["lower"]), 4),
                "upper_I1":  round(float(row["upper"]), 4),
            })
        bt_df = pd.DataFrame(bt_rows)
        bt_df["F_statistic"]   = round(float(bt.stat), 4)
        bt_df["upper_p_value"] = round(float(bt.p_values["upper"]), 4)
        bt_df["lower_p_value"] = round(float(bt.p_values["lower"]), 4)
        bt_df.to_csv(self.save_dir / "ardl_bounds_test.csv", index=False)
        print(f"  ✓ saved  {self.save_dir / 'ardl_bounds_test.csv'}")

        # diagnostics
        pd.DataFrame([diag]).to_csv(
            self.save_dir / "ardl_diagnostics.csv", index=False)
        print(f"  ✓ saved  {self.save_dir / 'ardl_diagnostics.csv'}")

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self, max_p: int = 4, max_q: int = 4) -> None:
        print("\n" + "=" * 70)
        print("  ARDL BOUNDS-TESTING  –  FULL PIPELINE")
        print(f"  Dependent: {self.dependent}  |  Regressors: {self.regressors}")
        print("=" * 70)

        print("\n  [1/5] Lag selection …")
        self._select_lags(max_p, max_q)

        print("\n  [2/5] Fitting UECM …")
        self._fit_uecm()

        print("\n  [3/5] Bounds test …")
        self._run_bounds_test()

        print("\n  [4/5] Extracting equations …")
        self._extract_coefficients()
        self._print_equations()

        print("  [5/5] Diagnostics & saving …")
        diag = self._diagnostics()
        self._plot_residuals()
        self._save_results(diag)
        print()


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    model = ARDLBoundsTest(df,
                           dependent  = "Inflation",
                           regressors = ["MPR", "ExchangeRate", "M2"])
    model.run_full_analysis()


if __name__ == "__main__":
    main()
