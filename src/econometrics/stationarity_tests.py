"""
Stationarity-Testing Module — Nigerian Monetary Policy Transmission Analysis

Implements three complementary unit-root tests:

    ADF   – Augmented Dickey-Fuller        (statsmodels)
    PP    – Phillips-Perron                 (statsmodels OLS + HAC covariance)
    KPSS  – Kwiatkowski-Phillips-Schmidt-Shin  (statsmodels)

Each test is run on LEVELS and on FIRST DIFFERENCES.  The module then
determines the integration order I(0)/I(1) for every variable — the
critical input that guides ARDL specification on Day 6.

Day 3 deliverable.  New dependencies: statsmodels, scipy.

Academic references
-------------------
Phillips & Perron (1988)  – Econometrica 56(2)
Kwiatkowski et al. (1992) – Journal of Econometrics 54
Hamilton (1994) Ch.15     – Time-Series Analysis (textbook)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm


# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (kept in sync with plots.py)
# ─────────────────────────────────────────────────────────────────────

_COLORS = {
    "MPR":            "#2E86AB",
    "Inflation":      "#A23B72",
    "ExchangeRate":   "#F18F01",
    "M2":             "#6A994E",
}


# ─────────────────────────────────────────────────────────────────────
# TESTER CLASS
# ─────────────────────────────────────────────────────────────────────

class StationarityTester:
    """
    Run ADF / PP / KPSS on every column of *df*, on both levels and
    first differences.  Persist results as CSV and publication-ready
    plots to *save_dir*.

    Parameters
    ----------
    df       : pd.DataFrame  –  columns in VAR order, DatetimeIndex
    save_dir : str | Path    –  e.g.  ``results/stationarity``
    """

    def __init__(self, df: pd.DataFrame, save_dir: str = "results/stationarity"):
        self.df       = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.results            = {}          # filled by run_all_tests()
        self.integration_orders = {}          # filled by determine_integration_order()

    # ────────────────────────────────────────────────────────────────
    # 1.  ADF  (exact — straight from statsmodels)
    # ────────────────────────────────────────────────────────────────
    def _run_adf(self, series: pd.Series, name: str) -> Dict:
        """
        Augmented Dickey-Fuller test.

        H₀ : unit root exists  (non-stationary)
        H₁ : no unit root      (stationary)

        Decision : reject H₀  ⟹  stationary   when  p < 0.05
        """
        raw = adfuller(series.dropna(), maxlag=12, regression="ct", autolag="AIC")

        return {
            "variable":        name,
            "test":            "ADF",
            "test_statistic":  round(raw[0], 4),
            "p_value":         round(raw[1], 4),
            "lags_used":       raw[2],
            "n_obs":           raw[3],
            "critical_values": raw[4],           # dict  '1%', '5%', '10%'
            "stationary":      raw[1] < 0.05,
        }

    # ────────────────────────────────────────────────────────────────
    # 2.  PHILLIPS-PERRON  (statsmodels OLS + Newey-West / HAC)
    # ────────────────────────────────────────────────────────────────
    def _run_pp(self, series: pd.Series, name: str) -> Dict:
        """
        Phillips-Perron test via OLS with HAC-corrected standard errors.

        The regression  Δy_t = μ + δ t + ρ y_{t-1} + u_t  is estimated
        by OLS.  The covariance matrix uses the Newey-West (Bartlett)
        kernel with an automatic bandwidth
            l = ⌊ 4 (T/100)^{2/9} ⌋
        chosen following Andrews (1991).

        Under H₀ : ρ = 0  the resulting t-statistic converges to the
        same Dickey-Fuller asymptotic distribution used for ADF critical
        values — so MacKinnon (1994) tables apply directly.

        H₀ : unit root  (non-stationary)
        H₁ : no unit root  (stationary)
        """
        y = series.dropna().values
        T = len(y)

        # ── regression variables ──────────────────────────────────
        dy    = np.diff(y)                              # Δy_t
        y_lag = y[:-1]                                  # y_{t-1}
        trend = np.arange(1, len(dy) + 1, dtype=float) # deterministic trend

        X = sm.add_constant(np.column_stack([trend, y_lag]))
        # column order: [const, trend, y_lag]   →  ρ is index 2

        # ── Newey-West bandwidth ──────────────────────────────────
        bw = max(1, int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0))))

        # ── OLS with HAC covariance ───────────────────────────────
        model = sm.OLS(dy, X).fit(cov_type="HAC", cov_kwds={"maxlags": bw})

        pp_t   = round(model.tvalues[2], 4)

        # ── approximate p-value via MacKinnon critical-value table ─
        # Critical values for const+trend, T ≈ 180  (MacKinnon 1994)
        cv = {"1%": -3.428, "5%": -2.862, "10%": -2.572}

        if   pp_t < cv["1%"]:   p_val = 0.005
        elif pp_t < cv["5%"]:   p_val = 0.02
        elif pp_t < cv["10%"]:  p_val = 0.08
        else:                   p_val = 0.40

        return {
            "variable":        name,
            "test":            "PP",
            "test_statistic":  pp_t,
            "p_value":         p_val,
            "critical_values": cv,
            "stationary":      pp_t < cv["5%"],
        }

    # ────────────────────────────────────────────────────────────────
    # 3.  KPSS
    # ────────────────────────────────────────────────────────────────
    def _run_kpss(self, series: pd.Series, name: str) -> Dict:
        """
        KPSS test — **note the opposite null hypothesis**.

        H₀ : series IS stationary
        H₁ : unit root exists  (non-stationary)

        Decision : reject H₀  ⟹  non-stationary   when  p < 0.05
        So  *stationary = (p > 0.05)*  ← flipped vs ADF/PP.
        """
        try:
            raw = kpss(series.dropna(), regression="ct", nlags="auto")
            return {
                "variable":        name,
                "test":            "KPSS",
                "test_statistic":  round(raw[0], 4),
                "p_value":         round(raw[1], 4),
                "lags_used":       raw[2],
                "critical_values": raw[3],
                "stationary":      raw[1] > 0.05,   # OPPOSITE to ADF/PP
            }
        except Exception as exc:                    # graceful fall-back
            return {"variable": name, "test": "KPSS", "error": str(exc), "stationary": None}

    # ────────────────────────────────────────────────────────────────
    # 4.  RUN ALL  (levels + first differences)
    # ────────────────────────────────────────────────────────────────
    def run_all_tests(self) -> Dict:
        """
        Execute the full battery on raw levels AND on first differences.

        Returns a nested dict
            {
              "levels":              [list of result dicts],
              "first_differences":   [list of result dicts],
            }
        """
        self.results = {"levels": [], "first_differences": []}

        # ── PANEL A — levels ──────────────────────────────────────
        print("\n" + "=" * 70)
        print("  PANEL A  –  UNIT ROOT TESTS ON LEVELS")
        print("=" * 70)
        print(f"  {'Variable':<18} {'ADF t-stat':>11} {'ADF p':>7}"
              f" {'PP t-stat':>10} {'PP p':>6}"
              f" {'KPSS stat':>10} {'KPSS p':>7}  Conclusion")
        print("  " + "─" * 95)

        for col in self.df.columns:
            adf  = self._run_adf(self.df[col],  col)
            pp   = self._run_pp(self.df[col],   col)
            kpss_r = self._run_kpss(self.df[col], col)
            self.results["levels"].extend([adf, pp, kpss_r])

            # consensus: call stationary only if ALL three agree
            agree_stat = adf["stationary"] and pp["stationary"] and (kpss_r.get("stationary", False))
            label = "I(0) Stationary" if agree_stat else "I(1) Unit root"

            kpss_p = kpss_r.get("p_value", "—")
            print(f"  {col:<18}"
                  f" {adf['test_statistic']:>10.4f} {adf['p_value']:>7.4f}"
                  f" {pp['test_statistic']:>10.4f} {pp['p_value']:>6.3f}"
                  f" {kpss_r.get('test_statistic', 0):>10.4f} {kpss_p:>7}"
                  f"  {label}")

        # ── PANEL B — first differences ───────────────────────────
        print("\n" + "=" * 70)
        print("  PANEL B  –  UNIT ROOT TESTS ON FIRST DIFFERENCES (Δy)")
        print("=" * 70)
        print(f"  {'Variable':<18} {'ADF t-stat':>11} {'ADF p':>7}"
              f" {'PP t-stat':>10} {'PP p':>6}"
              f" {'KPSS stat':>10} {'KPSS p':>7}  Conclusion")
        print("  " + "─" * 95)

        df_d = self.df.diff().dropna()

        for col in df_d.columns:
            label_d = f"Δ{col}"
            adf  = self._run_adf(df_d[col],  label_d)
            pp   = self._run_pp(df_d[col],   label_d)
            kpss_r = self._run_kpss(df_d[col], label_d)
            self.results["first_differences"].extend([adf, pp, kpss_r])

            agree_stat = adf["stationary"] and pp["stationary"] and (kpss_r.get("stationary", False))
            flag = "Stationary ✓" if agree_stat else "Non-stat  ✗"

            kpss_p = kpss_r.get("p_value", "—")
            print(f"  {label_d:<18}"
                  f" {adf['test_statistic']:>10.4f} {adf['p_value']:>7.4f}"
                  f" {pp['test_statistic']:>10.4f} {pp['p_value']:>6.3f}"
                  f" {kpss_r.get('test_statistic', 0):>10.4f} {kpss_p:>7}"
                  f"  {flag}")

        return self.results

    # ────────────────────────────────────────────────────────────────
    # 5.  INTEGRATION-ORDER DETERMINATION
    # ────────────────────────────────────────────────────────────────
    def determine_integration_order(self) -> Dict[str, str]:
        """
        Rule:
            • I(0)  if ADF rejects at 5 % on levels
            • I(1)  if ADF does NOT reject on levels  BUT  rejects on Δy
            • I(2)? if neither level nor Δy rejects  (flag for review)

        Returns  { 'MPR': 'I(1)', 'Inflation': 'I(0)', … }
        """
        if not self.results:
            raise RuntimeError("Call run_all_tests() first.")

        print("\n" + "=" * 70)
        print("  INTEGRATION ORDER DETERMINATION")
        print("=" * 70)
        print(f"\n  {'Variable':<18} {'Level':<18} {'First Diff':<18} {'Order'}")
        print("  " + "─" * 60)

        for col in self.df.columns:
            # pull the ADF dicts for this variable
            lev = next((r for r in self.results["levels"]
                        if r.get("variable") == col and r["test"] == "ADF"), None)
            dif = next((r for r in self.results["first_differences"]
                        if r.get("variable") == f"Δ{col}" and r["test"] == "ADF"), None)

            if lev and lev["stationary"]:
                order, lev_txt, dif_txt = "I(0)", "Stationary", "—"
            elif dif and dif["stationary"]:
                order, lev_txt, dif_txt = "I(1)", "Unit root",   "Stationary"
            else:
                order, lev_txt, dif_txt = "I(2)?", "Unit root",  "Unit root"

            self.integration_orders[col] = order
            print(f"  {col:<18} {lev_txt:<18} {dif_txt:<18} {order}")

        # ── ARDL / VAR implications ───────────────────────────────
        i0 = [v for v, o in self.integration_orders.items() if o == "I(0)"]
        i1 = [v for v, o in self.integration_orders.items() if o == "I(1)"]

        print("\n  " + "─" * 60)
        print(f"  I(0) variables : {i0 if i0 else 'none'}")
        print(f"  I(1) variables : {i1 if i1 else 'none'}")
        print()

        if i0 and i1:
            print("  📌 ARDL  –  mixed I(0)/I(1)  →  bounds testing is the IDEAL approach.")
            print("  📌 VAR   –  cointegration test needed (Day 4) before deciding levels vs Δ.")
        elif not i0:
            print("  📌 ARDL  –  all I(1)  →  bounds testing still valid; Johansen also possible.")
            print("  📌 VAR   –  test for cointegration; if found, use VECM.")
        else:
            print("  📌 ARDL  –  all I(0)  →  simple OLS valid; ARDL can still capture dynamics.")

        return self.integration_orders

    # ────────────────────────────────────────────────────────────────
    # 6.  ACF / PACF PLOTS
    # ────────────────────────────────────────────────────────────────
    def plot_acf_pacf(self):
        """Two figure-grids: one for levels, one for first differences."""
        n = len(self.df.columns)

        for label, data in [("levels", self.df), ("first_differences", self.df.diff().dropna())]:
            fig, axes = plt.subplots(n, 2, figsize=(14, 3.8 * n))
            fig.suptitle(
                f"ACF & PACF — {'Levels' if label == 'levels' else 'First Differences'}",
                fontsize=15, fontweight="bold", y=1.02,
            )

            for i, col in enumerate(data.columns):
                col_name = col if label == "levels" else f"Δ{col}"
                c = _COLORS.get(col, "#333333")

                plot_acf(data[col].dropna(),  lags=24, ax=axes[i, 0],
                         alpha=0.05, color=c)
                axes[i, 0].set_title(f"{col_name} – ACF",  fontsize=11, fontweight="bold")
                axes[i, 0].set_xlabel("Lag (months)")
                axes[i, 0].grid(True, alpha=0.2)

                plot_pacf(data[col].dropna(), lags=24, ax=axes[i, 1],
                          alpha=0.05, color=c, method="ywm")
                axes[i, 1].set_title(f"{col_name} – PACF", fontsize=11, fontweight="bold")
                axes[i, 1].set_xlabel("Lag (months)")
                axes[i, 1].grid(True, alpha=0.2)

            fig.tight_layout()
            path = self.save_dir / f"acf_pacf_{label}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"  ✓ saved  {path}")
            plt.close(fig)

    # ────────────────────────────────────────────────────────────────
    # 7.  SAVE RESULTS TO CSV
    # ────────────────────────────────────────────────────────────────
    def save_results(self):
        """Persist every result dict as flat CSV rows."""
        if not self.results:
            print("  Nothing to save — run run_all_tests() first.")
            return

        for panel, key in [("levels", "unit_root_tests_levels.csv"),
                           ("first_differences", "unit_root_tests_first_differences.csv")]:
            rows = []
            for r in self.results[panel]:
                if "error" in r:
                    continue                            # skip failed KPSS gracefully
                row = {
                    "Variable":        r["variable"],
                    "Test":            r["test"],
                    "Test_Statistic":  r["test_statistic"],
                    "P_Value":         r.get("p_value", "approx"),
                    "Stationary":      r.get("stationary"),
                }
                # flatten critical values into their own columns
                cv = r.get("critical_values", {})
                if isinstance(cv, dict):
                    for k, v in cv.items():
                        row[f"CV_{k}"] = round(v, 3) if isinstance(v, (int, float)) else v
                rows.append(row)

            if rows:
                path = self.save_dir / key
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  ✓ saved  {path}")

        # integration-order summary
        if self.integration_orders:
            path = self.save_dir / "integration_orders.csv"
            pd.DataFrame(
                [{"Variable": v, "Order": o} for v, o in self.integration_orders.items()]
            ).to_csv(path, index=False)
            print(f"  ✓ saved  {path}")

    # ────────────────────────────────────────────────────────────────
    # 8.  FULL PIPELINE
    # ────────────────────────────────────────────────────────────────
    def run_full_analysis(self):
        """One call does everything: test → order → plot → save."""
        print("\n" + "=" * 70)
        print("  STATIONARITY ANALYSIS — FULL PIPELINE")
        print("=" * 70)

        self.run_all_tests()
        self.determine_integration_order()

        print("\n  [Plotting] ACF / PACF …")
        self.plot_acf_pacf()

        print("\n  [Saving]   CSV results …")
        self.save_results()

        print("\n" + "=" * 70)
        print("  ✓ STATIONARITY ANALYSIS COMPLETE")
        print("=" * 70)

        return self.results, self.integration_orders


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    tester = StationarityTester(df, save_dir="results/stationarity")
    tester.run_full_analysis()


if __name__ == "__main__":
    main()
