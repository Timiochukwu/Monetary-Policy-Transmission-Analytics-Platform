"""
VAR Model — Nigerian Monetary Policy Transmission

4-variable Vector Autoregression with Cholesky identification.

Ordering  (Pesaran & Shin, 1998; Christiano et al., 1999):
    MPR  →  ExchangeRate  →  M2  →  Inflation

Steps
-----
1.  Lag order selection  (AIC, up to 12)
2.  Estimate VAR(p)
3.  Granger causality  (all 12 ordered pairs, F-test)
4.  Print coefficient tables + significance

Day 7 deliverable.  Requires statsmodels.

References
----------
Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48.
Christiano, L. J., Eichenbaum, M. & Evans, C. L. (1999).
  Monetary policy shocks: What have we learned and to what end?
  Handbook of Macroeconomics, 1, 65–148.
"""

import sys
import pandas   as pd
import numpy    as np
from pathlib    import Path
from typing     import List

from statsmodels.tsa.api import VAR


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class VARModel:
    """
    4-variable VAR with Cholesky identification.

    Parameters
    ----------
    df       : pd.DataFrame   –  macro data (columns reordered to Cholesky)
    ordering : list[str]      –  Cholesky ordering
    save_dir : str | Path     –  output directory
    """

    def __init__(self,
                 df: pd.DataFrame,
                 ordering: List[str] | None = None,
                 save_dir: str = "results/var"):
        self.ordering = ordering or ['MPR', 'ExchangeRate', 'M2', 'Inflation']
        self.df       = df[self.ordering]          # enforce column order
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model      = None
        self.res        = None
        self.opt_lag    : int            = 0
        self.lag_table  : pd.DataFrame   = pd.DataFrame()

    # ─────────────────────────────────────────────────────────────
    # LAG SELECTION
    # ─────────────────────────────────────────────────────────────
    def select_lags(self, max_lags: int = 12) -> int:
        """
        Fit VAR(1) … VAR(max_lags), record AIC/BIC/HQIC/FPE,
        return the AIC-optimal lag.
        """
        self.model = VAR(self.df)
        records    = []

        for lag in range(1, max_lags + 1):
            try:
                r = self.model.fit(lag)
                records.append({
                    "lag":  lag,
                    "AIC":  round(r.aic,  4),
                    "BIC":  round(r.bic,  4),
                    "HQIC": round(r.hqic, 4),
                    "FPE":  round(r.fpe,  4),
                })
            except Exception:
                break          # not enough observations for this lag

        self.lag_table = pd.DataFrame(records)
        self.opt_lag   = int(self.lag_table.loc[
            self.lag_table["AIC"].idxmin(), "lag"])

        print(f"  [Lag selection] AIC-optimal lag  :  {self.opt_lag}")
        print(f"  [Lag selection] (BIC = {self.lag_table.loc[self.lag_table['BIC'].idxmin(), 'lag']}"
              f", HQIC = {self.lag_table.loc[self.lag_table['HQIC'].idxmin(), 'lag']})")
        return self.opt_lag

    # ─────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────
    def fit(self, lags: int | None = None) -> None:
        """Estimate VAR(p).  Defaults to AIC-optimal lag."""
        if lags is None:
            lags = self.opt_lag
        if self.model is None:
            self.model = VAR(self.df)
        self.res = self.model.fit(lags)
        print(f"  [VAR] fitted  –  VAR({lags}), {len(self.df)} obs")

    # ─────────────────────────────────────────────────────────────
    # GRANGER CAUSALITY  (all ordered pairs)
    # ─────────────────────────────────────────────────────────────
    def granger_causality(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        F-test Granger causality for every ordered pair.

        H₀ : ``causing`` does NOT Granger-cause ``caused``
        """
        rows = []
        for caused in self.ordering:
            for causing in self.ordering:
                if caused == causing:
                    continue
                try:
                    gc = self.res.test_causality(
                        caused, causing=causing, kind='f', signif=alpha)
                    rows.append({
                        "causing":     causing,
                        "caused":      caused,
                        "F_statistic": round(float(gc.test_statistic), 4),
                        "p_value":     round(float(gc.pvalue),         4),
                        "significant": bool(gc.pvalue < alpha),
                    })
                except Exception as exc:
                    rows.append({
                        "causing": causing,
                        "caused":  caused,
                        "error":   str(exc),
                    })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────
    # PRINT
    # ─────────────────────────────────────────────────────────────
    def _print_lag_table(self) -> None:
        print("\n" + "=" * 70)
        print("  VAR  –  LAG SELECTION")
        print("=" * 70)
        print(f"\n  {'Lag':>4} {'AIC':>10} {'BIC':>10} {'HQIC':>10} {'FPE':>10}")
        print("  " + "─" * 48)
        for _, row in self.lag_table.iterrows():
            aic_flag = " ◄" if int(row["lag"]) == self.opt_lag else "  "
            print(f"  {int(row['lag']):>4} {row['AIC']:>10.3f} {row['BIC']:>10.3f}"
                  f" {row['HQIC']:>10.3f} {row['FPE']:>10.3f}{aic_flag}")
        print(f"  ◄ = AIC-optimal\n")

    def _print_coefficients(self) -> None:
        """Per-equation coefficient tables with significance stars."""
        print("=" * 70)
        print("  VAR COEFFICIENT TABLES")
        print("=" * 70)

        # VARResults stores params as a DataFrame: rows = lagged vars, cols = equations
        # tvalues / pvalues may or may not exist; compute manually if absent
        params = self.res.params           # shape (p*k + intercept, k)

        # try to get p-values; fall back to t-dist computation
        try:
            pvalues = self.res.pvalues
        except AttributeError:
            from scipy.stats import t as t_dist
            tvals   = self.res.tvalues
            n       = len(self.df)
            k       = len(self.ordering)
            p       = self.opt_lag
            df_resid = n - p * k - 1       # intercept counts as 1
            pvalues = 2 * (1 - t_dist.cdf(np.abs(tvals), df_resid))
            pvalues = pd.DataFrame(pvalues, index=tvals.index, columns=tvals.columns)

        tvalues = self.res.tvalues
        stderr  = self.res.stderr

        for eq in self.ordering:
            print(f"\n  Equation: {eq}")
            print("  " + "─" * 65)
            print(f"  {'Variable':<22} {'Coeff':>10} {'Std Err':>10}"
                  f" {'t-stat':>8} {'p-val':>8}  Sig")
            print("  " + "─" * 65)
            for var_lag in params.index:
                c  = params.loc[var_lag, eq]
                se = stderr.loc[var_lag, eq]
                t  = tvalues.loc[var_lag, eq]
                p  = pvalues.loc[var_lag, eq]
                stars = ""
                if p < 0.01:   stars = "***"
                elif p < 0.05: stars = "** "
                elif p < 0.10: stars = "*  "
                print(f"  {var_lag:<22} {c:>10.4f} {se:>10.4f}"
                      f" {t:>8.3f} {p:>8.4f}  {stars}")

        print("\n  Significance codes: *** p<0.01  ** p<0.05  * p<0.10\n")

    def _print_granger(self, gc_df: pd.DataFrame) -> None:
        print("=" * 70)
        print("  GRANGER CAUSALITY  (F-test, α = 0.05)")
        print("=" * 70)
        print(f"\n  {'Causing  →':<18} {'→  Caused':<18} {'F-stat':>8} {'p-val':>8}  Result")
        print("  " + "─" * 65)
        for _, row in gc_df.iterrows():
            if "error" in row and pd.notna(row.get("error")):
                print(f"  {row['causing']:<18} {row['caused']:<18}  ERROR: {row['error']}")
                continue
            flag = "Causes ✓" if row["significant"] else "No cause ✗"
            print(f"  {row['causing']:<18} {row['caused']:<18}"
                  f" {row['F_statistic']:>8.3f} {row['p_value']:>8.4f}  {flag}")

        # highlight significant links
        sig = gc_df[gc_df["significant"] == True] if "significant" in gc_df.columns else gc_df.iloc[0:0]  # noqa: E712
        print()
        if len(sig) > 0:
            print("  📌 Significant Granger-causal links (5 %):")
            for _, row in sig.iterrows():
                print(f"     {row['causing']}  →  {row['caused']}")
        else:
            print("  📌 No Granger-causal links detected at 5 %.")
        print()

    # ─────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────
    def _save_results(self, gc_df: pd.DataFrame) -> None:
        # lag selection
        self.lag_table.to_csv(self.save_dir / "var_lag_selection.csv", index=False)
        print(f"  ✓ saved  {self.save_dir / 'var_lag_selection.csv'}")

        # coefficient matrix
        self.res.params.to_csv(self.save_dir / "var_coefficients.csv")
        print(f"  ✓ saved  {self.save_dir / 'var_coefficients.csv'}")

        # Granger causality
        gc_df.to_csv(self.save_dir / "var_granger_causality.csv", index=False)
        print(f"  ✓ saved  {self.save_dir / 'var_granger_causality.csv'}")

        # summary
        summary = pd.DataFrame([{
            "n_obs":         len(self.df),
            "optimal_lag":   self.opt_lag,
            "n_variables":   len(self.ordering),
            "ordering":      " → ".join(self.ordering),
        }])
        summary.to_csv(self.save_dir / "var_summary.csv", index=False)
        print(f"  ✓ saved  {self.save_dir / 'var_summary.csv'}")

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self, max_lags: int = 12) -> None:
        print("\n" + "=" * 70)
        print("  VAR  –  FULL PIPELINE")
        print(f"  Ordering: {' → '.join(self.ordering)}")
        print("=" * 70)

        print("\n  [1/4] Lag selection …")
        self.select_lags(max_lags)
        self._print_lag_table()

        print("  [2/4] Fitting VAR …")
        self.fit()

        print("\n  [3/4] Coefficient tables …")
        self._print_coefficients()

        print("  [4/4] Granger causality & saving …")
        gc_df = self.granger_causality()
        self._print_granger(gc_df)
        self._save_results(gc_df)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    var = VARModel(df)
    var.run_full_analysis()


if __name__ == "__main__":
    main()
