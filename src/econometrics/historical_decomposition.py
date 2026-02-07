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
