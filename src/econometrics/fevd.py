"""
Forecast Error Variance Decomposition — Nigerian Monetary Policy

Decomposes the forecast error variance of each variable into the
contribution from each structural shock (orthogonalized via Cholesky).

Key question for thesis:
  "What percentage of Inflation variance is explained by MPR shocks?"

Ordering (from Day 7):  MPR → ExchangeRate → M2 → Inflation

Horizon: 24 months

Day 9 deliverable.  Requires statsmodels and matplotlib.

References
----------
Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48.
Lutkepohl, H. (2005). New introduction to multiple time series analysis.
  Springer, Chapter 2.
"""

import sys
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class FEVDAnalyzer:
    """
    Compute and visualise forecast error variance decomposition.

    Parameters
    ----------
    var_results : statsmodels VARResults   –  fitted VAR model (from Day 7)
    ordering    : list[str]                –  Cholesky ordering
    save_dir    : str | Path               –  output directory
    """

    def __init__(self, var_results, ordering: list[str], save_dir: str = "results/fevd"):
        self.res      = var_results
        self.ordering = ordering
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.fevd_obj = None
        self.decomp   = None           # FEVD decomposition (np.ndarray)
        self.periods  = 0

    # ─────────────────────────────────────────────────────────────
    # COMPUTE
    # ─────────────────────────────────────────────────────────────
    def compute(self, periods: int = 24) -> np.ndarray:
        """
        Compute FEVD for 'periods' horizons.

        Returns
        -------
        np.ndarray  shape (neqs, periods, neqs)
            decomp[i, h, j] = fraction of variance of variable i at horizon h+1
                              explained by shock j
        """
        self.periods  = periods
        self.fevd_obj = self.res.fevd(periods)
        self.decomp   = self.fevd_obj.decomp   # shape: (K, T, K)
        return self.decomp

    # ─────────────────────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────────────────────
    def plot_variance_decomposition(self, response_var: str) -> None:
        """
        Stacked area plot: variance of 'response_var' decomposed by shock source.

        X-axis: horizon (1..periods)
        Y-axis: cumulative fraction (0..1)
        Each shock is a colored band.
        """
        response_idx = self.ordering.index(response_var)
        neqs         = len(self.ordering)

        # Extract the decomposition: all horizons, response var, all shocks
        # decomp[response_idx, :, :] shape: (periods, neqs)
        data = self.decomp[response_idx, :, :]  # (T, K)

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(1, self.periods + 1)

        # Color palette (match Day 2 convention)
        colors = {
            "MPR":          "#2E86AB",
            "ExchangeRate": "#F18F01",
            "M2":           "#6A994E",
            "Inflation":    "#A23B72",
        }
        color_list = [colors.get(v, "#CCCCCC") for v in self.ordering]

        # Stacked area
        ax.stackplot(x, data.T, labels=self.ordering, colors=color_list, alpha=0.8)

        ax.set_title(f"Variance Decomposition of {response_var}",
                     fontsize=15, fontweight="bold")
        ax.set_xlabel("Horizon (months)", fontsize=12)
        ax.set_ylabel("Fraction of forecast error variance", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.25, axis="y", linestyle=":")
        fig.tight_layout()

        path = self.save_dir / f"fevd_{response_var}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    def plot_all_decompositions(self) -> None:
        """Generate one stacked-area plot for each response variable."""
        for response_var in self.ordering:
            self.plot_variance_decomposition(response_var)

    # ─────────────────────────────────────────────────────────────
    # TABLE
    # ─────────────────────────────────────────────────────────────
    def _print_table(self, response_var: str) -> None:
        """Print FEVD table for one response variable."""
        response_idx = self.ordering.index(response_var)
        print(f"\n  Response variable: {response_var}")
        print("  " + "─" * 75)
        print(f"  {'Horizon':>8}  " + "  ".join(f"{v:>12}" for v in self.ordering)
              + "    Sum")
        print("  " + "─" * 75)

        # Print selected horizons
        horizons = [h for h in [0, 1, 2, 5, 11, 23] if h < self.periods]
        for h_idx in horizons:
            row = "  " + f"{h_idx + 1:>8}  "  # horizon is 1-indexed in table
            row_sum = 0.0
            for shock_idx in range(len(self.ordering)):
                val = self.decomp[response_idx, h_idx, shock_idx]
                row += f"{val:>12.4f}  "
                row_sum += val
            row += f"{row_sum:>6.4f}"
            print(row)
        print()

    def print_all_tables(self) -> None:
        """Print FEVD tables for all response variables."""
        print("\n" + "=" * 70)
        print("  FORECAST ERROR VARIANCE DECOMPOSITION")
        print("=" * 70)
        for response_var in self.ordering:
            self._print_table(response_var)

        # Highlight key result
        print("=" * 70)
        print("  KEY RESULT — MPR → Inflation transmission")
        print("=" * 70)
        inf_idx = self.ordering.index("Inflation")
        mpr_idx = self.ordering.index("MPR")

        # Print the share of Inflation variance explained by MPR at key horizons
        print(f"\n  {'Horizon':>10}  {'MPR → Inflation':>18}  (% of variance)")
        print("  " + "─" * 50)
        for h in [0, 2, 5, 11, 23]:
            if h < self.periods:
                share = self.decomp[inf_idx, h, mpr_idx]
                print(f"  {h + 1:>10}  {share:>18.2%}")
        print()

    # ─────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────
    def save_tables(self) -> None:
        """Save one CSV per response variable."""
        for response_var in self.ordering:
            response_idx = self.ordering.index(response_var)

            # Build a DataFrame: rows = horizons, cols = shock sources
            df = pd.DataFrame(
                self.decomp[response_idx, :, :],
                columns=self.ordering
            )
            df.insert(0, "horizon", np.arange(1, self.periods + 1))

            path = self.save_dir / f"fevd_{response_var}.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ saved  {path}")

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self, periods: int = 24) -> None:
        print("\n" + "=" * 70)
        print("  FORECAST ERROR VARIANCE DECOMPOSITION  –  FULL PIPELINE")
        print(f"  Ordering (Cholesky): {' → '.join(self.ordering)}")
        print("=" * 70)

        print(f"\n  [1/4] Computing FEVD ({periods} months) …")
        self.compute(periods)

        print("  [2/4] Plotting …")
        self.plot_all_decompositions()

        print("\n  [3/4] Printing tables …")
        self.print_all_tables()

        print("  [4/4] Saving CSVs …")
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

    # Re-fit the VAR
    var = VARModel(df)
    var.select_lags(max_lags=12)
    var.fit()

    # FEVD analysis
    fevd_analyzer = FEVDAnalyzer(var.res, var.ordering)
    fevd_analyzer.run_full_analysis(periods=24)


if __name__ == "__main__":
    main()
