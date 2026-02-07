"""
Impulse Response Functions — Nigerian Monetary Policy Transmission

Computes orthogonalized (Cholesky) impulse responses from the fitted
VAR model.  Focus: response of all variables to a +1 SD shock in MPR.

Ordering (from Day 7):  MPR → ExchangeRate → M2 → Inflation

Horizon: 24 months (captures medium-run transmission)

Day 8 deliverable.  Requires statsmodels and matplotlib.

References
----------
Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48.
Pesaran, H. H. & Shin, Y. (1998). Generalized impulse response analysis
  in linear multivariate models. Economics Letters, 58(1), 17–28.
"""

import sys
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class IRFAnalyzer:
    """
    Compute and visualise orthogonalized impulse responses.

    Parameters
    ----------
    var_results : statsmodels VARResults   –  fitted VAR model (from Day 7)
    ordering    : list[str]                –  Cholesky ordering
    save_dir    : str | Path               –  output directory
    """

    def __init__(self, var_results, ordering: list[str], save_dir: str = "results/irf"):
        self.res      = var_results
        self.ordering = ordering
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.irf_obj  = None
        self.irfs     = None           # orthogonalized IRFs (np.ndarray)
        self.periods  = 0

    # ─────────────────────────────────────────────────────────────
    # COMPUTE
    # ─────────────────────────────────────────────────────────────
    def compute(self, periods: int = 24) -> np.ndarray:
        """
        Compute orthogonalized IRFs for 'periods' horizons.

        Returns
        -------
        np.ndarray  shape (periods+1, neqs, neqs)
            irfs[t, i, j] = response of variable i at time t to shock j
        """
        self.periods  = periods
        self.irf_obj  = self.res.irf(periods)
        self.irfs     = self.irf_obj.orth_irfs   # shape: (T+1, K, K)
        return self.irfs

    # ─────────────────────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────────────────────
    def plot_responses_to_shock(self, shock_var: str) -> None:
        """
        4-panel plot: responses of all variables to a shock in 'shock_var'.

        Each subplot shows one response variable over time.
        """
        shock_idx = self.ordering.index(shock_var)
        neqs      = len(self.ordering)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Impulse Responses to +1 SD Shock in {shock_var}",
                     fontsize=16, fontweight="bold", y=0.995)

        for i, (ax, response_var) in enumerate(zip(axes.flat, self.ordering)):
            # Extract the IRF: all time steps, response var i, shock var shock_idx
            y = self.irfs[:, i, shock_idx]
            x = np.arange(self.periods + 1)

            ax.plot(x, y, color="#2E86AB", linewidth=2.5, label="Response")
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.fill_between(x, 0, y, where=(y > 0), color="#2E86AB", alpha=0.15,
                            interpolate=True)
            ax.fill_between(x, 0, y, where=(y < 0), color="#A23B72", alpha=0.15,
                            interpolate=True)

            ax.set_title(f"Response: {response_var}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Months", fontsize=11)
            ax.set_ylabel("Response (pp or units)", fontsize=11)
            ax.grid(True, alpha=0.25, linestyle=":")
            ax.legend(loc="best", fontsize=10)

        fig.tight_layout()
        path = self.save_dir / f"irf_{shock_var}_shock.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    def plot_all_shocks(self) -> None:
        """Generate one 4-panel plot for each shock variable (4 plots total)."""
        for shock_var in self.ordering:
            self.plot_responses_to_shock(shock_var)

    # ─────────────────────────────────────────────────────────────
    # TABLE
    # ─────────────────────────────────────────────────────────────
    def _print_table(self, shock_var: str) -> None:
        """Print IRF table for one shock variable."""
        shock_idx = self.ordering.index(shock_var)
        print(f"\n  Shock: {shock_var}  (responses at selected horizons)")
        print("  " + "─" * 70)
        print(f"  {'Horizon':>8}  " + "  ".join(f"{v:>12}" for v in self.ordering))
        print("  " + "─" * 70)

        # Print selected horizons: 0, 1, 3, 6, 12, 24
        horizons = [h for h in [0, 1, 3, 6, 12, 24] if h <= self.periods]
        for h in horizons:
            row = "  " + f"{h:>8}  "
            for i, _ in enumerate(self.ordering):
                val = self.irfs[h, i, shock_idx]
                row += f"{val:>12.4f}  "
            print(row)
        print()

    def print_all_tables(self) -> None:
        """Print IRF tables for all shock variables."""
        print("\n" + "=" * 70)
        print("  IMPULSE RESPONSE FUNCTIONS  (Orthogonalized / Cholesky)")
        print("=" * 70)
        for shock_var in self.ordering:
            self._print_table(shock_var)

    # ─────────────────────────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────────────────────────
    def save_tables(self) -> None:
        """Save one CSV per shock variable."""
        for shock_var in self.ordering:
            shock_idx = self.ordering.index(shock_var)

            # Build a DataFrame: rows = horizons, cols = response variables
            df = pd.DataFrame(
                self.irfs[:, :, shock_idx],
                columns=self.ordering
            )
            df.insert(0, "horizon", np.arange(self.periods + 1))

            path = self.save_dir / f"irf_{shock_var}_shock.csv"
            df.to_csv(path, index=False)
            print(f"  ✓ saved  {path}")

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_analysis(self, periods: int = 24) -> None:
        print("\n" + "=" * 70)
        print("  IMPULSE RESPONSE FUNCTIONS  –  FULL PIPELINE")
        print(f"  Ordering (Cholesky): {' → '.join(self.ordering)}")
        print("=" * 70)

        print(f"\n  [1/4] Computing IRFs ({periods} months) …")
        self.compute(periods)

        print("  [2/4] Plotting …")
        self.plot_all_shocks()

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

    # Re-fit the VAR (or load from saved results — here we refit for simplicity)
    var = VARModel(df)
    var.select_lags(max_lags=12)
    var.fit()

    # IRF analysis
    irf_analyzer = IRFAnalyzer(var.res, var.ordering)
    irf_analyzer.run_full_analysis(periods=24)


if __name__ == "__main__":
    main()
