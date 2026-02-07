"""
Policy Shock Simulation — Nigerian Monetary Policy Transmission

Scales the orthogonalized IRFs to a realistic policy scenario:
    +100 basis points (1 percentage point) MPR hike

Produces:
  1. Scaled IRF tables (all variables, 24-month horizon)
  2. Comparison plot (1 SD shock vs 100 bps shock)
  3. Policy brief (Markdown) for CBN decision-makers

Day 10 deliverable.  Requires the fitted VAR from Day 7 and IRFs from Day 8.

References
----------
Central Bank of Nigeria (2023). Monetary Policy Committee Communiqué.
Bernanke, B. S. & Blinder, A. S. (1992). The Federal Funds Rate and the
  Channels of Monetary Transmission. American Economic Review, 82(4), 901–921.
"""

import sys
import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path
from datetime   import datetime


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class PolicySimulator:
    """
    Simulate a +100 bps MPR shock and generate a policy brief.

    Parameters
    ----------
    var_results : statsmodels VARResults   –  fitted VAR (from Day 7)
    irf_obj     : statsmodels IRAnalysis   –  IRF object (from Day 8)
    ordering    : list[str]                –  Cholesky ordering
    save_dir    : str | Path               –  output directory
    """

    def __init__(self, var_results, irf_obj, ordering: list[str],
                 save_dir: str = "results/policy_simulation"):
        self.res      = var_results
        self.irf_obj  = irf_obj
        self.ordering = ordering
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.shock_sd       = None    # MPR shock standard deviation
        self.scaling_factor = None    # 100 bps / shock_sd
        self.scaled_irfs    = None    # scaled IRFs (np.ndarray)

    # ─────────────────────────────────────────────────────────────
    # COMPUTE
    # ─────────────────────────────────────────────────────────────
    def compute_scaling_factor(self, policy_shock_bps: int = 100) -> float:
        """
        Compute the scaling factor to convert a 1 SD shock to 'policy_shock_bps'.

        MPR is in percentage points.  A 1 SD shock is `shock_sd` pp.
        To get a `policy_shock_bps` bp shock, multiply IRFs by:
            (policy_shock_bps / 100) / shock_sd
        """
        # Cholesky decomposition of VAR residuals
        # The diagonal of P (lower-triangular) gives the shock SDs
        cov_u = self.res.sigma_u               # covariance of reduced-form errors
        P     = np.linalg.cholesky(cov_u)      # P P' = cov_u

        mpr_idx       = self.ordering.index("MPR")
        self.shock_sd = P[mpr_idx, mpr_idx]    # SD of orthogonalized MPR shock

        self.scaling_factor = (policy_shock_bps / 100.0) / self.shock_sd
        return self.scaling_factor

    def scale_irfs(self) -> np.ndarray:
        """Scale the IRFs by the policy shock magnitude."""
        # Extract IRFs for MPR shock only
        mpr_idx = self.ordering.index("MPR")
        # irf_obj.orth_irfs shape: (T+1, K, K)
        # We want all responses to MPR shock: [:, :, mpr_idx]
        irfs_mpr = self.irf_obj.orth_irfs[:, :, mpr_idx]  # shape: (T+1, K)

        self.scaled_irfs = irfs_mpr * self.scaling_factor
        return self.scaled_irfs

    # ─────────────────────────────────────────────────────────────
    # PLOT
    # ─────────────────────────────────────────────────────────────
    def plot_comparison(self, policy_shock_bps: int = 100) -> None:
        """
        4-panel plot: 1 SD shock (blue) vs scaled policy shock (orange).
        """
        mpr_idx   = self.ordering.index("MPR")
        irfs_1sd  = self.irf_obj.orth_irfs[:, :, mpr_idx]  # (T+1, K)
        periods   = len(irfs_1sd) - 1

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Policy Simulation: +{policy_shock_bps} bps MPR Shock vs +1 SD Shock",
                     fontsize=16, fontweight="bold", y=0.995)

        x = np.arange(periods + 1)

        for i, (ax, var) in enumerate(zip(axes.flat, self.ordering)):
            y_1sd    = irfs_1sd[:, i]
            y_scaled = self.scaled_irfs[:, i]

            ax.plot(x, y_1sd, color="#2E86AB", linewidth=2, label=f"+1 SD shock (±{self.shock_sd:.2f} pp)")
            ax.plot(x, y_scaled, color="#F18F01", linewidth=2.5, linestyle="--",
                    label=f"+{policy_shock_bps} bps shock")
            ax.axhline(0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)

            ax.set_title(f"Response: {var}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Months", fontsize=11)
            ax.set_ylabel("Response (pp or units)", fontsize=11)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.25, linestyle=":")

        fig.tight_layout()
        path = self.save_dir / f"policy_shock_{policy_shock_bps}bps_comparison.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────
    # TABLE
    # ─────────────────────────────────────────────────────────────
    def save_table(self, policy_shock_bps: int = 100) -> None:
        """Save scaled IRF table as CSV."""
        df = pd.DataFrame(self.scaled_irfs, columns=self.ordering)
        df.insert(0, "horizon", np.arange(len(df)))

        path = self.save_dir / f"policy_shock_{policy_shock_bps}bps_responses.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ saved  {path}")

    def print_table(self, policy_shock_bps: int = 100) -> None:
        """Print key horizons from the scaled IRF."""
        print("\n" + "=" * 70)
        print(f"  POLICY SIMULATION: +{policy_shock_bps} BPS MPR SHOCK")
        print("=" * 70)
        print(f"\n  Shock magnitude : +{policy_shock_bps} basis points (1 pp)")
        print(f"  Shock SD        : {self.shock_sd:.4f} pp")
        print(f"  Scaling factor  : {self.scaling_factor:.4f}\n")

        print(f"  {'Horizon':>8}  " + "  ".join(f"{v:>12}" for v in self.ordering))
        print("  " + "─" * 70)

        horizons = [0, 1, 3, 6, 12, 24]
        for h in horizons:
            if h < len(self.scaled_irfs):
                row = f"  {h:>8}  "
                for i in range(len(self.ordering)):
                    val = self.scaled_irfs[h, i]
                    row += f"{val:>12.4f}  "
                print(row)
        print()

    # ─────────────────────────────────────────────────────────────
    # POLICY BRIEF
    # ─────────────────────────────────────────────────────────────
    def generate_policy_brief(self, policy_shock_bps: int = 100) -> str:
        """Generate a Markdown policy brief for CBN decision-makers."""
        inf_idx = self.ordering.index("Inflation")
        er_idx  = self.ordering.index("ExchangeRate")

        # Key numbers
        inf_impact   = self.scaled_irfs[0, inf_idx]
        inf_3mo      = self.scaled_irfs[3, inf_idx]
        inf_6mo      = self.scaled_irfs[6, inf_idx]
        inf_12mo     = self.scaled_irfs[12, inf_idx]
        inf_peak     = self.scaled_irfs[:, inf_idx].min()   # peak (most negative)
        inf_peak_idx = int(self.scaled_irfs[:, inf_idx].argmin())

        er_12mo = self.scaled_irfs[12, er_idx]

        brief = f"""# Policy Brief — Monetary Policy Rate Transmission Simulation

**Central Bank of Nigeria**
**Economic Policy Department**
**Date:** {datetime.now().strftime("%B %Y")}

---

## Executive Summary

This brief quantifies the expected impact of a **+100 basis point (1 percentage point) increase in the Monetary Policy Rate (MPR)** on key macroeconomic variables over a 24-month horizon.

The analysis is based on a Vector Autoregression (VAR) model estimated on monthly data from January 2010 to January 2025 (181 observations).

---

## Key Findings

### Inflation Response

| Horizon | Cumulative Effect on Inflation (pp) |
|---------|-------------------------------------|
| **Impact (month 0)** | {inf_impact:+.2f} |
| **3 months** | {inf_3mo:+.2f} |
| **6 months** | {inf_6mo:+.2f} |
| **12 months** | {inf_12mo:+.2f} |
| **Peak effect** | {inf_peak:+.2f} (month {inf_peak_idx}) |

**Interpretation:**
- A 100 bps MPR hike reduces headline inflation by approximately **{abs(inf_12mo):.2f} percentage points** after 12 months.
- The peak effect occurs at month {inf_peak_idx}, with a cumulative reduction of **{abs(inf_peak):.2f} pp**.
- The delayed response reflects **price stickiness** in Nigeria's consumer markets and the time required for credit-channel transmission.

### Exchange Rate Response

- At 12 months, the naira is expected to {'appreciate' if er_12mo < 0 else 'depreciate'} by approximately **{abs(er_12mo):.1f} units** per dollar.
- This reflects the **uncovered interest parity (UIP) channel** and capital flow dynamics.

### Policy Implications

1. **Timing**: Maximum inflation-reduction impact occurs at month {inf_peak_idx}. The MPC should anticipate a **{inf_peak_idx}-month policy lag**.

2. **Magnitude**: A 100 bps hike is expected to reduce inflation by ~{abs(inf_12mo):.1f} pp at the 12-month horizon. To achieve the CBN's inflation target band (6–9 %), multiple rate adjustments may be required if current inflation exceeds 15 %.

3. **Trade-offs**: The exchange-rate response suggests {'potential FX-market volatility' if abs(er_12mo) > 5 else 'limited FX-market disruption'}. The MPC should coordinate with the FX Management Department.

4. **Limitations**: The model assumes:
   - No structural breaks (2016 devaluation, 2023 FX unification effects are smoothed)
   - Linear transmission (nonlinear effects in high-inflation regimes are not captured)
   - Ceteris paribus (no fiscal or external shocks)

---

## Methodology

- **Model**: VAR(11) with Cholesky identification
- **Ordering**: MPR → Exchange Rate → Money Supply (M2) → Inflation
- **Sample**: 2010-01 to 2025-01 (181 monthly observations)
- **Shock magnitude**: +100 bps (scaled from orthogonalized shock SD = {self.shock_sd:.4f} pp)

---

## Recommendation

Based on this simulation:

- A **100 bps MPR hike** is expected to contribute a **−{abs(inf_12mo):.2f} pp** inflation reduction over 12 months.
- If current inflation is {15 + abs(inf_12mo):.1f} % and the target is 7.5 %, the MPC should consider **multiple 100 bps hikes** over consecutive meetings, subject to real-sector impact assessments.

---

## Contact

For technical details, see:
- Full VAR estimation results: `results/var/`
- Impulse response functions: `results/irf/`
- FEVD analysis: `results/fevd/`

Generated by: `src/econometrics/policy_simulation.py`
Model timestamp: {datetime.now().isoformat()}

---
"""
        return brief

    def save_policy_brief(self, policy_shock_bps: int = 100) -> None:
        """Save the policy brief as Markdown."""
        brief = self.generate_policy_brief(policy_shock_bps)
        path  = self.save_dir / f"policy_brief_{policy_shock_bps}bps.md"
        path.write_text(brief)
        print(f"  ✓ saved  {path}")

    # ─────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ─────────────────────────────────────────────────────────────
    def run_full_simulation(self, policy_shock_bps: int = 100) -> None:
        print("\n" + "=" * 70)
        print(f"  POLICY SIMULATION  –  +{policy_shock_bps} BPS MPR SHOCK")
        print("=" * 70)

        print(f"\n  [1/5] Computing scaling factor …")
        self.compute_scaling_factor(policy_shock_bps)
        print(f"        MPR shock SD       : {self.shock_sd:.4f} pp")
        print(f"        Scaling factor     : {self.scaling_factor:.4f}")

        print(f"  [2/5] Scaling IRFs …")
        self.scale_irfs()

        print(f"  [3/5] Plotting comparison …")
        self.plot_comparison(policy_shock_bps)

        print(f"  [4/5] Printing table …")
        self.print_table(policy_shock_bps)

        print(f"  [5/5] Saving outputs …")
        self.save_table(policy_shock_bps)
        self.save_policy_brief(policy_shock_bps)


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

    # Compute IRFs
    irf_obj = var.res.irf(24)

    # Policy simulation
    sim = PolicySimulator(var.res, irf_obj, var.ordering)
    sim.run_full_simulation(policy_shock_bps=100)


if __name__ == "__main__":
    main()
