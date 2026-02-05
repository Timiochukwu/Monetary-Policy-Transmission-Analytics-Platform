"""
Cointegration Testing Module — Nigerian Monetary Policy Transmission

Two complementary tests:
    Engle-Granger  –  pairwise, two-step OLS  (statsmodels.tsa.stattools.coint)
    Johansen       –  multivariate, maximum-likelihood  (statsmodels VAR johansen)

Run order
---------
1.  Engle-Granger on every pair of variables (both directions)
2.  Johansen on the full 4-variable system
3.  Johansen restricted to the two *confirmed* I(1) variables
    (MPR, ExchangeRate) as a robustness check

The integration orders that feed into this module are read from the
CSV produced by Day 3's StationarityTester.

Day 4 deliverable.  No new dependencies beyond statsmodels.

References
----------
Engle & Granger (1987)  – Econometrica 55(2)
Johansen (1991)         – Econometrica 59(2)
MacKinnon, Haug & Michelon (1999) – critical values for Johansen
"""

import pandas   as pd
import numpy    as np
import matplotlib.pyplot as plt
from pathlib    import Path
from typing     import Dict, List

from statsmodels.tsa.stattools             import coint
from statsmodels.tsa.vector_ar.vecm        import coint_johansen
import statsmodels.api                     as sm


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class CointegrationTester:
    """
    Runs Engle-Granger and Johansen cointegration tests.

    Parameters
    ----------
    df            : pd.DataFrame   –  processed macro data (DatetimeIndex)
    int_orders    : dict | None    –  e.g. {'MPR':'I(1)', 'Inflation':'I(0)'}
                                      If None, read from integration_orders.csv.
    save_dir      : str | Path     –  output folder
    """

    def __init__(self,
                 df: pd.DataFrame,
                 int_orders: Dict[str, str] | None = None,
                 save_dir: str = "results/stationarity"):
        self.df       = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # load integration orders from CSV if not supplied
        if int_orders is None:
            io_path = self.save_dir / "integration_orders.csv"
            if io_path.exists():
                io_df = pd.read_csv(io_path)
                int_orders = dict(zip(io_df["Variable"], io_df["Order"]))
            else:
                int_orders = {}
        self.int_orders = int_orders

        self.eg_results      = []
        self.johansen_results = {}

    # ────────────────────────────────────────────────────────────────
    # ENGLE-GRANGER  (pairwise)
    # ────────────────────────────────────────────────────────────────
    def _run_eg(self, y_name: str, x_name: str) -> Dict:
        """
        Two-step Engle-Granger for one (y, x) pair.

        Step 1: OLS  y_t = α + β x_t + e_t
        Step 2: ADF on residuals  e_t.  If stationary → cointegrated.

        H₀ : no cointegration
        """
        y = self.df[y_name].dropna()
        x = self.df[x_name].dropna()
        common = y.index.intersection(x.index)
        y, x   = y.loc[common], x.loc[common]

        try:
            t_stat, p_val, crit = coint(y.values, x.values)
            return {
                "dependent":   y_name,
                "independent": x_name,
                "t_statistic": round(t_stat, 4),
                "p_value":     round(p_val,  4),
                "critical_values": {"1%": round(crit[0], 3),
                                    "5%": round(crit[1], 3),
                                    "10%": round(crit[2], 3)},
                "cointegrated": bool(p_val < 0.05),
            }
        except Exception as exc:
            return {"dependent": y_name, "independent": x_name,
                    "error": str(exc), "cointegrated": None}

    def run_all_pairwise_eg(self) -> List[Dict]:
        """All ordered pairs (both directions)."""
        cols = list(self.df.columns)
        self.eg_results = []
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i != j:
                    self.eg_results.append(self._run_eg(cols[i], cols[j]))
        return self.eg_results

    # ────────────────────────────────────────────────────────────────
    # JOHANSEN  (multivariate)
    # ────────────────────────────────────────────────────────────────
    def _run_johansen(self, variables: List[str],
                      det_order: int = 0, nlags: int = 12) -> Dict:
        """
        Johansen maximum-likelihood cointegration test.

        det_order  –  -1 : none
                       0 : constant
                       1 : constant + trend

        Returns a dict with eigenvalues, trace stats, max-eigenvalue
        stats, critical values, and the determined rank (number of
        cointegrating vectors at 5 %).
        """
        data = self.df[variables].dropna().values
        T, p = data.shape

        try:
            result = coint_johansen(data, det_order, nlags)

            # coint_johansen returns a results object with pre-computed stats
            eigen        = result.eig          # eigenvalues (descending)
            trace_stats  = result.lr1          # trace statistics
            maxeig_stats = result.lr2          # max-eigenvalue statistics
            cvt          = result.cvt          # trace CVs  : cols [10%, 5%, 1%]
            cvx          = result.max_eig_stat_crit_vals  # max-eig CVs: cols [10%, 5%, 1%]

            # ── determine rank via sequential trace test at 5 % ──
            # cvt[r, 1] = 5 % critical value for trace at H₀: rank = r
            rank_trace  = 0
            for r in range(p):
                if trace_stats[r] > cvt[r, 1]:
                    rank_trace += 1
                else:
                    break

            rank_maxeig = 0
            for r in range(p):
                if maxeig_stats[r] > cvx[r, 1]:
                    rank_maxeig += 1
                else:
                    break

            return {
                "variables":        variables,
                "T":                T,
                "det_order":        det_order,
                "nlags":            nlags,
                "eigenvalues":      eigen.tolist(),
                "trace_stats":      trace_stats.tolist(),
                "trace_cv_5pct":    cvt[:, 1].tolist(),
                "maxeig_stats":     maxeig_stats.tolist(),
                "maxeig_cv_5pct":   cvx[:, 1].tolist(),
                "rank_trace":       int(rank_trace),
                "rank_maxeig":      int(rank_maxeig),
            }
        except Exception as exc:
            return {"variables": variables, "error": str(exc)}

    def run_johansen_full(self) -> Dict:
        """
        Two Johansen runs:
            'all'       –  all 4 variables  (flag: some may be I(0))
            'i1_only'   –  confirmed I(1) variables only
        """
        all_vars = list(self.df.columns)
        i1_vars  = [v for v, o in self.int_orders.items() if o == "I(1)"]

        print("\n  [Johansen] full system …")
        self.johansen_results["all"] = self._run_johansen(all_vars)

        if len(i1_vars) >= 2:
            print(f"  [Johansen] I(1)-only  ({i1_vars}) …")
            self.johansen_results["i1_only"] = self._run_johansen(i1_vars)

        return self.johansen_results

    # ────────────────────────────────────────────────────────────────
    # RESIDUAL PLOTS  (EG diagnostic)
    # ────────────────────────────────────────────────────────────────
    def plot_eg_residuals(self):
        """
        For each cointegrated pair, plot the OLS residuals.
        Stationary residuals confirm the long-run equilibrium.
        """
        cointegrated_pairs = [
            r for r in self.eg_results
            if r.get("cointegrated") is True
        ]
        if not cointegrated_pairs:
            print("  [Residuals] no cointegrated pairs to plot.")
            return

        n = len(cointegrated_pairs)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n),
                                 squeeze=False)
        fig.suptitle("Engle-Granger Residuals (Equilibrium Errors)",
                     fontsize=15, fontweight="bold", y=1.01)

        for ax, pair in zip(axes.flat, cointegrated_pairs):
            y = self.df[pair["dependent"]]
            x = self.df[pair["independent"]]
            common = y.index.intersection(x.index)
            y, x   = y.loc[common], x.loc[common]

            # OLS to get residuals
            X   = sm.add_constant(x.values)
            ols = sm.OLS(y.values, X).fit()
            resid = pd.Series(ols.resid, index=common)

            ax.plot(resid.index, resid.values,
                    color="#2C3E50", linewidth=1.2)
            ax.axhline(0, color="red", linestyle="--", alpha=0.5)
            ax.fill_between(resid.index,
                            resid.mean() - resid.std(),
                            resid.mean() + resid.std(),
                            color="#2C3E50", alpha=0.08)
            ax.set_title(f"{pair['dependent']}  =  α + β·{pair['independent']}  +  e",
                         fontsize=11, fontweight="bold")
            ax.set_ylabel("Residual", fontsize=10)
            ax.grid(True, alpha=0.2)

        axes.flat[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()

        path = self.save_dir / "eg_residuals.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
        plt.close(fig)

    # ────────────────────────────────────────────────────────────────
    # PRINT + SAVE
    # ────────────────────────────────────────────────────────────────
    def _print_eg(self):
        print("\n" + "=" * 70)
        print("  ENGLE-GRANGER  –  pairwise cointegration (two-step)")
        print("=" * 70)
        print(f"  {'Dependent':<16} {'Independent':<16} {'t-stat':>8}"
              f" {'p-value':>8}  Result")
        print("  " + "─" * 65)
        for r in self.eg_results:
            if "error" in r:
                print(f"  {r['dependent']:<16} {r['independent']:<16}"
                      f"  ERROR: {r['error']}")
                continue
            flag = "Cointegrated ✓" if r["cointegrated"] else "Not coint.   ✗"
            print(f"  {r['dependent']:<16} {r['independent']:<16}"
                  f" {r['t_statistic']:>8.4f} {r['p_value']:>8.4f}  {flag}")

    def _print_johansen(self):
        for label, res in self.johansen_results.items():
            tag = "FULL SYSTEM" if label == "all" else "I(1)-ONLY ROBUSTNESS"
            print("\n" + "=" * 70)
            print(f"  JOHANSEN  –  {tag}")
            print(f"  Variables : {res.get('variables', '—')}")
            print("=" * 70)

            if "error" in res:
                print(f"  ERROR: {res['error']}")
                continue

            p = len(res["eigenvalues"])

            # trace table
            print(f"\n  {'r':<4} {'Trace stat':>12} {'CV 5 %':>10}  Decision")
            print("  " + "─" * 45)
            for r in range(p):
                ts  = res["trace_stats"][r]
                cv  = res["trace_cv_5pct"][r]
                dec = "Reject  H₀: rank=r" if ts > cv else "Fail to reject"
                print(f"  {r:<4} {ts:>12.3f} {cv:>10.3f}  {dec}")
            print(f"\n  → Trace test rank  :  {res['rank_trace']}")

            # max-eigenvalue table
            print(f"\n  {'r':<4} {'MaxEig stat':>12} {'CV 5 %':>10}  Decision")
            print("  " + "─" * 45)
            for r in range(p):
                ms  = res["maxeig_stats"][r]
                cv  = res["maxeig_cv_5pct"][r]
                dec = "Reject  H₀: rank=r" if ms > cv else "Fail to reject"
                print(f"  {r:<4} {ms:>12.3f} {cv:>10.3f}  {dec}")
            print(f"\n  → Max-eig rank     :  {res['rank_maxeig']}")

    def save_results(self):
        # EG
        if self.eg_results:
            rows = [r for r in self.eg_results if "error" not in r]
            if rows:
                path = self.save_dir / "engle_granger_pairwise.csv"
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  ✓ saved  {path}")

        # Johansen
        for label, res in self.johansen_results.items():
            if "error" in res:
                continue
            p    = len(res["eigenvalues"])
            rows = []
            for r in range(p):
                rows.append({
                    "r":                  r,
                    "Eigenvalue":         round(res["eigenvalues"][r], 6),
                    "Trace_Statistic":    round(res["trace_stats"][r], 3),
                    "Trace_CV_5pct":      round(res["trace_cv_5pct"][r], 3),
                    "Trace_Reject":       res["trace_stats"][r] > res["trace_cv_5pct"][r],
                    "MaxEig_Statistic":   round(res["maxeig_stats"][r], 3),
                    "MaxEig_CV_5pct":     round(res["maxeig_cv_5pct"][r], 3),
                    "MaxEig_Reject":      res["maxeig_stats"][r] > res["maxeig_cv_5pct"][r],
                })
            path = self.save_dir / f"johansen_{label}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  ✓ saved  {path}")

    # ────────────────────────────────────────────────────────────────
    # MASTER PIPELINE
    # ────────────────────────────────────────────────────────────────
    def run_full_analysis(self):
        """EG pairwise  →  Johansen  →  residual plots  →  save."""
        print("\n" + "=" * 70)
        print("  COINTEGRATION ANALYSIS  –  FULL PIPELINE")
        print("=" * 70)

        # 1 ── Engle-Granger
        print("\n  [1/4] Engle-Granger pairwise …")
        self.run_all_pairwise_eg()
        self._print_eg()

        # 2 ── Johansen
        print("\n  [2/4] Johansen …")
        self.run_johansen_full()
        self._print_johansen()

        # 3 ── residual plot
        print("\n  [3/4] EG residual plots …")
        self.plot_eg_residuals()

        # 4 ── save
        print("\n  [4/4] Saving CSVs …")
        self.save_results()

        # ── ARDL / VAR summary ──
        coint_count = sum(1 for r in self.eg_results
                          if r.get("cointegrated") is True)
        j_rank = self.johansen_results.get("all", {}).get("rank_trace", "?")

        print("\n" + "=" * 70)
        print("  SUMMARY  –  implications for Week 2 models")
        print("=" * 70)
        print(f"\n  Engle-Granger : {coint_count} cointegrated ordered pairs")
        print(f"  Johansen rank : {j_rank} (full system, trace test)")
        print()
        if coint_count > 0 or (isinstance(j_rank, int) and j_rank > 0):
            print("  📌 Long-run equilibrium relationships EXIST.")
            print("     → ARDL bounds test will confirm / quantify them.")
            print("     → VAR should use LEVELS (not differences).")
        else:
            print("  📌 No cointegration detected at 5 %.")
            print("     → ARDL bounds test may still find cointegration")
            print("       (it is more powerful than EG for small samples).")
            print("     → VAR should difference the I(1) variables.")
        print("\n" + "=" * 70)

        return self.eg_results, self.johansen_results


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    tester = CointegrationTester(df, save_dir="results/stationarity")
    tester.run_full_analysis()


if __name__ == "__main__":
    main()
