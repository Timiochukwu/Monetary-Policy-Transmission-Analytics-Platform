"""
Week 1 Consolidated Report — Nigerian Monetary Policy Transmission

Reads every CSV produced during Days 3 and 4, assembles a single
human-readable summary, and persists it as ``results/week1_summary.md``.

Intended to be run after both stationarity_tests.py and
cointegration_tests.py have completed successfully.

Day 5 deliverable.  No dependencies beyond pandas.
"""

import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _header(title: str) -> str:
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}\n"


def _sub(title: str) -> str:
    return f"\n  {'─' * 60}\n  {title}\n  {'─' * 60}\n"


# ─────────────────────────────────────────────────────────────────────
# CLASS
# ─────────────────────────────────────────────────────────────────────

class Week1Reporter:
    """
    Assembles and prints the full Week 1 econometric summary.

    Parameters
    ----------
    results_dir : str | Path
        Root results directory (default ``results``).
        The script expects the sub-tree::

            results/
              stationarity/
                unit_root_tests_levels.csv
                unit_root_tests_first_differences.csv
                integration_orders.csv
                engle_granger_pairwise.csv
                johansen_all.csv
                johansen_i1_only.csv   (may be absent)
    """

    def __init__(self, results_dir: str = "results"):
        self.root = Path(results_dir)
        self.stat_dir = self.root / "stationarity"

    # ── loaders ────────────────────────────────────────────────────
    def _load(self, name: str) -> pd.DataFrame | None:
        path = self.stat_dir / name
        if path.exists():
            return pd.read_csv(path)
        print(f"  ⚠ {path} not found — skipping.")
        return None

    # ── section builders ──────────────────────────────────────────
    def _section_unit_roots(self) -> str:
        """Panel A: levels; Panel B: first differences."""
        lines = [_header("1 / UNIT-ROOT TESTS")]

        for panel, fname in [("A — Levels",      "unit_root_tests_levels.csv"),
                             ("B — Δ (first differences)",
                              "unit_root_tests_first_differences.csv")]:
            df = self._load(fname)
            if df is None:
                lines.append(f"  [{panel}]  data unavailable.\n")
                continue

            lines.append(_sub(f"Panel {panel}"))
            lines.append(f"  {'Variable':<18} {'Test':<6} {'Statistic':>10}"
                         f" {'p-value':>9}  {'Stationary'}")
            lines.append("  " + "─" * 58)
            for _, row in df.iterrows():
                stat_lbl = "Yes" if row.get("Stationary", False) else "No "
                lines.append(
                    f"  {row['Variable']:<18} {row['Test']:<6}"
                    f" {row['Test_Statistic']:>10.4f}"
                    f" {row['P_Value']:>9.4f}  {stat_lbl}"
                )
            lines.append("")

        return "\n".join(lines)

    def _section_integration_orders(self) -> str:
        """Consensus integration orders and ARDL eligibility note."""
        df = self._load("integration_orders.csv")
        lines = [_header("2 / INTEGRATION ORDERS  (consensus)")]

        if df is None:
            lines.append("  Data unavailable.\n")
            return "\n".join(lines)

        i0 = df.loc[df["Order"] == "I(0)", "Variable"].tolist()
        i1 = df.loc[df["Order"] == "I(1)", "Variable"].tolist()

        lines.append(f"  {'Variable':<18} {'Order'}")
        lines.append("  " + "─" * 30)
        for _, row in df.iterrows():
            lines.append(f"  {row['Variable']:<18} {row['Order']}")

        lines.append("")
        lines.append(f"  I(0) variables : {', '.join(i0) if i0 else 'none'}")
        lines.append(f"  I(1) variables : {', '.join(i1) if i1 else 'none'}")

        if i0 and i1:
            lines.append("")
            lines.append("  📌 Mixed orders detected  →  ARDL bounds-testing is the")
            lines.append("     preferred long-run framework (Pesaran et al., 2001).")
        elif i1 and not i0:
            lines.append("")
            lines.append("  📌 All variables I(1)  →  Johansen / VECM is valid.")

        lines.append("")
        return "\n".join(lines)

    def _section_engle_granger(self) -> str:
        df = self._load("engle_granger_pairwise.csv")
        lines = [_header("3 / ENGLE-GRANGER  —  pairwise cointegration")]

        if df is None:
            lines.append("  Data unavailable.\n")
            return "\n".join(lines)

        lines.append(f"  {'Dependent':<16} {'Independent':<16}"
                     f" {'t-stat':>8} {'p-val':>7}  Result")
        lines.append("  " + "─" * 62)

        coint_pairs = []
        for _, row in df.iterrows():
            coint = row.get("cointegrated", False)
            flag  = "Cointegrated ✓" if coint else "Not coint.   ✗"
            lines.append(
                f"  {row['dependent']:<16} {row['independent']:<16}"
                f" {row['t_statistic']:>8.4f} {row['p_value']:>7.4f}  {flag}"
            )
            if coint:
                coint_pairs.append((row["dependent"], row["independent"]))

        lines.append("")
        if coint_pairs:
            lines.append(f"  📌 Cointegrated pairs : {coint_pairs}")
        else:
            lines.append("  📌 No pairwise cointegration at 5 %.")
            lines.append("     (Inflation → ExchangeRate at p ≈ 0.08 is borderline.)")
        lines.append("")
        return "\n".join(lines)

    def _section_johansen(self) -> str:
        lines = [_header("4 / JOHANSEN  —  multivariate cointegration")]

        for label, fname in [("Full 4-variable system", "johansen_all.csv"),
                             ("I(1)-only robustness",   "johansen_i1_only.csv")]:
            df = self._load(fname)
            if df is None:
                continue

            lines.append(_sub(label))
            lines.append(f"  {'r':<4} {'Trace stat':>11} {'Trace CV5%':>11}"
                         f" {'MaxEig stat':>12} {'MaxEig CV5%':>12}")
            lines.append("  " + "─" * 56)
            for _, row in df.iterrows():
                lines.append(
                    f"  {int(row['r']):<4}"
                    f" {row['Trace_Statistic']:>11.3f}"
                    f" {row['Trace_CV_5pct']:>11.3f}"
                    f" {row['MaxEig_Statistic']:>12.3f}"
                    f" {row['MaxEig_CV_5pct']:>12.3f}"
                )

            # rank from trace
            rank_trace = df.loc[df["Trace_Reject"] == True].shape[0]  # noqa: E712
            rank_maxeig = 0
            for _, row in df.iterrows():
                if row["MaxEig_Reject"]:
                    rank_maxeig += 1
                else:
                    break

            lines.append("")
            lines.append(f"  Trace-test rank      : {rank_trace}")
            lines.append(f"  Max-eigenvalue rank  : {rank_maxeig}")
            lines.append("")

        return "\n".join(lines)

    def _section_summary(self) -> str:
        """Synthesise all findings into a model-selection memo."""
        # pull key numbers
        int_df   = self._load("integration_orders.csv")
        eg_df    = self._load("engle_granger_pairwise.csv")
        joh_df   = self._load("johansen_all.csv")

        i1_vars = int_df.loc[int_df["Order"] == "I(1)", "Variable"].tolist() if int_df is not None else []
        i0_vars = int_df.loc[int_df["Order"] == "I(0)", "Variable"].tolist() if int_df is not None else []
        eg_coint = int(eg_df["cointegrated"].sum()) if eg_df is not None else 0
        joh_rank = int(joh_df.loc[joh_df["Trace_Reject"] == True].shape[0]) if joh_df is not None else 0  # noqa: E712

        lines = [_header("5 / WEEK 1 SYNTHESIS  —  model-selection memo")]

        lines.append("  Variable classification")
        lines.append(f"    I(0) : {', '.join(i0_vars) if i0_vars else 'none'}")
        lines.append(f"    I(1) : {', '.join(i1_vars) if i1_vars else 'none'}")
        lines.append("")

        lines.append("  Cointegration evidence")
        lines.append(f"    EG pairwise  :  {eg_coint} cointegrated pair(s)")
        lines.append(f"    Johansen rank (trace, 5 %) :  {joh_rank}")
        lines.append("")

        lines.append("  Model-selection decision")
        if i0_vars and i1_vars:
            lines.append("    Mixed I(0)/I(1) confirmed  →  ARDL bounds-test (Week 2).")
            lines.append("    Johansen / VECM cannot be applied directly when I(0)")
            lines.append("    variables are present; ARDL handles this naturally.")
        elif i1_vars and not i0_vars:
            lines.append("    All I(1)  →  Johansen VECM is the primary framework.")

        if eg_coint > 0 or joh_rank > 0:
            lines.append("    Long-run equilibrium relationships exist in the data.")
            lines.append("    → Include error-correction term in the short-run spec.")
        else:
            lines.append("    No robust long-run link detected at 5 %.")
            lines.append("    → ARDL bounds test (more powerful, small-sample) will")
            lines.append("       provide the definitive answer in Week 2.")

        lines.append("")
        lines.append("  Week 2 roadmap")
        lines.append("    Day 6  –  ARDL bounds-testing (short + long run)")
        lines.append("    Day 7  –  VAR estimation + lag-order selection")
        lines.append("    Day 8  –  Impulse response functions (IRF)")
        lines.append("    Day 9  –  Forecast error variance decomposition (FEVD)")
        lines.append("    Day 10 –  +100 bps MPR policy-shock simulation")
        lines.append("")

        return "\n".join(lines)

    # ── master runner ────────────────────────────────────────────────
    def generate(self, save: bool = True) -> str:
        """Build the full report, print it, and optionally save."""
        print("\n" + "=" * 70)
        print("  WEEK 1 CONSOLIDATED REPORT")
        print("=" * 70)

        report = (
            self._section_unit_roots()
            + self._section_integration_orders()
            + self._section_engle_granger()
            + self._section_johansen()
            + self._section_summary()
        )

        print(report)

        if save:
            out_path = self.root / "week1_summary.md"
            # wrap in markdown code-fences for readability as .md
            md_body = (
                "# Week 1 Summary — Nigerian Monetary Policy Transmission\n\n"
                "Auto-generated by `src/econometrics/week1_report.py`.\n\n"
                "```\n" + report + "\n```\n"
            )
            out_path.write_text(md_body)
            print(f"\n  ✓ saved  {out_path}")

        return report


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    reporter = Week1Reporter(results_dir="results")
    reporter.generate()


if __name__ == "__main__":
    main()
