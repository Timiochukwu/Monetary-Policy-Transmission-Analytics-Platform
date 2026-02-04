"""
Visualization Module — Nigerian Monetary Policy Transmission Analysis

Generates publication-quality plots for EDA and thesis presentation.
All figures save at 300 DPI into the results/ tree.

Day 2 deliverable.  Dependencies: pandas, numpy, matplotlib, seaborn (Day 1 only).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# PROJECT-WIDE CONSTANTS  (single source of truth for every plot)
# ─────────────────────────────────────────────────────────────────────

COLORS = {
    "MPR":            "#2E86AB",
    "Inflation":      "#A23B72",
    "ExchangeRate":   "#F18F01",
    "M2":             "#6A994E",
}

LABELS = {                          # (title, y-axis unit)
    "MPR":            ("Monetary Policy Rate",          "% per annum"),
    "Inflation":      ("Headline Inflation (CPI)",     "% YoY"),
    "ExchangeRate":   ("Exchange Rate",                "NGN / USD"),
    "M2":             ("Broad Money Supply (M2)",      "₦ Billions"),
}

# Nigerian structural-break episodes embedded in the dataset
EVENTS = {
    "2016-06-01": ("2016 Devaluation",   "#E74C3C"),
    "2020-03-01": ("COVID-19",           "#E67E22"),
    "2023-06-01": ("2023 FX Unification","#8E44AD"),
}


# ─────────────────────────────────────────────────────────────────────
# HELPER — apply a consistent look to every Axes object
# ─────────────────────────────────────────────────────────────────────

def _style_ax(ax, title, ylabel, xlabel=None):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)


# ─────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ─────────────────────────────────────────────────────────────────────

class MacroPlotter:
    """
    Stateless plotting façade.

    Instantiate once with your processed DataFrame and an output
    directory.  Every method returns the matplotlib Figure so you can
    show or tweak it interactively, *and* persists a 300-DPI PNG.

    Parameters
    ----------
    df       : pd.DataFrame
        Columns  MPR | ExchangeRate | M2 | Inflation, DatetimeIndex.
    save_dir : str | Path
        Root folder for saved images (e.g. ``results``).
    """

    def __init__(self, df: pd.DataFrame, save_dir: str = "results"):
        self.df = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update({"font.size": 11, "figure.dpi": 100})

    # ── internal ──────────────────────────────────────────────────────
    def _save(self, fig, filename: str):
        path = self.save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")

    def _draw_events(self, ax, label_top=True):
        """Overlay vertical lines + labels for structural-break episodes."""
        ymin, ymax = ax.get_ylim()
        for date_str, (label, color) in EVENTS.items():
            ts = pd.Timestamp(date_str)
            if ts < self.df.index[0] or ts > self.df.index[-1]:
                continue
            ax.axvline(ts, color=color, linestyle="--", alpha=0.6, linewidth=1.4)
            if label_top:
                ax.text(
                    ts, ymax * 0.97, label,
                    fontsize=8, color=color, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75),
                )

    # ── 1. four-panel time series ─────────────────────────────────────
    def plot_time_series(self) -> plt.Figure:
        """Individual time-series plot for every variable."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle(
            "Nigerian Macroeconomic Variables  (2010–2025)",
            fontsize=16, fontweight="bold", y=1.01,
        )

        for ax, col in zip(axes, self.df.columns):
            title, ylabel = LABELS.get(col, (col, ""))
            ax.plot(self.df.index, self.df[col],
                    color=COLORS.get(col, "#333"), linewidth=2)
            ax.axhline(self.df[col].mean(),
                       color="red", linestyle="--", alpha=0.4, linewidth=1,
                       label=f"Mean  {self.df[col].mean():.1f}")
            _style_ax(ax, title, ylabel)
            ax.legend(loc="upper left", fontsize=9)
            self._draw_events(ax, label_top=(ax is axes[0]))

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "time_series_levels.png")
        return fig

    # ── 2. MPR vs Inflation overlay ───────────────────────────────────
    def plot_mpr_vs_inflation(self) -> plt.Figure:
        """Key-relationship overlay: the two series examiners care about."""
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(self.df.index, self.df["MPR"],
                label="Monetary Policy Rate", color=COLORS["MPR"], linewidth=2.5)
        ax.plot(self.df.index, self.df["Inflation"],
                label="Inflation", color=COLORS["Inflation"], linewidth=2.5)

        self._draw_events(ax)
        ax.set_title(
            "MPR vs Inflation — Transmission Dynamics",
            fontsize=14, fontweight="bold",
        )
        ax.set_ylabel("Percent  (%)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        self._save(fig, "mpr_vs_inflation.png")
        return fig

    # ── 3. rolling statistics ─────────────────────────────────────────
    def plot_rolling_statistics(self, window: int = 12) -> plt.Figure:
        """12-month rolling mean ± 1 σ band for every variable."""
        fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
        fig.suptitle(
            f"Rolling Statistics  ({window}-Month Window)",
            fontsize=15, fontweight="bold", y=1.01,
        )

        for ax, col in zip(axes, self.df.columns):
            title, ylabel = LABELS.get(col, (col, ""))
            raw  = self.df[col]
            mean = raw.rolling(window).mean()
            std  = raw.rolling(window).std()

            ax.plot(self.df.index, raw,  color=COLORS[col], alpha=0.25, linewidth=1, label="Raw")
            ax.plot(self.df.index, mean, color=COLORS[col], linewidth=2.5,            label=f"{window}-mo mean")
            ax.fill_between(self.df.index, mean - std, mean + std,
                            color=COLORS[col], alpha=0.12, label="± 1 σ")

            _style_ax(ax, f"{title} — rolling", ylabel)
            ax.legend(loc="upper left", fontsize=9)

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "rolling_statistics.png")
        return fig

    # ── 4. first differences (bar chart) ──────────────────────────────
    def plot_first_differences(self) -> plt.Figure:
        """Month-on-month change, positive in theme colour, negative in grey."""
        df_d = self.df.diff().dropna()
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        fig.suptitle(
            "First Differences  (Month-on-Month Change)",
            fontsize=15, fontweight="bold", y=1.01,
        )

        for ax, col in zip(axes, df_d.columns):
            title, ylabel = LABELS.get(col, (col, ""))
            vals = df_d[col].values
            colours = [COLORS.get(col, "#333") if v >= 0 else "#BDBDBD" for v in vals]

            ax.bar(df_d.index, vals, width=18, color=colours, edgecolor="none")
            ax.axhline(0, color="black", linewidth=0.7)
            _style_ax(ax, f"Δ {title}", f"Δ {ylabel}")

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "first_differences.png")
        return fig

    # ── 5. distributions ──────────────────────────────────────────────
    def plot_distributions(self) -> plt.Figure:
        """Histogram + mean / median lines for each variable."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Variable Distributions", fontsize=15, fontweight="bold")

        for idx, col in enumerate(self.df.columns):
            ax = axes[idx // 2, idx % 2]
            title, xlabel = LABELS.get(col, (col, ""))

            ax.hist(self.df[col], bins=28, color=COLORS.get(col, "#333"),
                    edgecolor="black", alpha=0.55, linewidth=0.6)

            # vertical lines: mean (dashed-red) and median (dotted-green)
            mn, md = self.df[col].mean(), self.df[col].median()
            ax.axvline(mn, color="red",       linestyle="--", linewidth=1.8,
                       label=f"Mean   {mn:.1f}")
            ax.axvline(md, color="darkgreen", linestyle=":",  linewidth=1.8,
                       label=f"Median {md:.1f}")

            _style_ax(ax, title, "Frequency", xlabel)
            ax.legend(fontsize=9)

        fig.tight_layout()
        self._save(fig, "distributions.png")
        return fig

    # ── 6. correlation heatmap ────────────────────────────────────────
    def plot_correlation_heatmap(self) -> plt.Figure:
        """Annotated correlation matrix.  Returns (fig, corr DataFrame)."""
        corr = self.df.corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, linewidths=1.5, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 13, "weight": "bold"},
            ax=ax,
        )
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "correlation_heatmap.png")
        return fig

    # ── 7. pairwise scatter + OLS trend lines ─────────────────────────
    def plot_scatter_pairs(self) -> plt.Figure:
        """Four economically-motivated scatter plots with r printed."""
        pairs = [
            ("MPR",            "Inflation",      "MPR → Inflation  (target)"),
            ("ExchangeRate",   "Inflation",      "Exchange Rate → Inflation  (pass-through)"),
            ("MPR",            "ExchangeRate",   "MPR → Exchange Rate  (interest parity)"),
            ("M2",             "Inflation",      "M2 → Inflation  (quantity theory)"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("Key Pairwise Relationships", fontsize=15, fontweight="bold")

        for ax, (x, y, title) in zip(axes.flat, pairs):
            ax.scatter(self.df[x], self.df[y],
                       color="#2C3E50", alpha=0.55, edgecolors="white", s=55, zorder=3)

            # OLS trend line
            coeffs = np.polyfit(self.df[x], self.df[y], 1)
            x_line = np.linspace(self.df[x].min(), self.df[x].max(), 120)
            ax.plot(x_line, np.polyval(coeffs, x_line),
                    "r--", linewidth=2, alpha=0.7, zorder=2)

            # Pearson r badge
            r = self.df[x].corr(self.df[y])
            ax.text(0.04, 0.94, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.85))

            _style_ax(ax, title, LABELS.get(y, (y, ""))[0], LABELS.get(x, (x, ""))[0])

        fig.tight_layout()
        self._save(fig, "scatter_pairs.png")
        return fig

    # ── 8. min-max-normalised overlay ─────────────────────────────────
    def plot_normalised_overlay(self) -> plt.Figure:
        """All four variables rescaled to [0, 1] — reveals relative moves."""
        normed = (self.df - self.df.min()) / (self.df.max() - self.df.min())

        fig, ax = plt.subplots(figsize=(14, 6))
        for col in normed.columns:
            ax.plot(normed.index, normed[col],
                    color=COLORS[col], linewidth=1.8,
                    label=LABELS.get(col, (col, ""))[0])

        self._draw_events(ax)
        ax.set_title("Normalised Variables  (0–1 scale)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Normalised value", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        self._save(fig, "normalised_overlay.png")
        return fig

    # ── 9. growth-rate volatility (rolling std of pct change) ─────────
    def plot_volatility(self, window: int = 12) -> plt.Figure:
        """Rolling standard-deviation of monthly % change — measures risk."""
        pct = self.df.pct_change().dropna() * 100          # monthly % change
        vol = pct.rolling(window).std()                     # rolling volatility

        fig, ax = plt.subplots(figsize=(14, 5))
        for col in vol.columns:
            ax.plot(vol.index, vol[col],
                    color=COLORS[col], linewidth=1.8,
                    label=LABELS.get(col, (col, ""))[0])

        self._draw_events(ax)
        ax.set_title(f"Rolling Volatility  ({window}-Month Std Dev of Monthly % Change)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Volatility  (% pts)", fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.25)

        fig.tight_layout()
        self._save(fig, "volatility.png")
        return fig

    # ── master runner ─────────────────────────────────────────────────
    def generate_all(self):
        """Iterate through every plot method, save, and close."""
        print("=" * 60)
        print("  GENERATING ALL EDA VISUALISATIONS")
        print("=" * 60)

        steps = [
            ("1/9  Time series (levels)",        self.plot_time_series),
            ("2/9  MPR vs Inflation",            self.plot_mpr_vs_inflation),
            ("3/9  Rolling statistics",          self.plot_rolling_statistics),
            ("4/9  First differences",           self.plot_first_differences),
            ("5/9  Distributions",               self.plot_distributions),
            ("6/9  Correlation heatmap",         self.plot_correlation_heatmap),
            ("7/9  Scatter pairs",               self.plot_scatter_pairs),
            ("8/9  Normalised overlay",          self.plot_normalised_overlay),
            ("9/9  Volatility",                  self.plot_volatility),
        ]

        for label, fn in steps:
            print(f"\n  [{label}]")
            fn()
            plt.close("all")          # free memory between figures

        print("\n" + "=" * 60)
        print("  ✓ ALL 9 PLOTS SAVED")
        print("=" * 60)


# ─────────────────────────────────────────────────────────────────────
# CLI ENTRY-POINT
# ─────────────────────────────────────────────────────────────────────

def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader = NigerianMacroDataLoader(data_dir="data")
    df     = loader.load_and_prepare()

    plotter = MacroPlotter(df, save_dir="results")
    plotter.generate_all()


if __name__ == "__main__":
    main()
