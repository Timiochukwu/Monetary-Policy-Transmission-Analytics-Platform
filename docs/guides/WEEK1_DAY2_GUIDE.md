# Week 1 – Day 2: Visualisation & Exploratory Analysis
## Complete Build Guide

**Duration**: 4-5 hours
**Difficulty**: ⭐⭐ (Beginner-Intermediate)
**Dependencies added today**: None (everything from Day 1 is enough)
**File you build**: `src/visualization/plots.py`

---

## 🎯 Objective

Turn raw numbers into **pictures that tell a story**.  By the end of
this guide you will:

1. Understand *why* each plot is chosen
2. Build a reusable `MacroPlotter` class from scratch
3. Generate 9 publication-quality figures
4. Visually identify the structural breaks that will matter for your
   econometric models later

---

## 📚 What You Need Before You Start

- Day 1 completed (folder structure, data loaded, `data_loader.py` working)
- Python virtual-env active
- Day 1 packages installed (`pip install -r requirements.txt`)

Quick smoke-test:
```bash
cd Monetary-Policy-Transmission-Analytics-Platform
python src/data_ingestion/data_loader.py
# Should print: ✓ Loaded 181 observations …
```

---

## 🧠 Part 1 — Why These Plots? (Theory First)

Before writing a single line of code, understand the **purpose** of
every figure.  A thesis examiner will ask *"why did you include this?"*

| # | Plot | What it shows | Why it matters |
|---|------|---------------|----------------|
| 1 | Time series (levels) | Raw trajectory of each variable | Visual stationarity check — do series wander or revert? |
| 2 | MPR vs Inflation | The **target relationship** | Does tightening actually cool inflation? |
| 3 | Rolling statistics | Trend + volatility over time | Detects regime changes and heteroskedasticity |
| 4 | First differences | Month-on-month shocks | Shows the *changes* the models will actually use |
| 5 | Distributions | Shape of data | Normality check; skewness flags potential log-transforms |
| 6 | Correlation heatmap | Pairwise linear relationships | Quick sense of signal strength before formal modelling |
| 7 | Scatter pairs | Four key economic channels | Visually confirms the transmission channels |
| 8 | Normalised overlay | Relative movements on one chart | Immediately reveals co-movement and lead/lag |
| 9 | Volatility | Rolling std of monthly % change | Shows which periods are "risky" — critical for policy context |

**Rule of thumb**: every plot should answer a single question.  If you
need a caption longer than one sentence to explain it, the plot is doing
too much.

---

## 🏗️ Part 2 — Project Constants (colours, labels, events)

**Why constants?**  Consistency across 9 figures.  If MPR is blue in
chart 1, it must be blue in chart 7.  Hard-coded colours scattered
through 9 functions = maintenance nightmare.

At the top of `plots.py` you define three dictionaries:

```python
# ── single source of truth ──────────────────────────────────
COLORS = {
    "MPR":            "#2E86AB",       # calm blue
    "Inflation":      "#A23B72",       # burgundy
    "ExchangeRate":   "#F18F01",       # amber
    "M2":             "#6A994E",       # forest green
}
```

**Colour psychology** (yes, it matters in thesis presentations):
- Blue = stable/policy
- Red/burgundy = prices/urgency
- Amber = exchange rates/FX risk
- Green = money/growth

```python
LABELS = {                             # (title string, y-axis unit)
    "MPR":            ("Monetary Policy Rate",          "% per annum"),
    "Inflation":      ("Headline Inflation (CPI)",     "% YoY"),
    "ExchangeRate":   ("Exchange Rate",                "NGN / USD"),
    "M2":             ("Broad Money Supply (M2)",      "₦ Billions"),
}
```

**Labels** carry two pieces of information: the human-readable name and
the unit.  Extracting them from a dict instead of hard-coding in every
function keeps things DRY (Don't Repeat Yourself).

```python
EVENTS = {
    "2016-06-01": ("2016 Devaluation",    "#E74C3C"),
    "2020-03-01": ("COVID-19",            "#E67E22"),
    "2023-06-01": ("2023 FX Unification", "#8E44AD"),
}
```

**Events** are the structural breaks embedded in Nigerian macro history.
Any examiner who knows Nigeria will ask about these.  Marking them
visually is both academically correct and intellectually honest.

---

## 🐍 Part 3 — Build `plots.py` Step by Step

**Create file**: `src/visualization/plots.py`

### 3.1 Imports & style setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
```

**Why each import?**
| Import | Used for |
|--------|----------|
| `pandas` | DataFrame slicing / `.rolling()` / `.pct_change()` |
| `numpy` | `polyfit` (trend lines), `linspace` (x-grids) |
| `matplotlib` | All drawing |
| `seaborn` | The correlation heatmap (one liner) |
| `pathlib` | Save-path construction (cross-platform) |

### 3.2 Helper functions

```python
def _style_ax(ax, title, ylabel, xlabel=None):
    """Apply a consistent look to every Axes."""
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=9)
```

**Why a helper?**  Nine plots × three label calls = 27 places where
font size could drift.  One function = one place to change.

```python
def _draw_events(self, ax, label_top=True):
    """Vertical dashed lines + badges for structural breaks."""
    ymin, ymax = ax.get_ylim()
    for date_str, (label, color) in EVENTS.items():
        ts = pd.Timestamp(date_str)
        if ts < self.df.index[0] or ts > self.df.index[-1]:
            continue                       # skip events outside sample
        ax.axvline(ts, color=color, linestyle="--", alpha=0.6, linewidth=1.4)
        if label_top:
            ax.text(ts, ymax * 0.97, label,        # <-- near top
                    fontsize=8, color=color, ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))
```

**`label_top=True` on only the first subplot** avoids label clutter in
a shared-x 4-panel grid.

### 3.3 The class shell

```python
class MacroPlotter:
    def __init__(self, df: pd.DataFrame, save_dir: str = "results"):
        self.df       = df
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-darkgrid")
        plt.rcParams.update({"font.size": 11, "figure.dpi": 100})

    def _save(self, fig, filename: str):
        path = self.save_dir / filename
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  ✓ saved  {path}")
```

**Key design choices**:
- `save_dir` is injected — makes unit-testing and CI easy
- `mkdir(parents=True, exist_ok=True)` — never crashes if dir exists
- `dpi=300` for saving, `dpi=100` for screen — print vs display
- `bbox_inches="tight"` — prevents axis labels being chopped off

### 3.4 Plot 1 — Time series (levels)

```python
    def plot_time_series(self) -> plt.Figure:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle("Nigerian Macroeconomic Variables  (2010–2025)",
                     fontsize=16, fontweight="bold", y=1.01)

        for ax, col in zip(axes, self.df.columns):
            title, ylabel = LABELS.get(col, (col, ""))

            ax.plot(self.df.index, self.df[col],
                    color=COLORS.get(col, "#333"), linewidth=2)

            # mean reference line
            ax.axhline(self.df[col].mean(),
                       color="red", linestyle="--", alpha=0.4,
                       label=f"Mean  {self.df[col].mean():.1f}")

            _style_ax(ax, title, ylabel)
            ax.legend(loc="upper left", fontsize=9)
            self._draw_events(ax, label_top=(ax is axes[0]))

        axes[-1].set_xlabel("Date", fontsize=11)
        fig.tight_layout()
        self._save(fig, "time_series_levels.png")
        return fig
```

**What to look for when you run it**:
- Does the series wander away from the mean and never come back?
  → Likely I(1) (unit root) — formal test in Day 3
- Does it oscillate around the mean?
  → Likely I(0) (stationary)
- Sharp one-time jumps?
  → Structural breaks — mark them with EVENTS

**Teaching moment**: `sharex=True` links all four x-axes.  Zooming one
zooms all.  Essential for visual comparison of timing.

### 3.5 Plot 2 — MPR vs Inflation

```python
    def plot_mpr_vs_inflation(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(self.df.index, self.df["MPR"],
                label="Monetary Policy Rate", color=COLORS["MPR"], linewidth=2.5)
        ax.plot(self.df.index, self.df["Inflation"],
                label="Inflation", color=COLORS["Inflation"], linewidth=2.5)

        self._draw_events(ax)
        ax.set_title("MPR vs Inflation — Transmission Dynamics",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Percent  (%)", fontsize=11)
        ax.legend(loc="upper left", fontsize=11)

        fig.tight_layout()
        self._save(fig, "mpr_vs_inflation.png")
        return fig
```

**Why this specific plot?**  It is the single most important visual in
your thesis.  An examiner looks at it and asks: *"does tightening
(↑MPR) actually reduce inflation?"*  The answer from the picture sets
expectations for every model result that follows.

**What to notice**:
- 2016: MPR stays high while inflation surges → pass-through dominates
- 2022-2024: MPR rises sharply but inflation follows with a lag →
  transmission is working, but slowly

### 3.6 Plot 3 — Rolling statistics

```python
    def plot_rolling_statistics(self, window: int = 12) -> plt.Figure:
        # … for each variable:
        mean = raw.rolling(window).mean()
        std  = raw.rolling(window).std()

        ax.plot(…, raw,  alpha=0.25, label="Raw")          # light trace
        ax.plot(…, mean, linewidth=2.5, label="12-mo mean")
        ax.fill_between(…, mean - std, mean + std,
                        alpha=0.12, label="± 1 σ")         # confidence band
```

**Key concept — window size**:
- `window=12` = one calendar year
- For monetary policy that transmits over 6-12 months, a 12-month
  window smooths noise while preserving real regime shifts
- A 3-month window would be too noisy; 24-month would hide the 2023
  FX shock entirely

**What to notice**:
- Width of the band = volatility.  Does it widen around events?
- If so → **heteroskedasticity** — your residuals are not constant
  variance.  Important for model diagnostics later.

### 3.7 Plot 4 — First differences

```python
    def plot_first_differences(self) -> plt.Figure:
        df_d = self.df.diff().dropna()          # Δy_t = y_t − y_{t-1}
        # … bar chart, positive in theme colour, negative in grey
        vals    = df_d[col].values
        colours = [COLORS[col] if v >= 0 else "#BDBDBD" for v in vals]
        ax.bar(df_d.index, vals, width=18, color=colours)
```

**Why first differences matter**:
- Most econometric models work on *stationary* data
- First-differencing removes a linear trend → often makes data
  stationary (formal proof in Day 3)
- This chart is a **preview** of what your models will see

**Bar vs line chart?**  Bars emphasise discrete monthly shocks.  Lines
smooth them — less honest for a model that treats each month as a
separate observation.

### 3.8 Plot 5 — Distributions

```python
        ax.hist(self.df[col], bins=28, …)
        ax.axvline(mn, color="red",       linestyle="--", label=f"Mean …")
        ax.axvline(md, color="darkgreen", linestyle=":",  label=f"Median …")
```

**Why mean AND median?**
- If they differ noticeably → **skewed distribution**
- Skewed data can cause OLS bias
- Log-transformation often fixes this (we'll decide in Week 2)

**Bin count = 28**: rule of thumb for 181 observations is
`bins ≈ √N ≈ 13`.  We use 28 for visual granularity.  If bars are
too thin, reduce; if too wide, increase.

### 3.9 Plot 6 — Correlation heatmap

```python
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    square=True, vmin=-1, vmax=1, …)
```

**Interpretation guide**:
| r range | Meaning |
|---------|---------|
| 0.7 – 1.0 | Strong positive |
| 0.3 – 0.7 | Moderate |
| −0.3 – 0.3 | Weak / no linear relationship |
| −0.7 – −0.3 | Moderate negative |
| −1.0 – −0.7 | Strong negative |

**Warning**: correlation ≠ causation.  MPR and ExchangeRate may
correlate at 0.8 simply because both trend upward over time.  After
differencing the correlation might vanish.  The heatmap is descriptive,
not causal.

### 3.10 Plot 7 — Scatter pairs

```python
        coeffs = np.polyfit(self.df[x], self.df[y], 1)   # OLS line
        x_line = np.linspace(self.df[x].min(), self.df[x].max(), 120)
        ax.plot(x_line, np.polyval(coeffs, x_line), "r--", linewidth=2)
```

**The four pairs are chosen deliberately**:
1. MPR → Inflation:  the **policy target** channel
2. ExchangeRate → Inflation:  the **import-price pass-through** channel
3. MPR → ExchangeRate:  the **interest-rate parity** channel
4. M2 → Inflation:  the **quantity theory** channel

Each one maps to a transmission mechanism your VAR will identify.

**`np.polyfit` in one line?**  It fits a degree-1 polynomial (straight
line) via OLS.  Equivalent to `sklearn LinearRegression` but zero extra
imports.

### 3.11 Plot 8 — Normalised overlay

```python
        normed = (self.df - self.df.min()) / (self.df.max() - self.df.min())
```

**Why normalise?**  MPR lives in 6–28; M2 in 9,000–64,000.  On the
same y-axis, M2 dominates and MPR disappears.  Min-max scales every
series to [0, 1] so visual co-movement becomes apparent.

**What to notice**:
- Do MPR and ExchangeRate move together?  (Interest parity)
- Does Inflation lag MPR by a few months?  (Transmission lag)
- When does M2 diverge from the pack?  (Liquidity shocks)

### 3.12 Plot 9 — Volatility

```python
        pct = self.df.pct_change().dropna() * 100   # monthly % Δ
        vol = pct.rolling(window).std()              # rolling σ
```

**`pct_change()` vs `diff()`?**
- `diff()` = absolute change (Naira amount)
- `pct_change()` = relative change (%)
- Volatility in *percentage* terms is comparable across variables

**What to notice**:
- Exchange rate volatility spikes in 2016, 2020, 2023 → regime events
- Is inflation volatility clustering?  (→ ARCH effects — advanced topic)

### 3.13 Master runner

```python
    def generate_all(self):
        steps = [
            ("1/9  Time series",        self.plot_time_series),
            ("2/9  MPR vs Inflation",   self.plot_mpr_vs_inflation),
            # … etc
        ]
        for label, fn in steps:
            print(f"  [{label}]")
            fn()
            plt.close("all")       # ← IMPORTANT: frees RAM
```

**`plt.close("all")`**: each figure allocates memory.  Without closing,
nine 300-DPI figures eat ~200 MB.  Close after saving.

### 3.14 CLI entry point

```python
def main():
    import sys
    sys.path.insert(0, ".")
    from src.data_ingestion.data_loader import NigerianMacroDataLoader

    loader  = NigerianMacroDataLoader(data_dir="data")
    df      = loader.load_and_prepare()
    plotter = MacroPlotter(df, save_dir="results")
    plotter.generate_all()

if __name__ == "__main__":
    main()
```

**`sys.path.insert(0, ".")`**: tells Python "look in the current working
directory for packages".  You must run from the project root:
```bash
cd Monetary-Policy-Transmission-Analytics-Platform
python src/visualization/plots.py
```

---

## 🚀 Part 4 — Run It

```bash
# Make sure you're at the project root
cd Monetary-Policy-Transmission-Analytics-Platform

# Run  (MPLBACKEND=Agg if you have no display / are on a server)
MPLBACKEND=Agg python src/visualization/plots.py
```

**Expected output**:
```
============================================================
  GENERATING ALL EDA VISUALISATIONS
============================================================

  [1/9  Time series (levels)]
  ✓ saved  results/time_series_levels.png

  [2/9  MPR vs Inflation]
  ✓ saved  results/mpr_vs_inflation.png
  …
  [9/9  Volatility]
  ✓ saved  results/volatility.png

============================================================
  ✓ ALL 9 PLOTS SAVED
============================================================
```

**Verify the output files exist**:
```bash
ls -lh results/*.png
# 9 files, each 200-500 KB at 300 DPI
```

**Open one to check** (on your local machine):
```bash
# macOS
open results/mpr_vs_inflation.png

# Linux
xdg-open results/mpr_vs_inflation.png

# Windows
start results\mpr_vs_inflation.png
```

---

## 📝 Part 5 — Read the Plots, Answer the Questions

This is the most important part of Day 2.  Open each PNG and write
down your answers.  They become the **EDA section of your thesis**.

### Questions to answer (write them down!)

**From Plot 1 (time series)**:
- [ ] Which variables appear to trend upward without mean-reversion?
- [ ] Which variables appear to oscillate around a central value?
- [ ] Where do you see abrupt level shifts?

**From Plot 2 (MPR vs Inflation)**:
- [ ] Does inflation respond to MPR increases? With what lag?
- [ ] Are there periods where MPR rises but inflation also rises?
  What might explain that?

**From Plot 3 (rolling statistics)**:
- [ ] Which variable has the widest volatility band?
- [ ] Does the band width change over time? (heteroskedasticity)

**From Plot 4 (first differences)**:
- [ ] Do the bars cluster (many positive followed by many negative)?
  That would suggest serial correlation.
- [ ] Which variable has the largest single-month shock?

**From Plot 6 (correlation)**:
- [ ] What is the strongest pairwise correlation?
- [ ] Is it between variables you'd expect to be related?

**From Plot 8 (normalised overlay)**:
- [ ] Do any two variables move almost identically?
- [ ] Which variable moves most independently?

**From Plot 9 (volatility)**:
- [ ] Do volatility spikes coincide with the marked events?

---

## 🔧 Part 6 — Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
```bash
# Wrong: running from inside src/
cd Monetary-Policy-Transmission-Analytics-Platform   # project root
python src/visualization/plots.py
```

### "seaborn-v0_8-darkgrid not found"
Older seaborn used `seaborn-darkgrid`.  Check your version:
```python
import seaborn; print(seaborn.__version__)
# If < 0.12: use "seaborn-darkgrid"
# If >= 0.12: use "seaborn-v0_8-darkgrid"
```

### Plots are blank / white
You're likely on a headless server.  Set the backend before importing:
```bash
MPLBACKEND=Agg python src/visualization/plots.py
```

### "Permission denied" when saving
```bash
chmod -R u+w results/
```

---

## ✅ Day 2 Completion Checklist

### Code
- [ ] `src/visualization/plots.py` created (full file)
- [ ] Script runs without errors
- [ ] 9 PNG files appear in `results/`
- [ ] All files are < 1 MB each (sanity check)

### Understanding
- [ ] Can explain why MPR vs Inflation is the single most important plot
- [ ] Know the difference between `diff()` and `pct_change()`
- [ ] Understand why colours and labels are stored as constants
- [ ] Know what `plt.close("all")` does and why
- [ ] Can read a correlation heatmap and state what r = 0.85 means

### Git
```bash
git add src/visualization/plots.py results/*.png
git commit -m "Day 2: EDA visualisation module — 9 publication-quality plots"
git push -u origin claude/monetary-policy-analytics-platform-03tRK
```

---

## 🎓 Key Concepts Recap

| Concept | Takeaway |
|---------|----------|
| Constants | Single source of truth prevents drift |
| `sharex=True` | Linked axes for temporal comparison |
| `dpi=300` | Print-quality; 150 is screen-only |
| Rolling window | 12 months matches policy-transmission horizon |
| Min-max normalisation | Puts heterogeneous units on one chart |
| Mean vs median | Gap reveals skewness |
| `polyfit` degree 1 | OLS in two lines, no sklearn needed |

---

## 🚀 Day 3 Preview

**Objective**: Formal stationarity testing (ADF, PP, KPSS)

**What you'll build**: `src/econometrics/stationarity_tests.py`

**New dependencies**: `statsmodels`, `scipy`  (install on Day 3 only)

**Why it matters**: The plots today gave you *visual* answers to
"is this stationary?"  Day 3 gives you **statistical proof** — the
kind that passes peer review.

---

*Guide v1.0  ·  Day 2  ·  2025-02*
