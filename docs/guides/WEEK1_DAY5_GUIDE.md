# Week 1 · Day 5 — Week 1 Review & Consolidated Report

> **What you will build:** `src/econometrics/week1_report.py`
> **What you will produce:** `results/week1_summary.md` — a single
> document that pulls together every Day 3 and Day 4 finding.
> **Why it matters:** Before moving to Week 2's heavier models, you
> need a clear, written summary of *what the data told us*.  This
> script is that summary — and it doubles as a draft for your thesis
> Chapter 2 (Literature Review + Data Description).

---

## 0  What Was Built This Week

| Day | Deliverable | Key output |
|-----|-------------|------------|
| 1 | Data loader + validator | 181-row cleaned DataFrame |
| 2 | 9 EDA visualisations | `results/*.png` |
| 3 | Stationarity tests (ADF / PP / KPSS) | `integration_orders.csv` |
| 4 | Cointegration tests (EG + Johansen) | `engle_granger_pairwise.csv`, `johansen_*.csv` |
| **5** | **Consolidated report** | **`results/week1_summary.md`** |

---

## 1  The Report Script — How It Works

`week1_report.py` is deliberately simple: it *reads* CSVs that Days 3
and 4 already produced and *formats* them into a unified narrative.
No econometric computation happens here — it is a **presentation
layer** on top of the earlier modules.

### 1.1  Class structure

```python
class Week1Reporter:
    def __init__(self, results_dir="results"):
        # points to results/stationarity/ sub-tree

    # One method per report section:
    def _section_unit_roots()          → str   # Panel A + B
    def _section_integration_orders()  → str   # I(0)/I(1) table + ARDL flag
    def _section_engle_granger()       → str   # 12-pair table
    def _section_johansen()            → str   # trace + max-eig tables
    def _section_summary()             → str   # synthesis + Week 2 roadmap

    def generate(save=True)            → str   # orchestrator
```

Each `_section_*` method:
1. Calls `self._load(filename)` to read the relevant CSV.
2. Formats it into aligned, readable text.
3. Returns the text as a string (so `generate()` can concatenate).

### 1.2  Graceful missing files

If a CSV does not exist (e.g. you skipped Day 4), the loader prints a
warning and returns `None`.  Each section checks for `None` and
substitutes a "data unavailable" message instead of crashing.

```python
def _load(self, name):
    path = self.stat_dir / name
    if path.exists():
        return pd.read_csv(path)
    print(f"  ⚠ {path} not found — skipping.")
    return None
```

### 1.3  The synthesis section

Section 5 (`_section_summary`) does the only piece of *interpretation*
in the script.  It checks three conditions:

| Condition | Action |
|-----------|--------|
| Both I(0) and I(1) variables present | Print "ARDL preferred" |
| All I(1) | Print "Johansen VECM valid" |
| EG or Johansen evidence of cointegration | Print "long-run equilibrium exists → include ECT" |
| No cointegration at all | Print "ARDL bounds test is more powerful — check in Week 2" |

---

## 2  Run the Full Week 1 Pipeline

Run Days 3 → 4 → 5 in order (each depends on the previous):

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform

# Day 3 — stationarity (produces integration_orders.csv)
python src/econometrics/stationarity_tests.py

# Day 4 — cointegration (reads integration_orders.csv)
MPLBACKEND=Agg python src/econometrics/cointegration_tests.py

# Day 5 — consolidated report (reads all CSVs)
python src/econometrics/week1_report.py
```

The final output `results/week1_summary.md` is a Markdown file you can
open in any editor, GitHub, or paste directly into a thesis draft.

---

## 3  Reading the Summary — Quick Guide

The report has five numbered sections:

1. **Unit-Root Tests** — the raw ADF / PP / KPSS numbers in panels for
   levels and first differences.  Scan for "Yes" / "No" in the
   Stationary column.

2. **Integration Orders** — the *consensus* decision for each variable.
   This is the single most important table: it determines which models
   are valid in Week 2.

3. **Engle-Granger** — the pairwise cointegration table.  Look for the
   ✓ marks.  Note which variable is *dependent* (it adjusts to
   restore equilibrium).

4. **Johansen** — two sub-tables (full system, I(1)-only).  Compare
   trace rank vs max-eigenvalue rank.  If they disagree, trust max-eig.

5. **Synthesis** — the model-selection memo.  This is what you would
   write in your thesis's "Preliminary Analysis" section.

---

## 4  The Interpretation Document

`docs/interpretation.md` expands on every finding with Nigerian
economic context.  Key points:

- **MPR is I(1)** because the CBN adjusts rates persistently.
- **Exchange rate is I(1)** because of structural FX-market shifts.
- **M2 is I(0)** because reserve requirements and loan dynamics
  keep money supply mean-reverting.
- **Inflation is I(0)** because it cycles around a (high) equilibrium.
- The **Inflation–M2 cointegrating link** is the quantity-theory
  channel (money → prices).
- **Johansen rank disagreement** (trace = 4, max-eig = 1) is resolved
  by favouring max-eigenvalue.

Read it in full before Week 2.

---

## 5  What Changes in Week 2?

| Week 1 question | Week 2 question |
|-----------------|-----------------|
| Are the series stationary? | How do they relate in the *long run*? |
| Do they cointegrate? | How fast does the system *adjust* after a shock? |
| Which model framework? | Estimate the model and *quantify* transmission. |

### Week 2 day map

| Day | Model | Output |
|-----|-------|--------|
| 6 | ARDL bounds-test | Long-run equation + ECM |
| 7 | VAR (4-variable, Cholesky) | Coefficient estimates |
| 8 | Impulse Response Functions | MPR shock → Inflation path |
| 9 | Forecast Error Variance Decomposition | Share of Inflation variance from MPR |
| 10 | +100 bps policy simulation | Policy-scenario report |

---

## 6  Checklist

- [ ] `stationarity_tests.py` has been run (Day 3 CSVs exist)
- [ ] `cointegration_tests.py` has been run (Day 4 CSVs exist)
- [ ] `week1_report.py` runs and saves `results/week1_summary.md`
- [ ] You have read `docs/interpretation.md`
- [ ] You can state in one sentence *why* ARDL is the right framework
- [ ] You can explain the Johansen trace vs max-eigenvalue disagreement
- [ ] You are ready for Week 2 — ARDL and VAR
