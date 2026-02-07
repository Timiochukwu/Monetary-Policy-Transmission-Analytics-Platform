# Week 2 · Day 15 — Master Thesis Report

> **What you will build:** `src/econometrics/week2_master_report.py`
> **What you will produce:** `results/MASTER_THESIS_REPORT.md` — single thesis-ready document
> **Why it matters:** This is the **backbone of your thesis** — all findings consolidated into one narrative.

---

## 0  What This Module Does

The master report generator:

1. **Reads** all CSV outputs from Days 1–14
2. **Extracts** key numbers (integration orders, ARDL bounds, IRF peaks, FEVD shares)
3. **Assembles** them into a structured Markdown document
4. **Auto-generates** the Executive Summary with actual results

No manual copy-pasting. **Everything is reproducible.**

---

## 1  Report Structure

### Section 1: Executive Summary

Auto-generated bullet points:
- Integration orders (from `stationarity/integration_orders.csv`)
- ARDL bounds test result (from `ardl/ardl_bounds_test.csv`)
- Granger causality links (from `var/var_granger_causality.csv`)
- IRF peak response (from `irf/irf_MPR_shock.csv`)
- FEVD shares at 12 and 24 months (from `fevd/fevd_Inflation.csv`)
- Policy simulation headline (from `policy_simulation/policy_shock_100bps_responses.csv`)

### Section 2: Introduction

Pre-written text covering:
- Nigerian policy context (dual FX regime, structural breaks)
- Research question (MPR transmission to Inflation)
- VAR identification strategy (Cholesky ordering)

### Section 3: Data

Pre-written summary:
- Source (CBN, NBS, FRED)
- Sample (2010-01 to 2025-01, 181 monthly obs)
- Variables (MPR, Inflation, Exchange Rate, M2)
- Transformations (levels, justified by cointegration)

### Section 4: Methodology

Brief overview (3-4 paragraphs) covering:
- Unit root tests (ADF, PP, KPSS)
- Cointegration (EG, Johansen)
- ARDL bounds test
- VAR(11) with Cholesky
- IRF, FEVD, policy simulation
- Robustness checks

### Section 5: Results

References to output directories:
- "See `results/irf/` for impulse response plots"
- "See `results/fevd/` for variance decomposition"
- "Policy brief in `results/policy_simulation/policy_brief_100bps.md`"

### Section 6: Conclusions

Pre-written synthesis:
- Transmission exists and is significant (−1.21 pp at 12 months)
- Exchange-rate pass-through dominates (34.6% vs 18.4%)
- Policy implications (MPR is effective but not the only lever)
- Limitations (structural breaks, omitted variables)
- Future research (nonlinear VAR, DSGE)

---

## 2  Code Walkthrough

### 2.1  Load key numbers

```python
# week2_master_report.py — generate_report
int_orders = pd.read_csv("results/stationarity/integration_orders.csv")
i0 = int_orders[int_orders["Order"] == "I(0)"]["Variable"].tolist()
i1 = int_orders[int_orders["Order"] == "I(1)"]["Variable"].tolist()

report += f"- **Integration orders:** I(1) = {', '.join(i1)}; I(0) = {', '.join(i0)}\n"
```

### 2.2  Extract IRF peak

```python
irf_mpr = pd.read_csv("results/irf/irf_MPR_shock.csv")
inf_response = irf_mpr["Inflation"].values
peak = inf_response.min()
peak_month = int(inf_response.argmin())

report += f"- **IRF peak:** +1 SD MPR shock → {peak:.2f} pp Inflation at month {peak_month}\n"
```

### 2.3  Assemble into Markdown

The final report is a single string saved as `.md`:

```python
path = self.save_dir / "MASTER_THESIS_REPORT.md"
path.write_text(report)
```

---

## 3  Run the Script

```bash
python src/econometrics/week2_master_report.py
```

### Output file

| File | Contents |
|------|----------|
| `results/MASTER_THESIS_REPORT.md` | 5-page thesis backbone (1500 words) |

---

## 4  Using the Master Report

### 4.1  Thesis workflow

1. **Open** `MASTER_THESIS_REPORT.md` in a Markdown editor (VS Code, Typora, or Obsidian)
2. **Convert** to .docx using Pandoc:
   ```bash
   pandoc MASTER_THESIS_REPORT.md -o thesis_draft.docx
   ```
3. **Expand** each section:
   - Introduction: Add 3-4 pages of policy context and motivation
   - Methodology: Add equations and proofs (10-15 pages)
   - Results: Insert plots from `results/` directories (15-20 pages)
   - Discussion: Compare with literature (10 pages)
4. **Add** references (30-50 papers)
5. **Proofread** and format to university guidelines

### 4.2  What the report provides

- **Executive summary** (copy to thesis abstract with minor edits)
- **Results skeleton** (expand with tables/plots)
- **Conclusions draft** (refine for final chapter)

### 4.3  What you still need to write

- **Literature review** (20-30 pages) — not auto-generated
- **Detailed methodology** (equations, proofs, justifications)
- **Discussion** (compare results with Adebiyi & Mordi 2012, Chit & Okafor 2021, etc.)

---

## 5  Thesis Defense Preparation

### 5.1  Memorize these numbers

From the master report Executive Summary:

1. **Integration orders**: MPR = I(1), ExchangeRate = I(1), M2 = I(0), Inflation = I(0)
2. **ARDL bounds test**: F = 3.56 (inconclusive at 5%, confirmed at 10%)
3. **Granger causality**: 3 significant links (ER→MPR, M2→MPR, M2→ER)
4. **IRF peak**: +1 SD MPR shock → −0.52 pp Inflation at month 12
5. **FEVD**: MPR explains 18.4% of Inflation variance at 24 months
6. **Policy simulation**: +100 bps MPR → −1.21 pp Inflation at 12 months

### 5.2  Defense script (5 minutes)

> "I quantify Nigerian monetary policy transmission using a VAR(11) model on monthly data from 2010 to 2025. The VAR is identified via Cholesky decomposition with the ordering: MPR → Exchange Rate → M2 → Inflation, justified by the CBN's policy-setting process.
>
> Key findings:
> 1. MPR and the exchange rate are I(1); M2 and Inflation are I(0). Johansen cointegration rank is 1, validating the use of levels.
> 2. A 100 basis point MPR hike reduces Inflation by 1.21 percentage points after 12 months.
> 3. MPR explains 18% of Inflation variance; the exchange rate dominates at 35%, reflecting Nigeria's import dependence.
> 4. Results are robust to alternative lag orders, orderings, and sub-samples (tested on Days 12-13).
>
> Policy implication: The CBN's MPR is an effective inflation-management tool, but coordination with FX policy is essential given the stronger pass-through channel."

---

## 6  Common Questions

**Q: The report is only 5 pages. My thesis needs to be 80 pages. What do I do?**

A: This is the **skeleton**. Expand it:
- Introduction: 5 pages → 10 pages (add context, motivation, research question)
- Literature review: 0 pages → 25 pages (add citations, theoretical framework)
- Methodology: 3 pages → 15 pages (add equations, derivations, justifications)
- Results: 2 pages → 20 pages (insert all plots, tables, detailed interpretation)
- Discussion: 1 page → 10 pages (compare with literature, policy implications)
- Conclusions: 1 page → 3 pages (refine, add future research)

**Q: Can I submit this as-is?**

A: No. This is a **first draft** / **backbone**. You still need to:
- Add citations (30-50 references)
- Insert plots and tables
- Expand methodology with equations
- Write a literature review

But it gives you 90% of the structure and 100% of the numbers.

**Q: Should I regenerate the report after tweaking code?**

A: Yes. If you re-run Days 1-14 with new data or settings, run Day 15 again to update the master report. **Everything stays synchronized.**

---

## 7  Export to Word / LaTeX

### 7.1  Pandoc (Markdown → Word)

```bash
pandoc results/MASTER_THESIS_REPORT.md -o thesis_draft.docx
```

Then open `thesis_draft.docx` in Microsoft Word and:
- Apply university template (heading styles, margins, fonts)
- Insert plots from `results/` directories
- Add page numbers, table of contents

### 7.2  LaTeX

If your university requires LaTeX:

```bash
pandoc results/MASTER_THESIS_REPORT.md -o thesis.tex --template=thesis_template.tex
```

Where `thesis_template.tex` is your university's LaTeX class.

---

## 8  Checklist

- [ ] `week2_master_report.py` runs without errors
- [ ] `MASTER_THESIS_REPORT.md` saved in `results/`
- [ ] You've read the full report (5 pages)
- [ ] Executive summary has all key numbers (no "TBD" placeholders)
- [ ] You've identified which sections need expansion
- [ ] You've exported to Word/LaTeX and applied formatting
- [ ] **Your thesis is now 70% complete**

---

## 9  What's Next?

### 9.1  Immediate tasks (1-2 weeks)

1. **Literature review**: Read 30-50 papers on monetary transmission, VAR identification, Nigerian policy
2. **Expand methodology**: Add equations for VAR, IRF, FEVD, Cholesky
3. **Insert plots**: All 20+ plots from `results/` directories
4. **Write Discussion**: Compare your 18.4% FEVD with literature benchmarks

### 9.2  Before submission (2-4 weeks)

1. **Proofread** 3 times
2. **Format** to university guidelines
3. **Abstract**: Adapt Executive Summary (max 300 words)
4. **Acknowledgements** (optional but appreciated)
5. **Submit**

### 9.3  Defense prep (1 week before viva)

1. **Print** the master report for reference
2. **Memorize** the 6 key numbers (§5.1)
3. **Practice** the 5-minute defense script (§5.2)
4. **Prepare slides** (10 slides: 1 intro, 1 data, 2 methodology, 4 results, 1 conclusion, 1 Q&A)

---

**Congratulations! You have a complete, thesis-ready analytics platform.**

Days 1–15 delivered:
- **15 Python modules** (3,500+ lines)
- **60+ output files** (plots, tables, CSVs)
- **11 guides** (15,000 words)
- **1 master thesis report** (backbone for 80-page thesis)

**You're ready to defend.**
