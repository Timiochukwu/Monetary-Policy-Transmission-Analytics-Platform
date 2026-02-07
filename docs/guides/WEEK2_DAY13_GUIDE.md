# Week 2 · Day 13 — Robustness Checks

> **What you will build:** `src/econometrics/robustness_checks.py`
> **What you will produce:** Lag-order comparison, ordering sensitivity, sub-sample analysis
> **Why it matters:** Examiners will ask: **"Are your results robust to alternative specifications?"**

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| Lag selection (AIC vs BIC) | Day 7 guide |
| Cholesky ordering | Day 7 + `docs/identification_strategy.md` |
| Sub-sample splits | Day 12 (stability tests) |

---

## 1  Why Robustness Matters

Your baseline specification (VAR(11), MPR-first ordering, full sample) is based on **choices**. Examiners want to know: "If I change one choice, do the results collapse?"

Robustness checks demonstrate that **key findings are not artifacts of arbitrary modeling decisions**.

---

## 2  Test 1: Alternative Lag Orders

### 2.1  The trade-off

- **Too few lags** (VAR(3)): Omitted variable bias, IRFs are wrong
- **Too many lags** (VAR(15)): Overfitting, noisy IRFs, fewer observations

AIC selected VAR(11). BIC selected VAR(3). What if we use VAR(6)?

### 2.2  What to check

Compare the **peak Inflation response to MPR shock** across VAR(3), VAR(6), VAR(11).

If they're all negative and similar in magnitude (e.g., −0.9 to −1.3 pp), the result is robust.

### 2.3  Code

```python
# robustness_checks.py — compare_lag_orders
for lag in [3, 6, 11]:
    var = VAR(self.df).fit(lag)
    irf_obj = var.irf(24)
    inf_response = irf_obj.orth_irfs[:, inf_idx, mpr_idx]
    peak_response = inf_response.min()
```

### 2.4  Output table

```
  Lag   AIC      BIC     Peak Inf Resp   Peak Month
   3    -0.968   -0.039     -0.9234         10
   6    -1.255    0.554     -1.1042         11
  11    -2.186    1.134     -1.2966         12  ◄ (baseline)
```

**Interpretation:** Peak response varies between −0.92 and −1.30 pp. All are economically significant and statistically negative. Result is robust.

---

## 3  Test 2: Alternative Cholesky Ordering

### 3.1  Why ordering matters

Cholesky imposes a **contemporaneous causal chain**. Variables ordered first absorb more variance mechanically.

**Baseline:** MPR → ExchangeRate → M2 → Inflation
- Justified by policy exogeneity (CBN sets MPR first)

**Alternative:** Inflation → ExchangeRate → M2 → MPR
- "Inflation-first" ordering (what if Inflation drives policy?)

### 3.2  What to check

Compare the **peak Inflation response to MPR shock** under both orderings.

If the alternative ordering produces a similar response, the result is not an artifact of the ordering.

### 3.3  Expected outcome

The peak response will differ slightly (Inflation-first ordering attributes more variance to Inflation's own shocks, less to MPR). But the **sign and order of magnitude** should be the same.

### 3.4  Code

```python
# robustness_checks.py — compare_orderings
# Baseline
df_baseline = self.df[["MPR", "ExchangeRate", "M2", "Inflation"]]
var_baseline = VAR(df_baseline).fit(11)
irf_baseline = var_baseline.irf(24)

# Alternative
df_alt = self.df[["Inflation", "ExchangeRate", "M2", "MPR"]]
var_alt = VAR(df_alt).fit(11)
irf_alt = var_alt.irf(24)
```

---

## 4  Test 3: Sub-Sample Analysis

### 4.1  The idea

Split at 2020-01-01 (pre-COVID vs post-COVID). Check if transmission strengthened or weakened.

### 4.2  Expected outcomes

| Scenario | Pre-2020 | Post-2020 | Interpretation |
|----------|----------|-----------|----------------|
| Similar  | −1.1 pp  | −1.3 pp   | Transmission stable over time |
| Weaker post | −1.5 pp | −0.7 pp | COVID weakened transmission (credit channel impaired) |
| Stronger post | −0.8 pp | −1.6 pp | CBN credibility improved post-COVID |

### 4.3  Code

```python
# robustness_checks.py — compare_subsamples
split_dt = pd.to_datetime("2020-01-01")
df_pre   = self.df.loc[:split_dt]
df_post  = self.df.loc[split_dt:]

var_pre  = VAR(df_pre).fit(11)
var_post = VAR(df_post).fit(11)
```

### 4.4  Caveat

Post-2020 sample has only ~60 observations. VAR(11) has 176 parameters. This is **overfit territory**. If the post-2020 VAR fails to converge, reduce lag to VAR(3) for the sub-sample only.

---

## 5  Run the Script

```bash
python src/econometrics/robustness_checks.py
```

### Output files

| File | Contents |
|------|----------|
| `results/robustness/lag_order_comparison.csv` | AIC, BIC, peak response for each lag |
| (Ordering comparison printed to console) | Peak response under both orderings |
| (Sub-sample comparison printed to console) | Peak response pre vs post 2020 |

---

## 6  Thesis Phrasing

### 6.1  If results are robust

> "I test robustness to three alternative specifications: (i) lag orders VAR(3) and VAR(6) (vs baseline VAR(11)); (ii) an Inflation-first Cholesky ordering (vs MPR-first); (iii) sub-sample estimation pre- and post-2020. The peak Inflation response to a 100 bps MPR shock ranges from −0.9 to −1.4 pp across all specifications (Table X), confirming that the transmission mechanism is robust to modeling choices."

### 6.2  If sub-sample shows large difference

> "Sub-sample analysis reveals that the transmission channel weakened post-2020 (peak response: −0.7 pp vs −1.5 pp pre-2020). This may reflect credit-market fragmentation during the pandemic, or the CBN's accommodative stance (holding MPR at 11.5% for 18 months). The full-sample result (−1.2 pp) is a weighted average of the two regimes."

---

## 7  Common Questions

**Q: Why not test every possible lag from 1 to 12?**

A: That's 12 tests → multiple-testing problem. Test only the extremes (AIC vs BIC selection) and one middle ground.

**Q: What if the Inflation-first ordering gives a positive peak response?**

A: This would be a **price puzzle** (Sims, 1992). It suggests the ordering is wrong, or a key variable (e.g., oil prices) is omitted. Discuss in your thesis and justify the MPR-first ordering using institutional knowledge.

**Q: The post-2020 VAR won't converge. What do I do?**

A: Reduce lag to VAR(3) for the post-2020 sub-sample only. Note this in a footnote: "Due to limited observations (60), the post-2020 VAR uses 3 lags instead of 11."

**Q: How many robustness checks are enough?**

A: These three (lag, ordering, sub-sample) are standard. If you want more:
- Exclude outliers (e.g., 2016-06, 2020-03)
- Add oil prices as exogenous
- Use sign restrictions instead of Cholesky

---

## 8  Examiner Defense Script

**Examiner:** "Your results depend on using VAR(11). What if AIC is wrong?"

**You:** "I test VAR(3) and VAR(6) as alternatives. Peak responses range from −0.92 to −1.30 pp, all statistically significant. The economic conclusion (negative transmission) is unchanged. AIC is preferred for IRF analysis because it captures longer dynamics."

**Examiner:** "Why MPR first? Nigeria has high inflation — shouldn't Inflation drive policy?"

**You:** "The MPC sets MPR based on lagged data (published 2 weeks before the meeting). Within-month, MPR cannot respond to contemporaneous Inflation shocks. This justifies the ordering. I also test an Inflation-first ordering as robustness — the peak response is [X] pp, slightly smaller but still negative."

---

## 9  Checklist

- [ ] `robustness_checks.py` runs without errors
- [ ] Lag-order comparison shows similar peak responses (within 0.5 pp)
- [ ] Ordering comparison confirms negative transmission in both cases
- [ ] Sub-sample analysis shows transmission in both regimes
- [ ] You can defend any large differences (e.g., COVID weakened channel)
- [ ] Ready for Day 14: Forecast evaluation
