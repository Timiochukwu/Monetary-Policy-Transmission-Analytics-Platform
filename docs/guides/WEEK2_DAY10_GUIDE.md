# Week 2 · Day 10 — Policy Shock Simulation (+100 bps MPR)

> **What you will build:** `src/econometrics/policy_simulation.py`
> **What you will produce:** Policy brief, comparison plot, scaled IRF table
> **Why it matters:** This translates your IRF into **actionable policy numbers** for the CBN.

---

## 0  Pre-requisites

| Concept | Where covered |
|---------|---------------|
| IRF (Day 8) | The 1 SD shock that we're scaling |
| Cholesky decomposition | Day 7 + Day 8 guides |
| Shock standard deviation | Diagonal of Cholesky factor |

---

## 1  The Scaling Problem

The Day 8 IRF shows the response to a **+1 standard deviation shock**. But policymakers don't think in standard deviations — they think in **basis points**.

When the CBN raises MPR by 100 basis points (1 percentage point), what happens?

### 1.1  The scaling formula

```
Shock SD              =  P[0,0]   (from Cholesky decomposition of Σ_u)
Scaling factor        =  (100 / 100) / Shock SD  =  1.0 / Shock SD
Scaled IRF (100 bps)  =  IRF (1 SD)  ×  scaling factor
```

In the Nigerian data, the MPR shock SD ≈ 0.40 pp, so:
```
Scaling factor  =  1.0 / 0.40  =  2.50
```

This means a 100 bps shock is **2.5 times larger** than a 1 SD shock.

---

## 2  Code Walkthrough

### 2.1  Extract shock SD from VAR

```python
# policy_simulation.py — compute_scaling_factor
cov_u = self.res.sigma_u              # covariance of reduced-form errors
P     = np.linalg.cholesky(cov_u)     # Cholesky: P P' = cov_u

mpr_idx       = self.ordering.index("MPR")
self.shock_sd = P[mpr_idx, mpr_idx]   # diagonal element = shock SD
```

### 2.2  Scale the IRFs

```python
# Extract IRFs for MPR shock (column mpr_idx)
irfs_mpr = self.irf_obj.orth_irfs[:, :, mpr_idx]   # shape: (T+1, K)

# Scale by the factor
self.scaled_irfs = irfs_mpr * self.scaling_factor
```

Now `scaled_irfs[12, inf_idx]` gives the Inflation response at month 12 to a **100 bps** shock.

---

## 3  Reading the Output

### 3.1  Comparison plot

The plot shows two lines for each variable:
- **Blue (solid)**: +1 SD shock (±0.40 pp)
- **Orange (dashed)**: +100 bps shock (±1.00 pp)

Focus on the **Inflation panel** (bottom-right):
- Peak response: **−1.21 pp at month 12**
- Interpretation: A 100 bps MPR hike reduces Inflation by 1.21 percentage points after one year

### 3.2  Policy brief

The script generates a Markdown document (`policy_brief_100bps.md`) with:

**Key numbers auto-extracted:**
- Impact effect (month 0)
- 3-month, 6-month, 12-month responses
- Peak response and timing
- Exchange-rate impact

**Recommendations section:**
- If current Inflation is 15% and target is 7.5%, the brief calculates how many 100 bps hikes are needed
- Timing guidance (peak effect at month 12 → plan ahead)
- Coordination with FX policy

---

## 4  Run the Script

```bash
cd /path/to/Monetary-Policy-Transmission-Analytics-Platform
python src/econometrics/policy_simulation.py
```

### Output files

| File | Contents |
|------|----------|
| `results/policy_simulation/policy_shock_100bps_comparison.png` | 4-panel: 1 SD vs 100 bps |
| `results/policy_simulation/policy_shock_100bps_responses.csv` | Scaled IRF table (all horizons) |
| `results/policy_simulation/policy_brief_100bps.md` | CBN-ready policy brief |

---

## 5  Using the Policy Brief

### 5.1  In your thesis

Excerpt from the brief in your Results chapter:

> "Based on the VAR(11) simulation, a 100 basis point increase in MPR is expected to reduce headline inflation by 1.21 percentage points after 12 months, with the maximum impact occurring at month 12. This transmission lag reflects price stickiness in Nigeria's consumer markets and the time required for credit-channel effects to propagate."

### 5.2  For the CBN

Print the policy brief and include it as an **Appendix** in your thesis. Some examiners appreciate seeing the "policy deliverable."

If you do an internship at the CBN's Research Department, this brief is the exact format they use for MPC briefings.

---

## 6  Common Questions

**Q: Why is the scaling factor constant across all variables?**

A: The shock size (100 bps to MPR) is fixed. The responses of other variables (ExchangeRate, M2, Inflation) scale proportionally because the VAR is linear.

**Q: The Inflation response is negative. Does that mean MPR and Inflation move in opposite directions?**

A: Yes. A **contractionary** MPR hike (positive shock) reduces Inflation (negative response). This is the expected transmission mechanism.

**Q: What if I want to simulate a 50 bps or 200 bps shock?**

A: Change `policy_shock_bps` in the script:
```python
sim.run_full_simulation(policy_shock_bps=50)   # or 200
```

**Q: The brief says "multiple 100 bps hikes are needed." How many?**

A: If current Inflation is 15% and target is 7.5%, you need to close a 7.5 pp gap. Since one hike delivers −1.21 pp, you'd need:
```
7.5 / 1.21  ≈  6.2  →  7 consecutive hikes
```

But this is simplistic — the effects compound, and each hike takes 12 months to peak. The CBN would space them out (e.g., 100 bps every 2 months).

---

## 7  Thesis Phrasing

> "I simulate a 100 basis point monetary policy tightening using the orthogonalized impulse response functions scaled by the empirical shock standard deviation (0.40 percentage points). The simulation predicts a cumulative inflation reduction of 1.21 percentage points at the 12-month horizon, consistent with the quantity-theory and exchange-rate pass-through channels identified in the variance decomposition."

---

## 8  Checklist

- [ ] `policy_simulation.py` runs without errors
- [ ] Policy brief saved in `results/policy_simulation/`
- [ ] You understand the scaling factor (1.0 / shock_sd)
- [ ] You can state the **headline number**: −1.21 pp at 12 months
- [ ] Ready for Day 11: Historical shock decomposition
