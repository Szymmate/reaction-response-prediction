# ReAction: Behavioral Response Prediction Challenge

## Background

A mid-sized European SaaS company runs an internal Learning & Development (L&D) platform.
Employees are enrolled in short skill-building modules (negotiation, public speaking, conflict
resolution, technical writing, leadership). For each upcoming module, the platform recommends
one of five *intervention types*:

- `peer_workshop` — group practice session
- `self_paced_video` — async video course
- `coaching_1on1` — one-to-one coaching call
- `simulation_exercise` — scenario-based role play
- `reading_pack` — curated article / book set

The company has gathered three sources of behavioral data during employee onboarding and a
voluntary wellness program:

1. **Big Five work-engagement psychometrics** — 24-item scale administered at onboarding
2. **Big Five personality composites** — abbreviated OCEAN scores
3. **Wearable aggregates** — 30-day rolling means for employees who opted in to the program
   (HRV, sleep variability, steps, active minutes)

For approximately **5 % of past situations**, an outcome was recorded: how the employee
engaged with the recommended intervention (`response_category`). For the remaining 95 %, the
intervention was delivered but no feedback was collected.

**Product goal:** given an employee's behavioral profile and the situational context of an
upcoming L&D event, predict the probability distribution over the five `response_category`
outcomes. This prediction will inform intervention matching — flagging cases where the default
recommendation is unlikely to succeed and surfacing alternatives. Because this operates in an
EU HR context, calibrated probabilities (not just rankings) are required from day one.

---

## Dataset

Three parquet files are provided in the `data/` directory. To read them:

```bash
pip install pandas pyarrow
# or: uv pip install pandas pyarrow
```

```python
import pandas as pd
persons    = pd.read_parquet("data/persons.parquet")
situations = pd.read_parquet("data/situations.parquet")
responses  = pd.read_parquet("data/responses.parquet")
```

### Files

| File | Rows | Description |
|---|---|---|
| `data/persons.parquet` | ~20 000 | One row per employee |
| `data/situations.parquet` | ~120 000 | One row per past L&D event (~6 per employee) |
| `data/responses.parquet` | ~6 000 | Labeled subset: observed engagement outcome |

### Schema: `persons.parquet`

| Column | Type | Notes |
|---|---|---|
| `person_id` | string | Primary key |
| `age` | float | 22–64; ~2 % missing |
| `tenure_months` | float | 0–300; ~3 % missing |
| `job_family` | string | One of 8 families (engineering, sales, operations, product, marketing, finance, hr, legal) |
| `region` | string | DE / FR / PL / ES / NL |
| `ocean_O` … `ocean_N` | float | Big Five z-scores; ~4–25 % missing (see distribution) |
| `engagement_q01` … `engagement_q24` | float | Work-engagement items, Likert 1–5; ~6–18 % missing |
| `burnout_composite` | float | Derived burnout index 0–1; ~5–55 % missing (see distribution) |
| `wellness_optin` | bool | Whether employee joined the wearable program |
| `wearable_hrv_mean_30d` | float | Mean HRV over 30 d; absent when `wellness_optin = False` |
| `wearable_sleep_var_30d` | float | Sleep variability (higher = worse); same |
| `wearable_steps_mean_30d` | float | Mean daily steps; same |
| `wearable_active_min_30d` | float | Mean active minutes/day; same |

> Note: wearable columns are absent for ~55 % of employees (non-opt-ins). This is a structural
> absence, not a stochastic missing-data pattern.

### Schema: `situations.parquet`

| Column | Type | Notes |
|---|---|---|
| `situation_id` | string | Primary key |
| `person_id` | string | FK → persons |
| `module_topic` | string | One of 5 topics |
| `recommended_intervention` | string | One of 5 interventions |
| `context_domain` | string | `formal_training` / `on_the_job` / `external_event` |
| `workload_index` | float | 0–1, recent calendar density |
| `team_size` | int | Team size at time of event |
| `manager_support_score` | float | Manager survey score 1–5; ~15 % missing |
| `season` | string | Q1–Q4 |
| `ctx_feat_001` … `ctx_feat_050` | float | Pre-computed context features |

### Schema: `responses.parquet`

| Column | Type | Notes |
|---|---|---|
| `situation_id` | string | FK → situations |
| `response_category` | string | `completed_effective` / `completed_ineffective` / `partial` / `dropped_out` / `declined` |
| `observed_at` | datetime | Timestamp of outcome collection (typically 4–8 weeks after event) |

---

## Tasks

You have four tasks. Each is stated as a **goal**, not a recipe. We are evaluating your
judgment about *what* to do and *why*, not your ability to follow instructions.

---

### Task 1 — Data Audit & Representation

**Goal.** Build a documented feature pipeline that handles the three data modalities and
produces a per-situation representation (person × situational context) suitable for predicting
`response_category`. The prediction target is per-situation, not per-person — your pipeline
should reflect how person-level and situation-level features combine at inference time.

Your submission should include:

- A written audit (~400–600 words) covering distributions, correlations, and anything you found
  surprising or concerning about the data — including your assessment of missing-data patterns.
- The pipeline code with comments explaining each non-obvious design choice.
- A list of questions you would ask the data collection team before committing to this
  representation in production.

---

### Task 2 — Response Prediction

**Goal.** Build a model that outputs calibrated `response_category` probability distributions
for unlabeled employees in new situations.

Requirements:

- Output must be probability distributions over all five categories (not just predicted labels).
- The labeled set covers ~5 % of situations. Justify your training-data strategy given this ratio
  — specifically, what role the unlabeled 95 % plays in your approach and why.
- Your evaluation must include a comparison that lets us assess whether your chosen
  training-data strategy actually improved over a supervised-only baseline.

Deliverable: a reproducible training script or notebook plus held-out evaluation results.

---

### Task 3 — Evaluation

**Goal.** Evaluate your model in a form that an EU-facing deployment review board would accept.

Your report should cover:

- Calibration and discrimination reported separately, with at least one calibration plot.
- Performance sliced by `age` cohort and `context_domain`.
- An explicit assessment of whether the labeled set is a representative sample of the full
  population of situations, and what implications this has for your reported metrics.
- A deployment recommendation: is the model ready to ship, and under what conditions?

---

### Task 4 — Deployment Design Memo

**Goal.** Write a concise (~400–600 word) technical memo describing how you would deploy and
operate this system inside the EU SaaS company.

Cover at minimum: inference latency and serving architecture, monitoring signals you would
track after launch, how you handle the fact that outcome labels arrive 4–8 weeks after
inference, and what feedback-loop risks exist. Include any GDPR or EU AI Act considerations you
consider load-bearing.

No code is required for this task.

---

## Deliverables

Submit as a git repository or zip archive containing:

```
README.md               — setup instructions; how to reproduce every result
data/                   — the provided parquet files (do not modify)
notebooks/ or src/      — your code for Tasks 1–2
report.md or report.pdf — written responses for Tasks 1 (audit), 3, and 4
```

---

## Notes

- **Time budget.** Max 8 hours.
- **Libraries.** Any Python library is permitted. We are not testing library knowledge.
- **LLM assistance.** You may use LLMs to help with coding. Disclose this in your README.
- **The 50 context features.** They are real columns; you decide whether and how to use them.
- **No benchmark.** There is no external leaderboard or target metric to beat. We evaluate
  reasoning and rigor, not whether you hit a specific number.
- **Language.** English or Polish are both accepted.
