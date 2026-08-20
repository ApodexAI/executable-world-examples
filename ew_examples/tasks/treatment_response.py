"""Predict treatment response from pre-treatment data only.

The shape: two cohorts. Cohort A has pre-treatment measurements, some POST-treatment
measurements, and the outcome. Cohort B has the pre-treatment measurements only, and
is what you are scored on. You choose which columns to build a model from; the task
fits it and reports how it does.

The trap is the whole task, and it is the most expensive mistake in applied
prediction work:

  THE STRONGEST PREDICTORS IN THE TABLE ARE THE ONES YOU MAY NOT USE.

The POST_* columns are measured after treatment starts. They correlate beautifully
with the outcome, because they are partly consequences of it. Using them produces a
model with a superb fit and no value whatsoever: at the moment you would need to
make a prediction, those values do not exist yet. They are also absent from cohort
B, which is the point at which a leaky model stops being merely wrong and becomes
unrunnable.

Nothing here is real patient data. The cohort is generated from a seed, so it is
identical on every machine and involves no dataset at all.
"""
from __future__ import annotations

import random

from ..engine import Action, ActionError, Budget, Task

# Pre-treatment, and legitimate to model from.
ALLOWED = ("age", "bmi_base", "hba1c", "sex_male", "genetic_risk_score",
           "prior_weight_loss_attempts", "egfr")
# Measured AFTER treatment begins. Present in cohort A, absent from cohort B,
# forbidden as model inputs. They are here because a real extract contains them.
FORBIDDEN = ("POST_early_weight_change_4w", "POST_gi_side_effect_severity",
             "POST_adherence_proxy")
OUTCOME = "delta_bmi_percent"


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting. Small systems only.

    Written out rather than pulled from a library so the pack keeps its promise of
    needing nothing installed. Ridge-regularised by the caller, which also keeps
    this stable when two chosen columns are near-duplicates.
    """
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            continue
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [(m[i][n] / m[i][i]) if abs(m[i][i]) > 1e-12 else 0.0 for i in range(n)]


def _fit(rows: list[dict], feats: list[str], lam: float = 1e-3) -> list[float]:
    """Ridge least squares with an intercept. Returns [b0, b1, ...]."""
    x = [[1.0] + [float(r[f]) for f in feats] for r in rows]
    y = [float(r[OUTCOME]) for r in rows]
    k = len(feats) + 1
    xtx = [[sum(x[i][a] * x[i][c] for i in range(len(x))) for c in range(k)]
           for a in range(k)]
    for i in range(1, k):
        xtx[i][i] += lam * len(x)
    xty = [sum(x[i][a] * y[i] for i in range(len(x))) for a in range(k)]
    return _solve(xtx, xty)


def _r2(rows: list[dict], feats: list[str], beta: list[float]) -> float:
    y = [float(r[OUTCOME]) for r in rows]
    mean = sum(y) / len(y)
    pred = [beta[0] + sum(beta[i + 1] * float(r[f]) for i, f in enumerate(feats))
            for r in rows]
    ss_res = sum((a - b) ** 2 for a, b in zip(y, pred))
    ss_tot = sum((a - mean) ** 2 for a in y)
    return round(1.0 - ss_res / ss_tot, 4) if ss_tot else 0.0


class TreatmentResponse(Task):
    task_id = "treatment_response"
    brief = f"""\
Predict six-month treatment response, and make it work on a cohort you have never
seen.

  cohort A  (train + validation)  pre-treatment columns, POST_* columns, and the
                                  outcome {OUTCOME}
  cohort B  (scored)              pre-treatment columns ONLY. No outcome, and no
                                  POST_* columns at all.

You do not write a model. You choose the COLUMNS, and `fit_report` fits a
ridge-regularised linear model on cohort A and tells you how it does. Then you
submit your chosen columns and are scored on cohort B.

  allowed    {', '.join(ALLOWED)}
  forbidden  {', '.join(FORBIDDEN)}

WHY THE FORBIDDEN ONES ARE THERE, AND WHY THIS IS A GATE
They were measured AFTER treatment began, so they are partly consequences of the
outcome. They will look like your best predictors — `correlation` will say so
plainly. They are also absent from cohort B entirely. A model built on them has a
superb fit and no value: at the moment you would actually need a prediction, those
numbers do not exist yet.

Submitting any forbidden column scores ZERO. Not a penalty — zero. This is the one
mistake in applied prediction work that is both easy and total.

SCORED against a ladder of baselines, so the number means something:
  mean-only, clinical-only, clinical+genetic, and every allowed column.
Beating 'every allowed column' means you found which columns actually carry signal.

ACTIONS
  inspect_data()                -> schema, sizes, which columns are which   [free]
  sample_rows(split, n<=20)     -> actual rows                             [1 query]
  correlation(column)           -> that column's correlation with the outcome  [1 query]
  fit_report(features)          -> train and validation R² for your choice  [1 fit]
  submit(features)              -> fits on all of A, scores on B; ends the episode
"""

    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        self._rng = rng

        def make(n: int, external: bool) -> list[dict]:
            out = []
            for _ in range(n):
                age = rng.gauss(52, 11)
                bmi = rng.gauss(34, 5)
                hba1c = rng.gauss(7.1, 0.9)
                male = 1.0 if rng.random() < 0.45 else 0.0
                grs = rng.gauss(0, 1)
                attempts = float(rng.randint(0, 5))
                egfr = rng.gauss(88, 14)
                # The truth: response depends on pre-treatment state only.
                y = (-3.0
                     - 0.18 * (bmi - 34)
                     - 1.1 * grs
                     + 0.9 * (hba1c - 7.1)
                     + 0.04 * (age - 52)
                     + 0.35 * male
                     + rng.gauss(0, 2.2))
                row = {"age": round(age, 1), "bmi_base": round(bmi, 1),
                       "hba1c": round(hba1c, 2), "sex_male": male,
                       "genetic_risk_score": round(grs, 3),
                       "prior_weight_loss_attempts": attempts,
                       "egfr": round(egfr, 1), OUTCOME: round(y, 3)}
                # POST_* are functions OF the outcome, which is exactly why they
                # correlate so well and are worthless for prediction.
                if not external:
                    row["POST_early_weight_change_4w"] = round(
                        0.42 * y + rng.gauss(0, 0.7), 3)
                    row["POST_gi_side_effect_severity"] = round(
                        max(0.0, -0.30 * y + rng.gauss(0, 1.1)), 3)
                    row["POST_adherence_proxy"] = round(
                        min(1.0, max(0.0, 0.5 - 0.06 * y + rng.gauss(0, 0.12))), 3)
                out.append(row)
            return out

        rows = make(600, external=False)
        self._train, self._val = rows[:450], rows[450:]
        self._external = make(300, external=True)   # cohort B: no POST_*, but we
        # keep the outcome host-side to score against.

    # ---- baselines --------------------------------------------------------
    def _baselines(self) -> dict:
        sets = {
            "mean_only": [],
            "clinical_only": ["age", "bmi_base", "hba1c", "sex_male"],
            "clinical_plus_genetic": ["age", "bmi_base", "hba1c", "sex_male",
                                      "genetic_risk_score"],
            "all_allowed": list(ALLOWED),
        }
        out = {}
        for name, feats in sets.items():
            if not feats:
                y = [r[OUTCOME] for r in self._external]
                mean = sum(r[OUTCOME] for r in self._train) / len(self._train)
                ss_res = sum((v - mean) ** 2 for v in y)
                m = sum(y) / len(y)
                ss_tot = sum((v - m) ** 2 for v in y)
                out[name] = round(1.0 - ss_res / ss_tot, 4)
            else:
                beta = _fit(self._train + self._val, feats)
                out[name] = _r2(self._external, feats, beta)
        return out

    # ---- actions ----------------------------------------------------------
    def initial_budget(self) -> Budget:
        return Budget({"queries": 20, "fits": 8})

    def actions(self) -> dict[str, Action]:
        return {
            "inspect_data": Action(0, self._inspect, "schema, sizes, column roles"),
            "sample_rows": Action(1, self._rows, "actual rows from a split"),
            "correlation": Action(1, self._corr, "a column's correlation with the outcome"),
            "fit_report": Action(1, self._fit_report, "train and validation R²"),
            "submit": Action(0, self._submit, "score on cohort B; ends the episode"),
        }

    def _inspect(self, ep, p):
        return {
            "cohort_a": {"train_rows": len(self._train), "val_rows": len(self._val),
                         "columns": list(ALLOWED) + list(FORBIDDEN) + [OUTCOME]},
            "cohort_b": {"rows": len(self._external), "columns": list(ALLOWED),
                         "note": "no outcome, and no POST_* columns exist here"},
            "allowed_columns": list(ALLOWED),
            "forbidden_columns": list(FORBIDDEN),
            "outcome": OUTCOME,
            "note": "POST_* columns were measured after treatment began. They are "
                    "provided because a real extract contains them, and they are "
                    "forbidden as model inputs.",
        }

    def _split(self, name: str) -> list[dict]:
        s = {"train": self._train, "val": self._val,
             "validation": self._val}.get(str(name or "train"))
        if s is None:
            raise ActionError("split must be 'train' or 'val' (cohort B rows are "
                              "not shown; that is what you are scored on)")
        return s

    def _rows(self, ep, p):
        rows = self._split(p.get("split", "train"))
        n = max(1, min(20, int(p.get("n", 5))))
        ep.budget.spend("queries", 1)
        return {"split": p.get("split", "train"), "rows": rows[:n]}

    def _corr(self, ep, p):
        col = str(p.get("column", ""))
        if col not in ALLOWED + FORBIDDEN:
            raise ActionError(f"unknown column {col!r}")
        ep.budget.spend("queries", 1)
        rows = self._train
        xs = [float(r[col]) for r in rows]
        ys = [float(r[OUTCOME]) for r in rows]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        vx = sum((a - mx) ** 2 for a in xs) ** 0.5
        vy = sum((b - my) ** 2 for b in ys) ** 0.5
        r = round(cov / (vx * vy), 4) if vx and vy else 0.0
        out = {"column": col, "correlation_with_outcome": r}
        if col in FORBIDDEN:
            out["warning"] = (
                "this column is FORBIDDEN as a model input. A strong correlation "
                "here is evidence that it is downstream of the outcome, not that "
                "it is useful. It does not exist in cohort B.")
        return out

    def _features(self, p) -> list[str]:
        raw = p.get("features") or p.get("columns") or []
        if isinstance(raw, str):
            raw = [raw]
        feats = [str(f) for f in raw]
        if not feats:
            raise ActionError('pass features=["bmi_base", ...]')
        unknown = [f for f in feats if f not in ALLOWED + FORBIDDEN]
        if unknown:
            raise ActionError(f"unknown columns: {unknown}")
        if len(feats) != len(set(feats)):
            raise ActionError("duplicate columns in the feature list")
        return feats

    def _fit_report(self, ep, p):
        feats = self._features(p)
        ep.budget.spend("fits", 1)
        beta = _fit(self._train, feats)
        out = {"features": feats,
               "train_r2": _r2(self._train, feats, beta),
               "val_r2": _r2(self._val, feats, beta)}
        leaks = [f for f in feats if f in FORBIDDEN]
        if leaks:
            out["warning"] = (
                f"this fit uses forbidden columns {leaks}. The R² above is real and "
                f"also meaningless: those columns are absent from cohort B, so this "
                f"model cannot be evaluated there at all. Submitting it scores zero.")
        return out

    def _submit(self, ep, p):
        feats = self._features(p)
        res = ep.finish(feats)
        return {"status": "submitted", "episode_complete": True,
                "score": res["score"], "gate_passed": res["gate_passed"],
                "feedback": res["feedback"]}

    # ---- scoring ----------------------------------------------------------
    def score(self, feats) -> dict:
        feats = list(feats or [])
        leaks = [f for f in feats if f in FORBIDDEN]
        base = self._baselines()
        fb: list[str] = []

        if leaks:
            return {"score": 0.0, "gate_passed": False, "leaked_columns": leaks,
                    "external_r2": None, "baselines": base,
                    "feedback": [
                        f"GATE FAILED: {leaks} are post-treatment columns. The model "
                        f"cannot even be applied to cohort B, where they do not "
                        f"exist. `correlation` said so, and `fit_report` warned "
                        f"about it.",
                        "This is the whole task. A model whose best feature is a "
                        "consequence of the thing it predicts is not a model."]}

        beta = _fit(self._train + self._val, feats)
        ext = _r2(self._external, feats, beta)
        ladder = base["all_allowed"]
        floor = base["clinical_only"]
        # Normalised against the ladder: 1.0 means matching the every-allowed-column
        # baseline, above 1.0 means beating it by dropping columns that add noise.
        denom = ladder - base["mean_only"]
        norm = (ext - base["mean_only"]) / denom if denom else 0.0
        score = round(max(0.0, min(1.2, norm)), 4)

        fb.append(f"external R² {ext} against baselines mean={base['mean_only']}, "
                  f"clinical={floor}, clinical+genetic="
                  f"{base['clinical_plus_genetic']}, all_allowed={ladder}")
        if ext > ladder:
            fb.append("beat the all-columns baseline — dropping columns that carry "
                      "no signal is the actual skill here")
        elif ext < floor:
            fb.append("below the clinical-only baseline: the chosen columns are "
                      "carrying less than four obvious ones would")
        dropped = [f for f in ALLOWED if f not in feats]
        if dropped:
            fb.append(f"columns you left out: {dropped}")
        return {"score": score, "gate_passed": True, "external_r2": ext,
                "baselines": base, "n_features": len(feats), "feedback": fb}
