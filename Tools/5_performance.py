"""Print the most up-to-date performance metrics for every model head.

Reads the artifacts the daily retrain refreshes and prints, for each head, all
of the metrics that head has - logloss, AUC, ECE, Brier, edge, accuracy and
top-10/day for the probability heads; MAE, dispersion and bias for the
count/regression heads. The same report is saved to Tools/performance.txt.

Both evaluation suites are shown side by side:

  SELECTION  scored on its test year (the iteration surface - the numbers a
             feature/param change is judged on).   metrics_select.json
  SHIPPING   the newest season, scored by the models that actually serve
             today.   metrics.json

Neither column is an untouched test (2026-07-14 data-role decision): 2025 is
the decision year (it carries calibration + feature-selection information) and
2026 is a partial season that a handful of past decisions have already seen.
Read both as UPPER-BOUND estimates. The only leakage-free read on the board is
the graded forward record - `python Tools/4_grade_results.py --all`.

Sources (all rewritten by the 06:00 pipeline - train.py, then evaluate_deep.py
--set-baseline, so these numbers are always current as of the last train):

  metrics.json / metrics_select.json      full per-head metrics from train.py
                                          (logloss, Brier, AUC, acc, the decile
                                          calibration table, top-10/day, MAE,
                                          means, cal-year dispersion)
  eval_baseline_2026.json /               condensed north-star snapshots from
  eval_baseline_select_2025.json          evaluate_deep --set-baseline - supply
                                          total MAE and the observed test-year
                                          dispersion that metrics.json omits
  sim_grade_2025.csv / sim_grade_2026.csv the SIM_BLEND-vs-incumbent shoot-out
                                          per head (printed as an appendix)

ECE is computed here from the metrics.json decile calibration table
(sum n*|pred-actual| / sum n - the same formula evaluate_deep uses) and edge
from logloss_baserate - logloss, so every derived number is consistent with the
logloss/Brier printed next to it. No model or data frame is loaded.

Usage:
    python Tools/5_performance.py
    python Tools/5_performance.py --out path\\to\\file.txt   # override output
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "Model" / "artifacts"
DEFAULT_OUT = ROOT / "Tools" / "performance.txt"

# metrics.json (SHIPPING) is written first, eval_baseline_2026 right after; the
# selection pair mirrors them on the decision year.
SUITES = [
    ("SELECTION", "metrics_select.json", "eval_baseline_select_2025.json"),
    ("SHIPPING", "metrics.json", "eval_baseline_2026.json"),
]
SIM_GRADE = {"SELECTION": "sim_grade_2025.csv",
             "SHIPPING": "sim_grade_2026.csv"}

# friendly head labels (train.PROPS + the count/game heads); unknown heads fall
# back to their raw key, so a head added by a future retrain still prints.
LABELS = {
    "hr": "home run", "hit": "1+ hit", "hits2": "2+ hits",
    "tb2": "2+ total bases", "run": "run scored", "rbi": "1+ RBI",
    "bb": "1+ walk", "sb": "stolen base", "single": "1+ single",
    "double": "1+ double", "bk": "1+ batter K", "bk2": "2+ batter K",
    "hrr2": "2+ H+R+RBI", "hrr3": "3+ H+R+RBI", "bk3": "3+ batter K",
    "tb3": "3+ total bases", "tb4": "4+ total bases", "hrr4": "4+ H+R+RBI",
    "triple": "1+ triple", "rbi2": "2+ RBI", "run2": "2+ runs",
    "xbk": "batter Ks (mean)", "xhrr": "H+R+RBI (mean)",
    "xtb": "total bases (mean)", "xh": "hits (mean)", "xrun": "runs (mean)",
    "xrbi": "RBIs (mean)", "xbb": "walks (mean)", "k": "starter strikeouts",
    "outs": "starter outs", "pbb": "starter walks",
    "pha": "starter hits allowed", "per": "starter earned runs",
    "team_runs": "team runs (per-team mean)", "total": "game total runs",
    "team_total": "team total runs", "winner": "home-win probability",
    "score": "per-team score (sim)",
}

# display groups + order; anything unseen is appended under "Other heads".
GROUPS = [
    ("Batter - probability heads",
     ["hr", "hit", "hits2", "tb2", "run", "rbi", "bb", "sb", "single",
      "double", "bk", "bk2", "hrr2", "hrr3", "bk3", "tb3", "tb4", "hrr4",
      "triple", "rbi2", "run2"]),
    ("Batter - count / mean heads",
     ["xbk", "xhrr", "xtb", "xh", "xrun", "xrbi", "xbb"]),
    ("Pitcher heads", ["k", "outs", "pbb", "pha", "per"]),
    ("Game heads", ["team_runs", "total", "team_total", "winner", "score"]),
]

# per-head detail rows: (merged key, label). Only rows a head actually has are
# printed, so probability heads and count heads each show only their own set.
DETAIL = [
    ("n_test", "n (test rows)"),
    ("base_rate", "base rate"),
    ("logloss", "logloss                (lower better)"),
    ("logloss_baserate", "  base-rate logloss"),
    ("logloss_naive_seasonrate", "  naive season-rate logloss"),
    ("edge", "  edge = base - model   (higher better)"),
    ("auc", "AUC                    (higher better)"),
    ("brier", "Brier                  (lower better)"),
    ("brier_baserate", "  base-rate Brier"),
    ("brier_naive_seasonrate", "  naive season-rate Brier"),
    ("ece", "ECE                    (lower better)"),
    ("acc", "accuracy"),
    ("acc_home_baseline", "  always-home accuracy"),
    ("top10_daily_hit_rate", "top-10/day hit rate    (higher better)"),
    ("mae", "MAE                    (lower better)"),
    ("mae_baseline", "  baseline MAE"),
    ("mae_gain", "  MAE gain=base-model   (higher better)"),
    ("mean_actual", "mean actual"),
    ("mean_pred", "mean predicted"),
    ("bias", "  bias = pred - actual"),
    ("dispersion_cal", "dispersion cal-yr model (1.0 ideal)"),
    ("dispersion_obs", "dispersion observed     (1.0 ideal)"),
    ("best_iter", "best iteration"),
    ("blend_gbm_weight", "GBM blend weight"),
    ("blend_ml_weight", "ML blend weight"),
    ("n_train", "n (train rows)"),
]

# scalars copied straight from a metrics.json head record
PASS_THROUGH = (
    "n_train", "n_test", "base_rate", "best_iter", "blend_gbm_weight",
    "blend_ml_weight", "auc", "acc", "logloss", "logloss_baserate",
    "brier", "brier_baserate", "top10_daily_hit_rate", "mae", "mae_baseline",
    "mean_actual", "mean_pred", "dispersion_cal", "acc_home_baseline",
    "logloss_naive_seasonrate", "brier_naive_seasonrate",
)


def ece_from_bins(bins):
    """Expected calibration error from the stored decile table: the pick-
    weighted mean |predicted - actual|, the same number evaluate_deep.ece()
    computes live (the bins ARE its qcut deciles)."""
    tot = sum(b["n"] for b in bins)
    if not tot:
        return None
    return sum(b["n"] * abs(b["pred"] - b["actual"]) for b in bins) / tot


def load_suite(name, metrics_name, baseline_name):
    """Merge one suite's metrics.json + eval_baseline into per-head records."""
    mpath, bpath = ART / metrics_name, ART / baseline_name
    mheads, loose = {}, {}
    years = []
    if mpath.exists():
        for k, v in json.loads(mpath.read_text()).items():
            m = re.match(r"^(.+)_(\d{4})$", k)
            if isinstance(v, dict) and m:          # a per-head record
                mheads[m.group(1)] = v
                years.append(int(m.group(2)))
            else:                                  # loose scalar (e.g. dispersion)
                loose[k] = v
    # loose "{head}_dispersion_YYYY" is that head's cal-year model dispersion
    cal_disp = {}
    for k, v in loose.items():
        m = re.match(r"^(.+)_dispersion_\d{4}$", k)
        if m and isinstance(v, (int, float)):
            cal_disp[m.group(1)] = v

    bheads = defaultdict(dict)
    if bpath.exists():
        for k, v in json.loads(bpath.read_text()).items():
            head, metric = k.rsplit("_", 1)        # metric names carry no "_"
            bheads[head][metric] = v

    heads = {}
    for h in set(mheads) | set(bheads) | set(cal_disp):
        rec, b = mheads.get(h, {}), bheads.get(h, {})
        merged = {key: rec[key] for key in PASS_THROUGH if key in rec}
        # derived, straight off metrics.json so they match the logloss/Brier
        # printed beside them
        cal = rec.get("calibration")
        if cal:
            merged["ece"] = ece_from_bins(cal)
        if "logloss" in rec and "logloss_baserate" in rec:
            merged["edge"] = rec["logloss_baserate"] - rec["logloss"]
        if "mae" in rec and "mae_baseline" in rec:
            merged["mae_gain"] = rec["mae_baseline"] - rec["mae"]
        if "mean_pred" in rec and "mean_actual" in rec:
            merged["bias"] = rec["mean_pred"] - rec["mean_actual"]
        # eval_baseline fills what metrics.json lacks (total's MAE, the observed
        # test-year dispersion; and any binary field for an odd partial record)
        for bk, mk in (("mae", "mae"), ("dispersion", "dispersion_obs"),
                       ("ece", "ece"), ("edge", "edge"), ("auc", "auc"),
                       ("acc", "acc"), ("logloss", "logloss"),
                       ("top10", "top10_daily_hit_rate")):
            if bk in b and mk not in merged:
                merged[mk] = b[bk]
        if h in cal_disp and "dispersion_cal" not in merged:
            merged["dispersion_cal"] = cal_disp[h]
        heads[h] = merged

    year = Counter(years).most_common(1)[0][0] if years else None
    if year is None:                               # fall back to the filename
        m = re.search(r"(\d{4})", baseline_name)
        year = int(m.group(1)) if m else None
    return {"name": name, "year": year, "heads": heads, "loose": loose,
            "mtime": mpath.stat().st_mtime if mpath.exists() else None,
            "bmtime": bpath.stat().st_mtime if bpath.exists() else None,
            "metrics_name": metrics_name, "baseline_name": baseline_name}


# --------------------------------------------------------------- formatting

def num(v, pct=False):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if pct and isinstance(v, (int, float)):
        return f"{v:.1%}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v == int(v) and abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.4f}"
    return str(v)


def table(headers, rows, aligns=None):
    """Monospace table with a rule under the header."""
    cols = len(headers)
    w = [len(str(h)) for h in headers]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], len(str(r[i])))
    aligns = aligns or ["<"] + [">"] * (cols - 1)
    fmt = lambda cells: "  ".join(
        f"{str(c):{aligns[i]}{w[i]}}" for i, c in enumerate(cells))
    return "\n".join([fmt(headers), "  ".join("-" * x for x in w)]
                     + [fmt(r) for r in rows])


def ordered_heads(suites):
    """Every head seen in any suite, in GROUPS order, grouped."""
    seen = set().union(*(s["heads"] for s in suites))
    groups, placed = [], set()
    for title, members in GROUPS:
        hs = [h for h in members if h in seen]
        placed.update(hs)
        if hs:
            groups.append((title, hs))
    extra = sorted(seen - placed)
    if extra:
        groups.append(("Other heads", extra))
    return groups


def is_prob(head, suites):
    return any("auc" in s["heads"].get(head, {}) for s in suites)


# ------------------------------------------------------------------ report

def build_report(suites):
    out, p = [], lambda s="": out.append(s)
    sel, conf = suites
    p("=" * 78)
    p("MODEL HEAD PERFORMANCE - most recent retrain")
    p("=" * 78)
    p(f"generated  {datetime.now():%Y-%m-%d %H:%M}")
    for s in suites:
        yr = s["year"]
        when = (datetime.fromtimestamp(s["mtime"]).strftime("%Y-%m-%d %H:%M")
                if s["mtime"] else "MISSING")
        age = ""
        if s["mtime"]:
            days = (datetime.now() - datetime.fromtimestamp(s["mtime"])).days
            if days >= 2:
                age = f"  (!) {days}d old - retrain may not have run"
        p(f"{s['name']:<9} year {yr}   {s['metrics_name']} @ {when}{age}")
        if s["mtime"] and s["bmtime"] and s["bmtime"] < s["mtime"] - 3600:
            p(f"          note: {s['baseline_name']} is older than its "
              f"metrics file - total MAE / observed dispersion may lag")
    p()
    p("Suites: SELECTION = the decision year | SHIPPING = the season the "
      "serving models are scored on.")
    p("NEITHER is an untouched test: 2025 carries calibration + feature-"
      "selection information, and")
    p("2026 is a partial season already seen by past decisions - read both as "
      "UPPER-BOUND estimates.")
    p("The leakage-free read is the graded forward record: "
      "python Tools/4_grade_results.py --all")
    p()
    p("Probability heads: logloss/edge/AUC/Brier/ECE/top-10-a-day.  Count heads:"
      " MAE/dispersion.")
    p("edge = base-rate logloss - model logloss.  ECE from the metrics.json "
      "decile table.")
    p("dispersion: variance/mean, 1.0 = Poisson-ideal (>1 = real counts wilder "
      "than modeled).")

    groups = ordered_heads(suites)

    # ---- compact summary tables ---------------------------------------
    prob_rows, count_rows = [], []
    for _title, heads in groups:
        for h in heads:
            for s in suites:
                r = s["heads"].get(h)
                if not r:
                    continue
                if is_prob(h, suites):
                    prob_rows.append([
                        h, s["name"][:4], num(r.get("n_test")),
                        num(r.get("base_rate"), pct=True), num(r.get("logloss")),
                        num(r.get("edge")), num(r.get("auc")),
                        num(r.get("brier")), num(r.get("ece")),
                        num(r.get("top10_daily_hit_rate"), pct=True)])
                else:
                    count_rows.append([
                        h, s["name"][:4], num(r.get("n_test")),
                        num(r.get("mae")), num(r.get("mae_gain")),
                        num(r.get("dispersion_cal")), num(r.get("dispersion_obs")),
                        num(r.get("mean_actual")), num(r.get("mean_pred")),
                        num(r.get("bias"))])
    if prob_rows:
        p("\n" + "-" * 78)
        p("SUMMARY - probability heads")
        p("-" * 78)
        p(table(["head", "suite", "n", "base", "logloss", "edge", "AUC",
                 "Brier", "ECE", "top10"], prob_rows))
    if count_rows:
        p("\n" + "-" * 78)
        p("SUMMARY - count / regression heads")
        p("-" * 78)
        p(table(["head", "suite", "n", "MAE", "gain", "disp_cal", "disp_obs",
                 "mean_act", "mean_prd", "bias"], count_rows))

    # ---- per-head detail ----------------------------------------------
    p("\n" + "=" * 78)
    p("PER-HEAD DETAIL - every metric each head has, both suites")
    p("=" * 78)
    y = {s["name"]: s["year"] for s in suites}
    for title, heads in groups:
        p("\n" + "#" * 78)
        p(f"# {title}")
        p("#" * 78)
        for h in heads:
            label = LABELS.get(h, h)
            p(f"\n=== {h}  -  {label} ===")
            rows = []
            for key, lab in DETAIL:
                sv = sel["heads"].get(h, {}).get(key)
                cv = conf["heads"].get(h, {}).get(key)
                if sv is None and cv is None:
                    continue
                pct = key in ("base_rate", "top10_daily_hit_rate", "acc",
                              "acc_home_baseline")
                rows.append([lab, num(sv, pct=pct), num(cv, pct=pct)])
            if rows:
                p(table([" ", f"SELECTION {y['SELECTION']}",
                         f"SHIPPING {y['SHIPPING']}"], rows,
                        aligns=["<", ">", ">"]))
            else:
                p("  (no metrics recorded)")

    # ---- in-season recal offsets (loose, informative) -----------------
    off = conf["loose"].get("inseason_offsets") or sel["loose"].get(
        "inseason_offsets")
    if isinstance(off, dict):
        p("\n" + "-" * 78)
        p("In-season recalibration offsets (drift guard; serving-OFF reference)")
        p("-" * 78)
        p(table(["prop", "offset"],
                [[k, num(v)] for k, v in off.items()]))

    # ---- SIM_BLEND vs incumbent appendix ------------------------------
    for s in suites:
        path = ART / SIM_GRADE[s["name"]]
        if not path.exists():
            continue
        with path.open(newline="") as f:
            rdr = list(csv.reader(f))
        if len(rdr) < 2:
            continue
        p("\n" + "-" * 78)
        p(f"SIM_BLEND vs incumbent - {s['name']} {s['year']}  ({path.name})")
        p("-" * 78)
        p("  incumbent vs PA-sim per head; AUC for prob heads, MAE for count "
          "('(MAE)').")
        p("  dAUC/dMAE & dLL negative = sim worse (INC+ = keep incumbent).")
        p(table(rdr[0], rdr[1:]))

    p("\n" + "=" * 78)
    p(f"sources: {', '.join(s['metrics_name'] for s in suites)} + "
      f"eval_baseline_*.json (Model/artifacts/)")
    p("=" * 78)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output file (default: {DEFAULT_OUT})")
    args = ap.parse_args()

    suites = [load_suite(*cfg) for cfg in SUITES]
    if not any(s["heads"] for s in suites):
        raise SystemExit(f"no metrics found in {ART} - has train.py run?")
    report = build_report(suites)
    print(report)
    args.out.write_text(report + "\n", encoding="utf-8")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
