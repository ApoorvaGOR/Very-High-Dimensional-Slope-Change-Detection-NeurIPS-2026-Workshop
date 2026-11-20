import json
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from domain_reduction import DEFAULT_REFINEMENTS

BENCH_FIELDS = ["minlp_status", "minlp_incumbent_SSE", "minlp_best_bound_SSE", "minlp_gap_percent",
                "minlp_solver_runtime_seconds", "minlp_build_runtime_seconds", "minlp_total_runtime_seconds",
                "minlp_solution_count", "minlp_node_count", "minlp_breakpoint", "minlp_selected_count",
                "exact_minus_minlp_incumbent_SSE", "exact_minus_minlp_bound_SSE", "exact_within_minlp_bounds", "exact_bound_violating_dimensions", "minlp_error"]

TIME_FIELDS = ["data_generation_seconds", "data_loading_seconds", "exact_end_to_end_seconds", "validation_seconds", "bound_construction_seconds", "numerical_runtime_seconds",
               "statistics_seconds", "interval_solver_seconds", "recovery_seconds", "selection_seconds",
               "excel_writing_seconds", "data_npz_writing_seconds", "wall_clock_seconds"]
DOMAIN_CHECKPOINT_METRICS = ("feasible_SSE", "lower_bound_SSE", "search_seconds",
                             "intervals_retained", "intervals_evaluated")
DOMAIN_NUMERIC_FIELDS = ["domain_breakpoint", "domain_feasible_SSE", "domain_direct_residual_SSE",
                         "domain_numerical_runtime_seconds", "domain_end_to_end_seconds",
                         "domain_validation_seconds", "domain_npz_writing_seconds", "domain_excel_writing_seconds",
                         *[f"domain_{depth}_{metric}" for depth in DEFAULT_REFINEMENTS
                           for metric in DOMAIN_CHECKPOINT_METRICS]]
NUMERIC_FIELDS = TIME_FIELDS + ["breakpoint", "optimal_SSE", "baseline_SSE"] + DOMAIN_NUMERIC_FIELDS
FIELDS = ["dataset", "setting", "J", "T", "seed", "k", "status", "breakpoint", "optimal_SSE", "baseline_SSE", "direct_residual_SSE", "objective_validation_abs_error", "input_sha256",
          *TIME_FIELDS, *BENCH_FIELDS, "domain_status", *DOMAIN_NUMERIC_FIELDS, "domain_error", "error"]


def style_sheet(ws):
    ws.freeze_panes = "B2"
    ws.sheet_view.showGridLines = False
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="183153")
        cell.font = Font(bold=True, color="FFFFFF")
        cell.alignment = Alignment(wrap_text=True)
    for col in ws.columns:
        letter = col[0].column_letter
        ws.column_dimensions[letter].width = min(36, max(14, max(len(str(c.value or "")) for c in list(col)[:100]) + 2))


def save_detail_excel(path, x, Y, dates, tickers, result, setting, seed, k):
    """Write T x J raw/fitted arrays (transposed), avoiding Excel's 16,384-column limit."""
    J, T_local = Y.shape
    if J > 16382:
        raise ValueError("Detail workbook cannot accommodate J > 16382 columns")
    z = float(result["shared_breakpoint"])
    alpha = np.asarray(result["alpha"])
    beta = np.asarray(result["left_slope"])
    delta = np.asarray(result["delta"])
    hinge = np.maximum(0.0, x - z)
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("Summary")
    loss_label, loss_key = ("Feasible SSE", "feasible_loss") if "feasible_loss" in result else ("Optimal SSE", "optimal_loss")
    for label, value in (("Setting", setting), ("J", J), ("T", T_local), ("Seed", seed), ("k", k),
                         ("Estimated breakpoint", z), (loss_label, float(result[loss_key])),
                         ("Baseline SSE", float(result["C"]))):
        ws.append([label, value])
    raw = wb.create_sheet("Raw data")
    fitted = wb.create_sheet("Fitted data")
    residual = wb.create_sheet("Residuals")
    headers = ["time_index", "date_or_time", *map(str, tickers)]
    for sheet in (raw, fitted, residual):
        sheet.append(headers)
    for t in range(T_local):
        obs = Y[:, t]
        pred = alpha + beta * x[t] + delta * hinge[t]
        date = str(dates[t])
        raw.append([float(x[t]), date, *map(float, obs)])
        fitted.append([float(x[t]), date, *map(float, pred)])
        residual.append([float(x[t]), date, *map(float, obs - pred)])
    coef = wb.create_sheet("Stream coefficients")
    coef.append(["stream", "selected", "alpha", "left_slope", "delta", "right_slope",
                 "left_intercept", "right_intercept", "SSE"])
    selected = result.get("selected_mask", np.ones(J, dtype=bool))
    for j in range(J):
        coef.append([str(tickers[j]), bool(selected[j]), float(alpha[j]), float(beta[j]),
                     float(delta[j]), float(result["right_slope"][j]),
                     float(result["left_intercept"][j]), float(result["right_intercept"][j]),
                     float(result["losses_by_dimension"][j])])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def stats(rows, field):
    status, success = ("domain_status", "OK") if field in DOMAIN_NUMERIC_FIELDS else ("status", "ok")
    values = np.asarray([float(r[field]) for r in rows if r[status] == success and r[field] != ""], dtype=float)
    return (float(values.mean()) if values.size else None,
            float(values.std(ddof=1)) if values.size > 1 else None)


def write_instance_workbook(path, rows, startup_warmup):
    """Exactly one worksheet per seed, each showing its full timing breakdown."""
    wb = Workbook()
    wb.remove(wb.active)
    for row in rows:
        ws = wb.create_sheet(f"Seed_{int(row['seed']):06d}")
        ws.append(["Metric", "Value"])
        for name in FIELDS:
            value = row[name]
            if name in ("J", "T", "seed", "k"):
                value = int(value)
            elif name in NUMERIC_FIELDS and value != "":
                value = float(value)
            ws.append([name, value])
        ws.append(["one_time_numba_compile_seconds_excluded", startup_warmup.get(row["setting"])])
        ws.append(["one_time_domain_numba_compile_seconds_excluded", startup_warmup.get("domain_reduction")
                   if row["setting"] == "sparse" else None])
        style_sheet(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def write_final_summary(path, rows, args, startup_warmups):
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    metrics = ["minlp_incumbent_SSE", "minlp_best_bound_SSE", "minlp_gap_percent",
               "minlp_solver_runtime_seconds", "minlp_build_runtime_seconds", "minlp_total_runtime_seconds",
               "exact_minus_minlp_incumbent_SSE", "exact_minus_minlp_bound_SSE",
               "numerical_runtime_seconds", "exact_end_to_end_seconds", "bound_construction_seconds", "excel_writing_seconds",
               "data_generation_seconds", "statistics_seconds", "interval_solver_seconds",
               "selection_seconds", "recovery_seconds", "data_npz_writing_seconds", "wall_clock_seconds",
               "optimal_SSE", "breakpoint", *DOMAIN_NUMERIC_FIELDS]
    summary.append(["Dataset", "Setting", "J", "T", "k", "Completed", "Success", "Failed",
                    "One-time Numba compile (s), excluded", *[x for m in metrics for x in (f"Mean {m}", f"SD {m}")]])
    groups = sorted({(r["dataset"], r["setting"], int(r["J"]), int(r["T"]), int(r["k"])) for r in rows})
    for dataset, setting, J, T_local, k in groups:
        subset = [r for r in rows if (r["dataset"],r["setting"],int(r["J"]),int(r["T"]),int(r["k"])) == (dataset,setting,J,T_local,k)]
        good = [r for r in subset if r["status"] == "ok"]
        summary.append([dataset,setting,J,T_local,k,len(subset),len(good),len(subset)-len(good),
                        startup_warmups.get(setting), *[v for m in metrics for v in stats(subset,m)]])
    raw = wb.create_sheet("All replications")
    raw.append(FIELDS)
    for r in rows:
        raw.append([int(r[k]) if k in ("J", "T", "seed", "k") else
                    float(r[k]) if k in NUMERIC_FIELDS and r[k] != "" else r[k]
                    for k in FIELDS])
    if "domain_reduction" in startup_warmups:
        checkpoints = wb.create_sheet("Domain reduction")
        identifiers = ["dataset", "setting", "J", "T", "seed", "k"]
        checkpoints.append([*identifiers, "refinements", *DOMAIN_CHECKPOINT_METRICS, "exact_optimal_SSE"])
        for r in rows:
            if r["domain_status"] == "OK":
                for depth in DEFAULT_REFINEMENTS:
                    checkpoints.append([r["dataset"], r["setting"], *[int(r[k]) for k in identifiers[2:]],
                                        depth, *[float(r[f"domain_{depth}_{m}"]) for m in DOMAIN_CHECKPOINT_METRICS],
                                        float(r["optimal_SSE"]) if r["optimal_SSE"] != "" else None])
        style_sheet(checkpoints)
    comparison = wb.create_sheet("MINLP comparison")
    comparison_fields = ["dataset", "setting", "J", "T", "seed", "k", "status", "optimal_SSE",
                         "numerical_runtime_seconds", *BENCH_FIELDS]
    comparison.append(comparison_fields)
    for r in rows:
        comparison.append([r.get(f, "") for f in comparison_fields])
    style_sheet(comparison)
    config = wb.create_sheet("Configuration")
    for k, v in vars(args).items():
        config.append([k, json.dumps(v) if isinstance(v, (list, dict)) else str(v) if isinstance(v, Path) else v])
    config.append(["Numba compilation", "Compiled once per solver at startup; measured separately, excluded from numerical runtime. Its sample SD across seeds is not statistically defined."])
    if "domain_reduction" in startup_warmups:
        config.append(["Domain reduction Numba warmup (s)", startup_warmups["domain_reduction"]])
        config.append(["Domain reduction timings", "Checkpoint search times include statistics and all prior refinement rounds. Final numerical runtime also includes coefficient recovery."])
    config.append(["Coordinate stabilization", "Exact solver uses affine time rescaling and response centering; coefficients exported in original coordinates, raw-unit SSE retained."])
    config.append(["Excel timing", "Per-seed detail Excel writing only. Checkpoint runtime workbooks and final summary writing are excluded from this field."])
    config.append(["Runtime", "numerical_runtime_seconds is solver's reported numerical runtime; excludes warmup, simulation and writing."])
    config.append(["MINLP", "Manuscript Appendix C; bounded parameter domain may exclude the unconstrained polynomial-time solution."])
    config.append(["MINLP time limit", "Gurobi TimeLimit applies to optimize(), not model construction; runtime recorded separately."])
    config.append(["MINLP gap", "100 * Gurobi MIPGap; blank when no incumbent exists."])
    config.append(["SD", "Sample standard deviation (ddof=1), computed over successful seeds."])
    config.append(["Simulator", "Synthetic correlated stock-price paths; not guaranteed exact shared-breakpoint PWL signals."])
    for ws in (summary, raw):
        style_sheet(ws)
    config.column_dimensions["A"].width = 32
    config.column_dimensions["B"].width = 100
    wb.save(path)

