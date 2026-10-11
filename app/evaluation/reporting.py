"""Human-readable report formatting - a thin presentation layer over
`StrategyContextEvaluation`/`ComparisonRow`. Every number shown here is
read directly off the model; nothing is computed in this module.
Produces the shape the mission's own "Final Engineering Principle"
section explicitly asks for (evidence + walk-forward + status +
warnings - never a score or a verdict).
"""

from __future__ import annotations

from app.evaluation.comparison import MULTIPLE_TESTING_CAVEAT, ComparisonRow
from app.evaluation.models import StrategyContextEvaluation


def _context_label(evaluation: StrategyContextEvaluation) -> str:
    f = evaluation.context_filter
    parts = [
        evaluation.instrument_symbol,
        evaluation.timeframe,
        f.session,
        f.trend_state.value if f.trend_state else None,
        f.volatility_state.value if f.volatility_state else None,
        f.structure_state.value if f.structure_state else None,
        evaluation.direction,
    ]
    return " / ".join(p for p in parts if p) or "(no context filter)"


def format_report(evaluation: StrategyContextEvaluation) -> str:
    p = evaluation.probability
    lines = [
        f"Strategy:",
        f"    {evaluation.strategy_name or '(all strategies)'}",
        "",
        f"Context:",
        f"    {_context_label(evaluation)}",
        "",
        "Evidence:",
        f"    observations = {p.counts.total_observations} ({p.counts.resolved_observations} resolved)",
        f"    P(win) = {p.p_win if p.p_win is not None else 'unavailable'}",
        f"    P(loss) = {p.p_loss if p.p_loss is not None else 'unavailable'}",
        f"    P(breakeven) = {p.p_breakeven if p.p_breakeven is not None else 'unavailable'}",
        f"    P(unresolved) = {p.p_unresolved if p.p_unresolved is not None else 'unavailable'}",
    ]
    if p.win_interval is not None:
        lines.append(
            f"    interval = [{p.win_interval.lower_bound:.3f}, {p.win_interval.upper_bound:.3f}] "
            f"({p.win_interval.confidence_level:.0%})"
        )
    else:
        lines.append("    interval = unavailable")
    lines.append(f"    expected_R = {p.expected_r if p.expected_r is not None else 'unavailable'}")

    if evaluation.walk_forward is not None:
        s = evaluation.walk_forward.stability
        lines += [
            "",
            "Walk-forward:",
            f"    windows = {s.number_of_windows}",
            f"    positive windows = {s.windows_with_positive_expected_r}",
            f"    mean expected_R = {s.mean_expected_r if s.mean_expected_r is not None else 'unavailable'}",
            f"    expected_R range = {s.expected_r_range if s.expected_r_range is not None else 'unavailable'}",
            f"    win rate range = {s.win_rate_range if s.win_rate_range is not None else 'unavailable'}",
        ]

    lines += ["", "Status:", f"    {evaluation.evidence_tier.value.upper()}"]

    if evaluation.warnings:
        lines += ["", "Warnings:"] + [f"    - {w}" for w in evaluation.warnings]

    if evaluation.robustness_summary is not None:
        lines += [
            "",
            "Robustness (M3 reference):",
            f"    experiment_id = {evaluation.robustness_summary.get('parent_experiment_id')}",
            f"    stability_label = {evaluation.robustness_summary.get('stability_label')}",
        ]

    return "\n".join(lines)


def format_comparison_report(rows: list[ComparisonRow]) -> str:
    lines = [f"Comparing {len(rows)} configuration(s):", ""]
    for row in rows:
        interval = (
            f"[{row.win_interval_lower:.3f}, {row.win_interval_upper:.3f}]"
            if row.win_interval_lower is not None
            else "unavailable"
        )
        lines += [
            row.label,
            f"    n = {row.total_observations} ({row.resolved_observations} resolved)",
            f"    P(win) = {row.p_win if row.p_win is not None else 'unavailable'}  interval = {interval}",
            f"    expected_R = {row.expected_r if row.expected_r is not None else 'unavailable'}",
            f"    evidence_tier = {row.evidence_tier.value}",
            "",
        ]
    if len(rows) > 1:
        lines.append(MULTIPLE_TESTING_CAVEAT)
    return "\n".join(lines)
