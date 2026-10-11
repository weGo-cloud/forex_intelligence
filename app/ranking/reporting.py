"""Human-readable ranking report - presentation only, same discipline
as `app.evaluation.reporting`: every number is read directly off a
`RankedEvaluation`, nothing computed here."""

from __future__ import annotations

from app.ranking.ranking import RANKING_CAVEAT, RankedEvaluation


def format_ranking_report(results: list[RankedEvaluation]) -> str:
    ranked = [r for r in results if r.rank is not None]
    excluded = [r for r in results if r.rank is None]

    lines = [f"Ranked {len(ranked)} of {len(results)} evaluated configuration(s):", ""]
    for r in ranked:
        p = r.evaluation.probability
        lines += [
            f"#{r.rank}  {r.label}",
            f"    FIAS = {r.fias.score:.3f}  (sample_adequacy={r.fias.components.sample_adequacy}, "
            f"win_rate_confidence={r.fias.components.win_rate_confidence}, "
            f"stability={r.fias.components.stability}, robustness={r.fias.components.robustness})",
            f"    n = {p.counts.resolved_observations}  P(win) = {p.p_win}  expected_R = {p.expected_r}",
            f"    q-value = {r.q_value:.4f}  significant = {r.significant}" if r.q_value is not None else "    q-value = unavailable",
            "",
        ]

    if excluded:
        lines.append("Excluded from ranking:")
        for r in excluded:
            lines.append(f"  - {r.label}: {'; '.join(r.exclusion_reasons)}")
        lines.append("")

    if len(results) > 1:
        lines.append(RANKING_CAVEAT)

    return "\n".join(lines)
