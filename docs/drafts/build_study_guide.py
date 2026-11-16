"""Assemble docs/STUDY_GUIDE.md from the chapter drafts in this folder.

    python docs/drafts/build_study_guide.py

The drafts stay the editable sources; the merged guide is regenerated, never edited by hand.
Per part: status lines are dropped, "Draft by Agent ..." sentences become an author line,
headings are demoted one level (outside code fences) so the guide has a single H1, and the
director's ruling replaces the placeholder in part 3.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).parent
PARTS = [
    ("STUDY_GUIDE_part0_front.md", None),
    ("STUDY_GUIDE_part1_data_features.md", None),
    ("STUDY_GUIDE_part2a_validation.md", None),
    ("STUDY_GUIDE_part2b_linear_models.md", None),
    ("STUDY_GUIDE_part2c_boosting_optuna.md", None),
    ("STUDY_GUIDE_part3_ensembles.md", None),
    ("STUDY_GUIDE_part4_final_selection.md", None),
    ("INTERPRETABILITY_E066.md", "# Study guide, part 5: what the final model relies on and where it fails"),
    ("STUDY_GUIDE_part6_implementation.md", None),
    ("STUDY_GUIDE_part7_defending.md", None),
]
RULING = (
    "**Director's ruling (D22, 2026-11-17).** Single CatBoost E066 is frozen. The equal-weight average has the "
    "best point estimate on both sets of folds, but its advantage (0.0010) is a third of a standard error and "
    "fails the rule the auditor wrote before reading the finalist records (at least 0.0010 better, beyond one "
    "standard error, at least 10 of 15 folds). Ties go to the simpler model. This is a parsimony ruling, not "
    "evidence that the average is worse; part 4 gives the full account."
)
DRAFT_BY = re.compile(
    r"Draft by Agent [A-D] \(([a-z-]+)\) for the director to merge into `docs/STUDY_GUIDE\.md`\.\s*"
    r"(Agent B will review the modelling content\.\s*)?")


def clean(text: str, new_title: str | None, demote: bool) -> str:
    out, in_code, title_done = [], False, False
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code:
            if line.startswith("DRAFT") or line.startswith("<!-- STATUS"):
                continue
            if line.strip() == "<<DIRECTOR RULING>>":
                line = RULING
            line = DRAFT_BY.sub(lambda m: f"Author: {m.group(1)}. ", line)
            if line.startswith("#"):
                if new_title and not title_done and line.startswith("# "):
                    line = new_title
                if line.startswith("# "):
                    title_done = True
                if demote:
                    line = "#" + line
        out.append(line)
    return "\n".join(out).strip() + "\n"


def main() -> None:
    chunks = []
    for i, (name, title) in enumerate(PARTS):
        chunks.append(clean((HERE / name).read_text(encoding="utf-8"), title, demote=i > 0))
    guide = "\n\n---\n\n".join(chunks)
    assert "<<" not in guide, "unfilled placeholder left in the guide"
    out = HERE.parent / "STUDY_GUIDE.md"
    out.write_text(guide, encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(guide.split())} words, {guide.count(chr(10))} lines")


if __name__ == "__main__":
    main()
