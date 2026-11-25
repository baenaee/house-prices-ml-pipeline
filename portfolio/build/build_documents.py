"""Render the portfolio PDFs with headless Chrome (Markdown -> HTML -> PDF).

Run from the repository root, after make_figures.py:
    python portfolio/build/build_documents.py            # all documents
    python portfolio/build/build_documents.py report     # only one: report | slides | guide

Outputs
    portfolio/HousePrices_Technical_Report.pdf   built from docs/RESEARCH_REPORT.md + figures
    portfolio/HousePrices_Slides.pdf             built from portfolio/build/slides.html
    portfolio/figures/social_preview.png         1280x640 card for GitHub / LinkedIn link previews
    career_prep/HousePrices_Study_Guide.pdf      built from career_prep/*.md + docs/STUDY_GUIDE.md
                                                 (only if career_prep/ exists; it is git-ignored)
Requires: mistune 3, Google Chrome or Microsoft Edge.
"""
from __future__ import annotations

import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "portfolio" / "build"
TMP = BUILD / "_html"
TMP.mkdir(exist_ok=True)
BASE_URI = ROOT.as_uri() + "/"

AUTHOR = "Enrique Baena"
DATE = "November 2026"

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chrome", "msedge",
]


def chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("Chrome/Edge not found")


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           "--generate-pdf-document-outline", "--virtual-time-budget=15000",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True)
    print("wrote", pdf_path.relative_to(ROOT))


# ------------------------------------------------------------------ markdown
_md = mistune.create_markdown(escape=False, plugins=["table", "strikethrough"])


def slug(text: str, used: set[str]) -> str:
    s = re.sub(r"<[^>]+>", "", text)
    s = re.sub(r"[^a-z0-9]+", "-", html.unescape(s).lower()).strip("-")[:60] or "s"
    base, k = s, 2
    while s in used:
        s, k = f"{base}-{k}", k + 1
    used.add(s)
    return s


def md_to_html(text: str, used: set[str], toc: list | None = None, toc_levels=(1, 2)) -> str:
    out = _md(text)

    def head(m):
        level, inner = int(m.group(1)), m.group(2)
        hid = slug(inner, used)
        if toc is not None and level in toc_levels:
            toc.append((level, re.sub(r"<[^>]+>", "", inner), hid))
        return f'<h{level} id="{hid}">{inner}</h{level}>'

    out = re.sub(r"<h([1-4])>(.*?)</h\1>", head, out, flags=re.S)
    # images alone in a paragraph become captioned figures
    out = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)"\s*/?></p>',
                 lambda m: f'<figure><img src="{m.group(1)}" alt=""><figcaption>{m.group(2)}</figcaption></figure>',
                 out)
    return out


def toc_html(toc: list) -> str:
    items = []
    for level, text, hid in toc:
        cls = "toc1" if level == 1 else "toc2"
        items.append(f'<li class="{cls}"><a href="#{hid}">{text}</a></li>')
    return '<nav class="toc"><h2 class="toc-title">Contents</h2><ul>' + "".join(items) + "</ul></nav>"


DOC_CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page); font: 9pt 'Segoe UI', Arial, sans-serif; color: #8a8984; } }
@page :first { @bottom-center { content: none; } }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; font-size: 10.3pt; line-height: 1.5;
       color: #1d1d1b; margin: 0; }
h1, h2, h3, h4 { color: #0b0b0b; line-height: 1.25; break-after: avoid; }
h1 { font-size: 22pt; margin: 0 0 14pt; padding-bottom: 6pt; border-bottom: 3px solid #2a78d6; break-before: page; }
h2 { font-size: 14.5pt; margin: 20pt 0 8pt; color: #1b4f8f; }
h3 { font-size: 12pt; margin: 14pt 0 6pt; }
h4 { font-size: 10.8pt; margin: 12pt 0 4pt; color: #52514e; }
p { margin: 0 0 7pt; orphans: 3; widows: 3; }
a { color: #1b5fb0; text-decoration: none; }
blockquote { margin: 8pt 0; padding: 8pt 12pt; background: #f3f7fd; border-left: 4px solid #2a78d6; border-radius: 4px; }
blockquote p:last-child { margin-bottom: 0; }
code { font-family: Consolas, 'Cascadia Mono', monospace; font-size: 8.9pt; background: #f1f0ec; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f5f2; border: 1px solid #e6e5e1; border-radius: 6px; padding: 8pt 10pt; overflow-wrap: anywhere;
      white-space: pre-wrap; break-inside: avoid; font-size: 8.4pt; line-height: 1.4; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.9pt; line-height: 1.35; }
th { background: #eef3fa; color: #0b0b0b; text-align: left; font-weight: 600; }
th, td { border: 1px solid #dcdad4; padding: 3.5pt 6pt; vertical-align: top; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
figure { margin: 10pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 118mm; }
figcaption { font-size: 8.8pt; color: #52514e; margin-top: 4pt; font-style: italic; }
hr { border: none; border-top: 1px solid #e6e5e1; margin: 14pt 0; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }
.cover { height: 252mm; display: flex; flex-direction: column; justify-content: space-between; break-after: page; }
.cover .band { background: linear-gradient(135deg, #1b4f8f, #2a78d6); color: white; padding: 34mm 14mm 20mm; border-radius: 10px; }
.cover .kicker { font-size: 11pt; letter-spacing: .14em; text-transform: uppercase; opacity: .85; }
.cover .title { font-size: 30pt; font-weight: 700; line-height: 1.15; margin: 10pt 0 12pt; }
.cover .subtitle { font-size: 13pt; opacity: .95; }
.cover .meta { font-size: 10.5pt; color: #52514e; padding: 0 4mm; }
.cover .meta b { color: #0b0b0b; }
.cover .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8pt; margin: 0 0 10pt; }
.cover .stat { border: 1px solid #dcdad4; border-radius: 8px; padding: 9pt; }
.cover .stat .v { font-size: 18pt; font-weight: 700; color: #1b4f8f; }
.cover .stat .l { font-size: 8.5pt; color: #52514e; }
.toc { break-after: page; }
.toc-title { font-size: 20pt; color: #0b0b0b; margin-top: 0; }
.toc ul { list-style: none; padding: 0; columns: 1; }
.toc li.toc1 { font-weight: 700; margin-top: 9pt; font-size: 11pt; }
.toc li.toc2 { margin-left: 14pt; font-size: 9.6pt; }
.abstract { background: #f6f5f2; border-radius: 8px; padding: 10pt 14pt; margin: 0 0 12pt; }
.abstract h2 { margin-top: 0; font-size: 12pt; }
.note { font-size: 9pt; color: #52514e; }
.report h2 { font-size: 15pt; color: #0b0b0b; border-bottom: 2px solid #2a78d6; padding-bottom: 4pt; margin-top: 22pt; }
"""


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><base href='{BASE_URI}'>"
            f"<title>{html.escape(title)}</title><style>{DOC_CSS}</style></head><body>{body}</body></html>")


def stats_html(stats) -> str:
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in stats) + "</div>"


# ------------------------------------------------------------------ technical report
REPORT_FIGS = {
    "## 1. Summary": [("portfolio/figures/results_ladder.png", "Figure 1. The results ladder on the selection folds; every bar is a registered experiment.")],
    "## 2. Problem, data and protocol": [
        ("portfolio/figures/pipeline_overview.svg", "Figure 2. End-to-end flow. Green: deterministic, run once. Blue: refitted inside every CV fold. Orange: decisions and guarded steps."),
        ("portfolio/figures/validation_protocol.svg", "Figure 3. How the 1,460 labelled houses are used."),
        ("portfolio/figures/leakage_rule.svg", "Figure 4. Which steps may run before cross-validation.")],
    "### 4.3 Outlier policy": [("portfolio/figures/outlier_policy.png", "Figure 5. Outlier removal restricted to training folds, validation on complete folds.")],
    "### 4.5 Boosted trees": [("portfolio/figures/catboost_optuna_history.png", "Figure 6. CatBoost Optuna history across Stage 1 and Stage 2.")],
    "### 4.6 Ensembles": [("portfolio/figures/nested_stacking.svg", "Figure 7. Nested stacking inside one outer fold.")],
    "## 6. Finalists on fresh folds": [("portfolio/figures/finalists_fresh_folds.png", "Figure 8. Selection-fold versus fresh-fold scores of the eight finalists.")],
    "## 9. What the model relies on": [
        ("outputs/figures/E066_permutation_top25.png", "Figure 9. Permutation importance (validation folds) of the final model."),
        ("outputs/figures/E066_shap_beeswarm.png", "Figure 10. SHAP summary of the final model."),
        ("portfolio/figures/error_by_price_decile.png", "Figure 11. Out-of-fold RMSE by price decile.")],
}

ABSTRACT = """
We build a leakage-safe, fully reproducible regression pipeline for the Kaggle *House Prices: Advanced Regression
Techniques* competition (Ames, Iowa; 1,460 labelled houses, 79 features). A 15 % stratified holdout is locked before
any modelling; all fitted preprocessing, models and ensemble weights are learned inside the training part of each
cross-validation fold; ensembles are stacked with nested cross-validation; and every one of 83 experiments is stored in
an append-only registry from which its score can be reproduced with one command. Selection optimism is measured by
re-scoring eight finalists on fresh folds (+0.003 to +0.004 RMSE), and the final model is chosen with a decision rule
recorded before the comparison was read. The frozen model, a tuned CatBoost regressor, reaches **0.1217 RMSLE** on
fresh folds and **0.1065** on the single-use holdout (95 % bootstrap interval 0.092–0.121). Two common practices do not
survive honest validation: removing the well-known outliers from training folds worsens every model, and no blend or
stacked ensemble improves on the single model by more than a third of a standard error. Errors concentrate at the price
extremes and in a handful of atypical houses.
"""


def build_report() -> None:
    src = (ROOT / "docs" / "RESEARCH_REPORT.md").read_text(encoding="utf-8").splitlines()
    src = src[1:]  # drop the markdown title; the cover carries it
    out, pending = [], []
    for line in src:
        if line.startswith("#") and pending:
            out.extend(pending)
            pending = []
        out.append(line)
        for key, figs in REPORT_FIGS.items():
            if line.startswith(key):
                pending = [""] + [f"![{cap}]({p})\n" for p, cap in figs]
    out.extend(pending)
    used: set[str] = set()
    body = md_to_html("\n".join(out), used)
    cover = f"""
<section class="cover">
  <div class="band">
    <div class="kicker">Technical report</div>
    <div class="title">Leakage-safe machine learning for house-price prediction</div>
    <div class="subtitle">A reproducible, audited study on the Kaggle Ames House Prices data</div>
  </div>
  <div>
    {stats_html([("0.1217", "RMSLE, fresh-fold CV (final CatBoost)"), ("0.1065", "RMSLE, holdout scored once"),
                 ("83", "logged, reproducible experiments"), ("217", "automated tests")])}
    <div class="abstract"><h2>Abstract</h2>{_md(ABSTRACT)}</div>
    <div class="meta"><b>{AUTHOR}</b> · {DATE} · Code and data: see the repository README.<br>
    Implementation carried out with Claude Code agent teams under the author's methodology and review;
    every number is traceable to <code>experiments/registry.jsonl</code>.</div>
  </div>
</section>"""
    html_path = TMP / "report.html"
    html_path.write_text(page("Technical report — House Prices", cover + "<main class='report'>" + body + "</main>"),
                         encoding="utf-8")
    print_pdf(html_path, ROOT / "portfolio" / "HousePrices_Technical_Report.pdf")


# ------------------------------------------------------------------ slides
def build_slides() -> None:
    src = BUILD / "slides.html"
    html_path = TMP / "slides.html"
    html_path.write_text(src.read_text(encoding="utf-8").replace("{{BASE}}", BASE_URI), encoding="utf-8")
    print_pdf(html_path, ROOT / "portfolio" / "HousePrices_Slides.pdf")


# ------------------------------------------------------------------ social preview (GitHub / LinkedIn card)
def build_social() -> None:
    html_path = TMP / "social.html"
    html_path.write_text((BUILD / "social_preview.html").read_text(encoding="utf-8").replace("{{BASE}}", BASE_URI),
                         encoding="utf-8")
    out = ROOT / "portfolio" / "figures" / "social_preview.png"
    subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1280,640",
                    f"--screenshot={out}", html_path.as_uri()], check=True, capture_output=True)
    print("wrote", out.relative_to(ROOT))


# ------------------------------------------------------------------ study guide
def build_guide() -> None:
    cp = ROOT / "career_prep"
    if not (cp / "01_interview_edition.md").exists():
        print("career_prep/ not found: study guide skipped")
        return
    used: set[str] = set()
    toc: list = []
    part1 = md_to_html((cp / "01_interview_edition.md").read_text(encoding="utf-8"), used, toc)

    deep = (ROOT / "docs" / "STUDY_GUIDE.md").read_text(encoding="utf-8")
    deep = re.sub(r"^# ", "## ", deep, flags=re.M)
    deep = "# Part II — The deep reference guide\n\n" \
           "*This part is `docs/STUDY_GUIDE.md` from the repository, unchanged: about 34,000 words covering every " \
           "method, number and decision in depth, with self-check questions and answers at the end of each part. " \
           "Use Part I to learn the story; use Part II when an interviewer digs into a specific topic.*\n\n" + deep
    part2 = md_to_html(deep, used, toc)
    part3 = md_to_html((cp / "02_github_linkedin_cv.md").read_text(encoding="utf-8"), used, toc)

    cover = f"""
<section class="cover">
  <div class="band">
    <div class="kicker">Study guide · interview preparation</div>
    <div class="title">House Prices: a leakage-safe ML pipeline, explained end to end</div>
    <div class="subtitle">Everything you need to explain and defend the project in an interview</div>
  </div>
  <div>
    {stats_html([("≈ 0.12", "expected RMSLE (≈ 12 % price error)"), ("83", "logged experiments"),
                 ("217", "tests · 36 audit findings"), ("1 ×", "holdout used exactly once")])}
    <div class="abstract"><h2>Inside</h2>
    <p><b>Part I — Interview edition.</b> The pitch in 30 s / 2 min / 5 min, the ML flow diagram, the numbers to memorise,
    every concept explained from zero (validation, leakage, linear models, boosting, Optuna, ensembles, interpretability),
    the engineering, how to talk about AI-assisted work, a question bank with model answers, a glossary and a one-page cheat sheet.</p>
    <p><b>Part II — The deep reference guide.</b> The full 34,000-word guide from the repository.</p>
    <p><b>Part III — Publishing.</b> GitHub upload step by step, LinkedIn profile and post, CV bullets (English and Spanish).</p></div>
    <div class="meta"><b>{AUTHOR}</b> · {DATE} · private preparation document (kept out of the public repository)</div>
  </div>
</section>"""
    body = cover + toc_html(toc) + part1 + part2 + part3
    html_path = TMP / "guide.html"
    html_path.write_text(page("Study guide — House Prices", body), encoding="utf-8")
    print_pdf(html_path, cp / "HousePrices_Study_Guide.pdf")


if __name__ == "__main__":
    which = sys.argv[1:] or ["report", "slides", "social", "guide"]
    for w in which:
        {"report": build_report, "slides": build_slides, "social": build_social, "guide": build_guide}[w]()
