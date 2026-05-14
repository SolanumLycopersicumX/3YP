#!/usr/bin/env python3
"""Generate a 16:9 slide-style speaker-script PDF.

Each page corresponds one-to-one with the final presentation slides.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from build_presentation_speech_qa_pdf import SLIDES, tex_escape


ROOT = Path(__file__).resolve().parents[1]
OUT_TEX = ROOT / "01_Reports" / "final_project_presentation_speaker_slides_11314389.tex"
OUT_PDF = ROOT / "01_Reports" / "final_project_presentation_speaker_slides_11314389.pdf"


def build_tex() -> str:
    total = len(SLIDES)
    lines = [
        r"\documentclass[aspectratio=169,10pt]{beamer}",
        r"\usepackage{fontspec}",
        r"\usepackage{xcolor}",
        r"\usepackage{ragged2e}",
        r"\usepackage{tikz}",
        r"\usetikzlibrary{calc}",
        r"\setmainfont{TeX Gyre Heros}",
        r"\setsansfont{TeX Gyre Heros}",
        r"\definecolor{uomPurple}{HTML}{660099}",
        r"\definecolor{uomGold}{HTML}{FFCC33}",
        r"\definecolor{textDark}{HTML}{222222}",
        r"\definecolor{panelGrey}{HTML}{F7F7F7}",
        r"\definecolor{noteCream}{HTML}{FFF7D6}",
        r"\setbeamertemplate{navigation symbols}{}",
        r"\setbeamertemplate{footline}{}",
        r"\setbeamercolor{normal text}{fg=textDark,bg=white}",
        r"\begin{document}",
    ]

    for slide in SLIDES:
        n = slide["n"]
        title = tex_escape(slide["title"])
        time = tex_escape(slide["time"])
        script = tex_escape(slide["script"])
        detail = tex_escape(slide["detail"])
        lines.extend(
            [
                r"\begin{frame}[plain]",
                r"\vspace*{0.16cm}",
                r"\begin{columns}[T,totalwidth=\textwidth]",
                r"\begin{column}{0.72\textwidth}",
                rf"{{\fontsize{{17}}{{18.5}}\selectfont\bfseries Slide {n}: {title}}}",
                r"\end{column}",
                r"\begin{column}{0.25\textwidth}",
                r"\raggedleft",
                rf"\colorbox{{noteCream}}{{\parbox{{0.88\linewidth}}{{\centering\bfseries Target: {time}}}}}",
                r"\end{column}",
                r"\end{columns}",
                r"\vspace{0.10cm}",
                r"{\color{uomPurple}\rule{\textwidth}{1.2pt}}",
                r"\vspace{0.18cm}",
                r"{\color{uomPurple}\bfseries Speaker script}",
                r"\vspace{0.06cm}",
                r"\setlength{\fboxsep}{6pt}",
                r"\noindent\colorbox{panelGrey}{\begin{minipage}{0.94\textwidth}",
                r"\justifying\fontsize{11.0}{13.4}\selectfont",
                script,
                r"\end{minipage}}",
                r"\vspace{0.22cm}",
                r"\noindent\colorbox{noteCream}{\begin{minipage}{0.94\textwidth}",
                r"\fontsize{9.6}{11.6}\selectfont",
                rf"{{\bfseries If asked:}} {detail}",
                r"\end{minipage}}",
                r"\vfill",
                rf"\hfill{{\color{{textDark}}\scriptsize {n} / {total}}}",
                r"\end{frame}",
                "",
            ]
        )

    lines.append(r"\end{document}")
    return "\n".join(lines)


def main() -> None:
    OUT_TEX.write_text(build_tex(), encoding="utf-8")
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-halt-on-error", OUT_TEX.name],
        cwd=OUT_TEX.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"Wrote {OUT_TEX}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
