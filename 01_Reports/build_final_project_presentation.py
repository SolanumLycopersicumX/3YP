#!/usr/bin/env python3
"""Build the final 3YP presentation from the Manchester 16:9 template.

The deck is generated as raw OOXML so it does not depend on python-pptx.
It keeps the template masters/layouts and adds explicit page numbers because
the template's slide-number placeholders are disabled in exported PDFs.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageSequence


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "05_Documentation" / "Master_169 presentation.pptx"
OUT_PPTX = ROOT / "01_Reports" / "final_project_presentation_11314389.pptx"
OUT_NOTES = ROOT / "01_Reports" / "final_project_presentation_11314389_timing_notes.md"
EMU_PER_INCH = 914400

PURPLE = "660099"
GOLD = "FFCC33"
DARK = "333333"
TEXT = "111111"
MID = "595959"
LIGHT = "F2F2F2"
PANEL = "F7F7F7"
TEAL = "007C89"
BLUE = "2F5597"
GREEN = "2E7D32"
RED = "B03A2E"
ORANGE = "C55A11"


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def ensure_png_from_pdf(pdf_path: Path, out_path: Path, dpi: int = 200) -> Path:
    if out_path.exists() and out_path.stat().st_mtime >= pdf_path.stat().st_mtime:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-singlefile", str(pdf_path), str(stem)],
        check=True,
        capture_output=True,
        text=True,
    )
    return out_path


def ensure_gif_frame(gif_path: Path, out_path: Path) -> Path:
    if out_path.exists() and out_path.stat().st_mtime >= gif_path.stat().st_mtime:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(gif_path) as im:
        frame = next(ImageSequence.Iterator(im)).convert("RGBA")
        frame.save(out_path)
    return out_path


def ensure_video_frame(video_path: Path, out_path: Path) -> Path:
    if not video_path.exists():
        if out_path.exists():
            return out_path
        return ensure_gif_frame(ROOT / "05_Documentation" / "rl_arm.gif", out_path)
    if out_path.exists() and out_path.stat().st_mtime >= video_path.stat().st_mtime:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read a video frame from {video_path}")
        cv2.imwrite(str(out_path), frame)
    except Exception:
        # Fallback to the simulation GIF if OpenCV cannot read the video.
        ensure_gif_frame(ROOT / "05_Documentation" / "rl_arm.gif", out_path)
    return out_path


@dataclass
class ImageAsset:
    source: Path
    media_name: str | None = None


@dataclass
class Slide:
    title: str
    layout: int = 2
    objects: list[str] = field(default_factory=list)
    rels: list[tuple[str, str, str]] = field(default_factory=list)
    notes: str = ""
    timing: str = ""


class DeckBuilder:
    def __init__(self) -> None:
        self.next_shape_id = 2
        self.next_image_id = 1
        self.assets: dict[Path, str] = {}

    def shape_id(self) -> int:
        sid = self.next_shape_id
        self.next_shape_id += 1
        return sid

    def reset_shapes(self) -> None:
        self.next_shape_id = 2

    def media_name(self, source: Path) -> str:
        source = source.resolve()
        if source in self.assets:
            return self.assets[source]
        suffix = source.suffix.lower()
        if suffix == ".jpg":
            suffix = ".jpeg"
        name = f"image_final_{self.next_image_id}{suffix}"
        self.next_image_id += 1
        self.assets[source] = name
        return name

    def text_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        paragraphs: list[dict],
        *,
        fill: str | None = None,
        line: str | None = None,
        radius: str = "rect",
        name: str = "Text",
        margin: float = 0.08,
    ) -> str:
        sid = self.shape_id()
        fill_xml = (
            f"<a:solidFill><a:srgbClr val=\"{fill}\"/></a:solidFill>"
            if fill
            else "<a:noFill/>"
        )
        line_xml = (
            f"<a:ln w=\"9525\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln>"
            if line
            else "<a:ln><a:noFill/></a:ln>"
        )
        paras = "\n".join(self.paragraph(p) for p in paragraphs)
        return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}
        </p:spPr>
        <p:txBody>
          <a:bodyPr wrap="square" lIns="{emu(margin)}" tIns="{emu(margin)}" rIns="{emu(margin)}" bIns="{emu(margin)}" anchor="t"/>
          <a:lstStyle/>
          {paras}
        </p:txBody>
      </p:sp>"""

    def paragraph(self, p: dict) -> str:
        text = esc(p.get("text", ""))
        size = int(p.get("size", 20) * 100)
        color = p.get("color", TEXT)
        bold = ' b="1"' if p.get("bold") else ""
        align = p.get("align", "l")
        bullet = p.get("bullet", False)
        before = int(p.get("before", 0))
        after = int(p.get("after", 0))
        ppr_extra = ""
        if bullet:
            ppr_extra = (
                '<a:buFont typeface="Arial"/><a:buChar char="&#x2022;"/>'
                '<a:defRPr/>'
            )
            ppr = (
                f'<a:pPr marL="{emu(0.22)}" indent="-{emu(0.16)}" algn="{align}">'
                f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef>'
                f'<a:spcAft><a:spcPts val="{after}"/></a:spcAft>{ppr_extra}</a:pPr>'
            )
        else:
            ppr = (
                f'<a:pPr algn="{align}">'
                f'<a:spcBef><a:spcPts val="{before}"/></a:spcBef>'
                f'<a:spcAft><a:spcPts val="{after}"/></a:spcAft></a:pPr>'
            )
        return (
            f"<a:p>{ppr}<a:r><a:rPr lang=\"en-US\" sz=\"{size}\"{bold}>"
            f"<a:solidFill><a:srgbClr val=\"{color}\"/></a:solidFill>"
            f"<a:latin typeface=\"Arial\"/><a:cs typeface=\"Arial\"/></a:rPr>"
            f"<a:t>{text}</a:t></a:r></a:p>"
        )

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        line: str | None = None,
        name: str = "Rectangle",
    ) -> str:
        sid = self.shape_id()
        line_xml = (
            f"<a:ln w=\"9525\"><a:solidFill><a:srgbClr val=\"{line}\"/></a:solidFill></a:ln>"
            if line
            else "<a:ln><a:noFill/></a:ln>"
        )
        return f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}
        </p:spPr>
      </p:sp>"""

    def line(self, x: float, y: float, w: float, *, color: str = PURPLE, height: float = 0.03) -> str:
        return self.rect(x, y, w, height, fill=color, line=None, name="Accent")

    def stat_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        label: str,
        *,
        color: str = PURPLE,
    ) -> str:
        return self.text_box(
            x,
            y,
            w,
            h,
            [
                {"text": value, "size": 28, "bold": True, "color": color, "align": "c"},
                {"text": label, "size": 11.5, "color": DARK, "align": "c"},
            ],
            fill=PANEL,
            line="D9D9D9",
            margin=0.05,
            name="Stat",
        )

    def picture(
        self,
        slide: Slide,
        source: Path,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        name: str = "Picture",
        border: str | None = None,
    ) -> str:
        media_name = self.media_name(source)
        rid = f"rId{len(slide.rels) + 2}"
        slide.rels.append(
            (
                rid,
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                f"../media/{media_name}",
            )
        )
        with Image.open(source) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        fw = iw * scale
        fh = ih * scale
        fx = x + (w - fw) / 2
        fy = y + (h - fh) / 2
        sid = self.shape_id()
        border_xml = (
            f"<a:ln w=\"12700\"><a:solidFill><a:srgbClr val=\"{border}\"/></a:solidFill></a:ln>"
            if border
            else ""
        )
        return f"""
      <p:pic>
        <p:nvPicPr><p:cNvPr id="{sid}" name="{name} {sid}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr>
          <a:xfrm><a:off x="{emu(fx)}" y="{emu(fy)}"/><a:ext cx="{emu(fw)}" cy="{emu(fh)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{border_xml}
        </p:spPr>
      </p:pic>"""

    def title(self, slide: Slide, text: str, subtitle: str | None = None) -> None:
        slide.objects.append(
            self.text_box(
                0.45,
                1.32,
                8.4,
                0.62,
                [{"text": text, "size": 25, "bold": True, "color": TEXT}],
                margin=0,
                name="Slide title",
            )
        )
        slide.objects.append(self.line(0.45, 1.98, 11.0, color=PURPLE, height=0.025))

    def page_number(self, slide: Slide, n: int, total: int) -> None:
        slide.objects.append(
            self.text_box(
                12.15,
                7.08,
                0.75,
                0.22,
                [{"text": f"{n} / {total}", "size": 9.2, "color": DARK, "align": "r"}],
                margin=0,
                name="Page number",
            )
        )

    def content_xml(self, slide: Slide) -> str:
        objects = "\n".join(slide.objects)
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {objects}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""

    def rels_xml(self, slide: Slide) -> str:
        rels = [
            (
                "rId1",
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
                f"../slideLayouts/slideLayout{slide.layout}.xml",
            )
        ] + slide.rels
        body = "\n".join(
            f'  <Relationship Id="{rid}" Type="{typ}" Target="{target}"/>' for rid, typ, target in rels
        )
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{body}
</Relationships>
"""


def title_slide(builder: DeckBuilder, total: int) -> Slide:
    slide = Slide(
        title="Title",
        layout=1,
        timing="0:12",
        notes="Open with the system-level claim: this is an integrated BCI-to-robot control pipeline, not only an EEG classifier.",
    )
    builder.reset_shapes()
    slide.objects.append(
        builder.text_box(
            0.50,
            2.10,
            9.8,
            1.05,
            [
                {"text": "A Brain-Computer Interface Control", "size": 27, "bold": True, "color": DARK},
                {"text": "System Design Based on Deep Learning", "size": 27, "bold": True, "color": DARK},
            ],
            margin=0,
            name="Presentation title",
        )
    )
    slide.objects.append(builder.line(0.55, 4.00, 10.7, color=PURPLE, height=0.025))
    slide.objects.append(
        builder.text_box(
            0.55,
            4.75,
            7.6,
            0.95,
            [
                {"text": "Third Year Individual Project Final Presentation", "size": 18, "color": MID},
                {"text": "Zheng Xu | Student ID: 11314389 | Supervisor: Alex Casson", "size": 14.5, "color": MID},
                {"text": "May 2026", "size": 13, "color": MID},
            ],
            margin=0,
            name="Presentation subtitle",
        )
    )
    builder.page_number(slide, 1, total)
    return slide


def build_slides(builder: DeckBuilder, assets: dict[str, Path]) -> list[Slide]:
    total = 16
    slides = [title_slide(builder, total)]

    # Slide 2
    s = Slide("Objective", timing="0:35", notes="Tell the markers exactly what the presentation will cover and that no prior report reading is assumed.")
    builder.reset_shapes()
    builder.title(s, "Objective of This Presentation", "What I will explain in the next 15 minutes")
    agenda = [
        ("1", "Why MI-BCI robotic control is difficult"),
        ("2", "Which datasets and methods were selected"),
        ("3", "How the EEGTransformer decoder works"),
        ("4", "How channel reduction supports OpenBCI"),
        ("5", "How DQN closes the control loop"),
        ("6", "What the results, limits, and next steps show"),
    ]
    for i, (num, text) in enumerate(agenda):
        row, col = divmod(i, 2)
        x = 0.65 + col * 6.1
        y = 2.55 + row * 1.15
        s.objects.append(builder.stat_card(x, y, 0.65, 0.72, num, "", color=PURPLE))
        s.objects.append(
            builder.text_box(
                x + 0.85,
                y + 0.05,
                4.8,
                0.56,
                [{"text": text, "size": 15.5, "bold": True, "color": DARK}],
                margin=0,
            )
        )
    s.objects.append(
        builder.text_box(
            0.65,
            6.02,
            11.55,
            0.48,
            [{"text": "Main message: the contribution is an integrated decoder -> controller -> deployment-oriented pipeline.", "size": 15.5, "bold": True, "color": PURPLE, "align": "c"}],
            fill="F4EFF8",
            line="DDD0E8",
        )
    )
    slides.append(s)

    # Slide 3
    s = Slide("Problem", timing="0:55", notes="Frame the project as a systems problem: noisy perception, fragile open-loop control, and expensive hardware.")
    builder.reset_shapes()
    builder.title(s, "Project Aim and Problem", "From offline EEG labels to reliable robotic control")
    s.objects.append(
        builder.text_box(
            0.65,
            2.42,
            4.1,
            0.95,
            [
                {"text": "Project aim", "size": 16, "bold": True, "color": PURPLE},
                {"text": "Translate motor-imagery EEG into robotic arm commands using deep learning and reinforcement learning.", "size": 13.5, "color": DARK},
            ],
            fill=PANEL,
            line="D9D9D9",
        )
    )
    problems = [
        ("Non-stationary EEG", "Different users and sessions shift the signal distribution."),
        ("Open-loop fragility", "A single misclassified epoch can push the arm in the wrong direction."),
        ("Deployment gap", "High-density 64-channel studies do not map cleanly to low-cost hardware."),
    ]
    colors = [RED, ORANGE, BLUE]
    for i, (head, body) in enumerate(problems):
        x = 0.65 + i * 4.15
        s.objects.append(builder.stat_card(x, 3.82, 1.0, 0.72, str(i + 1), "", color=colors[i]))
        s.objects.append(
            builder.text_box(
                x + 1.12,
                3.77,
                2.75,
                1.02,
                [
                    {"text": head, "size": 14.5, "bold": True, "color": colors[i]},
                    {"text": body, "size": 12.2, "color": DARK},
                ],
                fill="FFFFFF",
                line="D9D9D9",
            )
        )
    s.objects.append(
        builder.text_box(
            0.65,
            5.65,
            11.55,
            0.55,
            [{"text": "The project therefore evaluates an offline/simulated perception -> decision -> action loop, not classification accuracy alone.", "size": 14.5, "bold": True, "color": DARK, "align": "c"}],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    # Slide 4
    s = Slide("Background", timing="0:45", notes="Give the minimum MI-BCI background needed for markers who have not read the report.")
    builder.reset_shapes()
    builder.title(s, "Background: Motor-Imagery BCI", "The signal, task, and control mapping used in this project")
    flow = [
        ("EEG", "Scalp electrodes record microvolt brain activity"),
        ("Motor imagery", "Labels depend on dataset; imagined movement becomes command evidence"),
        ("Mu/Beta rhythms", "8-30 Hz activity over sensorimotor cortex changes"),
        ("Robot command", "Predicted intention becomes a directional control cue"),
    ]
    for i, (head, body) in enumerate(flow):
        x = 0.7 + i * 3.0
        s.objects.append(builder.stat_card(x, 2.52, 0.62, 0.60, str(i + 1), "", color=PURPLE))
        s.objects.append(
            builder.text_box(
                x,
                3.28,
                2.62,
                1.18,
                [
                    {"text": head, "size": 15, "bold": True, "color": DARK, "align": "c"},
                    {"text": body, "size": 11.5, "color": DARK, "align": "c"},
                ],
                fill=PANEL,
                line="D9D9D9",
                margin=0.05,
            )
        )
        if i < 3:
            s.objects.append(builder.line(x + 2.72, 2.82, 0.52, color=GOLD, height=0.04))
    s.objects.append(
        builder.text_box(
            0.8,
            5.30,
            5.4,
            0.85,
            [
                {"text": "Why motor imagery?", "size": 16, "bold": True, "color": PURPLE},
                {"text": "It requires no external stimulus and supports voluntary assistive control, but single-trial EEG is noisy and user-dependent.", "size": 12.8, "color": DARK},
            ],
            fill="F4EFF8",
            line="DDD0E8",
        )
    )
    s.objects.append(
        builder.text_box(
            6.75,
            5.30,
            5.4,
            0.85,
            [
                {"text": "Control challenge", "size": 16, "bold": True, "color": PURPLE},
                {"text": "A robot must recover from wrong cues over time, so the controller must be closed-loop rather than a fixed label-to-action map.", "size": 12.8, "color": DARK},
            ],
            fill="F4EFF8",
            line="DDD0E8",
        )
    )
    slides.append(s)

    # Slide 5
    s = Slide("Literature methods", timing="0:55", notes="Keep literature concise: why existing classifiers are useful, what they miss, and why this project adds a controller.")
    builder.reset_shapes()
    builder.title(s, "Literature Review: Method Gap", "Existing methods improve decoding, but usually stop before closed-loop robotic control")
    rows = [
        ("CSP / FBCSP", "Interpretable spatial filters", "Sensitive to covariance drift"),
        ("EEGNet / CNNs", "Learn compact spatial-temporal filters", "Mostly independent single-trial decisions"),
        ("EEG-Conformer / ATCNet", "Stronger CNN-attention decoders", "Classifier output still treated as endpoint"),
        ("RL for EEG", "Useful for feature selection or adaptation", "Less often used for actuator control"),
    ]
    for i, (method, strength, gap) in enumerate(rows):
        y = 2.45 + i * 0.83
        s.objects.append(builder.text_box(0.68, y, 2.55, 0.55, [{"text": method, "size": 13.5, "bold": True, "color": PURPLE}], fill=PANEL, line="D9D9D9", margin=0.05))
        s.objects.append(builder.text_box(3.35, y, 3.55, 0.55, [{"text": strength, "size": 12.2, "color": DARK}], fill="FFFFFF", line="D9D9D9", margin=0.05))
        s.objects.append(builder.text_box(7.04, y, 4.65, 0.55, [{"text": gap, "size": 12.2, "color": DARK}], fill="FFFFFF", line="D9D9D9", margin=0.05))
    s.objects.append(builder.stat_card(0.8, 5.95, 2.1, 0.72, "73.80%", "Ours on BCI IV-2a", color=PURPLE))
    s.objects.append(builder.stat_card(3.25, 5.95, 2.1, 0.72, "77.7%", "EEG-Conformer", color=BLUE))
    s.objects.append(builder.stat_card(5.70, 5.95, 2.1, 0.72, "80.5%", "ATCNet", color=BLUE))
    s.objects.append(
        builder.text_box(
            8.25,
            5.88,
            3.85,
            0.86,
            [{"text": "Positioning: indicative literature context, not a direct leaderboard; the main contribution is the closed-loop pipeline.", "size": 12.2, "bold": True, "color": DARK}],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    # Slide 6
    s = Slide("Datasets", timing="1:00", notes="Explain why the selected datasets are complementary and why other public datasets were not added.")
    builder.reset_shapes()
    builder.title(s, "Dataset and Method Choice", "Three datasets test different deployment pressures")
    s.objects.append(builder.picture(s, assets["dataset"], 6.05, 2.15, 5.9, 3.98, name="Dataset comparison"))
    dataset_cards = [
        ("BCI IV-2a", "22 ch | 4 classes | 9 subjects", "Standard multi-class benchmark for literature comparison."),
        ("BCI IV-2b", "3 ch | 2 classes | 9 subjects", "Minimal-channel stress test under strong hardware constraint."),
        ("PhysioNet EEGMMIDB", "64 ch | 4 classes | 109 subjects", "Large subject pool for cross-subject pre-training and fine-tuning."),
    ]
    for i, (name, meta, reason) in enumerate(dataset_cards):
        s.objects.append(
            builder.text_box(
                0.65,
                2.22 + i * 1.16,
                4.95,
                0.90,
                [
                    {"text": name, "size": 14.5, "bold": True, "color": PURPLE},
                    {"text": meta, "size": 11.3, "bold": True, "color": MID},
                    {"text": reason, "size": 11.5, "color": DARK},
                ],
                fill=PANEL,
                line="D9D9D9",
                margin=0.06,
            )
        )
    s.objects.append(
        builder.text_box(
            0.65,
            5.88,
            11.3,
            0.48,
            [{"text": "The datasets are evaluated independently because their montages, class definitions, and published protocols differ.", "size": 12.8, "bold": True, "color": DARK, "align": "c"}],
            fill="F4EFF8",
            line="DDD0E8",
        )
    )
    slides.append(s)

    # Slide 7
    s = Slide("Architecture", timing="1:00", notes="Walk left to right through EEG processing, EEGTransformer, DQN, and hardware interfaces.")
    builder.reset_shapes()
    builder.title(s, "Overall System Architecture", "Signal preparation -> decoded intention -> closed-loop control")
    s.objects.append(builder.picture(s, assets["architecture"], 0.55, 2.28, 11.9, 3.95, name="System architecture"))
    s.objects.append(
        builder.text_box(
            0.75,
            6.30,
            11.3,
            0.36,
            [{"text": "Classifier output is noisy DQN evidence; OpenBCI was interface/synthetic-board validated, not live EEG controlled.", "size": 12.3, "bold": True, "color": PURPLE, "align": "c"}],
            margin=0,
        )
    )
    slides.append(s)

    # Slide 8
    s = Slide("Preprocessing", timing="1:00", notes="Use the scoped ablation evidence to justify bandpass-only preprocessing and avoid overclaiming ICA findings.")
    builder.reset_shapes()
    builder.title(s, "Preprocessing and Evidence", "Bandpass filtering had the strongest support; ICA benefit was small and inconsistent")
    steps = [
        ("Raw EEG", "EDF / MAT loading, channel standardisation"),
        ("8-30 Hz bandpass", "Keeps motor-imagery Mu/Beta information"),
        ("Epoch + resample", "Fixed input tensor for EEGTransformer"),
        ("Normalise", "Training statistics reused on held-out data"),
    ]
    for i, (head, body) in enumerate(steps):
        x = 0.68 + i * 3.0
        s.objects.append(
            builder.text_box(
                x,
                2.45,
                2.55,
                1.10,
                [
                    {"text": head, "size": 14.2, "bold": True, "color": PURPLE, "align": "c"},
                    {"text": body, "size": 11.2, "color": DARK, "align": "c"},
                ],
                fill=PANEL,
                line="D9D9D9",
                margin=0.06,
            )
        )
        if i < 3:
            s.objects.append(builder.line(x + 2.65, 2.98, 0.48, color=GOLD, height=0.04))
    s.objects.append(builder.stat_card(1.10, 4.55, 2.6, 0.95, "+18.44%", "5-subject PhysioNet filter ablation", color=GREEN))
    s.objects.append(builder.stat_card(4.10, 4.55, 2.6, 0.95, "+1.51%", "3-subject BCI IV-2a ICA test", color=ORANGE))
    s.objects.append(
        builder.text_box(
            7.20,
            4.45,
            4.65,
            1.18,
            [
                {"text": "Resulting choice", "size": 16, "bold": True, "color": PURPLE},
                {"text": "Use a bandpass-only pipeline for the main experiments. ICA is reported as limited-scope evidence, not a universal conclusion.", "size": 12.2, "color": DARK},
            ],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    # Slide 9
    s = Slide("EEGTransformer", timing="0:50", notes="Explain architecture at concept level, not every layer: CNN spatial bias plus Transformer temporal context.")
    builder.reset_shapes()
    builder.title(s, "EEGTransformer Design", "A CNN-Transformer decoder for trial-scale MI-EEG classification")
    s.objects.append(builder.picture(s, assets["eegtransformer"], 0.55, 2.15, 8.3, 3.25, name="EEGTransformer architecture"))
    callouts = [
        ("CNN front-end", "Learns temporal filters and data-driven spatial filters."),
        ("Transformer encoder", "Models long-range structure across the MI window."),
        ("Residual fusion", "Combines local EEG features with global context before classification."),
    ]
    for i, (head, body) in enumerate(callouts):
        s.objects.append(
            builder.text_box(
                9.05,
                2.20 + i * 1.02,
                3.15,
                0.78,
                [
                    {"text": head, "size": 13.8, "bold": True, "color": PURPLE},
                    {"text": body, "size": 11.3, "color": DARK},
                ],
                fill=PANEL,
                line="D9D9D9",
                margin=0.05,
            )
        )
    s.objects.append(builder.stat_card(1.20, 5.88, 2.0, 0.72, "64 x 1000", "Main PhysioNet input", color=PURPLE))
    s.objects.append(builder.stat_card(3.65, 5.88, 2.0, 0.72, "15 tokens", "After pooling", color=PURPLE))
    s.objects.append(builder.stat_card(6.10, 5.88, 2.0, 0.72, "4 classes", "MI command output", color=PURPLE))
    slides.append(s)

    # Slide 10
    s = Slide("Transfer learning", timing="0:50", notes="Explain cross-subject pre-training followed by subject-specific fine-tuning as the calibration strategy.")
    builder.reset_shapes()
    builder.title(s, "Training and Transfer Learning", "Cross-subject model first, then participant-specific calibration")
    s.objects.append(builder.picture(s, assets["finetune"], 4.15, 2.15, 7.8, 3.22, name="Fine tuning comparison"))
    s.objects.append(builder.stat_card(0.85, 2.35, 2.6, 0.95, "56.54%", "Pooled cross-subject PhysioNet", color=ORANGE))
    s.objects.append(builder.stat_card(0.85, 3.55, 2.6, 0.95, "88.78%", "Fine-tuned mean | 10 subjects", color=GREEN))
    s.objects.append(builder.stat_card(0.85, 4.75, 2.6, 0.95, "+32 pts", "Gap reduced in fine-tuning eval", color=PURPLE))
    s.objects.append(
        builder.text_box(
            4.35,
            5.80,
            7.3,
            0.56,
            [{"text": "Interpretation: fine-tuning adapts a transferable MI representation; broader full-cohort validation remains future work.", "size": 12.2, "bold": True, "color": DARK, "align": "c"}],
            fill="F4EFF8",
            line="DDD0E8",
        )
    )
    slides.append(s)

    # Slide 11
    s = Slide("Classification results", timing="1:00", notes="State the classification result carefully: competitive and sufficient for the system study, not best-in-literature.")
    builder.reset_shapes()
    builder.title(s, "Classification Results: Useful Decoder, Not SOTA", "Strong enough decoder for the closed-loop control study")
    s.objects.append(builder.stat_card(0.80, 2.30, 2.2, 0.95, "73.80%", "BCI IV-2a | 22 ch | 4 class", color=PURPLE))
    s.objects.append(builder.stat_card(3.30, 2.30, 2.2, 0.95, "82.87%", "BCI IV-2b | 3 ch | 2 class", color=PURPLE))
    s.objects.append(builder.stat_card(5.80, 2.30, 2.2, 0.95, "88.78%", "PhysioNet | fine-tuned", color=PURPLE))
    rows = [
        ("FBCSP + SVM", "67.6%", "Classical baseline"),
        ("EEGNet / ShallowConvNet", "68-72%", "Compact CNN family"),
        ("EEG-Conformer", "77.7%", "CNN + Transformer"),
        ("ATCNet", "80.5%", "Attention temporal CNN"),
        ("EEGTransformer (ours)", "73.80%", "Competitive, system-oriented"),
    ]
    for i, (method, acc, note) in enumerate(rows):
        y = 3.75 + i * 0.45
        fill = "F4EFF8" if "ours" in method else "FFFFFF"
        s.objects.append(builder.text_box(0.85, y, 4.0, 0.33, [{"text": method, "size": 10.7, "bold": "ours" in method, "color": DARK}], fill=fill, line="D9D9D9", margin=0.02))
        s.objects.append(builder.text_box(4.92, y, 1.25, 0.33, [{"text": acc, "size": 10.7, "bold": True, "color": PURPLE, "align": "c"}], fill=fill, line="D9D9D9", margin=0.02))
        s.objects.append(builder.text_box(6.25, y, 5.2, 0.33, [{"text": note, "size": 10.7, "color": DARK}], fill=fill, line="D9D9D9", margin=0.02))
    s.objects.append(
        builder.text_box(
            8.40,
            2.30,
            3.15,
            0.95,
            [{"text": "Literature context: competitive with CNN baselines, below stronger recent attention models; contribution is system integration.", "size": 11.8, "bold": True, "color": DARK}],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    # Slide 12
    s = Slide("Channel reduction", timing="1:00", notes="Emphasise offline evidence for OpenBCI-compatible channel reduction and clearly state the validation boundary.")
    builder.reset_shapes()
    builder.title(s, "OpenBCI-Oriented Channel Reduction", "Offline mapping from 64-channel laboratory EEG to an 8-channel consumer setup")
    s.objects.append(builder.picture(s, assets["channel_reduction"], 0.55, 2.08, 6.25, 3.8, name="Channel reduction"))
    s.objects.append(builder.picture(s, assets["channel_importance"], 7.05, 2.08, 5.05, 2.75, name="Channel importance"))
    s.objects.append(builder.stat_card(7.15, 5.10, 1.55, 0.75, "C3", "Largest ablation drop", color=PURPLE))
    s.objects.append(builder.stat_card(8.92, 5.10, 1.55, 0.75, "8 ch", "OpenBCI-compatible montage", color=PURPLE))
    s.objects.append(builder.stat_card(10.70, 5.10, 1.55, 0.75, "72.54%", "Fine-tuned accuracy", color=GREEN))
    s.objects.append(
        builder.text_box(
            0.78,
            6.15,
            11.2,
            0.44,
            [{"text": "Boundary: this is offline evidence for an 8-channel montage; native live OpenBCI validation is future work.", "size": 12.7, "bold": True, "color": DARK, "align": "c"}],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    # Slide 13
    s = Slide("RL method", timing="1:00", notes="Show the robot-control formulation; the robot GIF plays in PPTX, while exported PDF shows a static frame.")
    builder.reset_shapes()
    builder.title(s, "Robotic Control and RL Method", "The controller learns to correct noisy intention cues over time")
    s.objects.append(builder.picture(s, assets["robot_frame"], 0.70, 2.10, 4.35, 2.45, name="Physical robot arm frame", border="D9D9D9"))
    s.objects.append(builder.picture(s, assets["robot_gif"], 0.11, 4.72, 2.64, 2.64, name="Robot arm animated GIF", border="D9D9D9"))
    s.objects.append(
        builder.text_box(
            2.85,
            4.95,
            2.25,
            1.15,
            [{"text": "Top: physical SO-101 interface frame. Bottom: animated control GIF plays in PPTX; PDF shows a static frame.", "size": 10.2, "color": MID}],
            margin=0.02,
        )
    )
    mdp = [
        ("State", "End-effector position, target position, distance, optional EEG class + confidence"),
        ("Action", "Four discrete directions: left, right, up, down"),
        ("Reward", "Target reaching, distance improvement, step penalty, boundary penalty"),
        ("Environment", "PyBullet / planar reaching simplification before richer SO-101 manipulation"),
    ]
    for i, (head, body) in enumerate(mdp):
        s.objects.append(
            builder.text_box(
                5.45,
                2.10 + i * 0.88,
                6.45,
                0.64,
                [
                    {"text": head, "size": 13.5, "bold": True, "color": PURPLE},
                    {"text": body, "size": 11.3, "color": DARK},
                ],
                fill=PANEL if i % 2 == 0 else "FFFFFF",
                line="D9D9D9",
                margin=0.05,
            )
        )
    s.objects.append(
        builder.text_box(
            5.45,
            5.95,
            6.45,
            0.48,
            [{"text": "Controller role: use the current state to correct transient EEG misclassifications instead of blindly executing each label.", "size": 12.5, "bold": True, "color": DARK, "align": "c"}],
            fill="F4EFF8",
            line="DDD0E8",
        )
    )
    slides.append(s)

    # Slide 14
    s = Slide("RL results", timing="1:00", notes="Compare the three DQN architectures under this simulated 2D reaching protocol and present Light Transformer as the practical trade-off.")
    builder.reset_shapes()
    builder.title(s, "RL Results", "Transformer-based policies reached the best simulated 2D control performance")
    s.objects.append(builder.picture(s, assets["rl_comparison"], 0.55, 2.10, 6.55, 4.0, name="RL architecture comparison"))
    s.objects.append(builder.stat_card(7.45, 2.40, 2.0, 0.85, "97%", "CNN+LSTM final reach rate", color=ORANGE))
    s.objects.append(builder.stat_card(9.80, 2.40, 2.0, 0.85, "99%", "Light Transformer reach rate", color=GREEN))
    s.objects.append(builder.stat_card(7.45, 3.70, 2.0, 0.85, "100%", "2D benchmark reach rate", color=GREEN))
    s.objects.append(builder.stat_card(9.80, 3.70, 2.0, 0.85, "10.39", "Transformer final reward", color=PURPLE))
    s.objects.append(
        builder.text_box(
            7.45,
            5.18,
            4.35,
            0.86,
            [{"text": "Under this simulated reaching protocol, the full Transformer scores highest; Light Transformer is a practical trade-off.", "size": 12.2, "bold": True, "color": DARK}],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    # Slide 15
    s = Slide("End-to-end", timing="1:00", notes="State the central interpretation: 82.22% decoding does not cap simulated control because the loop can recover; EEG mainly accelerates training.")
    builder.reset_shapes()
    builder.title(s, "End-to-End Results", "Closed-loop control remains stable with imperfect EEG predictions")
    s.objects.append(builder.stat_card(0.80, 2.28, 1.95, 0.82, "82.22%", "EEG accuracy on 630 trials", color=ORANGE))
    s.objects.append(builder.stat_card(3.05, 2.28, 1.95, 0.82, "98.7%", "EEG-aware reach rate", color=GREEN))
    s.objects.append(builder.stat_card(5.30, 2.28, 1.95, 0.82, "99.0%", "State-only reach rate", color=GREEN))
    s.objects.append(builder.stat_card(7.55, 2.28, 1.95, 0.82, "668s", "With EEG state", color=PURPLE))
    s.objects.append(builder.stat_card(9.80, 2.28, 1.95, 0.82, "1130s", "Without EEG state", color=BLUE))
    s.objects.append(
        builder.text_box(
            0.85,
            3.58,
            5.35,
            1.72,
            [
                {"text": "Why control can exceed decoding", "size": 15.0, "bold": True, "color": PURPLE},
                {"text": "The DQN does not execute each class label blindly.", "size": 11.7, "color": DARK, "bullet": True},
                {"text": "It observes state, target distance, class, and confidence over multiple steps.", "size": 11.7, "color": DARK, "bullet": True},
                {"text": "Reward feedback allows wrong directional cues to be corrected.", "size": 11.7, "color": DARK, "bullet": True},
            ],
            fill=PANEL,
            line="D9D9D9",
        )
    )
    s.objects.append(
        builder.text_box(
            6.50,
            3.58,
            5.35,
            1.72,
            [
                {"text": "What EEG added in this test", "size": 15.0, "bold": True, "color": PURPLE},
                {"text": "Final reach rate was similar: 98.7% with EEG vs 99.0% state-only.", "size": 11.7, "color": DARK, "bullet": True},
                {"text": "Training was faster with EEG evidence: 668 s vs 1130 s.", "size": 11.7, "color": DARK, "bullet": True},
                {"text": "This supports usefulness, not proof that EEG is essential in the simplified task.", "size": 11.7, "color": DARK, "bullet": True},
            ],
            fill="FFFFFF",
            line="D9D9D9",
        )
    )
    s.objects.append(
        builder.text_box(
            0.85,
            5.88,
            11.35,
            0.48,
            [
                {"text": "Main interpretation: closed-loop DQN compensates for imperfect EEG; EEG evidence mainly accelerates learning in this simulation.", "size": 12.0, "bold": True, "color": DARK, "align": "c"},
            ],
            fill="FFF7D6",
            line="E6D27A",
            margin=0.04,
        )
    )
    slides.append(s)

    # Slide 16
    s = Slide("Conclusion", timing="1:00", notes="Close with contributions, validation boundaries, and next steps. Do not start Q&A content here.")
    builder.reset_shapes()
    builder.title(s, "Discussion, Conclusion, and Next Steps", "What was proven, what was not, and how the work should continue")
    shown = [
        "EEGTransformer works competitively: 73.80%, 82.87%, 88.78%",
        "Fine-tuning reduces cross-subject weakness: 56.54% -> 88.78%",
        "8-channel montage remains usable offline: 72.54%",
        "End-to-end simulation: 98.7% EEG-aware, 99.0% state-only",
    ]
    not_shown = [
        "No live human-subject EEG control",
        "BrainFlow was synthetic-board interface validation",
        "8-channel OpenBCI result is offline, not native live deployment",
        "4 s online window may be too slow for real-time use",
    ]
    next_steps = [
        "Ethics-approved live-user OpenBCI study",
        "Native real-time 8-channel inference path",
        "Shorter overlapping windows or asynchronous MI",
        "Richer SO-101 manipulation beyond planar reaching",
    ]
    panels = [
        ("What was shown", shown, PANEL),
        ("What was not shown", not_shown, "FFF7D6"),
        ("Next steps", next_steps, "FFFFFF"),
    ]
    for i, (head, items, fill) in enumerate(panels):
        x = 0.65 + i * 3.92
        s.objects.append(
            builder.text_box(
                x,
                2.25,
                3.55,
                3.10,
                [{"text": head, "size": 15.0, "bold": True, "color": PURPLE}]
                + [{"text": item, "size": 10.9, "color": DARK, "bullet": True} for item in items],
                fill=fill,
                line="D9D9D9",
            )
        )
    s.objects.append(
        builder.text_box(
            0.65,
            5.95,
            11.55,
            0.48,
            [{"text": "Validation boundary: EEG classification is offline; control is simulated; live human closed-loop testing requires ethical approval.", "size": 12.6, "bold": True, "color": DARK, "align": "c"}],
            fill="FFF7D6",
            line="E6D27A",
        )
    )
    slides.append(s)

    for n, slide in enumerate(slides[1:], start=2):
        builder.page_number(slide, n, total)
    return slides


def presentation_xml(total: int) -> str:
    sld_ids = "\n".join(f'    <p:sldId id="{256 + i}" r:id="rId{3 + i}"/>' for i in range(total))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId1"/>
    <p:sldMasterId id="2147483660" r:id="rId2"/>
  </p:sldMasterIdLst>
  <p:notesMasterIdLst><p:notesMasterId r:id="rId19"/></p:notesMasterIdLst>
  <p:handoutMasterIdLst><p:handoutMasterId r:id="rId20"/></p:handoutMasterIdLst>
  <p:sldIdLst>
{sld_ids}
  </p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle>
    <a:defPPr><a:defRPr lang="en-US"/></a:defPPr>
    <a:lvl1pPr marL="0" algn="l" defTabSz="914400"><a:defRPr sz="1800" kern="1200"><a:solidFill><a:schemeClr val="tx1"/></a:solidFill><a:latin typeface="+mn-lt"/></a:defRPr></a:lvl1pPr>
  </p:defaultTextStyle>
</p:presentation>
"""


def presentation_rels(total: int) -> str:
    rels = [
        ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster1.xml"),
        ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster", "slideMasters/slideMaster2.xml"),
    ]
    rels += [
        (f"rId{3 + i}", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide", f"slides/slide{i + 1}.xml")
        for i in range(total)
    ]
    rels += [
        ("rId19", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesMaster", "notesMasters/notesMaster1.xml"),
        ("rId20", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/handoutMaster", "handoutMasters/handoutMaster1.xml"),
        ("rId21", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps", "presProps.xml"),
        ("rId22", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/viewProps", "viewProps.xml"),
        ("rId23", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme", "theme/theme1.xml"),
        ("rId24", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/tableStyles", "tableStyles.xml"),
    ]
    body = "\n".join(f'  <Relationship Id="{rid}" Type="{typ}" Target="{target}"/>' for rid, typ, target in rels)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{body}
</Relationships>
"""


def update_content_types(path: Path, total: int) -> None:
    text = path.read_text(encoding="utf-8")
    if 'Extension="gif"' not in text:
        text = text.replace(
            "</Types>",
            '  <Default Extension="gif" ContentType="image/gif"/>\n</Types>',
        )
    text = re.sub(
        r'\s*<Override PartName="/ppt/slides/slide\d+\.xml" ContentType="application/vnd\.openxmlformats-officedocument\.presentationml\.slide\+xml"\s*/>',
        "",
        text,
    )
    slide_overrides = "\n".join(
        f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, total + 1)
    )
    text = text.replace("</Types>", f"\n{slide_overrides}\n</Types>")
    path.write_text(text, encoding="utf-8")


def update_document_properties(tmpdir: Path, total: int) -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>A Brain-Computer Interface Control System Design Based on Deep Learning</dc:title>
  <dc:creator>Zheng Xu</dc:creator>
  <cp:lastModifiedBy>Zheng Xu</cp:lastModifiedBy>
  <cp:revision>1</cp:revision>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""
    (tmpdir / "docProps" / "core.xml").write_text(core, encoding="utf-8")

    app_path = tmpdir / "docProps" / "app.xml"
    app = app_path.read_text(encoding="utf-8")
    app = re.sub(r"<TotalTime>\d+</TotalTime>", "<TotalTime>15</TotalTime>", app)
    app = re.sub(r"<Slides>\d+</Slides>", f"<Slides>{total}</Slides>", app)
    app = re.sub(r"<Words>\d+</Words>", "<Words>950</Words>", app)
    app = re.sub(
        r"<vt:lpstr>PowerPoint Presentation</vt:lpstr>",
        "<vt:lpstr>A Brain-Computer Interface Control System Design Based on Deep Learning</vt:lpstr>",
        app,
        count=1,
    )
    app_path.write_text(app, encoding="utf-8")


def write_notes(slides: list[Slide]) -> None:
    total_seconds = 0
    lines = [
        "# Final Project Presentation Timing Notes",
        "",
        "Target: keep the spoken section under 15 minutes. The title slide is deliberately short; each content slide is planned around one minute.",
        "",
        "| Slide | Timing | Speaking goal |",
        "|---:|---:|---|",
    ]
    for i, slide in enumerate(slides, start=1):
        mm, ss = slide.timing.split(":")
        total_seconds += int(mm) * 60 + int(ss)
        lines.append(f"| {i} | {slide.timing} | {slide.notes} |")
    lines.extend(
        [
            "",
            f"Planned total: {total_seconds // 60}:{total_seconds % 60:02d}.",
            "",
            "Important claim boundaries:",
            "- Do not claim the EEG classifier is state of the art on BCI IV-2a; it is competitive and sufficient for the system study.",
            "- State clearly that live human closed-loop testing was not performed because ethical approval was not available.",
            "- Treat the 8-channel OpenBCI result as offline evidence, not as a completed native live deployment.",
            "- Explain that EEG input accelerated DQN training, while final reach rates for EEG-aware and state-only agents were similar.",
        ]
    )
    OUT_NOTES.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    build_assets = ROOT / "01_Reports" / "_presentation_assets"
    assets = {
        "architecture": ROOT / "01_Reports" / "architecture_diagram.png",
        "dataset": ROOT / "03_Experiments" / "dataset_comparison" / "dataset_comparison.png",
        "finetune": ROOT / "03_Experiments" / "physionet_ctnet_finetune" / "finetune_comparison.png",
        "channel_reduction": ROOT / "03_Experiments" / "Channel_Reduction" / "channel_reduction" / "channel_reduction_comparison.png",
        "channel_importance": ROOT / "03_Experiments" / "Channel_Reduction" / "channel_reduction" / "importance" / "channel_importance_bar.png",
        "rl_comparison": ROOT / "03_Experiments" / "architecture_comparison_v2" / "comparison_v2.png",
        "e2e_overall": ROOT / "03_Experiments" / "E2E_Evaluation" / "rl_control_test" / "overall_performance.png",
        "eegtransformer": ensure_png_from_pdf(
            ROOT / "01_Reports" / "plotneuralnet_eegtransformer_v13" / "eegtransformer_plotnn.pdf",
            build_assets / "eegtransformer_plotnn.png",
        ),
        "robot_gif": ROOT / "05_Documentation" / "rl_arm.gif",
        "robot_frame": ensure_video_frame(
            ROOT / "05_Documentation" / "phy_control_arm.mp4",
            build_assets / "phy_control_arm_frame.png",
        ),
    }

    builder = DeckBuilder()
    slides = build_slides(builder, assets)
    total = len(slides)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(TEMPLATE, "r") as zf:
            zf.extractall(tmpdir)

        slides_dir = tmpdir / "ppt" / "slides"
        rels_dir = slides_dir / "_rels"
        for old in slides_dir.glob("slide*.xml"):
            old.unlink()
        for old in rels_dir.glob("slide*.xml.rels"):
            old.unlink()

        for i, slide in enumerate(slides, start=1):
            (slides_dir / f"slide{i}.xml").write_text(builder.content_xml(slide), encoding="utf-8")
            (rels_dir / f"slide{i}.xml.rels").write_text(builder.rels_xml(slide), encoding="utf-8")

        (tmpdir / "ppt" / "presentation.xml").write_text(presentation_xml(total), encoding="utf-8")
        (tmpdir / "ppt" / "_rels" / "presentation.xml.rels").write_text(presentation_rels(total), encoding="utf-8")
        update_content_types(tmpdir / "[Content_Types].xml", total)
        update_document_properties(tmpdir, total)

        media_dir = tmpdir / "ppt" / "media"
        media_dir.mkdir(exist_ok=True)
        for source, media_name in builder.assets.items():
            shutil.copy2(source, media_dir / media_name)

        OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(OUT_PPTX, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(tmpdir))

    write_notes(slides)
    print(f"Wrote {OUT_PPTX}")
    print(f"Wrote {OUT_NOTES}")


if __name__ == "__main__":
    main()
