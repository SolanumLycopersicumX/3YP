#!/usr/bin/env python3
"""Generate speaker script and viva Q&A PDF for the final presentation."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_TEX = ROOT / "01_Reports" / "final_project_presentation_speech_and_qa_11314389.tex"
OUT_PDF = ROOT / "01_Reports" / "final_project_presentation_speech_and_qa_11314389.pdf"


SLIDES = [
    {
        "n": 1,
        "title": "Title",
        "time": "0:12",
        "script": (
            "Good morning. My project is titled A Brain-Computer Interface Control System "
            "Design Based on Deep Learning. The main point is that this is not only an EEG "
            "classification project. It is a system design project that links motor-imagery "
            "EEG decoding, closed-loop reinforcement-learning control, and deployment-oriented "
            "channel reduction."
        ),
        "detail": "Make the system-level framing clear immediately.",
    },
    {
        "n": 2,
        "title": "Objective of This Presentation",
        "time": "0:35",
        "script": (
            "I will cover six points. First, why MI-BCI robotic control is difficult. Second, "
            "which datasets and methods I used. Third, how the EEGTransformer decoder works. "
            "Fourth, why channel reduction matters for OpenBCI-style hardware. Fifth, how the "
            "DQN closes the control loop. Finally, I will explain what the results prove, what "
            "they do not prove, and what the next steps should be."
        ),
        "detail": "Do not spend time reading every item; give the audience the route map.",
    },
    {
        "n": 3,
        "title": "Project Aim and Problem",
        "time": "0:55",
        "script": (
            "The project aim is to translate motor-imagery EEG into robotic arm commands using "
            "deep learning and reinforcement learning. The difficulty is not just classification. "
            "There are three connected problems: EEG is non-stationary across users and sessions, "
            "open-loop control is fragile because one wrong class can become one wrong command, "
            "and many strong EEG systems rely on high-density lab equipment. My project treats "
            "this as a pipeline problem: perception, decision-making, and deployability."
        ),
        "detail": "If asked, say the final task metric is control success, not only offline accuracy.",
    },
    {
        "n": 4,
        "title": "Background: Motor-Imagery BCI",
        "time": "0:45",
        "script": (
            "Motor imagery means the user imagines movement without physically moving. In EEG, "
            "this changes Mu and Beta rhythms, especially in the 8 to 30 Hz range over the "
            "sensorimotor cortex. A basic BCI pipeline records EEG, extracts these patterns, "
            "classifies the imagined action, and maps the result to a command. The problem is "
            "that single-trial EEG is noisy, so a robot needs feedback and correction rather "
            "than blindly following every label."
        ),
        "detail": "Key terms: MI, Mu/Beta, ERD/ERS, noisy single-trial classification.",
    },
    {
        "n": 5,
        "title": "Literature Review: Method Gap",
        "time": "0:55",
        "script": (
            "Classical CSP methods are interpretable, while CNN and CNN-attention models often "
            "improve offline classification. However, many studies still stop at trial-level "
            "decoding. RL has also been explored in EEG pipelines, but often for feature selection "
            "or adaptation rather than controlling a robotic actuator. My gap is therefore a "
            "system gap: previous work often treats EEG decoding, robotic control, and low-channel "
            "deployment separately. This project connects them in one pipeline."
        ),
        "detail": "Do not claim state of the art classification; claim system integration.",
    },
    {
        "n": 6,
        "title": "Dataset and Method Choice",
        "time": "1:00",
        "script": (
            "I selected these datasets because they are open, motor-imagery focused, established "
            "enough for comparison, and complementary. BCI IV-2a gives a standard 22-channel, "
            "4-class benchmark. BCI IV-2b gives an extreme 3-channel binary stress test, which is "
            "useful for low-channel deployment thinking. PhysioNet gives 109 subjects and 64 "
            "channels, so it supports cross-subject pre-training, fine-tuning, and channel reduction. "
            "I did not add datasets such as OpenBMI, High-Gamma, BNCI, or Jeong because they either "
            "duplicate a role already covered here, emphasise a different paradigm, or add extra "
            "preprocessing and comparison factors beyond this project scope."
        ),
        "detail": "Dataset choice was focused coverage, not maximum dataset count; IV-2b is not used for the 4-command control loop because it has only two classes.",
    },
    {
        "n": 7,
        "title": "Overall System Architecture",
        "time": "1:00",
        "script": (
            "The system has four main parts. First, EEG is filtered, segmented, normalised, and "
            "passed to the EEGTransformer. Second, the classifier outputs a motor-imagery class "
            "and confidence. Third, the DQN receives robot state, target information, and optionally "
            "the classifier evidence. Fourth, the selected action drives the simulated robotic "
            "environment or the software interface. The important design choice is that the EEG "
            "class is a noisy observation for the controller, not the final action."
        ),
        "detail": "If asked about hardware: BrainFlow/OpenBCI and SO-101 interfaces were software/interface validated, not live EEG controlled.",
    },
    {
        "n": 8,
        "title": "Preprocessing and Evidence",
        "time": "1:00",
        "script": (
            "The final preprocessing choice is intentionally simple: standard loading, 8 to 30 Hz "
            "bandpass filtering, epoching or resampling, and normalisation using training statistics. "
            "This is justified by the ablations. The bandpass filter improved PhysioNet fine-tuning "
            "by 18.44 percentage points on average in the five-subject test. ICA gave only a small "
            "and inconsistent 1.51 percentage-point average gain on three BCI IV-2a subjects, so I "
            "reported it as tested but did not make it the default."
        ),
        "detail": "Say ICA was limited in this project, not universally useless.",
    },
    {
        "n": 9,
        "title": "EEGTransformer Design",
        "time": "0:50",
        "script": (
            "The EEGTransformer combines two useful biases. The CNN front-end learns local temporal "
            "filters and data-driven spatial filters across electrodes, which is appropriate for EEG. "
            "The Transformer encoder then models longer relationships across the motor-imagery trial, "
            "for example cue-aligned activity, sustained ERD, and recovery. In the main PhysioNet "
            "configuration, the input is 64 channels by 1000 samples, which is reduced to 15 tokens "
            "before 4-class classification."
        ),
        "detail": "CNN handles spatial/local temporal structure; Transformer handles within-trial temporal context.",
    },
    {
        "n": 10,
        "title": "Training and Transfer Learning",
        "time": "0:50",
        "script": (
            "Cross-subject generalisation is a major weakness in EEG. The pooled PhysioNet model "
            "achieved 56.54 percent accuracy, showing that one general model does not fully handle "
            "inter-user variability. Subject-specific fine-tuning increased the mean result to "
            "88.78 percent on the evaluated subjects. My interpretation is that the pooled model "
            "learns transferable MI structure, but each user still needs calibration because EEG "
            "depends on anatomy, electrode placement, and imagery strategy."
        ),
        "detail": "Fine-tuning helps, but full-cohort validation and online adaptation are future work.",
    },
    {
        "n": 11,
        "title": "Classification Results",
        "time": "1:00",
        "script": (
            "The classifier achieved 73.80 percent on BCI IV-2a, 82.87 percent on BCI IV-2b, "
            "and 88.78 percent on fine-tuned PhysioNet. These results show that the decoder is "
            "usable and competitive with classical or compact CNN baselines. I would not claim "
            "it is state of the art, because methods such as EEG-Conformer and ATCNet report "
            "higher results under their own protocols. For this project, the classifier is strong "
            "enough to support the closed-loop control study."
        ),
        "detail": "Best answer if challenged: competitive and system-oriented, not leaderboard-leading.",
    },
    {
        "n": 12,
        "title": "OpenBCI-Oriented Channel Reduction",
        "time": "1:00",
        "script": (
            "The channel reduction study asks whether a 64-channel laboratory montage can move "
            "toward an 8-channel OpenBCI-compatible setup. The final set is C3, C4, FC3, FC4, "
            "CP3, CP4, Cz, and FCz. C3 was especially important, which is neurophysiologically "
            "plausible because it lies over left sensorimotor cortex. After fine-tuning, the "
            "8-channel setup reached 72.54 percent. This supports offline feasibility, but it "
            "does not prove live OpenBCI control."
        ),
        "detail": "Boundary: offline 8-channel evidence, not native live hardware validation.",
    },
    {
        "n": 13,
        "title": "Robotic Control and RL Method",
        "time": "1:00",
        "script": (
            "The controller is formulated as a DQN problem. The base state contains end-effector "
            "position, target position, and distance to the target. In the EEG-augmented version, "
            "the predicted MI class and confidence are added. The four actions are left, right, "
            "up, and down. The reward combines target reaching, distance improvement, a step cost, "
            "and penalties for boundary or unstable movement. This lets the controller learn "
            "corrective actions rather than directly executing every EEG class."
        ),
        "detail": "Quantitative RL results are from simulated reaching, not physical closed-loop EEG control.",
    },
    {
        "n": 14,
        "title": "RL Results",
        "time": "1:00",
        "script": (
            "I compared three DQN architectures under the same simulated 2D reaching protocol. "
            "CNN+LSTM reached 97 percent final evaluation reach rate. Light Transformer reached "
            "99 percent with fewer parameters. The full Transformer DQN reached 100 percent and "
            "the highest final reward, 10.39. I interpret this as evidence that sequential "
            "Transformer-style policies can learn strong reaching behaviour. For deployment, "
            "Light Transformer remains attractive because it gives nearly the same reach rate with "
            "lower model complexity."
        ),
        "detail": "Full Transformer is best result; Light Transformer is the practical trade-off.",
    },
    {
        "n": 15,
        "title": "End-to-End Results",
        "time": "1:00",
        "script": (
            "This is the central result. The EEGTransformer classified 630 PhysioNet trials at "
            "82.22 percent accuracy. In the control loop, the EEG-aware DQN reached 98.7 percent, "
            "while the state-only DQN reached 99.0 percent. This means control success is not capped "
            "by instantaneous EEG accuracy, because the DQN can recover across steps. EEG evidence "
            "helped mainly by accelerating learning: training took 668 seconds with EEG evidence "
            "and 1130 seconds without it. So the honest conclusion is useful learning signal, not "
            "proof that EEG was essential in this simplified task."
        ),
        "detail": "This page is likely to be challenged; be precise about training speed vs final reach rate.",
    },
    {
        "n": 16,
        "title": "Discussion, Conclusion, and Next Steps",
        "time": "1:00",
        "script": (
            "To conclude, the project shows a feasible integrated MI-BCI pipeline: a competitive "
            "EEGTransformer decoder, subject-specific adaptation, offline 8-channel reduction, "
            "and simulated closed-loop DQN control. The main boundary is equally important: EEG "
            "classification is offline, control is simulated, and BrainFlow validation used a "
            "synthetic board. Live human OpenBCI control was not performed because ethical approval "
            "was not available. The next steps are an ethics-approved live-user study, native "
            "8-channel real-time inference, shorter windows, and richer SO-101 manipulation."
        ),
        "detail": "End by showing technical confidence and honesty about validation limits.",
    },
]


POINTED_QA = [
    ("Slide 2: What does an integrated pipeline mean?", "It means EEG decoding, DQN control, channel reduction, and hardware-oriented interfaces are evaluated together rather than treating classification as the endpoint."),
    ("Slide 3: Why is open-loop control fragile?", "Because a wrong EEG class maps directly to a wrong command. Closed-loop control can recover from transient mistakes using feedback."),
    ("Slide 4: Why use 8 to 30 Hz?", "Motor imagery is strongly associated with Mu and Beta rhythms in this band, especially over sensorimotor cortex."),
    ("Slide 5: Is the classifier state of the art?", "No. It is competitive and sufficient for the system study, but some recent methods report higher IV-2a accuracy."),
    ("Slide 6: Why these datasets rather than others?", "They are open, MI-focused, established, and complementary across channel count, class difficulty, and subject scale. OpenBMI, High-Gamma, BNCI, and Jeong-style datasets are valuable, but they either duplicate one selected role, use a less matched paradigm, or add extra comparison factors beyond the project scope."),
    ("Slide 6: Why not merge all datasets?", "They differ in channels, classes, sampling rates, subjects, and protocols, so independent evaluation avoids misleading comparisons."),
    ("Slide 6: Why is IV-2b not used for the control mapping?", "It has only two classes, while the final control task uses four directions mapped from four MI classes."),
    ("Slide 7: Why is the EEG class not the final action?", "Because the classifier is noisy. The DQN treats class and confidence as evidence alongside robot state and target feedback."),
    ("Slide 8: Why not use ICA by default?", "ICA gave only a small, inconsistent average gain in the tested subjects and sometimes reduced accuracy, so bandpass-only was safer for the main pipeline."),
    ("Slide 9: Why CNN plus Transformer?", "The CNN gives EEG-specific spatial and local temporal filtering, while the Transformer models longer within-trial temporal relationships."),
    ("Slide 10: What does 56.54% to 88.78% show?", "It shows that cross-subject generalisation is hard, but subject-specific fine-tuning can adapt the pooled representation effectively."),
    ("Slide 11: How should the three classifier numbers be interpreted?", "73.80%, 82.87%, and 88.78% show usable decoding across complementary settings, not a single unified leaderboard."),
    ("Slide 12: What is the final 8-channel subset?", "C3, C4, FC3, FC4, CP3, CP4, Cz, and FCz."),
    ("Slide 12: Why is C3 important?", "C3 lies over left sensorimotor cortex and is informative for contralateral right-hand motor imagery; its ablation caused the largest drop."),
    ("Slide 13: What are the DQN state, action, and reward?", "State is arm position, target, distance, and optionally EEG class/confidence. Actions are four directions. Reward encourages reaching and distance improvement while penalising inefficient or unsafe movement."),
    ("Slide 14: Which DQN architecture performed best?", "Transformer DQN achieved the best evaluation reach rate at 100% and the highest reward. Light Transformer is the practical lightweight trade-off."),
    ("Slide 15: Why can 82.22% EEG accuracy still give about 99% reach rate?", "Because the DQN uses closed-loop sequential feedback, so one wrong class does not force task failure."),
    ("Slide 15: Did EEG actually help the DQN?", "Yes, but mainly in training speed. The EEG-aware model trained in 668 s versus 1130 s, while final reach rates were similar."),
    ("Slide 16: What exactly was not validated?", "Live human OpenBCI closed-loop control was not validated. The project used offline EEG datasets, PyBullet simulation, and synthetic-board BrainFlow tests."),
]


REFLECTIVE_QA = [
    ("What was the most challenging part of the project?", "The hardest part was designing a robust interface between noisy EEG decoding and sequential control. A motor-imagery prediction is not a clean command; it is a delayed, probabilistic observation affected by subject variability and trial noise. The technical challenge was therefore deciding how to pass class and confidence into the DQN state, and how to shape the reward so the controller could recover from individual wrong labels instead of amplifying them."),
    ("What is the best part of the project?", "The strongest part is the system-level integration. The end-to-end result shows why closing the loop matters: imperfect EEG classification can still support high simulated control success because the controller can correct errors."),
    ("What is the weakest part of the project?", "The weakest part is the mismatch between the EEG decoding stage and the simplified control benchmark. The decoder is trained on cue-based motor-imagery trials, while the controller is evaluated in a simplified 2D reaching task. Technically, the next improvement is to reduce that gap with a native real-time state representation, shorter windows, and a tighter mapping between MI evidence and robot-control decisions."),
    ("Which result would you trust least?", "I would treat the 8-channel result cautiously because it is a channel-subset study rather than a native 8-channel acquisition study. It is promising for OpenBCI-style deployment, but the exact performance may change with electrode placement, impedance, artefacts, and shorter inference windows."),
    ("What would you do next?", "I would build a native 8-channel real-time inference path, test shorter and overlapping windows, add confidence-based rejection, and evaluate whether the DQN still benefits when EEG evidence arrives asynchronously rather than as fixed trial segments."),
    ("What would you improve technically?", "I would reduce latency by testing shorter and overlapping windows, add confidence rejection, and explore asynchronous MI detection so the system is less dependent on a fixed 4-second window."),
    ("If you restarted the project, what would you do differently?", "I would make the deployment constraints part of the first experimental design: train native 8-channel models earlier, define the control-state interface earlier, and measure latency and confidence calibration alongside accuracy from the beginning."),
    ("What is the main risk of overclaiming?", "The main risk is mixing metrics from different stages. Classification accuracy, channel-reduction accuracy, DQN reach rate, and end-to-end training speed answer different questions, so they should be interpreted as complementary evidence rather than one single proof of performance."),
    ("Why is the project still valuable as a staged system study?", "Because it identifies and tests the main technical components needed for deployment: transferable decoding, subject adaptation, channel reduction, controller robustness, confidence-aware control, and software integration boundaries."),
    ("What is the one-sentence conclusion?", "The project shows that EEGTransformer decoding, subject adaptation, OpenBCI-oriented channel reduction, and DQN closed-loop control can form a feasible MI-BCI robotic-control pipeline, with the next technical priority being lower-latency native 8-channel integration."),
]


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def paragraph(text: str) -> str:
    return tex_escape(text) + "\n\n"


def build_tex() -> str:
    planned_seconds = sum(int(s["time"].split(":")[0]) * 60 + int(s["time"].split(":")[1]) for s in SLIDES)
    planned = f"{planned_seconds // 60}:{planned_seconds % 60:02d}"
    buffer_seconds = 15 * 60 - planned_seconds
    buffer = f"{buffer_seconds // 60}:{buffer_seconds % 60:02d}"

    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,margin=18mm]{geometry}",
        r"\usepackage{xcolor}",
        r"\usepackage{enumitem}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue}",
        r"\definecolor{uomPurple}{HTML}{660099}",
        r"\definecolor{lightPanel}{HTML}{F7F7F7}",
        r"\definecolor{warnPanel}{HTML}{FFF7D6}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{5pt}",
        r"\setlist[itemize]{leftmargin=*,itemsep=2pt,topsep=2pt}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries Final Project Presentation Speaker Script and Q\&A}\\",
        r"\vspace{2mm}",
        r"{A Brain-Computer Interface Control System Design Based on Deep Learning}\\",
        r"\vspace{1mm}",
        r"{Zheng Xu | Student ID: 11314389}",
        r"\end{center}",
        "",
        r"\section*{Timing Summary}",
        paragraph(
            f"Planned spoken time: {planned}. This leaves {buffer} under a 15:00 limit, "
            "so it satisfies the requirement to keep at least 30 seconds of safety buffer. "
            "All content slides are designed to be spoken in one minute or less."
        ),
        r"\begin{longtable}{p{12mm}p{32mm}p{18mm}p{96mm}}",
        r"\textbf{Slide} & \textbf{Title} & \textbf{Target} & \textbf{Speaking focus}\\\hline",
    ]

    for slide in SLIDES:
        lines.append(
            f"{slide['n']} & {tex_escape(slide['title'])} & {tex_escape(slide['time'])} & {tex_escape(slide['detail'])}\\\\"
        )

    lines.extend([r"\end{longtable}", r"\newpage", r"\section*{Slide-by-Slide Speaker Script}"])

    for slide in SLIDES:
        lines.append(rf"\subsection*{{Slide {slide['n']}: {tex_escape(slide['title'])} ({tex_escape(slide['time'])})}}")
        lines.append(paragraph(slide["script"]))
        lines.append(r"\textbf{If asked:} " + tex_escape(slide["detail"]) + "\n")

    lines.extend([r"\newpage", r"\section*{Q\&A: Slide-Pointing Detail Questions}"])
    for idx, (q, a) in enumerate(POINTED_QA, start=1):
        lines.append(rf"\subsection*{{Q{idx}. {tex_escape(q)}}}")
        lines.append(paragraph(a))

    lines.extend([r"\newpage", r"\section*{Q\&A: Reflective and Critical Questions}"])
    for idx, (q, a) in enumerate(REFLECTIVE_QA, start=1):
        lines.append(rf"\subsection*{{R{idx}. {tex_escape(q)}}}")
        lines.append(paragraph(a))

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
