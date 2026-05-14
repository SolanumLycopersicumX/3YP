# Final Project Presentation Timing Notes

Target: keep the spoken section under 15 minutes. The title slide is deliberately short; each content slide is planned around one minute.

| Slide | Timing | Speaking goal |
|---:|---:|---|
| 1 | 0:12 | Open with the system-level claim: this is an integrated BCI-to-robot control pipeline, not only an EEG classifier. |
| 2 | 0:35 | Tell the markers exactly what the presentation will cover and that no prior report reading is assumed. |
| 3 | 0:55 | Frame the project as a systems problem: noisy perception, fragile open-loop control, and expensive hardware. |
| 4 | 0:45 | Give the minimum MI-BCI background needed for markers who have not read the report. |
| 5 | 0:55 | Keep literature concise: why existing classifiers are useful, what they miss, and why this project adds a controller. |
| 6 | 1:00 | Explain why the selected datasets are complementary and why other public datasets were not added. |
| 7 | 1:00 | Walk left to right through EEG processing, EEGTransformer, DQN, and hardware interfaces. |
| 8 | 1:00 | Use the scoped ablation evidence to justify bandpass-only preprocessing and avoid overclaiming ICA findings. |
| 9 | 0:50 | Explain architecture at concept level, not every layer: CNN spatial bias plus Transformer temporal context. |
| 10 | 0:50 | Explain cross-subject pre-training followed by subject-specific fine-tuning as the calibration strategy. |
| 11 | 1:00 | State the classification result carefully: competitive and sufficient for the system study, not best-in-literature. |
| 12 | 1:00 | Emphasise offline evidence for OpenBCI-compatible channel reduction and clearly state the validation boundary. |
| 13 | 1:00 | Show the robot-control formulation; the robot GIF plays in PPTX, while exported PDF shows a static frame. |
| 14 | 1:00 | Compare the three DQN architectures under this simulated 2D reaching protocol and present Light Transformer as the practical trade-off. |
| 15 | 1:00 | State the central interpretation: 82.22% decoding does not cap simulated control because the loop can recover; EEG mainly accelerates training. |
| 16 | 1:00 | Close with contributions, validation boundaries, and next steps. Do not start Q&A content here. |

Planned total: 14:02.

Important claim boundaries:
- Do not claim the EEG classifier is state of the art on BCI IV-2a; it is competitive and sufficient for the system study.
- State clearly that live human closed-loop testing was not performed because ethical approval was not available.
- Treat the 8-channel OpenBCI result as offline evidence, not as a completed native live deployment.
- Explain that EEG input accelerated DQN training, while final reach rates for EEG-aware and state-only agents were similar.
