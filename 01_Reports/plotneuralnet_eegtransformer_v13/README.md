`eegtransformer_plotnn.py` generates a standalone `PlotNeuralNet` diagram for the EEGTransformer subfigure.

Build from this directory with:

```bash
python eegtransformer_plotnn.py
pdflatex -interaction=nonstopmode -halt-on-error eegtransformer_plotnn.tex
```

The figure is aligned to the main PhysioNet configuration described in [final_report_draft_v11.tex](/home/tomato/3YP/01_Reports/final_report_draft_v11.tex:457):

- `F1 = 20`
- `D = 2`
- temporal kernels `64` and `16`
- pooling sizes `8` and `8`
- token embedding dimension `40`
- transformer depth `6`
- attention heads `2`

Dimension labels in the rendered figure use:

- `F` for feature maps
- `T` for temporal samples after convolution / pooling
- `E` for token embedding width
- `N` for token count
- `L` for flattened feature length
- `K` for output classes
