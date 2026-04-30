import sys

from pycore.tikzeng import to_begin, to_connection, to_end, to_generate, to_head


def color_preamble():
    return r"""
\usetikzlibrary{calc,fit,positioning}
\definecolor{InputFill}{RGB}{236,239,241}
\definecolor{TemporalFill}{RGB}{214,227,239}
\definecolor{SpatialFill}{RGB}{214,231,224}
\definecolor{PatchFill}{RGB}{239,232,218}
\definecolor{TokenFill}{RGB}{241,225,216}
\definecolor{PosFill}{RGB}{228,220,237}
\definecolor{TransFill}{RGB}{219,227,240}
\definecolor{TransBandFill}{RGB}{199,210,229}
\definecolor{HeadFill}{RGB}{229,223,216}
\definecolor{SoftmaxFill}{RGB}{233,222,228}
\definecolor{SumFill}{RGB}{222,231,226}
\definecolor{ResidualSoft}{RGB}{161,150,182}
\def\InputColor{InputFill}
\def\TemporalColor{TemporalFill}
\def\SpatialColor{SpatialFill}
\def\PatchColor{PatchFill}
\def\TokenColor{TokenFill}
\def\PosColor{PosFill}
\def\TransColor{TransFill}
\def\TransBandColor{TransBandFill}
\def\HeadColor{HeadFill}
\def\SoftmaxColor{SoftmaxFill}
\def\SumColor{SumFill}
\tikzset{
    stage/.style={font=\bfseries\normalsize, text=black!80, fill=none, inner sep=1pt},
    annot/.style={font=\small, text=black!82, align=left, fill=none, inner sep=1pt},
    edgeannot/.style={font=\small, text=black!86, fill=none, inner sep=1pt},
    legendannot/.style={font=\normalsize, text=black!82, align=left, fill=none, inner sep=1pt},
    group/.style={draw=black!35, dashed, rounded corners=6pt, inner sep=10pt}
}
\pgfdeclarelayer{stage0}
\pgfdeclarelayer{flow0}
\pgfdeclarelayer{stage1}
\pgfdeclarelayer{flow1}
\pgfdeclarelayer{stage2}
\pgfdeclarelayer{flow2}
\pgfdeclarelayer{stage3}
\pgfdeclarelayer{flow3}
\pgfdeclarelayer{stage4}
\pgfdeclarelayer{flow4}
\pgfdeclarelayer{stage5}
\pgfdeclarelayer{flow5}
\pgfdeclarelayer{stage6}
\pgfdeclarelayer{flow6}
\pgfdeclarelayer{stage7}
\pgfdeclarelayer{flow7}
\pgfdeclarelayer{stage8}
\pgfdeclarelayer{flow8}
\pgfdeclarelayer{stage9}
\pgfdeclarelayer{flow9}
\pgfdeclarelayer{stage10}
\pgfdeclarelayer{flow10}
\pgfdeclarelayer{stage11}
\pgfdeclarelayer{flow11}
\pgfdeclarelayer{stage12}
\pgfsetlayers{stage0,flow0,stage1,flow1,stage2,flow2,stage3,flow3,stage4,flow4,stage5,flow5,stage6,flow6,stage7,flow7,stage8,flow8,stage9,flow9,stage10,flow10,stage11,flow11,stage12,main}
"""


def box(name, caption, xlabel, zlabel, fill, offset, to, width, height, depth, opacity=1.0, layer=None):
    xlabel_tex = '{{"{0}", }}'.format(xlabel)
    pic = rf"""
\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={xlabel_tex},
        zlabel={zlabel},
        fill={fill},
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};
"""
    if layer:
        return rf"""
\begin{{pgfonlayer}}{{{layer}}}
{pic}\end{{pgfonlayer}}
"""
    return pic


def banded_box(
    name,
    caption,
    xlabel_left,
    xlabel_right,
    zlabel,
    fill,
    bandfill,
    offset,
    to,
    width_left,
    width_right,
    height,
    depth,
    layer=None,
):
    xlabel_tex = '{{"{0}", "{1}"}}'.format(xlabel_left, xlabel_right)
    pic = rf"""
\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={xlabel_tex},
        zlabel={zlabel},
        fill={fill},
        bandfill={bandfill},
        height={height},
        width={{ {width_left}, {width_right} }},
        depth={depth}
        }}
    }};
"""
    if layer:
        return rf"""
\begin{{pgfonlayer}}{{{layer}}}
{pic}\end{{pgfonlayer}}
"""
    return pic


def sum_ball(name, offset, to, radius=2.0, opacity=0.7, layer=None):
    pic = rf"""
\pic[shift={{{offset}}}] at {to}
    {{Ball={{
        name={name},
        fill=\SumColor,
        opacity={opacity},
        radius={radius},
        logo=$+$
        }}
    }};
"""
    if layer:
        return rf"""
\begin{{pgfonlayer}}{{{layer}}}
{pic}\end{{pgfonlayer}}
"""
    return pic


def raw(text):
    return text


arch = [
    to_head("."),
    color_preamble(),
    to_begin(),
    box(
        "input",
        r"",
        r"F=1",
        r"",
        r"\InputColor",
        "(0,0,0)",
        "(0,0,0)",
        2.0,
        18,
        40,
        layer="stage0",
    ),
    box(
        "temp",
        r"",
        r"F=20",
        r"",
        r"\TemporalColor",
        "(2.2,0,0)",
        "(input-east)",
        2.2,
        18,
        40,
        layer="stage1",
    ),
    box(
        "spatial",
        r"",
        r"F=40",
        r"",
        r"\SpatialColor",
        "(1.8,0,0)",
        "(temp-east)",
        2.4,
        16,
        34,
        layer="stage2",
    ),
    box(
        "pool1",
        r"",
        r"F=40",
        r"",
        r"\PatchColor",
        "(1.4,0,0)",
        "(spatial-east)",
        1.5,
        12,
        24,
        opacity=1.0,
        layer="stage3",
    ),
    box(
        "sep",
        r"",
        r"F=40",
        r"",
        r"\PatchColor",
        "(1.4,0,0)",
        "(pool1-east)",
        2.2,
        12,
        24,
        layer="stage4",
    ),
    box(
        "pool2",
        r"",
        r"F=40",
        r"",
        r"\PatchColor",
        "(1.2,0,0)",
        "(sep-east)",
        1.5,
        9,
        18,
        opacity=1.0,
        layer="stage5",
    ),
    box(
        "tokens",
        r"",
        r"E=40",
        r"",
        r"\TokenColor",
        "(1.6,0,0)",
        "(pool2-east)",
        2.8,
        9,
        18,
        layer="stage6",
    ),
    box(
        "pos",
        r"",
        r"E=40",
        r"",
        r"\PosColor",
        "(1.2,0,0)",
        "(tokens-east)",
        1.5,
        9,
        18,
        layer="stage7",
    ),
    banded_box(
        "trans",
        r"",
        r"E=40",
        r"",
        r"",
        r"\TransColor",
        r"\TransBandColor",
        "(1.7,0,0)",
        "(pos-east)",
        2.4,
        0.9,
        11,
        20,
        layer="stage8",
    ),
    sum_ball("sum1", "(1.4,0,0)", "(trans-east)", layer="stage9"),
    box(
        "flat",
        r"",
        r"L=600",
        r"",
        r"\HeadColor",
        "(1.3,0,0)",
        "(sum1-east)",
        1.5,
        7,
        14,
        layer="stage10",
    ),
    box(
        "fc",
        r"",
        r"K=4",
        r"",
        r"\HeadColor",
        "(1.2,0,0)",
        "(flat-east)",
        1.5,
        6,
        12,
        layer="stage11",
    ),
    box(
        "out",
        r"",
        r"K=4",
        r"",
        r"\SoftmaxColor",
        "(1.2,0,0)",
        "(fc-east)",
        1.6,
        6,
        12,
        opacity=1.0,
        layer="stage12",
    ),
    raw(
        r"""
\path (pool2-south) + (0,-3.25,0) coordinate (title-row);
\node[stage, anchor=north] at (input-xlabel |- title-row) {EEG};
\node[stage, anchor=north] at (temp-xlabel |- title-row) {T-Conv};
\node[stage, anchor=north] at (spatial-xlabel |- title-row) {DWConv};
\node[stage, anchor=north] at (pool1-xlabel |- title-row) {Pool1};
\node[stage, anchor=north] at (sep-xlabel |- title-row) {SepConv};
\node[stage, anchor=north] at (pool2-xlabel |- title-row) {Pool2};
\node[stage, anchor=north] at (tokens-xlabel |- title-row) {Tokens};
\node[stage, anchor=north] at (pos-xlabel |- title-row) {PosEnc};
\node[stage, anchor=north] at (trans-xlabel |- title-row) {TrEnc $\times 6$};
\node[stage, anchor=north] at (sum1-anchor |- title-row) {Add};
\node[stage, anchor=north] at (flat-xlabel |- title-row) {Flat};
\node[stage, anchor=north] at (fc-xlabel |- title-row) {FC};
\node[stage, anchor=north] at (out-xlabel |- title-row) {Out};

\node[legendannot, anchor=west] at ($(input-west |- title-row)+(-0.2,-0.95,0)$) {Legend: $F$ feature maps, $E$ embedding width, $L$ flattened length, $K$ classes};

\begin{pgfonlayer}{flow0}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(input-east)+(0.10,0,0)$) -- ($(temp-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow1}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(temp-east)+(0.10,0,0)$) -- ($(spatial-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow2}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(spatial-east)+(0.10,0,0)$) -- ($(pool1-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow3}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(pool1-east)+(0.10,0,0)$) -- ($(sep-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow4}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(sep-east)+(0.10,0,0)$) -- ($(pool2-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow5}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(pool2-east)+(0.10,0,0)$) -- ($(tokens-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow6}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(tokens-east)+(0.10,0,0)$) -- ($(pos-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow7}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(pos-east)+(0.10,0,0)$) -- ($(trans-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow8}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(trans-east)+(0.10,0,0)$) -- ($(sum1-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow9}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(sum1-east)+(0.10,0,0)$) -- ($(flat-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow10}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(flat-east)+(0.10,0,0)$) -- ($(fc-west)+(-0.10,0,0)$);
\end{pgfonlayer}
\begin{pgfonlayer}{flow11}
\draw[-Stealth, line width=0.9pt, draw=black!90] ($(fc-east)+(0.10,0,0)$) -- ($(out-west)+(-0.10,0,0)$);
\end{pgfonlayer}

\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw=ResidualSoft,opacity=0.72]
\renewcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw=ResidualSoft] (-0.3,0) -- ++(0.3,0);}
\path (pos-southeast) -- (pos-northeast) coordinate[pos=1.35] (pos-top);
\path (sum1-south) -- (sum1-north) coordinate[pos=1.35] (sum1-top);
\draw [copyconnection]
    (pos-northeast)
    -- (pos-top)
    -- node {\copymidarrow} (sum1-top)
    -- (sum1-north);
\node[annot, rotate=45, anchor=south west] at ($(sum1-top)+(-0.22,0.16,0)$) {residual fusion};

\node[edgeannot, rotate=45, anchor=south west] at ($(input-northwest)+(-0.04,0.26,0)$) {Input: $1 \times 64 \times 1000$};
\node[edgeannot, rotate=45, anchor=south west] at ($(temp-northwest)+(-0.02,0.26,0)$) {kernel $64$, $F_1=20$};
\node[edgeannot, rotate=45, anchor=south west] at ($(spatial-northwest)+(-0.02,0.24,0)$) {\shortstack[l]{depthwise spatial conv\\[-0.15ex]over 64 channels, $D=2$}};
\node[edgeannot, rotate=45, anchor=south west] at ($(sep-northwest)+(-0.02,0.24,0)$) {kernel $16$ + avg pool/dropout};
\node[edgeannot, rotate=45, anchor=south west] at ($(tokens-northwest)+(-0.02,0.24,0)$) {$15$ tokens, embed dim $40$};
\node[edgeannot, rotate=45, anchor=south west] at ($(pos-northwest)+(-0.02,0.24,0)$) {learned positional encoding};
\node[edgeannot, rotate=45, anchor=south west] at ($(trans-northwest)+(-0.02,0.26,0)$) {$6$ blocks, $2$ heads, FFN $\times 4$};
\node[edgeannot, rotate=45, anchor=south west] at ($(out-northwest)+(-0.02,0.24,0)$) {4-class probabilities};
"""
    ),
    to_end(),
]


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(arch, namefile + ".tex")


if __name__ == "__main__":
    main()
