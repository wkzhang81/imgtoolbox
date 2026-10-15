"""
IHC image HED color deconvolution + DAB quantification.

Original script: ihc/ihc.py (logic lives in process_ihc; here we only adapt the parameters and summary).
"""

from __future__ import annotations

from pathlib import Path

from ..loader import load_script, project_root
from ..registry import Param, Tool, register


def run(params: dict) -> str:
    m = load_script("ihc")

    input_dir = Path(params["input_dir"])
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    output_dir = params["output_dir"] or str(project_root() / "outputs" / "ihc")

    quant, csv_path = m.process_ihc(str(input_dir), output_dir)

    return (
        f"Images processed: {len(quant)}\n"
        f"Output directory: {output_dir}\n"
        f"Quantification table: {csv_path}"
    )


register(Tool(
    key="ihc",
    label="IHC HED separation + DAB quantification",
    description=(
        "Convert IHC color images into HED space, reconstruct separate hematoxylin / eosin / DAB grayscale images, "
        "and tally the total DAB-channel intensity per image (CSV)."
    ),
    params=[
        Param(key="input_dir", label="Input image directory", type="dir"),
        Param(key="output_dir", label="Output directory", type="dir", default="",
              help="If empty, uses outputs/ihc.", optional=True),
    ],
    run=run,
))
