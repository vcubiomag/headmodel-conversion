"""Command-line entry point: `headmodel-convert <m2m_dir> <out_dir>`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cyclopts

from . import SizingConfig, convert

app = cyclopts.App(help="Convert a CHARM m2m directory into Ansys-importable STEP files.")


@app.default
def main(
    m2m_dir: Path,
    out_dir: Path,
    *,
    method: Literal["mmg", "charm"] = SizingConfig.method,
    hausd_mm: float = SizingConfig.mmg_hausd_mm,
    hmax_mm: float = SizingConfig.mmg_hmax_mm,
) -> None:
    """Convert one subject and export one STEP file per tissue.

    Parameters
    ----------
    m2m_dir
        A CHARM `m2m_<sub>` directory containing `<sub>.msh`.
    out_dir
        Directory to write `<tissue>.step` files into.
    method
        `mmg` (default) remeshes the whole tet volume with MMG3D -- conformal and
        valid for every tissue, including the folded ones. `charm` exports CHARM's
        own triangulation uncoarsened (~1.24M faces; too large for Ansys).
    hausd_mm
        MMG Hausdorff bound (mm): how far a boundary may move. The quality/size
        knob -- larger coarsens more (smoothing sulci).
    hmax_mm
        MMG maximum edge length (mm).
    """
    config = SizingConfig(method=method, mmg_hausd_mm=hausd_mm, mmg_hmax_mm=hmax_mm)
    written = convert(m2m_dir, out_dir, config)
    print(f"wrote {len(written)} tissue STEP files to {out_dir}")


if __name__ == "__main__":
    app()
