"""Command-line entry point: `headmodel-convert <m2m_dir> <out_dir>`."""

from __future__ import annotations

from pathlib import Path

import cyclopts

from . import SizingConfig, convert
from .densities import ROIConfig

app = cyclopts.App(help="Convert a CHARM m2m directory into Ansys-importable STEP files.")


@app.default
def main(
    m2m_dir: Path,
    out_dir: Path,
    *,
    roi: list[str] | None = None,
    roi_radius_mm: float = ROIConfig.radius_mm,
    roi_edge_mm: float = ROIConfig.edge_mm,
) -> None:
    """Decimate one subject and export one STEP file per tissue.

    Parameters
    ----------
    m2m_dir
        A CHARM `m2m_<sub>` directory containing `<sub>.msh` and `eeg_positions/`.
    out_dir
        Directory to write `<tissue>.step` files into.
    roi
        10-10 electrode labels marking TMS coil sites to hold near CHARM
        resolution, one `--roi` per site (e.g. `--roi C3 --roi F3 --roi FCz`).
        Omit to disable ROI refinement entirely.
    roi_radius_mm
        Distance (mm) from each ROI site over which resolution ramps back up to
        the tissue's base edge length.
    roi_edge_mm
        Target edge length (mm) at the ROI sites themselves.
    """
    config = SizingConfig(
        roi=ROIConfig(electrodes=tuple(roi or ()), radius_mm=roi_radius_mm, edge_mm=roi_edge_mm)
    )
    written = convert(m2m_dir, out_dir, config)
    print(f"wrote {len(written)} tissue STEP files to {out_dir}")


if __name__ == "__main__":
    app()
