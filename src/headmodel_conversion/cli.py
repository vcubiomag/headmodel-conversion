"""Command-line entry point: `headmodel-convert <m2m_dir> <out_dir>`."""

from __future__ import annotations

from pathlib import Path

import cyclopts

from . import convert

app = cyclopts.App(help="Convert a CHARM m2m directory into Ansys-importable STEP bodies.")


@app.default
def main(m2m_dir: Path, out_dir: Path) -> None:
    """Decimate one subject and export per-tissue STEP solids.

    Parameters
    ----------
    m2m_dir
        A CHARM `m2m_<sub>` directory containing `<sub>.msh` and `eeg_positions/`.
    out_dir
        Directory to write `<tissue>.step` files into.
    """
    written = convert(m2m_dir, out_dir)
    print(f"wrote {len(written)} STEP bodies to {out_dir}")


if __name__ == "__main__":
    app()
