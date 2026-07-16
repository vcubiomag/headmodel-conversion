"""CHARM head model -> decimated, Ansys-importable STEP bodies.

`convert()` is the whole pipeline for one subject: read the CHARM `.msh`, build
the multi-material interface complex, QEM-decimate each tissue-pair patch to its
TMS-tuned target density (keeping shared feature curves fixed so the result stays
conformal), then export one STEP file per tissue via OpenCASCADE, holding every
B-rep solid of that tissue as its own named product.
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyvista

from .densities import SizingConfig, vertex_target_sizes
from .mesher import decimate_interfaces, write_step
from .surfaces import VOLUME_KEY_TO_LABEL, shell_arrays, tissue_solids

__all__ = ["convert", "SizingConfig"]

VTK_TETRA = 10


Part = tuple[str, list[tuple[np.ndarray, np.ndarray]]]


def _write_step_job(job: tuple[str, list[Part]]) -> str:
    # Runs in a worker process: OCCT's STEP transfer serializes on in-process
    # global singletons, so separate processes -- not threads -- are what let
    # independent tissue files export in parallel.
    path, parts = job
    write_step(parts, path)
    return path


def _read_charm_mesh(msh_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, tets (M,4), tet gmsh-physical labels) from a CHARM `.msh`."""
    mesh = pyvista.read(msh_path)
    tet_mask = mesh.celltypes == VTK_TETRA
    labels = np.asarray(mesh.cell_data["gmsh:physical"])[tet_mask].astype(np.int32)
    tets = np.asarray(mesh.cells_dict[VTK_TETRA], dtype=np.int64)
    return np.ascontiguousarray(mesh.points, dtype=np.float64), tets, labels


def _read_roi_points(m2m_dir: Path, config: SizingConfig) -> np.ndarray | None:
    """Coordinates of the coil-site electrodes, in mesh subject space."""
    if not config.roi.electrodes:
        return None

    csv_path = m2m_dir / "eeg_positions" / config.roi.csv_name
    if not csv_path.exists():
        print(f"  ! ROI CSV not found ({csv_path.name}); skipping ROI refinement")
        return None

    wanted = set(config.roi.electrodes)
    found: dict[str, list[float]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            # rows are: Electrode,x,y,z,Name
            if len(row) >= 5 and row[4] in wanted:
                found[row[4]] = [float(row[1]), float(row[2]), float(row[3])]

    missing = wanted - found.keys()
    if missing:
        print(f"  ! ROI electrodes missing from {csv_path.name}: {sorted(missing)}")
    return np.array(list(found.values()), dtype=np.float64) if found else None


def convert(
    m2m_dir: str | Path,
    out_dir: str | Path,
    config: SizingConfig | None = None,
) -> list[Path]:
    """Convert one CHARM `m2m` directory into one STEP file per tissue.

    Each file holds every solid of that tissue as a separately named product
    (e.g. `eye_balls.step` holds `eye_ball_L` and `eye_ball_R`). Returns the
    written `.step` paths, sorted.

    The STEP export runs in a process pool, so callers must be import-safe
    (guard the entry point with ``if __name__ == "__main__":``) on `spawn`
    start-method platforms such as macOS and Windows.
    """
    m2m_dir = Path(m2m_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = config or SizingConfig()

    sub_id = m2m_dir.name.removeprefix("m2m_")
    msh_path = m2m_dir / f"{sub_id}.msh"
    if not msh_path.exists():
        raise FileNotFoundError(f"CHARM mesh not found: {msh_path}")

    print(f"[{sub_id}] reading {msh_path.name}")
    points, tets, labels = _read_charm_mesh(msh_path)
    print(f"  input: {len(points):,} nodes, {len(tets):,} tets")

    roi_points = _read_roi_points(m2m_dir, config)
    sizes = vertex_target_sizes(points, tets, labels, roi_points, config)

    print("  decimating interface surfaces...")
    d_points, d_tris, d_refs = decimate_interfaces(points, tets, labels, sizes)
    print(f"  decimated complex: {len(d_points):,} nodes, {len(d_tris):,} triangles")

    jobs: list[tuple[int, str, list[Part]]] = []
    for tag, label in VOLUME_KEY_TO_LABEL.items():
        solids = tissue_solids(d_points, d_tris, d_refs, tag, label)
        if not solids:
            continue
        parts = [(name, [shell_arrays(s) for s in shells]) for name, shells in solids]
        n_faces = sum(len(tris) for _, arrs in parts for _, tris in arrs)
        print(f"  {label}: {len(parts)} part(s), {n_faces} faces")
        jobs.append((n_faces, str(out_dir / f"{label}.step"), parts))

    # Largest first: there are only ever a handful of tissues, and one of them
    # (gray matter) dwarfs the rest, so it sets the makespan -- start it before
    # the small ones can queue ahead of it.
    jobs.sort(key=lambda job: job[0], reverse=True)
    payloads = [(path, parts) for _, path, parts in jobs]

    workers = max(1, min(len(payloads), os.cpu_count() or 1, 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        written = [Path(p) for p in pool.map(_write_step_job, payloads)]

    return sorted(written)
