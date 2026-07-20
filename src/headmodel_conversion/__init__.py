"""CHARM head model -> coarsened, Ansys-importable STEP bodies.

`convert()` is the whole pipeline for one subject: read the CHARM `.msh`, remesh
the tet volume with MMG3D, cut the multi-material interface complex off the
result, resolve each tissue's self-contacts, decompose it into solids, and export
one STEP file per tissue via OpenCASCADE, holding every B-rep solid of that
tissue as its own named product.

Remeshing the *volume* is the load-bearing choice. A tissue's two walls are
remeshed together and never crossed, so the complex stays conformal and
non-self-intersecting by construction and every tissue resolves into a valid
solid -- which coarsening the surfaces alone could not achieve for the folded
tissues (gray matter, CSF, cortical bone), whose walls come within a millimetre
of each other through the sulci.

Each stage checks its own invariant and raises rather than handing the next stage
something it cannot use -- a tissue that cannot be exported validly is a failure
to look at, not a file to write.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pyvista

from .mesher import boundary_facets, mmg_remesh, write_step
from .surfaces import VOLUME_KEY_TO_LABEL, tissue_solids

__all__ = ["convert", "SizingConfig"]

VTK_TETRA = 10


Part = tuple[str, list[tuple[np.ndarray, np.ndarray]]]


@dataclass
class SizingConfig:
    """How to coarsen the CHARM mesh before export. `method` picks the pipeline:

    - ``"mmg"`` (default): remesh the whole tet volume with MMG3D under
      `mmg_hausd_mm`. Conformal and non-self-intersecting by construction, so
      every tissue -- including the folded ones -- exports as a valid solid.
      `mmg_hausd_mm` is the headline quality/size knob (bigger coarsens more by
      smoothing sulci); the others cap edge length and size gradation.
    - ``"charm"``: CHARM's own triangulation, no coarsening. Every tissue valid,
      but ~1.24M faces / ~2.7 GB per subject -- too large for Ansys.
    """

    method: Literal["mmg", "charm"] = "mmg"

    # Tuned on sub-001 (anisotropic): ~320k boundary faces, ~3.7x fewer than CHARM,
    # at ~0.3 mm typical / ~2 mm max deviation. Past hausd ~2 the face count plateaus
    # (curvature/topology floor) while fidelity only degrades, so ~1.5-2.0 is the
    # sound floor; hgrad 2.0 coarsens faster than MMG's 1.3 default at equal fidelity.
    mmg_hausd_mm: float = 1.5
    mmg_hmax_mm: float = 15.0
    mmg_hmin_mm: float = 0.2
    mmg_hgrad: float = 2.0
    # Anisotropic size map: long thin triangles along a sulcus's low-curvature axis,
    # roughly halving the face count vs isotropic at equal fidelity. Angle detection
    # must stay on -- MMG3D skips boundary remeshing entirely without it. FE-quality
    # passes are skipped (`nofem`): Ansys remeshes, so it only needs valid boundaries.
    mmg_aniso: bool = True
    mmg_angle_detect: bool = True
    mmg_nofem: bool = True


def _write_step_job(job: tuple[str, list[Part]]) -> str:
    """Export one tissue's STEP file. Runs in a worker process.

    OCCT's STEP transfer serializes on in-process global singletons, so separate
    processes -- not threads -- are what let independent tissue files export in
    parallel.
    """
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

    if config.method == "mmg":
        print(f"  MMG3D remesh (hausd {config.mmg_hausd_mm} mm)...")
        points, tets, labels = mmg_remesh(
            points,
            np.ascontiguousarray(tets, dtype=np.int32),
            np.ascontiguousarray(labels, dtype=np.int32),
            config.mmg_hausd_mm,
            config.mmg_hmax_mm,
            config.mmg_hmin_mm,
            config.mmg_hgrad,
            1,
            int(config.mmg_aniso),
            int(config.mmg_angle_detect),
            int(config.mmg_nofem),
        )
        print(f"  remeshed: {len(points):,} nodes, {len(tets):,} tets")

    tris, refs = boundary_facets(points, tets, labels)
    print(f"  exporting {len(tris):,} boundary triangles")

    jobs: list[tuple[int, str, list[Part]]] = []
    failed: list[str] = []
    for tag, label in VOLUME_KEY_TO_LABEL.items():
        # A tissue whose walls cannot be resolved into valid solids is a geometry
        # problem to report, not a reason to lose the tissues that did resolve.
        # Under `mmg` this list is expected to stay empty; an entry in it means the
        # remesh produced something the solid decomposition could not accept.
        try:
            parts = tissue_solids(points, tris, refs, tag, label)
        except RuntimeError as e:
            print(f"  {label}: FAILED -- {e}")
            failed.append(label)
            continue
        if not parts:
            continue
        n_faces = sum(len(tris) for _, shells in parts for _, tris in shells)
        print(f"  {label}: {len(parts)} part(s), {n_faces} faces")
        jobs.append((n_faces, str(out_dir / f"{label}.step"), parts))

    if failed:
        print(
            f"  ! {len(failed)}/{len(VOLUME_KEY_TO_LABEL)} tissues failed to "
            f"resolve into valid solids: {', '.join(failed)}"
        )

    # Largest first: there are only ever a handful of tissues, and one of them
    # (gray matter) dwarfs the rest, so it sets the makespan -- start it before
    # the small ones can queue ahead of it.
    jobs.sort(key=lambda job: job[0], reverse=True)
    payloads = [(path, parts) for _, path, parts in jobs]

    workers = max(1, min(len(payloads), os.cpu_count() or 1, 8))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        written = [Path(p) for p in pool.map(_write_step_job, payloads)]

    return sorted(written)
