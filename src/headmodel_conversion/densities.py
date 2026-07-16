"""TMS-tuned target edge lengths and the per-vertex sizing field for decimation.

CHARM's native surface is already ~1.0-1.4 mm, and edge-collapse can only
*coarsen*, so the targets here are deliberately **coarser** than CHARM to cut the
BREP face count for Ansys. Smooth, low-field tissues (scalp, skull, deep white
matter) are coarsened hard; the cortex (GM/CSF) is coarsened only moderately; and
within `roi.radius_mm` of the coil sites everything is held near CHARM resolution
(`roi.edge_mm`) so accuracy is preserved where the E-field is evaluated. Maxwell
does its own adaptive volumetric FEM refinement, so these surfaces only need
geometric fidelity, not solver resolution.

An edge length L (mm) corresponds to a node density rho = 2 / (sqrt(3) * L^2)
(nodes/mm^2); `rho_to_edge` converts the other way if you prefer to think in
densities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# tag -> target edge length (mm). Keys match the gmsh physical tags. Edit freely.
EDGE_LENGTH: dict[int, float] = {
    1: 2.0,  # white_matter    - interior low priority
    2: 1.7,  # gray_matter     - cortex; moderate (ROI keeps it fine near coils)
    3: 2.0,  # csf             - moderate
    5: 3.5,  # scalp           - smooth, low field
    6: 3.0,  # eye_balls
    7: 3.0,  # cortical_bone   - smooth
    8: 3.0,  # cancellous_bone
    9: 3.0,  # blood
    10: 3.0,  # muscle
}

DEFAULT_EDGE = 3.0


def rho_to_edge(rho: float) -> float:
    """Isotropic edge length (mm) for a surface node density (nodes/mm^2)."""
    return float(np.sqrt(2.0 / (np.sqrt(3.0) * rho)))


@dataclass
class ROIConfig:
    """Keep near-CHARM resolution around the TMS coil target sites.

    `electrodes` are 10-10 labels read from the subject's eeg_positions CSV
    (mesh subject space): M1=C3, DLPFC=F3, SMA=FCz, PPC=P3 & P4. Within
    `radius_mm` of any site the target edge length ramps from `edge_mm` (fine, at
    the site) up to the tissue's base edge length (at the radius).
    """

    electrodes: tuple[str, ...] = ("C3", "F3", "FCz", "P3", "P4")
    radius_mm: float = 25.0
    edge_mm: float = 1.0
    csv_name: str = "EEG10-10_UI_Jurak_2007.csv"


@dataclass
class SizingConfig:
    edge_length: dict[int, float] = field(default_factory=lambda: dict(EDGE_LENGTH))
    default_edge: float = DEFAULT_EDGE
    roi: ROIConfig = field(default_factory=ROIConfig)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def vertex_target_sizes(
    points: np.ndarray,
    tets: np.ndarray,
    tet_labels: np.ndarray,
    roi_points: np.ndarray | None,
    config: SizingConfig,
) -> np.ndarray:
    """Per-vertex target edge length (mm) for the whole mesh.

    Each vertex takes the *finest* (smallest) base edge length over its incident
    tets, so an interface vertex between tissues A and B is decimated no coarser
    than min(L_A, L_B). Near the ROI sites the target ramps down to
    `roi.edge_mm`, protecting resolution where it matters.
    """
    n = len(points)
    edge_of_tag = {
        tag: config.edge_length.get(int(tag), config.default_edge)
        for tag in np.unique(tet_labels)
    }
    tet_edge = np.array([edge_of_tag[int(t)] for t in tet_labels], dtype=np.float64)

    base = np.full(n, np.inf, dtype=np.float64)
    for k in range(tets.shape[1]):
        np.minimum.at(base, tets[:, k], tet_edge)
    base[~np.isfinite(base)] = config.default_edge

    if roi_points is None or len(roi_points) == 0:
        return base

    dmin = np.full(n, np.inf, dtype=np.float64)
    for r in roi_points:
        dmin = np.minimum(dmin, np.linalg.norm(points - r, axis=1))
    s = _smoothstep(dmin / config.roi.radius_mm)
    ramped = config.roi.edge_mm + (base - config.roi.edge_mm) * s
    return np.minimum(base, ramped)
