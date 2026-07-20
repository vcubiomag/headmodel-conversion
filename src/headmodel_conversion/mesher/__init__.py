"""Python side of the C++ mesher: the `_mesher_ext` entry points, plus the
interface-complex extraction that feeds them."""

from __future__ import annotations

import numpy as np

from ._mesher_ext import heal_soup, mmg_remesh, repair_solids, write_step

__all__ = ["boundary_facets", "write_step", "mmg_remesh", "repair_solids", "heal_soup"]


def boundary_facets(
    points: np.ndarray, tets: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interface triangles with a per-pair reference `ref = lo*1000 + hi`.

    A facet is on a tissue boundary when it bounds the meshed region (used by one
    tet) or separates two different labels; `lo`/`hi` are the tissue labels on its
    two sides, with `hi = 0` for the outside.

    Each triangle is wound so its normal points out of `lo` and into `hi`. The tet
    it was cut from is the only thing that knows which way that is, so the winding
    has to be taken here and carried through -- orientation is not a property of
    the patch the facet later lands in. A tissue's own boundary is then just its
    facets, reversed wherever the tissue sits on the `hi` side.
    """
    # A tet's faces only have well-defined outward windings once the tet itself is
    # positively oriented, so fix the sign before reading any winding off it.
    edge = points[tets[:, 1:]] - points[tets[:, [0]]]
    vol6 = np.einsum("ij,ij->i", edge[:, 0], np.cross(edge[:, 1], edge[:, 2]))
    tets = tets.copy()
    neg = vol6 < 0.0
    tets[neg, 0], tets[neg, 1] = tets[neg, 1], tets[neg, 0]

    # The outward-facing windings of a positively-oriented tet.
    faces = np.concatenate(
        [tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [0, 3, 2]], tets[:, [1, 2, 3]]]
    )
    face_label = np.tile(labels, 4)

    # Sort each triple (v0<=v1<=v2) branchlessly and pack it into one uint64 key,
    # so the dedup below is a fast 1-D unique instead of a lexicographic sort over
    # a (4*T, 3) array (which dominated this function on large meshes). Node ids
    # fit in 21 bits (< 2^21 = 2.1M) so three of them fit in 63 bits.
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    lo = np.minimum(np.minimum(a, b), c).astype(np.uint64)
    hi = np.maximum(np.maximum(a, b), c).astype(np.uint64)
    mid = (a + b + c).astype(np.uint64) - lo - hi
    if hi.max() >= (1 << 21):
        raise ValueError("node count exceeds 21-bit facet key packing limit")
    keys = (lo << np.uint64(42)) | (mid << np.uint64(21)) | hi

    uniq_keys, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    lab_min = np.full(len(uniq_keys), np.iinfo(np.int32).max, dtype=np.int64)
    lab_max = np.zeros(len(uniq_keys), dtype=np.int64)
    np.minimum.at(lab_min, inv, face_label)
    np.maximum.at(lab_max, inv, face_label)

    interior = (counts == 2) & (lab_min == lab_max)
    boundary = ~interior & (counts <= 2)  # drop any non-manifold (count > 2) facets

    # Of the (up to two) tets sharing a facet, keep the winding contributed by the
    # `lo`-labelled one: its outward normal points out of lo and into hi. A surface
    # facet has only the one tet and `hi = 0`, so the same rule covers it unchanged.
    lo_side = face_label == lab_min[inv]
    src = np.empty(len(uniq_keys), dtype=np.int64)
    src[inv[lo_side]] = np.flatnonzero(lo_side)
    tris = faces[src[boundary]]

    refs = lab_min[boundary] * 1000 + np.where(counts[boundary] == 1, 0, lab_max[boundary])
    return tris.astype(np.int32), refs.astype(np.int32)
