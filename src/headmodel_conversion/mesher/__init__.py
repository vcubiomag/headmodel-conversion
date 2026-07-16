"""Python side of the surface decimator: build the interface complex, decimate
each material-pair patch, and stitch the results back into one shared-vertex
complex."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _mesher_ext

write_step = _mesher_ext.write_step


def _boundary_facets(tets: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interface triangles with a per-pair reference `ref = lo*1000 + hi`.

    A facet is on a tissue boundary when it bounds the meshed region (used by one
    tet) or separates two different labels; `lo`/`hi` are the sorted tissue labels
    on its two sides (outside = 0).
    """
    faces = np.concatenate(
        [tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]
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
    bkeys = uniq_keys[boundary]
    mask21 = np.uint64((1 << 21) - 1)
    tris = np.stack(
        [
            (bkeys >> np.uint64(42)).astype(np.int64),
            ((bkeys >> np.uint64(21)) & mask21).astype(np.int64),
            (bkeys & mask21).astype(np.int64),
        ],
        axis=1,
    )
    refs = lab_min[boundary] * 1000 + np.where(counts[boundary] == 1, 0, lab_max[boundary])
    return tris.astype(np.int32), refs.astype(np.int32)


def decimate_interfaces(
    points: np.ndarray,
    tets: np.ndarray,
    labels: np.ndarray,
    sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate every interface patch and stitch into one shared-vertex complex.

    `sizes` is the per-vertex target edge length. Each patch is QEM-decimated
    under that field with its feature-boundary curves held fixed, so patches
    still meet exactly and the reassembled per-tissue surfaces stay conformal.
    Returns (points, triangles, refs).
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    sizes = np.ascontiguousarray(sizes, dtype=np.float64)
    tris, refs = _boundary_facets(tets, labels)

    def _decimate_one(ref: int) -> tuple[np.ndarray, np.ndarray]:
        patch_tris = tris[refs == ref]
        used = np.unique(patch_tris)
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        local_pts = np.ascontiguousarray(points[used])
        local_faces = remap[patch_tris].astype(np.int32)
        local_targets = np.ascontiguousarray(sizes[used])
        # decimate_patch drops the GIL for its CGAL work, so patches run in
        # parallel across the pool.
        return _mesher_ext.decimate_patch(local_pts, local_faces, local_targets)

    uniq_refs = np.unique(refs)
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_decimate_one, uniq_refs))

    out_points: list[np.ndarray] = []
    out_faces: list[np.ndarray] = []
    out_refs: list[np.ndarray] = []
    offset = 0
    for ref, (dp, df) in zip(uniq_refs, results):
        out_points.append(dp)
        out_faces.append(df.astype(np.int64) + offset)
        out_refs.append(np.full(len(df), ref, dtype=np.int32))
        offset += len(dp)

    P = np.vstack(out_points)
    F = np.vstack(out_faces)
    R = np.concatenate(out_refs)

    # Feature-curve vertices are preserved exactly, so they coincide across
    # patches -- merge them (and any coincident interior vertices) globally.
    uP, inv = np.unique(P, axis=0, return_inverse=True)
    F = inv.ravel()[F.ravel()].reshape(-1, 3).astype(np.int32)
    return uP, F, R
