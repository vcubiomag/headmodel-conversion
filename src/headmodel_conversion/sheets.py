"""Split a self-touching tissue boundary into sheets, so it becomes a manifold.

A CHARM tissue touches itself. White matter has ~20 edges where two of its walls
meet along a curve, grey matter ~28, CSF ~36. The soup is still closed and
coherently wound there -- every such edge carries an even number of faces, half
wound each way -- but four faces on one edge is not something a halfedge mesh can
hold, so it has to be split into separate sheets first.

CGAL's `orient_polygon_soup` will do that, and it is what the pipeline used to
lean on, but it decides combinatorially: given four faces on an edge it has no
way to tell which two belong to the same wall, so rather than guess it tears the
edge open. That leaves the tissue with holes it never had, `repair_solids`
requires a closed soup, and the tissue fails to export at all. It is not a small
effect -- it took out seven of the nine tissues, before any coarsening, and the
only two that survived (eyeballs, muscle) were the two with no self-touching
edges. `stitch_borders` afterwards repairs some of the tears and mispairs others.

The information the combinatorial pass lacks is angular. Around the axis of a
self-touching edge the faces alternate: the tissue fills the wedge between one
face and the next, and it is the two faces bounding one wedge that belong to the
same wall. Sorting the faces by angle recovers exactly that, and pairs them with
no ambiguity to resolve.

Cost is negligible: only edges carrying more than two faces need the angular
sort, and there are a few dozen of those in a tissue of half a million.
"""

from __future__ import annotations

import numpy as np


def _corner_union(n_corners: int, pairs: np.ndarray) -> np.ndarray:
    """Connected-component label per corner, given (a, b) corner pairs to join.

    Union-find with path halving; `pairs` is small enough that the Python loop is
    not worth replacing with a sparse-graph dependency.
    """
    parent = np.arange(n_corners, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    roots = np.array([find(i) for i in range(n_corners)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.ravel()


def _pair_around_edge(
    points: np.ndarray, tris: np.ndarray, u: int, v: int, faces: np.ndarray
) -> list[tuple[int, int]]:
    """Which faces on edge (u,v) bound a common wedge of tissue, hence one sheet.

    The soup is wound outward, so a face traversing the edge v->u has the tissue
    on the increasing-angle side of it and a face traversing u->v has the tissue
    on the decreasing-angle side. Sorted by angle about the axis, each v->u face
    therefore opens a wedge that the next u->v face closes, and those two are the
    wedge's walls -- the same sheet, meeting along this edge.
    """
    axis = points[v] - points[u]
    n = np.linalg.norm(axis)
    if n == 0:
        return []
    axis = axis / n

    # Any vector perpendicular to the axis will do as the zero-angle reference.
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    ref = np.cross(axis, tmp)
    ref /= np.linalg.norm(ref)
    ref2 = np.cross(axis, ref)

    angles, forward = [], []
    for f in faces:
        tri = tris[f]
        w = points[[x for x in tri if x != u and x != v][0]]
        r = w - points[u]
        r = r - np.dot(r, axis) * axis
        if np.linalg.norm(r) == 0:
            return []  # degenerate: apex sits on the axis
        angles.append(np.arctan2(np.dot(r, ref2), np.dot(r, ref)) % (2 * np.pi))
        # True when this face traverses the edge as u->v.
        i = int(np.flatnonzero(tri == u)[0])
        forward.append(tri[(i + 1) % 3] == v)

    # Walking the faces in angular order, a well-formed edge alternates: each v->u
    # face opens a wedge of tissue and the next u->v face closes it. Match them off
    # as we go, so one face can only ever close the single wedge it bounds -- taking
    # "the next u->v face" per v->u face independently would hand the same face to
    # several wedges and leave others with none, which tears the edge instead of
    # splitting it.
    #
    # The matching must be perfect: on a closed soup an edge carries equal numbers
    # of u->v and v->u faces, and every face must be matched exactly once, or the
    # unmatched one is left a lone face on its sheet-edge -- a torn hole. The wrap
    # pass (order twice, because a wedge may straddle angle zero) is what forces a
    # face seen before its partner to still get one, so it must not re-push a face
    # already waiting (`pushed`) nor re-match one already paired (`matched`);
    # without those guards the second lap double-uses some faces and orphans others.
    order = list(np.argsort(angles))
    matched = [False] * len(faces)
    pushed = [False] * len(faces)
    out, pending = [], []
    for idx in order + order:
        if matched[idx]:
            continue
        if not forward[idx]:
            if not pushed[idx]:
                pending.append(idx)
                pushed[idx] = True
        elif pending:
            a = pending.pop()
            matched[a] = matched[idx] = True
            out.append((int(faces[a]), int(faces[idx])))
    return out


def split_sheets(
    points: np.ndarray, tris: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate vertices so the soup becomes a manifold, without tearing it.

    Returns (points, tris, src) where every edge carries at most two faces, every
    vertex has a single fan, and `src[i]` is the input vertex that output vertex i
    was copied from -- so a caller's per-vertex fields can follow the duplication.
    Copies keep their coordinates exactly, so the geometry (and conformality with
    the neighbouring tissue's file) is untouched; only identity changes.
    """
    nf = len(tris)
    if nf == 0:
        return points, tris, np.arange(len(points), dtype=np.int64)

    # Corner c = 3*f + k is face f's k-th vertex. Two faces sharing an edge and a
    # sheet put their corners at that edge's endpoints into one class; the classes
    # are the vertex copies.
    e_lo = np.minimum(tris[:, [0, 1, 2]], tris[:, [1, 2, 0]]).ravel()
    e_hi = np.maximum(tris[:, [0, 1, 2]], tris[:, [1, 2, 0]]).ravel()
    e_face = np.repeat(np.arange(nf), 3)

    key = e_lo.astype(np.int64) * (int(tris.max()) + 1) + e_hi
    order = np.argsort(key, kind="stable")
    key_s, face_s = key[order], e_face[order]
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    counts = np.diff(np.r_[starts, len(key_s)])

    pairs: list[tuple[int, int]] = []
    for s, c in zip(starts, counts):
        faces = face_s[s : s + c]
        if c == 2:
            pairs.append((int(faces[0]), int(faces[1])))
        elif c > 2:
            u, v = int(e_lo[order[s]]), int(e_hi[order[s]])
            pairs.extend(_pair_around_edge(points, tris, u, v, faces))

    # Turn face pairs into corner pairs: two sheet-adjacent faces share an edge, so
    # they agree at both of its endpoints.
    corner_of = {}
    for f in range(nf):
        for k in range(3):
            corner_of[(f, int(tris[f, k]))] = 3 * f + k

    cpairs = []
    for f1, f2 in pairs:
        shared = set(tris[f1].tolist()) & set(tris[f2].tolist())
        for v in shared:
            cpairs.append((corner_of[(f1, v)], corner_of[(f2, v)]))

    labels = _corner_union(3 * nf, np.array(cpairs, dtype=np.int64).reshape(-1, 2))

    new_tris = labels.reshape(nf, 3).astype(np.int32)
    src = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    src[labels] = tris.ravel()
    return np.ascontiguousarray(points[src], dtype=np.float64), new_tris, src
