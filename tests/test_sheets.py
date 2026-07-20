"""`split_sheets` must separate a self-touching tissue without tearing it.

CGAL's `orient_polygon_soup` tears a pinch edge open rather than splitting it,
which is what stopped seven of the nine CHARM tissues exporting at all -- before
any coarsening was involved. These pin the property that matters: the result is
manifold, still closed, and encloses the same volume it went in with.
"""

from __future__ import annotations

import numpy as np
import pytest

from headmodel_conversion.sheets import split_sheets

# Outward-wound unit box.
BOX_TRIS = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


def soup(*parts):
    """Concatenate parts and weld coincident points -- how a tissue's own lumps
    become one non-manifold soup in the pipeline."""
    points, tris, offset = [], [], 0
    for p, t in parts:
        points.append(p)
        tris.append(t.astype(np.int64) + offset)
        offset += len(p)
    P, T = np.vstack(points), np.vstack(tris)
    uP, inv = np.unique(P, axis=0, return_inverse=True)
    return uP, inv.ravel()[T.ravel()].reshape(-1, 3).astype(np.int32)


def edge_valences(tris):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    _, counts = np.unique(np.sort(e, axis=1), axis=0, return_counts=True)
    return counts


def volume(points, tris):
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


def test_plain_box_is_left_alone():
    """Nothing to split: a manifold soup must come back untouched."""
    P, T = soup((box((0, 0, 0), (1, 1, 1)), BOX_TRIS))
    P2, T2, src = split_sheets(P, T)
    assert len(P2) == len(P), "duplicated vertices on a manifold soup"
    assert volume(P2, T2) == pytest.approx(1.0)
    assert set(edge_valences(T2)) == {2}


def test_edge_pinch_is_split_not_torn():
    """The regression. Two lumps meeting on one edge: that edge carries four faces,
    which no halfedge mesh can hold. It must come apart into two manifold sheets --
    torn open (valence 1) is the failure this exists to prevent."""
    P, T = soup((box((0, 0, 0), (1, 1, 1)), BOX_TRIS), (box((1, 1, 0), (2, 2, 1)), BOX_TRIS))
    assert 4 in set(edge_valences(T)), "fixture is not actually pinched"

    P2, T2, _ = split_sheets(P, T)
    assert set(edge_valences(T2)) == {2}, "still non-manifold, or torn open"
    assert volume(P2, T2) == pytest.approx(2.0), "split changed the enclosed volume"
    assert len(T2) == len(T), "split added or dropped faces"


def test_vertex_pinch_is_split():
    """Two lumps meeting at a single corner: no edge is non-manifold, but the
    vertex has two fans and must become two vertices."""
    P, T = soup((box((0, 0, 0), (1, 1, 1)), BOX_TRIS), (box((1, 1, 1), (2, 2, 2)), BOX_TRIS))
    assert set(edge_valences(T)) == {2}, "fixture should pinch at a vertex, not an edge"
    P2, T2, _ = split_sheets(P, T)
    assert len(P2) == len(P) + 1, "the shared corner was not split into two"
    assert volume(P2, T2) == pytest.approx(2.0)


def test_split_preserves_coordinates_exactly():
    """Copies must land on their original, or the tissue stops matching its
    neighbours' files and conformality is gone."""
    P, T = soup((box((0, 0, 0), (1, 1, 1)), BOX_TRIS), (box((1, 1, 0), (2, 2, 1)), BOX_TRIS))
    P2, T2, src = split_sheets(P, T)
    assert np.array_equal(P2, P[src]), "a copy moved off its original coordinate"
    assert set(map(tuple, P2)) == set(map(tuple, P)), "the coordinate set changed"


def test_disjoint_lumps_are_untouched():
    """Two lumps that never touch have nothing to resolve."""
    P, T = soup((box((0, 0, 0), (1, 1, 1)), BOX_TRIS), (box((5, 5, 5), (6, 6, 6)), BOX_TRIS))
    P2, T2, _ = split_sheets(P, T)
    assert len(P2) == len(P)
    assert volume(P2, T2) == pytest.approx(2.0)
