"""`_boundary_facets` must wind every triangle out of `lo` and into `hi`.

Everything downstream inherits its orientation from here, so these check the
winding against tets whose geometry is known by construction rather than against
anything the rest of the pipeline produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from headmodel_conversion.mesher import boundary_facets as _boundary_facets


def normal(points: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = points[tri]
    return np.cross(b - a, c - a)


def test_single_tet_faces_point_outward():
    """A lone tet is all surface facets: lo is the tissue, hi = 0 (outside)."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tets = np.array([[0, 1, 2, 3]], dtype=np.int64)
    tris, refs = _boundary_facets(points, tets, np.array([5], dtype=np.int32))

    assert len(tris) == 4
    assert set(refs.tolist()) == {5 * 1000 + 0}  # lo=5 (tissue), hi=0 (outside)

    centroid = points.mean(axis=0)
    for tri in tris:
        face_centre = points[tri].mean(axis=0)
        assert np.dot(normal(points, tri), face_centre - centroid) > 0


def test_single_tet_outward_regardless_of_input_winding():
    """A negatively-oriented tet must be fixed up, not trusted."""
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    flipped = np.array([[0, 2, 1, 3]], dtype=np.int64)  # negative signed volume
    tris, _ = _boundary_facets(points, flipped, np.array([5], dtype=np.int32))

    centroid = points.mean(axis=0)
    for tri in tris:
        face_centre = points[tri].mean(axis=0)
        assert np.dot(normal(points, tri), face_centre - centroid) > 0


def test_interface_points_from_lo_into_hi():
    """The shared facet of two labelled tets faces out of the lower label."""
    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
    )
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    labels = np.array([5, 2], dtype=np.int32)  # tet at z>0 is 5, at z<0 is 2
    tris, refs = _boundary_facets(points, tets, labels)

    shared = refs == 2 * 1000 + 5
    assert shared.sum() == 1
    # lo=2 lives at z<0, so its outward normal on the z=0 interface points +z.
    assert normal(points, tris[shared][0])[2] > 0


def test_interface_winding_is_independent_of_tet_order():
    """Same facet, same winding, whichever tet is listed first."""
    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
    )
    a = _boundary_facets(
        points,
        np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64),
        np.array([5, 2], dtype=np.int32),
    )
    b = _boundary_facets(
        points,
        np.array([[0, 1, 2, 4], [0, 1, 2, 3]], dtype=np.int64),
        np.array([2, 5], dtype=np.int32),
    )
    na = normal(points, a[0][a[1] == 2005][0])
    nb = normal(points, b[0][b[1] == 2005][0])
    assert np.allclose(na, nb)


def test_interior_facets_dropped():
    """A facet between two tets of the same label is not a boundary."""
    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
    )
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    tris, refs = _boundary_facets(points, tets, np.array([5, 5], dtype=np.int32))

    assert len(tris) == 6  # the 8 outer faces less the 2 sharing the interior facet
    assert set(refs.tolist()) == {5 * 1000 + 0}


@pytest.mark.parametrize("label_hi", [2, 5])
def test_two_tissue_boundary_is_coherently_oriented(label_hi):
    """Each tissue's own boundary closes coherently once `hi` facets are flipped.

    This is the property the whole export rests on: every edge of a tissue's
    surface traversed exactly once in each direction.
    """
    points = np.array(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64
    )
    tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    tris, refs = _boundary_facets(points, tets, np.array([5, 2], dtype=np.int32))

    lo, hi = refs // 1000, refs % 1000
    mine = (lo == label_hi) | (hi == label_hi)
    faces = tris[mine].copy()
    faces[hi[mine] == label_hi] = faces[hi[mine] == label_hi][:, [0, 2, 1]]

    directed: dict[tuple[int, int], int] = {}
    for f in faces:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            directed[(int(u), int(v))] = directed.get((int(u), int(v)), 0) + 1

    assert all(n == 1 for n in directed.values()), "edge traversed twice the same way"
    for u, v in directed:
        assert (v, u) in directed, f"edge {u}->{v} has no opposite"
