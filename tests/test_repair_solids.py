"""`repair_solids` turns one tissue's outward-wound soup into valid solids.

The cases here are built from boxes whose nesting and manifoldness are known by
construction, so they pin the two things the STEP export actually depends on:
that a cavity is recognised as a cavity rather than a separate lump, and that
two lumps merely touching along an edge come apart instead of being exported as
one non-manifold shell.
"""

from __future__ import annotations

import numpy as np
import pytest

from headmodel_conversion.mesher import _mesher_ext

# The 12 outward-wound triangles of a box, over the 8 corners emitted by `box`.
BOX_TRIS = np.array(
    [
        [0, 3, 2],
        [0, 2, 1],  # z lo, normal -z
        [4, 5, 6],
        [4, 6, 7],  # z hi, normal +z
        [0, 1, 5],
        [0, 5, 4],  # y lo, normal -y
        [3, 7, 6],
        [3, 6, 2],  # y hi, normal +y
        [0, 4, 7],
        [0, 7, 3],  # x lo, normal -x
        [1, 2, 6],
        [1, 6, 5],  # x hi, normal +x
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
    """Concatenate (points, tris) parts and weld coincident points.

    The weld mirrors the pipeline's global merge, which is what turns two lumps
    that share coordinates into a single non-manifold soup.
    """
    points, tris, offset = [], [], 0
    for p, t in parts:
        points.append(p)
        tris.append(t.astype(np.int64) + offset)
        offset += len(p)
    P, T = np.vstack(points), np.vstack(tris)
    uP, inv = np.unique(P, axis=0, return_inverse=True)
    return uP, inv.ravel()[T.ravel()].reshape(-1, 3).astype(np.int32)


def volume(points, tris):
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0


def repair(points, tris):
    return _mesher_ext.repair_solids(np.ascontiguousarray(points), np.ascontiguousarray(tris))


def test_single_box_is_one_solid():
    solids = repair(*soup((box((0, 0, 0), (1, 1, 1)), BOX_TRIS)))
    assert len(solids) == 1
    assert len(solids[0]) == 1
    assert volume(*solids[0][0]) == pytest.approx(1.0)


def test_disjoint_boxes_are_separate_solids():
    solids = repair(
        *soup(
            (box((0, 0, 0), (1, 1, 1)), BOX_TRIS),
            (box((5, 5, 5), (6, 6, 6)), BOX_TRIS),
        )
    )
    assert len(solids) == 2
    assert all(len(s) == 1 for s in solids)


def test_cavity_is_nested_not_a_separate_lump():
    """A hollow box: the inner shell is a cavity of the outer, not its own solid."""
    inner = BOX_TRIS[:, [0, 2, 1]].copy()  # reversed: normals point out of the material
    solids = repair(
        *soup(
            (box((0, 0, 0), (10, 10, 10)), BOX_TRIS),
            (box((4, 4, 4), (6, 6, 6)), inner),
        )
    )
    assert len(solids) == 1, "cavity was split off as its own solid"
    outer, *voids = solids[0]
    assert len(voids) == 1
    assert volume(*outer) == pytest.approx(1000.0)
    # The cavity comes back wound out of the material, so its volume reads negative.
    assert volume(*voids[0]) == pytest.approx(-8.0)


def test_two_cavities():
    inner = BOX_TRIS[:, [0, 2, 1]].copy()
    solids = repair(
        *soup(
            (box((0, 0, 0), (10, 10, 10)), BOX_TRIS),
            (box((1, 1, 1), (2, 2, 2)), inner),
            (box((7, 7, 7), (8, 8, 8)), inner),
        )
    )
    assert len(solids) == 1
    assert len(solids[0]) == 3  # outer + 2 cavities


def test_edge_touching_boxes_are_split():
    """Two lumps sharing only an edge are non-manifold; repair must separate them.

    This is the 4-use edge seen throughout the real tissues: two sheets meeting
    along a line, which a halfedge mesh cannot represent and Ansys rejects.
    """
    points, tris = soup(
        (box((0, 0, 0), (1, 1, 1)), BOX_TRIS),
        (box((1, 1, 0), (2, 2, 1)), BOX_TRIS),
    )
    # The weld really did create a non-manifold edge: 4 faces on the shared edge.
    uses: dict[tuple[int, int], int] = {}
    for f in tris:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            key = (min(int(u), int(v)), max(int(u), int(v)))
            uses[key] = uses.get(key, 0) + 1
    assert sorted(set(uses.values())) == [2, 4]

    solids = repair(points, tris)
    assert len(solids) == 2, "edge-touching lumps were not separated"
    for s in solids:
        assert len(s) == 1
        assert volume(*s[0]) == pytest.approx(1.0)


def test_inside_out_soup_is_rejected():
    """An inward-wound shell is not a body; it must not silently become one."""
    inward = BOX_TRIS[:, [0, 2, 1]].copy()
    points, tris = soup((box((0, 0, 0), (1, 1, 1)), inward))
    solids = repair(points, tris)
    assert solids == [], "an inside-out shell was exported as a solid"
