"""Tests of the split stage's cut positions: the edge check's bounds and the outlier filter.

The regression case reproduces Vega 2022-05-21 left (radius 400, 1500-px padded frame): chop B
near x = 895, outside IDL's edge-check range [400, 800], and one wrong position at chunk 89
inside it, which IDL's reverts carried into every later chunk and through chop B's whole pass.
Chunk 90 there moved both positions by the same 33 px (the throw unchanged): real motion, kept.
"""
import numpy as np

from reduce_nomic import geometry


def vega_like_positions(n=100):
    """Build chunk positions shaped like Vega 2022-05-21 left's.

    Parameters
    ----------
    n : int, default 100
        Number of chunks (at least 91).

    Returns
    -------
    raw : list of [pos1, pos2]
        Per chunk, chop A near (634, 583) and chop B drifting from x = 890 to 901, y near 583;
        chunk 90 has both moved by (-34, -20) and chunk 89's chop B is wrong, at (752, 497).
    truth : list of [pos1, pos2]
        Where the star really was: as ``raw`` but with chunk 89's chop B on its track.
    """
    truth = []
    for c in range(n):
        xb = 890 + int(round(11 * c / (n - 1)))
        truth.append([[634 + (c % 3) - 1, 583 + (c % 2)], [xb, 583 - (c % 2)]])
    truth[90] = [[truth[90][0][0] - 34, truth[90][0][1] - 20], [truth[90][1][0] - 34, truth[90][1][1] - 20]]
    raw = [[list(p1), list(p2)] for p1, p2 in truth]
    raw[89][1] = [752, 497]
    return raw, truth


def settings(**over):
    """Split settings of the Vega nights.

    Parameters
    ----------
    **over
        Settings to change.

    Returns
    -------
    st : dict
        ``radius`` 400, ``framew`` 1500, edge check on with the frame's own bounds, outlier
        filter at 20 px, updated with ``over``.
    """
    st = {"radius": 400, "framew": 1500, "auto_positions_edgecheck": 1, "edgecheck_idl_bounds": False,
          "pos_outlier_px": 20.0}
    st.update(over)
    return st


def test_edge_checked_frame_bounds():
    """A chop B position inside the padded frame is kept; IDL's bounds revert it."""
    assert geometry.edge_checked([895, 584], [892, 584], 400, 1500) == [895, 584]
    assert geometry.edge_checked([895, 584], [892, 584], 400) == [892, 584]
    assert geometry.edge_checked([1150, 584], [892, 584], 400, 1500) == [892, 584]


def test_edge_checked_first_chunk():
    """Without a previous position, the frame's bounds clip; IDL's bounds keep the position."""
    assert geometry.edge_checked([1200, 300], None, 400, 1500) == [1100, 400]
    assert geometry.edge_checked([1200, 300], None, 400) == [1200, 300]


def test_filter_positions_rebuilds_the_wrong_chop():
    """Chunk 89's chop B is rebuilt from chop A and the throw; chunk 90's common shift is kept."""
    raw, truth = vega_like_positions()
    out, replaced = geometry.filter_positions(raw, 20.0)
    assert replaced == [(89, 2)]
    assert np.all(np.abs(np.subtract(out[89][1], truth[89][1])) <= 2)
    assert out[89][0] == raw[89][0]
    assert all(out[c] == raw[c] for c in range(len(raw)) if c != 89)


def test_filter_positions_both_wrong():
    """When the throw is off and both positions are far from their neighbors, both are replaced."""
    raw = [[[634, 583], [895, 583]] for _ in range(9)]
    raw[4] = [[500, 700], [1000, 300]]
    out, replaced = geometry.filter_positions(raw, 20.0)
    assert replaced == [(4, 1), (4, 2)]
    assert out[4] == [[634, 583], [895, 583]]


def test_filter_positions_off_and_short_runs():
    """Tolerance 0 and runs of fewer than 3 chunks leave the positions as found."""
    raw, _ = vega_like_positions()
    assert geometry.filter_positions(raw, 0.0) == (raw, [])
    two = [[[634, 583], [895, 583]], [[100, 100], [752, 497]]]
    assert geometry.filter_positions(two, 20.0) == (two, [])


def test_split_cut_positions_vega_regression():
    """Every cut lands within 2 px of the star; IDL's logic left chop B at x = 752."""
    raw, truth = vega_like_positions()
    use, replaced, cut = geometry.split_cut_positions(raw, settings())
    for cn in (1, 2):
        d = np.abs(np.asarray(cut[cn]) - np.asarray([t[cn - 1] for t in truth]))
        assert d.max() <= 2, (cn, int(d.max()))
    _, _, idl_cut = geometry.split_cut_positions(raw, settings(edgecheck_idl_bounds=True, pos_outlier_px=0.0))
    assert all(q[0] == 752 for q in idl_cut[2])
    assert idl_cut[1] == [r[0] for r in raw]                 # chop A lies inside IDL's range


def test_split_cut_positions_edgecheck_off():
    """With the edge check off, the cuts are the filtered positions themselves."""
    raw, _ = vega_like_positions()
    use, _, cut = geometry.split_cut_positions(raw, settings(auto_positions_edgecheck=0))
    assert cut[1] == [u[0] for u in use] and cut[2] == [u[1] for u in use]
