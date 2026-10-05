"""Tests for gromov._sliced.py"""

# Author: Remyx AI <ai@remyx.ai>
#
# License: MIT License

import numpy as np
import pytest

import ot


def test_sliced_fused_gromov_wasserstein(nx):
    rng = np.random.RandomState(42)
    n1, n2 = 9, 6
    xs = rng.randn(n1, 2)
    xt = rng.randn(n2, 2)
    C1 = ot.dist(xs, xs)
    C1 /= C1.max()
    C2 = ot.dist(xt, xt)
    C2 /= C2.max()
    Y1 = rng.randn(n1, 3)
    Y2 = rng.randn(n2, 3)
    p = ot.unif(n1)
    q = ot.unif(n2)

    C1b, C2b, Y1b, Y2b, pb, qb = nx.from_numpy(C1, C2, Y1, Y2, p, q)

    res = ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, seed=0)
    resb = nx.to_numpy(
        ot.gromov.sliced_fused_gromov_wasserstein(C1b, C2b, Y1b, Y2b, seed=0)
    )
    resb2 = nx.to_numpy(
        ot.gromov.sliced_fused_gromov_wasserstein(C1b, C2b, Y1b, Y2b, seed=0)
    )

    # finite, non-negative and deterministic given the seed
    assert np.isfinite(res)
    assert res >= 0
    assert np.isfinite(resb)
    assert resb >= 0
    np.testing.assert_allclose(resb, resb2, atol=1e-12)

    # distance between a graph and itself is 0 (subsamples cover all nodes)
    res_same = nx.to_numpy(
        ot.gromov.sliced_fused_gromov_wasserstein(C1b, C1b, Y1b, Y1b, seed=0)
    )
    np.testing.assert_allclose(res_same, 0.0, atol=1e-12)

    # custom node weights
    resw = nx.to_numpy(
        ot.gromov.sliced_fused_gromov_wasserstein(C1b, C2b, Y1b, Y2b, p=pb, q=qb, seed=1)
    )
    assert np.isfinite(resw)
    assert resw >= 0

    # pure feature / pure structure trade-offs
    for alpha in [0.0, 1.0]:
        res_a = nx.to_numpy(
            ot.gromov.sliced_fused_gromov_wasserstein(
                C1b, C2b, Y1b, Y2b, alpha=alpha, seed=0
            )
        )
        assert np.isfinite(res_a)
        assert res_a >= 0

    # log of the per-subsample sliced distances and per-projection costs
    resl, logl = ot.gromov.sliced_fused_gromov_wasserstein(
        C1b, C2b, Y1b, Y2b, seed=0, log=True, n_subsamples=3, n_projections=7
    )
    assert np.isfinite(nx.to_numpy(resl))
    assert len(logl["sliced_dists"]) == 3
    assert len(logl["projected_emds"]) == 3
    assert len(logl["projections"]) == 3
    assert nx.to_numpy(logl["projected_emds"][0]).shape == (7,)
    assert nx.to_numpy(logl["projections"][0]).shape == (n2 + 3, 7)


def test_sliced_fused_gromov_wasserstein_projections(nx):
    # with same-size graphs (node subsamples are then the full node set) and
    # user-provided projections, no randomness remains: the solver must match
    # ot.sliced.sliced_wasserstein_distance on the fused representations
    # exactly, in every backend
    rng = np.random.RandomState(0)
    n = 6
    xs = rng.randn(n, 2)
    xt = rng.randn(n, 2)
    C1 = ot.dist(xs, xs)
    C1 /= C1.max()
    C2 = ot.dist(xt, xt)
    C2 /= C2.max()
    Y1 = rng.randn(n, 3)
    Y2 = rng.randn(n, 3)
    projections = rng.randn(n + 3, 20)

    alpha = 0.3
    # fused node representations built as in the solver
    X1 = np.concatenate(
        [np.sqrt(alpha / n) * np.sort(C1, axis=-1), np.sqrt(1.0 - alpha) * Y1], axis=1
    )
    X2 = np.concatenate(
        [np.sqrt(alpha / n) * np.sort(C2, axis=-1), np.sqrt(1.0 - alpha) * Y2], axis=1
    )
    expected = ot.sliced.sliced_wasserstein_distance(X1, X2, projections=projections)

    C1b, C2b, Y1b, Y2b, projb = nx.from_numpy(C1, C2, Y1, Y2, projections)
    res = nx.to_numpy(
        ot.gromov.sliced_fused_gromov_wasserstein(
            C1b, C2b, Y1b, Y2b, alpha=alpha, n_subsamples=3, projections=projb
        )
    )
    np.testing.assert_allclose(res, expected, rtol=1e-7, atol=1e-12)

    # node weights are supported on the deterministic path as well
    p = ot.unif(n)
    q = ot.unif(n)
    expected_w = ot.sliced.sliced_wasserstein_distance(X1, X2, p, q, projections=projections)

    pb, qb = nx.from_numpy(p, q)
    res_w = nx.to_numpy(
        ot.gromov.sliced_fused_gromov_wasserstein(
            C1b, C2b, Y1b, Y2b, p=pb, q=qb, alpha=alpha, n_subsamples=3, projections=projb
        )
    )
    np.testing.assert_allclose(res_w, expected_w, rtol=1e-7, atol=1e-12)


def _reference_calc_sftlb(
    C1, C2, Y1, Y2, alpha, seed, n_subsamples=10, n_projections=100, subsample_dim=None
):
    """Numpy transcription of the reference SFGW implementation used as oracle.

    Ported from functions ``tlb_process`` and ``calc_SFTLB`` of
    ``graph_distances.py`` in the MIT-licensed repository
    https://github.com/MoePien/slicing_fused_gromov_wasserstein accompanying
    [100]_ (Piening & Beinert, 2025). The reference code is::

        g0_sort, _ = graph0[0].sort(dim=-1)
        g0_sort = np.sqrt(((1 - alpha) / mindim)) * g0_sort
        g0feat = np.sqrt(alpha) * graph0[2]
        ...
        perm = torch.randperm(gdim0)[:mindim].sort().values
        g0_subsort = g0_sort[perm, :]; g0_subsort = g0_subsort[:, perm]
        g0feat_sub = g0feat[perm, :]
        g0_cat = torch.concatenate([g0_subsort, g0feat_sub], dim=-1)
        dist += ot.sliced.sliced_wasserstein_distance(g0_cat, g1_cat, n_projections=...)

    The reference convention is kept here: ``alpha`` weights the features. The
    reference draws its randomness from the global torch/numpy generators and
    has no seed parameter, so the draws are replayed with the protocol
    documented in ``sliced_fused_gromov_wasserstein``: seed the generator once,
    then, for each subsample, draw one node permutation per graph followed by
    the projection directions of the sliced Wasserstein estimator. Note that
    the structure term is normalized by the actual subsample size (the
    reference always divides by min(ns, nt), which coincides with the subsample
    size when ``subsample_dim`` is None).
    """
    rs = np.random.RandomState(seed)
    n1, n2 = C1.shape[0], C2.shape[0]
    m = min(n1, n2) if subsample_dim is None else subsample_dim
    g0_sort = np.sqrt((1.0 - alpha) / m) * np.sort(C1, axis=-1)
    g1_sort = np.sqrt((1.0 - alpha) / m) * np.sort(C2, axis=-1)
    g0feat = np.sqrt(alpha) * Y1
    g1feat = np.sqrt(alpha) * Y2
    dist = 0.0
    for _ in range(n_subsamples):
        perm0 = np.sort(rs.permutation(n1)[:m])
        perm1 = np.sort(rs.permutation(n2)[:m])
        g0_cat = np.concatenate([g0_sort[perm0][:, perm0], g0feat[perm0]], axis=-1)
        g1_cat = np.concatenate([g1_sort[perm1][:, perm1], g1feat[perm1]], axis=-1)
        # replicate the projection draws and the 1D OT costs of
        # ot.sliced.sliced_wasserstein_distance on the seeded generator
        projections = rs.randn(g0_cat.shape[1], n_projections)
        projections = projections / np.sqrt(np.sum(projections**2, 0, keepdims=True))
        emds = ot.lp.wasserstein_1d(
            g0_cat @ projections, g1_cat @ projections, None, None, p=2
        )
        dist += (np.sum(emds) / n_projections) ** (1.0 / 2.0)
    return dist / n_subsamples


def test_sliced_fused_gromov_wasserstein_reference_parity():
    # numerical parity with the reference implementation of [100]_: the MIT
    # reference module is not a POT dependency (it requires torch, grakel and
    # geomloss and offers no seed parameter), so the oracle is a verbatim
    # transcription of its algorithm driven with the documented RNG protocol
    rng = np.random.RandomState(0)
    n1, n2 = 9, 6
    xs = rng.randn(n1, 2)
    xt = rng.randn(n2, 2)
    C1 = ot.dist(xs, xs)
    C1 /= C1.max()
    C2 = ot.dist(xt, xt)
    C2 /= C2.max()
    Y1 = rng.randn(n1, 3)
    Y2 = rng.randn(n2, 3)

    for alpha_pot in [0.5, 0.25, 0.8]:
        # POT weights the structure with alpha while the reference weights the
        # features with alpha, hence alpha_reference = 1 - alpha_pot
        expected = _reference_calc_sftlb(
            C1, C2, Y1, Y2, alpha=1.0 - alpha_pot, seed=42, n_subsamples=5,
            n_projections=20
        )
        res = ot.gromov.sliced_fused_gromov_wasserstein(
            C1, C2, Y1, Y2, alpha=alpha_pot, n_subsamples=5, n_projections=20, seed=42
        )
        np.testing.assert_allclose(res, expected, rtol=1e-10, atol=1e-12)

    # node subsamples smaller than both graphs
    expected = _reference_calc_sftlb(
        C1, C2, Y1, Y2, alpha=0.5, seed=42, n_subsamples=4, n_projections=20,
        subsample_dim=4
    )
    res = ot.gromov.sliced_fused_gromov_wasserstein(
        C1, C2, Y1, Y2, alpha=0.5, n_subsamples=4, n_projections=20,
        subsample_dim=4, seed=42
    )
    np.testing.assert_allclose(res, expected, rtol=1e-10, atol=1e-12)


def test_sliced_fused_gromov_wasserstein_value_errors():
    C1 = np.eye(4)
    C2 = np.eye(3)
    Y1 = np.ones((4, 2))
    Y2 = np.ones((3, 2))

    # feature dimensions differ
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, Y1, np.ones((3, 5)))
    # non-square structure matrix
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(np.ones((4, 3)), C2, Y1, Y2)
    # features do not match the structure size
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, np.ones((3, 2)), Y2)
    # wrong weight size
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, p=np.ones(5))
    # subsample larger than the smallest graph
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, subsample_dim=10)
    # alpha outside [0, 1]
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, alpha=1.5)
    # not a positive number of subsamples
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, n_subsamples=0)
    # projections with a wrong number of rows (expected min(4, 3) + 2 = 5)
    with pytest.raises(ValueError):
        ot.gromov.sliced_fused_gromov_wasserstein(
            C1, C2, Y1, Y2, projections=np.ones((4, 3))
        )
