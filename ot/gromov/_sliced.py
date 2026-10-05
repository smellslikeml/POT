# -*- coding: utf-8 -*-
"""
Sliced Fused Gromov-Wasserstein solver.

"""

# Author: Remyx AI <ai@remyx.ai>
#
# License: MIT License
#
# This solver is a backend-agnostic port (with attribution) of the
# MIT-licensed reference implementation of the Sliced Fused Gromov-Wasserstein
# distance: https://github.com/MoePien/slicing_fused_gromov_wasserstein
# (functions ``calc_SFTLB`` and ``tlb_process`` in ``graph_distances.py``),
# which accompanies the paper [100]. The random projections and the 1D OT
# costs are delegated to :func:`ot.sliced.sliced_wasserstein_distance`.

import warnings

from ..backend import get_backend
from ..utils import list_to_array
from ..sliced import sliced_wasserstein_distance


def sliced_fused_gromov_wasserstein(
    C1,
    C2,
    Y1,
    Y2,
    p=None,
    q=None,
    alpha=0.5,
    n_subsamples=10,
    n_projections=100,
    subsample_dim=None,
    projections=None,
    seed=None,
    log=False,
):
    r"""Returns the Sliced Fused Gromov-Wasserstein (SFGW) distance between the
    attributed graphs :math:`(\mathbf{C_1}, \mathbf{Y_1}, \mathbf{p})` and
    :math:`(\mathbf{C_2}, \mathbf{Y_2}, \mathbf{q})`
    (see :ref:`[100] <references-sliced-fused-gromov-wasserstein>`).

    Instead of solving the non-convex quadratic program of
    :func:`ot.gromov.fused_gromov_wasserstein2`, the distance is estimated by
    a Monte-Carlo average of sliced Wasserstein distances [31]_ between fused
    node representations built from random node subsamples of the two graphs.
    Each node of a graph is described by concatenating:

    - its row of the structure matrix with entries sorted in increasing order
      (which removes the dependence on the ordering of the other nodes),
      restricted to the subsampled nodes and rescaled by
      :math:`\sqrt{\alpha / m}`, and
    - its features, rescaled by :math:`\sqrt{1 - \alpha}`,

    where :math:`m` is the size of the node subsamples. The distance between
    the two sets of :math:`m` fused representations is then estimated with the
    sliced Wasserstein distance over ``n_projections`` random projections:

    .. math::
        \mathrm{SFGW} = \frac{1}{S} \sum_{s=1}^{S}
        \mathrm{SW}_2 \left( \mathbf{X}^{(s)}_1, \mathbf{X}^{(s)}_2 \right)

    where :math:`S` is ``n_subsamples``, each
    :math:`\mathbf{X}^{(s)}_i \in \mathbb{R}^{m \times (m + d)}` concatenates
    :math:`\sqrt{\alpha / m} \, \mathrm{sort}(\mathbf{C}_i)_{I_s I_s}` and
    :math:`\sqrt{1 - \alpha} \, (\mathbf{Y}_i)_{I_s}` for a random node
    subsample :math:`I_s` of size :math:`m` (indices sorted in increasing
    order), and :math:`\mathrm{SW}_2` is computed with uniform weights by
    default.

    .. note:: ``alpha`` follows the convention of
        :func:`ot.gromov.fused_gromov_wasserstein`: it weights the structure
        term while ``1 - alpha`` weights the feature term. Note that the
        reference implementation of [100]_ uses the inverse convention
        (``alpha`` weights the features).
    .. note:: This function is backend-compatible and will work on arrays from
        all compatible backends. The projections and the 1D OT costs of each
        subsample are computed by :func:`ot.sliced.sliced_wasserstein_distance`.
    .. note:: This is a Monte-Carlo estimator: the returned value depends on
        the random number generator. Use ``seed`` for reproducibility.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix representative of the structure in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix representative of the structure in the target space
    Y1 : array-like, shape (ns, d)
        Feature matrix in the source space
    Y2 : array-like, shape (nt, d)
        Feature matrix in the target space
    p : array-like, shape (ns,), optional
        Distribution in the source space. If None, uniform weights are used on
        the subsampled nodes of the source graph.
    q : array-like, shape (nt,), optional
        Distribution in the target space. If None, uniform weights are used on
        the subsampled nodes of the target graph.
    alpha : float, optional
        Trade-off parameter (0 <= alpha <= 1) between the features
        (alpha = 0) and the structure (alpha = 1)
    n_subsamples : int, optional
        Number of random node subsamples used in the Monte-Carlo average
    n_projections : int, optional
        Number of projections used by the sliced Wasserstein distance of each
        subsample
    subsample_dim : int, optional
        Size :math:`m` of the node subsamples. If None, ``min(ns, nt)`` is
        used, so that all nodes of the smaller graph are subsampled.
    projections : array-like, shape (subsample_dim + d, n_projections), optional
        Projection matrix used by the sliced Wasserstein distance of every
        subsample (``n_projections`` is not used in this case; ``seed`` still
        controls the node subsamples). Must be expressed in the same backend
        as the input arrays.
    seed : int or RandomState or None, optional
        Seed used to seed the backend random number generator. It controls
        both the node subsamples and the random projections.
    log : bool, optional
        If True, sliced_fused_gromov_wasserstein also returns a log dict.

    Returns
    -------
    distance : float or array-like
        Sliced Fused Gromov-Wasserstein distance for the given parameters
    log : dict, optional
        Log dictionary return only if log==True in parameters, with keys:

        - ``"sliced_dists"``: list of the ``n_subsamples`` sliced Wasserstein
          distances (one per subsample)
        - ``"projected_emds"``: list of the per-projection 1D OT costs of each
          subsample (each of shape (n_projections,))
        - ``"projections"``: list of the projection matrices used by each
          subsample (each of shape (subsample_dim + d, n_projections))

    Examples
    --------
    >>> import numpy as np
    >>> C1 = np.array([[0.0, 1.0], [1.0, 0.0]])
    >>> C2 = np.array([[0.0, 2.0], [2.0, 0.0]])
    >>> Y1 = np.array([[0.0], [1.0]])
    >>> Y2 = np.array([[1.0], [0.0]])
    >>> sliced_fused_gromov_wasserstein(C1, C1, Y1, Y1, seed=0)  # doctest: +NORMALIZE_WHITESPACE
    0.0
    >>> r1 = sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, seed=0)
    >>> r2 = sliced_fused_gromov_wasserstein(C1, C2, Y1, Y2, seed=0)
    >>> r1 == r2
    True

    .. _references-sliced-fused-gromov-wasserstein:
    References
    ----------
    .. [100] Piening, M., & Beinert, R. (2025). "A Novel Sliced Fused
        Gromov-Wasserstein Distance." arXiv preprint arXiv:2508.02364.
        Reference (MIT-licensed) implementation:
        https://github.com/MoePien/slicing_fused_gromov_wasserstein

    .. [31] Bonneel, Nicolas, et al. "Sliced and radon wasserstein barycenters
        of measures." Journal of Mathematical Imaging and Vision 51.1 (2015):
        22-45
    """

    C1, C2, Y1, Y2 = list_to_array(C1, C2, Y1, Y2)

    arr = [C1, C2, Y1, Y2]
    if p is not None:
        p = list_to_array(p)
        arr.append(p)
    if q is not None:
        q = list_to_array(q)
        arr.append(q)
    if projections is not None:
        projections = list_to_array(projections)
        arr.append(projections)

    nx = get_backend(*arr)

    if not isinstance(alpha, (int, float)):
        alpha = float(nx.to_numpy(alpha))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1], got {}".format(alpha))

    n1, n2 = C1.shape[0], C2.shape[0]

    if C1.ndim != 2 or C1.shape[0] != C1.shape[1]:
        raise ValueError(
            "C1 must be a square structure matrix, got shape {}".format(C1.shape)
        )
    if C2.ndim != 2 or C2.shape[0] != C2.shape[1]:
        raise ValueError(
            "C2 must be a square structure matrix, got shape {}".format(C2.shape)
        )
    if Y1.ndim != 2 or Y1.shape[0] != n1:
        raise ValueError(
            "Y1 must be a feature matrix with {} rows, got shape {}".format(
                n1, Y1.shape
            )
        )
    if Y2.ndim != 2 or Y2.shape[0] != n2:
        raise ValueError(
            "Y2 must be a feature matrix with {} rows, got shape {}".format(
                n2, Y2.shape
            )
        )
    if Y1.shape[1] != Y2.shape[1]:
        raise ValueError(
            "Y1 and Y2 must have the same number of features, got {} and {}".format(
                Y1.shape[1], Y2.shape[1]
            )
        )
    if p is not None and p.shape[0] != n1:
        raise ValueError(
            "p must be a distribution with {} samples, got shape {}".format(
                n1, p.shape
            )
        )
    if q is not None and q.shape[0] != n2:
        raise ValueError(
            "q must be a distribution with {} samples, got shape {}".format(
                n2, q.shape
            )
        )

    n_subsamples = int(n_subsamples)
    if n_subsamples < 1:
        raise ValueError("n_subsamples must be a positive integer")

    if subsample_dim is None:
        n_sub = min(n1, n2)
    else:
        n_sub = int(subsample_dim)
        if not 1 <= n_sub <= min(n1, n2):
            raise ValueError(
                "subsample_dim must be in [1, {}], got {}".format(min(n1, n2), n_sub)
            )

    d = n_sub + Y1.shape[1]
    if projections is not None:
        if projections.ndim != 2 or projections.shape[0] != d:
            raise ValueError(
                "projections must have shape ({}, n_projections), got shape {}".format(
                    d, projections.shape
                )
            )

    if not nx.is_floating_point(C1):
        warnings.warn(
            "Input structure matrix consists of integer. The distance will be "
            "computed with a loss of precision. If this behaviour is unwanted, "
            "please make sure your input structure matrix consists of floating "
            "point elements.",
            stacklevel=2,
        )

    # Fused representations of the nodes (port of tlb_process in the reference
    # implementation): rows of the structure matrices sorted in increasing
    # order and rescaled by sqrt(alpha / m), features rescaled by
    # sqrt(1 - alpha). Note that alpha weights the structure, as in
    # ot.gromov.fused_gromov_wasserstein, which is the inverse of the
    # convention of the reference implementation.
    scale_structure = nx.sqrt(nx.full((1,), alpha / n_sub, type_as=C1))
    scale_features = nx.sqrt(nx.full((1,), 1.0 - alpha, type_as=C1))

    C1_sorted = nx.sort(C1) * scale_structure
    C2_sorted = nx.sort(C2) * scale_structure
    Y1_scaled = Y1 * scale_features
    Y2_scaled = Y2 * scale_features

    if seed is not None:
        nx.seed(seed)

    dist = 0.0
    log_sub = {"sliced_dists": [], "projected_emds": [], "projections": []}

    for _ in range(n_subsamples):
        # random node subsample of size n_sub of each graph, with indices
        # sorted in increasing order as in the reference implementation
        idx1 = nx.sort(nx.randperm(n1, type_as=C1)[:n_sub])
        idx2 = nx.sort(nx.randperm(n2, type_as=C2)[:n_sub])

        X1 = nx.concatenate([C1_sorted[idx1][:, idx1], Y1_scaled[idx1]], axis=1)
        X2 = nx.concatenate([C2_sorted[idx2][:, idx2], Y2_scaled[idx2]], axis=1)

        # weights restricted to the subsampled nodes and renormalized
        if p is None:
            a = None
        else:
            a = p[idx1]
            a = a / nx.sum(a)
        if q is None:
            b = None
        else:
            b = q[idx2]
            b = b / nx.sum(b)

        sw, log_sw = sliced_wasserstein_distance(X1, X2, a, b, n_projections=n_projections,
                                                 projections=projections, log=True)
        dist = dist + sw
        if log:
            log_sub["sliced_dists"].append(sw)
            log_sub["projected_emds"].append(log_sw["projected_emds"])
            log_sub["projections"].append(log_sw["projections"])

    res = dist / n_subsamples
    if log:
        return res, log_sub
    return res
