# -*- coding: utf-8 -*-
"""
Gromov-Wasserstein and Fused-Gromov-Wasserstein utils.
"""

# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#         Rémi Flamary <remi.flamary@unice.fr>
#         Titouan Vayer <titouan.vayer@irisa.fr>
#         Cédric Vincent-Cuaz <cedvincentcuaz@gmail.com>
#         Quang Huy Tran <quang-huy.tran@univ-ubs.fr>
#
# License: MIT License

from ..utils import list_to_array, euclidean_distances
from ..backend import get_backend
from ..lp import emd

try:
    from networkx.algorithms.community import asyn_fluidc
    from networkx import from_numpy_array

    networkx_import = True
except ImportError:
    networkx_import = False

try:
    from sklearn.cluster import SpectralClustering, KMeans

    sklearn_import = True
except ImportError:
    sklearn_import = False

import numpy as np
import warnings


def _transform_matrix(C1, C2, loss_fun="square_loss", nx=None):
    r"""Return transformed structure matrices for Gromov-Wasserstein fast computation

    Returns the matrices involved in the computation of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2})`
    with the selected loss function as the loss function of Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    fC1 : array-like, shape (ns, ns)
        :math:`\mathbf{f1}(\mathbf{C1})` matrix in Eq. (6)
    fC2 : array-like, shape (nt, nt)
        :math:`\mathbf{f2}(\mathbf{C2})` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)


    .. _references-transform_matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        C1, C2 = list_to_array(C1, C2)
        nx = get_backend(C1, C2)

    if loss_fun == "square_loss":

        def f1(a):
            return a**2

        def f2(b):
            return b**2

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == "kl_loss":

        def f1(a):
            return a * nx.log(a + 1e-18) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-18)
    else:
        raise ValueError(
            f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
        )

    fC1 = f1(C1)
    fC2 = f2(C2)
    hC1 = h1(C1)
    hC2 = h2(C2)

    return fC1, fC2, hC1, hC2


def init_matrix(C1, C2, p, q, loss_fun="square_loss", nx=None):
    r"""Return loss matrices and tensors for Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Probability distribution in the source space
    q : array-like, shape (nt,)
        Probability distribution in the target space
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        C1, C2, p, q = list_to_array(C1, C2, p, q)
        nx = get_backend(C1, C2, p, q)

    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, nx)
    constC1 = nx.dot(
        nx.dot(fC1, nx.reshape(p, (-1, 1))), nx.ones((1, len(q)), type_as=q)
    )
    constC2 = nx.dot(
        nx.ones((len(p), 1), type_as=p), nx.dot(nx.reshape(q, (1, -1)), fC2.T)
    )
    constC = constC1 + constC2

    return constC, hC1, hC2


def tensor_product(constC, hC1, hC2, T, nx=None):
    r"""Return the tensor for Gromov-Wasserstein fast computation

    The tensor is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-tensor-product>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    tens : array-like, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result


    .. _references-tensor-product:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    if nx is None:
        constC, hC1, hC2, T = list_to_array(constC, hC1, hC2, T)
        nx = get_backend(constC, hC1, hC2, T)

    A = -nx.dot(nx.dot(hC1, T), hC2.T)
    tens = constC + A
    # tens -= tens.min()
    return tens


def gwloss(constC, hC1, hC2, T, nx=None):
    r"""Return the Loss for Gromov-Wasserstein

    The loss is computed as described in Proposition 1 Eq. (6) in :ref:`[12] <references-gwloss>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    loss : float
        Gromov-Wasserstein loss


    .. _references-gwloss:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    tens = tensor_product(constC, hC1, hC2, T, nx)
    if nx is None:
        tens, T = list_to_array(tens, T)
        nx = get_backend(tens, T)

    return nx.sum(tens * T)


def gwggrad(constC, hC1, hC2, T, nx=None):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
        Gromov-Wasserstein gradient


    .. _references-gwggrad:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    return 2 * tensor_product(constC, hC1, hC2, T, nx)  # [12] Prop. 2 misses a 2 factor


def _spectral_norm(C, nx=None):
    r"""Return the spectral norm (largest singular value) of a square matrix.

    The norm is computed from eigendecompositions provided by the backend
    (:code:`nx.eigh`) so that the function works on arrays from all compatible
    backends. Symmetric matrices are handled directly through the eigenvalues
    of :math:`\mathbf{C}`, while for general matrices the eigenvalues of the
    symmetric matrix :math:`\mathbf{C}^T \mathbf{C}` are used:

    .. math::
        \|\mathbf{C}\|_2 = \max_i |\lambda_i(\mathbf{C})|
        \quad \text{if } \mathbf{C} = \mathbf{C}^T, \qquad
        \|\mathbf{C}\|_2 = \sqrt{\lambda_{\max}(\mathbf{C}^T \mathbf{C})}
        \quad \text{otherwise.}

    Parameters
    ----------
    C : array-like, shape (n, n)
        Square matrix.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    norm : array-like
        Spectral norm :math:`\|\mathbf{C}\|_2` of the matrix.
    """
    if nx is None:
        C = list_to_array(C)
        nx = get_backend(C)

    if nx.allclose(C, C.T, atol=1e-10):
        eigvals = nx.eigh(C)[0]
        return nx.max(nx.abs(eigvals))

    eigvals = nx.eigh(nx.dot(C.T, C))[0]
    return nx.sqrt(nx.maximum(nx.max(eigvals), 0.0))


def _project(
    Tin,
    p,
    q,
    y0,
    max_iter,
    tol,
    nx=None,
    accelerated=True,
    restart=True,
    restart_iter=500,
    lr_const=2.0,
    return_err=False,
):
    r"""(internal) Project a matrix onto the transport polytope of :math:`(\mathbf{p}, \mathbf{q})`.

    Euclidean projection of :math:`\mathbf{T}_{in}` (which need not be
    nonnegative) onto :math:`\{\mathbf{T} \geq 0: \mathbf{T}\mathbf{1} = \mathbf{p},
    \mathbf{T}^T\mathbf{1} = \mathbf{q}\}` using Nesterov-accelerated gradient
    ascent on the smooth dual of the projection problem, as described in [97]_.

    Parameters
    ----------
    Tin : array-like, shape (ns, nt)
        Matrix to project onto the transport polytope.
    p : array-like, shape (ns,)
        Probability distribution in the source space.
    q : array-like, shape (nt,)
        Probability distribution in the target space.
    y0 : array-like, shape (ns + nt,), optional
        Initial dual potentials (row then column), used for warm starting.
    max_iter : int
        Maximum number of dual gradient iterations.
    tol : float
        Stopping target for the marginal residual; the dual iterate stops when
        the residual is below ``tol / (6 + norm(y))``.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    accelerated : bool, optional
        Use Nesterov acceleration of the dual gradient ascent (default True).
    restart : bool, optional
        Use periodic momentum restarts (default True).
    restart_iter : int, optional
        Restart interval for the momentum (default 500).
    lr_const : float, optional
        Dual step size is ``lr_const / (ns + nt)`` (default 2., the practical
        value used in the reference implementation; use at most 1 for the safe
        Lipschitz step of the accelerated dual).
    return_err : bool, optional
        Also return the residual trajectory of the dual iterations.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Approximate Euclidean projection of `Tin` onto the polytope.
    y : array-like, shape (ns + nt,)
        Final dual potentials, that can be used to warm start another call.
    stats : dict
        Convergence information with keys ``n_iter``, ``converged``,
        ``final_err`` and ``threshold`` (and ``err`` when `return_err` is True).
    """
    if nx is None:
        Tin, p, q = list_to_array(Tin, p, q)
        nx = get_backend(Tin, p, q)

    m, n = Tin.shape
    lr = lr_const / (m + n)
    if y0 is None:
        y = nx.zeros(m + n, type_as=Tin)
    else:
        y = nx.copy(y0)
    z = nx.copy(y)
    t = 1.0

    err_traj = [] if return_err else None

    k = -1
    for k in range(max_iter):
        Tz = nx.maximum(Tin + z[:m, None] + z[None, m:], 0.0)
        grad = nx.concatenate([nx.sum(Tz, 1) - p, nx.sum(Tz, 0) - q])
        err = nx.norm(grad)
        normz = nx.norm(z)

        if return_err:
            err_traj.append(err)

        if err <= tol / (6.0 + normz):
            y = z
            break

        y_next = z - lr * grad
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)) if accelerated else t
        if restart and (k + 1) % restart_iter == 0:
            t = t_next = 1.0
        beta = (t - 1.0) / t_next
        z = y_next + beta * (y_next - y)
        y, t = y_next, t_next

    Tproj = nx.maximum(Tin + y[:m, None] + y[None, m:], 0.0)
    grad = nx.concatenate([nx.sum(Tproj, 1) - p, nx.sum(Tproj, 0) - q])
    final_err = float(nx.to_numpy(nx.norm(grad)))
    threshold = tol / (6.0 + float(nx.to_numpy(nx.norm(y))))
    stats = {
        "n_iter": k + 1,
        "converged": final_err <= threshold,
        "final_err": final_err,
        "threshold": threshold,
    }
    if return_err:
        stats["err"] = err_traj

    return Tproj, y, stats


def proj_transport_polytope(
    Tin,
    p,
    q,
    y0=None,
    max_iter=1000,
    tol=1e-9,
    accelerated=True,
    restart=True,
    restart_iter=500,
    lr_const=2.0,
    log=False,
    nx=None,
):
    r"""Projection onto the transport polytope of distributions :math:`(\mathbf{p}, \mathbf{q})`.

    Returns the (approximate) Euclidean projection of :math:`\mathbf{T}_{in}`
    onto the set :math:`\{\mathbf{T} \geq 0: \mathbf{T}\mathbf{1} = \mathbf{p},
    \mathbf{T}^T\mathbf{1} = \mathbf{q}\}`, solved with Nesterov-accelerated
    gradient ascent on the smooth dual of the projection problem [97]_. The
    dual problem is a concave quadratic maximization with Lipschitz gradient
    whose iterates read :math:`\mathbf{T}(\mathbf{y}) = \max(\mathbf{T}_{in} +
    \mathbf{y}_{rows} + \mathbf{y}_{cols}, 0)` for dual potentials
    :math:`\mathbf{y}`.

    This projection is the inner solver of the inexact projected-gradient
    Gromov-Wasserstein solver :func:`ot.gromov.gromov_wasserstein_ipg` but can
    also be used standalone.

    .. note:: This function is backend-compatible and will work on arrays from
        all compatible backends.

    Parameters
    ----------
    Tin : array-like, shape (ns, nt)
        Matrix to project; it need not be nonnegative.
    p : array-like, shape (ns,)
        Probability distribution in the source space.
    q : array-like, shape (nt,)
        Probability distribution in the target space.
    y0 : array-like, shape (ns + nt,), optional
        Initial dual potentials (row then column), used for warm starting.
        If let to its default value None, a zero start is taken.
    max_iter : int, optional
        Maximum number of dual gradient iterations (default 1000).
    tol : float, optional
        Stopping target for the marginal residual; the dual iterates stop when
        the residual is below ``tol / (6 + norm(y))`` (default 1e-9).
    accelerated : bool, optional
        Use Nesterov acceleration of the dual gradient ascent (default True).
    restart : bool, optional
        Use periodic momentum restarts (default True).
    restart_iter : int, optional
        Restart interval for the momentum (default 500).
    lr_const : float, optional
        Dual step size is ``lr_const / (ns + nt)`` (default 2., the practical
        value used in the reference implementation; use at most 1 for the safe
        Lipschitz step of the accelerated dual).
    log : bool, optional
        Record log if True.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Approximate Euclidean projection of `Tin` onto the transport polytope.
    log : dict
        Projection information with keys ``y`` (dual potentials), ``err``
        (residual trajectory), ``converged``, ``n_iter``, ``final_err``
        and ``threshold``.

    :ref:`[97] <references-proj-transport-polytope>`

    .. _references-proj-transport-polytope:

    References
    ----------
    .. [97] Liang, J., & Yang, R. "A Provably Convergent and Practical
        Algorithm for Gromov-Wasserstein Optimal Transport". Advances in
        Neural Information Processing Systems (NeurIPS), 2026.
    """
    arr = [Tin, p, q]
    if y0 is not None:
        arr.append(y0)

    if nx is None:
        arr = list_to_array(*arr)
        nx = get_backend(*arr)

    Tin, p, q = arr[0], arr[1], arr[2]
    y0 = arr[3] if len(arr) > 3 else None

    Tproj, y, stats = _project(
        Tin,
        p,
        q,
        y0,
        max_iter,
        tol,
        nx=nx,
        accelerated=accelerated,
        restart=restart,
        restart_iter=restart_iter,
        lr_const=lr_const,
        return_err=log,
    )

    if log:
        log = {
            "y": y,
            "err": stats["err"],
            "converged": stats["converged"],
            "n_iter": stats["n_iter"],
            "final_err": stats["final_err"],
            "threshold": stats["threshold"],
        }
        return Tproj, log

    return Tproj


def init_matrix_semirelaxed(C1, C2, p, loss_fun="square_loss", nx=None):
    r"""Return loss matrices and tensors for semi-relaxed Gromov-Wasserstein fast computation

    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of semi-relaxed Gromov-Wasserstein discrepancy.

    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`
    and adapted to the semi-relaxed problem where the second marginal is not a constant anymore.

    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces

    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a^2

                        f_2(b) &= b^2

                        h_1(a) &= a

                        h_2(b) &= 2b

    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :

    .. math::

        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)

        \mathrm{with} \ f_1(a) &= a \log(a) - a

                        f_2(b) &= b

                        h_1(a) &= a

                        h_2(b) &= \log(b)
    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Probability distribution in the source space
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6) adapted to srGW
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    fC2t: array-like, shape (nt, nt)
        :math:`\mathbf{f2}(\mathbf{C2})^\top` matrix in Eq. (6)


    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
        "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
        International Conference on Learning Representations (ICLR), 2022.
    """
    if nx is None:
        C1, C2, p = list_to_array(C1, C2, p)
        nx = get_backend(C1, C2, p)

    fC1, fC2, hC1, hC2 = _transform_matrix(C1, C2, loss_fun, nx)

    constC = nx.dot(
        nx.dot(fC1, nx.reshape(p, (-1, 1))), nx.ones((1, C2.shape[0]), type_as=p)
    )

    fC2t = fC2.T
    return constC, hC1, hC2, fC2t


def semirelaxed_init_plan(
    C1,
    C2,
    p,
    M=None,
    alpha=1.0,
    method="product",
    use_target=True,
    random_state=0,
    nx=None,
):
    r"""
    Heuristics to initialize the semi-relaxed (F)GW transport plan
    :math:`\mathbf{T} \in \mathcal{U}_{nt}(\mathbf{p})`, between a graph
    :math:`(\mathbf{C1}, \mathbf{p})` and a structure matrix :math:`\mathbf{C2}`,
    where :math:`\mathcal{U}_{nt}(\mathbf{p}) = \{\mathbf{T} \in \mathbb{R}_{+}^{ns * nt}, \mathbf{T} \mathbf{1}_{nt} = \mathbf{p} \}`.
    Available methods are:
        - "product" or "random_product": :math:`\mathbf{T} = \mathbf{pq}^{T}`
          with :math:`\mathbf{q}` uniform or randomly samples in the nt probability simplex.

        - "random": random sampling in :math:`\mathcal{U}_{nt}(\mathbf{p})`.

        - "fluid": Fluid algorithm from networkx for graph partitioning.

        - "spectral", "kmeans" : Spectral or Kmeans clustering from sklearn.

        - "fluid_soft", "spectral_soft", "kmeans_soft": :math:`\mathbf{T}_0` given
          by corresponding clustering with target marginal :math:`\mathbf{q}_0`, further
          centered as :math:`\mathbf{T} = (\mathbf{T}_0 + \mathbf{pq}_0^T) / 2` .

    If a metric cost matrix between features across domains :math:`\mathbf{M}`
    is a provided, it will be used as cost matrix in a semi-relaxed Wasserstein
    problem providing :math:`\mathbf{T}_M \in \mathcal{U}_{nt}(\mathbf{p})`. Then
    the outputted transport plan is :math:`\alpha \mathbf{T}  + (1 - \alpha ) \mathbf{T}_{M}`.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space.
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space.
    p : array-like, shape (ns,), optional.
        Probability distribution in the source space. If let to None, uniform
        weights are assumed on C1.
    M : array-like, shape (ns, nt), optional.
        Metric cost matrix between features across domains.
    alpha : float, optional
        Trade-off parameter (0 <= alpha <= 1)
    method : str, optional
        Method to initialize the transport plan. The default is 'product'.
    use_target : bool, optional.
        Whether or not to use the target structure/features to further align
        transport plan provided by the `method`.
    random_state: int, optional
        Random seed used for stochastic methods.
    nx : backend, optional
        POT backend.

    Returns
    -------
    T : array-like, shape (ns, ns)
        Admissible transport plan for the sr(F)GW problems.

    References
    ----------
    .. [48]  Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.

    """
    list_partitioning_methods = [
        "fluid",
        "spectral",
        "kmeans",
        "fluid_soft",
        "spectral_soft",
        "kmeans_soft",
    ]

    if method not in list_partitioning_methods + [
        "product",
        "random_product",
        "random",
    ]:
        raise ValueError(f"Unsupported initialization method = {method}.")

    if (method in ["kmeans", "kmeans_soft"]) and (not sklearn_import):
        raise ValueError(f"Scikit-learn must be installed to use method = {method}")

    if (method in ["fluid", "fluid_soft"]) and (not networkx_import):
        raise ValueError(f"Networkx must be installed to use method = {method}")

    if nx is None:
        nx = get_backend(C1, C2, p, M)

    n = C1.shape[0]
    m = C2.shape[0]
    min_size = min(n, m)

    if method in list_partitioning_methods:
        if n > m:  # partition C1 to deduce map from C1 to C2
            C_to_partition = nx.to_numpy(C1)
        elif m > n:  # partition C2 to deduce map from C1 to C2
            C_to_partition = nx.to_numpy(C2)
        else:  # equal size -> simple Wasserstein alignment
            C_to_partition = None
            warnings.warn(
                "Both structures have the same size so no partitioning is"
                "performed to initialize the transport plan even though"
                f"initialization method is {method}",
                stacklevel=2,
            )

        def get_transport_from_partition(part):
            if n > m:  # partition C1 to deduce map from C1 to C2
                T_ = nx.eye(m, type_as=C1)[part]
                T_ = p[:, None] * T_
                q = nx.sum(T_, 0)

                factored_C1 = nx.dot(nx.dot(T_.T, C1), T_) / nx.outer(q, q)

                # alignment of both structure seen as feature matrices
                if use_target:
                    M_structure = euclidean_distances(factored_C1, C2)
                    T_emd = emd(q, q, M_structure)
                    inv_q = 1.0 / q

                    T = nx.dot(T_, inv_q[:, None] * T_emd)
                else:
                    T = T_

            elif m > n:
                T_ = nx.eye(n, type_as=C1)[part] / m  # assume uniform masses on C2
                q = nx.sum(T_, 0)

                factored_C2 = nx.dot(nx.dot(T_.T, C2), T_) / nx.outer(q, q)

                # alignment of both structure seen as feature matrices
                M_structure = euclidean_distances(factored_C2, C1)
                T_emd = emd(q, p, M_structure)
                inv_q = 1.0 / q

                T = nx.dot(T_, inv_q[:, None] * T_emd).T
                q = nx.sum(T, 0)  # uniform one
            else:
                # alignment of both structure seen as feature matrices
                M_structure = euclidean_distances(C1, C2)
                q = p
                T = emd(p, q, M_structure)

            return T, q

    # Handle initialization via structure information

    if method == "product":
        q = nx.ones(m, type_as=C1) / m
        T = nx.outer(p, q)

    elif method == "random_product":
        np.random.seed(random_state)
        q = np.random.uniform(0, m, size=(m,))
        q = q / q.sum()
        q = nx.from_numpy(q, type_as=p)
        T = nx.outer(p, q)

    elif method == "random":
        np.random.seed(random_state)
        U = np.random.uniform(0, n * m, size=(n, m))
        U = (p / U.sum(1))[:, None] * U
        T = nx.from_numpy(U, type_as=C1)

    elif method in ["fluid", "fluid_soft"]:
        # compute fluid partitioning on the biggest graph
        if C_to_partition is None:
            T, q = get_transport_from_partition(None)
        else:
            graph = from_numpy_array(C_to_partition)
            part_sets = asyn_fluidc(graph, min_size, seed=random_state)
            part = np.zeros(C_to_partition.shape[0], dtype=int)
            for iset_, set_ in enumerate(part_sets):
                set_ = list(set_)
                part[set_] = iset_
            part = nx.from_numpy(part)

            T, q = get_transport_from_partition(part)

        if "soft" in method:
            T = (T + nx.outer(p, q)) / 2.0

    elif method in ["spectral", "spectral_soft"]:
        # compute spectral partitioning on the biggest graph
        if C_to_partition is None:
            T, q = get_transport_from_partition(None)
        else:
            sc = SpectralClustering(
                n_clusters=min_size, random_state=random_state, affinity="precomputed"
            ).fit(C_to_partition)
            part = sc.labels_
            T, q = get_transport_from_partition(part)

        if "soft" in method:
            T = (T + nx.outer(p, q)) / 2.0

    elif method in ["kmeans", "kmeans_soft"]:
        # compute spectral partitioning on the biggest graph
        if C_to_partition is None:
            T, q = get_transport_from_partition(None)
        else:
            km = KMeans(n_clusters=min_size, random_state=random_state, n_init=1).fit(
                C_to_partition
            )

            part = km.labels_
            T, q = get_transport_from_partition(part)

        if "soft" in method:
            T = (T + nx.outer(p, q)) / 2.0

    if M is not None:
        # Add feature information solving a semi-relaxed Wasserstein problem
        # get minimum by rows as binary mask
        TM = nx.ones(1, type_as=p) * (M == nx.reshape(nx.min(M, axis=1), (-1, 1)))
        TM *= nx.reshape((p / nx.sum(TM, axis=1)), (-1, 1))

        T = alpha * T + (1 - alpha) * TM

    return T


def update_barycenter_structure(
    Ts,
    Cs,
    lambdas,
    p=None,
    loss_fun="square_loss",
    target=True,
    check_zeros=True,
    nx=None,
):
    r"""
    Updates :math:`\mathbf{C}` according to the inner loss L with the `S`
    :math:`\mathbf{T}_s` couplings calculated at each iteration of variants of
    the GW barycenter problem (e.g GW :ref:`[12]`, srGW :ref:`[48]`).
    If `target=True` it solves for:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad
        \sum_s \lambda_s \sum_{i,j,k,l}
        L(\mathbf{C}^{(s)}_{i,k}, \mathbf{C}_{j,l}) \mathbf{T}^{(s)}_{i,j} \mathbf{T}^{(s)}_{k,l}

    Else it solves the symmetric problem:

    .. math::

        \mathbf{C}^* = \mathop{\arg \min}_{\mathbf{C}\in \mathbb{R}^{N \times N}} \quad
        \sum_s \lambda_s \sum_{i,j,k,l}
        L(\mathbf{C}_{j,l}, \mathbf{C}^{(s)}_{i,k}) \mathbf{T}^{(s)}_{i,j} \mathbf{T}^{(s)}_{k,l}

    Where :

    - :math:`\mathbf{C}^{(s)}`: pairwise matrix in the s^{th} source space .
    - :math:`\mathbf{C}`: pairwise matrix in the target space.
    - :math:`L`: inner divergence for the GW loss

    Parameters
    ----------
    Ts : list of S array-like of shape (ns, N) if `target=True` else (N, ns).
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Cs : list of S array-like, shape(ns, ns)
        Metric cost matrices.
    lambdas : list of float,
        List of the `S` spaces' weights.
    p : array-like, shape (N,) or (S,N)
        Masses or list of masses in the targeted barycenter.
    loss_fun : str, optional. Default is 'square_loss'
        Name of loss function to use in ['square_loss', 'kl_loss'].
    target: bool, optional. Default is True.
        Whether the barycenter is positioned as target (True) or source (False).
    check_zeros: bool, optional. Default is True.
        Whether to check if marginals on the barycenter contains zeros or not.
        Can be set to False to gain time if marginals are known to be positive.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    C : array-like, shape (`nt`, `nt`)
        Updated :math:`\mathbf{C}` matrix.

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.

    """

    if nx is None:
        arr = [*Ts, *Cs, p]
        nx = get_backend(*arr)

    S = len(Ts)

    if p is None:
        p = nx.concatenate(
            [nx.sum(Ts[s], int(not target))[None, :] for s in range(S)], axis=0
        )

    # compute coefficients for the barycenter coming from marginals

    if len(p.shape) == 1:  # shared target masses potentially with zeros
        if check_zeros:
            inv_p = nx.nan_to_num(1.0 / p, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            inv_p = 1.0 / p

        prod = nx.outer(inv_p, inv_p)

    else:
        quotient = sum([lambdas[s] * nx.outer(p[s], p[s]) for s in range(S)])
        if check_zeros:
            prod = nx.nan_to_num(1.0 / quotient, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            prod = 1.0 / quotient

    # compute coefficients for the barycenter coming from Ts and Cs

    if loss_fun == "square_loss":
        if target:
            list_structures = [
                lambdas[s] * nx.dot(nx.dot(Ts[s].T, Cs[s]), Ts[s]) for s in range(S)
            ]
        else:
            list_structures = [
                lambdas[s] * nx.dot(nx.dot(Ts[s], Cs[s]), Ts[s].T) for s in range(S)
            ]

        return sum(list_structures) * prod

    elif loss_fun == "kl_loss":
        if target:
            list_structures = [
                lambdas[s] * nx.dot(nx.dot(Ts[s].T, Cs[s]), Ts[s]) for s in range(S)
            ]

            return sum(list_structures) * prod
        else:
            list_structures = [
                lambdas[s]
                * nx.dot(nx.dot(Ts[s], nx.log(nx.maximum(Cs[s], 1e-16))), Ts[s].T)
                for s in range(S)
            ]

            return nx.exp(sum(list_structures) * prod)

    else:
        raise ValueError(f"not supported loss_fun = {loss_fun}")


def update_barycenter_feature(
    Ts,
    Ys,
    lambdas,
    p=None,
    loss_fun="square_loss",
    target=True,
    check_zeros=True,
    nx=None,
):
    r"""Updates the feature with respect to the `S` :math:`\mathbf{T}_s`
    couplings calculated at each iteration of variants of the FGW
    barycenter problem with inner wasserstein loss `loss_fun`
    (e.g FGW :ref:`[24]`, srFGW :ref:`[48]`).
    If `target=True` the barycenter is considered as the target else as the source.

    Parameters
    ----------
    Ts : list of S array-like of shape (ns, N) if `target=True` else (N, ns).
        The `S` :math:`\mathbf{T}_s` couplings calculated at each iteration.
    Ys : list of S array-like, shape (ns, d)
        Feature matrices.
    lambdas : list of float
        List of the `S` spaces' weights
    p : array-like, shape (N,) or (S,N)
        Masses or list of masses in the targeted barycenter.
    loss_fun : str, optional. Default is 'square_loss'
        Name of loss function to use in ['square_loss'].
    target: bool, optional. Default is True.
        Whether the barycenter is positioned as target (True) or source (False).
    check_zeros: bool, optional. Default is True.
        Whether to check if marginals on the barycenter contains zeros or not.
        Can be set to False to gain time if marginals are known to be positive.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    X : array-like, shape (N, d)

    References
    ----------
    .. [24] Vayer Titouan, Chapel Laetitia, Flamary Rémi, Tavenard Romain and Courty Nicolas
        "Optimal Transport for structured data with application on graphs"
        International Conference on Machine Learning (ICML). 2019.

    .. [48] Cédric Vincent-Cuaz, Rémi Flamary, Marco Corneli, Titouan Vayer, Nicolas Courty.
            "Semi-relaxed Gromov-Wasserstein divergence and applications on graphs"
            International Conference on Learning Representations (ICLR), 2022.
    """
    if nx is None:
        arr = [*Ts, *Ys, p]
        nx = get_backend(*arr)

    if loss_fun != "square_loss":
        raise ValueError(f"not supported loss_fun = {loss_fun}")

    S = len(Ts)

    if target:
        list_features = [lambdas[s] * nx.dot(Ts[s].T, Ys[s]) for s in range(S)]
    else:
        list_features = [lambdas[s] * nx.dot(Ts[s], Ys[s]) for s in range(S)]

    if p is None:
        p = nx.concatenate(
            [nx.sum(Ts[s], int(not target))[None, :] for s in range(S)], axis=0
        )

    if len(p.shape) == 1:  # shared target masses potentially with zeros
        if check_zeros:
            inv_p = nx.nan_to_num(1.0 / p, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            inv_p = 1.0 / p
    else:
        p_sum = sum([lambdas[s] * p[s] for s in range(S)])
        if check_zeros:
            inv_p = nx.nan_to_num(1.0 / p_sum, nan=1.0, posinf=1.0, neginf=1.0)
        else:
            inv_p = 1.0 / p_sum

    return sum(list_features) * inv_p[:, None]


############################################################################
# Methods related to fused unbalanced GW and unbalanced Co-Optimal Transport.
############################################################################


def div_to_product(pi, a, b, pi1=None, pi2=None, divergence="kl", mass=True, nx=None):
    r"""Fast computation of the Bregman divergence between an arbitrary measure and a product measure.
    Only support for Kullback-Leibler and half-squared L2 divergences.

    - For half-squared L2 divergence:

    .. math::
        \frac{1}{2} || \pi - a \otimes b ||^2
        = \frac{1}{2} \Big[ \sum_{i, j} \pi_{ij}^2 + (\sum_i a_i^2) ( \sum_j b_j^2) - 2 \sum_{i, j} a_i \pi_{ij} b_j \Big]

    - For Kullback-Leibler divergence:

    .. math::
        KL(\pi | a \otimes b)
        = \langle \pi, \log \pi \rangle - \langle \pi_1, \log a \rangle
        - \langle \pi_2, \log b \rangle - m(\pi) + m(a) m(b)

    where :

    - :math:`\pi` is the (`dim_a`, `dim_b`) transport plan
    - :math:`\pi_1` and :math:`\pi_2` are the marginal distributions
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are source and target unbalanced distributions
    - :math:`m` denotes the mass of the measure

    Parameters
    ----------
    pi : array-like (dim_a, dim_b)
        Transport plan
    a : array-like (dim_a,)
        Unnormalized histogram of dimension `dim_a`
    b : array-like (dim_b,)
        Unnormalized histogram of dimension `dim_b`
    pi1 : array-like (dim_a,), optional (default = None)
        Marginal distribution with respect to the first dimension of the transport plan
        Only used in case of Kullback-Leibler divergence.
    pi2 : array-like (dim_a,), optional (default = None)
        Marginal distribution with respect to the second dimension of the transport plan
        Only used in case of Kullback-Leibler divergence.
    divergence : string, default = "kl"
        Bregman divergence, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    mass : bool, optional. Default is False.
        Only used in case of Kullback-Leibler divergence.
        If False, calculate the relative entropy.
        If True, calculate the Kullback-Leibler divergence.
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    -------
    Bregman divergence between an arbitrary measure and a product measure.
    """

    arr = [pi, a, b, pi1, pi2]

    if nx is None:
        nx = get_backend(*arr, pi1, pi2)

    if divergence == "kl":
        if pi1 is None:
            pi1 = nx.sum(pi, 1)
        if pi2 is None:
            pi2 = nx.sum(pi, 0)

        res = (
            nx.sum(pi * nx.log(pi + 1.0 * (pi == 0)))
            - nx.sum(pi1 * nx.log(a))
            - nx.sum(pi2 * nx.log(b))
        )
        if mass:
            res = res - nx.sum(pi1) + nx.sum(a) * nx.sum(b)

    elif divergence == "l2":
        res = (
            nx.sum(pi**2) + nx.sum(a**2) * nx.sum(b**2) - 2 * nx.dot(a, nx.dot(pi, b))
        ) / 2

    return res


def div_between_product(mu, nu, alpha, beta, divergence, nx=None):
    r"""Fast computation of the Bregman divergence between two product measures.
    Only support for Kullback-Leibler and half-squared L2 divergences.

    For half-squared L2 divergence:

    .. math::
        \frac{1}{2} || \mu \otimes \nu, \alpha \otimes \beta ||^2
        = \frac{1}{2} \Big[ ||\alpha||^2 ||\beta||^2 + ||\mu||^2 ||\nu||^2 - 2 \langle \alpha, \mu \rangle \langle \beta, \nu \rangle \Big]

    For Kullback-Leibler divergence:

    .. math::
        KL(\mu \otimes \nu, \alpha \otimes \beta)
        = m(\mu) * KL(\nu, \beta) + m(\nu) * KL(\mu, \alpha) + (m(\mu) - m(\alpha)) * (m(\nu) - m(\beta))

    where:

    - :math:`\mu` and :math:`\alpha` are two measures having the same shape.
    - :math:`\nu` and :math:`\beta` are two measures having the same shape.
    - :math:`m` denotes the mass of the measure

    Parameters
    ----------
    mu : array-like
        vector or matrix
    nu : array-like
        vector or matrix
    alpha : array-like
        vector or matrix with the same shape as `\mu`
    beta : array-like
        vector or matrix with the same shape as `\nu`
    divergence : string, default = "kl"
        Bregman divergence, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    Bregman divergence between two product measures.
    """

    if nx is None:
        nx = get_backend(mu, nu, alpha, beta)

    if divergence == "kl":
        m_mu, m_nu = nx.sum(mu), nx.sum(nu)
        m_alpha, m_beta = nx.sum(alpha), nx.sum(beta)
        const = (m_mu - m_alpha) * (m_nu - m_beta)
        res = (
            m_nu * nx.kl_div(mu, alpha, mass=True)
            + m_mu * nx.kl_div(nu, beta, mass=True)
            + const
        )

    elif divergence == "l2":
        res = (
            nx.sum(alpha**2) * nx.sum(beta**2)
            - 2 * nx.sum(alpha * mu) * nx.sum(beta * nu)
            + nx.sum(mu**2) * nx.sum(nu**2)
        ) / 2

    return res


# Support functions for BCD schemes
def uot_cost_matrix(data, pi, tuple_p, hyperparams, divergence, reg_type, nx=None):
    r"""The Block Coordinate Descent algorithm for FUGW and UCOOT
    requires solving an UOT problem in each iteration.
    In particular, we need to specify the following inputs:

    - Cost matrix

    - Hyperparameters (marginal-relaxations and regularization)

    - Reference measures in the marginal-relaxation and regularization terms

    This method returns the cost matrix.
    The method :any:`ot.gromov.uot_parameters_and_measures` returns the rest of the inputs.

    Parameters
    ----------
    data : tuple of arrays
        vector or matrix
    pi : array-like
        vector or matrix
    tuple_p : tuple of arrays
        Tuple of reference measures in the marginal-relaxation terms
        w.r.t the (either sample or feature) coupling
    hyperparams : tuple of floats
        Hyperparameters of marginal-relaxation and regularization terms
        in the fused unbalanced across-domain divergence
    divergence : string, default = "kl"
        Bregman divergence, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    reg_type : string,
        Type of regularization term in the fused unbalanced across-domain divergence

        - `reg_type = "joint"` corresponds to FUGW

        - `reg_type = "independent"` corresponds to UCOOT
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    Cost matrix of the UOT subroutine for UCOOT and FUGW
    """

    X_sqr, Y_sqr, X, Y, M = data
    rho_x, rho_y, eps = hyperparams
    a, b = tuple_p

    if nx is None:
        nx = get_backend(X, Y, a, b)

    pi1, pi2 = nx.sum(pi, 1), nx.sum(pi, 0)
    A, B = nx.dot(X_sqr, pi1), nx.dot(Y_sqr, pi2)
    uot_cost = A[:, None] + B[None, :] - 2 * nx.dot(nx.dot(X, pi), Y.T)
    if M is not None:
        uot_cost = uot_cost + M

    if divergence == "kl":
        if rho_x != float("inf") and rho_x != 0:
            uot_cost = uot_cost + rho_x * nx.kl_div(pi1, a, mass=False)
        if rho_y != float("inf") and rho_y != 0:
            uot_cost = uot_cost + rho_y * nx.kl_div(pi2, b, mass=False)
        if reg_type == "joint" and eps > 0:
            uot_cost = uot_cost + eps * div_to_product(
                pi, a, b, pi1, pi2, divergence, mass=False, nx=nx
            )

    return uot_cost


def uot_parameters_and_measures(
    pi, tuple_weights, hyperparams, reg_type, divergence, nx
):
    r"""The Block Coordinate Descent algorithm for FUGW and UCOOT
    requires solving an UOT problem in each iteration.
    In particular, we need to specify the following inputs:

    - Cost matrix

    - Hyperparameters (marginal-relaxations and regularization)

    - Reference measures in the marginal-relaxation and regularization terms

    The method :any:`ot.gromov.uot_cost_matrix` returns the cost matrix.
    This method returns the rest of the inputs.

    Parameters
    ----------
    pi : array-like
        vector or matrix
    tuple_weights : tuple of arrays
        Tuple of reference measures in the marginal-relaxation and regularization terms
        w.r.t the (either sample or feature) coupling
    hyperparams : tuple of floats
        Hyperparameters of marginal-relaxation and regularization terms
        in the fused unbalanced across-domain divergence
    reg_type : string,
        Type of regularization term in the fused unbalanced across-domain divergence

        - `reg_type = "joint"` corresponds to FUGW

        - `reg_type = "independent"` corresponds to UCOOT
    divergence : string, default = "kl"
        Bregman divergence, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    Tuple of hyperparameters and distributions (weights)
    """

    rho_x, rho_y, eps = hyperparams
    wx, wy, wxy = tuple_weights

    if divergence == "l2":
        pi1, pi2 = nx.sum(pi, 1), nx.sum(pi, 0)
        l2_pi1, l2_pi2, l2_pi = nx.sum(pi1**2), nx.sum(pi2**2), nx.sum(pi**2)

        weighted_wx = wx * nx.sum(pi1 * wx) / l2_pi1
        weighted_wy = wy * nx.sum(pi2 * wy) / l2_pi2
        weighted_wxy = wxy * nx.sum(pi * wxy) / l2_pi if reg_type == "joint" else wxy
        weighted_w = (weighted_wx, weighted_wy, weighted_wxy)

        new_rho = (rho_x * l2_pi1, rho_y * l2_pi2)
        new_eps = eps * l2_pi if reg_type == "joint" else eps

    elif divergence == "kl":
        mass = nx.sum(pi)
        new_rho = (rho_x * mass, rho_y * mass)
        new_eps = mass * eps if reg_type == "joint" else eps
        weighted_w = tuple_weights

    return weighted_w, new_rho, new_eps


def fused_unbalanced_across_spaces_cost(
    M_linear,
    data,
    tuple_pxy_samp,
    tuple_pxy_feat,
    pi_samp,
    pi_feat,
    hyperparams,
    divergence,
    reg_type,
    nx,
):
    r"""Return the fused unbalanced across-space divergence between two spaces

    Parameters
    ----------
    M_linear : tuple of arrays
        Pair of cost matrices corresponding to the Wasserstein terms w.r.t sample and feature couplings
    data : tuple of arrays
        Tuple of input spaces represented as matrices
    tuple_pxy_samp : tuple of arrays
        Tuple of reference measures in the marginal-relaxation and regularization terms
        w.r.t the sample coupling
    tuple_pxy_feat : tuple of arrays
        Tuple of reference measures in the marginal-relaxation and regularization terms
        w.r.t the feature coupling
    pi_samp : array-like
        Sample coupling
    pi_feat : array-like
        Feature coupling
    hyperparams : tuple of floats
        Hyperparameters of marginal-relaxation and regularization terms
        in the fused unbalanced across-domain divergence
    divergence : string, default = "kl"
        Bregman divergence, either "kl" (Kullback-Leibler divergence) or "l2" (half-squared L2 divergence)
    reg_type : string,
        Type of regularization term in the fused unbalanced across-domain divergence

        - `reg_type = "joint"` corresponds to FUGW

        - `reg_type = "independent"` corresponds to UCOOT
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.

    Returns
    ----------
    Fused unbalanced across-space divergence between two spaces
    """

    rho_x, rho_y, eps_samp, eps_feat = hyperparams
    M_samp, M_feat = M_linear
    px_samp, py_samp, pxy_samp = tuple_pxy_samp
    px_feat, py_feat, pxy_feat = tuple_pxy_feat
    X_sqr, Y_sqr, X, Y = data

    pi1_samp, pi2_samp = nx.sum(pi_samp, 1), nx.sum(pi_samp, 0)
    pi1_feat, pi2_feat = nx.sum(pi_feat, 1), nx.sum(pi_feat, 0)

    A_sqr = nx.dot(nx.dot(X_sqr, pi1_feat), pi1_samp)
    B_sqr = nx.dot(nx.dot(Y_sqr, pi2_feat), pi2_samp)
    AB = nx.dot(nx.dot(X, pi_feat), Y.T) * pi_samp
    linear_cost = A_sqr + B_sqr - 2 * nx.sum(AB)

    ucoot_cost = linear_cost
    if M_samp is not None:
        ucoot_cost = ucoot_cost + nx.sum(pi_samp * M_samp)
    if M_feat is not None:
        ucoot_cost = ucoot_cost + nx.sum(pi_feat * M_feat)

    if rho_x != float("inf") and rho_x != 0:
        ucoot_cost = ucoot_cost + rho_x * div_between_product(
            pi1_samp, pi1_feat, px_samp, px_feat, divergence, nx
        )
    if rho_y != float("inf") and rho_y != 0:
        ucoot_cost = ucoot_cost + rho_y * div_between_product(
            pi2_samp, pi2_feat, py_samp, py_feat, divergence, nx
        )

    if reg_type == "joint" and eps_samp != 0:
        div_cost = div_between_product(
            pi_samp, pi_feat, pxy_samp, pxy_feat, divergence, nx
        )
        ucoot_cost = ucoot_cost + eps_samp * div_cost
    elif reg_type == "independent":
        if eps_samp != 0:
            div_samp = div_to_product(
                pi_samp,
                px_samp,
                py_samp,
                pi1_samp,
                pi2_samp,
                divergence,
                mass=True,
                nx=nx,
            )
            ucoot_cost = ucoot_cost + eps_samp * div_samp
        if eps_feat != 0:
            div_feat = div_to_product(
                pi_feat,
                px_feat,
                py_feat,
                pi1_feat,
                pi2_feat,
                divergence,
                mass=True,
                nx=nx,
            )
            ucoot_cost = ucoot_cost + eps_feat * div_feat

    return linear_cost, ucoot_cost
