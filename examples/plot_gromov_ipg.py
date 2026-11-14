# -*- coding: utf-8 -*-
"""
===================================================
Gromov-Wasserstein with the inexact projected gradient solver
===================================================

.. note::
    Example added in release: 0.9.8.

This example illustrates the computation of Gromov-Wasserstein (GW) matchings
between two metric measure spaces with the inexact projected gradient (IPG)
solver :func:`ot.gromov.gromov_wasserstein_ipg` [97], and compares it with the
conditional gradient (CG) solver :func:`ot.gromov.gromov_wasserstein` [12].

Contrary to the conditional gradient solver, which solves a linear optimal
transport problem at each iteration, the IPG solver takes a gradient step on
the (nonconvex) GW objective and then projects the iterate onto the transport
polytope only approximately, with a Nesterov-accelerated dual gradient ascent
whose dual potentials are warm started between iterations. This yields a
single-loop algorithm with convergence guarantees to a stationary point of the
GW problem [97], at a lower per-iteration cost.

We sample two 2D Gaussian distributions, the target being a stretched and
rotated version of the source, and recover their GW coupling with both
solvers.

[12] Gabriel Peyré, Marco Cuturi, and Justin Solomon (2016),
"Gromov-Wasserstein averaging of kernel and distance matrices".
International Conference on Machine Learning (ICML).

[97] Liang, J., & Yang, R. "A Provably Convergent and Practical Algorithm for
Gromov-Wasserstein Optimal Transport". Advances in Neural Information
Processing Systems (NeurIPS), 2026.

"""

# License: MIT License

# sphinx_gallery_thumbnail_number = 1

import numpy as np
import matplotlib.pylab as pl

import ot
from ot.gromov import gromov_wasserstein, gromov_wasserstein2
from ot.gromov import gromov_wasserstein_ipg, gromov_wasserstein_ipg2

#############################################################################
#
# Generate the two distributions
# ------------------------------
#

np.random.seed(0)

n_samples = 30  # nb samples

mu_s = np.array([0, 0])
cov_s = np.array([[1, 0], [0, 1]])

xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s)

# the target is a stretched, rotated and noisy version of the source
theta = np.pi / 4
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
xt = 2.5 * xs.dot(rot) + 3 + 0.5 * np.random.randn(n_samples, 2)

p = ot.unif(n_samples)
q = ot.unif(n_samples)

fig = pl.figure(1)
pl.scatter(xs[:, 0], xs[:, 1], c="C0", label="Source samples")
pl.scatter(xt[:, 0], xt[:, 1], c="C3", label="Target samples")
pl.legend()
pl.title("Source and target distributions")

#############################################################################
#
# Compute the GW matchings with both solvers
# ------------------------------------------
#

C1 = ot.dist(xs, xs)
C2 = ot.dist(xt, xt)

C1 /= C1.max()
C2 /= C2.max()

G_cg = gromov_wasserstein(C1, C2, p, q, "square_loss")
gw_cg = gromov_wasserstein2(C1, C2, p, q, "square_loss")

G_ipg, log = gromov_wasserstein_ipg(
    C1,
    C2,
    p,
    q,
    "square_loss",
    max_iter=1000,
    max_iter_proj=1000,
    proj_lr_const=1.0,
    log=True,
)
gw_ipg = gromov_wasserstein_ipg2(
    C1,
    C2,
    p,
    q,
    "square_loss",
    max_iter=1000,
    max_iter_proj=1000,
    proj_lr_const=1.0,
)

print(f"GW loss (conditional gradient)  : {gw_cg:.4f}")
print(f"GW loss (inexact proj. gradient): {gw_ipg:.4f}")
print(f"Max. coupling difference        : {np.abs(G_cg - G_ipg).max():.4f}")

#############################################################################
#
# Plot the couplings
# ------------------
#
# Both solvers recover a coupling aligning the two distributions up to the
# isometric transformation relating them.

pl.figure(2, (12, 4))

pl.subplot(131)
pl.imshow(G_cg, cmap="Greys")
pl.title("Coupling\n(conditional gradient)")

pl.subplot(132)
pl.imshow(G_ipg, cmap="Greys")
pl.title("Coupling\n(inexact projected gradient)")

pl.subplot(133)
pl.imshow(np.abs(G_cg - G_ipg), cmap="Greys")
pl.title("Coupling difference")

pl.show()
