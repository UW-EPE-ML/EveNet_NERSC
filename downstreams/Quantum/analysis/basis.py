from typing import Any

import numpy as np
import vector
from vector._methods import VectorProtocolSpatial


def helicity_basis(particle: vector.Vector) -> dict[str, VectorProtocolSpatial | Any]:
    """
    Helicity basis: https://arxiv.org/pdf/2305.07075
    Returns the helicity basis for a given particle
    """

    k_hat = particle.to_pxpypz().unit()

    # Define beam direction
    p_hat = vector.Vector(x=0, y=0, z=1)
    y_p = p_hat.dot(k_hat)
    r_p = np.sqrt(1 - y_p ** 2)

    r_hat = 1 / r_p * (p_hat - y_p * k_hat)
    r_hat = r_hat.unit()

    n_hat = r_hat.cross(k_hat)
    n_hat = n_hat.unit()

    return {"k": k_hat, "r": r_hat, "n": n_hat}


if __name__ == "__main__":
    p = vector.Vector(x=44.9603, y=-0.893463, z=-505.204, t=507.204)
    print(helicity_basis(p))