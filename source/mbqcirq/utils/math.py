"""Collect utility math functions and tabulated constant values, vectors,
matrices etc...
"""

import numpy as np

from typing import List


#################
#####   Constants
#################


identity = np.array([[1., 0.], [0., 1.]])
sigma_x = np.array([[0., 1.], [1., 0.]])
sigma_y = np.array([[0., -1.j], [1.j, 0.]])
sigma_z = np.array([[1., 0.], [0., -1.]])
pauli_2x2_basis = {'identity': identity,
                          'sigma_x': sigma_x,
                          'sigma_y': sigma_y,
                          'sigma_z': sigma_z}


#################
####    Functions
#################


def decompose_into_pauli_2x2_basis(matrix: np.ndarray) -> List[complex]:
    """Return the coefficient for the decomposition of a generic 2x2 matrix
    into the pauli basis for 2x2 matrices, so that if :math:`A` is a 2x2 matrix
    then

    .. math::

        A = \sum_{i=0}^3 a_k \sigma_k

    where :math:`\sigma_0` is the :math:`2 \times 2` identity matrix,
    :math:`\sigma_i` for :math:`i=1, 2, 3` are the x, y, and z pauli matrices,
    while :math:`\{a_k\}_{k=0}^3` are the coefficients of the decomposition.


    :param matrix: Matrix to decompose.
    :type matrix: np.ndarray
    :return: The coefficient :math:`a_0, a_1, a_2, a_3` of the decomposition.
    :rtype: List[complex]
    """
    assert isinstance(matrix, np.ndarray), 'matrix should be a numpy array.'
    assert matrix.shape == (2, 2), 'matrix should have shape (2,2) (i.e. a ' \
                                   '2x2 matrix), but it has shape ' \
                                   '{}'.format(matrix.shape)
    return [complex(0.5 * np.trace(np.matmul(b, matrix)))
            for b in pauli_2x2_basis.values()]
