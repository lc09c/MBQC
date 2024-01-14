# title: 'single_qubits.py'
# author: Curcuraci L.
# date: 13/01/2024
#
# scope: Collect all the single-qubit measurement-based quantum gates.

"""Single-qubit measurement-based quantum gates."""

#################
#####   Libraries
#################


import numpy as np
import cirq
import networkx as nx

from mbqcirq.measurements.projectors import Projector_B_phi
from mbqcirq.gates import MBQGate

from typing import Dict, Any


###############
#####   Classes
###############


class SU2Gate(MBQGate):
    """MBQGate implementing an arbitrary SU2 rotation. The SU2 rotation is
    parameterized using the 3 euler angle using the x-z-x axis configuration.

    More precisely, the operator implemented is the following:

    .. math::

       U[\alpha, \beta, \gamma] = e^{-i\alpha X/2} e^{-i\beta Z/2}
            e^{-i\gamma X/2}.

    """
    def __init__(
            self, alpha: float,
            beta: float,
            gamma: float,
            gate_label: str = '') -> None:
        """Initialize the gate.

        :param alpha: euler angle of the first rotation around the x-axis.
        :type alpha: float
        :param beta: euler angle of the second rotation around the z-axis.
        :type beta: float
        :param gamma: euler angle of the third rotation around the x-axis.
        :type gamma: float
        :param gate_label: label used to identify the gate.
        :type gate_label: str
        """
        super(SU2Gate, self).__init__(gate_label=gate_label)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self._gate_graph = nx.path_graph(5)
        self._operation_dictionary = {
            0: {
                'gates': [Projector_B_phi, ],
                'parameters': [self._f_params_q0_t0, ],
                'previous_measurement_dependencies': [None, ]
            },
            1: {
                'gates': [Projector_B_phi, ],
                'parameters': [self._f_params_q1_t0, ],
                'previous_measurement_dependencies':
                    [(f'b_q0[{self.gate_label}]', ), ]
            },
            2: {
                'gates': [Projector_B_phi],
                'parameters': [self._f_params_q2_t0, ],
                'previous_measurement_dependencies':
                    [(f'b_q1[{self.gate_label}]', ), ]
            },
            3: {
                'gates': [Projector_B_phi],
                'parameters': [self._f_params_q3_t0, ],
                'previous_measurement_dependencies':
                    [(f'b_q2[{self.gate_label}]', ), ]
            },
            4: {
                'gates': [cirq.XPowGate, cirq.ZPowGate],
                'parameters': [self._f_params_q4_t0, self._f_params_q4_t1],
                'previous_measurement_dependencies':
                    [(f'b_q1[{self.gate_label}]', f'b_q3[{self.gate_label}]'),
                     (f'b_q0[{self.gate_label}]', f'b_q2[{self.gate_label}]')]
            }
        }

        self._setup()

    def _setup(self) -> None:
        self._initialize_graph_nodes()

    def _f_params_q0_t0(self) -> Dict[str, Any]:
        return dict(phi=0,
                    measurement_label=f'b_q0[{self.gate_label}]')

    def _f_params_q1_t0(self, b_q0: float) -> Dict[str, Any]:
        return dict(phi=-self.alpha * (-1) ** b_q0,
                    measurement_label=f'b_q1[{self.gate_label}]')

    def _f_params_q2_t0(self, b_q1: float) -> Dict[str, Any]:
        return dict(phi=-self.beta * (-1) ** b_q1,
                    measurement_label=f'b_q2[{self.gate_label}]')

    def _f_params_q3_t0(self, b_q2: float) -> Dict[str, Any]:
        return dict(phi=-self.gamma * (-1) ** b_q2,
                    measurement_label=f'b_q3[{self.gate_label}]')

    def _f_params_q4_t0(self, b_q1: float, b_q3: float) -> Dict[str, Any]:
        return dict(exponent=b_q1 + b_q3)

    def _f_params_q4_t1(self, b_q0: float, b_q2: float) -> Dict[str, Any]:
        return dict(exponent=b_q0 + b_q2)



class XPowGate(SU2Gate):
    """Power of a X pauli gate, i.e.

    .. math::

       X^\theta = e^{i\theta X/2},

    where :math:`X` is the X pauli matrix.

    .. note::

       The gate implement the power of X pauli gate up to a global phase factor.
       More precisely the gate here has a global phase equal to
       :math:`i^\theta`, i.e. it implements :math:`(iX)^\theta`. This is due to
       the fact this gate is obtained directly from the SU2 gate, without
       correcting for the phase.

    """
    def __init__(self, exponent: float = 1.) -> None:
        """Initialize the gate.

        :param exponent: exponent of the gate.
        :type exponent: float
        """
        super(XPowGate, self).__init__(alpha=0.,
                                       beta=0.,
                                       gamma=- np.pi * exponent)


class X(XPowGate):
    """X pauli gate.

    .. math::

       X = \begin{bmatrix}
       0 & 1 \\
       1 & 0
       \end{bmatrix}

    .. note::

       The gate implement the X pauli gate up to a global phase factor. More
       precisely the gate here has a global phase equal to :math:`i`, i.e.
       it implements :math:`iX`. This is due to the fact this gate is obtained
       directly from the SU2 gate, without correcting for the phase.

    """
    def __init__(self) -> None:
        super(X, self).__init__(exponent=1.0)


class YPowGate(SU2Gate):
    """Power of a Y pauli gate, i.e.

    .. math::

       Y^\theta = e^{i\theta Y/2},

    where :math:`Y` is the Y pauli matrix.

    .. note::

       The gate implement the power of Y pauli gate up to a global phase factor.
       More precisely the gate here has a global phase equal to
       :math:`i^\theta`, i.e. it implements :math:`(iY)^\theta`. This is due
       to the fact this gate is obtained directly from the SU2 gate, without
       correcting for the phase.

    """
    def __init__(self, exponent: float = 1.) -> None:
        """Initialize the gate.

        :param exponent: exponent of the gate.
        :type exponent: float
        """
        super(YPowGate, self).__init__(alpha=0.,
                                       beta=- np.pi * exponent,
                                       gamma=- np.pi * exponent)


class Y(YPowGate):
    """Y pauli gate.

    .. math::

       Y = \begin{bmatrix}
       0 & -i \\
       i & 0
       \end{bmatrix}

    .. note::

       The gate implement the Y pauli gate up to a global phase factor. More
       precisely the gate here has a global phase equal to :math:`i`, i.e.
       it implements :math:`iY`. This is due to the fact this gate is obtained
       directly from the SU2 gate, without correcting for the phase.

    """
    def __init__(self) -> None:
        super(Y, self).__init__(exponent=1.0)


class ZPowGate(SU2Gate):
    """Power of a Z pauli gate, i.e.

    .. math::

       Z^\theta = e^{i\theta Z/2},

    where :math:`Z` is the Z pauli matrix.

    .. note::

       The gate implement the power of Z pauli gate up to a global phase factor.
       More precisely the gate here has a global phase equal to
       :math:`i^\theta`, i.e. it implements :math:`(iZ)^\theta`. This is due to
       the fact this gate is obtained directly from the SU2 gate, without
       correcting for the phase.

    """
    def __init__(self, exponent: float = 1.) -> None:
        """Initialize the gate.

        :param exponent: exponent of the gate.
        :type exponent: float
        """
        super(ZPowGate, self).__init__(alpha=0.,
                                       beta=- np.pi * exponent,
                                       gamma=0.)


class Z(ZPowGate):
    """Z pauli gate.

    .. math::

       Z = \begin{bmatrix}
       1 & 0 \\
       0 & -1
       \end{bmatrix}

    .. note::

       The gate implement the Z pauli gate up to a global phase factor. More
       precisely the gate here has a global phase equal to :math:`i`, i.e.
       it implements :math:`iZ`. This is due to the fact this gate is obtained
       directly from the SU2 gate, without correcting for the phase.

    """
    def __init__(self) -> None:
        super(Z, self).__init__(exponent=1.0)
