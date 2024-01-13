# title: 'single_qubits.py'
# author: Curcuraci L.
# date: 13/01/2024
#
# scope: Collect all the single-qubit measurement-based quantum gates.

"""Single-qubit measurement-based quantum gates."""

#################
#####   Libraries
#################


import cirq
import networkx as nx

from mbqcirq.measurements.projectors import Projector_B_phi
from mbqcirq.gates import MBQGate

from typing import Dict, Any


###############
#####   Classes
###############


class SU2Gate(MBQGate):
    """MBQGate implementing an arbitrary SU2 rotation."""
    def __init__(
            self, alpha: float,
            beta: float,
            gamma: float,
            gate_label: str = '') -> None:
        """

        :param alpha:
        :param beta:
        :param gamma:
        :param gate_label:
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

