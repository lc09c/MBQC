"""Projective measurements."""


#################
#####   Libraries
#################


import numpy as np
import cirq

from typing import Iterable, Union, List,  Any


###############
#####   Classes
###############


class Projector_X(cirq.Gate):
    """Projective measurement in the X basis."""

    def __init__(
            self,
            measurement_label: str = 'X',
    ) -> None:
        """Initialize projector.

        :param measurement_label: name used for the measurement result in
            after simulation.
        :type measurement_label: str
        """
        super(Projector_X, self)
        self.measurement_label = measurement_label

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(
            self, qubit: List[Union[cirq.NamedQubit, cirq.LineQubit,
                                    cirq.GridQubit]]
    ) -> Iterable[cirq.Gate]:

        Xq = cirq.X(qubit[0])
        yield cirq.measure_single_paulistring(Xq,
                                              key=f'{self.measurement_label}')

    def _circuit_diagram_info_(self, args: Any) -> str:
        return 'proj_Y'

    @property
    def basis(self):
        """State after measurement the qubit is left if 0 or 1 is measured."""

        return {0: (f'|+⟩', np.array([1, 1]) / np.sqrt(2)),
                1: (f'|-⟩', np.array([1, -1]) / np.sqrt(2))}


class Projector_Y(cirq.Gate):
    """Projective measurement in the Y basis."""

    def __init__(
            self,
            measurement_label: str = 'Y',
    ) -> None:
        """Initialize projector.

        :param measurement_label: name used for the measurement result in
            after simulation.
        :type measurement_label: str
        """
        super(Projector_X, self)
        self.measurement_label = measurement_label

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(
            self, qubit: List[Union[cirq.NamedQubit, cirq.LineQubit,
                                    cirq.GridQubit]]
    ) -> Iterable[cirq.Gate]:
        Yq = cirq.Y(qubit[0])
        yield cirq.measure_single_paulistring(Yq,
                                              key=f'{self.measurement_label}')

    def _circuit_diagram_info_(self, args: Any) -> str:
        return 'proj_Y'

    @property
    def basis(self):
        """State after measurement the qubit is left if 0 or 1 is measured."""

        return {0: (f'|+,y⟩', np.array([1, 1j]) / np.sqrt(2)),
                1: (f'|-,y⟩', np.array([1, -1j]) / np.sqrt(2))}


class Projector_Z(cirq.Gate):
    """Projective measurement in the computational basis (Z basis)."""

    def __init__(
            self,
            measurement_label: str = 'Z',
    ) -> None:
        """Initialize projector.

        :param measurement_label: name used for the measurement result in
            after simulation.
        :type measurement_label: str
        """
        super(Projector_Z, self)
        self.measurement_label = measurement_label

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(
            self, qubit: List[Union[cirq.NamedQubit, cirq.LineQubit,
                                    cirq.GridQubit]]
    ) -> Iterable[cirq.Gate]:
        yield cirq.measure(qubit[0],
                           key=f'{self.measurement_label}')

    def _circuit_diagram_info_(self, args: Any) -> str:
        return 'proj_Z'

    @property
    def basis(self):
        """State after measurement the qubit is left if 0 or 1 is measured."""

        return {0: (f'|0⟩', np.array([1, 0])),
                1: (f'|1⟩', np.array([0, 1]))}


class Projector_B_phi(cirq.Gate):
    """Projective measurement on the basis spanned by

    .. math::

        |+\rangle_\phi = \frac{|0\rangle + e^{i \ phi} |1\rangle}{2}
        |-\rangle_\phi = \frac{|0\rangle - e^{i \ phi} |1\rangle}{2}.

    .. note::

       The measurement is not performed on this basis directly, rather the
       qubit is measured in the computation basis after it is rotated using
       :math:`HP(-\phi/\pi)`, which is such that

       .. math::

          HP(-\phi/\pi) |+\rangle_\phi = | 0 \rangle
          HP(-\phi/\pi) |-\rangle_\phi = | 1 \rangle,

       where :math:`H` is the hadamard gate, and

       .. math::

          P(\eta) = \begin{bmatrix}
          1 & 0 \\
          0 & e^{i\eta}
          \end{bmatrix},

       is the phase shift operator. This means that given a state

       ..math::

         |\psi\rangle = \alpha |+\rangle_\phi + \beta|-\rangle_\phi,

       the rotation above map the state into

       ..math::

         |\psi\rangle = \alpha |0\rangle + \beta|1\rangle,

       In this way one can perform the projective measurement in the
       :math:`|+\rangle_\phi, |-\rangle_\phi` basis, using a projective
       measurement in the computational basis. Because of that, after a
       measurement, the state is left in :math:`|0\rangle` or :math:`|1\rangle`.
       To restore the state to the expected post measurement result, one
       needs to apply the inverse rotation, i.e. :math:`P(\phi / \pi)H`, to
       the collapsed state in the computational basis.

    """
    def __init__(
            self, phi: float,
            measurement_label: str = 'B_phi',
            collapse_qubit_state: bool = False
    ) -> None:
        """Initialize projector.

        :param phi: phase angle between the 0 an 1 kets of the computational
            basis.
        :type phi: float
        :param measurement_label: name used for the measurement result in
            after simulation.
        :type measurement_label: str
        :param collapse_qubit_state: if True, the state of the qubit after the
            measurement is restored to ordinary state associated to the
            measurement outcome.
        :type collapse_qubit_state: bool
        """
        super(Projector_B_phi, self)
        self.phi = phi
        self.measurement_label = measurement_label
        self.collapse_qubit_state = collapse_qubit_state

    def _num_qubits_(self) -> int:
        return 1

    def _decompose_(
            self, qubit: List[Union[cirq.NamedQubit, cirq.LineQubit,
                                    cirq.GridQubit]]
    ) -> Iterable[cirq.Gate]:
        yield cirq.ZPowGate(exponent=-self.phi/np.pi)(qubit[0])
        yield cirq.H(qubit[0])
        yield cirq.measure(qubit[0],
                           key=f'{self.measurement_label}')
        if self.collapse_qubit_state:
            yield cirq.H(qubit[0])
            yield cirq.ZPowGate(exponent=self.phi/np.pi)(qubit[0])

    def _circuit_diagram_info_(self, args: Any) -> str:
        return f'proj_B_phi({self.phi})'

    @property
    def basis(self):
        """State after measurement the qubit is left if 0 or 1 is measured."""

        return {0: (f'|+, φ={self.phi}⟩',
                    np.array([1, np.exp(1j * self.phi)]) / np.sqrt(2)),
                1: (f'|-, φ={self.phi}⟩',
                    np.array([1, - np.exp(1j * self.phi)]) / np.sqrt(2))}

