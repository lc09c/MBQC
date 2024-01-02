"""Generic graph states."""


import cirq
import qsimcirq
import abc

from mbqcirq.graphs import Graph

from abc import ABC

from typing import Union, Tuple, List


###############
#####   Classes
###############


class AbstractGraphState(ABC):
    """abstract class for the construction of  a graph state from the
    underling graph."""
    def __init__(self) -> None:
        """Initialize graph state."""

        self._graph = None
        self._qubits = None
        self._initialization_circuit = None
        self._state_vector = None

        # self._setup()

    @abc.abstractmethod
    def _setup(self):
        """Setup operations."""
        pass

    @property
    def graph(self):
        """Get the graph of the state."""
        return self._graph

    def _initialize_qubits(self) -> None:
        """Initialize qubits."""
        if self._graph.description.dimension == 1:
            self._qubits = [cirq.LineQubit(v) for v in self._graph.nodes]

        elif self._graph.description.dimension == 2:
            self._qubits = [cirq.GridQubit(*v) for v in self._graph.nodes]

        else:
            raise NotImplemented

    @property
    def qubits(self):
        """List of cubits of the state."""
        return self._qubits

    def _produce_initialization_circuit(self) -> None:
        """Initialization circuit for a graph state."""
        self._initialization_circuit = cirq.Circuit()
        for qubit in self._qubits:
            self._initialization_circuit.append(cirq.H(qubit))

        for edge in self._graph.edges:
            n1, n2 = edge
            qubit1 = self._qubits[self._graph.nodes.index(n1)]
            qubit2 = self._qubits[self._graph.nodes.index(n2)]
            self._initialization_circuit.append(cirq.CZ(qubit1, qubit2))

    @property
    def initialization_circuit(self):
        """Circuit which can be used to produce the cluster state from the
        default initialization (:math:`|0\rangle^{\otimes n}`).
        """

        return self._initialization_circuit

    def _compute_state_vector(self) -> None:
        """Compute the state vector by simulating the initialization
        circuit.
        """
        sim = qsimcirq.QSimSimulator()
        res = sim.simulate(self._initialization_circuit)
        self._state_vector = res.final_state_vector

    @property
    def state_vector(self):
        """State vector of the cluster state."""
        return self._state_vector

    def get_qubit(self,  *args) -> Union[cirq.LineQubit, cirq.GridQubit]:
        """Get qubit in a specific node of the graph.

        :param args: Index(es) indicating the graph node of the desired qubit.
        :type args: Union[int, (int, int)]
        :return: The selected qubit.
        :rtype: Union[cirq.LineQubit, cirq.GridQubit]
        """
        if len(args) == 1:
            idx = self._graph.nodes.index(*args)

        if len(args) == 2:
            idx = self._graph.nodes.index(args)

        return self._qubits[idx]


class GraphState(AbstractGraphState):
    """Generic graph state constructed from a graph."""

    def __init__(self, graph: Graph) -> None:
        """Initialize a graph state.

        :param graph: Graph describing the graph state.
        :type graph: Graph
        """
        super(GraphState, self).__init__()
        self._graph = graph

        self._setup()

    def _setup(self):
        self._initialize_qubits()
        self._produce_initialization_circuit()
        self._compute_state_vector()
