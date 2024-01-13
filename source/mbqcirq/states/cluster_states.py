"""Cluster states used for measurement-based quantum computation."""


#################
#####   Libraries
#################


import numpy as np
import random
import copy
import cirq
import qsimcirq

from mbqcirq.graphs.regular import Linear1dGraph, Grid2dGraph
from mbqcirq.states.graph_states import AbstractGraphState

from typing import Union, Dict, Tuple


###############
#####   Classes
###############


class ClusterState1d(AbstractGraphState):
    """1-dimensional cluster state on a 1d chain."""

    def __init__(self, n_qubits: int) -> None:
        """Initialize the 1-d cluster state.

        :param n_qubits: Number of qubits in the 1d chain.
        :type n_qubits: int
        """
        super(ClusterState1d, self).__init__()
        self.n_qubits = n_qubits

        self._setup()

    def _setup(self) -> None:
        self._graph = Linear1dGraph(n_nodes=self.n_qubits)
        self._initialize_qubits()
        self._produce_initialization_circuit()
        self._compute_state_vector()


class ClusterState2d(AbstractGraphState):
    """2-dimensional cluster state on a a 2d grid."""

    def __init__(self, n_qubits_rows: int, n_qubits_cols: int) -> None:
        """Initialize the 2-d cluster state.

        :param n_qubits_rows: number of qubits in a row of the 2d grid.
        :type n_qubits_rows: int
        :param n_qubits_cols: number of qubits in a column of the 2d grid.
        :type n_qubits_cols: int
        """
        super(ClusterState2d, self).__init__()
        self.n_qubits_rows = n_qubits_rows
        self.n_qubits_cols = n_qubits_cols

        self._setup()

    def _setup(self):
        self._graph = Grid2dGraph(n_rows=self.n_qubits_rows,
                                  n_cols=self.n_qubits_cols)
        self._initialize_qubits()
        self._produce_initialization_circuit()
        self._compute_state_vector()


class TestClusterState:
    """Test if a state is a cluster state by checking the eigenvalues of the
    correlation operations.

    Given the graph :math:`C` of a cluster state :math:`|\phi_C\rangle`, it is
    known that

    .. math::

       K_a |\phi_C\rangle = k |\phi_C\rangle

    where :math:`a` is a generic node of graph :math`C`:, :math:`k = \pm 1`,
    and :math:`K_a` is the correlation operator, defend as

    .. math::

       K_a = X_a \bigotimes_{b \in N(a)} Z_b

    with :math:`N(a)` is the set of neighborhood nodes of the node :math:`a`,
    and :math:`X_c, Z_c` are the x and z Pauli operators for the qubit
    associated to the node :math:`c`. This test computes the scalar
    product :math:`\langle \psi| K_a | \psi \rangle` for a generic
    :math:`|C|`-qubits state :math:`| \psi \rangle` for each node of the graph
    associated to the state. If :math:`| \psi \rangle` cluster state then

    .. math::

       |\langle \psi| K_a | \psi \rangle| = 1

    .. note::

       This test should work for graph states as well.

    """
    def __init__(
            self, state: AbstractGraphState,
            sample_nodes: bool = False,
            tested_nodes_size: float = 0.2,
            seed: int = 42
    ) -> None:
        """Initialize test.

        :param state: State under test.
        :type self: AbstractGraphState
        :param sample_nodes: If True only a randomly extracted section of the
            graph nodes are checked.
        :type sample_nodes: bool
        :param tested_nodes_size: Fraction of the total number of nodes in the
            graph which is checked is the test is performed by sampling the
            nodes.
         :type tested_nodes_size: float
         :param seed: Seed used for the randomization.
         :type state: int
        """
        self.state = state
        self.sample_nodes = sample_nodes
        self.tested_nodes_size = tested_nodes_size
        self.seed = seed

        self._setup()

    def _setup(self):
        """Setup operations."""
        self._nodes_to_test = self.state.graph.nodes
        if self.sample_nodes:
            random.seed(self.seed)
            n_nodes_to_test = max([1, int(len(self.state.graph) *
                                          self.tested_nodes_size)])
            _shuffled_nodes_to_test = copy.deepcopy(self._nodes_to_test)
            random.shuffle(_shuffled_nodes_to_test)
            self._nodes_to_test = _shuffled_nodes_to_test[:n_nodes_to_test]


    def _compute_scalar_product(self):
        """Compute the scalar product for the test in each node of the graph."""
        self._scalar_products = []
        simulator = qsimcirq.QSimSimulator()
        for node in self._nodes_to_test:
            neighborhood = self.state.graph.get_neighborhood(node)
            test_circuit = cirq.Circuit()
            qubit_a = self.state.get_qubit(node)
            test_circuit.append(cirq.X(qubit_a))
            for nn_node in neighborhood:
                qubit_b = self.state.get_qubit(nn_node)
                test_circuit.append(cirq.Z(qubit_b))

            for other_node in self.state.graph.nodes:
                qubit_c = self.state.get_qubit(other_node)
                test_circuit.append(cirq.I(qubit_c))

            result = simulator.simulate(test_circuit,
                                        initial_state=self.state.state_vector)
            k = cirq.dot(self.state.state_vector, result.final_state_vector)
            self._scalar_products.append(k)

    def run(self) -> bool:
        """Run the test.

        :return: If True the state is a cluster state.
        :rtype: bool
        """
        self._compute_scalar_product()
        test_result = True
        for k in self._scalar_products:
            test_result &= np.isclose(np.abs(k.real), 1.) and \
                           np.isclose(k.imag, 0.0)

        return test_result

    def get_scalar_products(
            self) -> Dict[Union[int, Tuple[int, int]], complex]:
        """Get the scalar products result.

        :return: Dictionary containing result of the scalar product for each
            correlation operator (e.g. node) tested.
        :rtype: Dict[Union[int, Tuple[int, int]], np.complex]
        """
        if hasattr(self, '_scalar_products'):
            return {n: k for n, k in zip(self._nodes_to_test,
                                         self._scalar_products)}

