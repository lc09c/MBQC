"""Cluster states used for measurement-based quantum computation."""


#################
#####   Libraries
#################


from mbqcirq.graphs.regular import Linear1dGraph, Grid2dGraph
from mbqcirq.states.graph_states import AbstractGraphState


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
