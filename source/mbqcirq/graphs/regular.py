

#################
#####   Libraries
#################


import networkx as nx
from ..graphs import Graph


#############
#####   Class
#############


class Linear1dGraph(Graph):
    """1-dimensional graph representing a linear chain of nodes."""

    def __init__(self, n_nodes: int) -> None:
        """Initialize graph.

        :param n_nodes: number of nodes in the graph.
        :type n_nodes: int
        """
        super(Linear1dGraph, self).__init__()

        self.n_nodes = n_nodes

        self._setup()

    def _setup(self):
        self.graph = nx.path_graph(n=self.n_nodes)
        self._describe_graph()

    def _describe_graph(self) -> None:
        self._description.is_grid = False
        self._description.dimension = 1


class Grid2dGraph(Graph):
    """2-dimensional graph representing a regular grid of nodes in a 2d
    space.
    """

    def __init__(self, n_rows: int, n_cols: int) -> None:
        """Initialize graph.

        :param n_rows: number of nodes used in a row.
        :type n_rows: int
        :param n_cols: number of nodes used in a column.
        :type n_cols: int
        """
        super(Grid2dGraph, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols

        self._setup()

    def _setup(self) -> None:
        self.graph = nx.grid_2d_graph(m=self.n_cols,
                                      n=self.n_rows)
        self.n_nodes = self.n_cols * self.n_rows

        self._describe_graph()

    def _describe_graph(self) -> None:
        self._description.is_grid = True
        self._description.dimension = 2
