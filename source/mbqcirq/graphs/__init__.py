


#################
#####   Libraries
#################
import abc

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx

from abc import ABC

from typing import Literal, Optional


########################
#####   Abstract Classes
########################


class GraphDescription:
    """Description of the basic properties of a graph."""
    is_grid: Optional[bool] = None
    """If the graph is a 2d grid or not."""
    dimension: Optional[int] = None
    """The dimension of the space in which the graph has a meaningful 
    interpretation."""


class Graph(ABC):
    """Abstract class for a generic graph."""

    def __init__(self):
        """Initialize graph."""
        self._description = GraphDescription()

    @abc.abstractmethod
    def _setup(self):
        """Setup operations."""
        self.graph = None
        self.n_nodes = 0

    @abc.abstractmethod
    def _describe_graph(self) -> None:
        """Add information to the _description attribute of this class."""
        pass

    @property
    def description(self) -> GraphDescription:
        """Get graph description."""
        return self._description

    @property
    def nodes(self):
        """Get graph nodes."""
        return list(self.graph.nodes)

    @property
    def edges(self):
        """Get graph edges."""
        return list(self.graph.edges)

    def get_neighborhood(self, node):
        """Get nearest neighborhood of a given node."""
        return list(nx.neighbors(self.graph, node))

    def _regular_layout(self) -> None:
        """Compute node positions for plotting in case of regular graphs."""
        pos = {}
        for v in self.graph.nodes:
            if isinstance(v, tuple):
                pos.update({v: v})

            elif isinstance(v, int):
                pos.update({v: (0.333333, v)})

        return pos

    def plot(
            self, layout: Literal['regular', 'planar', 'spring'] = 'regular'
    ) -> None:
        """Plot graph.

        :param layout: layout method used to define the node position in the
            plot.
        :type layout: Literal['regular', 'planar', 'spring']
        """
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, seed=42)

        elif layout == 'planar':
            pos = nx.planar_layout(self.graph)

        else:
            pos = self._regular_layout()

        plt.figure()
        nx.draw(self.graph, pos, node_color="tab:green", with_labels=True)
        plt.show()

    def __len__(self):
        return self.n_nodes
