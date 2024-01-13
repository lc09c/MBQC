# title: '__init__.py'
# author: Curcuraci L.
# date: 13/01/2024
#
# scope: Collect all common methods and functions for all the measurement-based
# quantum gates.

"""Measurement-based quantum gates."""

#################
#####   Libraries
#################


import abc
import networkx as nx


###############
#####   Classes
###############


class MBQGate(abc.ABC):
    """Abstract class for measurement-based quantum gates."""

    def __init__(self, gate_label: str) -> None:
        """Initialize the measurement-based quantum gate

        :param gate_label: label associated to gate.
        :type gate_label: str
        """
        self.gate_label = gate_label
        self._gate_graph = None
        """Gate graph of the MBQ gate."""
        self._operation_dictionary = {}
        """Operation  dictionary describing the individual node operations."""

    def _setup(self) -> None:
        """Setup operations."""
        pass

    def _initialize_graph_nodes(self) -> None:
        """Initialize the graph nodes."""
        nx.set_node_attributes(self._gate_graph, self._operation_dictionary)

    @property
    def gate_graph(self) -> nx.Graph:
        """Get the gate graph"""
        return self._gate_graph

