import math
from collections import Counter
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
from pyannote.core import SlidingWindowFeature

from pyannote.audio.utils.permutation import mae_cost_func, permutate


class SegmentationConstraints(nx.Graph):
    """ """

    @classmethod
    def from_segmentation(
        cls,
        segmentations: SlidingWindowFeature,
        identifier: Any = None,
        cannot: bool = True,
        must: bool = True,
    ):

        identifier = identifier or id(segmentations)
        constraints_graph = cls()

        chunks = segmentations.sliding_window
        _, num_frames, _ = segmentations.data.shape
        max_lookahead = math.floor(chunks.duration / chunks.step - 1)

        # for each pair (c, C) of adjacent chunks
        for C, (chunk, segmentation) in enumerate(segmentations):
            for c in range(max(0, C - max_lookahead), C + 1):

                # cannot link active speakers from the same chunk
                if cannot and c == C:

                    # loop on pair of active speakers
                    for this, that in combinations(
                        np.where(np.any(segmentation, axis=0))[0], r=2
                    ):
                        # compute how much they overlap
                        both_active_frames = np.sum(
                            segmentation[:, this] * segmentation[:, that]
                        )

                        # compute how much they
                        cost = np.mean(
                            np.abs(segmentation[:, this] - segmentation[:, that])
                        )

                        # add constraint
                        constraints_graph.add_edge(
                            (chunk, this),
                            (chunk, that),
                            link="cannot",
                            origin=identifier,
                            both_active_frames=both_active_frames,
                            cost=cost,
                        )

                # speakers from adjacent chunks
                elif must:

                    # extract temporal support common to both chunks
                    shift = round((C - c) * num_frames * chunks.step / chunks.duration)

                    if shift < 0:
                        this_segmentations = segmentation[-shift:]
                        that_segmentations = segmentations[c, : num_frames + shift]
                    else:
                        this_segmentations = segmentation[: num_frames - shift]
                        that_segmentations = segmentations[c, shift:]

                    # find the optimal bijective mapping between their respective speakers
                    _, (permutation,), (cost_matrix,) = permutate(
                        this_segmentations[np.newaxis],
                        that_segmentations,
                        cost_func=mae_cost_func,
                        return_cost=True,
                    )

                    permutation_cost = sum(
                        cost_matrix[this, that] for this, that in enumerate(permutation)
                    )

                    # connect each pair of mapped speakers by an edge weighted by how much
                    # active frames they have in common
                    for this, that in enumerate(permutation):

                        both_active_frames = np.sum(
                            this_segmentations[:, this] * that_segmentations[:, that]
                        )
                        if both_active_frames == 0:
                            continue

                        constraints_graph.add_edge(
                            (chunk, this),
                            (chunks[c], that),
                            link="must",
                            origin=identifier,
                            both_active_frames=both_active_frames,
                            cost=cost_matrix[this, that],
                            permutation_cost=permutation_cost,
                        )

        return constraints_graph

    def remove_constraints(self, function, copy: bool = False):
        graph = self.copy() if copy else self
        graph.remove_edges_from(
            (n1, n2, edge_data)
            for (n1, n2, edge_data) in list(graph.edges(data=True))
            if function(n1, n2, edge_data)
        )
        return graph

    def augment(
        self, other: "SegmentationConstraints", copy: bool = False
    ) -> "SegmentationConstraints":
        """Add other constraints

        * only if the nodes exist in the original constraint graph
        * only if the edge does not exist already

        Parameters
        ----------
        other : SegmentationConstraints
            Other constraints
        copy : bool, optional

        Returns
        -------


        """

        graph = self.copy() if copy else self

        for node1, node2, data in other.edges(data=True):

            # do not add edge if nodes do not exist already
            if node1 not in graph or node2 not in graph:
                continue

            # do not add edge if it exists already
            if graph.has_edge(node1, node2):
                # TODO: warn in case of conflict
                continue

            graph.add_edge(node1, node2, **data)

        return graph

    def propagate(self, cannot: bool = False, must: bool = False, copy: bool = False):

        graph = self.copy() if copy else self

        cannot_link_subgraph: nx.Graph = self.edge_subgraph(
            (n1, n2) for (n1, n2, link) in self.edges(data="link") if link == "cannot"
        )

        must_link_subgraph: nx.Graph = self.edge_subgraph(
            (n1, n2) for (n1, n2, link) in self.edges(data="link") if link == "must"
        )

        # propagate "cannot link" constraints
        if cannot:

            # loop on (node1 ≠ node2) edges
            for node1, node2 in cannot_link_subgraph.edges():

                # node1 ≠ node2 and node1 = other_node implies node2 ≠ other_node
                # (unless there already exists an edge between node2 and other_node)
                if node1 in must_link_subgraph:
                    for other_node in must_link_subgraph.neighbors(node1):
                        if graph.has_edge(node2, other_node):
                            continue
                        graph.add_edge(
                            node2, other_node, link="cannot", origin="propagated"
                        )

                # same but switching node1/node2 roles
                if node2 in must_link_subgraph:
                    for other_node in must_link_subgraph.neighbors(node2):
                        if graph.has_edge(node1, other_node):
                            continue
                        graph.add_edge(
                            node1, other_node, link="cannot", origin="propagated"
                        )

        # propagate "must link" constraints
        if must:

            # loop on (node1 = node2) edges
            for node1, node2 in must_link_subgraph.edges():

                # node1 = node2 and node1 = other_node implies node2 = other_node
                # (unless there already exists an edge between node2 and other_node)
                for other_node in must_link_subgraph.neighbors(node1):
                    if graph.has_edge(node2, other_node):
                        continue
                    graph.add_edge(node2, other_node, link="must", origin="propagated")

                # same but switching node1/node2 roles
                for other_node in must_link_subgraph.neighbors(node2):
                    if graph.has_edge(node1, other_node):
                        continue
                    graph.add_edge(node1, other_node, link="must", origin="propagated")

        return graph

    def __str__(self):
        num_chunks = len(set(chunk for chunk, _ in self.nodes()))
        num_must = len([_ for _, _, link in self.edges(data="link") if link == "must"])
        num_cannot = len(
            [_ for _, _, link in self.edges(data="link") if link == "cannot"]
        )
        origin = Counter(origin for _, _, origin in self.edges(data="origin"))

        return f"{num_chunks} chunks / {num_must} must-link / {num_cannot} cannot-link / {origin}"
