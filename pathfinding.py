import networkx as nx
import numpy as np

heuristic_fn = lambda u, v: abs(u[0] - u[1]) + abs(v[0] - v[1])
class FretboardGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
    def add_edges(self, positions, transition_matrix):
        for i, (string1, fret1) in enumerate(positions):
            for j, (string2, fret2) in enumerate(positions):
                # Calculate edge weight
                w = -np.log(transition_matrix[i][j])
                self.graph.add_edge((string1, fret1), (string2, fret2), weight=w)

    def get_path(self, src, dst, heuristic=heuristic_fn, weight='weight'):




