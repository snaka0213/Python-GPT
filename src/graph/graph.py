#!/usr/bin/env python3

class OrientedGraph(object):
    def __init__(self, num_of_nodes: int):
        self._nodes = [i for i in range(num_of_nodes)]
        self._edges = [[] for i in range(num_of_nodes)]

    def add_edge(self, v, w):
        # add edge: node v -> node w
        self._edges[v].append(w)

