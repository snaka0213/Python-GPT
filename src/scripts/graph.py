#!/usr/bin/env python3

class OrientedGraph(object):
    def __init__(self, nodes: list):
        self.nodes = nodes
        self.edges = {}
        for node in nodes:
            self.edges[node] = []

    def add_edge(self, v, w):
        # add edge: node v -> node w
        self.edges[v].append(w)

