import numpy as np
from typing import Set


class ExplorationState:
    def __init__(self, initial_pos):
        self.dist = 0
        self.current_node = FrontierNode(initial_pos, None)
        self.history = [self.current_node]
    
    def go_to(self, node, dist):
        self.dist += dist
        self.current_node = node
        self.history.append(node)
    
    def backtrack(self, history_idx):
        self.dist = history_idx
        self.history = self.history[:history_idx + 1]
        self.current_node = self.history[history_idx]
    
    def reset(self, current_node):
        self.dist = 0
        self.current_node = current_node
        self.history = [current_node]


class FrontierNode:
    nodes = set()
    
    def __init__(self, pos, parent):
        self.pos = np.array(pos)
        # stores children
        self.unexplored: Set[FrontierNode] = set()
        self.explored: Set[FrontierNode] = set()
        self.unreachable: Set[FrontierNode] = set()
        self.parent: FrontierNode = parent
        self.is_root = len(FrontierNode.nodes) == 0
        FrontierNode.nodes.add(self)
    
    def has_children(self):
        return len(self.unexplored) > 0 or len(self.explored) > 0 or len(self.unreachable) > 0
    
    def update(self, dist, d_max=1):
        self.unexplored = set.union(self.unexplored, self.unreachable)
        unreachable = set()
        for node in self.unexplored:
            if dist[node] > d_max:
                unreachable.add(node)
        self.unreachable = set.union(self.unreachable, unreachable)
        self.unexplored.difference_update(unreachable)
    
    def add_child(self, pos):
        child = FrontierNode(pos, self)
        self.unexplored.add(child)
        return child
    
    def get_closest_node(self, dest, dist):
        node = min(self.unexplored, key=lambda x: dist[tuple(x.pos), tuple(dest)])
        return node
    
    def get_reachable_nodes(self, dist, d_max=1):
        reachable = set()
        for node in self.unexplored:
            if dist[node] <= d_max:
                reachable.add(node)
        return reachable
    
    def get_nearest_child(self, dist):
        return min(self.unexplored, key=lambda x: dist[x])
    
    def is_alive(self, frontier_idxs):
        # checks if this frontier is still alive
        # note: we check if the frontier point is still in the current list of frontiers
        return tuple(self.pos) in frontier_idxs
    
    def prune_children(self, frontier_idxs):
        dead_children = set()
        for n in self.unexplored:
            if not n.is_alive(frontier_idxs):
                dead_children.add(n)
        self.unexplored.difference_update(dead_children)
        for n in self.explored:
            if not n.is_alive(frontier_idxs):
                dead_children.add(n)
        self.explored.difference_update(dead_children)
        for n in self.unreachable:
            if not n.is_alive(frontier_idxs):
                dead_children.add(n)
        self.unreachable.difference_update(dead_children)
        return dead_children
