"""Geometry-only drafts. A gesture is one undoable command."""
import numpy as np


class GeometryDraft:
    def __init__(self, points, edges=None):
        self.points = np.asarray(points, dtype=float).reshape(-1, 3).copy()
        self.edges = np.asarray(edges if edges is not None else [], dtype=int).reshape(-1, 2).copy()
        self.original_points, self.original_edges = self.points.copy(), self.edges.copy()
        self.nodes, self.edge_ids = set(), set()
        self.undo_stack, self.redo_stack = [], []
        self.gesture = None

    @property
    def changed(self):
        return not (np.array_equal(self.points, self.original_points) and np.array_equal(self.edges, self.original_edges))

    def state(self):
        return self.points.copy(), self.edges.copy(), set(self.nodes), set(self.edge_ids)

    def restore(self, state):
        self.points, self.edges = state[0].copy(), state[1].copy()
        self.nodes, self.edge_ids = set(state[2]), set(state[3])

    def record(self, before, title):
        if np.array_equal(before[0], self.points) and np.array_equal(before[1], self.edges):
            return False
        self.undo_stack.append((title, before, self.state()))
        self.undo_stack[:] = self.undo_stack[-100:]
        self.redo_stack.clear()
        return True

    def undo(self):
        if self.gesture is not None:
            self.cancel_gesture()
        if not self.undo_stack:
            return False
        command = self.undo_stack.pop()
        self.restore(command[1])
        self.redo_stack.append(command)
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        command = self.redo_stack.pop()
        self.restore(command[2])
        self.undo_stack.append(command)
        return True

    def clear_selection(self):
        self.nodes.clear()
        self.edge_ids.clear()

    def select(self, kind, index, additive=False):
        if not additive:
            self.clear_selection()
        selected = self.nodes if kind == 'node' else self.edge_ids
        if additive and index in selected:
            selected.remove(index)
        else:
            selected.add(int(index))

    def select_branch(self, edge_id, additive=False):
        if not 0 <= int(edge_id) < len(self.edges):
            return
        adjacent = [[] for _ in self.points]
        for index, (a, b) in enumerate(self.edges):
            adjacent[a].append(index)
            adjacent[b].append(index)
        found, queue = {int(edge_id)}, [int(edge_id)]
        while queue:
            for node in self.edges[queue.pop()]:
                if len(adjacent[node]) == 2:
                    for index in adjacent[node]:
                        if index not in found:
                            found.add(index)
                            queue.append(index)
        if not additive:
            self.clear_selection()
        self.edge_ids.update(found)

    def connect(self, start, end):
        a, b = int(start), int(end)
        if a == b or not (0 <= a < len(self.points) and 0 <= b < len(self.points)):
            return False
        if any(set(edge) == {a, b} for edge in self.edges):
            return False
        before = self.state()
        self.edges = np.vstack([self.edges, [a, b]])
        self.clear_selection()
        self.edge_ids.add(len(self.edges)-1)
        return self.record(before, 'Connect nodes')

    def delete_selection(self):
        if not self.nodes and not self.edge_ids:
            return False
        before = self.state()
        removed_nodes = set(self.nodes)
        keep_edges = [i for i, e in enumerate(self.edges)
                      if i not in self.edge_ids and not removed_nodes.intersection(e)]
        edges = self.edges[keep_edges]
        if len(self.edge_ids) > 1:
            # Removing a branch preserves attachment nodes still used elsewhere.
            touched = set(self.edges[sorted(self.edge_ids)].ravel())
            removed_nodes.update(touched-set(edges.ravel()))
        keep = [i for i in range(len(self.points)) if i not in removed_nodes]
        remap = np.full(len(self.points), -1, dtype=int)
        remap[keep] = np.arange(len(keep))
        self.points = self.points[keep]
        self.edges = remap[edges].reshape(-1, 2)
        self.clear_selection()
        return self.record(before, 'Delete selection')

    def begin_gesture(self):
        self.gesture = self.state()

    def move_selection(self, delta):
        if self.gesture is None or not self.nodes:
            return
        delta = np.asarray(delta, dtype=float).reshape(3)
        if np.all(np.isfinite(delta)):
            indices = sorted(self.nodes)
            self.points[indices] = self.gesture[0][indices]+delta

    def finish_gesture(self):
        before, self.gesture = self.gesture, None
        return before is not None and self.record(before, 'Move nodes')

    def cancel_gesture(self):
        if self.gesture is not None:
            self.restore(self.gesture)
            self.gesture = None

