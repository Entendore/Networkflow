"""
Network Flow Algorithm Learner - Extended Edition
Algorithms: Edmonds-Karp, Ford-Fulkerson (DFS), Dinic's, Push-Relabel
Improved with: ABC, Template Method, Observer, Command, Factory, logging, __slots__, validation
"""

import sys
import math
import json
import random
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem, QGraphicsPathItem,
    QToolBar, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QComboBox, QSlider, QTextEdit, QSplitter, QMessageBox, QFileDialog,
    QGroupBox, QFormLayout, QScrollArea, QTabWidget, QFrame, QInputDialog,
    QUndoStack, QUndoCommand, QSizePolicy, QSpinBox
)
from PySide6.QtCore import Qt, QPointF, QTimer, QLineF, Signal, QEvent
from PySide6.QtGui import (
    QPen, QBrush, QColor, QFont, QPainter, QPainterPath, QAction, QPixmap,
    QKeySequence, QShortcut
)

# ═══════════════════════════════════════════════
#  CONSTANTS  (single source of truth — no magic numbers)
# ═══════════════════════════════════════════════

NODE_RADIUS = 26
ARROW_SIZE = 12
GRID_SPACING = 50
GRID_EXTENT = 2000
SCENE_RECT = (-2000, -2000, 4000, 4000)
PUSH_RELABEL_MAX_STEPS = 500
MAX_EDGE_CAPACITY = 9999
DEFAULT_CAPACITY = 10
DEFAULT_ANIM_SPEED_MS = 800
MIN_ANIM_SPEED_MS = 100
MAX_ANIM_SPEED_MS = 2000
DEFAULT_WINDOW_WIDTH = 1500
DEFAULT_WINDOW_HEIGHT = 900
MIN_WINDOW_WIDTH = 1300
MIN_WINDOW_HEIGHT = 800
RANDOM_MIN_NODES = 5
RANDOM_MAX_NODES = 12
RANDOM_MIN_CAPACITY = 1
RANDOM_MAX_CAPACITY = 30
RANDOM_EDGE_PROBABILITY = 0.35
RANDOM_LAYOUT_SPREAD_X = 500
RANDOM_LAYOUT_SPREAD_Y = 400


# ═══════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════

logger = logging.getLogger("networkflow")
logger.setLevel(logging.DEBUG)

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
logger.addHandler(_handler)


# ═══════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════

class Edge:
    """Represents a directed edge with capacity and flow."""
    __slots__ = ("u", "v", "capacity", "flow")

    def __init__(self, u: int, v: int, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Edge capacity must be positive, got {capacity}")
        self.u = u
        self.v = v
        self.capacity = capacity
        self.flow = 0

    @property
    def residual_capacity(self) -> int:
        """Residual capacity: how much more flow can be pushed forward."""
        return self.capacity - self.flow

    def __repr__(self) -> str:
        return f"Edge(u={self.u}, v={self.v}, cap={self.capacity}, flow={self.flow})"

    def __str__(self) -> str:
        return f"{self.u}→{self.v} [{self.flow}/{self.capacity}]"


class GraphObserver(ABC):
    """Observer interface — receive notifications when the graph mutates."""

    @abstractmethod
    def on_graph_changed(self, event: str = "changed") -> None:
        """Called after any mutation to the graph."""


class Graph:
    """
    Mutable directed graph for network flow.

    Implements the *Observable* pattern: call ``attach(observer)`` to
    register any object that implements :class:`GraphObserver`.
    """

    def __init__(self):
        self.nodes: dict[int, QPointF] = {}
        self.edges: list[Edge] = []
        self.adj: dict[int, list[int]] = defaultdict(list)
        self.source: Optional[int] = None
        self.sink: Optional[int] = None
        self._next_id: int = 0
        self._observers: list[GraphObserver] = []

    # ── Serialization ───────────────────────────

    def to_dict(self) -> dict:
        """Serialize graph to a JSON-compatible dict."""
        return {
            "nodes": {str(nid): [pos.x(), pos.y()] for nid, pos in self.nodes.items()},
            "edges": [{"u": e.u, "v": e.v, "capacity": e.capacity} for e in self.edges],
            "source": self.source,
            "sink": self.sink,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        """Deserialize graph from a JSON-compatible dict."""
        g = cls()
        for nid_str, pos_list in data.get("nodes", {}).items():
            nid = int(nid_str)
            pos = QPointF(pos_list[0], pos_list[1])
            g.nodes[nid] = pos
            g._next_id = max(g._next_id, nid + 1)
        for edge_data in data.get("edges", []):
            eidx = len(g.edges)
            g.edges.append(Edge(edge_data["u"], edge_data["v"], edge_data["capacity"]))
            g.adj[edge_data["u"]].append(eidx)
        g.source = data.get("source")
        g.sink = data.get("sink")
        return g

    # ── Random graph generation ──────────────────

    @classmethod
    def generate_random(cls, num_nodes: int = 8, seed: Optional[int] = None) -> "Graph":
        """
        Generate a random layered flow network.
        Nodes are arranged in layers; edges flow left-to-right.
        Source is node 0, sink is the last node.
        """
        rng = random.Random(seed)
        g = cls()
        num_nodes = max(4, min(num_nodes, 20))

        # Create nodes in a layered layout
        layers_count = max(2, num_nodes // 3)
        nodes_per_layer = max(1, num_nodes // layers_count)
        node_id = 0
        layer_starts: list[int] = []
        nid = 0
        for layer_idx in range(layers_count):
            layer_starts.append(nid)
            count = nodes_per_layer
            if layer_idx == layers_count - 1:
                count = num_nodes - nid
            for j in range(count):
                x = 80 + layer_idx * (RANDOM_LAYOUT_SPREAD_X / layers_count) + rng.uniform(-30, 30)
                y = 100 + j * (RANDOM_LAYOUT_SPREAD_Y / max(1, count - 1)) + rng.uniform(-20, 20)
                g.nodes[nid] = QPointF(x, y)
                nid += 1
        g._next_id = nid

        g.source = 0
        g.sink = nid - 1

        # Create edges: within-layer and between adjacent layers
        for layer_idx in range(layers_count - 1):
            start_a = layer_starts[layer_idx]
            if layer_idx + 1 < len(layer_starts):
                start_b = layer_starts[layer_idx + 1]
            else:
                continue
            # Determine end of layer A
            if layer_idx + 1 < len(layer_starts):
                end_a = layer_starts[layer_idx + 1]
            else:
                end_a = nid
            end_b = nid

            for u in range(start_a, end_a):
                for v in range(start_b, end_b):
                    if rng.random() < RANDOM_EDGE_PROBABILITY:
                        cap = rng.randint(RANDOM_MIN_CAPACITY, RANDOM_MAX_CAPACITY)
                        eidx = len(g.edges)
                        g.edges.append(Edge(u, v, cap))
                        g.adj[u].append(eidx)

            # Some within-layer edges (non-source, non-sink layers)
            if layer_idx > 0:
                for u in range(start_a, end_a):
                    for v in range(start_a, end_a):
                        if u != v and rng.random() < RANDOM_EDGE_PROBABILITY * 0.3:
                            cap = rng.randint(RANDOM_MIN_CAPACITY, RANDOM_MAX_CAPACITY)
                            eidx = len(g.edges)
                            g.edges.append(Edge(u, v, cap))
                            g.adj[u].append(eidx)

        # Ensure source and sink are connected
        if not g.adj.get(0):
            # Connect source to all nodes in next layer
            next_start = layer_starts[1] if len(layer_starts) > 1 else 1
            next_end = layer_starts[2] if len(layer_starts) > 2 else nid
            for v in range(next_start, min(next_end, nid)):
                cap = rng.randint(RANDOM_MIN_CAPACITY, RANDOM_MAX_CAPACITY)
                eidx = len(g.edges)
                g.edges.append(Edge(0, v, cap))
                g.adj[0].append(eidx)

        # Ensure sink is reachable
        sink = g.sink
        if not any(e.v == sink for e in g.edges):
            prev_start = layer_starts[-2] if len(layer_starts) >= 2 else 0
            prev_end = layer_starts[-1] if len(layer_starts) >= 2 else sink
            for u in range(prev_start, min(prev_end, sink)):
                cap = rng.randint(RANDOM_MIN_CAPACITY, RANDOM_MAX_CAPACITY)
                eidx = len(g.edges)
                g.edges.append(Edge(u, sink, cap))
                g.adj[u].append(eidx)

        return g

    # ── Observer management ──────────────────────

    def attach(self, observer: GraphObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug("Observer attached: %s", observer)

    def detach(self, observer: GraphObserver) -> None:
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify(self, event: str = "changed") -> None:
        for obs in self._observers:
            try:
                obs.on_graph_changed(event)
            except Exception:
                logger.exception("Observer %s failed on_graph_changed", obs)

    # ── Node operations ─────────────────────────

    def add_node(self, pos: QPointF) -> int:
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = pos
        logger.debug("Node %d added at (%.0f, %.0f)", nid, pos.x(), pos.y())
        self._notify("node_added")
        return nid

    def remove_node(self, nid: int) -> None:
        if nid not in self.nodes:
            logger.warning("remove_node: node %d does not exist", nid)
            return
        self.edges = [e for e in self.edges if e.u != nid and e.v != nid]
        self._rebuild_adj()
        del self.nodes[nid]
        if self.source == nid:
            self.source = None
        if self.sink == nid:
            self.sink = None
        logger.debug("Node %d removed", nid)
        self._notify("node_removed")

    # ── Edge operations ──────────────────────────

    def add_edge(self, u: int, v: int, capacity: int) -> int:
        if u not in self.nodes:
            raise ValueError(f"Node {u} does not exist")
        if v not in self.nodes:
            raise ValueError(f"Node {v} does not exist")
        if capacity <= 0 or capacity > MAX_EDGE_CAPACITY:
            raise ValueError(f"Capacity must be in (0, {MAX_EDGE_CAPACITY}], got {capacity}")
        for e in self.edges:
            if e.u == u and e.v == v:
                raise ValueError(f"Edge {u}→{v} already exists")

        idx = len(self.edges)
        self.edges.append(Edge(u, v, capacity))
        self.adj[u].append(idx)
        logger.debug("Edge %d→%d added (cap=%d)", u, v, capacity)
        self._notify("edge_added")
        return idx

    def _rebuild_adj(self) -> None:
        self.adj = defaultdict(list)
        for i, e in enumerate(self.edges):
            self.adj[e.u].append(i)

    # ── Flow operations ──────────────────────────

    def reset_flows(self) -> None:
        for e in self.edges:
            e.flow = 0
        logger.debug("All flows reset to 0")

    def clear(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.adj.clear()
        self.source = None
        self.sink = None
        self._next_id = 0
        logger.debug("Graph cleared")
        self._notify("cleared")

    def __repr__(self) -> str:
        return (f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)}, "
                f"source={self.source}, sink={self.sink})")

    def __len__(self) -> int:
        return len(self.nodes)


# ═══════════════════════════════════════════════
#  ALGORITHM STEP  (Factory helpers for clean construction)
# ═══════════════════════════════════════════════

class AlgoStep:
    """
    Immutable-ish snapshot of one algorithm step.
    Use the ``_make_*`` factory methods for construction.
    """
    __slots__ = (
        "step_type", "path", "path_edges", "bottleneck", "total_flow",
        "flows_snapshot", "heights", "excess", "push_from", "push_to",
        "push_amount", "relabel_node", "old_height", "new_height",
        "level", "description",
    )

    def __init__(self):
        self.step_type: str = ""
        self.path: list[int] = []
        self.path_edges: list[int] = []
        self.bottleneck: int = 0
        self.total_flow: int = 0
        self.flows_snapshot: dict[int, int] = {}
        self.heights: dict[int, int] = {}
        self.excess: dict[int, int] = {}
        self.push_from: Optional[int] = None
        self.push_to: Optional[int] = None
        self.push_amount: int = 0
        self.relabel_node: Optional[int] = None
        self.old_height: int = 0
        self.new_height: int = 0
        self.level: dict[int, int] = {}
        self.description: str = ""

    # ── Factory helpers ──────────────────────────

    @staticmethod
    def _snapshot_flows(edges: list[Edge]) -> dict[int, int]:
        return {i: e.flow for i, e in enumerate(edges)}

    @staticmethod
    def make_done(total_flow: int, edges: list[Edge], *,
                  heights: dict[int, int] | None = None,
                  excess: dict[int, int] | None = None) -> "AlgoStep":
        """Create a 'done' terminal step."""
        s = AlgoStep()
        s.step_type = "done"
        s.total_flow = total_flow
        s.flows_snapshot = AlgoStep._snapshot_flows(edges)
        s.description = f"Algorithm complete.\nMaximum flow = {total_flow}"
        if heights:
            s.heights = dict(heights)
        if excess:
            s.excess = dict(excess)
        return s

    @staticmethod
    def make_augment(iteration: int, search_type: str,
                     path: list[int], path_edges: list[int],
                     bottleneck: int, total_flow: int,
                     edges: list[Edge]) -> "AlgoStep":
        """Create an 'augment' step (for BFS / DFS based algos)."""
        s = AlgoStep()
        s.step_type = "augment"
        s.path = list(path)
        s.path_edges = list(path_edges)
        s.bottleneck = bottleneck
        s.total_flow = total_flow
        s.flows_snapshot = AlgoStep._snapshot_flows(edges)
        path_str = " -> ".join(str(n) for n in path)
        s.description = (
            f"Iteration {iteration} ({search_type}):\n"
            f"  Path: {path_str}\n"
            f"  Bottleneck: {bottleneck}\n"
            f"  Total flow: {total_flow}"
        )
        return s

    @staticmethod
    def make_level_graph(iteration: int, level: dict[int, int],
                         total_flow: int, edges: list[Edge],
                         sink: int) -> "AlgoStep":
        """Create a 'level_graph' step (Dinic's Phase 1)."""
        s = AlgoStep()
        s.step_type = "level_graph"
        s.level = dict(level)
        s.total_flow = total_flow
        s.flows_snapshot = AlgoStep._snapshot_flows(edges)
        level_str = ", ".join(f"{n}:{l}" for n, l in sorted(level.items()))
        s.description = (
            f"Iteration {iteration} - Phase 1 (BFS):\n"
            f"  Level graph: {level_str}\n"
            f"  Distance S->T: {level.get(sink, '?')}\n"
            f"  Now finding blocking flow via DFS..."
        )
        return s

    @staticmethod
    def make_blocking_flow(iteration: int, pushed: int,
                           iteration_flow: int, total_flow: int,
                           edges: list[Edge]) -> "AlgoStep":
        """Create an 'augment' step for Dinic's blocking flow (Phase 2)."""
        s = AlgoStep()
        s.step_type = "augment"
        s.bottleneck = pushed
        s.total_flow = total_flow
        s.flows_snapshot = AlgoStep._snapshot_flows(edges)
        s.description = (
            f"Iteration {iteration} - Phase 2 (DFS):\n"
            f"  Found augmenting path with flow = {pushed}\n"
            f"  Iteration flow so far: {iteration_flow}\n"
            f"  Total flow: {total_flow}"
        )
        return s

    @staticmethod
    def make_push(push_amt: int, u: int, v: int,
                  heights: dict[int, int], excess: dict[int, int],
                  total_flow: int, edges: list[Edge]) -> "AlgoStep":
        """Create a 'push' step (Push-Relabel)."""
        s = AlgoStep()
        s.step_type = "push"
        s.push_from = u
        s.push_to = v
        s.push_amount = push_amt
        s.total_flow = total_flow
        s.heights = dict(heights)
        s.excess = dict(excess)
        s.flows_snapshot = AlgoStep._snapshot_flows(edges)
        s.description = (
            f"PUSH {push_amt} from {u} -> {v}\n"
            f"  Condition: h({u})={heights[u]} = h({v})+1={heights[v]+1}\n"
            f"  Residual capacity before push: {push_amt + (edges[0].residual_capacity if edges else '?')}"
        )
        return s

    @staticmethod
    def make_relabel(node: int, old_h: int, new_h: int,
                     heights: dict[int, int], excess: dict[int, int],
                     total_flow: int, edges: list[Edge]) -> "AlgoStep":
        """Create a 'relabel' step (Push-Relabel)."""
        s = AlgoStep()
        s.step_type = "relabel"
        s.relabel_node = node
        s.old_height = old_h
        s.new_height = new_h
        s.total_flow = total_flow
        s.heights = dict(heights)
        s.excess = dict(excess)
        s.flows_snapshot = AlgoStep._snapshot_flows(edges)
        s.description = (
            f"RELABEL node {node}\n"
            f"  Old height: {old_h}\n"
            f"  New height: {new_h}\n"
            f"  (min neighbor height + 1)"
        )
        return s

    def __repr__(self) -> str:
        return (f"AlgoStep(type={self.step_type!r}, flow={self.total_flow}, "
                f"desc={self.description[:50]!r}...)")


# ═══════════════════════════════════════════════
#  BASE ALGORITHM  (ABC + Template Method helpers)
# ═══════════════════════════════════════════════

class FlowAlgorithm(ABC):
    """
    Abstract base for all max-flow algorithms.
    Provides shared path-finding and augmentation utilities
    (Template Method pattern: subclasses call these helpers).
    """
    name: str = "Base"
    short_desc: str = ""

    def __init__(self, graph: Graph):
        self.graph = graph
        self.steps: list[AlgoStep] = []
        self.current_step: int = -1
        self.max_flow: int = 0

    @abstractmethod
    def run_all(self) -> None:
        """Execute the algorithm end-to-end, populating ``self.steps``."""

    # ── Navigation ───────────────────────────────

    def go_to_step(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.steps):
            return
        self.current_step = idx
        snap = self.steps[idx].flows_snapshot
        for eidx, flow_val in snap.items():
            if eidx < len(self.graph.edges):
                self.graph.edges[eidx].flow = flow_val
        self.max_flow = self.steps[idx].total_flow

    def step_forward(self) -> bool:
        if self.current_step + 1 < len(self.steps):
            self.go_to_step(self.current_step + 1)
            return True
        return False

    def step_backward(self) -> bool:
        if self.current_step > 0:
            self.go_to_step(self.current_step - 1)
            return True
        return False

    # ── Shared helpers (Template Method) ─────────

    def _reset_state(self) -> None:
        """Common reset before running any algorithm."""
        self.graph.reset_flows()
        self.steps.clear()
        self.current_step = -1
        self.max_flow = 0

    def _has_source_sink(self) -> bool:
        if self.graph.source is None or self.graph.sink is None:
            logger.warning("Algorithm '%s' skipped: source/sink not set", self.name)
            return False
        return True

    def _find_path_bfs(self) -> Optional[tuple[list[int], list[int], int]]:
        """
        BFS to find an augmenting path in the residual graph.
        Returns (path_nodes, path_edge_indices, bottleneck) or None.
        """
        source, sink = self.graph.source, self.graph.sink
        parent: dict[int, int | None] = {source: None}
        parent_edge: dict[int, int] = {source: -1}
        visited: set[int] = {source}
        queue = deque([source])

        while queue:
            u = queue.popleft()
            if u == sink:
                return self._trace_path(parent, parent_edge, source, sink)
            # Forward edges
            for eidx in self.graph.adj[u]:
                e = self.graph.edges[eidx]
                if e.v not in visited and e.residual_capacity > 0:
                    visited.add(e.v)
                    parent[e.v] = u
                    parent_edge[e.v] = eidx
                    queue.append(e.v)
            # Backward edges (undo flow)
            for eidx, e in enumerate(self.graph.edges):
                if e.v == u and e.u not in visited and e.flow > 0:
                    visited.add(e.u)
                    parent[e.u] = u
                    parent_edge[e.u] = -(eidx + 1)
                    queue.append(e.u)
        return None

    def _find_path_dfs(self) -> Optional[tuple[list[int], list[int], int]]:
        """
        DFS to find an augmenting path in the residual graph.
        Returns (path_nodes, path_edge_indices, bottleneck) or None.
        """
        source, sink = self.graph.source, self.graph.sink
        parent: dict[int, int | None] = {source: None}
        parent_edge: dict[int, int] = {source: -1}
        visited: set[int] = {source}
        stack = [source]

        while stack:
            u = stack.pop()
            if u == sink:
                return self._trace_path(parent, parent_edge, source, sink)
            for eidx in self.graph.adj[u]:
                e = self.graph.edges[eidx]
                if e.v not in visited and e.residual_capacity > 0:
                    visited.add(e.v)
                    parent[e.v] = u
                    parent_edge[e.v] = eidx
                    stack.append(e.v)
            for eidx, e in enumerate(self.graph.edges):
                if e.v == u and e.u not in visited and e.flow > 0:
                    visited.add(e.u)
                    parent[e.u] = u
                    parent_edge[e.u] = -(eidx + 1)
                    stack.append(e.u)
        return None

    def _trace_path(self, parent: dict, parent_edge: dict,
                    source: int, sink: int) -> tuple[list[int], list[int], int]:
        """Trace back from sink to source, compute bottleneck and augment."""
        path: list[int] = []
        path_edges: list[int] = []
        node = sink
        bottleneck = math.inf

        while node != source:
            path.append(node)
            eidx = parent_edge[node]
            path_edges.append(eidx)
            if eidx >= 0:
                bottleneck = min(bottleneck, self.graph.edges[eidx].residual_capacity)
            else:
                bottleneck = min(bottleneck, self.graph.edges[-(eidx + 1)].flow)
            node = parent[node]
        path.append(source)
        path.reverse()
        path_edges.reverse()

        # Apply augmentation
        for eidx in path_edges:
            if eidx >= 0:
                self.graph.edges[eidx].flow += bottleneck
            else:
                self.graph.edges[-(eidx + 1)].flow -= bottleneck

        self.max_flow += bottleneck
        return path, path_edges, bottleneck

    def _append_done(self) -> None:
        """Append the terminal 'done' step."""
        self.steps.append(AlgoStep.make_done(self.max_flow, self.graph.edges))

    def compute_min_cut(self) -> set[int]:
        """
        Compute the minimum S-side cut after a run.
        Returns the set of nodes reachable from source in the residual graph.
        """
        source = self.graph.source
        if source is None:
            return set()
        visited: set[int] = set()
        queue = deque([source])
        visited.add(source)
        while queue:
            u = queue.popleft()
            for eidx in self.graph.adj[u]:
                e = self.graph.edges[eidx]
                if e.v not in visited and e.residual_capacity > 0:
                    visited.add(e.v)
                    queue.append(e.v)
            for eidx, e in enumerate(self.graph.edges):
                if e.v == u and e.u not in visited and e.flow > 0:
                    visited.add(e.u)
                    queue.append(e.u)
        return visited

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(steps={len(self.steps)}, "
                f"max_flow={self.max_flow})")


# ═══════════════════════════════════════════════
#  1. EDMONDS-KARP (BFS) — uses Template Method
# ═══════════════════════════════════════════════

class EdmondsKarpEngine(FlowAlgorithm):
    name = "Edmonds-Karp (BFS)"
    short_desc = "Uses BFS to find shortest augmenting path. O(VE2)."

    def run_all(self) -> None:
        self._reset_state()
        if not self._has_source_sink():
            return

        iteration = 0
        while True:
            iteration += 1
            result = self._find_path_bfs()
            if result is None:
                break
            path, path_edges, bottleneck = result
            self.steps.append(
                AlgoStep.make_augment(
                    iteration, "BFS", path, path_edges,
                    bottleneck, self.max_flow, self.graph.edges
                )
            )

        self._append_done()
        logger.info("Edmonds-Karp done: max_flow=%d, steps=%d", self.max_flow, len(self.steps))


# ═══════════════════════════════════════════════
#  2. FORD-FULKERSON (DFS) — uses Template Method
# ═══════════════════════════════════════════════

class FordFulkersonDFSEngine(FlowAlgorithm):
    name = "Ford-Fulkerson (DFS)"
    short_desc = "Uses DFS to find augmenting path. O(E*max_flow) for integers."

    def run_all(self) -> None:
        self._reset_state()
        if not self._has_source_sink():
            return

        iteration = 0
        while True:
            iteration += 1
            result = self._find_path_dfs()
            if result is None:
                break
            path, path_edges, bottleneck = result
            self.steps.append(
                AlgoStep.make_augment(
                    iteration, "DFS", path, path_edges,
                    bottleneck, self.max_flow, self.graph.edges
                )
            )

        self._append_done()
        logger.info("Ford-Fulkerson DFS done: max_flow=%d, steps=%d", self.max_flow, len(self.steps))


# ═══════════════════════════════════════════════
#  3. DINIC'S ALGORITHM
# ═══════════════════════════════════════════════

class DinicEngine(FlowAlgorithm):
    name = "Dinic's Algorithm"
    short_desc = "Level graph + blocking flow. O(V2E)."

    def run_all(self) -> None:
        self._reset_state()
        if not self._has_source_sink():
            return

        source, sink = self.graph.source, self.graph.sink
        iteration = 0

        while True:
            iteration += 1
            # Phase 1: BFS to build level graph
            level: dict[int, int] = {source: 0}
            queue = deque([source])
            while queue:
                u = queue.popleft()
                for eidx in self.graph.adj[u]:
                    e = self.graph.edges[eidx]
                    if e.v not in level and e.residual_capacity > 0:
                        level[e.v] = level[u] + 1
                        queue.append(e.v)

            if sink not in level:
                break

            # Record level graph step
            self.steps.append(
                AlgoStep.make_level_graph(
                    iteration, level, self.max_flow, self.graph.edges, sink
                )
            )

            # Phase 2: Blocking flow via DFS with current-arc optimization
            iter_ptr: dict[int, int] = {u: 0 for u in self.graph.nodes}
            iteration_flow = 0

            def _dfs(u: int, pushed: int) -> int:
                if u == sink:
                    return pushed
                while iter_ptr[u] < len(self.graph.adj[u]):
                    eidx = self.graph.adj[u][iter_ptr[u]]
                    e = self.graph.edges[eidx]
                    if level.get(e.v, -1) == level[u] + 1 and e.residual_capacity > 0:
                        tr = _dfs(e.v, min(pushed, e.residual_capacity))
                        if tr > 0:
                            e.flow += tr
                            # Update reverse edge
                            for ri, re in enumerate(self.graph.edges):
                                if re.u == e.v and re.v == u:
                                    re.flow -= tr
                                    break
                            return tr
                    iter_ptr[u] += 1
                return 0

            while True:
                pushed = _dfs(source, math.inf)
                if pushed == 0:
                    break
                iteration_flow += pushed
                self.max_flow += pushed
                self.steps.append(
                    AlgoStep.make_blocking_flow(
                        iteration, pushed, iteration_flow,
                        self.max_flow, self.graph.edges
                    )
                )

        self._append_done()
        logger.info("Dinic's done: max_flow=%d, steps=%d", self.max_flow, len(self.steps))


# ═══════════════════════════════════════════════
#  4. PUSH-RELABEL (PREFLOW-PUSH)
# ═══════════════════════════════════════════════

class PushRelabelEngine(FlowAlgorithm):
    name = "Push-Relabel"
    short_desc = "Maintains preflow with heights. O(V2E) or O(V3)."

    def run_all(self) -> None:
        self._reset_state()
        if not self._has_source_sink():
            return

        source, sink = self.graph.source, self.graph.sink
        nodes = list(self.graph.nodes.keys())

        # Initialize heights and excess
        height: dict[int, int] = {u: 0 for u in nodes}
        height[source] = len(nodes)
        excess: dict[int, int] = {u: 0 for u in nodes}

        # Initial preflow: saturate all edges from source
        for eidx in self.graph.adj[source]:
            e = self.graph.edges[eidx]
            if e.capacity > 0:
                e.flow = e.capacity
                excess[e.v] += e.capacity
                excess[source] -= e.capacity

        # Record initial step
        self.steps.append(
            AlgoStep.make_push(
                0, source, -1, height, excess,
                0, self.graph.edges
            )
        )
        # Override description for the initialization step
        self.steps[-1].description = (
            f"Initialization:\n"
            f"  h(S) = {len(nodes)}, h(v) = 0 for all other v\n"
            f"  Saturate all edges from source.\n"
            f"  Excess at each neighbor of S is now positive."
        )

        # Active nodes (excess > 0, not source or sink)
        active = deque(u for u in nodes if u != source and u != sink and excess[u] > 0)
        step_count = 0

        while active:
            u = active.popleft()
            if excess[u] <= 0:
                continue

            step_count += 1
            if step_count > PUSH_RELABEL_MAX_STEPS:
                logger.warning("Push-Relabel exceeded %d steps — stopping early", PUSH_RELABEL_MAX_STEPS)
                break

            # Try to push
            pushed_any = False
            for eidx in self.graph.adj[u]:
                e = self.graph.edges[eidx]
                if excess[u] <= 0:
                    break
                if e.residual_capacity > 0 and height[u] == height[e.v] + 1:
                    push_amt = min(excess[u], e.residual_capacity)
                    e.flow += push_amt
                    excess[u] -= push_amt
                    excess[e.v] += push_amt
                    pushed_any = True

                    self.steps.append(
                        AlgoStep.make_push(
                            push_amt, u, e.v, height, excess,
                            self.max_flow, self.graph.edges
                        )
                    )

                    if e.v != source and e.v != sink and excess[e.v] > 0:
                        if e.v not in active:
                            active.append(e.v)

            # If couldn't push, relabel
            if not pushed_any and excess[u] > 0:
                old_h = height[u]
                min_h = math.inf
                for eidx in self.graph.adj[u]:
                    e = self.graph.edges[eidx]
                    if e.residual_capacity > 0:
                        min_h = min(min_h, height[e.v])
                if min_h < math.inf:
                    height[u] = min_h + 1

                    self.steps.append(
                        AlgoStep.make_relabel(
                            u, old_h, height[u], height, excess,
                            self.max_flow, self.graph.edges
                        )
                    )

                if excess[u] > 0:
                    active.append(u)

        # Compute final max flow = excess at sink
        self.max_flow = excess.get(sink, 0)

        self._append_done()
        logger.info("Push-Relabel done: max_flow=%d, steps=%d", self.max_flow, len(self.steps))


# ═══════════════════════════════════════════════
#  ALGORITHM REGISTRY
# ═══════════════════════════════════════════════

ALGORITHMS: dict[str, type[FlowAlgorithm]] = {
    "Edmonds-Karp (BFS)": EdmondsKarpEngine,
    "Ford-Fulkerson (DFS)": FordFulkersonDFSEngine,
    "Dinic's Algorithm": DinicEngine,
    "Push-Relabel": PushRelabelEngine,
}


# ═══════════════════════════════════════════════
#  GRAPHICS ITEMS
# ═══════════════════════════════════════════════

class NodeItem(QGraphicsEllipseItem):
    COLORS: dict[str, QColor] = {
        "default": QColor(70, 130, 180),
        "source": QColor(46, 204, 113),
        "sink": QColor(231, 76, 60),
        "hover": QColor(100, 160, 220),
        "path": QColor(255, 165, 0),
        "push_from": QColor(255, 100, 100),
        "push_to": QColor(100, 200, 255),
        "relabel": QColor(200, 100, 255),
    }

    def __init__(self, node_id: int, pos: QPointF, parent=None):
        super().__init__(-NODE_RADIUS, -NODE_RADIUS, NODE_RADIUS * 2, NODE_RADIUS * 2, parent)
        self.node_id = node_id
        self.setPos(pos)
        self.setBrush(QBrush(self.COLORS["default"]))
        self.setPen(QPen(QColor(40, 40, 40), 2))
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.setAcceptHoverEvents(True)

        self.label = QGraphicsTextItem(str(node_id), self)
        self.label.setDefaultTextColor(Qt.white)
        self.label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self._center_label()

        # Height badge (for push-relabel)
        self.height_label = QGraphicsTextItem("", self)
        self.height_label.setDefaultTextColor(QColor(180, 50, 50))
        self.height_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.height_label.setPos(-10, -NODE_RADIUS - 18)
        self.height_label.setVisible(False)

        # Excess badge
        self.excess_label = QGraphicsTextItem("", self)
        self.excess_label.setDefaultTextColor(QColor(200, 50, 50))
        self.excess_label.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.excess_label.setPos(NODE_RADIUS - 8, -NODE_RADIUS - 4)
        self.excess_label.setVisible(False)

        self.role: str = "default"
        self.edges: list["EdgeItem"] = []

    def _center_label(self) -> None:
        br = self.label.boundingRect()
        self.label.setPos(-br.width() / 2, -br.height() / 2)

    def set_role(self, role: str) -> None:
        self.role = role
        color = self.COLORS.get(role, self.COLORS["default"])
        self.setBrush(QBrush(color))
        if role == "source":
            self.label.setPlainText("S")
        elif role == "sink":
            self.label.setPlainText("T")
        else:
            self.label.setPlainText(str(self.node_id))
        self._center_label()

    def set_highlight(self, highlight_type: str) -> None:
        if highlight_type:
            self.setBrush(QBrush(self.COLORS.get(highlight_type, self.COLORS["path"])))
            self.setPen(QPen(QColor(200, 100, 0), 3))
        else:
            self.set_role(self.role)
            self.setPen(QPen(QColor(40, 40, 40), 2))

    def set_height(self, h: int) -> None:
        self.height_label.setPlainText(f"h={h}")
        self.height_label.setVisible(True)

    def set_excess(self, ex: int) -> None:
        if ex > 0:
            self.excess_label.setPlainText(f"+{ex}")
            self.excess_label.setVisible(True)
        else:
            self.excess_label.setVisible(False)

    def clear_annotations(self) -> None:
        self.height_label.setVisible(False)
        self.excess_label.setVisible(False)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            for edge in self.edges:
                edge.update_position()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        if self.role == "default":
            self.setBrush(QBrush(self.COLORS["hover"]))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self.role == "default":
            self.setBrush(QBrush(self.COLORS["default"]))
        super().hoverLeaveEvent(event)

    def __repr__(self) -> str:
        return f"NodeItem(id={self.node_id}, role={self.role!r})"


class EdgeItem(QGraphicsObject):
    def __init__(self, edge_idx: int, source_item: NodeItem, target_item: NodeItem,
                 capacity: int, parent=None):
        super().__init__(parent)
        self.edge_idx = edge_idx
        self.source_item = source_item
        self.target_item = target_item
        self.capacity = capacity
        self.flow = 0
        self.setZValue(1)
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setAcceptHoverEvents(True)

        self._line_path = QPainterPath()
        self._arrow_path = QPainterPath()
        self._is_path_edge = False

        source_item.edges.append(self)
        target_item.edges.append(self)
        self.update_position()

    def hoverEnterEvent(self, event) -> None:
        if not self._is_path_edge and not self.isSelected():
            self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if not self._is_path_edge and not self.isSelected():
            self.update()
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        """Double-click to edit edge capacity."""
        new_cap, ok = QInputDialog.getInt(
            None, "Edit Edge Capacity",
            f"Edge {self.source_item.node_id} -> {self.target_item.node_id}\n"
            f"Current capacity: {self.capacity}",
            self.capacity, 1, MAX_EDGE_CAPACITY
        )
        if ok and new_cap != self.capacity:
            self.capacity = new_cap
            self.update()
        super().mouseDoubleClickEvent(event)

    def boundingRect(self):
        return self._line_path.boundingRect().united(
            self._arrow_path.boundingRect()
        ).adjusted(-20, -20, 20, 20)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)

        if self._is_path_edge:
            line_color = QColor(255, 140, 0)
            line_width = 3.5
        elif self.isSelected():
            line_color = QColor(255, 200, 50)
            line_width = 3
        elif self.isUnderMouse():
            line_color = QColor(120, 120, 180)
            line_width = 2.5
        else:
            line_color = QColor(80, 80, 80)
            line_width = 2

        pen = QPen(line_color, line_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(self._line_path)

        painter.setBrush(QBrush(line_color))
        painter.drawPath(self._arrow_path)

        # Edge label: flow/capacity
        label = f"{self.flow}/{self.capacity}"
        color = QColor(255, 140, 0) if self._is_path_edge else QColor(40, 40, 40)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QPen(color))

        p1 = self.source_item.pos()
        p2 = self.target_item.pos()
        mid = QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2)
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.sqrt(dx * dx + dy * dy) or 1
        offset = 16
        nx = -dy / length * offset
        ny = dx / length * offset

        painter.drawText(QPointF(mid.x() + nx - 15, mid.y() + ny + 4), label)

    def update_position(self) -> None:
        self.prepareGeometryChange()
        p1 = self.source_item.pos()
        p2 = self.target_item.pos()
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.sqrt(dx * dx + dy * dy) or 1
        ux, uy = dx / length, dy / length

        sx = p1.x() + ux * NODE_RADIUS
        sy = p1.y() + uy * NODE_RADIUS
        ex = p2.x() - ux * (NODE_RADIUS + ARROW_SIZE)
        ey = p2.y() - uy * (NODE_RADIUS + ARROW_SIZE)

        self._line_path = QPainterPath()
        self._line_path.moveTo(sx, sy)
        self._line_path.lineTo(ex, ey)

        angle = math.atan2(ey - sy, ex - sx)
        arrow_tip = QPointF(p2.x() - ux * NODE_RADIUS, p2.y() - uy * NODE_RADIUS)
        a1 = QPointF(
            arrow_tip.x() - ARROW_SIZE * math.cos(angle - math.pi / 7),
            arrow_tip.y() - ARROW_SIZE * math.sin(angle - math.pi / 7),
        )
        a2 = QPointF(
            arrow_tip.x() - ARROW_SIZE * math.cos(angle + math.pi / 7),
            arrow_tip.y() - ARROW_SIZE * math.sin(angle + math.pi / 7),
        )
        self._arrow_path = QPainterPath()
        self._arrow_path.moveTo(arrow_tip)
        self._arrow_path.lineTo(a1)
        self._arrow_path.lineTo(a2)
        self._arrow_path.closeSubpath()
        self.update()

    def set_path_highlight(self, on: bool) -> None:
        self._is_path_edge = on
        self.update()

    def __repr__(self) -> str:
        return (f"EdgeItem(idx={self.edge_idx}, "
                f"{self.source_item.node_id}->{self.target_item.node_id})")


# ═══════════════════════════════════════════════
#  COMMAND PATTERN  (Undo/Redo for graph operations)
# ═══════════════════════════════════════════════

class AddNodeCommand(QUndoCommand):
    """Undoable command: add a node to the graph."""

    def __init__(self, graph: Graph, node_items: dict, scene: QGraphicsScene,
                 pos: QPointF, parent=None):
        super().__init__("Add Node", parent)
        self.graph = graph
        self.node_items = node_items
        self.scene = scene
        self.pos = pos
        self.nid: int = -1
        self.item: Optional[NodeItem] = None

    def redo(self) -> None:
        self.nid = self.graph.add_node(self.pos)
        self.item = NodeItem(self.nid, self.pos)
        self.scene.addItem(self.item)
        self.node_items[self.nid] = self.item

    def undo(self) -> None:
        if self.item and self.nid in self.node_items:
            self.scene.removeItem(self.item)
            del self.node_items[self.nid]
            self.graph.remove_node(self.nid)


class AddEdgeCommand(QUndoCommand):
    """Undoable command: add an edge to the graph."""

    def __init__(self, graph: Graph, edge_items: list, scene: QGraphicsScene,
                 node_items: dict, source_item: NodeItem, target_item: NodeItem,
                 capacity: int, parent=None):
        super().__init__(AddEdgeCommand._label(source_item.node_id, target_item.node_id), parent)
        self.graph = graph
        self.edge_items = edge_items
        self.scene = scene
        self.node_items = node_items
        self.source_item = source_item
        self.target_item = target_item
        self.capacity = capacity
        self.edge_item: Optional[EdgeItem] = None

    @staticmethod
    def _label(u: int, v: int) -> str:
        return f"Add Edge {u}->{v}"

    def redo(self) -> None:
        try:
            eidx = self.graph.add_edge(
                self.source_item.node_id, self.target_item.node_id, self.capacity
            )
            self.edge_item = EdgeItem(eidx, self.source_item, self.target_item, self.capacity)
            self.scene.addItem(self.edge_item)
            self.edge_items.append(self.edge_item)
        except ValueError as exc:
            logger.warning("AddEdgeCommand redo skipped: %s", exc)

    def undo(self) -> None:
        if self.edge_item and self.edge_item in self.edge_items:
            self.scene.removeItem(self.edge_item)
            self.edge_items.remove(self.edge_item)
            # Remove the data edge (last one matching)
            for i in range(len(self.graph.edges) - 1, -1, -1):
                e = self.graph.edges[i]
                if e.u == self.source_item.node_id and e.v == self.target_item.node_id:
                    self.graph.edges.pop(i)
                    self.graph._rebuild_adj()
                    break


class DeleteNodeCommand(QUndoCommand):
    """Undoable command: delete a node (and all connected edges) from the graph."""

    def __init__(self, graph: Graph, node_items: dict, edge_items: list,
                 scene: QGraphicsScene, node_item: NodeItem, parent=None):
        super().__init__(f"Delete Node {node_item.node_id}", parent)
        self.graph = graph
        self.node_items = node_items
        self.edge_items = edge_items
        self.scene = scene
        self.node_item = node_item
        self.nid = node_item.node_id
        self.pos = self.node_item.pos()
        self.role = self.node_item.role
        self.removed_edge_items: list[EdgeItem] = []
        self.removed_edges: list[tuple[int, int, int]] = []

    def redo(self) -> None:
        nid = self.nid
        # Save and remove edge items
        self.removed_edge_items = [
            ei for ei in self.edge_items
            if ei.source_item.node_id == nid or ei.target_item.node_id == nid
        ]
        for ei in self.removed_edge_items:
            self.scene.removeItem(ei)
            self.edge_items.remove(ei)

        # Save and remove data edges
        self.removed_edges = [
            (e.u, e.v, e.capacity) for e in self.graph.edges
            if e.u == nid or e.v == nid
        ]
        self.graph.remove_node(nid)

        self.scene.removeItem(self.node_item)
        del self.node_items[nid]

    def undo(self) -> None:
        nid = self.nid
        # Restore data node
        self.graph.nodes[nid] = self.pos
        self.graph._next_id = max(self.graph._next_id, nid + 1)

        # Restore data edges
        for u, v, cap in self.removed_edges:
            try:
                self.graph.add_edge(u, v, cap)
            except ValueError:
                pass  # may already exist during replay

        # Restore visual node
        item = NodeItem(nid, self.pos)
        self.scene.addItem(item)
        self.node_items[nid] = item
        if self.role == "source":
            self.graph.source = nid
            item.set_role("source")
        elif self.role == "sink":
            self.graph.sink = nid
            item.set_role("sink")

        # Rebuild visual edges
        for i, e in enumerate(self.graph.edges):
            src = self.node_items.get(e.u)
            tgt = self.node_items.get(e.v)
            if src and tgt:
                # Check if visual edge already exists
                existing = any(
                    ei.source_item is src and ei.target_item is tgt
                    for ei in self.edge_items
                )
                if not existing:
                    ei = EdgeItem(i, src, tgt, e.capacity)
                    ei.flow = e.flow
                    self.scene.addItem(ei)
                    self.edge_items.append(ei)


class ClearAllCommand(QUndoCommand):
    """Undoable command: clear the entire graph."""

    def __init__(self, graph: Graph, node_items: dict, edge_items: list,
                 scene: QGraphicsScene, parent=None):
        super().__init__("Clear All", parent)
        self.graph = graph
        self.node_items = node_items
        self.edge_items = edge_items
        self.scene = scene
        # Snapshot current state
        self._snapshot_nodes: dict[int, QPointF] = dict(graph.nodes)
        self._snapshot_edges: list[tuple[int, int, int]] = [
            (e.u, e.v, e.capacity) for e in graph.edges
        ]
        self._snapshot_source = graph.source
        self._snapshot_sink = graph.sink
        self._snapshot_next_id = graph._next_id

    def redo(self) -> None:
        for ei in list(self.edge_items):
            self.scene.removeItem(ei)
        for ni in list(self.node_items.values()):
            self.scene.removeItem(ni)
        self.edge_items.clear()
        self.node_items.clear()
        self.graph.clear()

    def undo(self) -> None:
        # Re-add all nodes
        for nid, pos in self._snapshot_nodes.items():
            self.graph.nodes[nid] = pos
            item = NodeItem(nid, pos)
            self.scene.addItem(item)
            self.node_items[nid] = item
        self.graph._next_id = self._snapshot_next_id
        # Re-add all edges
        for u, v, cap in self._snapshot_edges:
            try:
                self.graph.add_edge(u, v, cap)
            except ValueError:
                pass
        self.graph.source = self._snapshot_source
        self.graph.sink = self._snapshot_sink
        # Set roles
        if self._snapshot_source is not None and self._snapshot_source in self.node_items:
            self.node_items[self._snapshot_source].set_role("source")
        if self._snapshot_sink is not None and self._snapshot_sink in self.node_items:
            self.node_items[self._snapshot_sink].set_role("sink")
        # Rebuild edge visuals
        for i, e in enumerate(self.graph.edges):
            src = self.node_items.get(e.u)
            tgt = self.node_items.get(e.v)
            if src and tgt:
                ei = EdgeItem(i, src, tgt, e.capacity)
                self.scene.addItem(ei)
                self.edge_items.append(ei)


# ═══════════════════════════════════════════════
#  GRAPH VIEW
# ═══════════════════════════════════════════════

class ToolMode(Enum):
    SELECT = auto()
    ADD_NODE = auto()
    ADD_EDGE = auto()
    DELETE = auto()
    DELETE_EDGE = auto()
    SET_SOURCE = auto()
    SET_SINK = auto()


class GraphView(QGraphicsView):
    graph_changed = Signal()
    edge_capacity_changed = Signal(int, int, int)  # edge_idx, old_cap, new_cap
    edge_deleted = Signal(int)  # edge_idx
    graph_saved = Signal(str)  # filepath
    graph_loaded = Signal(str)  # filepath
    random_generated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(*SCENE_RECT)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor(245, 245, 250)))
        self._draw_grid()

        self.graph = Graph()
        self.node_items: dict[int, NodeItem] = {}
        self.edge_items: list[EdgeItem] = []

        self.tool_mode = ToolMode.SELECT
        self.edge_start_node: Optional[NodeItem] = None
        self.temp_line = None

        # Undo/Redo stack
        self.undo_stack = QUndoStack(self)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

    def _draw_grid(self) -> None:
        """Draw grid lines using a cached pixmap for performance."""
        size = GRID_EXTENT * 2 + 1
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(245, 245, 250))  # match background
        painter = QPainter(pixmap)
        painter.translate(GRID_EXTENT, GRID_EXTENT)
        pen = QPen(QColor(220, 220, 230), 1)
        painter.setPen(pen)
        for i in range(-GRID_EXTENT, GRID_EXTENT + 1, GRID_SPACING):
            painter.drawLine(i, -GRID_EXTENT, i, GRID_EXTENT)
            painter.drawLine(-GRID_EXTENT, i, GRID_EXTENT, i)
        painter.end()
        self._grid_pixmap = pixmap
        self.setBackgroundBrush(QBrush(pixmap))

    def set_tool(self, mode: ToolMode) -> None:
        self.tool_mode = mode
        self.edge_start_node = None
        if self.temp_line:
            self.scene.removeItem(self.temp_line)
            self.temp_line = None

        cursors = {
            ToolMode.SELECT: Qt.ArrowCursor,
            ToolMode.ADD_NODE: Qt.CrossCursor,
            ToolMode.ADD_EDGE: Qt.CrossCursor,
            ToolMode.DELETE: Qt.PointingHandCursor,
            ToolMode.DELETE_EDGE: Qt.PointingHandCursor,
            ToolMode.SET_SOURCE: Qt.PointingHandCursor,
            ToolMode.SET_SINK: Qt.PointingHandCursor,
        }
        self.setCursor(cursors.get(mode, Qt.ArrowCursor))
        self.setDragMode(QGraphicsView.RubberBandDrag if mode == ToolMode.SELECT else QGraphicsView.NoDrag)

    def _get_edge_item(self, item) -> Optional["EdgeItem"]:
        """Walk up the item hierarchy to find an EdgeItem."""
        while item:
            if isinstance(item, EdgeItem):
                return item
            item = item.parentItem()
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            if self.tool_mode == ToolMode.ADD_NODE:
                cmd = AddNodeCommand(
                    self.graph, self.node_items, self.scene, scene_pos
                )
                self.undo_stack.push(cmd)
                self.graph_changed.emit()
                return

            item = self.itemAt(event.pos())
            node = self._get_node_item(item)

            if node:
                if self.tool_mode == ToolMode.ADD_EDGE:
                    if self.edge_start_node is None:
                        self.edge_start_node = node
                        self.edge_start_node.setBrush(QBrush(QColor(255, 220, 100)))
                        self.temp_line = self.scene.addLine(
                            QLineF(node.pos(), scene_pos),
                            QPen(QColor(100, 100, 255), 2, Qt.DashLine)
                        )
                    else:
                        if node != self.edge_start_node:
                            self._add_edge_between(self.edge_start_node, node)
                        self.edge_start_node.set_role(self.edge_start_node.role)
                        self.edge_start_node = None
                        if self.temp_line:
                            self.scene.removeItem(self.temp_line)
                            self.temp_line = None
                    return

                elif self.tool_mode == ToolMode.DELETE:
                    cmd = DeleteNodeCommand(
                        self.graph, self.node_items, self.edge_items,
                        self.scene, node
                    )
                    self.undo_stack.push(cmd)
                    self.graph_changed.emit()
                    return

                elif self.tool_mode == ToolMode.DELETE_EDGE:
                    edge_item = self._get_edge_item(item)
                    if edge_item:
                        self._delete_edge(edge_item)
                    return

                elif self.tool_mode == ToolMode.SET_SOURCE:
                    self.graph.source = node.node_id
                    for ni in self.node_items.values():
                        if ni.role == "source":
                            ni.set_role("default")
                    node.set_role("source")
                    self.graph_changed.emit()
                    return

                elif self.tool_mode == ToolMode.SET_SINK:
                    self.graph.sink = node.node_id
                    for ni in self.node_items.values():
                        if ni.role == "sink":
                            ni.set_role("default")
                    node.set_role("sink")
                    self.graph_changed.emit()
                    return

        elif event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.tool_mode == ToolMode.ADD_EDGE and self.edge_start_node and self.temp_line:
            scene_pos = self.mapToScene(event.pos())
            self.temp_line.setLine(QLineF(self.edge_start_node.pos(), scene_pos))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
            if self.tool_mode == ToolMode.SELECT:
                self.setDragMode(QGraphicsView.RubberBandDrag)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def _get_node_item(self, item) -> Optional[NodeItem]:
        """Walk up the item hierarchy to find a NodeItem."""
        while item:
            if isinstance(item, NodeItem):
                return item
            item = item.parentItem()
        return None

    def _add_edge_between(self, source_item: NodeItem, target_item: NodeItem) -> None:
        capacity, ok = QInputDialog.getInt(
            self, "Edge Capacity",
            f"Capacity for edge {source_item.node_id} -> {target_item.node_id}:",
            DEFAULT_CAPACITY, 1, MAX_EDGE_CAPACITY
        )
        if not ok:
            return

        cmd = AddEdgeCommand(
            self.graph, self.edge_items, self.scene, self.node_items,
            source_item, target_item, capacity
        )
        self.undo_stack.push(cmd)
        self.graph_changed.emit()

    def _delete_edge(self, edge_item: EdgeItem) -> None:
        """Delete a single edge from the graph."""
        u = edge_item.source_item.node_id
        v = edge_item.target_item.node_id
        self.scene.removeItem(edge_item)
        self.edge_items.remove(edge_item)
        # Remove from source/target's edge lists
        if edge_item in edge_item.source_item.edges:
            edge_item.source_item.edges.remove(edge_item)
        if edge_item in edge_item.target_item.edges:
            edge_item.target_item.edges.remove(edge_item)
        # Remove data edge
        for i in range(len(self.graph.edges) - 1, -1, -1):
            e = self.graph.edges[i]
            if e.u == u and e.v == v:
                self.graph.edges.pop(i)
                self.graph._rebuild_adj()
                break
        logger.debug("Edge %d->%d deleted", u, v)
        self._rebuild_edge_items()
        self.edge_deleted.emit(edge_item.edge_idx)
        self.graph_changed.emit()

    # ── File I/O ────────────────────────────────

    def save_graph_to_file(self, filepath: str) -> None:
        """Save current graph to a JSON file."""
        data = self.graph.to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Graph saved to %s", filepath)
        self.graph_saved.emit(filepath)

    def load_graph_from_file(self, filepath: str) -> None:
        """Load graph from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        g = Graph.from_dict(data)
        self.clear_all()
        self.undo_stack.clear()
        # Rebuild visuals from loaded graph
        for nid, pos in g.nodes.items():
            item = NodeItem(nid, pos)
            self.scene.addItem(item)
            self.node_items[nid] = item
        self.graph = g
        if g.source is not None and g.source in self.node_items:
            self.node_items[g.source].set_role("source")
        if g.sink is not None and g.sink in self.node_items:
            self.node_items[g.sink].set_role("sink")
        self._rebuild_edge_items()
        logger.info("Graph loaded from %s", filepath)
        self.graph_loaded.emit(filepath)
        self.graph_changed.emit()

    def generate_random_graph(self, num_nodes: int, seed: Optional[int] = None) -> None:
        """Generate and display a random flow network."""
        g = Graph.generate_random(num_nodes, seed)
        self.clear_all()
        self.undo_stack.clear()
        for nid, pos in g.nodes.items():
            item = NodeItem(nid, pos)
            self.scene.addItem(item)
            self.node_items[nid] = item
        self.graph = g
        if g.source is not None and g.source in self.node_items:
            self.node_items[g.source].set_role("source")
        if g.sink is not None and g.sink in self.node_items:
            self.node_items[g.sink].set_role("sink")
        self._rebuild_edge_items()
        logger.info("Random graph generated: %d nodes, %d edges", len(g.nodes), len(g.edges))
        self.random_generated.emit()
        self.graph_changed.emit()

    def _rebuild_edge_items(self) -> None:
        for ei in self.edge_items:
            self.scene.removeItem(ei)
        self.edge_items.clear()
        for i, e in enumerate(self.graph.edges):
            src = self.node_items[e.u]
            tgt = self.node_items[e.v]
            ei = EdgeItem(i, src, tgt, e.capacity)
            ei.flow = e.flow
            self.scene.addItem(ei)
            self.edge_items.append(ei)

    def refresh_edge_display(self) -> None:
        for i, ei in enumerate(self.edge_items):
            if i < len(self.graph.edges):
                ei.flow = self.graph.edges[i].flow
                ei.capacity = self.graph.edges[i].capacity
                ei.update()

    def clear_highlights(self) -> None:
        for ni in self.node_items.values():
            ni.set_highlight("")
            ni.clear_annotations()
        for ei in self.edge_items:
            ei.set_path_highlight(False)

    def highlight_step(self, step: AlgoStep) -> None:
        self.clear_highlights()

        if step.step_type == "augment":
            for nid in step.path:
                if nid in self.node_items:
                    self.node_items[nid].set_highlight("path")
            for eidx in step.path_edges:
                real_idx = eidx if eidx >= 0 else -(eidx + 1)
                if 0 <= real_idx < len(self.edge_items):
                    self.edge_items[real_idx].set_path_highlight(True)

        elif step.step_type == "level_graph":
            level_colors = [
                QColor(100, 180, 255), QColor(150, 220, 150),
                QColor(255, 220, 100), QColor(255, 150, 100),
                QColor(220, 130, 220), QColor(180, 180, 255),
            ]
            for nid, lvl in step.level.items():
                if nid in self.node_items:
                    color = level_colors[lvl % len(level_colors)]
                    self.node_items[nid].setBrush(QBrush(color))
                    self.node_items[nid].setPen(QPen(QColor(40, 40, 40), 2))

        elif step.step_type == "push":
            if step.push_from is not None and step.push_from in self.node_items:
                self.node_items[step.push_from].set_highlight("push_from")
            if step.push_to is not None and step.push_to in self.node_items:
                self.node_items[step.push_to].set_highlight("push_to")
            for ei in self.edge_items:
                if (step.push_from is not None and
                    ei.source_item.node_id == step.push_from and
                    step.push_to is not None and
                    ei.target_item.node_id == step.push_to):
                    ei.set_path_highlight(True)

        elif step.step_type == "relabel":
            if step.relabel_node is not None and step.relabel_node in self.node_items:
                self.node_items[step.relabel_node].set_highlight("relabel")

        # Show heights/excess if present (push-relabel)
        if step.heights:
            for nid, h in step.heights.items():
                if nid in self.node_items:
                    self.node_items[nid].set_height(h)
        if step.excess:
            for nid, ex in step.excess.items():
                if nid in self.node_items:
                    self.node_items[nid].set_excess(ex)

    def clear_all(self) -> None:
        for ei in list(self.edge_items):
            self.scene.removeItem(ei)
        for ni in list(self.node_items.values()):
            self.scene.removeItem(ni)
        self.edge_items.clear()
        self.node_items.clear()
        self.graph.clear()

    def load_graph_data(self, nodes, edges, source, sink) -> None:
        self.clear_all()
        self.undo_stack.clear()
        for nid, pos in nodes:
            self.graph.nodes[nid] = QPointF(*pos)
            self.graph._next_id = max(self.graph._next_id, nid + 1)
            item = NodeItem(nid, QPointF(*pos))
            self.scene.addItem(item)
            self.node_items[nid] = item

        for u, v, cap in edges:
            try:
                self.graph.add_edge(u, v, cap)
            except ValueError as exc:
                logger.warning("load_graph_data: skipping edge %d->%d: %s", u, v, exc)

        self.graph.source = source
        self.graph.sink = sink
        if source is not None and source in self.node_items:
            self.node_items[source].set_role("source")
        if sink is not None and sink in self.node_items:
            self.node_items[sink].set_role("sink")

        self._rebuild_edge_items()
        self.graph_changed.emit()


# ═══════════════════════════════════════════════
#  PRESETS
# ═══════════════════════════════════════════════

PRESETS: dict[str, dict] = {
    "Simple (4 nodes)": {
        "nodes": [(0, (100, 200)), (1, (300, 100)), (2, (300, 300)), (3, (500, 200))],
        "edges": [(0, 1, 10), (0, 2, 10), (1, 2, 2), (1, 3, 4), (2, 3, 9)],
        "source": 0, "sink": 3,
    },
    "Classic Textbook (6 nodes)": {
        "nodes": [
            (0, (80, 250)), (1, (250, 100)), (2, (250, 400)),
            (3, (450, 100)), (4, (450, 400)), (5, (620, 250)),
        ],
        "edges": [
            (0, 1, 16), (0, 2, 13), (1, 2, 10), (1, 3, 12),
            (2, 1, 4), (2, 4, 14), (3, 2, 9), (3, 5, 20),
            (4, 3, 7), (4, 5, 4),
        ],
        "source": 0, "sink": 5,
    },
    "Bipartite Matching": {
        "nodes": [
            (0, (80, 250)),
            (1, (250, 100)), (2, (250, 250)), (3, (250, 400)),
            (4, (450, 100)), (5, (450, 250)), (6, (450, 400)),
            (7, (620, 250)),
        ],
        "edges": [
            (0, 1, 1), (0, 2, 1), (0, 3, 1),
            (1, 4, 1), (1, 5, 1),
            (2, 5, 1), (2, 6, 1),
            (3, 4, 1), (3, 6, 1),
            (4, 7, 1), (5, 7, 1), (6, 7, 1),
        ],
        "source": 0, "sink": 7,
    },
    "Dense Network": {
        "nodes": [
            (0, (80, 250)), (1, (220, 80)), (2, (220, 250)), (3, (220, 420)),
            (4, (420, 150)), (5, (420, 350)), (6, (580, 250)),
        ],
        "edges": [
            (0, 1, 8), (0, 2, 12), (0, 3, 6),
            (1, 4, 5), (1, 2, 3),
            (2, 4, 7), (2, 5, 4),
            (3, 2, 2), (3, 5, 8),
            (4, 6, 10), (5, 6, 9), (4, 5, 3),
        ],
        "source": 0, "sink": 6,
    },
}


# ═══════════════════════════════════════════════
#  LEARNING CONTENT (per algorithm)
# ═══════════════════════════════════════════════

LEARN_OVERVIEW = """
<h2>Network Flow - Overview</h2>

<h3>What is a Flow Network?</h3>
<p>A <b>flow network</b> is a directed graph where each edge has a <i>capacity</i>
and flow must satisfy:</p>
<ul>
<li><b>Capacity constraint:</b> 0 &le; flow(e) &le; capacity(e)</li>
<li><b>Conservation:</b> For every node except S and T: inflow = outflow</li>
<li><b>Skew symmetry:</b> flow(u&rarr;v) = &minus;flow(v&rarr;u)</li>
</ul>

<h3>Goal</h3>
<p>Find the <b>maximum flow</b> from source S to sink T &mdash; the greatest amount
of "stuff" that can be pushed through the network.</p>

<h3>Two Main Paradigms</h3>
<table border="1" cellpadding="6" style="border-collapse: collapse;">
<tr><th>Paradigm</th><th>Idea</th><th>Algorithms</th></tr>
<tr><td><b>Augmenting Paths</b></td><td>Repeatedly find S&rarr;T paths in residual graph and push flow</td>
<td>Ford-Fulkerson, Edmonds-Karp, Dinic's</td></tr>
<tr><td><b>Preflow-Push</b></td><td>Maintain a "preflow" with heights; push excess downhill</td>
<td>Push-Relabel</td></tr>
</table>

<h3>Residual Graph</h3>
<p>For each edge (u,v) with capacity c and flow f:</p>
<ul>
<li><b>Forward edge:</b> (u,v) with residual capacity c &minus; f</li>
<li><b>Backward edge:</b> (v,u) with residual capacity f (lets us "undo" flow)</li>
</ul>
<p>The algorithm terminates when no S&rarr;T path exists in the residual graph.</p>

<h3>Max-Flow Min-Cut Theorem</h3>
<p>The maximum flow equals the capacity of the <b>minimum cut</b> &mdash; a partition
of nodes into two sets (S-side, T-side) that minimizes the total capacity of
edges crossing from S-side to T-side.</p>
"""

LEARN_EDMONDS_KARP = """
<h2>Edmonds-Karp Algorithm</h2>

<h3>Core Idea</h3>
<p>A refinement of Ford-Fulkerson that uses <b>BFS</b> to always pick the
<b>shortest augmenting path</b> (in terms of number of edges).</p>

<h3>Pseudocode</h3>
<pre style="background:#f0f0f0; padding:10px; border-radius:4px;">
max_flow = 0
while True:
    path = BFS(residual_graph, S, T)  # shortest path
    if path is None:
        break
    bottleneck = min(residual_capacity(e) for e in path)
    for e in path:
        if e is forward: flow(e) += bottleneck
        else:            flow(e) -= bottleneck
    max_flow += bottleneck
return max_flow
</pre>

<h3>Why BFS Matters</h3>
<ul>
<li>DFS-based Ford-Fulkerson can take <b>O(E &middot; max_flow)</b> &mdash; bad when capacities are huge.</li>
<li>BFS guarantees the shortest path length <b>never decreases</b>.</li>
<li>Each edge can become critical at most O(V) times.</li>
<li>Result: <b>O(V &middot; E&sup2;)</b> &mdash; independent of capacities!</li>
</ul>

<h3>Key Insight</h3>
<p>The distance from S to any node in the residual graph is <b>monotonically
non-decreasing</b> across iterations. This is what gives the polynomial bound.</p>

<h3>When to Use</h3>
<ul>
<li>Default choice for general max-flow problems.</li>
<li>Simple to implement, reliable performance.</li>
<li>Good for small to medium graphs.</li>
</ul>

<h3>Complexity</h3>
<p><b>Time:</b> O(V &middot; E&sup2;) &nbsp; | &nbsp; <b>Space:</b> O(V + E)</p>
"""

LEARN_FORD_FULKERSON_DFS = """
<h2>Ford-Fulkerson with DFS</h2>

<h3>Core Idea</h3>
<p>The original Ford-Fulkerson method &mdash; find <i>any</i> augmenting path using
<b>Depth-First Search</b> and push flow along it.</p>

<h3>Pseudocode</h3>
<pre style="background:#f0f0f0; padding:10px; border-radius:4px;">
max_flow = 0
while True:
    path = DFS(residual_graph, S, T)  # any path
    if path is None:
        break
    bottleneck = min(residual_capacity(e) for e in path)
    push bottleneck along path
    max_flow += bottleneck
return max_flow
</pre>

<h3>The Danger of DFS</h3>
<p>DFS can pick <b>terrible paths</b> that force the algorithm to undo flow
repeatedly. Classic bad example:</p>
<pre style="background:#fff8e0; padding:8px; border-left:3px solid orange;">
S -> A (cap 10^9), S -> B (cap 10^9)
A -> B (cap 1), A -> T (cap 10^9)
B -> T (cap 10^9)

DFS might alternate: S->A->B->T and S->B->A->T
each pushing only 1 unit -> 2*10^9 iterations!
</pre>

<h3>When is DFS OK?</h3>
<ul>
<li>When capacities are <b>small integers</b> (bounded by max_flow).</li>
<li>When the graph structure doesn't allow pathological paths.</li>
<li>As a teaching tool to understand the basic idea.</li>
</ul>

<h3>Complexity</h3>
<p><b>Time:</b> O(E &middot; max_flow) for integer capacities<br>
<b>Space:</b> O(V + E)<br>
<b>Warning:</b> May not terminate on irrational capacities!</p>

<h3>vs Edmonds-Karp</h3>
<p>Try both on the "Classic Textbook" preset &mdash; notice how DFS often takes
<b>more iterations</b> than BFS!</p>
"""

LEARN_DINIC = """
<h2>Dinic's Algorithm</h2>

<h3>Core Idea</h3>
<p>Combines BFS (to build a <b>level graph</b>) with DFS (to find a
<b>blocking flow</b>). Much faster than Edmonds-Karp on many graphs.</p>

<h3>Two Phases per Iteration</h3>
<ol>
<li><b>Phase 1 &mdash; BFS:</b> Compute level[v] = distance from S to v in residual graph.
Only keep edges (u,v) where level[v] = level[u] + 1. This is the <i>level graph</i>.</li>
<li><b>Phase 2 &mdash; DFS:</b> Find a <b>blocking flow</b> in the level graph &mdash;
a flow such that every S&rarr;T path in the level graph has at least one saturated edge.
Use DFS with <i>current-arc optimization</i>.</li>
<li>Repeat until T is unreachable from S.</li>
</ol>

<h3>Pseudocode</h3>
<pre style="background:#f0f0f0; padding:10px; border-radius:4px;">
max_flow = 0
while True:
    level = BFS_levels(residual_graph, S)
    if level[T] is undefined: break

    # Blocking flow via DFS with current-arc
    iter_ptr = {v: 0 for v in nodes}
    while True:
        pushed = DFS_with_ptr(S, inf, level, iter_ptr)
        if pushed == 0: break
        max_flow += pushed
return max_flow
</pre>

<h3>Current-Arc Optimization</h3>
<p>During DFS, remember which edge we last explored from each node.
Never re-explore dead ends &mdash; this is what makes the blocking flow
phase run in O(V &middot; E) instead of O(V &middot; E&sup2;).</p>

<h3>Why It's Fast</h3>
<ul>
<li>Each iteration increases the S&rarr;T distance by at least 1.</li>
<li>At most V iterations.</li>
<li>Each iteration: O(V &middot; E) for blocking flow.</li>
<li>Total: <b>O(V&sup2; &middot; E)</b>.</li>
</ul>

<h3>When to Use</h3>
<ul>
<li>Best general-purpose algorithm in practice.</li>
<li>Very fast on unit-capacity graphs: O(E &middot; sqrt(V)).</li>
<li>Often beats Push-Relabel on typical inputs.</li>
</ul>

<h3>Visualization Tip</h3>
<p>Watch the <b>level graph</b> steps &mdash; nodes are colored by their level
(distance from S). The algorithm only pushes flow "downhill" in levels.</p>
"""

LEARN_PUSH_RELABEL = """
<h2>Push-Relabel Algorithm</h2>

<h3>Core Idea</h3>
<p>A completely different paradigm! Instead of augmenting paths, we maintain
a <b>preflow</b> (where nodes can have excess flow) and <b>heights</b> on nodes.
Flow always goes "downhill" &mdash; from higher to lower height.</p>

<h3>Key Invariants</h3>
<ul>
<li><b>Height function h:</b> h(S) = |V|, h(T) = 0, and for every residual edge (u,v):
h(u) &le; h(v) + 1.</li>
<li><b>Preflow:</b> For every node v != S: excess(v) &ge; 0 (inflow &ge; outflow).</li>
<li><b>Active node:</b> A node with excess &gt; 0 (and not S or T).</li>
</ul>

<h3>Two Operations</h3>
<p><b>PUSH(u, v):</b> If h(u) = h(v) + 1 and residual capacity exists, push
min(excess(u), residual) from u to v.</p>
<p><b>RELABEL(u):</b> If u is active but can't push anywhere, set
h(u) = min{h(v) : (u,v) is residual edge} + 1.</p>

<h3>Pseudocode</h3>
<pre style="background:#f0f0f0; padding:10px; border-radius:4px;">
h(S) = |V|, h(v) = 0 for all v != S
Saturate all edges from S  # initial preflow
active = {v : excess(v) > 0, v != S, T}

while active not empty:
    pick u from active
    if exists v with h(u) = h(v)+1 and residual(u,v) > 0:
        PUSH(u, v)
        if v became active: add v to active
    else:
        RELABEL(u)
        re-add u to active

return excess(T)
</pre>

<h3>Visualization</h3>
<ul>
<li><b>h=X</b> above each node = its current height.</li>
<li><b>+X</b> badge = excess flow at that node.</li>
<li><b>Red node</b> = pushing flow from here.</li>
<li><b>Blue node</b> = receiving flow.</li>
<li><b>Purple node</b> = being relabeled.</li>
</ul>

<h3>Complexity</h3>
<ul>
<li>Generic: <b>O(V&sup2; &middot; E)</b></li>
<li>With FIFO selection: <b>O(V&sup3;)</b></li>
<li>With highest-label selection: <b>O(V&sup2; &middot; sqrt(E))</b> &mdash; fastest in theory!</li>
</ul>

<h3>When to Use</h3>
<ul>
<li>Very large, dense graphs.</li>
<li>When you need the fastest theoretical bound.</li>
<li>Often preferred in competitive programming.</li>
</ul>

<h3>vs Augmenting Paths</h3>
<p>Push-relabel never maintains a valid flow until the end &mdash; it works with
"broken" preflows. This local approach is what makes it so efficient.</p>
"""

LEARN_COMPARISON = """
<h2>Algorithm Comparison</h2>

<table border="1" cellpadding="8" style="border-collapse: collapse; width: 100%;">
<tr style="background:#e0e8f0;">
<th>Algorithm</th><th>Time</th><th>Idea</th><th>Best For</th>
</tr>
<tr>
<td><b>Ford-Fulkerson (DFS)</b></td>
<td>O(E &middot; f*)</td>
<td>Any augmenting path</td>
<td>Teaching; small integer capacities</td>
</tr>
<tr style="background:#f8f8f8;">
<td><b>Edmonds-Karp (BFS)</b></td>
<td>O(V &middot; E&sup2;)</td>
<td>Shortest augmenting path</td>
<td>General purpose; simple to implement</td>
</tr>
<tr>
<td><b>Dinic's</b></td>
<td>O(V&sup2; &middot; E)</td>
<td>Level graph + blocking flow</td>
<td>Best in practice; unit capacities</td>
</tr>
<tr style="background:#f8f8f8;">
<td><b>Push-Relabel</b></td>
<td>O(V&sup2; &middot; sqrt(E))</td>
<td>Preflow + heights</td>
<td>Huge dense graphs; theory</td>
</tr>
</table>

<h3>Practical Recommendations</h3>
<ol>
<li><b>Start with Edmonds-Karp</b> &mdash; simple and reliable.</li>
<li><b>Use Dinic's</b> when performance matters &mdash; it's what most libraries use.</li>
<li><b>Use Push-Relabel</b> for massive graphs or competitive programming.</li>
<li><b>Avoid plain Ford-Fulkerson (DFS)</b> in production &mdash; it can be exponentially slow.</li>
</ol>

<h3>Try This Experiment</h3>
<p>Load the "Classic Textbook" preset and run all 4 algorithms. Compare:</p>
<ul>
<li>Number of iterations / steps</li>
<li>The paths chosen (DFS vs BFS differ!)</li>
<li>How Dinic's groups work by level</li>
<li>How Push-Relabel's heights evolve</li>
</ul>

<h3>Further Reading</h3>
<ul>
<li><b>CLRS</b> &mdash; Introduction to Algorithms, Chapter 26</li>
<li><b>Jeff Erickson</b> &mdash; Algorithms, Chapter on Max Flow (free online)</li>
<li><b>CP-Algorithms</b> &mdash; maxflow section for competitive programming</li>
</ul>
"""

LEARN_TABS: dict[str, str] = {
    "Overview": LEARN_OVERVIEW,
    "Edmonds-Karp": LEARN_EDMONDS_KARP,
    "Ford-Fulkerson DFS": LEARN_FORD_FULKERSON_DFS,
    "Dinic's": LEARN_DINIC,
    "Push-Relabel": LEARN_PUSH_RELABEL,
    "Comparison": LEARN_COMPARISON,
}

# Map algorithm name -> learning tab name
ALGO_TO_TAB: dict[str, str] = {
    "Edmonds-Karp (BFS)": "Edmonds-Karp",
    "Ford-Fulkerson (DFS)": "Ford-Fulkerson DFS",
    "Dinic's Algorithm": "Dinic's",
    "Push-Relabel": "Push-Relabel",
}


# ═══════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Network Flow Algorithm Learner - Extended")
        self.setMinimumSize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self.engine: Optional[FlowAlgorithm] = None
        self.comparison_results: list[tuple[str, int, int]] = []  # (name, max_flow, steps)
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self._auto_step)
        self.anim_speed = DEFAULT_ANIM_SPEED_MS
        self._show_residual = False

        self._init_ui()
        self._init_toolbar()
        self._init_menu_bar()
        self._init_shortcuts()
        self._init_status_bar()

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left: Graph view
        self.graph_view = GraphView()
        self.graph_view.setMinimumWidth(600)
        splitter.addWidget(self.graph_view)

        # Right: Controls + Tabs
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        # -- Algorithm + Preset selectors --
        top_group = QGroupBox("Setup")
        top_layout = QFormLayout(top_group)

        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(ALGORITHMS.keys()))
        self.algo_combo.currentTextChanged.connect(self._on_algo_changed)
        self.algo_combo.setStyleSheet("font-weight: bold; padding: 4px;")
        top_layout.addRow("Algorithm:", self.algo_combo)

        self.algo_desc = QLabel("")
        self.algo_desc.setWordWrap(True)
        self.algo_desc.setStyleSheet("color: #555; font-style: italic; padding: 4px;")
        top_layout.addRow(self.algo_desc)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        top_layout.addRow("Preset:", self.preset_combo)

        btn_load = QPushButton("Load Preset")
        btn_load.clicked.connect(lambda: self._load_preset(self.preset_combo.currentText()))
        top_layout.addRow(btn_load)

        right_layout.addWidget(top_group)

        # -- Algorithm Controls --
        algo_group = QGroupBox("Run Controls")
        algo_layout = QVBoxLayout(algo_group)

        btn_row1 = QHBoxLayout()
        self.btn_run = QPushButton("Run All")
        self.btn_run.setStyleSheet("background: #27ae60; color: white; padding: 8px; font-weight: bold;")
        self.btn_run.clicked.connect(self._run_all)
        btn_row1.addWidget(self.btn_run)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setStyleSheet("background: #e74c3c; color: white; padding: 8px; font-weight: bold;")
        self.btn_reset.clicked.connect(self._reset)
        btn_row1.addWidget(self.btn_reset)
        algo_layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.btn_step_back = QPushButton("Back")
        self.btn_step_back.clicked.connect(self._step_back)
        btn_row2.addWidget(self.btn_step_back)

        self.btn_step_fwd = QPushButton("Fwd")
        self.btn_step_fwd.clicked.connect(self._step_forward)
        btn_row2.addWidget(self.btn_step_fwd)
        algo_layout.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.graph_view.undo_stack.undo)
        btn_row3.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self.graph_view.undo_stack.redo)
        btn_row3.addWidget(self.btn_redo)
        algo_layout.addLayout(btn_row3)

        self.btn_auto = QPushButton("Auto Play")
        self.btn_auto.setCheckable(True)
        self.btn_auto.clicked.connect(self._toggle_auto_play)
        algo_layout.addWidget(self.btn_auto)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(MIN_ANIM_SPEED_MS, MAX_ANIM_SPEED_MS)
        self.speed_slider.setValue(DEFAULT_ANIM_SPEED_MS)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel(f"{DEFAULT_ANIM_SPEED_MS}ms")
        speed_layout.addWidget(self.speed_label)
        algo_layout.addLayout(speed_layout)

        # Step slider for direct seeking
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Step:"))
        self.step_slider = QSlider(Qt.Horizontal)
        self.step_slider.setRange(0, 0)
        self.step_slider.setValue(0)
        self.step_slider.setEnabled(False)
        self.step_slider.valueChanged.connect(self._on_step_slider_changed)
        slider_layout.addWidget(self.step_slider)
        algo_layout.addLayout(slider_layout)

        right_layout.addWidget(algo_group)

        # -- Step Info --
        step_group = QGroupBox("Current Step")
        step_layout = QVBoxLayout(step_group)

        self.step_counter_label = QLabel("Step: - / -")
        self.step_counter_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        step_layout.addWidget(self.step_counter_label)

        self.flow_label = QLabel("Max Flow: -")
        self.flow_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.flow_label.setStyleSheet("color: #2980b9; padding: 6px;")
        step_layout.addWidget(self.flow_label)

        self.min_cut_label = QLabel("Min-Cut: -")
        self.min_cut_label.setFont(QFont("Segoe UI", 10))
        self.min_cut_label.setStyleSheet("color: #7f8c8d; padding: 2px;")
        step_layout.addWidget(self.min_cut_label)

        self.step_detail = QTextEdit()
        self.step_detail.setReadOnly(True)
        self.step_detail.setMaximumHeight(180)
        self.step_detail.setStyleSheet("""
            QTextEdit {
                background: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                padding: 6px;
            }
        """)
        step_layout.addWidget(self.step_detail)

        right_layout.addWidget(step_group)

        # -- Comparison Group --
        cmp_group = QGroupBox("Algorithm Comparison")
        cmp_layout = QVBoxLayout(cmp_group)

        btn_compare_all = QPushButton("Run All Algorithms")
        btn_compare_all.setStyleSheet("background: #8e44ad; color: white; padding: 8px; font-weight: bold;")
        btn_compare_all.clicked.connect(self._run_comparison)
        cmp_layout.addWidget(btn_compare_all)

        self.comparison_table = QTextEdit()
        self.comparison_table.setReadOnly(True)
        self.comparison_table.setMaximumHeight(140)
        self.comparison_table.setStyleSheet("""
            QTextEdit {
                background: #f4f6f9;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                padding: 6px;
            }
        """)
        self.comparison_table.setPlaceholderText("Click 'Run All Algorithms' to compare...")
        cmp_layout.addWidget(self.comparison_table)

        # Residual graph toggle
        self.btn_residual = QPushButton("Show Residual Graph")
        self.btn_residual.setCheckable(True)
        self.btn_residual.clicked.connect(self._toggle_residual_graph)
        cmp_layout.addWidget(self.btn_residual)

        right_layout.addWidget(cmp_group)

        # -- Learning Tabs --
        self.learn_tabs = QTabWidget()
        self.learn_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                background: white;
            }
            QTabBar::tab {
                padding: 6px 12px;
                margin: 2px;
                background: #ecf0f1;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
                font-weight: bold;
            }
        """)

        self.tab_texts: dict[str, QTextEdit] = {}
        for tab_name, html_content in LEARN_TABS.items():
            text_edit = QTextEdit()
            text_edit.setHtml(html_content)
            text_edit.setReadOnly(True)
            text_edit.setStyleSheet("""
                QTextEdit {
                    background: #fafafa;
                    font-size: 13px;
                    padding: 8px;
                    border: none;
                }
            """)
            self.learn_tabs.addTab(text_edit, tab_name)
            self.tab_texts[tab_name] = text_edit

        right_layout.addWidget(self.learn_tabs, stretch=1)

        scroll = QScrollArea()
        scroll.setWidget(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(460)
        splitter.addWidget(scroll)

        splitter.setSizes([850, 420])

        # Initialize description
        self._on_algo_changed(self.algo_combo.currentText())

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("Tools")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.tool_actions: dict[ToolMode, QAction] = {}
        tools = [
            ("Select / Move", ToolMode.SELECT),
            ("Add Node", ToolMode.ADD_NODE),
            ("Add Edge", ToolMode.ADD_EDGE),
            ("Delete Node", ToolMode.DELETE),
            ("Delete Edge", ToolMode.DELETE_EDGE),
            ("Set Source (S)", ToolMode.SET_SOURCE),
            ("Set Sink (T)", ToolMode.SET_SINK),
        ]

        for name, mode in tools:
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, m=mode: self._set_tool(m))
            toolbar.addAction(action)
            self.tool_actions[mode] = action

        self.tool_actions[ToolMode.SELECT].setChecked(True)
        toolbar.addSeparator()

        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self._clear_all)
        toolbar.addAction(clear_action)

        toolbar.addSeparator()

        # Random graph button with spinbox
        toolbar.addWidget(QLabel("  Random: "))
        self.random_spin = QSpinBox()
        self.random_spin.setRange(RANDOM_MIN_NODES, RANDOM_MAX_NODES)
        self.random_spin.setValue(8)
        self.random_spin.setFixedWidth(50)
        toolbar.addWidget(self.random_spin)
        random_action = QAction("Generate", self)
        random_action.triggered.connect(self._on_generate_random)
        toolbar.addAction(random_action)

    def _init_status_bar(self) -> None:
        self.statusBar().showMessage("Ready. Select an algorithm and load a preset to begin.")

    def _init_menu_bar(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        save_action = QAction("Save Graph...", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_graph)
        file_menu.addAction(save_action)

        open_action = QAction("Open Graph...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_graph)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        zoom_in_action.triggered.connect(lambda: self.graph_view.scale(1.25, 1.25))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        zoom_out_action.triggered.connect(lambda: self.graph_view.scale(1 / 1.25, 1 / 1.25))
        view_menu.addAction(zoom_out_action)

        reset_view_action = QAction("Reset View", self)
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)

    def _init_shortcuts(self) -> None:
        """Setup keyboard shortcuts for quick access."""
        # Undo/Redo
        QShortcut(QKeySequence.StandardKey.Undo, self,
                  lambda: self.graph_view.undo_stack.undo())
        QShortcut(QKeySequence.StandardKey.Redo, self,
                  lambda: self.graph_view.undo_stack.redo())

        # Run / Reset
        QShortcut(QKeySequence("F5"), self, self._run_all)
        QShortcut(QKeySequence("Escape"), self, self._reset)

        # Step forward/back
        QShortcut(QKeySequence("Right"), self, self._step_forward)
        QShortcut(QKeySequence("Left"), self, self._step_back)

        # Auto play toggle
        QShortcut(QKeySequence("Space"), self, self._toggle_auto_play_shortcut)

        # Tool shortcuts
        QShortcut(QKeySequence("1"), self, lambda: self._set_tool(ToolMode.SELECT))
        QShortcut(QKeySequence("2"), self, lambda: self._set_tool(ToolMode.ADD_NODE))
        QShortcut(QKeySequence("3"), self, lambda: self._set_tool(ToolMode.ADD_EDGE))
        QShortcut(QKeySequence("4"), self, lambda: self._set_tool(ToolMode.DELETE))
        QShortcut(QKeySequence("5"), self, lambda: self._set_tool(ToolMode.SET_SOURCE))
        QShortcut(QKeySequence("6"), self, lambda: self._set_tool(ToolMode.SET_SINK))

    def _toggle_auto_play_shortcut(self) -> None:
        """Toggle auto-play via Space shortcut."""
        self.btn_auto.setChecked(not self.btn_auto.isChecked())
        self._toggle_auto_play(self.btn_auto.isChecked())

    def _set_tool(self, mode: ToolMode) -> None:
        for m, a in self.tool_actions.items():
            a.setChecked(m == mode)
        self.graph_view.set_tool(mode)

    def _on_algo_changed(self, algo_name: str) -> None:
        algo_cls = ALGORITHMS.get(algo_name)
        if algo_cls:
            self.algo_desc.setText(algo_cls.short_desc)
        # Auto-switch to corresponding learning tab
        tab_name = ALGO_TO_TAB.get(algo_name)
        if tab_name and tab_name in self.tab_texts:
            idx = list(LEARN_TABS.keys()).index(tab_name)
            self.learn_tabs.setCurrentIndex(idx)

    def _load_preset(self, name: str) -> None:
        if name not in PRESETS:
            return
        preset = PRESETS[name]
        self._reset()
        self.graph_view.load_graph_data(
            preset["nodes"], preset["edges"], preset["source"], preset["sink"]
        )
        self.statusBar().showMessage(f"Loaded preset: {name}")

    def _run_all(self) -> None:
        if self.graph_view.graph.source is None or self.graph_view.graph.sink is None:
            QMessageBox.warning(self, "Missing Source/Sink",
                                "Please set both a Source (S) and Sink (T) node first.")
            return
        if not self.graph_view.graph.edges:
            QMessageBox.warning(self, "No Edges", "Please add edges to the graph first.")
            return

        try:
            algo_name = self.algo_combo.currentText()
            algo_cls = ALGORITHMS[algo_name]
            self.engine = algo_cls(self.graph_view.graph)
            self.engine.run_all()

            if self.engine.steps:
                self.engine.go_to_step(0)
                self._update_display()

            self.statusBar().showMessage(
                f"{algo_name} complete. Max flow = {self.engine.max_flow}. "
                f"Total steps: {len(self.engine.steps)}"
            )
        except Exception as exc:
            logger.exception("Error running algorithm")
            QMessageBox.critical(self, "Algorithm Error", f"An error occurred:\n{exc}")

    def _step_forward(self) -> None:
        if self.engine and self.engine.step_forward():
            self._update_display()

    def _step_back(self) -> None:
        if self.engine and self.engine.step_backward():
            self._update_display()

    def _reset(self) -> None:
        self.anim_timer.stop()
        self.btn_auto.setChecked(False)
        self.engine = None
        self.graph_view.graph.reset_flows()
        self.graph_view.refresh_edge_display()
        self.graph_view.clear_highlights()
        self.step_counter_label.setText("Step: - / -")
        self.flow_label.setText("Max Flow: -")
        self.min_cut_label.setText("Min-Cut: -")
        self.step_detail.clear()
        self.statusBar().showMessage("Reset. Ready.")

    def _clear_all(self) -> None:
        self.anim_timer.stop()
        self.btn_auto.setChecked(False)
        self.engine = None
        self.graph_view.clear_all()
        self.graph_view.undo_stack.clear()
        self.step_counter_label.setText("Step: - / -")
        self.flow_label.setText("Max Flow: -")
        self.min_cut_label.setText("Min-Cut: -")
        self.step_detail.clear()
        self.step_slider.blockSignals(True)
        self.step_slider.setRange(0, 0)
        self.step_slider.setValue(0)
        self.step_slider.setEnabled(False)
        self.step_slider.blockSignals(False)
        self.comparison_table.clear()
        self.statusBar().showMessage("Graph cleared.")

    def _toggle_auto_play(self, checked: bool) -> None:
        if checked:
            if self.engine is None:
                self._run_all()
                if self.engine is None:
                    self.btn_auto.setChecked(False)
                    return
            self.anim_timer.start(self.anim_speed)
            self.btn_auto.setText("Pause")
        else:
            self.anim_timer.stop()
            self.btn_auto.setText("Auto Play")

    def _auto_step(self) -> None:
        if self.engine:
            if not self.engine.step_forward():
                self.anim_timer.stop()
                self.btn_auto.setChecked(False)
                self.btn_auto.setText("Auto Play")
            self._update_display()

    def _on_speed_changed(self, value: int) -> None:
        self.anim_speed = value
        self.speed_label.setText(f"{value}ms")
        if self.anim_timer.isActive():
            self.anim_timer.setInterval(value)

    def _on_step_slider_changed(self, value: int) -> None:
        """Jump directly to a step via the slider."""
        if self.engine and 0 <= value < len(self.engine.steps):
            self.engine.go_to_step(value)
            self._update_display()

    def _save_graph(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                self.graph_view.save_graph_to_file(filepath)
                self.statusBar().showMessage(f"Graph saved to {filepath}")
            except Exception as exc:
                logger.exception("Save failed")
                QMessageBox.critical(self, "Save Error", f"Failed to save: {exc}")

    def _open_graph(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Graph", "", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            try:
                self._reset()
                self.graph_view.load_graph_from_file(filepath)
                self.statusBar().showMessage(f"Graph loaded from {filepath}")
            except Exception as exc:
                logger.exception("Load failed")
                QMessageBox.critical(self, "Load Error", f"Failed to load: {exc}")

    def _on_generate_random(self) -> None:
        num_nodes = self.random_spin.value()
        self._reset()
        self.graph_view.generate_random_graph(num_nodes)
        self.statusBar().showMessage(f"Generated random graph with {num_nodes} nodes")

    def _reset_view(self) -> None:
        """Reset zoom/pan to default."""
        self.graph_view.resetTransform()

    def _run_comparison(self) -> None:
        """Run all algorithms on the current graph and display comparison."""
        if (self.graph_view.graph.source is None or
                self.graph_view.graph.sink is None or
                not self.graph_view.graph.edges):
            QMessageBox.warning(self, "Cannot Compare",
                                "Set source, sink, and add edges first.")
            return

        # Snapshot current graph state (flows)
        original_flows = [e.flow for e in self.graph_view.graph.edges]
        self.comparison_results = []

        for algo_name, algo_cls in ALGORITHMS.items():
            # Reset flows before each run
            for i, e in enumerate(self.graph_view.graph.edges):
                e.flow = original_flows[i] if i < len(original_flows) else 0
            try:
                engine = algo_cls(self.graph_view.graph)
                engine.run_all()
                self.comparison_results.append((
                    algo_name, engine.max_flow, len(engine.steps)
                ))
            except Exception as exc:
                logger.exception("Comparison run failed for %s", algo_name)
                self.comparison_results.append((
                    algo_name, -1, -1
                ))

        # Reset flows back
        for i, e in enumerate(self.graph_view.graph.edges):
            e.flow = original_flows[i] if i < len(original_flows) else 0
        self.graph_view.refresh_edge_display()
        self.graph_view.clear_highlights()

        # Display results
        self._display_comparison()
        self.statusBar().showMessage("Comparison complete.")

    def _display_comparison(self) -> None:
        """Format comparison results as a table in the text widget."""
        if not self.comparison_results:
            return
        lines = [f"{'Algorithm':<24} {'Max Flow':>8} {'Steps':>6}"]
        lines.append("-" * 42)
        for name, max_flow, steps in self.comparison_results:
            flow_str = str(max_flow) if max_flow >= 0 else "ERROR"
            steps_str = str(steps) if steps >= 0 else "-"
            lines.append(f"{name:<24} {flow_str:>8} {steps_str:>6}")
        lines.append("-" * 42)
        # Check if all max flows agree
        flows = [r[1] for r in self.comparison_results if r[1] >= 0]
        if flows and len(set(flows)) == 1:
            lines.append(f"All algorithms agree: max flow = {flows[0]}")
        elif flows:
            lines.append(f"WARNING: max flows differ! ({', '.join(str(f) for f in set(flows))})")
        self.comparison_table.setPlainText("\n".join(lines))

    def _toggle_residual_graph(self, checked: bool) -> None:
        """Toggle display of residual graph overlay on edges."""
        self._show_residual = checked
        self.btn_residual.setText(
            "Hide Residual Graph" if checked else "Show Residual Graph"
        )
        if not checked:
            # Remove residual edge items
            for item in list(self.graph_view.scene.items()):
                if isinstance(item, QGraphicsLineItem) and hasattr(item, '_is_residual'):
                    self.graph_view.scene.removeItem(item)
        elif self.engine:
            self._draw_residual_overlay()

    def _draw_residual_overlay(self) -> None:
        """Draw residual edges as dashed lines on the graph."""
        # Clear previous residual items
        for item in list(self.graph_view.scene.items()):
            if isinstance(item, QGraphicsLineItem) and hasattr(item, '_is_residual'):
                self.graph_view.scene.removeItem(item)

        if not self.graph_view.graph.edges:
            return

        residual_pen = QPen(QColor(0, 180, 0, 120), 1.5, Qt.DashLine)
        for eidx, e in enumerate(self.graph_view.graph.edges):
            src_item = self.graph_view.node_items.get(e.u)
            tgt_item = self.graph_view.node_items.get(e.v)
            if not src_item or not tgt_item:
                continue
            # Forward residual
            if e.residual_capacity > 0:
                line = QGraphicsLineItem(QLineF(src_item.pos(), tgt_item.pos()))
                line.setPen(residual_pen)
                line.setZValue(0)
                line._is_residual = True
                self.graph_view.scene.addItem(line)
            # Backward residual (reverse direction)
            if e.flow > 0:
                p1 = tgt_item.pos()
                p2 = src_item.pos()
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                length = math.sqrt(dx * dx + dy * dy) or 1
                ux, uy = dx / length, dy / length
                offset = 8
                # Slight offset so it doesn't overlap exactly
                nx, ny = -uy * offset, ux * offset
                offset_p1 = QPointF(p1.x() + nx, p1.y() + ny)
                offset_p2 = QPointF(p2.x() + nx, p2.y() + ny)
                reverse_pen = QPen(QColor(180, 0, 0, 120), 1.5, Qt.DotLine)
                line = QGraphicsLineItem(QLineF(offset_p1, offset_p2))
                line.setPen(reverse_pen)
                line.setZValue(0)
                line._is_residual = True
                self.graph_view.scene.addItem(line)

    def _update_display(self) -> None:
        if not self.engine:
            return

        self.graph_view.refresh_edge_display()

        step = self.engine.steps[self.engine.current_step]
        self.graph_view.highlight_step(step)

        total = len(self.engine.steps)
        current = self.engine.current_step + 1
        self.step_counter_label.setText(f"Step: {current} / {total}")
        self.flow_label.setText(f"Max Flow: {step.total_flow}")

        # Sync step slider (block signals to avoid feedback loop)
        self.step_slider.blockSignals(True)
        self.step_slider.setRange(0, max(0, total - 1))
        self.step_slider.setValue(self.engine.current_step)
        self.step_slider.setEnabled(True)
        self.step_slider.blockSignals(False)

        # Update residual graph overlay if toggled on
        if self._show_residual:
            self._draw_residual_overlay()

        # Show min-cut info on the final step
        if step.step_type == "done":
            s_side = self.engine.compute_min_cut()
            cut_cap = sum(
                e.capacity for e in self.graph_view.graph.edges
                if e.u in s_side and e.v not in s_side
            )
            self.min_cut_label.setText(
                f"Min-Cut capacity: {cut_cap}  (S-side: {{{', '.join(map(str, sorted(s_side)))}}})"
            )
            # Highlight min-cut edges
            for ei in self.graph_view.edge_items:
                u_id = ei.source_item.node_id
                v_id = ei.target_item.node_id
                if u_id in s_side and v_id not in s_side:
                    ei.set_path_highlight(True)
        else:
            self.min_cut_label.setText("Min-Cut: (visible at completion)")

        self.step_detail.setPlainText(step.description)


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    app.setStyleSheet("""
        QMainWindow { background: #ffffff; }
        QGroupBox {
            font-weight: bold;
            font-size: 13px;
            border: 1px solid #ccc;
            border-radius: 6px;
            margin-top: 10px;
            padding-top: 14px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }
        QPushButton {
            padding: 6px 14px;
            border: 1px solid #aaa;
            border-radius: 4px;
            background: #ecf0f1;
            font-size: 12px;
        }
        QPushButton:hover { background: #d5dbdb; }
        QPushButton:pressed { background: #bdc3c7; }
        QComboBox {
            padding: 4px 8px;
            border: 1px solid #aaa;
            border-radius: 4px;
        }
        QToolBar {
            background: #f5f6fa;
            border-bottom: 1px solid #ddd;
            padding: 4px;
            spacing: 4px;
        }
        QLabel { font-size: 12px; }
    """)

    window = MainWindow()
    window.show()
    window._load_preset("Classic Textbook (6 nodes)")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
