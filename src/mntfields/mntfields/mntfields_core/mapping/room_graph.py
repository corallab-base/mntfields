"""
Room connectivity graph.

Nodes = rooms, edges = doorways.  Provides shortest-path queries and
JSON serialisation for persistence between mapping and navigation modes.

Pure algorithm — no ROS dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from .room_segmenter import Doorway, RoomSegment


class RoomGraph:
    """
    Lightweight graph over detected rooms connected by doorways.

    Usage::

        g = RoomGraph()
        g.update(rooms, doorways)
        path = g.shortest_path(room_a_id, room_b_id)
        g.save("model_dir/room_graph.json")
    """

    def __init__(self) -> None:
        self.rooms: dict[int, RoomSegment] = {}
        self.doorways: dict[int, Doorway] = {}
        self._graph: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Build / update
    # ------------------------------------------------------------------

    def update(self, rooms: list[RoomSegment], doorways: list[Doorway]) -> None:
        """Rebuild the graph from the latest segmentation results."""
        self.rooms = {r.id: r for r in rooms}
        self.doorways = {d.id: d for d in doorways}
        self._graph = nx.Graph()

        for r in rooms:
            self._graph.add_node(r.id,
                                 centroid=r.centroid.tolist(),
                                 bbox=r.bounding_box.tolist())

        for d in doorways:
            r1, r2 = d.room_ids
            if r1 in self.rooms and r2 in self.rooms:
                weight = float(np.linalg.norm(
                    self.rooms[r1].centroid - self.rooms[r2].centroid))
                self._graph.add_edge(r1, r2,
                                     doorway_id=d.id,
                                     position=d.position.tolist(),
                                     width=d.width,
                                     weight=max(weight, 0.1))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def shortest_path(self, from_room: int, to_room: int
                      ) -> list[dict]:
        """
        Shortest path through the room graph.

        Returns a list of waypoints::

            [
              {"type": "room",    "id": 1, "position": [x, y]},
              {"type": "doorway", "id": 5, "position": [x, y]},
              {"type": "room",    "id": 2, "position": [x, y]},
              ...
            ]

        Raises ``nx.NetworkXNoPath`` if no path exists.
        """
        if from_room == to_room:
            r = self.rooms[from_room]
            return [{"type": "room", "id": from_room,
                     "position": r.centroid.tolist()}]

        node_path = nx.shortest_path(self._graph, from_room, to_room,
                                     weight="weight")
        result: list[dict] = []
        for i, node in enumerate(node_path):
            r = self.rooms[node]
            result.append({"type": "room", "id": node,
                           "position": r.centroid.tolist()})
            if i < len(node_path) - 1:
                edge = self._graph.edges[node, node_path[i + 1]]
                did = edge["doorway_id"]
                d = self.doorways[did]
                result.append({"type": "doorway", "id": did,
                               "position": d.position.tolist()})
        return result

    def room_for_point(self, xy: np.ndarray) -> Optional[int]:
        """
        Return the room ID whose bounding box contains *xy*, or the nearest
        room if *xy* falls outside all boxes.
        """
        xy = np.asarray(xy).ravel()[:2]

        # exact containment
        for rid, room in self.rooms.items():
            if room.contains_point(xy):
                return rid

        # fallback: nearest centroid
        best_id: Optional[int] = None
        best_dist = float("inf")
        for rid, room in self.rooms.items():
            d = float(np.linalg.norm(room.centroid - xy))
            if d < best_dist:
                best_dist = d
                best_id = rid
        return best_id

    @property
    def room_ids(self) -> list[int]:
        return list(self.rooms.keys())

    @property
    def num_rooms(self) -> int:
        return len(self.rooms)

    @property
    def num_doorways(self) -> int:
        return len(self.doorways)

    def adjacent_rooms(self, room_id: int) -> list[int]:
        """Room IDs directly connected to *room_id* via a doorway."""
        if room_id not in self._graph:
            return []
        return list(self._graph.neighbors(room_id))

    def doorway_between(self, room_a: int, room_b: int) -> Optional[Doorway]:
        """Return the Doorway connecting two adjacent rooms, or None."""
        if not self._graph.has_edge(room_a, room_b):
            return None
        did = self._graph.edges[room_a, room_b]["doorway_id"]
        return self.doorways.get(did)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save graph to JSON."""
        data = {
            "rooms": {},
            "doorways": {},
            "edges": [],
        }
        for rid, room in self.rooms.items():
            data["rooms"][str(rid)] = {
                "id": rid,
                "centroid": room.centroid.tolist(),
                "bounding_box": room.bounding_box.tolist(),
                "segment_count": len(room.line_segments),
            }
        for did, door in self.doorways.items():
            data["doorways"][str(did)] = {
                "id": did,
                "position": door.position.tolist(),
                "width": door.width,
                "room_ids": list(door.room_ids),
            }
        for u, v, edata in self._graph.edges(data=True):
            data["edges"].append({
                "rooms": [u, v],
                "doorway_id": edata["doorway_id"],
                "weight": edata["weight"],
            })
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """Load graph from JSON (navigation mode)."""
        raw = json.loads(Path(path).read_text())

        self.rooms = {}
        for _key, rdata in raw["rooms"].items():
            room = RoomSegment(
                id=rdata["id"],
                line_segments=[],   # segments not persisted — only bounds needed
                bounding_box=np.array(rdata["bounding_box"]),
                centroid=np.array(rdata["centroid"]),
            )
            self.rooms[room.id] = room

        self.doorways = {}
        for _key, ddata in raw["doorways"].items():
            door = Doorway(
                id=ddata["id"],
                position=np.array(ddata["position"]),
                width=ddata["width"],
                room_ids=tuple(ddata["room_ids"]),
            )
            self.doorways[door.id] = door

        self._graph = nx.Graph()
        for room in self.rooms.values():
            self._graph.add_node(room.id,
                                 centroid=room.centroid.tolist(),
                                 bbox=room.bounding_box.tolist())
        for edata in raw["edges"]:
            r1, r2 = edata["rooms"]
            self._graph.add_edge(r1, r2,
                                 doorway_id=edata["doorway_id"],
                                 position=self.doorways[edata["doorway_id"]].position.tolist(),
                                 width=self.doorways[edata["doorway_id"]].width,
                                 weight=edata["weight"])
