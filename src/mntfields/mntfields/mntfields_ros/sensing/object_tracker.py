"""
object_tracker.py — Temporal accumulation of semantic detections.

Pure Python (no ROS deps).  Maintains persistent object tracks with
nearest-neighbour association, EMA position smoothing, and a
candidate → confirmed → lost lifecycle.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    map_xy: np.ndarray               # EMA-smoothed position [x, y]
    confidence: float                 # running-mean confidence
    observation_count: int = 1
    miss_count: int = 0
    status: str = "candidate"        # "candidate" | "confirmed" | "lost"
    first_seen: float = 0.0          # timestamp (seconds)
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "map_xy": self.map_xy.tolist(),
            "confidence": round(self.confidence, 3),
            "observation_count": self.observation_count,
            "miss_count": self.miss_count,
            "status": self.status,
            "first_seen_sec": self.first_seen,
            "last_seen_sec": self.last_seen,
        }


class ObjectTracker:
    """Multi-frame temporal tracker with simple nearest-neighbour association.

    Lifecycle:
        candidate  ──(obs_count >= confirm_threshold)──►  confirmed
        confirmed  ──(miss_count >= lost_threshold)────►  lost
        lost       ──(re-observed)─────────────────────►  confirmed
    """

    def __init__(
        self,
        confirm_threshold: int = 5,
        lost_threshold: int = 30,
        association_radius_m: float = 0.6,
        ema_alpha: float = 0.3,
    ):
        self.confirm_threshold = confirm_threshold
        self.lost_threshold = lost_threshold
        self.association_radius = association_radius_m
        self.ema_alpha = ema_alpha

        self._tracks: list[TrackedObject] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------
    def update(
        self, detections: list[dict], timestamp: float
    ) -> list[TrackedObject]:
        """Process a batch of per-frame detections.

        Each detection dict must contain:
            class_name: str
            map_xy:     [x, y]  (list or array, in map frame)
            confidence: float

        Returns the full list of active (non-lost) tracks after update.
        """
        matched_track_ids: set = set()

        for det in detections:
            cls = det["class_name"]
            xy = np.asarray(det["map_xy"], dtype=np.float32)
            conf = float(det["confidence"])

            best_track = None
            best_dist = self.association_radius

            for trk in self._tracks:
                if trk.class_name != cls:
                    continue
                if trk.status == "lost":
                    continue
                d = float(np.linalg.norm(trk.map_xy - xy))
                if d < best_dist:
                    best_dist = d
                    best_track = trk

            if best_track is not None:
                # Update existing track
                a = self.ema_alpha
                best_track.map_xy = a * xy + (1.0 - a) * best_track.map_xy
                best_track.confidence = a * conf + (1.0 - a) * best_track.confidence
                best_track.observation_count += 1
                best_track.miss_count = 0
                best_track.last_seen = timestamp
                # Promote / resurrect
                if best_track.observation_count >= self.confirm_threshold:
                    best_track.status = "confirmed"
                matched_track_ids.add(best_track.track_id)
            else:
                # New candidate track
                trk = TrackedObject(
                    track_id=self._next_id,
                    class_name=cls,
                    map_xy=xy.copy(),
                    confidence=conf,
                    observation_count=1,
                    miss_count=0,
                    status="candidate",
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._next_id += 1
                self._tracks.append(trk)
                matched_track_ids.add(trk.track_id)

        # Increment miss_count for unmatched active tracks
        for trk in self._tracks:
            if trk.track_id not in matched_track_ids and trk.status != "lost":
                trk.miss_count += 1
                if trk.status == "confirmed" and trk.miss_count >= self.lost_threshold:
                    trk.status = "lost"

        # Purge old candidates that never confirmed
        self._tracks = [
            t for t in self._tracks
            if not (t.status == "candidate" and t.miss_count >= self.lost_threshold)
        ]

        return self.get_active()

    # ------------------------------------------------------------------
    def get_active(self) -> list[TrackedObject]:
        """Return all non-lost tracks."""
        return [t for t in self._tracks if t.status != "lost"]

    def get_confirmed(self) -> list[TrackedObject]:
        """Return only confirmed tracks."""
        return [t for t in self._tracks if t.status == "confirmed"]

    def get_all(self) -> list[TrackedObject]:
        """Return every track including lost ones."""
        return list(self._tracks)
