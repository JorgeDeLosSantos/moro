"""Numerical data containers used by the visualization backends."""

from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np


@dataclass(slots=True)
class FrameData:
    """Numerical homogeneous transformation of one robot frame."""

    T: np.ndarray

    @property
    def position(self):
        return self.T[:3, 3]

    @property
    def rotation(self):
        return self.T[:3, :3]

    @property
    def x(self):
        return self.rotation[:, 0]

    @property
    def y(self):
        return self.rotation[:, 1]

    @property
    def z(self):
        return self.rotation[:, 2]


@dataclass(slots=True)
class DHSegmentData:
    """One numerical DH translation contribution, in base coordinates.

    `row_index` is 1-based. `kind` is `"d"` or `"a"`.
    The endpoints are independent, finite real arrays with shape `(3,)`
    when produced by :func:`evaluate_robot`. These segments describe DH
    geometry, not the physical shape of a robot link.
    """

    row_index: int
    kind: Literal["d", "a"]
    start: np.ndarray
    end: np.ndarray



@dataclass(slots=True)
class JointData:
    """Kinematic metadata for a single classic-DH joint.

    The index is the 1-based DH row/joint index; kind is "r" or "p".
    Origin is the previous frame origin p_(i-1), and axis is its positive
    z direction z_(i-1), both expressed in world/base coordinates.
    Together they define the motion axis, not physical actuator geometry.

    Origin and axis are independent finite float arrays of shape (3,).
    Any nonzero finite axis is normalized to unit length on construction.
    """

    index: int
    kind: Literal["r", "p"]
    origin: np.ndarray
    axis: np.ndarray

    def __post_init__(self):
        if isinstance(self.index, (bool, np.bool_)) or not isinstance(
            self.index, (int, np.integer)
        ) or self.index < 1:
            raise ValueError("Joint index must be a positive integer (1-based).")
        self.index = int(self.index)

        if self.kind not in ("r", "p"):
            raise ValueError("Joint kind must be 'r' or 'p'.")

        self.origin = self._as_vector(self.origin, "origin")
        self.axis = self._as_vector(self.axis, "axis")

        length = math.hypot(*self.axis)
        if not math.isfinite(length) or length == 0:
            raise ValueError("Joint axis must have a finite nonzero direction.")
        self.axis /= length

    @staticmethod
    def _as_vector(value, name: str) -> np.ndarray:
        if np.iscomplexobj(value):
            raise ValueError(f"Joint {name} must contain three finite real values.")
        try:
            vector = np.array(value, dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Joint {name} must contain three finite real values."
            ) from exc
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"Joint {name} must contain three finite real values.")
        return vector


@dataclass(slots=True)
class SceneData:
    """Evaluated numerical data for one robot configuration.

    dh_segments lists two contributions per DH row in d, a order.
    joint_data describes each motion axis in joint index order.
    Both fields are optional so legacy positional scene construction remains
    valid. The historic joints positions still denote frame origins.
    """

    joints: list[np.ndarray]
    frames: list[FrameData]
    dimension: float
    dh_segments: list[DHSegmentData] = field(default_factory=list)
    joint_data: list[JointData] = field(default_factory=list)

    @property
    def characteristic_length(self) -> float | None:
        """Sum of the evaluated absolute DH translation lengths.

        Returns `None` when geometric segments are unavailable (e.g. a
        legacy manually constructed scene). Returns `0.0` for explicitly
        evaluated but degenerate DH geometry. No absolute unit-dependent
        minimum is imposed, and the result follows mutations to segments.
        """
        if not self.dh_segments:
            return None

        length = math.fsum(
            math.dist(segment.start, segment.end)
            for segment in self.dh_segments
        )
        if not math.isfinite(length):
            raise ValueError("DH segments must have finite geometric lengths.")
        return length
