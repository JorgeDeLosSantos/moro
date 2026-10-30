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
class SceneData:
    """Evaluated numerical data for one robot configuration.

    `dh_segments` lists two contributions per DH row in `d, a` order.
    It is optional so callers constructing legacy scenes remain supported.
    """

    joints: list[np.ndarray]
    frames: list[FrameData]
    dimension: float
    dh_segments: list[DHSegmentData] = field(default_factory=list)

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
