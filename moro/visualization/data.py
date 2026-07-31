"""Numerical data containers used by the visualization backends."""

from dataclasses import dataclass

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
class SceneData:
    """Evaluated numerical data for one robot configuration."""

    joints: list[np.ndarray]
    frames: list[FrameData]
    dimension: float
