import torch as th
import numpy as np
from motornet.effector import Effector
from motornet.muscle import ReluMuscle
from motornet.skeleton import PointMass


class FingerForceEffector(Effector):
    """3-DOF isometric finger plant.

    Models one finger as a 3D point mass driven by 6 ReluMuscle actuators
    arranged in agonist/antagonist pairs along three orthogonal axes:

      axis 0 (+x / -x) : MCP flexion  / extension
      axis 1 (+y / -y) : MCP abduction / adduction
      axis 2 (+z / -z) : PIP flexion  / extension

    Muscle index:
      0  mcp_flex  – pulls mass in +x  (MCP flexion)
      1  mcp_ext   – pulls mass in -x  (MCP extension)
      2  mcp_abd   – pulls mass in +y  (MCP abduction)
      3  mcp_add   – pulls mass in -y  (MCP adduction)
      4  pip_flex  – pulls mass in +z  (PIP flexion)
      5  pip_ext   – pulls mass in -z  (PIP extension)

    The point-mass position represents the net displacement in 3D force space.
    High damping is applied to suppress oscillation and produce overdamped,
    quasi-isometric dynamics.
    """

    def __init__(
        self,
        timestep: float = 0.01,
        max_isometric_force: float = 500.,
        mass: float = 0.05,
        damping: float = 50.,
        pos_bound: float = 1.5,
        **kwargs,
    ):
        skeleton = PointMass(space_dim=3, mass=mass)
        super().__init__(
            skeleton=skeleton,
            muscle=ReluMuscle(),
            timestep=timestep,
            damping=damping,
            pos_lower_bound=-pos_bound,
            pos_upper_bound=pos_bound,
            **kwargs,
        )

        R = 2.0  # world-attachment radius (must be > pos_bound)
        configs = [
            ([[ R, 0., 0.], [0., 0., 0.]], 'mcp_flex'),
            ([[-R, 0., 0.], [0., 0., 0.]], 'mcp_ext'),
            ([[0.,  R, 0.], [0., 0., 0.]], 'mcp_abd'),
            ([[0., -R, 0.], [0., 0., 0.]], 'mcp_add'),
            ([[0., 0.,  R], [0., 0., 0.]], 'pip_flex'),
            ([[0., 0., -R], [0., 0., 0.]], 'pip_ext'),
        ]
        for coords, name in configs:
            self.add_muscle(
                path_fixation_body=[0, 1],   # 0 = worldspace, 1 = point mass
                path_coordinates=coords,
                name=name,
                max_isometric_force=max_isometric_force,
            )
