import numpy as np
from motornet.skeleton import TwoDofArm


class FingerSkeleton(TwoDofArm):
    """2-DOF planar finger (MCP + PIP joints).

    Joint angle convention: 0 = fully extended, positive = flexion.
    pos0 = MCP angle, pos1 = PIP angle relative to proximal phalanx.

    Joint limits are set by the enclosing Effector (RigidTendonFinger), not here,
    because TwoDofArm hardcodes its own limits before forwarding to Skeleton.
    Default parameters approximate the index finger.
    """

    def __init__(self, name='finger_skeleton', **kwargs):
        l1 = kwargs.pop('l1', 0.043)   # proximal phalanx length [m]
        l2 = kwargs.pop('l2', 0.026)   # middle phalanx length [m]
        m1 = kwargs.pop('m1', 0.004)   # proximal phalanx mass [kg]
        m2 = kwargs.pop('m2', 0.0025)  # middle phalanx mass [kg]
        i1 = kwargs.pop('i1', m1 * l1 ** 2 / 3)  # moment of inertia [kg·m²]
        i2 = kwargs.pop('i2', m2 * l2 ** 2 / 3)

        # Do NOT pass pos_lower/upper_bound here — TwoDofArm sets them internally
        # and would conflict. RigidTendonFinger passes the correct finger limits
        # to Effector.__init__, which propagates them via skeleton.build().
        super().__init__(
            name=name,
            m1=m1, m2=m2,
            l1g=l1 / 2, l2g=l2 / 2,
            i1=i1, i2=i2,
            l1=l1, l2=l2,
            **kwargs,
        )
