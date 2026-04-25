import torch as th
import numpy as np
from typing import Any
from motornet.environment import Environment


# 6 canonical targets: ±x (MCP flex/ext), ±y (MCP abd/add), ±z (PIP flex/ext)
_UNIT_TARGETS = th.zeros(6, 3)
for _i, (_row, _sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]):
    _UNIT_TARGETS[_i, _row] = _sign


class IsometricSingleFingerTask(Environment):
    """Single-finger isometric task.

    Each episode, one of six discrete movement directions is cued at random:

      0  MCP flexion   (+x)
      1  MCP extension (-x)
      2  MCP abduction (+y)
      3  MCP adduction (-y)
      4  PIP flexion   (+z)
      5  PIP extension (-z)

    The finger starts at rest (origin). The goal is to drive the point-mass
    to the cued target (unit_direction × target_distance) and hold it there.

    Observation: [goal_xyz (3), fingertip_xyz (3), proprioception (n_muscles × 2)]
    """

    _UNIT_TARGETS = _UNIT_TARGETS

    def __init__(self, *args, target_distance: float = 1.0, **kwargs):
        q_init = kwargs.pop('q_init', np.zeros((1, 3)))  # rest at origin
        # target_distance is needed in reset(), which _build_spaces() calls
        # during super().__init__(), so we store it via object.__setattr__
        # to bypass nn.Module's __setattr__ before its __init__ runs.
        object.__setattr__(self, 'target_distance', target_distance)
        super().__init__(*args, q_init=q_init, **kwargs)
        # goal signal is noiseless
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        self._set_generator(seed=seed)
        options = {} if options is None else options
        batch_size: int = options.get('batch_size', 1)
        joint_state = options.get('joint_state', None)
        deterministic: bool = options.get('deterministic', False)

        if joint_state is not None:
            if np.shape(self.detach(joint_state))[0] > 1:
                batch_size = np.shape(self.detach(joint_state))[0]
        else:
            joint_state = self.q_init  # always reset to rest

        # Effector.reset requires a tensor (MotorNet v0.2.0 stores q_init as numpy)
        if joint_state is not None and not th.is_tensor(joint_state):
            joint_state = th.tensor(joint_state, dtype=th.float32)

        self.effector.reset(options={"batch_size": batch_size, "joint_state": joint_state})

        # sample one random direction per trial in the batch
        if deterministic:
            idx = th.zeros(batch_size, dtype=th.long)  # always MCP flex for deterministic reset
        else:
            idx = th.randint(0, 6, (batch_size,))

        self.goal = (
            self._UNIT_TARGETS[idx] * self.target_distance
        ).to(self.device)

        self.elapsed = 0.
        action = th.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(self.obs_buffer["proprioception"])
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)
        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
        return obs, info
