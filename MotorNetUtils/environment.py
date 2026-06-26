from typing import Any

import numpy as np
import torch
import motornet as mn


class CentreOutReach(mn.environment.Environment):
    """Centre-out reaching task: the arm always starts from a fixed central posture and
    must reach a target placed at a fixed radius around the hand's starting position. By
    default the target direction is random (uniform on the circle) at every episode; pass
    ``angles`` to instead draw from a fixed set of directions.

    This is the classic "centre-out" paradigm. Compared with MotorNet's
    ``RandomTargetReach`` (random start, random target), here the start is fixed and the
    target lies on a ring of fixed radius, which makes it easy to analyse reaches by
    direction.

    What you can control
    --------------------
    radius : float
        Distance (m) from the central start position to every target.
    angles : sequence of float or None
        The set of target directions to draw from, as angles in **radians** (0 = +x axis,
        CCW positive). If ``None`` (default), the target direction is drawn from a
        *continuous* uniform distribution over the full circle ``[0, 2*pi)`` at every
        episode.
    start_joint : sequence of float or None
        Central posture in **joint space** (radians, length = n_dof). The hand's starting
        Cartesian position is the forward kinematics of this posture. If ``None``, the
        midpoint of each joint's limits is used.
    vision : bool
        If ``True`` the observation includes visual feedback (the fingertip x-y position).
    proprioception : bool
        If ``True`` the observation includes proprioceptive feedback (normalised muscle
        length and velocity for every muscle). At least one of ``vision`` /
        ``proprioception`` must be ``True``.
    vision_delay, proprioception_delay : float or None
        Feedback delay (seconds) for each modality. Must be an integer multiple of the
        effector timestep ``dt``. ``None`` means a single-timestep delay (the default).
    vision_noise, proprioception_noise : float
        Std. dev. of i.i.d. Gaussian noise added to each feedback channel.

    The observation vector is always ``[goal, (vision), (proprioception)]`` where the
    bracketed parts are present only if their flag is ``True`` (the goal / target is always
    given and is noiseless). ``observation_space`` is sized automatically from the flags.

    Notes
    -----
    Pick ``radius`` so the whole target ring stays inside the reachable workspace
    (``< sum(link lengths)``) and within the joint limits, otherwise some targets are
    physically unreachable.

    Examples
    --------
    ::

        import numpy as np
        from MotorNetUtils.environment import CentreOutReach

        # random target direction each episode:
        env = CentreOutReach(effector, radius=0.10,
                             vision=True, proprioception=True,
                             vision_delay=0.05, proprioception_delay=0.02)

        # or draw from a fixed 8-direction set:
        env = CentreOutReach(effector, radius=0.10,
                             angles=np.linspace(0, 2 * np.pi, 8, endpoint=False))

        obs, info = env.reset(options={"batch_size": 32})
        # reach specific directions (e.g. for evaluation / plotting a fan of targets):
        obs, info = env.reset(options={"batch_size": 8,
                                       "target_angle": np.linspace(0, 2 * np.pi, 8, endpoint=False)})
    """

    def __init__(self, effector, *, radius: float = 0.10, angles=None,
                 start_joint=None, vision: bool = True, proprioception: bool = True,
                 vision_delay: float = None, proprioception_delay: float = None,
                 vision_noise: float = 0.0, proprioception_noise: float = 0.0,
                 name: str = "CentreOutReach", **kwargs):
        # these must exist before super().__init__ (it calls _get_obs_size)
        self.vision_on = bool(vision)
        self.proprioception_on = bool(proprioception)
        if not (self.vision_on or self.proprioception_on):
            raise ValueError("at least one of `vision` / `proprioception` must be True.")
        self.radius = float(radius)
        self.angles = None if angles is None else np.asarray(angles, dtype=float).reshape(-1)

        super().__init__(
            effector, name=name, q_init=start_joint,
            vision_delay=vision_delay, proprioception_delay=proprioception_delay,
            vision_noise=vision_noise, proprioception_noise=proprioception_noise,
            **kwargs,
        )

        # default central posture = midpoint of the joint limits
        if self.q_init is None:
            lb = self.effector.skeleton.pos_lower_bound
            ub = self.effector.skeleton.pos_upper_bound
            self.q_init = ((lb + ub) / 2).detach().cpu().numpy().reshape(1, -1)
            self.nq_init = 1

        # the target / goal is given to the controller noiselessly
        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim

    def _get_obs_size(self) -> int:
        """Observation size given the active feedback channels (overrides the base so
        ``observation_space`` matches the toggled feedback)."""
        size = self.skeleton.space_dim                       # goal / target (always)
        if self.vision_on:
            size += self.skeleton.space_dim                  # fingertip x-y
        if self.proprioception_on:
            size += 2 * self.effector.n_muscles              # muscle length + velocity
        size += self.effector.n_muscles * self.action_frame_stacking
        return size

    def get_obs(self, action: torch.Tensor | None = None,
                deterministic: bool = False) -> torch.Tensor:
        """Assemble the observation as ``[goal, (vision), (proprioception), (actions)]``,
        including only the feedback channels that are switched on."""
        self.update_obs_buffer(action=action)

        obs_as_list = [self.goal]
        if self.vision_on:
            obs_as_list.append(self.obs_buffer["vision"][0])
        if self.proprioception_on:
            obs_as_list.append(self.obs_buffer["proprioception"][0])
        obs_as_list += self.obs_buffer["action"][:self.action_frame_stacking]

        obs = torch.cat(obs_as_list, dim=-1)
        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs if self.differentiable else self.detach(obs)

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Start every trial from the fixed central posture and place the target on the
        radial ring.

        Extra ``options`` keys (besides ``batch_size`` / ``deterministic``):
          ``target_angle`` : float or sequence of float, optional
              Explicit target direction(s) in radians to use this episode (overrides the
              random / fixed-set sampling). A scalar is applied to the whole batch.
        """
        self._set_generator(seed=seed)

        options = {} if options is None else options
        batch_size: int = options.get("batch_size", 1)
        deterministic: bool = options.get("deterministic", False)
        target_angle = options.get("target_angle", None)

        # always start from the fixed centre posture (effector.reset needs a tensor)
        q0 = torch.as_tensor(self.q_init, dtype=torch.float32, device=self.device)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": q0})
        self.elapsed = 0.

        # choose a target direction per trial, then place the target at `radius`
        if target_angle is not None:                         # explicit angles requested
            angle = np.broadcast_to(np.asarray(target_angle, dtype=float), (batch_size,)).copy()
        elif self.angles is not None:                        # draw from the fixed set
            angle = self.angles[self.np_random.integers(0, self.angles.size, size=batch_size)]
        else:                                                # continuous random direction
            angle = self.np_random.uniform(0., 2. * np.pi, size=batch_size)
        offset = np.stack([np.cos(angle), np.sin(angle)], axis=-1) * self.radius
        centre = self.states["fingertip"].detach()          # (batch, 2), same for all trials
        self.goal = centre + torch.as_tensor(offset, dtype=torch.float32, device=self.device)

        action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)

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
            "angle": torch.as_tensor(np.asarray(angle), dtype=torch.float32, device=self.device),
        }
        return obs, info


# ----------------------------------------------------------------------------
# Handwriting task
# ----------------------------------------------------------------------------

# A minimal single-stroke vector font. Each glyph is ONE polyline (list of
# (x, y) points) drawn without lifting the pen, in a unit box where the
# baseline is y = 0 and the cap height is y = 1 (x starts at 0). Letters are
# deliberately stylised so they can be written in a single continuous stroke;
# consecutive letters (and words) are joined by straight "ligature" segments.
FONT = {
    "A": [(0, 0), (0.4, 1), (0.8, 0), (0.62, 0.36), (0.18, 0.36)],
    "C": [(0.78, 0.78), (0.45, 1), (0.1, 0.78), (0, 0.4), (0.2, 0.08), (0.55, 0), (0.8, 0.18)],
    "D": [(0, 0), (0, 1), (0.45, 1), (0.75, 0.7), (0.75, 0.3), (0.45, 0), (0, 0)],
    "E": [(0.75, 1), (0, 1), (0, 0.5), (0.55, 0.5), (0, 0.5), (0, 0), (0.78, 0)],
    "G": [(0.78, 0.78), (0.45, 1), (0.1, 0.78), (0, 0.4), (0.2, 0.08), (0.55, 0),
          (0.78, 0.18), (0.78, 0.45), (0.5, 0.45)],
    "H": [(0, 0), (0, 1), (0, 0.5), (0.75, 0.5), (0.75, 1), (0.75, 0)],
    "I": [(0.1, 1), (0.6, 1), (0.35, 1), (0.35, 0), (0.1, 0), (0.6, 0)],
    "L": [(0, 1), (0, 0), (0.7, 0)],
    "M": [(0, 0), (0, 1), (0.4, 0.35), (0.8, 1), (0.8, 0)],
    "N": [(0, 0), (0, 1), (0.75, 0), (0.75, 1)],
    "O": [(0.4, 0), (0.08, 0.3), (0.08, 0.7), (0.4, 1), (0.72, 0.7), (0.72, 0.3), (0.4, 0)],
    "P": [(0, 0), (0, 1), (0.6, 1), (0.7, 0.7), (0.6, 0.5), (0, 0.5)],
    "R": [(0, 0), (0, 1), (0.6, 1), (0.7, 0.72), (0.55, 0.5), (0, 0.5), (0.7, 0)],
    "S": [(0.75, 0.82), (0.4, 1), (0.08, 0.82), (0.4, 0.5), (0.72, 0.32), (0.4, 0), (0.08, 0.18)],
    "T": [(0, 1), (0.7, 1), (0.35, 1), (0.35, 0)],
    "U": [(0, 1), (0, 0.25), (0.35, 0), (0.7, 0.25), (0.7, 1)],
    "V": [(0, 1), (0.35, 0), (0.7, 1)],
    "W": [(0, 1), (0.18, 0), (0.35, 0.6), (0.52, 0), (0.7, 1)],
    "Y": [(0, 1), (0.35, 0.5), (0.7, 1), (0.35, 0.5), (0.35, 0)],
}


def word_polyline(word: str, size: float = 1.0, letter_spacing: float = 0.3) -> np.ndarray:
    """Build the continuous stroke path that spells ``word``.

    Args:
      word: the text to write (letters from ``FONT``; spaces allowed, case-insensitive).
      size: cap height of the letters, in metres.
      letter_spacing: horizontal gap between letters, as a fraction of ``size``.

    Returns:
      An ``(N, 2)`` array of x-y points (metres), baseline at y = 0, starting at x = 0.
    """
    parts, x = [], 0.0
    for ch in word:
        if ch == " ":
            x += 0.6 + letter_spacing
            continue
        key = ch.upper()
        if key not in FONT:
            raise ValueError(f"character {ch!r} is not in the font; "
                             f"supported: {' '.join(sorted(FONT))} and space")
        p = np.asarray(FONT[key], dtype=float).copy()
        p[:, 0] += x
        parts.append(p)
        x += p[:, 0].max() - x + letter_spacing       # advance by this glyph's width + gap
    return np.concatenate(parts, axis=0) * size


def resample_polyline(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample a polyline to ``n`` points spaced evenly along its arc length
    (so the pen moves at roughly constant speed)."""
    seg = np.sqrt((np.diff(poly, axis=0) ** 2).sum(1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(poly[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, poly[:, 0]), np.interp(u, s, poly[:, 1])], axis=1)


class Handwriting(mn.environment.Environment):
    """Handwriting task: the hand must trace out the stroke path that spells a word.

    The target ("goal") is a point that moves, at constant speed, along the word's
    stroke path over the course of the episode; minimising the hand-to-target distance
    therefore makes the hand *write the word*. The word is centred (by its bounding box)
    on the hand's starting position and scaled to ``size``.

    What you can control
    --------------------
    words : str or sequence of str
        The word(s) to write. If several are given, one is drawn at random per trial
        (each episode). Pass ``options={"word": ...}`` to ``reset`` to force a specific
        word. Letters must be in ``FONT`` (A-Z subset, case-insensitive); spaces allowed.
    size : float
        Cap height of the letters (m). Keep the whole word inside the reachable workspace.
    letter_spacing : float
        Gap between letters, as a fraction of ``size``.
    origin : (x, y) or None
        Where the word's bounding-box centre is placed (m). If ``None`` (default) it is
        the hand's position at the start posture.
    start_joint : sequence of float or None
        Starting posture in joint space (radians). If ``None``, the midpoint of the joint
        limits is used.
    vision, proprioception : bool
        Whether each feedback modality is included in the observation (see CentreOutReach).
    vision_delay, proprioception_delay : float or None
        Feedback delays in seconds (integer multiples of the timestep ``dt``).
    vision_noise, proprioception_noise : float
        Feedback noise std. devs.

    The observation is ``[goal, (vision), (proprioception)]`` where ``goal`` is the *current*
    point on the word to be traced (always provided, noiseless).

    Examples
    --------
    ::

        from MotorNetUtils.environment import Handwriting
        env = Handwriting(effector, words="hi", size=0.07)
        obs, info = env.reset(options={"batch_size": 32})
        env.reference          # (batch, n_steps + 1, 2): the full target word per trial
        obs, info = env.reset(options={"batch_size": 1, "word": "draw"})
    """

    def __init__(self, effector, *, words="hi", size: float = 0.07,
                 letter_spacing: float = 0.3, origin=None, start_joint=None,
                 vision: bool = True, proprioception: bool = True,
                 vision_delay: float = None, proprioception_delay: float = None,
                 vision_noise: float = 0.0, proprioception_noise: float = 0.0,
                 name: str = "Handwriting", **kwargs):
        self.vision_on = bool(vision)
        self.proprioception_on = bool(proprioception)
        if not (self.vision_on or self.proprioception_on):
            raise ValueError("at least one of `vision` / `proprioception` must be True.")
        self.words = [words] if isinstance(words, str) else list(words)
        self.size = float(size)
        self.letter_spacing = float(letter_spacing)
        self._origin = None if origin is None else np.asarray(origin, dtype=float).reshape(2)

        super().__init__(
            effector, name=name, q_init=start_joint,
            vision_delay=vision_delay, proprioception_delay=proprioception_delay,
            vision_noise=vision_noise, proprioception_noise=proprioception_noise,
            **kwargs,
        )

        # number of step() calls per episode (the reference has one more: the start point)
        self.n_steps = int(round(self.max_ep_duration / self.dt))

        if self.q_init is None:
            lb = self.effector.skeleton.pos_lower_bound
            ub = self.effector.skeleton.pos_upper_bound
            self.q_init = ((lb + ub) / 2).detach().cpu().numpy().reshape(1, -1)
            self.nq_init = 1

        self.obs_noise[:self.skeleton.space_dim] = [0.] * self.skeleton.space_dim
        self.reference = None

    def _get_obs_size(self) -> int:
        size = self.skeleton.space_dim                       # goal / current target point
        if self.vision_on:
            size += self.skeleton.space_dim
        if self.proprioception_on:
            size += 2 * self.effector.n_muscles
        size += self.effector.n_muscles * self.action_frame_stacking
        return size

    def get_obs(self, action: torch.Tensor | None = None,
                deterministic: bool = False) -> torch.Tensor:
        self.update_obs_buffer(action=action)
        obs_as_list = [self.goal]
        if self.vision_on:
            obs_as_list.append(self.obs_buffer["vision"][0])
        if self.proprioception_on:
            obs_as_list.append(self.obs_buffer["proprioception"][0])
        obs_as_list += self.obs_buffer["action"][:self.action_frame_stacking]
        obs = torch.cat(obs_as_list, dim=-1)
        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)
        return obs if self.differentiable else self.detach(obs)

    def _build_reference(self, batch_size, origin, word_opt):
        """Per-trial target trajectories (batch, n_steps + 1, 2) and the chosen words."""
        T = self.n_steps + 1
        refs, chosen = [], []
        for b in range(batch_size):
            if word_opt is None:
                w = self.words[self.np_random.integers(0, len(self.words))]
            else:
                w = word_opt if isinstance(word_opt, str) else word_opt[b]
            poly = word_polyline(w, self.size, self.letter_spacing)
            poly = poly - (poly.min(0) + poly.max(0)) / 2 + origin[b]   # centre bbox on origin
            refs.append(resample_polyline(poly, T))
            chosen.append(w)
        ref = torch.as_tensor(np.stack(refs, 0), dtype=torch.float32, device=self.device)
        return ref, chosen

    def reset(self, *, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Start from the fixed posture and build the target word trajectory.

        Extra ``options`` keys (besides ``batch_size`` / ``deterministic``):
          ``word`` : str or sequence of str, optional
              Force the word(s) to write this episode instead of sampling from ``words``.
        """
        self._set_generator(seed=seed)
        options = {} if options is None else options
        batch_size: int = options.get("batch_size", 1)
        deterministic: bool = options.get("deterministic", False)
        word_opt = options.get("word", None)

        q0 = torch.as_tensor(self.q_init, dtype=torch.float32, device=self.device)
        self.effector.reset(options={"batch_size": batch_size, "joint_state": q0})
        self.elapsed = 0.

        centre = self.states["fingertip"].detach().cpu().numpy()        # (batch, 2)
        origin = centre if self._origin is None else np.tile(self._origin, (batch_size, 1))
        self.reference, self.words_this_episode = self._build_reference(batch_size, origin, word_opt)
        self._idx = 0
        self.goal = self.reference[:, 0, :]

        action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)
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
            "reference": self.reference if self.differentiable else self.detach(self.reference),
        }
        return obs, info

    def step(self, action, deterministic: bool = False, **kwargs):
        """Advance the moving target one point along the word, then step the effector."""
        self._idx = min(self._idx + 1, self.n_steps)
        self.goal = self.reference[:, self._idx, :]
        return super().step(action, deterministic=deterministic, **kwargs)


# ----------------------------------------------------------------------------
# Two-plant (dyadic) coordination task
# ----------------------------------------------------------------------------

class _DelayLine:
    """A fixed-length delay buffer. ``push(x)`` stores ``x`` and returns the value
    from ``n_steps`` steps ago (the buffer is seeded with the initial value on reset,
    so it is differentiable and well-defined from t = 0)."""

    def __init__(self, n_steps: int):
        self.n = int(n_steps)
        self.buf = None

    def reset(self, value):
        self.buf = [value] * (self.n + 1)
        return self.buf[0]

    def peek(self):
        return self.buf[0]

    def push(self, value):
        self.buf.append(value)
        self.buf.pop(0)
        return self.buf[0]


class DyadCoordination:
    """Two arms ("plants"), each driven by its own controller, learning to coordinate.

    Inspired by Tomassini et al. (2022, iScience): two effectors perform a paced
    left-right (x-axis) oscillation and must keep a target relative phase -- moving
    **in-phase** (same direction) or **anti-phase** (opposite directions). A shared
    metronome sets the pace; the coordination mode sets the relative phase.

    Each controller observes:
      - the coordination mode (+1 in-phase / -1 anti-phase) and the metronome clock
        (cos, sin of the pace phase),
      - its **own proprioception** (normalised muscle length & velocity), delayed,
      - its **own visual feedback** (its endpoint x-y), delayed,
      - the **partner's visual feedback** (the other arm's endpoint x-y), delayed.

    The three feedback delays are set independently (and can differ between own and
    partner vision, mirroring the paper's manipulations of visuomotor delay).

    This class is *not* a MotorNet ``Environment`` subclass: it wraps **two** effectors
    and returns a ``(obs_a, obs_b)`` tuple from ``reset`` / ``step`` and takes an
    ``(action_a, action_b)`` tuple in ``step``. The whole rollout is differentiable.

    Args:
      effector_a, effector_b: two built MotorNet effectors (typically identical).
      freq: metronome frequency (Hz). Either a scalar (fixed) or a ``(low, high)``
        range sampled uniformly per trial at each reset.
      amp: oscillation amplitude (m) along x. Scalar or ``(low, high)`` range (per trial).
      max_ep_duration: episode length (s). Scalar or ``(low, high)`` range (sampled once
        per reset, shared by the whole batch). The sampled ``freq`` and ``amp`` are added
        to each controller's observation (the clock alone carries phase, not amplitude).
      modes: which coordination modes to sample per episode -- any of ``"in"`` /
        ``"anti"`` (string or sequence). Each trial in the batch draws one at random.
      proprioception_delay, own_vision_delay, partner_vision_delay: feedback delays (s),
        rounded to the nearest multiple of the timestep ``dt``.
      feedback_noise: std of Gaussian noise added to every feedback channel.
      start_joint_a, start_joint_b: start postures (rad); default = midpoint of limits.

    Example
    -------
    ::

        env = DyadCoordination(eff_a, eff_b, freq=1.0, amp=0.06, max_ep_duration=3.0,
                               proprioception_delay=0.02, own_vision_delay=0.05,
                               partner_vision_delay=0.10, modes=("in", "anti"))
        (oa, ob), info = env.reset(options={"batch_size": 32})
        (oa, ob), terminated, info = env.step((action_a, action_b))
    """

    def __init__(self, effector_a, effector_b, *, freq=1.0, amp=0.06,
                 max_ep_duration=3.0, modes=("in", "anti"),
                 proprioception_delay=0.0, own_vision_delay=0.0, partner_vision_delay=0.0,
                 feedback_noise=0.0, start_joint_a=None, start_joint_b=None):
        self.eff = (effector_a, effector_b)
        self.dt = effector_a.dt
        # freq / amp / max_ep_duration may each be a scalar (fixed) or a (low, high)
        # range that is sampled uniformly at every reset. freq & amp are sampled per
        # trial in the batch; the episode duration is shared (all trials end together).
        self.freq_range = self._as_range(freq)
        self.amp_range = self._as_range(amp)
        self.dur_range = self._as_range(max_ep_duration)
        self.max_ep_duration = self.dur_range[1]
        self.feedback_noise = float(feedback_noise)
        self.device = torch.device("cpu")

        self.modes = [modes] if isinstance(modes, str) else list(modes)
        for m in self.modes:
            if m not in ("in", "anti"):
                raise ValueError(f"mode must be 'in' or 'anti', got {m!r}")

        steps = lambda d: int(round(d / self.dt))
        self._d_prop = steps(proprioception_delay)
        self._d_self_vis = steps(own_vision_delay)
        self._d_partner_vis = steps(partner_vision_delay)

        self._start = [start_joint_a, start_joint_b]
        for k in range(2):
            if self._start[k] is None:
                lb = self.eff[k].skeleton.pos_lower_bound
                ub = self.eff[k].skeleton.pos_upper_bound
                self._start[k] = ((lb + ub) / 2).detach().cpu().numpy().reshape(1, -1)

        self.n_muscles = (effector_a.n_muscles, effector_b.n_muscles)
        # obs = mode(1) + freq(1) + amp(1) + clock(2) + own proprio(2*nm)
        #       + own vision(2) + partner vision(2)
        self.obs_dim = tuple(1 + 1 + 1 + 2 + 2 * nm + 2 + 2 for nm in self.n_muscles)
        self.elapsed = None

    @staticmethod
    def _as_range(x):
        """Return a (low, high) tuple from either a scalar or a (low, high) sequence."""
        if isinstance(x, (tuple, list, np.ndarray)):
            return float(x[0]), float(x[1])
        return float(x), float(x)

    # -- feedback helpers ---------------------------------------------------
    def _proprio(self, k):
        ms = self.eff[k].states["muscle"]                       # (B, n_feat, n_musc)
        l0 = getattr(self.eff[k].muscle, "l0_ce", 1.0)
        vmax = getattr(self.eff[k].muscle, "vmax", 1.0)
        mlen = ms[:, 1:2, :] / l0
        mvel = ms[:, 2:3, :] / vmax
        return torch.cat([mlen, mvel], dim=-1).squeeze(1)       # (B, 2*n_musc)

    def _endpoint(self, k):
        return self.eff[k].states["fingertip"]                 # (B, 2)

    def _noise(self, x):
        if self.feedback_noise <= 0:
            return x
        return x + self.feedback_noise * torch.randn_like(x)

    def _clock(self, batch_size):
        phase = 2.0 * np.pi * self.freq * self.elapsed         # (batch,) tensor
        return phase, torch.cos(phase)[:, None], torch.sin(phase)[:, None]

    def _targets(self, batch_size):
        """Per-arm endpoint targets implied by the clock + coordination mode."""
        phase = 2.0 * np.pi * self.freq * self.elapsed         # (batch,)
        d = self.amp * torch.sin(phase)                        # arm-A displacement (batch,)
        zero = torch.zeros_like(d)
        tgt_a = self.center[0] + torch.stack([d, zero], dim=-1)
        tgt_b = self.center[1] + torch.stack([self.mode_sign[:, 0] * d, zero], dim=-1)
        return tgt_a, tgt_b

    def _observe(self, batch_size, push):
        """Build (obs_a, obs_b). ``push`` decides whether the delay lines advance."""
        _, cosp, sinp = self._clock(batch_size)
        freq_col, amp_col = self.freq[:, None], self.amp[:, None]
        prop = [self._noise(self._proprio(0)), self._noise(self._proprio(1))]
        endp = [self._noise(self._endpoint(0)), self._noise(self._endpoint(1))]

        def line(name, value):
            dl = self._lines[name]
            return dl.push(value) if push else dl.peek()

        obs = []
        for k, other in [(0, 1), (1, 0)]:
            own_prop = line(f"prop{k}", prop[k])
            own_vis = line(f"selfvis{k}", endp[k])
            partner_vis = line(f"partnervis{k}", endp[other])
            obs.append(torch.cat([self.mode_sign, freq_col, amp_col, cosp, sinp,
                                   own_prop, own_vis, partner_vis], dim=-1))
        return obs[0], obs[1]

    # -- API ----------------------------------------------------------------
    def reset(self, options=None):
        options = {} if options is None else options
        batch_size = options.get("batch_size", 1)
        mode_opt = options.get("mode", None)

        for k in range(2):
            q0 = torch.as_tensor(self._start[k], dtype=torch.float32, device=self.device)
            self.eff[k].reset(options={"batch_size": batch_size, "joint_state": q0})
        self.center = [self._endpoint(0).detach(), self._endpoint(1).detach()]
        self.elapsed = 0.0

        # sample this episode's task parameters (duration shared; freq & amp per trial)
        self.max_ep_duration = float(np.random.uniform(*self.dur_range))
        f = np.random.uniform(self.freq_range[0], self.freq_range[1], size=batch_size)
        a = np.random.uniform(self.amp_range[0], self.amp_range[1], size=batch_size)
        self.freq = torch.tensor(f, dtype=torch.float32, device=self.device)
        self.amp = torch.tensor(a, dtype=torch.float32, device=self.device)

        # coordination mode per trial: +1 in-phase, -1 anti-phase
        if mode_opt is None:
            pick = np.random.randint(0, len(self.modes), size=batch_size)
            signs = np.array([1.0 if self.modes[i] == "in" else -1.0 for i in pick])
        else:
            signs = np.full(batch_size, 1.0 if mode_opt == "in" else -1.0)
        self.mode_sign = torch.tensor(signs, dtype=torch.float32, device=self.device)[:, None]

        # seed the delay lines with the initial feedback values
        self._lines = {}
        for k, other in [(0, 1), (1, 0)]:
            self._lines[f"prop{k}"] = _DelayLine(self._d_prop)
            self._lines[f"prop{k}"].reset(self._proprio(k))
            self._lines[f"selfvis{k}"] = _DelayLine(self._d_self_vis)
            self._lines[f"selfvis{k}"].reset(self._endpoint(k))
            self._lines[f"partnervis{k}"] = _DelayLine(self._d_partner_vis)
            self._lines[f"partnervis{k}"].reset(self._endpoint(other))

        obs_a, obs_b = self._observe(batch_size, push=False)
        return (obs_a, obs_b), self._info(batch_size)

    def step(self, actions):
        action_a, action_b = actions
        batch_size = action_a.shape[0]
        self.eff[0].step(action_a)
        self.eff[1].step(action_b)
        self.elapsed += self.dt
        obs_a, obs_b = self._observe(batch_size, push=True)
        terminated = bool(self.elapsed >= self.max_ep_duration - 1e-9)
        return (obs_a, obs_b), terminated, self._info(batch_size)

    def _info(self, batch_size):
        tgt_a, tgt_b = self._targets(batch_size)
        return {
            "endpoint_a": self._endpoint(0), "endpoint_b": self._endpoint(1),
            "target_a": tgt_a, "target_b": tgt_b,
            "center_a": self.center[0], "center_b": self.center[1],
            "mode_sign": self.mode_sign, "freq": self.freq, "amp": self.amp,
        }
