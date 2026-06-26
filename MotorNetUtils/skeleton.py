import torch
import motornet as mn

class NLinkArm(mn.skeleton.Skeleton):
    """Planar arm with an arbitrary number of revolute joints (a drop-in, n-link
    generalisation of MotorNet's built-in ``TwoDofArm``).

    The arm lies in the horizontal x-y plane (no gravity). It is a serial chain of
    ``n`` rigid links connected by revolute (hinge) joints. The first joint is the
    "shoulder", pinned at the origin ``(0, 0)``; each subsequent joint sits at the
    distal end of the previous link. Adding a link just means adding one more entry
    to every per-link list below — e.g. 2 links = shoulder + elbow, 3 links =
    shoulder + elbow + wrist.

    Conventions
    -----------
    * **Joint angles** ``q`` are *relative* (each joint's angle is measured from its
      parent link). The *absolute* orientation of link ``k`` is the cumulative sum
      ``phi_k = q_1 + ... + q_k``. Positive = counter-clockwise.
    * **Bodies / fixation indices** (used when attaching muscles to the effector):
      ``0`` = worldspace (fixed in the lab frame), ``1`` = first link, ``2`` = second
      link, ... ``n`` = last link.
    * **Per-link parameters** are all measured from that link's *proximal* joint and
      must each be a length-``n`` sequence:

        ====  ===========================================================
        m     link masses (kg)
        l     link lengths (m)  -- also sets reach: max reach = sum(l)
        lg    centre-of-mass distance from the proximal joint (m), 0 < lg < l
        i     link moment of inertia about its own COM (kg.m^2)
        ====  ===========================================================

    Parameters
    ----------
    m, l, lg, i : sequences of float, all length ``n``
        Per-link inertial/geometric properties (see table above).
    pos_lower_bound, pos_upper_bound : sequences of float, length ``n``
        Per-joint angle limits in **radians** (use ``np.deg2rad([...])`` for degrees).
    viscosity : float, optional
        Joint viscous damping coefficient (torque proportional to -velocity).
        Default 0 (no damping).
    name : str, optional
        Skeleton name.

    Notes
    -----
    Internally the dynamics use recursive Newton-Euler (the mass matrix is read off
    by probing it with unit accelerations; the Coriolis/centrifugal torques come from
    the same recursion with zero acceleration). Forward kinematics and the
    muscle-path geometry (moment arms) are computed analytically. For ``n == 2`` with
    ``TwoDofArm``'s parameters it reproduces ``TwoDofArm`` to ~1e-6.

    Convenience attributes ``l1, l2, ...`` (the individual link lengths) are exposed
    so plotting helpers written for ``TwoDofArm`` keep working.

    Examples
    --------
    Build a 3-link arm and wrap it in an effector, then attach a muscle::

        import numpy as np, motornet as mn
        from MotorNetUtils.skeleton import NLinkArm

        arm = NLinkArm(
            m=[1.9, 1.5, 0.5],          # upper arm, forearm, hand
            l=[0.31, 0.27, 0.15],
            lg=[0.16, 0.16, 0.07],
            i=[0.013, 0.020, 0.003],
            pos_lower_bound=np.deg2rad([0, 0, -70]),    # shoulder, elbow, wrist
            pos_upper_bound=np.deg2rad([140, 160, 70]),
        )
        effector = mn.effector.Effector(skeleton=arm, muscle=mn.muscle.ReluMuscle())

        # a muscle pulling from the worldspace (body 0) onto the first link (body 1).
        # each fixation point is [along_the_bone, perpendicular] in that bone's frame.
        effector.add_muscle(
            path_fixation_body=[0, 1],
            path_coordinates=[[0.0, 0.04], [0.12, 0.0]],
            max_isometric_force=1000.,
            name="shoulder_flexor",
        )

    Useful attributes after construction: ``arm.n`` (number of links/joints),
    ``arm.dof`` (== n), ``arm.l`` (tensor of link lengths), ``arm.l1, arm.l2, ...``.
    """

    def __init__(self, m, l, lg, i, pos_lower_bound, pos_upper_bound,
                 viscosity=0.0, name="n_link_arm"):
        n = len(l)
        super().__init__(dof=n, space_dim=2, name=name,
                         pos_lower_bound=list(pos_lower_bound),
                         pos_upper_bound=list(pos_upper_bound))
        self.n = n
        self.c_viscosity = viscosity
        for nm, val in [("m", m), ("l", l), ("lg", lg), ("i", i)]:
            self.register_buffer(nm, torch.tensor(val, dtype=torch.float32))
        # expose l1, l2, ... so the plotting helpers and other code keep working
        for k in range(n):
            setattr(self, f"l{k + 1}", float(l[k]))

    def _angles(self, q):
        """Absolute link angles phi = cumsum(q), with cos and sin."""
        phi = torch.cumsum(q, dim=1)
        return phi, torch.cos(phi), torch.sin(phi)

    def _rne(self, q, qd, qdd):
        """Recursive Newton-Euler: joint torques for (q, qd, qdd). Batched."""
        B = q.shape[0]
        phi, c, s = self._angles(q)
        e = torch.stack([c, s], dim=-1)            # unit vector of each link
        omega = torch.cumsum(qd, dim=1)            # absolute angular velocity
        alpha = torch.cumsum(qdd, dim=1)           # absolute angular acceleration

        def zcross(scal, vec):                     # z-hat (scalar) x planar vector
            return torch.stack([-scal * vec[..., 1], scal * vec[..., 0]], dim=-1)

        # forward pass: COM linear accelerations (world frame)
        a_prox = torch.zeros(B, 2)
        a_com = []
        for k in range(self.n):
            rc = self.lg[k] * e[:, k]              # prox joint -> COM
            rd = self.l[k] * e[:, k]               # prox joint -> distal joint
            w2 = (omega[:, k] ** 2)[:, None]
            a_com.append(a_prox + zcross(alpha[:, k], rc) - w2 * rc)
            a_prox = a_prox + zcross(alpha[:, k], rd) - w2 * rd

        # backward pass: forces / moments (world frame, so all rotations are identity)
        f_next = torch.zeros(B, 2)
        n_next = torch.zeros(B)
        tau = [None] * self.n
        for k in reversed(range(self.n)):
            F = self.m[k] * a_com[k]               # net force on link k
            N = self.i[k] * alpha[:, k]            # net moment about its COM
            Pci = self.lg[k] * e[:, k]             # prox joint -> COM
            Pnext = self.l[k] * e[:, k]            # prox joint -> next joint
            cross_F = Pci[:, 0] * F[:, 1] - Pci[:, 1] * F[:, 0]
            cross_fn = Pnext[:, 0] * f_next[:, 1] - Pnext[:, 1] * f_next[:, 0]
            n_k = N + n_next + cross_F + cross_fn
            tau[k] = n_k
            f_next = F + f_next
            n_next = n_k
        return torch.stack(tau, dim=1)

    def _endpoint_jacobian(self, q):
        """Hand Jacobian J (B, 2, n): d(hand_xy)/d(q)."""
        phi, c, s = self._angles(q)
        seg = self.l[None, :, None] * torch.stack([-s, c], dim=-1)
        suffix = torch.flip(torch.cumsum(torch.flip(seg, dims=[1]), dim=1), dims=[1])
        return suffix.permute(0, 2, 1)

    def _ode(self, inputs, joint_state, endpoint_load):
        q, qd = joint_state[:, :self.n], joint_state[:, self.n:]
        zeros = torch.zeros_like(q)
        # mass matrix by probing RNE with unit accelerations (qd = 0 -> no bias term)
        cols = []
        for j in range(self.n):
            ej = torch.zeros_like(q)
            ej[:, j] = 1.0
            cols.append(self._rne(q, zeros, ej))
        M = torch.stack(cols, dim=2)               # (B, n, n)
        # Coriolis / centrifugal torques with zero acceleration
        c = self._rne(q, qd, zeros) + self.c_viscosity * qd
        # external endpoint load -> joint torques
        J = self._endpoint_jacobian(q)
        tau_ext = torch.einsum("bsn,bs->bn", J, endpoint_load)
        rhs = (inputs + tau_ext - c)[:, :, None]
        return torch.linalg.solve(M, rhs)[:, :, 0]

    def _integrate(self, dt, state_derivative, joint_state):
        old_pos, old_vel = joint_state.chunk(2, dim=1)
        new_vel = old_vel + state_derivative * dt
        new_pos = old_pos + old_vel * dt
        new_vel = self.clip_velocity(new_pos, new_vel)
        new_pos = self.clip_position(new_pos)
        return torch.cat([new_pos, new_vel], dim=1)

    def _joint2cartesian(self, joint_state):
        j = torch.reshape(joint_state, (-1, self.state_dim))
        q, qd = j[:, :self.n], j[:, self.n:]
        phi, c, s = self._angles(q)
        omega = torch.cumsum(qd, dim=1)
        px, py = (self.l * c).sum(1), (self.l * s).sum(1)
        vx = (self.l * (-s) * omega).sum(1)
        vy = (self.l * c * omega).sum(1)
        return torch.stack([px, py, vx, vy], dim=1)

    def _path2cartesian(self, path_coordinates, path_fixation_body, joint_state):
        q, qd = joint_state[:, :self.n], joint_state[:, self.n:]
        B = q.shape[0]
        phi, c, s = self._angles(q)
        npts = path_fixation_body.numel()
        body = path_fixation_body.reshape(-1).long()
        u = path_coordinates.reshape(2, npts)[0]          # along bone
        v = path_coordinates.reshape(2, npts)[1]          # perpendicular

        # angle of the bone each point sits on (0 for worldspace)
        A = torch.cat([torch.zeros(B, 1), phi], dim=1)
        ang = A[:, body]
        ca, sa = torch.cos(ang), torch.sin(ang)

        # proximal-joint origin of each body (sum of links before it)
        seg = self.l[None, :, None] * torch.stack([c, s], dim=-1)
        Pcum = torch.cat([torch.zeros(B, 1, 2), torch.cumsum(seg, dim=1)], dim=1)
        origin = Pcum[:, (body - 1).clamp(min=0), :]
        origin = torch.where((body == 0)[None, :, None], 0.0, origin)

        wx = origin[:, :, 0] + u * ca - v * sa
        wy = origin[:, :, 1] + u * sa + v * ca
        xy = torch.stack([wx, wy], dim=1)

        # derivative w.r.t. each joint angle (MotorNet turns this into moment arms)
        dR = torch.stack([-u * sa - v * ca, u * ca - v * sa], dim=1)   # wrt own bone angle
        dseg = self.l[None, :, None] * torch.stack([-s, c], dim=-1)
        G = torch.cat([torch.zeros(B, 1, 2), torch.cumsum(dseg, dim=1)], dim=1)
        dxy_ddof = torch.zeros(B, 2, self.n, npts)
        for a in range(1, self.n + 1):
            mask_rot = (body >= a).float()
            mask_org = (body - 1 >= a).float()
            Gb = G[:, (body - 1).clamp(min=0), :].permute(0, 2, 1)
            Ga = G[:, a - 1, :][:, :, None]
            dxy_ddof[:, :, a - 1, :] = mask_rot * dR + mask_org * (Gb - Ga)

        omega = torch.cumsum(qd, dim=1)
        dxy_dt = torch.einsum("bsan,ba->bsn", dxy_ddof, omega)
        return xy, dxy_dt, dxy_ddof