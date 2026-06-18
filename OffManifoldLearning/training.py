"""
End-to-end pipeline:

  1. Train MotorNet on single-channel force-matching task
     (L2 weight decay on readout layer only) -> "the patient's motor system".

  2. Collect calibration data F from this trained network; fit FA
     (k via cross-validated log-likelihood) -> L, Q (intrinsic manifold).

  3. Build D_on and D_off from L, Q.

  4. Re-purpose the SAME trained network for the center-out task:
     - its 15-D output now drives the cursor via D (D_on or D_off)
       instead of driving single-channel force
     - it receives visual error (target - cursor) instead of the
       single-channel error signal
     - starting from the single-channel-calibrated weights, allow
       continued training (fine-tuning) on the center-out task,
       and record the learning curve.

  Two independent copies of the calibrated network are fine-tuned
  separately: one with D_on, one with D_off. Comparing their learning
  curves tests whether the network's pre-existing (single-channel-shaped)
  repertoire supports faster learning under D_on than D_off.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold


# ----------------------------------------------------------------------
# 1. Motor network (same for both phases)
# ----------------------------------------------------------------------

class MotorNet(nn.Module):
    """
    Recurrent network producing a n_channels-D output.

    Generic interface: at each step, receives a task-specific input vector
    and the previous n_channels-D output, produces a new n_channels-D output
    (tanh-bounded).

    Used in two contexts:
      - Single-channel task: input = [channel one-hot (n_channels), target force (1),
        commanded-channel error (1)]
      - Center-out task: input = [visual error (2)]

    The readout layer maps hidden state -> n_channels output. It can be either
    trainable (default) or fixed to a user-supplied matrix. When fixed, only
    the GRU weights are updated during training, forcing the network to route
    control signals through the fixed random projection and naturally producing
    channel-specific co-activation (enslavement) patterns.
    """

    def __init__(self, task_input_size, n_channels=15, hidden_size=64, readout_matrix=None):
        """
        Parameters
        ----------
        task_input_size : int
            Dimensionality of the task-specific input at each time step.
        n_channels : int, optional
            Dimensionality of the network output. Default: 15.
        hidden_size : int, optional
            Number of GRU hidden units. Default: 64.
        readout_matrix : array-like of shape (n_channels, hidden_size), optional
            Fixed readout weight matrix. The readout is always frozen
            (requires_grad=False), so only the GRU is trained. When None
            (default), falls back to a rectangular identity matrix
            torch.eye(n_channels, hidden_size), which reads out the first
            n_channels hidden units directly and makes the output purely
            determined by GRU dynamics.
        """
        super().__init__()
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.task_input_size = task_input_size

        gru_input_size = task_input_size + n_channels  # + previous output
        self.gru_cell = nn.GRUCell(gru_input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, n_channels, bias=False)

        if readout_matrix is None:
            readout_matrix = torch.eye(n_channels, hidden_size)
        else:
            readout_matrix = torch.as_tensor(readout_matrix, dtype=torch.float32)
            assert readout_matrix.shape == (n_channels, hidden_size), (
                f"readout_matrix must have shape (n_channels={n_channels}, "
                f"hidden_size={hidden_size}), got {tuple(readout_matrix.shape)}"
            )

        with torch.no_grad():
            self.readout.weight.copy_(readout_matrix)
        for p in self.readout.parameters():
            p.requires_grad = False

    def init_hidden(self, batch_size):
        """
        Return a zero-filled initial hidden state.

        Parameters
        ----------
        batch_size : int
            Number of parallel trials.

        Returns
        -------
        h : torch.Tensor, shape (batch_size, hidden_size)
            Zero-initialised GRU hidden state.
        """
        return torch.zeros(batch_size, self.hidden_size, device='cpu')

    def step(self, task_input, f_prev, h):
        """
        Advance the network by one time step.

        Parameters
        ----------
        task_input : torch.Tensor, shape (batch_size, task_input_size)
            Task-specific input for the current time step (e.g., channel
            one-hot + target + error for the single-channel task, or 2-D
            visual error for the center-out task).
        f_prev : torch.Tensor, shape (batch_size, n_channels)
            Network output from the previous time step.
        h : torch.Tensor, shape (batch_size, hidden_size)
            GRU hidden state from the previous time step.

        Returns
        -------
        f_t : torch.Tensor, shape (batch_size, n_channels)
            Network output for the current time step (tanh-bounded).
        h_new : torch.Tensor, shape (batch_size, hidden_size)
            Updated GRU hidden state.
        """
        inp = torch.cat([task_input, f_prev], dim=-1)
        h_new = self.gru_cell(inp, h)
        f_t = torch.tanh(self.readout(h_new))
        return f_t, h_new


# ----------------------------------------------------------------------
# 2. Single-channel task
# ----------------------------------------------------------------------

def run_singlechannel_trial(model, channel_idx: int, target_force: float,
                            batch_size: int=20, n_steps: int=20):
    """
    Run a batch of single-channel force-matching trials, all with the same
    channel and target force.

    Parameters
    ----------
    model : MotorNet
        Recurrent network with n_channels-D output.
    channel_idx : int
        Index of the controlled channel (0-indexed). Shared across the batch.
    target_force : float
        Target force magnitude. Shared across the batch.
    batch_size : int
        Number of parallel trials to simulate.
    n_steps : int, optional
        Number of simulation time steps. Default: 20.

    Returns
    -------
    f_traj : torch.Tensor, shape (n_steps, batch_size, n_channels)
        Full network output trajectory across all time steps and batch items.
    """
    n_channels = model.n_channels

    target_force_t = torch.full((batch_size,), target_force)
    channel_onehot = (
        torch.nn.functional.one_hot(torch.tensor(channel_idx), n_channels)
        .float()
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    target_force_col = target_force_t.unsqueeze(-1)

    f_prev = torch.zeros(batch_size, n_channels, device='cpu')
    h = model.init_hidden(batch_size)

    f_traj = []
    for t in range(n_steps):
        commanded_force = f_prev[:, channel_idx]
        error = (target_force_t - commanded_force).unsqueeze(-1)
        task_input = torch.cat([channel_onehot, target_force_col, error], dim=-1)

        f_t, h = model.step(task_input, f_prev, h)
        f_traj.append(f_t)
        f_prev = f_t

    return torch.stack(f_traj, dim=0)


def train_singlechannel(model, n_epochs=500, n_steps=20, batch_size=32,
                         n_channels=15, lr=1e-3, lambda_other=1., verbose_every=50):
    """
    Train a MotorNet on the single-channel force-matching task.

    Each epoch samples a single random channel and a target force of ±1
    (drawn with equal probability), then runs a batch of identical trials.
    L2 weight decay is applied
    only to the readout layer to encourage a low-dimensional output correlation
    structure.

    Parameters
    ----------
    n_epochs : int, optional
        Number of training epochs. Default: 500.
    n_steps : int, optional
        Number of simulation time steps per trial. Default: 20.
    batch_size : int, optional
        Number of trials per epoch. Default: 32.
    n_channels : int, optional
        Dimensionality of the network output. Default: 15.
    hidden_size : int, optional
        Number of GRU hidden units. Default: 64.
    lr : float, optional
        Adam learning rate. Default: 1e-3.
    lambda_other : float, optional
        Penalty weight on non-commanded channel activations. When > 0, the
        network is penalised for activating channels other than the instructed
        one, encouraging individuation. The residual co-activation that remains
        despite this penalty is the enslavement. Default: 0.0 (no penalty).
    verbose_every : int, optional
        Print loss every this many epochs. Default: 50.

    Returns
    -------
    model : MotorNet
        Trained network (the "calibrated patient motor system").
    loss_history : list of float
        Per-epoch scalar loss values (total loss, commanded + other penalty).
    """
    n_channels = model.n_channels

    opt = torch.optim.Adam(model.gru_cell.parameters(), lr=lr)

    loss_history = []
    for epoch in range(n_epochs):
        channel_idx = int(torch.randint(0, n_channels, (1,)).item())
        target_force = float(torch.empty(1).uniform_(-1, 1).item())

        f_traj = run_singlechannel_trial(model, channel_idx, target_force, batch_size, n_steps)

        commanded = f_traj[:, :, channel_idx]
        other_mask = torch.ones(n_channels, dtype=torch.bool)
        other_mask[channel_idx] = False
        other = f_traj[:, :, other_mask]

        loss = ((commanded - target_force) ** 2).mean() + lambda_other * (other ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if (epoch + 1) % verbose_every == 0:
            print(f"  Epoch {epoch+1:4d} | commanded-channel loss = {loss.item():.4f}")

    return model, loss_history


def calibration(model, n_trials_per_channel, n_steps,
                force_levels=None):
    """
    Collect steady-state network outputs across all channels for manifold fitting.

    For each channel, runs n_trials_per_channel trials. Each trial draws a
    target force uniformly at random from a discrete set of force levels. The
    force level used on each trial is recorded and returned alongside the
    network outputs.

    Parameters
    ----------
    model : MotorNet
        Trained single-channel network (output of train_singlechannel).
    n_trials_per_channel : int
        Number of trials to run per channel.
    n_steps : int
        Number of simulation time steps per trial.
    force_levels : array-like of float, optional
        Discrete force magnitudes to sample from. Default:
        [-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0].

    Returns
    -------
    F : np.ndarray, shape (n_channels * n_trials_per_channel, n_channels)
        Matrix of final-step network outputs concatenated across all channels.
        Rows are individual trials; columns are output dimensions.
    channels : np.ndarray, shape (n_channels * n_trials_per_channel,)
        Instructed channel index for each trial.
    targets : np.ndarray, shape (n_channels * n_trials_per_channel,)
        Target force used on each trial, drawn from force_levels.
    """
    if force_levels is None:
        force_levels = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
    force_levels = np.array(force_levels)

    n_channels = model.n_channels
    all_f, all_channels, all_targets = [], [], []
    with torch.no_grad():
        for channel in range(n_channels):
            for _ in range(n_trials_per_channel):
                target_force = float(np.random.choice(force_levels))
                f_traj = run_singlechannel_trial(model, channel, target_force, 1, n_steps)
                all_f.append(f_traj[-1].cpu().numpy())
                all_channels.append(channel)
                all_targets.append(target_force)

    return (np.concatenate(all_f, axis=0),
            np.array(all_channels),
            np.array(all_targets))


# ----------------------------------------------------------------------
# 3. FA / manifold + decoder construction
# ----------------------------------------------------------------------

def crossval_fa_k(F, k_range, n_folds=4, random_state=0):
    """
    Select Factor Analysis dimensionality via cross-validated log-likelihood.

    Parameters
    ----------
    F : np.ndarray, shape (n_samples, n_features)
        Data matrix (e.g., standardised calibration outputs).
    k_range : iterable of int
        Candidate numbers of latent factors to evaluate.
    n_folds : int, optional
        Number of cross-validation folds. Default: 4.
    random_state : int, optional
        Random seed for reproducibility. Default: 0.

    Returns
    -------
    k_values : np.ndarray, shape (len(k_range),)
        Array of evaluated k values.
    mean_ll : np.ndarray, shape (len(k_range),)
        Mean held-out log-likelihood across folds for each k.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    k_values = np.array(list(k_range))
    fold_ll = np.full((n_folds, len(k_values)), np.nan)
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(F)):
        F_train, F_test = F[train_idx], F[test_idx]
        for ki, k in enumerate(k_values):
            fa = FactorAnalysis(n_components=k, random_state=random_state)
            fa.fit(F_train)
            fold_ll[fold_idx, ki] = fa.score(F_test)
    return k_values, np.nanmean(fold_ll, axis=0)


def fit_manifold(F, k):
    """
    Fit a Factor Analysis model and return the loading matrix.

    Parameters
    ----------
    F : np.ndarray, shape (n_samples, n_features)
        Data matrix (e.g., standardised calibration outputs).
    k : int
        Number of latent factors (intrinsic dimensionality).

    Returns
    -------
    L : np.ndarray, shape (n_features, k)
        Factor loading matrix. Columns span the estimated intrinsic manifold.
    """
    fa = FactorAnalysis(n_components=k, random_state=0)
    fa.fit(F)
    return fa.components_.T  # L: (n_channels, k)


def random_orthonormal(n_rows, n_cols, rng):
    """
    Generate a random matrix with orthonormal rows via QR decomposition.

    Parameters
    ----------
    n_rows : int
        Number of rows in the output (must be <= n_cols).
    n_cols : int
        Number of columns in the output.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    A : np.ndarray, shape (n_rows, n_cols)
        Matrix whose rows are orthonormal.
    """
    M = rng.normal(size=(n_cols, n_rows))
    Qm, _ = np.linalg.qr(M)
    return Qm.T


def angle_from_subspace(v, Q):
    """
    Compute the angle (in radians) between a vector and a subspace.

    Parameters
    ----------
    v : np.ndarray, shape (n,)
        Query vector.
    Q : np.ndarray, shape (n, k)
        Matrix whose columns form an orthonormal basis for the subspace.

    Returns
    -------
    angle : float
        Angle in radians between v and its projection onto the column space
        of Q. Returns 0 if v lies in the subspace, pi/2 if orthogonal.
    """
    v_par = Q @ (Q.T @ v)
    v_perp = v - v_par
    return np.arctan2(np.linalg.norm(v_perp), np.linalg.norm(v_par))


def make_D_on(A, Q, G):
    """
    Build the on-manifold decoder D_on = A @ G @ Q^T.

    Maps network output (n_channels-D) to 2-D cursor velocity by routing
    through the intrinsic manifold subspace Q.

    Parameters
    ----------
    A : np.ndarray, shape (2, k)
        Random orthonormal read-out from latent space to cursor space.
    Q : np.ndarray, shape (n_channels, k)
        Orthonormal basis of the intrinsic manifold (columns from QR of L).
    G : np.ndarray, shape (k, k)
        Gain / permutation matrix in latent space (identity for D_on).

    Returns
    -------
    D_on : np.ndarray, shape (2, n_channels)
        On-manifold decoder matrix.
    """
    return A @ G @ Q.T


def make_D_off(A, Q, G):
    """
    Build the off-manifold decoder D_off = A @ Q^T @ G.

    Maps network output to cursor velocity via a permutation G that routes
    activity outside the intrinsic manifold.

    Parameters
    ----------
    A : np.ndarray, shape (2, k)
        Random orthonormal read-out from latent space to cursor space.
    Q : np.ndarray, shape (n_channels, k)
        Orthonormal basis of the intrinsic manifold.
    G : np.ndarray, shape (n_channels, n_channels)
        Permutation matrix that scrambles the output dimensions.

    Returns
    -------
    D_off : np.ndarray, shape (2, n_channels)
        Off-manifold decoder matrix.
    """
    return A @ Q.T @ G


def search_offmanifold_permutation(A, Q, rng, n_candidates=20000,
                                    angle_threshold_deg=80):
    """
    Search for a permutation matrix G such that D_off rows are maximally
    off-manifold (large angles with respect to Q).

    Parameters
    ----------
    A : np.ndarray, shape (2, k)
        Random orthonormal read-out from latent space to cursor space.
    Q : np.ndarray, shape (n_channels, k)
        Orthonormal basis of the intrinsic manifold.
    rng : np.random.Generator
        NumPy random generator for reproducibility.
    n_candidates : int, optional
        Maximum number of random permutations to evaluate. Default: 20000.
    angle_threshold_deg : float, optional
        Stop early if the minimum row angle exceeds this value (degrees).
        Default: 80.

    Returns
    -------
    G : np.ndarray, shape (n_channels, n_channels)
        Best permutation matrix found.
    D_off : np.ndarray, shape (2, n_channels)
        Corresponding off-manifold decoder.
    angles : np.ndarray, shape (2,)
        Angle (degrees) between each row of D_off and the manifold subspace Q.
    """
    n = Q.shape[0]
    best, best_min_angle = None, -1
    for _ in range(n_candidates):
        perm = rng.permutation(n)
        G = np.eye(n)[perm]
        D_off = make_D_off(A, Q, G)
        angles = np.array([np.degrees(angle_from_subspace(D_off[i, :], Q))
                            for i in range(D_off.shape[0])])
        min_angle = angles.min()
        if min_angle > best_min_angle:
            best_min_angle, best = min_angle, (G, D_off, angles)
        if min_angle >= angle_threshold_deg:
            return G, D_off, angles
    print(f"  Warning: best permutation min angle = {best_min_angle:.1f} deg")
    return best


# ----------------------------------------------------------------------
# 4. Center-out task (reusing MotorNet, now driven by D)
# ----------------------------------------------------------------------

def run_centerout_trial(model, D, pos_target, n_steps, dt, cursor_start=None):
    """
    Run a batch of center-out reaching trials driven by decoder D.

    The network receives visual error (target - cursor position) at each step
    and produces n_channels-D output. Cursor velocity is obtained by
    projecting the output through D.

    Parameters
    ----------
    model : MotorNet
        Network adapted to the center-out task (task_input_size=2).
    D : torch.Tensor, shape (2, n_channels)
        Decoder mapping network output to 2-D cursor velocity.
    pos_target : torch.Tensor, shape (batch_size, 2)
        2-D target position for each trial in the batch.
    n_steps : int
        Number of simulation time steps.
    dt : float
        Time step size (scales velocity to position update).
    cursor_start : torch.Tensor, shape (batch_size, 2), optional
        Initial cursor position. Defaults to the origin if None.

    Returns
    -------
    cursor_traj : torch.Tensor, shape (n_steps + 1, batch_size, 2)
        Cursor position at each time step including the initial position.
    f_traj : torch.Tensor, shape (n_steps, batch_size, n_channels)
        Network output at each time step.
    """
    batch_size = pos_target.shape[0]
    n_channels = model.n_channels

    cursor = torch.zeros(batch_size, 2, device='cpu') if cursor_start is None else cursor_start.clone()
    f_prev = torch.zeros(batch_size, n_channels, device='cpu')
    h = model.init_hidden(batch_size)

    cursor_traj = [cursor]
    f_traj = []
    for t in range(n_steps):
        error = pos_target - cursor  # (batch, 2) -- this is task_input
        f_t, h = model.step(error, f_prev, h)
        velocity = f_t @ D.T
        cursor = cursor + dt * velocity
        cursor_traj.append(cursor)
        f_traj.append(f_t)
        f_prev = f_t

    return torch.stack(cursor_traj, dim=0), torch.stack(f_traj, dim=0)


def make_targets(n_targets, radius, rng):
    """
    Generate uniformly distributed 2-D target positions on a circle.

    Parameters
    ----------
    n_targets : int
        Number of targets to generate.
    radius : float
        Radius of the target circle.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    targets : np.ndarray, shape (n_targets, 2)
        2-D (x, y) target positions uniformly distributed on the circle.
    """
    theta = rng.uniform(0, 2 * np.pi, n_targets)
    return radius * np.column_stack([np.cos(theta), np.sin(theta)])


def adapt_calibrated_model_to_centerout(calibrated_model):
    """
    The calibrated model has gru_cell sized for the single-channel task's
    input (task_input_size = n_channels+1+1 = 17, plus 15 prev-output = 32).
    The center-out task's input is 2 (visual error) + 15 = 17.

    We build a new MotorNet with task_input_size=2, and copy over the
    READOUT weights (the "intrinsic repertoire" we want to preserve).
    The GRU is re-initialized (it must be, since input dimensionality
    differs) but the readout -- which is what shapes L/Q via its
    correlation structure -- carries over.
    """
    n_channels = calibrated_model.n_channels
    hidden_size = calibrated_model.hidden_size

    new_model = MotorNet(task_input_size=2, n_channels=n_channels,
                          hidden_size=hidden_size)

    # copy readout weights (preserves the emergent output correlation structure)
    new_model.readout.load_state_dict(calibrated_model.readout.state_dict())

    return new_model


def train_centerout(base_model, D, pos_target_all, n_epochs=400, n_steps=40, dt=0.25,
                     batch_size=32, lr=1e-3, f_reg=1e-3,
                     verbose_every=50, fix_readout=False):
    """
    base_model : MotorNet already adapted (task_input_size=2), with readout
                 weights copied from the calibrated single-channel model.
    D : (2, 15) decoder
    fix_readout : if True, freeze the readout layer during center-out
                   training (only GRU adapts) -- an alternative "fine-tuning"
                   regime if you want the intrinsic repertoire to stay fixed.
    """
    model = copy.deepcopy(base_model)
    D_t = torch.tensor(D, dtype=torch.float32, device='cpu')
    targets_t = torch.tensor(pos_target_all, dtype=torch.float32, device='cpu')
    n_targets = targets_t.shape[0]

    if fix_readout:
        for p in model.readout.parameters():
            p.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr=lr)

    loss_history = []
    for epoch in range(n_epochs):
        idx = torch.randint(0, n_targets, (batch_size,))
        pos_target = targets_t[idx]

        cursor_traj, f_traj = run_centerout_trial(model, D_t, pos_target, n_steps, dt)

        err = cursor_traj[1:] - pos_target.unsqueeze(0)
        pos_loss = (err ** 2).sum(dim=-1).mean()
        reg_loss = (f_traj ** 2).mean()
        loss = pos_loss + f_reg * reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_history.append(loss.item())
        if (epoch + 1) % verbose_every == 0:
            print(f"  Epoch {epoch+1:4d} | loss = {loss.item():.4f} (pos={pos_loss.item():.4f})")

    return model, loss_history


def evaluate_centerout(model, D, pos_target_all, n_steps, dt):
    """
    Evaluate a center-out model by computing final cursor distance to targets.

    Parameters
    ----------
    model : MotorNet
        Fine-tuned center-out network to evaluate.
    D : np.ndarray, shape (2, n_channels)
        Decoder matrix used during evaluation (same as during training).
    pos_target_all : np.ndarray, shape (n_targets, 2)
        All target positions to evaluate against.
    n_steps : int
        Number of simulation time steps per trial.
    dt : float
        Time step size.

    Returns
    -------
    final_dist : np.ndarray, shape (n_targets,)
        Euclidean distance from the cursor's final position to each target.
    """
    D_t = torch.tensor(D, dtype=torch.float32, device='cpu')
    targets_t = torch.tensor(pos_target_all, dtype=torch.float32, device='cpu')
    with torch.no_grad():
        cursor_traj, _ = run_centerout_trial(model, D_t, targets_t, n_steps, dt)
    final_dist = torch.norm(cursor_traj[-1] - targets_t, dim=-1).cpu().numpy()
    return final_dist


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    n_channels = 15

    # === Phase 1: calibrate "patient" via single-channel task ===
    print("Phase 1: training single-channel network (L2 on readout)...")
    calibrated_model, sf_loss = train_singlechannel(n_epochs=500, weight_decay=1e-2)

    # === Phase 2: estimate intrinsic manifold from calibrated network ===
    print("\nPhase 2: collecting calibration data and fitting FA...")
    F, channels, targets = calibration(calibrated_model, n_trials_per_channel=200, n_steps=20)
    F_z = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-8)

    k_values, mean_ll = crossval_fa_k(F_z, k_range=range(1, 11))
    eid = k_values[np.argmax(mean_ll)]
    print(f"  Estimated intrinsic dimensionality (EID): {eid}")

    L = fit_manifold(F_z, k=eid)
    Q, _ = np.linalg.qr(L)

    # === Phase 3: build decoders ===
    print("\nPhase 3: building D_on / D_off...")
    A = random_orthonormal(2, eid, rng)
    G_on = np.eye(eid)
    D_on = make_D_on(A, Q, G_on)

    G_off, D_off, off_angles = search_offmanifold_permutation(A, Q, rng)
    print(f"  Off-manifold decoder row angles from W: {off_angles} deg")

    # === Phase 4: adapt calibrated model to center-out interface ===
    print("\nPhase 4: adapting calibrated model to center-out task...")
    centerout_base_on = adapt_calibrated_model_to_centerout(calibrated_model)
    centerout_base_off = adapt_calibrated_model_to_centerout(calibrated_model)

    n_targets = 100
    radius = 4.3
    pos_target_all = make_targets(n_targets, radius, rng)

    # === Phase 5: fine-tune on center-out task, separately for D_on / D_off ===
    print("\nPhase 5a: fine-tuning under D_on...")
    model_on, loss_on = train_centerout(centerout_base_on, D_on, pos_target_all,
                                         n_epochs=400)
    dist_on = evaluate_centerout(model_on, D_on, pos_target_all, 40, 0.25)
    print(f"  Mean final distance (D_on): {dist_on.mean():.3f}")

    print("\nPhase 5b: fine-tuning under D_off...")
    model_off, loss_off = train_centerout(centerout_base_off, D_off, pos_target_all,
                                           n_epochs=400)
    dist_off = evaluate_centerout(model_off, D_off, pos_target_all, 40, 0.25)
    print(f"  Mean final distance (D_off): {dist_off.mean():.3f}")

    # === Plots ===
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(sf_loss)
    axes[0].set_title('Phase 1: single-channel calibration')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(k_values, mean_ll, 'o-', color='k')
    axes[1].axvline(eid, color='r', linestyle='--', label=f'EID={eid}')
    axes[1].set_title('Phase 2: manifold dimensionality')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('CV log-likelihood')
    axes[1].legend()

    axes[2].plot(loss_on, label='D_on')
    axes[2].plot(loss_off, label='D_off')
    axes[2].set_title('Phase 5: center-out fine-tuning')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()

    fig.tight_layout()
    fig.savefig('/mnt/user-data/outputs/full_pipeline_summary.png', dpi=150)
    plt.close(fig)

    print("\nDone.")