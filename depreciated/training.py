def train_controller(A0: np.ndarray,
                     B: np.ndarray,
                     W: np.ndarray,
                     sigma_u: float = .1,  # action noise for u (exploration)
                     eta_w: float = 1e-3,  # GD learning rate for W_pol
                     eta_a: float = 1e-5,  # RL learning rate for s
                     sigma_s: float = 0.02,  # exploration noise for s (trial-level)
                     beta: float = 0.02,  # reward-baseline update rate
                     lam_a: float = 1e-2,  # regularize s toward 1
                     n_trials: int = 1000,
                     ang: float | list | np.ndarray = None,
                     radius: float = 1.,
                     maxT: int = 1000,
                     dt: float = .01,
                     tol: float = .001,
                     sim_rehab: bool = False):
    rng = np.random.default_rng()

    if ang is None:
        ang = rng.uniform(0, 2 * np.pi, size=n_trials)

    N, K = A0.shape

    # force -> velocity mapping
    P = W @ B.T  # (2, N)

    # plant scaling parameters (we RL-learn these)
    s = np.ones(K, dtype=float)

    # policy parameters (we GD-learn these)
    W_pol = np.zeros((K, 2), dtype=float)

    # logging
    success = np.zeros((n_trials,), dtype=bool)
    nsteps = np.zeros(n_trials, dtype=int)
    loss = np.zeros(n_trials, dtype=float)
    meanDev = np.zeros_like(loss)
    velMax = np.zeros_like(loss)

    # RL baseline (for variance reduction)
    baseline = 0.0
    baseline_init = False

    for tr in range(n_trials):

        # target
        pos_star = radius * np.array([np.cos(ang[tr]), np.sin(ang[tr])], dtype=float)

        # --- RL: sample a plant for this trial (keep fixed during the trial) ---
        if sim_rehab:
            eps_s = rng.standard_normal(K)  # exploration direction
            s_trial = s + sigma_s * eps_s  #np.clip(s + sigma_s * eps_s, s_min, s_max)
        else:
            eps_s = None
            s_trial = s

        # state
        u = np.zeros((K,), dtype=float)
        pos = np.zeros((2,), dtype=float)
        dev = 0.0
        velMax_tr = 0.0
        current_loss = np.inf

        for t in range(maxT):
            print(f'doing trial {tr + 1} of {n_trials}, loss = {current_loss}')

            # plant for this trial
            A = A0 * s_trial[None, :]  # (N, K)

            # u -> force -> velocity -> position
            f = A @ u  # (N,)
            vel = P @ f  # (2,)
            pos = pos + dt * vel  # (2,)

            # error
            e = pos_star - pos  # (2,)

            # --- policy: sample action u around mean mu = W_pol @ e ---
            mu = W_pol @ e  # (K,)
            u = mu + sigma_u * rng.standard_normal(K)  # (K,)

            # --- GD update for W_pol (model-based) ---
            # d pos / d u = dt * (P @ A)
            J_u = dt * (P @ A)  # (2, K)
            g_u = J_u.T @ e  # (K,)  (descending loss uses +outer(g_u,e) given your convention)
            W_pol += eta_w * np.outer(g_u, e)  # (K,2)

            # metrics
            current_loss = np.linalg.norm(e)
            vel_mod = np.linalg.norm(vel)
            if vel_mod > velMax_tr:
                velMax_tr = vel_mod

            alpha = (pos @ pos_star) / ((pos_star @ pos_star) + 1e-12)
            proj = alpha * pos_star
            dev += np.linalg.norm(pos - proj)

            if current_loss < tol * radius:
                success[tr] = True
                break

        steps = t + 1
        nsteps[tr] = steps
        loss[tr] = current_loss
        meanDev[tr] = dev / steps
        velMax[tr] = velMax_tr

        # --- RL update for s (trial-level REINFORCE) ---
        if sim_rehab:
            # return: higher is better
            # simplest: negative terminal error, optional time penalty
            R = -current_loss  #- time_cost * steps

            # baseline (EMA) to reduce variance
            if not baseline_init:
                baseline = R
                baseline_init = True
            else:
                baseline = (1.0 - beta) * baseline + beta * R

            adv = R - baseline

            # REINFORCE for Gaussian perturbation of s:
            # s_trial = s + sigma_s * eps_s
            # grad log p(s_trial|s) = eps_s / sigma_s
            # using additive parametrization => update proportional to eps_s / sigma_s^2
            s += eta_a * adv * (eps_s / (sigma_s ** 2))

            # regularize toward 1.0 (like your previous lam_a term)
            s -= eta_a * lam_a * (s - 1.0)

            # keep stable / interpretable
            # s = np.clip(s, s_min, s_max)

    # return the final plant matrix (using current mean s)
    A = A0 * s[None, :]
    return W_pol, success, nsteps, loss, meanDev, velMax, A


def simulate_trial(A, B, W_dec, W_pol, sigma_u=.1, ang=0, radius=1., maxT=1000, dt=.01, tol=.001):
    rng = np.random.default_rng()
    pos_star = radius * np.array([np.cos(ang), np.sin(ang)])
    K = A.shape[1]
    u = np.zeros((K,), dtype=float)
    pos = np.zeros((2,), dtype=float)
    P = W_dec @ B.T
    traj = np.zeros((maxT, 2), dtype=float)
    for t in range(maxT):
        f = A @ u  # (N,)
        vel = P @ f
        pos = pos + dt * vel
        traj[t] = pos
        e = pos_star - pos
        u = W_pol @ e + sigma_u * rng.standard_normal(K)
        current_loss = np.sqrt(e[0] ** 2 + e[1] ** 2)
        if current_loss < tol * radius:
            break

    return traj, u