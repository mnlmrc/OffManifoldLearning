import os
import torch as th
import numpy as np
from motornet.policy import PolicyGRU
from motornet_sim.effector import FingerForceEffector
from motornet_sim.environment import IsometricSingleFingerTask


def run_episode(env, policy, batch_size: int = 64):
    """Roll out one episode end-to-end and return mean per-step losses."""
    obs, _ = env.reset(options={"batch_size": batch_size})
    h = policy.init_hidden(batch_size)

    n_steps = int(env.max_ep_duration / env.dt)
    pos_loss = th.tensor(0.)
    effort_loss = th.tensor(0.)

    for _ in range(n_steps):
        action, h = policy(obs, h)
        obs, _, _, _, info = env.step(action)

        fingertip = info["states"]["fingertip"]   # (batch, 3)
        goal = info["goal"]                        # (batch, 3)
        pos_loss += th.mean(th.sum((fingertip - goal) ** 2, dim=-1))
        effort_loss += th.mean(th.sum(action ** 2, dim=-1))

    return pos_loss / n_steps, effort_loss / n_steps


def train(
    n_batches: int = 2000,
    batch_size: int = 64,
    hidden_dim: int = 128,
    ep_duration: float = 1.0,
    dt: float = 0.01,
    lr: float = 1e-3,
    lambda_effort: float = 1e-4,
    target_distance: float = 1.0,
    save_dir: str = None,
):
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    effector = FingerForceEffector(timestep=dt)
    env = IsometricSingleFingerTask(
        effector=effector,
        max_ep_duration=ep_duration,
        target_distance=target_distance,
        differentiable=True,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = PolicyGRU(input_dim=obs_dim, hidden_dim=hidden_dim, output_dim=act_dim, device=device)

    optimizer = th.optim.Adam(policy.parameters(), lr=lr)

    losses, pos_losses = [], []
    for i in range(n_batches):
        optimizer.zero_grad()

        pos_loss, effort_loss = run_episode(env, policy, batch_size=batch_size)
        loss = pos_loss + lambda_effort * effort_loss
        loss.backward()

        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        pos_losses.append(pos_loss.item())

        if (i + 1) % 100 == 0:
            print(
                f"batch {i+1:4d}/{n_batches} | "
                f"loss={loss.item():.5f} | "
                f"pos={pos_loss.item():.5f} | "
                f"effort={effort_loss.item():.6f}"
            )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        th.save(policy.state_dict(), os.path.join(save_dir, "policy.pt"))
        np.save(os.path.join(save_dir, "losses.npy"), np.array(losses))
        np.save(os.path.join(save_dir, "pos_losses.npy"), np.array(pos_losses))
        print(f"Saved to {save_dir}")

    return env, policy, np.array(losses)


if __name__ == "__main__":
    train(n_batches=2000, batch_size=64, save_dir="results/isometric_finger")
