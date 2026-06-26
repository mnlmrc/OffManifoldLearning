import numpy as np
import matplotlib.pyplot as plt

def muscle_color(name, default="0.5"):
    if "flx" in name:
        return "r"
    if "ext" in name:
        return "b"
    return default


def link_points(q, lengths):
    """Joint positions [base, ..., tip] for an n-vector of joint angles q."""
    phi = np.cumsum(q)
    pts = [np.array([0.0, 0.0])]
    for k, L in enumerate(lengths):
        pts.append(pts[-1] + L * np.array([np.cos(phi[k]), np.sin(phi[k])]))
    return np.array(pts), phi


def to_world(body, coord, pts, phi):
    """Map a bone-local [along, perp] point on `body` to world coordinates."""
    if body == 0:
        base, ang = np.array([0.0, 0.0]), 0.0
    else:
        base, ang = pts[body - 1], phi[body - 1]
    d = np.array([np.cos(ang), np.sin(ang)])
    p = np.array([-np.sin(ang), np.cos(ang)])
    return base + coord[0] * d + coord[1] * p


def draw_arm(fig, ax, q, lengths, muscles, alpha=1.0, muscles_on=True):
    pts, phi = link_points(q, lengths)
    ax.plot(pts[:, 0], pts[:, 1], "-", color="0.5", lw=8, solid_capstyle="round", alpha=alpha, zorder=1)
    ax.plot(pts[:, 0], pts[:, 1], "o", color='k', ms=7, alpha=alpha, zorder=3)
    ax.plot(pts[-1, 0], pts[-1, 1], "o", color="white", mec="k", ms=8, alpha=alpha, zorder=3)
    if not muscles_on:
        return
    for fixation, coords, _, name in muscles:
        path = np.array([to_world(body, coord, pts, phi) for body, coord in zip(fixation, coords)])
        ax.plot(path[:, 0], path[:, 1], color=muscle_color(name), lw=2, zorder=2, label=name)
        ax.plot(path[:, 0], path[:, 1], ".", color=muscle_color(name), ms=6, zorder=2)