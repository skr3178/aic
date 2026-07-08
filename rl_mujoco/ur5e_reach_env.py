"""Single-env MuJoCo Gymnasium task: UR5e drives its tool flange (TCP) to a fixed
port target. Insertion-flavored *reach* — the minimal task that proves the RL loop
runs end-to-end on one MuJoCo env on this machine, before parallelizing with MJX.

Design notes for the MJX port later:
- rigid bodies only, no elasticity/cable plugin (MJX-incompatible), so this scene
  and reward transfer directly to a vmap'd MJX env.
- action = delta joint-position targets (position actuators), matching how the AIC
  controller drives the UR5e.
"""
from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCENE = os.path.join(_HERE, "ur5e_menagerie", "insertion_scene.xml")


class UR5eReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, scene_path: str = _SCENE, max_steps: int = 200,
                 action_scale: float = 0.05, success_dist: float = 0.02):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.action_scale = action_scale          # rad per step, per joint
        self.success_dist = success_dist
        self.n_sub = 5                             # physics substeps per env step

        self.nu = self.model.nu                    # 6 arm actuators
        self.tcp_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        self.port_sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "port_site")
        self.home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        # action: normalized delta joint targets in [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.nu,), dtype=np.float32)
        # obs: qpos(6) qvel(6) tcp(3) port(3) tcp-port(3) = 21
        obs_dim = self.nu * 2 + 9
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._step = 0

    def _tcp(self):
        return self.data.site_xpos[self.tcp_sid].copy()

    def _port(self):
        return self.data.site_xpos[self.port_sid].copy()

    def _obs(self):
        tcp, port = self._tcp(), self._port()
        return np.concatenate([
            self.data.qpos[: self.nu], self.data.qvel[: self.nu],
            tcp, port, tcp - port,
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        # small random joint perturbation for exploration diversity
        self.data.qpos[: self.nu] += self.np_random.uniform(-0.05, 0.05, self.nu)
        self.data.ctrl[:] = self.data.qpos[: self.nu]
        mujoco.mj_forward(self.model, self.data)
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        target = self.data.ctrl[: self.nu] + action * self.action_scale
        self.data.ctrl[: self.nu] = np.clip(target, self.ctrl_lo, self.ctrl_hi)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self._step += 1

        dist = float(np.linalg.norm(self._tcp() - self._port()))
        success = dist < self.success_dist
        # dense shaped reward: negative distance + exp bonus near target + success spike
        reward = -dist + 0.5 * np.exp(-(dist ** 2) / (0.05 ** 2))
        reward -= 1e-3 * float(np.sum(np.square(action)))    # small ctrl cost
        if success:
            reward += 10.0

        terminated = success
        truncated = self._step >= self.max_steps
        info = {"dist": dist, "is_success": success}
        return self._obs(), float(reward), terminated, truncated, info


if __name__ == "__main__":
    # quick sanity: load, reset, random rollout, print distances
    env = UR5eReachEnv()
    obs, _ = env.reset(seed=0)
    print("obs_dim", obs.shape, "act_dim", env.action_space.shape, "nu", env.nu)
    d0 = np.linalg.norm(env._tcp() - env._port())
    for _ in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
    print(f"start dist={d0:.3f}  after 50 random steps dist={info['dist']:.3f}  reward={r:.3f}")
    print("SANITY OK")
