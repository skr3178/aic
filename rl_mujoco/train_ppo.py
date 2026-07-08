"""Single-env PPO smoke test on UR5eReachEnv (stable-baselines3, torch/CUDA).
Goal: prove the RL loop trains end-to-end on ONE MuJoCo env on this machine.
Reports mean TCP->port distance and success rate before vs after training.
"""
import os, time, numpy as np
os.environ.setdefault("PYTHONNOUSERSITE", "1")

from ur5e_reach_env import UR5eReachEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import torch


def evaluate(model, n_ep=20, seed0=1000):
    env = UR5eReachEnv()
    dists, succ, rets = [], [], []
    for k in range(n_ep):
        obs, _ = env.reset(seed=seed0 + k)
        done = trunc = False
        ret = 0.0
        info = {"dist": np.nan, "is_success": False}
        while not (done or trunc):
            act = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
            obs, r, done, trunc, info = env.step(act)
            ret += r
        dists.append(info["dist"]); succ.append(float(info["is_success"])); rets.append(ret)
    return np.mean(dists), np.mean(succ), np.mean(rets)


def main():
    total_steps = int(os.environ.get("STEPS", 150_000))
    print(f"torch cuda: {torch.cuda.is_available()}  device: {torch.cuda.get_device_name(0)}")
    print(f"training PPO for {total_steps} steps on ONE env\n")

    env = Monitor(UR5eReachEnv())
    model = PPO("MlpPolicy", env, verbose=1, device="cuda",
                n_steps=2048, batch_size=128, gae_lambda=0.95, gamma=0.99,
                n_epochs=10, ent_coef=0.0, learning_rate=3e-4,
                policy_kwargs=dict(net_arch=[256, 256]), seed=0)

    d0, s0, r0 = evaluate(None)                       # random baseline
    print(f"[BEFORE] random policy: mean_dist={d0:.4f} m  success={s0:.0%}  return={r0:.1f}\n")

    t = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=False)
    dt = time.time() - t

    d1, s1, r1 = evaluate(model)                      # trained
    print(f"\n[AFTER]  trained policy: mean_dist={d1:.4f} m  success={s1:.0%}  return={r1:.1f}")
    print(f"[RESULT] dist {d0:.4f} -> {d1:.4f} m | success {s0:.0%} -> {s1:.0%} "
          f"| {total_steps/dt:.0f} steps/s over {dt:.0f}s")
    model.save(os.path.join(os.path.dirname(__file__), "ppo_ur5e_reach"))
    print("saved model -> ppo_ur5e_reach.zip")


if __name__ == "__main__":
    main()
