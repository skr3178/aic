#!/usr/bin/env python3
"""Loss/error curves: earlier successful 1024 run vs the interrupted 1024-on-v2 run."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(d):
    p = f"/home/skr/aic_data/{d}/history.json"
    return json.load(open(p)) if os.path.exists(p) else None

# earlier successful 1024 dual run (native 50-ep, 2 rails) — the direct predecessor config
nat = load("m35_native_1024")
# context: the 224 dual run on v1
dual = load("m35_dual_run")

# interrupted run (1024 aspect-crop on v2, all-rail) — only the epoch-6 endpoint survived (log lost on sleep)
intr = {"epoch": 6, "port_mm_med": 18.2, "plug_mm_med": 33.4, "train_loss": 0.072}

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

# ---- Panel 1: validation PORT position error ----
def eps(h): return [e["epoch"] for e in h]
def key(h, k): return [e[k] for e in h]
if nat:
    ax[0].plot(eps(nat), key(nat, "port_mm_med"), "-o", color="#2a7", lw=2, ms=5,
               label="earlier 1024 run (native, 2 rails) — best 10.9mm")
if dual:
    ax[0].plot(eps(dual), key(dual, "port_mm_med"), "--s", color="#79a", lw=1.5, ms=4, alpha=0.8,
               label="M3.5 dual 224 (v1) — best 12.6mm")
ax[0].plot(intr["epoch"], intr["port_mm_med"], "*", color="#d33", ms=22, mec="k", mew=0.6, zorder=5,
           label="interrupted 1024 run (v2, ALL rails) — epoch 6 = 18.2mm")
ax[0].annotate("only surviving point\n(full curve lost when\nmachine slept mid-run)",
               xy=(6, 18.2), xytext=(6.4, 30), fontsize=8.5, color="#d33",
               arrowprops=dict(arrowstyle="->", color="#d33", lw=1))
ax[0].set_title("Validation PORT position error", fontsize=12, fontweight="bold")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("median error (mm)")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_ylim(0, 70)

# ---- Panel 2: training loss ----
if nat:
    ax[1].plot(eps(nat), key(nat, "train_loss"), "-o", color="#2a7", lw=2, ms=5, label="earlier 1024 run (native)")
if dual:
    ax[1].plot(eps(dual), key(dual, "train_loss"), "--s", color="#79a", lw=1.5, ms=4, alpha=0.8, label="M3.5 dual 224 (v1)")
ax[1].plot(intr["epoch"], intr["train_loss"], "*", color="#d33", ms=22, mec="k", mew=0.6, zorder=5,
           label="interrupted 1024 run (v2) — epoch 6")
ax[1].set_title("Training loss", fontsize=12, fontweight="bold")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train loss")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="upper right")

fig.suptitle("Perception net — loss/error curves  (interrupted 1024-on-v2 vs earlier successful runs)",
             fontsize=12.5, fontweight="bold")
fig.text(0.5, 0.005,
         "Note: v2 val spans ALL 5 NIC + 2 SC rails (harder) vs the earlier runs' 2-rail val — absolute mm are not directly comparable; "
         "the interrupted run's per-epoch curve was lost on sleep, resume-relaunch is regenerating it.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
out = "/home/skr/aic_data/m6_dual_run/loss_curves.png"
fig.savefig(out, dpi=110)
print("wrote", out)
