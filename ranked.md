# Solution Approaches — Ranked (cable insertion)

The task is a **goal-conditioned POMDP**: hidden state = *where is the target port*; the policy only
gets 3 wrist-camera images + joints + force/torque (no ground-truth pose at evaluation). Approaches
sort into **analytic**, **learning**, and **hybrid**. "Trial-and-error" isn't a separate method — it's
the *search* inside force-control and inside RL.

| # | Approach | What it is | Fit for *this* problem |
|---|---|---|---|
| **Analytic / model-based** ||||
| 1 | Motion planning (RRT, etc.) | Plan a collision-free path to a known goal | Necessary but insufficient — assumes you *know* the port pose; doesn't handle perception or contact |
| 2 | Visual servoing (IBVS/PBVS) | Closed-loop control driving image features / an estimated pose to target | Attacks perception→motion without learning — if you can reliably extract port features |
| 3 | Force/compliance control + search | Impedance/admittance + spiral/tilt search, force-threshold seating | *The* classic peg-in-hole solver; the contact-phase answer (the "trial-and-error" part) |
| 4 | Classical 6-DoF pose estimation | CAD/template/fiducial matching → then plan+servo | Perception via classical CV; brittle to lighting/clutter |
| **Learning** ||||
| 5 | Imitation learning (ACT, diffusion policy) | Clone expert demos: pixels→actions | Toolkit's own baseline; needs an expert (CheatCode) |
| 6 | Reinforcement learning (PPO/SAC) | Learn from reward in sim | Powerful for contact, but sample-hungry, reward-shaping hard, sim-to-real gap |
| **Hybrid (usually strongest)** ||||
| 7 | **Learned perception + analytic control** ⭐ | NN estimates the *port pose* from images (trained on free GT labels), fed into CheatCode-style insertion | **Primary path.** Learn only the hard hidden part; keep control robust |
| 8 | Residual RL / RL-on-demos | Learn a *correction* on top of a scripted/planned base; or IL-warm-start → RL-finetune | SOTA for connector insertion; combines robustness + learned dexterity |
| 9 | VLA / foundation models | Fine-tune a large pretrained vision-language-action model (OpenVLA, π0, or Phase-1's Intrinsic Vision Model) | Best generalization; heavy; Phase-1-oriented |

## Why the problem is perception-dominated
With ground truth (CheatCode) it *looks* like path planning — but that's only because the port pose is
handed to you for free. Remove it and three real problems appear that planning doesn't touch:
1. **Perception** — locate a mm-scale port in pixels under randomized board pose + clutter (the hard 80%).
2. **Contact-rich insertion** — sub-mm tolerance peg-in-hole; drive straight down on a bad estimate → jam.
3. **Deformable cable + imperfect grasp** — the plug swings on a floppy cable, ~2 mm/0.04 rad grasp slop.

## What we chose (and why)
- **#7 (learned perception + analytic control)** is the primary strategy: train a net for `images → port
  pose` (we have *free* GT labels in training via TF), then reuse CheatCode's analytic descent.
  Lowest-risk, sample-efficient, interpretable. → this is what the collected dataset targets.
- **#5 (imitation learning)** kept open in parallel — we also recorded the expert's actions
  (`il_frames.parquet`), so end-to-end BC/ACT is available as an alternative/backup.
- **#3 (force-search)** for the final contact if the perception estimate has residual error; the F/T
  data is in the dataset.
- **#8 (residual RL)** as a later sharpening step; **#9 (VLA)** relevant if we reach Phase 1.

Dataset built for this: `perception_v1/` — `frames.parquet` = `(images → port pose)` for #7,
`il_frames.parquet` = `(images + proprio → expert action)` for #5.

## Related work: CoStream (arXiv:2606.26423)

*Chen et al., "CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation"* —
their headline task is **seating a GPU into a PCIe slot**, i.e. the same sub-mm connector-insertion
problem class. Rather than a rigid pipeline or a monolithic end-to-end policy, they **compose three
independent behaviors** over a shared **SE(3) task-space interface**, fused by right-multiplication into
one pose command per step and run by a **compliant (impedance/admittance) controller**:

- **Semantic behavior** (LLM + VLM): grounds instruction + scene into a **task-frame anchor** (a frame,
  not a trajectory); object-centric ⇒ SE(3)-equivariant across board poses.
- **Predictive behavior** (video world model + VLM critic): imagined-video rollouts → **nominal motion
  prior** in the task frame (3D-keypoint-lifted).
- **Reactive behavior** (tactile + F/T): high-rate **force/contact residual** + guard events.

Two levels: an open-vocab **stage compiler** (pre-grasp → grasp → insert → home) and **per-stage parallel
action composition**. Claim: monolithic-policy generalization without the data hunger, pipeline precision
without the brittleness; zero-shot transfer + perturbation recovery on 8 real tasks.

**Where it fits this table.** CoStream isn't one row — it's an *architecture for composing the rows*, in
the "Hybrid (usually strongest)" band ≈ **#7 ⊕ #3 ⊕ #9**:

| CoStream component | Row here |
|---|---|
| Semantic anchor (images → task frame, via foundation models) | **#7** learned perception, realized with **#9** instead of a trained pose net; grounding echoes **#4** |
| Predictive video-WM motion prior in task frame | motion-planning prior **#1**, generated by a learned world model (predictive cousin of **#5**) |
| Reactive tactile/F-T residual + compliant controller | **#3** force/compliance control + search; closed-loop like **#2** |

**Relevance to our choice.** It's essentially a generalized, foundation-model-powered version of our
**#7 + #3** bet (see "What we chose"): learn the hidden port frame, hand it to robust force-compliant
insertion. The paper is an explicit argument *for* that decomposition and *against* the monolithic
ACT/VLA baseline — the same end-to-end policy that scored **−21** in `baseline_scores.md`. Beyond our #7
it adds zero-shot foundation-model grounding and a video-WM motion prior (**#9-heavy, Phase-1**).
**Big caveat — we are sim-only.** CoStream is validated *entirely on real hardware*: real RGB-D, real
GelSight-style **tactile**, 8 real-world tasks. We have **only synthetic Gazebo data — no real-world data
at all**, and scoring happens in sim. Consequences:

- **No tactile in our setup**, and none in sim. CoStream's reactive behavior is tactile-first; ours can
  only be the **Gazebo F/T** signal (coarser, no contact geometry). Their high-rate tactile residual is
  not reproducible for us — the reactive layer degrades to force-threshold + compliance (#3).
- **No RGB-D / depth guaranteed** — 3 wrist **RGB** cameras only. Their depth-based scene parser and
  3D-keypoint lifting would need mono depth estimation or multi-view triangulation on our side.
- **Our data is a strength for #7, not #9.** Sim gives *unlimited* frames with **free GT port-pose
  labels from TF** — ideal for training the #7 perception net. It does *not* give the real-image
  distribution that CoStream's foundation-model grounding + video world model were tuned on; running
  those on Gazebo renders is an untested domain shift.
- **Sim-to-real is deferred (Phase 1).** Everything we build lands in Gazebo first; the real-world
  transfer CoStream demonstrates is out of scope for us right now (cf. `RL.md` sim-to-real notes).

Net: external validation that **learned-perception-anchor + compliant force-reactive control (#7+#3)**
beats an end-to-end policy for this task class — but adopt only the parts our **sim-only, RGB+F/T,
no-tactile** bench can support. The directly usable idea is the SE(3) composition of a *perception frame
anchor* (#7, trained on our free sim GT) with a *compliant force residual* (#3); the tactile reactive
layer and real-data foundation-model/video-WM layers are aspirational until we have real hardware/data.
