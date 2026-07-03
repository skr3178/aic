# Collection Pipeline — Fixes & Gotchas That Made It Work

Hard-won fixes from building the CheatCode data-collection pipeline (`gen_configs.py`,
`collector_node.py`, `run_collection.sh`, `finalize.py`). Each cost real debugging time — read
before touching the pipeline.

---

## 1. `set -o pipefail` silently broke engine-up detection  ⭐ (the big one)
- **Symptom:** container boots and the engine polls for the model forever; `policy.log`/
  `collector.log` empty; CheatCode + collector never launch; container eventually times out and
  the run "completes" having collected **nothing**. Intermittent — worked the first time.
- **Root cause:** the wait loop uses `docker logs "$c" | grep -q "aic_model"`. `grep -q`
  short-circuits (exits 0) on first match and closes the pipe → `docker logs` gets **SIGPIPE** →
  with `pipefail` the whole pipeline reports **failure** → the `if` sees non-zero → "engine up" is
  never detected → policy/collector never start. As logs grow, grep short-circuits earlier, so it
  fails more reliably (looked flaky).
- **Fix:** `set -u` only — **do not** use `pipefail` in scripts that do `cmd | grep -q`.
  (In `run_collection.sh`.)

## 2. Episode boundaries: TF frames don't get "deleted"  ⭐
- **Symptom:** the collector recorded one giant "episode 0" of ~725 frames spanning **both trials**;
  ep1 never started; no clean per-episode split.
- **Root cause:** the per-trial cable despawn stops *publishing* its plug TF frame, but tf2's Buffer
  keeps the last transform for `cache_time`. So `can_transform(base_link, plug, latest)` stays
  **true** long after the cable is gone → the collector never sees the boundary.
- **Fix:** detect "currently published" by **transform freshness**, not availability — look up the
  latest transform and check its stamp age:
  ```python
  tf = buf.lookup_transform("base_link", frame, Time())
  age = (self.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds * 1e-9
  return age < 0.75     # fresh => frame currently published
  ```
  (In `collector_node.py::has()`.) Debounce 3 frames on both edges.

## 3. SIGKILL loses the last episode → write incrementally
- **Symptom:** an episode dir had 725 PNGs but **no `frames.parquet`** (only written in `end()`).
- **Root cause:** the collector is torn down with `pkill -9` (SIGKILL) — can't be caught, so the
  `finally`/`end()` parquet flush never runs.
- **Fix:** write **`frames.jsonl` incrementally** (append + `flush()` per frame) as the crash-safe
  record; still write `frames.parquet` at `end()` for clean episodes. `finalize.py`/`n_frames`
  read parquet-or-jsonl. (In `collector_node.py`.)

## 4. `pkill -f` self-kills inline commands → use a FILE
- **Symptom:** an inline background command died instantly at its own `pkill`, before doing work.
- **Root cause:** `pkill -9 -f "aic_model --ros-args"` matches **its own shell's argv** when the
  command string contains that pattern (inline `bash -c '...'`).
- **Fix:** put the orchestrator in a **`.sh` file** (`run_collection.sh`). Its process argv is
  `bash run_collection.sh …` — it does **not** contain the pkill patterns, so `pkill` only hits the
  real policy/collector. For ad-hoc cleanup, kill by PID excluding self:
  ```bash
  me=$$; ps -eo pid,ppid,cmd | grep -E "collector_node.py|aic_model --ros" | grep -v grep \
    | awk -v me=$me '$1!=me && $2!=me{print $1}' | xargs -r kill -9
  ```

## 5. Long jobs must be FULLY DETACHED (harness reaps managed jobs)
- **Symptom:** the run wrapper reported exit 1/144 mid-run; policy/collector (children) died with it.
- **Fix:** launch the orchestrator itself fully detached so it's independent of the harness, and
  inside it launch policy+collector detached too:
  ```bash
  setsid nohup bash run_collection.sh 0 1 2 3 4 >full_run.log 2>&1 </dev/null & disown
  ```
  `run_collection.sh` writes `$$` to `~/aic_results/collect/run.pid`; a separate watcher polls
  `kill -0 <pid>` for liveness/completion (survives even if the watcher is reaped).

## 6. Root-owned results dir (container writes `/results` as root)
- **Symptom:** `rm -rf ~/aic_results/collect/chunk_0` → "Permission denied" on `*.mcap`/`metadata.yaml`.
- **Root cause:** the eval container writes scoring bags to the bind-mounted `/results` **as root**.
- **Fix:** don't `rm` results in the run path (the container overwrites `scoring.yaml`; bags just
  accumulate harmlessly). To reset, delete via a throwaway root container:
  ```bash
  docker run --rm -v /home/skr/aic_results/collect:/x alpine sh -c "rm -rf /x/chunk_*"
  ```

## 7. Foreground `sleep` is blocked by the harness
- `sleep` in a **foreground** Bash tool call fails (exit 1). Use it only inside background jobs /
  detached scripts. To wait-on-a-condition in the foreground, use a bounded `for` loop with the
  work in a `run_in_background` job, or `docker wait`.

## 8. gen_configs SFP-port balance bug
- **Symptom:** all 25 SFP episodes got `sfp_port_0`, none `sfp_port_1`.
- **Root cause:** balanced on the **global** index (`gidx % 2`), but the SFP/SC interleave puts SFP
  only on even global indices → always port_0.
- **Fix:** balance on the **SFP ordinal** (0,1,2… among SFP episodes), not the global index.

---

## Carried over from M0/M1 (still essential for collection)

- **GPU EGL rendering** — `-e NVIDIA_DRIVER_CAPABILITIES=all` **and** mount `10_nvidia.json` as the
  EGL vendor ICD. Without it Gazebo CPU-renders the 3 cameras → **RTF ≈ 0.03** (~30× slower). With
  it, headless collection runs at **RTF ≈ 0.95** (≈ real-time; 50 episodes ≈ ~45 min).
- **Exactly one `aic_model` node** — orphans → engine "More than one node with name 'aic_model'" →
  Tier-1 fail. `run_collection.sh` kills stale policy/collector before each chunk.
- **eval/generated configs not in the image** — bind-mount the config dir (`-v configs:/aic_cfg:ro`)
  and pass `aic_engine_config_file:=/aic_cfg/<file>.yaml`; the prebuilt image (2026-04-16) only ships
  `sample_config.yaml`.
- **`ground_truth:=true`** required — CheatCode reads GT TF; and the collector needs `/scoring/tf`→
  `/tf` for the pose labels.

## Verified-correct signals (what "working" looks like)
- collector log: `START ep N → END ep N: ~360 frames → START ep N+1 …`
- per episode: `frames.parquet` + `frames.jsonl` + `meta.json` + 3× PNG dirs.
- **Label chain (`verify_labels.py`)**: port-in-base `std ≈ [0,0,0] mm` across an episode (stable
  label, changing views) **and** the recorded port pose projects **inside the image** near the port
  (converging as the arm descends). This is the make-or-break perception check.
