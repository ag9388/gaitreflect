---
license: mit
pretty_name: GaitReflect-Bench
size_categories:
  - n<1K
task_categories:
  - text-generation
tags:
  - blender
  - 3d
  - rigging
  - animation
  - benchmark
  - agents
  - code-generation
  - verifiable-rewards
  - physics
  - beam
configs:
  - config_name: runs
    data_files: runs.jsonl
    default: true
  - config_name: episodes
    data_files: episodes.jsonl
  - config_name: loophole_run
    data_files: loophole_run/runs.jsonl
---

# GaitReflect-Bench v2

**30 broken walking robots, a deterministic Blender verifier with Bullet physics for falls, and every attempt by [Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B) to repair them.**

![ep24 broken input](gifs/ep24.gif)

Each episode is a rigged humanoid robot (10 bones, 164 vertices, an orange face and chest plate marking the front) with a 60-frame walk cycle: the thigh swings forward, the shin bends back, the stance foot stays planted and the hips travel forward. 1–3 defects are planted. A model reads a verifier report, writes a bpy script, and Blender re-runs the walk test. PASS means zero verifier errors. The model never grades its own work.

Demo and write-up: [AG1188/gaitreflect-beam](https://huggingface.co/spaces/AG1188/gaitreflect-beam) · Built for [Build with Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5).

## Results: Beam-1.0-501B-A23B, up to 3 fix rounds per episode

Via Hugging Face Inference Providers, `reasoning_effort=high`, JSON-schema structured output.

| Evidence level | Solved in 1 round | Solved within 3 | Avg rounds (solved) | Scripts that crashed | Tokens / episode |
|---|---|---|---|---|---|
| **L1** · measurements only | 8/30 | 18/30 | 1.94 | 4/71 | 24,748 |
| **L2** · + vertex indices, bone motion vs. the normal cycle, bone axes | 13/30 | **27/30** | 1.63 | 0/53 | 18,109 |
| **L3** · + near-answer hints | 11/30 | 23/30 | 1.74 | 0/61 | 20,328 |

Solved within 3 rounds:

| Subset | L1 | L2 | L3 |
|---|---|---|---|
| contains leak | 15/20 | 17/20 | 14/20 |
| contains balance | 11/19 | 18/19 | 14/19 |
| contains knee | 7/17 | 15/17 | 11/17 |
| 1 defect | 6/10 | 10/10 | 10/10 |
| 2 defects | 7/12 | 9/12 | 8/12 |
| 3 defects | 5/8 | **8/8** | 5/8 |

**Findings**

- **L2 gave the best result** and used the fewest tokens per episode.
- **More hinting was not better.** L3's hint ("restore values from the same phase one cycle earlier") points at already-broken frames when a defect lasts longer than one cycle, and Beam followed it.
- **A verifier gap, found by Beam.** In a first run (kept in `loophole_run/`), the balance check only failed a CoM that stayed outside the feet for 6 frames in a row, and the system prompt asks for "the smallest change that makes the metrics pass". Beam straightened the upper body on exactly the flagged frames: of 52 passing balance repairs, 46 snapped the torso by ≥15° in one frame and 26 still leaned 45–50° on some frames. The motion-continuity check and the ±20° upper-body limit close that gap. With full evidence (L2) the score was 27/30 both before and after.
- **Remaining failures** leave a foot in the floor, a jump in the motion, or the CoM outside the feet. Typical diagnoses: raising the foot that is in the floor instead of restoring the knee; copying values from a cycle that is itself broken; clamping a joint to its limit instead of restoring the normal pose.

## How the episodes are built

- `gen_episode.py` builds a robot from random proportions and a random gait. The hips are computed from the stance leg so the foot stays planted; the swing knee bends most mid-swing.
- Defects:
  - **leak**: a body part's vertices also weighted to a bone that is not its parent or child
  - **balance**: from a random frame the upper body leans 38–50° to one side, or slides 0.18–0.26 m sideways off the hips, until the CoM leaves the feet
  - **knee**: during swing, a shin bends forward 15–35° past straight (hyperextension), beyond the rig limit of −150°…+5°
- Plan: 10 single-defect, 12 two-defect, 8 three-defect episodes.
- **Every episode is verified solvable**: with no fix it fails with exactly the planted defect types, the generated oracle fix (which restores the normal walk values) passes in one round, and the robot walks forward with its toes pointing ahead. Seeds that failed were regenerated (5 seeds in 4 of the 30 episodes).

## Verifier (harness.py, Blender bpy + bmesh)

| Check | Error when |
|---|---|
| Face stretch | a face's area > 1.5× (or < 0.4×) its rest-pose area |
| Weight audit | a vertex carries weight on two bones that are not parent/child |
| Balance | the CoM's ground projection is > 5 cm outside the support polygon (ground-contact vertices) for 6+ consecutive frames |
| Ground penetration | any vertex more than 2 cm below the floor |
| Motion continuity | a bone's own rotation (location) changes by ≥15° (≥4 cm) in its frame-to-frame second difference |
| Joint limits | knee −150°…+5°, thigh −50°…+60°, upper-body tilt ±20° (rig spec on the bones) |
| Walk preserved | thigh swing below half the original stride, or the robot no longer travels forward with toes ahead |

Robots that fail the balance check are handed to Bullet (pybullet) at that frame for display: the pose is frozen and the parts fall as one rigid body. Out of scope: momentum (sudden stops), foot sliding, and outside forces such as pushes or slopes.

## Evidence levels

- **L1**: which checks failed and the measured values.
- **L2**: + vertex indices and bone distances, + each bone's deviation from the normal walk cycle at the worst frame (same phase one cycle earlier), + bone axes in world space, + frames of each motion jump.
- **L3**: + "likely leak group" verdicts, + repair strategies in the system prompt.

Blender 5 API notes are in the system prompt at every level.

## Files

| File | Contents |
|---|---|
| `runs.jsonl` | 90 rows (30 episodes × 3 levels): solved, rounds, crashes, tokens, and `rounds[]` with the exact prompt, Beam's diagnosis, fix script, Blender error (if any), errors before/after |
| `episodes.jsonl` | 30 rows: defects, robot parameters, ground truth (incl. leaked vertex indices), oracle fix script, walk check |
| `episodes.zip` | `episodes/ep01.blend` … `ep30.blend`, oracle scripts, `manifest.json` |
| `gifs/epXX.gif` | each broken input walking (and falling under Bullet) |
| `loophole_run/` | the first run, before the verifier gap was closed: same format, plus `recheck_with_v2_verifier.json` |
| `summary.md`, `summary.json` | the tables above |
| `*.py` | verifier, generator, loop, runner, sandboxed applier, Bullet baking, GIF rendering, packaging |

Beam's internal reasoning text and local file paths are not included.

## Reproduce

```bash
pip install bpy pybullet              # Blender 5.2 as a Python module; pybullet may build from source
python3 make_bench.py --out episodes   # generate + verify 30 episodes (no API)
export HF_TOKEN=...
python3 run_bench.py --effort high --token-budget 2300000
```

## Limitations

- One attempt per episode and level (n = 30 per level); expect several points of sampling noise.
- Box-based robots; production characters have more bones, smoother weights and IK.
- `loophole_run/` uses an earlier set of episodes (same plan, slightly different gait), so its numbers are not a paired comparison.

All 90 repairs (diagnoses and fix scripts) are by Beam. The verifier, generator and the demo page were built with Claude Code.
