---
title: GaitReflect Beam
emoji: 🦿
colorFrom: blue
colorTo: green
sdk: static
pinned: false
license: mit
short_description: Beam repairs 3D robot rigs; Blender and Bullet grade it
tags:
  - beam
  - reflection
  - blender
  - 3d
  - agents
  - verifiable-rewards
  - physics
---

![GaitReflect](assets/banner.png)

# GaitReflect v2

**Beam reads a deterministic Blender walk test, diagnoses why a rigged robot stretches, over-straightens a knee or can't stay on its feet, writes the bpy fix, and Blender verifies it. Robots that can't stand fall over under Bullet physics, not keyframes.**

Built for the [Build with Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5) challenge with [`Beam-1.0-501B-A23B`](https://huggingface.co/reflection/Beam-1.0-501B-A23B) via Hugging Face Inference Providers (`reasoning_effort=high`, JSON-schema output).

![Episode 24: broken input and Beam's repairs at L1, L2, L3](assets/ep24_beam.gif)

*Episode 24 (weight leak + upper body leaning 48° + right knee bending forward). The broken robot can't stand and falls under Bullet physics; Beam's L1 repair still falls; L2 passes and keeps walking; L3 fails on seams in its copied sway.*

## What changed in v2

![v1 bird knees vs v2 human knees](assets/knees_v1_v2.png)

| | v1 | v2 |
|---|---|---|
| Knees | thigh swung back, shin bent forward (a bird's knee) | thigh swings forward, shin bends back; defect = **hyperextension** past the rig limit (−150°…+5°) |
| Falls | a keyframe sank or tipped the root bone | the upper body leans or slides until the CoM leaves the feet's **support polygon** for 6+ frames; the fall is simulated with **Bullet** |
| Front | plain boxes | orange visor and chest plate |
| Walking | in place; left/right mirrored | hips travel forward with the stance foot planted; verifier checks travel and toe direction |
| Extra checks | | foot ≤ 2 cm into the floor, no sudden jumps in any bone's motion, upper body within ±20° |

## Result: 30 robots × 3 evidence levels, up to 3 rounds

![Solved within 3 rounds](assets/solved_chart.png)

| Evidence given to Beam | Solved in 1 round | Solved within 3 | Tokens / episode | Scripts crashed |
|---|---|---|---|---|
| L1 · measurements only | 8/30 | 18/30 | 24,748 | 4/71 |
| L2 · + vertex indices, bone motion vs. the normal cycle, bone axes | 13/30 | **27/30** | 18,109 | 0/53 |
| L3 · + near-answer hints | 11/30 | 23/30 | 20,328 | 0/61 |

- **L2 gave the best result**: 27/30, 8/8 on the three-defect robots, and the fewest tokens per episode.
- **More hinting was not better.** L3's hint ("restore values from the same phase one cycle earlier") points at already-broken frames when a defect lasts longer than one cycle, and Beam followed it.

## Beam read the verifier so precisely that it found a gap in it

![Before and after closing the loophole](assets/loophole_chart.png)

The first full run scored 29/30 at L3, but the GIF looked wrong. The system prompt asks for "the smallest change that makes the metrics pass", and the balance check only failed a CoM that stayed outside the feet for 6 frames in a row. Beam straightened the upper body on exactly the flagged frames and left the lean elsewhere: of 52 passing balance repairs, **46 snapped the torso by ≥15° in one frame** and 26 still leaned 45–50° on some frames. The gap was in our verifier. We added a motion-continuity check and a ±20° upper-body limit, regenerated the 30 episodes (same plan, smoother gait) and redid all 90 runs. **With full evidence (L2) the score did not move: 27/30 before and after.** Run 1 is kept in the dataset as `loophole_run/`.

## What the remaining failures teach the next version

22 of 90 runs still fail, each leaving a foot in the floor, a jump in the motion, or the CoM outside the feet:

- **Right symptom, wrong bone** (ep20, L1): Beam raised the foot that was in the floor; the cause was the knee keyed bending forward.
- **Copying a broken cycle** (ep21, L3): the "one cycle earlier" values were themselves broken.
- **Clamped to the limit, not restored** (ep11, L2): the torso was set to the +20° limit instead of upright.

## The loop

1. **Verify** (`harness.py`, Blender bpy + bmesh): face stretch, full weight audit, joint limits, floor penetration, motion continuity, support polygon, walk direction.
2. **Report**: structured prompt; the evidence level (L1–L3) sets how much it explains.
3. **Reflect**: Beam returns schema-checked JSON (`analysis`, `fix_code`, `expected`).
4. **Apply & re-verify** (`apply_fix.py`): the script runs behind a banned-pattern guard, then the test runs again. Up to 3 rounds. Beam never grades its own work.

Out of scope: momentum (sudden stops), foot sliding, and outside forces such as pushes or slopes.

## Files

- `index.html`: this demo (three.js explorer for all 30 robots, every Beam round)
- `harness.py`, `gen_episode.py`, `make_bench.py`, `reflect_loop.py`, `run_bench.py`, `apply_fix.py`, `blender_run.py`: the pipeline
- `bake.py` (Bullet falls + .glb), `build_viewer.py`, `render_gif.py`, `make_chart.py`, `make_post_images.py`, `build_dataset.py`
- `bench.json`, `bench_viewer.json`, `bench_runs.json`, `bench_models.bin` (120 rigs as concatenated .glb), `assets/`

Full data: [AG1188/gaitreflect-bench](https://huggingface.co/datasets/AG1188/gaitreflect-bench). v1 is in this Space's commit history.

## Caveats

- One attempt per episode and level (n = 30 per level); expect several points of sampling noise.
- Box-based robots (164 vertices, 144 faces, 10 bones); production characters have more bones, smoother weights and IK.
- Run 1 and run 2 use different (regenerated) episodes with the same plan.

All 90 repairs (diagnoses and fix scripts) are by Beam. The verifier, generator and this page were built with Claude Code.
