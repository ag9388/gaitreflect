## GaitReflect: Beam repairs broken 3D robot rigs, verified by Blender

🔗 Space: https://huggingface.co/spaces/AG1188/gaitreflect-beam
📊 Dataset: https://huggingface.co/datasets/AG1188/gaitreflect-bench

Rigging a walking robot is tedious: weights leak between bones, meshes stretch, the robot tips over. GaitReflect turns this into a verifiable loop where **Beam does the reasoning and Blender does the grading**:

1. **Verify**: Blender (bpy + bmesh) plays the walk cycle headless and measures face stretch, a full vertex-weight audit, center-of-mass height, joint limits, and which bone moved during a fall.
2. **Reflect**: the numbers go to `Beam-1.0-501B-A23B` as a structured report. Beam returns schema-checked JSON with a diagnosis per issue and a bpy fix script.
3. **Re-verify**: the script runs in a sandboxed Blender and the test runs again. PASS = zero verifier errors. Beam never grades its own work.

### GaitReflect-Bench: 30 generated robots × 3 evidence levels

30 rigged humanoids with random proportions and gaits, 1–3 planted defects each (weight leak, fall, knee flip), every episode verified solvable. Beam repaired each one at three evidence levels, up to 3 rounds.

| Evidence given to Beam | Solved in 1 round | Solved within 3 |
|---|---|---|
| L1 · measurements only | 13/30 | 21/30 |
| L2 · + vertex indices & bone motion | 27/30 | **30/30** |
| L3 · + near-answer hints | 29/30 | **30/30** |

- **L2 is enough.** Near-answer hints add little, so Beam reasons from the measurements rather than reading out the answer.
- **Every L1 failure is an unfixed fall.** With 3 defects at once, L1 solved 0/8 and L2 solved 8/8. Failed diagnoses were axis confusion, blaming a co-occurring defect, or a vague "outlier keyframe".
- **Thin evidence costs ~2× the tokens**, because Beam iterates through wrong guesses.

In the Space you can pick any of the 30 robots and watch the broken rig walk next to Beam's repairs at L1, L2 and L3 in 3D, plus every Beam round unedited and the full pipeline (stdlib Python + Blender 4.4+). The dataset has all 30 episodes and all 90 runs with prompts, diagnoses and scripts.

Caveats: one attempt per episode and level; box-based robots; episodes 1–10 ran via the Reflection API and 11–30 via HF Inference Providers (same model, route recorded per run).