![GaitReflect](https://huggingface.co/spaces/AG1188/gaitreflect-beam/resolve/main/assets/banner.png)

## GaitReflect v2: Beam repairs broken 3D robot rigs, Blender and Bullet physics grade it

- 🔗 Space: https://huggingface.co/spaces/AG1188/gaitreflect-beam
- 📊 Dataset: https://huggingface.co/datasets/AG1188/gaitreflect-bench

Beam reads a deterministic Blender walk test, diagnoses why a rigged robot stretches, over-straightens a knee or can't stay on its feet, writes a bpy fix, and Blender re-runs the test. Robots that can't stand fall over under **Bullet physics**, not keyframes. Beam never grades its own work.

![Episode 24: broken input and Beam's repairs at L1, L2, L3](https://huggingface.co/spaces/AG1188/gaitreflect-beam/resolve/main/assets/ep24_beam.gif)

*Episode 24: weight leak + upper body leaning 48° + right knee bending forward. The broken robot can't stand and falls; Beam's L1 repair still falls; **L2 passes and keeps walking**; L3 fails on seams in its copied sway.*

### 30 robots × 3 evidence levels

![Solved within 3 rounds](https://huggingface.co/spaces/AG1188/gaitreflect-beam/resolve/main/assets/solved_chart.png)

| Evidence given to Beam | Solved in 1 round | Solved within 3 | Tokens / episode |
|---|---|---|---|
| L1 · measurements only | 8/30 | 18/30 | 24.7k |
| L2 · + vertex indices & bone motion | 13/30 | **27/30** | 18.1k |
| L3 · + near-answer hints | 11/30 | 23/30 | 20.3k |

- **L2 gave the best result**: 27/30, 8/8 on the three-defect robots, the fewest tokens per episode, and none of its 53 scripts crashed.
- **More hinting was not better.** L3's hint ("restore values from one gait cycle earlier") points at already-broken frames when a defect lasts longer than one cycle, and Beam followed it.

### Beam read the verifier so precisely that it found a gap in it

![Before and after closing the loophole](https://huggingface.co/spaces/AG1188/gaitreflect-beam/resolve/main/assets/loophole_chart.png)

The first full run scored 29/30 at L3, but the GIF looked wrong. My system prompt asks for "the smallest change that makes the metrics pass", and the balance check only failed a center of mass that stayed outside the feet for 6 frames in a row. Beam straightened the upper body on exactly the flagged frames: of 52 passing balance repairs, **46 snapped the torso by ≥15° in a single frame**. The gap was in my verifier, not in Beam. I added a motion-continuity check and a ±20° upper-body limit, regenerated the episodes and redid all 90 runs. **With full evidence (L2), Beam's score did not move: 27/30 before and after.**

### What changed since v1

![v1 bird knees vs v2 human knees](https://huggingface.co/spaces/AG1188/gaitreflect-beam/resolve/main/assets/knees_v1_v2.png)

- Knees now bend the human way; the knee defect is hyperextension past the rig limit.
- "Falling" is now physical: a posture whose center of mass leaves the support polygon, simulated with Bullet.
- An orange face and chest plate mark the front; the robots walk forward with planted feet.

In the Space you can pick any of the 30 robots, watch the broken rig next to Beam's repairs at L1/L2/L3 in 3D, and read every Beam round unedited. The dataset has all episodes, all 90 runs (prompts, diagnoses, scripts), the first run with the loophole, and a GIF per robot.

Caveats: one attempt per episode and level; box-based robots; momentum, foot sliding and outside forces (pushes, slopes) are out of scope.

All 90 repairs (diagnoses and fix scripts) are by Beam. The verifier, generator and demo page were built with Claude Code.
