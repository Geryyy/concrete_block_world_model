# concrete_block_world_model

Persistent block world model and perception orchestrator for the concrete-block stack. Owns the single `world_model_node` executable that consumes detections + registration results from [concrete_block_perception](../concrete_block_perception/) and serves the persistent block state to the BT, the wall planner and the motion planner.

## Responsibilities

- Hold the **single source of truth** for block state (id, pose, pose-/task-/goal-status, confidence) across frames and pipeline modes — perception providers stay stateless.
- **Orchestrate perception**: drive detection → cutout → coarse pose → ICP registration, in continuous, scene-discovery and on-demand refine flows.
- Track grasped blocks by FK (using the grasp offset the BT supplies) while `task_status == TASK_MOVE`.
- Serve the pull-based query / write API and publish the state + RViz markers + per-stage timing.

## Role in the stack

```
detection_tracking_node ──TrackedDetections──┐
                                              ▼
                                      world_model_node ◄──► block_registration_node
                                       (this package)              (RegisterBlock action)
                                              │
                                              ├─► block_world_model    (BlockArray, persistent state)
                                              ├─► markers / debug/*     (RViz overlays)
                                              ├─► timing/continuous_*    (per-stage latency)
                                              └─► services: GetCoarseBlocks, GetPlanningScene,
                                                            RunPoseEstimation, SetBlockTaskStatus,
                                                            SetBlockGoal, UpsertBlock, SetPerceptionMode
```

## Contents

```text
config/   world_model.yaml plus seed YAMLs (none / pick_place / b0 / legacy_2block)
include/concrete_block_world_model/  Public headers
launch/world_node.launch.py
src/nodes/         world_model_node entrypoint + pipeline / continuous / TF-ROI / services split
src/world_model/   State manager, registration / refine / scene-discovery flows, config loader, block filter
src/utils/         Block geometry, image / coarse-pose helpers, visualization
test/              world_model_utils, block_filter, continuous_perception (gtest)
```

## Dependencies & interactions

| Direction | Package | Via |
|---|---|---|
| **in** | [concrete_block_perception](../concrete_block_perception/) | `TrackedDetectionArray` sub; `RegisterBlock` action client; `ExtractMaskCutout` / yolos segment service clients |
| **in/out** | [concrete_block_perception_interfaces](../concrete_block_perception_interfaces/) | transient detection / registration contracts |
| **out** | [concrete_block_world_model_interfaces](../concrete_block_world_model_interfaces/) | publishes `BlockArray`; serves the persistent query/write services |
| **lib** | `ros2_yolos_cpp` | vendored YOLO/ONNX inference used in the perception flows |
| **clients of this node** | [concrete_block_behavior_tree](../concrete_block_behavior_tree/), [concrete_block_assembly_planning](../concrete_block_assembly_planning/), [concrete_block_viz_common](../concrete_block_viz_common/), [concrete_block_rviz_plugins](../concrete_block_rviz_plugins/) | task-status / goal writes, block subscriptions, RViz panels |

This package depends on **no other CBS runtime package** — only on the two interface packages and the vendored YOLO lib — which keeps it the hub every other package talks *to*.

## Build & run

```bash
colcon build --packages-select concrete_block_world_model --symlink-install
source install/setup.bash
ros2 launch concrete_block_world_model world_node.launch.py
```
