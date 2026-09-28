# concrete_block_world_model

Persistent block world model for the concrete-block stack. Owns the single `world_model_node` executable, which measures blocks through [concrete_block_detector](../concrete_block_detector/) and serves the persistent block state to the BT, the wall planner and the motion planner.

## Responsibilities

- Hold the **single source of truth** for block state (id, pose, pose-/task-/goal-status, confidence) — the detector stays stateless.
- **Ask the detector**: every `run_pose_estimation` mode (SCENE_DISCOVERY, REFINE_BLOCK, REFINE_GRASPED) is one `DiscoverBlocks` call plus the world-model decision of what it means. SCENE_DISCOVERY associates every observation; the two refine modes send one pose prior and write back only the detection that matches the addressed block.
- Track grasped blocks by FK (using the grasp offset the BT supplies) while `task_status == TASK_MOVE`.
- Serve the pull-based query / write API and publish the state, the planner's collision scene and the RViz markers.

## Role in the stack

```
                                      world_model_node ──DiscoverBlocks──► concrete_block_detector
                                       (this package)
                                              │
                                              ├─► block_world_model      (BlockArray, persistent state)
                                              ├─► /crane/collision_scene (planner obstacles)
                                              ├─► markers / debug/*      (RViz overlays)
                                              └─► services: GetCoarseBlocks, GetPlanningScene,
                                                            RunPoseEstimation, SetBlockTaskStatus,
                                                            SetBlockGoal, UpsertBlock,
                                                            ClearBlockGoals, ClearWorldModel
```

## Contents

```text
config/   world_model.yaml plus seed YAMLs (none / pick_place / b0 / legacy_2block)
include/concrete_block_world_model/  Public headers
launch/world_node.launch.py
src/nodes/         world_model_node entrypoint + world publishing / grasp-TF / services split
src/world_model/   State manager (association), config loader
src/utils/         Block geometry, collision-scene conversion, visualization
test/              world_model_utils (gtest)
```

## Dependencies & interactions

| Direction | Package | Via |
|---|---|---|
| **in** | [concrete_block_detector](../concrete_block_detector/) | `DiscoverBlocks` service client (runtime only, via the shared interface package) |
| **out** | [concrete_block_world_model_interfaces](../concrete_block_world_model_interfaces/) | publishes `BlockArray`; serves the persistent query/write services |
| **out** | `crane_msgs` | latches `/crane/collision_scene` for the planner |
| **clients of this node** | [concrete_block_behavior_tree](../concrete_block_behavior_tree/), [concrete_block_assembly_planning](../concrete_block_assembly_planning/), [concrete_block_viz_common](../concrete_block_viz_common/), [concrete_block_rviz_plugins](../concrete_block_rviz_plugins/) | task-status / goal writes, block subscriptions, RViz panels |

This package depends on **no other CBS runtime package** — only on interface packages — which keeps it the hub every other package talks *to*. The detector is reached by service name (`scene_discovery.detector_service`), not by a build dependency.

## Build & run

```bash
colcon build --packages-select concrete_block_world_model --symlink-install
source install/setup.bash
ros2 launch concrete_block_world_model world_node.launch.py
```
