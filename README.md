# concrete_block_world_model

Persistent block world model and perception orchestrator for the concrete-block stack. Owns the single `world_model_node` executable that consumes detections + registration results from [concrete_block_perception](../concrete_block_perception/) and serves the persistent block state to the BT and motion planner.

## Role in the stack

```
detection_tracking_node ──TrackedDetections──┐
                                              ▼
                                      world_model_node ◄──► block_registration_node
                                       (this package)              (ICP action)
                                              │
                                              ├─► /cbp/block_world_model         (persistent state)
                                              ├─► /cbp/block_world_model_markers (RViz)
                                              └─► services: GetCoarseBlocks, GetPlanningScene,
                                                            RunPoseEstimation, SetBlockTaskStatus,
                                                            UpsertBlock, SetPerceptionMode
```

The persistent state survives across frames and pipeline modes; perception providers stay stateless.

## Contents

```text
config/   world_model.yaml plus seed YAMLs (none / pick_place / b0)
include/concrete_block_world_model/  Public headers
launch/world_node.launch.py
src/nodes/         world_model_node entrypoint + pipeline / TF-ROI / services split
src/world_model/   State manager, registration / refine / scene-discovery flows, config loader
src/utils/         Block geometry, image / coarse-pose helpers, visualization
test/test_world_model_utils.cpp
```

## Build & run

```bash
colcon build --packages-select concrete_block_world_model --symlink-install
source install/setup.bash
ros2 launch concrete_block_world_model world_node.launch.py
```

Interfaces are split across [concrete_block_perception_interfaces](../concrete_block_perception_interfaces/) (in: detections, registration) and [concrete_block_world_model_interfaces](../concrete_block_world_model_interfaces/) (out: block state, planning scene).
