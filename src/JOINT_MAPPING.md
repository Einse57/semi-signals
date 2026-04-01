# Joint & Region Mapping: Stage 1 vs VSViG (Stage 2)

## VSViG's Skeleton (15 joints)

VSViG uses **15 of the 17 COCO joints**, dropping both ears (indices 3, 4). It reorders them into 5 **equal-sized partitions of 3** — structurally critical to the inter/intra MRGC graph convolutions.

| Partition | VSViG local idx | COCO-17 source | Joints                          |
|-----------|----------------|----------------|---------------------------------|
| head      | 0, 1, 2        | 0, 1, 2        | nose, L eye, R eye              |
| r_arm     | 3, 4, 5        | 6, 8, 10       | R shoulder, R elbow, R wrist    |
| r_leg     | 6, 7, 8        | 12, 14, 16     | R hip, R knee, R ankle          |
| l_arm     | 9, 10, 11      | 5, 7, 9        | L shoulder, L elbow, L wrist    |
| l_leg     | 12, 13, 14     | 11, 13, 15     | L hip, L knee, L ankle          |

Defined in `config.py` as `VSVIG_JOINT_INDICES = [0, 1, 2, 6, 8, 10, 12, 14, 16, 5, 7, 9, 11, 13, 15]`.

## Our Stage 1 Regions (9 regions, 17+42 joints)

| Region  | COCO-17 indices | Extra joints           |
|---------|----------------|------------------------|
| head    | 0, 1, 2, 3, 4  | —                      |
| torso   | 5, 6, 11, 12   | —                      |
| l_arm   | 5, 7           | —                      |
| r_arm   | 6, 8           | —                      |
| l_hand  | 9              | +21 hand keypoints     |
| r_hand  | 10             | +21 hand keypoints     |
| l_leg   | 11, 13, 15     | —                      |
| r_leg   | 12, 14, 16     | —                      |
| overall | 0–16           | —                      |

## Key Differences

### 1. Ears (indices 3, 4)
We include them in `head`; VSViG drops them. VSViG needs exactly 3 joints per partition for the MRGC graph convolutions (5 × 3 = 15). Already handled by `VSVIG_JOINT_INDICES`.

### 2. Torso
We have a dedicated torso region (shoulders + hips). VSViG has **no torso partition** — shoulders are folded into arm partitions, hips into leg partitions. The torso is a relatively rigid reference frame; VSViG cares about limb motion relative to each other.

### 3. Hands
We split hands out as a separate region with 21 detailed finger keypoints each. VSViG treats wrists as the terminal joint of the arm partition. **VSViG has no concept of hand/finger granularity** — it uses 32×32 Gaussian-filtered RGB patch extraction around the wrist as a visual proxy (implemented in `patch_embedding.py`).

### 4. Shoulders/Hips Double-Counted
In our regions, shoulders appear in both `torso` and `l_arm`/`r_arm`; hips in both `torso` and `l_leg`/`r_leg`. VSViG assigns each joint to exactly one partition — no overlap.

## Alignment: Stage 1 → Stage 2

To feed Stage 2 from Stage 1:

1. **Select 15 joints** from the 17-point body pose via `VSVIG_JOINT_INDICES` (drop ears; reorder). Already defined and used by `extract_patches()`.

2. **Extract 32×32 RGB patches** around each of those 15 joints (already implemented in `patch_embedding.py`).

3. **Hand keypoints don't feed Stage 2 directly** — VSViG's graph is fixed at 15 nodes with 5×3 partitions. Options to incorporate hand detail:
   - Use the existing wrist patch (already captures hand appearance visually)
   - Expand the graph beyond 15 nodes (requires redesigning MRGC convolutions)
   - Encode hand motion as an auxiliary feature vector on the wrist node's embedding

4. **Our motion regions are purely for real-time visualization** — they don't need to match VSViG's partitions. The translation happens at the Stage 1→2 boundary when building the clip tensor.

## Summary Table

| Aspect             | Our Stage 1                    | VSViG Stage 2                    | Gap                              |
|--------------------|--------------------------------|----------------------------------|----------------------------------|
| Total body joints  | 17 (COCO)                      | 15 (COCO minus ears)             | Trivial — index select           |
| Partitioning       | 9 overlapping regions          | 5 strict 3-joint partitions      | Different purpose; mapping exists|
| Hands              | 21 keypoints × 2               | Wrist patch (visual)             | No direct path into graph        |
| Torso              | Explicit region                | None (split into arms/legs)      | Stage 1 only                     |
| Ears               | In head region                 | Dropped                          | Handled by `VSVIG_JOINT_INDICES` |
| Input to model     | Joint velocities               | 32×32 RGB patches + position     | Completely different signal      |
