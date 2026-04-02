# Joint & Region Mapping: semi-signals vs VSViG

## VSViG's Skeleton (15 joints)

VSViG uses **15 of the 17 COCO joints**, dropping both ears (indices 3, 4). It reorders them into 5 **equal-sized partitions of 3** — structurally critical to the inter/intra MRGC graph convolutions.

| Partition | VSViG local idx | COCO-17 source | Joints                          |
|-----------|----------------|----------------|---------------------------------|
| head      | 0, 1, 2        | 0, 1, 2        | nose, L eye, R eye              |
| r_arm     | 3, 4, 5        | 6, 8, 10       | R shoulder, R elbow, R wrist    |
| r_leg     | 6, 7, 8        | 12, 14, 16     | R hip, R knee, R ankle          |
| l_arm     | 9, 10, 11      | 5, 7, 9        | L shoulder, L elbow, L wrist    |
| l_leg     | 12, 13, 14     | 11, 13, 15     | L hip, L knee, L ankle          |

## Our Regions (8 regions, 17+42 joints)

| Region  | COCO-17 indices | Extra joints           | VSViG equivalent        |
|---------|----------------|------------------------|-------------------------|
| head    | 0, 1, 2        | —                      | head (exact match)      |
| l_arm   | 5, 7           | —                      | l_arm minus wrist       |
| r_arm   | 6, 8           | —                      | r_arm minus wrist       |
| l_hand  | 9              | +21 hand keypoints     | wrist portion of l_arm  |
| r_hand  | 10             | +21 hand keypoints     | wrist portion of r_arm  |
| l_leg   | 11, 13, 15     | —                      | l_leg (exact match)     |
| r_leg   | 12, 14, 16     | —                      | r_leg (exact match)     |
| overall | 0–16           | —                      | —                       |

## Alignment Notes

### Exact matches
- **head** uses the same 3 joints as VSViG (nose, L eye, R eye; ears dropped).
- **l_leg** / **r_leg** use the same 3 joints each (hip, knee, ankle).

### Arms → arms + hands
VSViG packs shoulder, elbow, and wrist into each arm partition. We split the wrist into its own region (`l_hand` / `r_hand`) because we extend it with 21 fine-grained hand keypoints. If you merge `l_arm + l_hand` (and likewise right), you recover VSViG's arm partition plus 21 extra finger joints.

### Torso
VSViG has **no torso partition** — shoulders are part of arms, hips part of legs. We follow the same approach: no dedicated torso region.

### Hands
VSViG has no concept of hand/finger granularity — it uses 32×32 Gaussian-filtered RGB patch extraction around the wrist as a visual proxy. We extend this with a full 21-keypoint hand model per side.

### Ears (COCO indices 3, 4)
Dropped, matching VSViG.

## Summary Table

| Aspect             | semi-signals                   | VSViG                            | Gap                              |
|--------------------|--------------------------------|----------------------------------|----------------------------------|
| Total body joints  | 17 (COCO)                      | 15 (COCO minus ears)             | Ears unused in regions           |
| Partitioning       | 8 regions (hands split out)    | 5 strict 3-joint partitions      | Arms split at wrist for hand model |
| Hands              | 21 keypoints × 2               | Wrist patch (visual)             | Extended, not compatible         |
| Torso              | None                           | None                             | Aligned                          |
| Ears               | Dropped from regions           | Dropped                          | Aligned                          |
| Input              | Joint velocities               | 32×32 RGB patches + position     | Different signal                 |
