# Flow Correctness Validation

- Data root: `/media/image/mxz/human/SeqAvatar/DNA-Rendering`
- Sequences: `0007_04, 0019_10`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Max vertices per pose/view: `4096`
- CUDA_VISIBLE_DEVICES: `0`

| Sequence | Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 8296992 | 0.3663 | 0.5934 | 1.1038 | 0.6173 | 0.3319 | 0.6311 | 0.7882 | 0.0280 | 0.5616 |
| 0019_10 | 8296992 | 1.9007 | 2.9220 | 5.1822 | 0.6505 | 0.3668 | 0.7862 | 0.9076 | 0.4904 | 2.5619 |

## Weighted Aggregate

| Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16593984 | 1.1335 | 1.7577 | 3.1430 | 0.6339 | 0.3493 | 0.7087 | 0.8479 | 0.2592 | 1.5617 |
