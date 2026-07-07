# Flow Correctness Validation

- Data root: `/media/image/mxz/human/SeqAvatar/DNA-Rendering`
- Sequences: `0051_09`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Max vertices per pose/view: `4096`
- CUDA_VISIBLE_DEVICES: `2`

| Sequence | Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0051_09 | 8296992 | 0.9745 | 1.2877 | 2.2861 | 0.7568 | 0.4263 | 0.6427 | 0.8067 | 0.1059 | 1.1828 |

## Weighted Aggregate

| Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8296992 | 0.9745 | 1.2877 | 2.2861 | 0.7568 | 0.4263 | 0.6427 | 0.8067 | 0.1059 | 1.1828 |
