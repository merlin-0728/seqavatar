# Flow Correctness Validation

- Data root: `/media/image/mxz/human/SeqAvatar/DNA-Rendering`
- Sequences: `0007_04`
- Views: train views only `0,2`
- Pose range: `1-2`
- Scale: `0.25`
- Max vertices per pose/view: `1024`
- CUDA_VISIBLE_DEVICES: `0`

| Sequence | Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 3812 | 0.0874 | 0.0883 | 0.1208 | 0.9893 | 0.7232 | 0.4793 | 0.6933 | 0.0035 | 0.0456 |

## Weighted Aggregate

| Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3812 | 0.0874 | 0.0883 | 0.1208 | 0.9893 | 0.7232 | 0.4793 | 0.6933 | 0.0035 | 0.0456 |
