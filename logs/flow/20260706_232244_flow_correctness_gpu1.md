# Flow Correctness Validation

- Data root: `/media/image/mxz/human/SeqAvatar/DNA-Rendering`
- Sequences: `0044_11`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Max vertices per pose/view: `4096`
- CUDA_VISIBLE_DEVICES: `1`

| Sequence | Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 8296992 | 0.4206 | 0.4494 | 0.7155 | 0.9361 | 0.5879 | 0.6281 | 0.7859 | 0.0494 | 0.3921 |

## Weighted Aggregate

| Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8296992 | 0.4206 | 0.4494 | 0.7155 | 0.9361 | 0.5879 | 0.6281 | 0.7859 | 0.0494 | 0.3921 |
