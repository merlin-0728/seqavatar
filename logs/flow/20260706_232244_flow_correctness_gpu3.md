# Flow Correctness Validation

- Data root: `/media/image/mxz/human/SeqAvatar/DNA-Rendering`
- Sequences: `0206_04, 0813_05`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Max vertices per pose/view: `4096`
- CUDA_VISIBLE_DEVICES: `3`

| Sequence | Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0206_04 | 8296992 | 0.5576 | 0.8078 | 1.4374 | 0.6902 | 0.3879 | 0.6740 | 0.8217 | 0.0579 | 0.7371 |
| 0813_05 | 8296992 | 1.1182 | 1.1096 | 2.1359 | 1.0078 | 0.5235 | 0.5033 | 0.7538 | 0.1903 | 1.2038 |

## Weighted Aggregate

| Count | Flow err | No-flow err | Minus-flow err | Flow/No | Flow/Minus | Better No | Better Minus | FB err | Flow mag |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16593984 | 0.8379 | 0.9587 | 1.7866 | 0.8490 | 0.4557 | 0.5886 | 0.7878 | 0.1241 | 0.9704 |
