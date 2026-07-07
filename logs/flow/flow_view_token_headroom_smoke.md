# Flow View-wise Token Headroom

- Sequences: `0007_04`
- Source train views: `0,4`
- Held-out validation train views: `2`
- Pose range/step: `1-1/1`
- Max vertices: `512`

| Seq | No px | Collapsed px | Viewwise px | Best-conf px | SMPL oracle px | Viewwise/Collapsed | Viewwise/No | Source views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.0716 | 0.0870 | 0.0818 | 0.0881 | 0.0000 | 0.9405 | 1.1421 | 2.0000 |

## Aggregate

```json
{
  "count": 499,
  "no_motion_mean_px": 0.07162169851567114,
  "collapsed_mean_px": 0.08697336411621125,
  "viewwise_mean_px": 0.08179641074280569,
  "best_conf_mean_px": 0.08806782384741409,
  "smpl_oracle_mean_px": 0.0,
  "collapsed_over_no": 1.2143437801489878,
  "viewwise_over_no": 1.1420618672553302,
  "best_conf_over_no": 1.229624899612573,
  "smpl_oracle_over_no": 0.0,
  "viewwise_over_collapsed": 0.940476565141389
}
```
