# Flow View-wise Token Headroom

- Sequences: `0044_11`
- Source train views: `0,4,8,12,16,20,24,28,32,36,40,44`
- Held-out validation train views: `2,6,10,14,18,22,26,30,34,38,42,46`
- Pose range/step: `1-99/5`
- Max vertices: `2048`

| Seq | No px | Collapsed px | Viewwise px | Best-conf px | SMPL oracle px | Viewwise/Collapsed | Viewwise/No | Source views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 0.4276 | 30.8881 | 0.3367 | 0.4145 | 0.0000 | 0.0109 | 0.7873 | 12.0000 |

## Aggregate

```json
{
  "count": 419040,
  "no_motion_mean_px": 0.4275908742946225,
  "collapsed_mean_px": 30.888122134516383,
  "viewwise_mean_px": 0.3366573669927388,
  "best_conf_mean_px": 0.4145190101549107,
  "smpl_oracle_mean_px": 0.0,
  "collapsed_over_no": 72.23756163054493,
  "viewwise_over_no": 0.7873352478537045,
  "best_conf_over_no": 0.9694290385376538,
  "smpl_oracle_over_no": 0.0,
  "viewwise_over_collapsed": 0.010899250058861174
}
```
