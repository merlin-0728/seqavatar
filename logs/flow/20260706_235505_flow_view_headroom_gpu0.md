# Flow View-wise Token Headroom

- Sequences: `0007_04, 0019_10`
- Source train views: `0,4,8,12,16,20,24,28,32,36,40,44`
- Held-out validation train views: `2,6,10,14,18,22,26,30,34,38,42,46`
- Pose range/step: `1-99/5`
- Max vertices: `2048`

| Seq | No px | Collapsed px | Viewwise px | Best-conf px | SMPL oracle px | Viewwise/Collapsed | Viewwise/No | Source views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.5907 | 26.9525 | 0.2964 | 0.4063 | 0.0000 | 0.0110 | 0.5017 | 12.0000 |
| 0019_10 | 3.0843 | 105.1044 | 1.8264 | 2.2496 | 0.0000 | 0.0174 | 0.5922 | 12.0000 |

## Aggregate

```json
{
  "count": 838080,
  "no_motion_mean_px": 1.837532580269074,
  "collapsed_mean_px": 65.48497313887488,
  "viewwise_mean_px": 1.0613863453340924,
  "best_conf_mean_px": 1.3279266982232025,
  "smpl_oracle_mean_px": 0.0,
  "collapsed_over_no": 35.637448740792266,
  "viewwise_over_no": 0.5776149803986992,
  "best_conf_over_no": 0.7226683828532451,
  "smpl_oracle_over_no": 0.0,
  "viewwise_over_collapsed": 0.016208090107683114
}
```
