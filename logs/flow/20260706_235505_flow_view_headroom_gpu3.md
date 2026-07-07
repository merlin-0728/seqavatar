# Flow View-wise Token Headroom

- Sequences: `0206_04, 0813_05`
- Source train views: `0,4,8,12,16,20,24,28,32,36,40,44`
- Held-out validation train views: `2,6,10,14,18,22,26,30,34,38,42,46`
- Pose range/step: `1-99/5`
- Max vertices: `2048`

| Seq | No px | Collapsed px | Viewwise px | Best-conf px | SMPL oracle px | Viewwise/Collapsed | Viewwise/No | Source views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0206_04 | 0.7869 | 88.5781 | 0.4583 | 0.6901 | 0.0000 | 0.0052 | 0.5824 | 12.0000 |
| 0813_05 | 0.9718 | 85.4189 | 0.8296 | 0.9798 | 0.0000 | 0.0097 | 0.8537 | 12.0000 |

## Aggregate

```json
{
  "count": 838080,
  "no_motion_mean_px": 0.8793702537132716,
  "collapsed_mean_px": 87.00321609752004,
  "viewwise_mean_px": 0.6439292195504882,
  "best_conf_mean_px": 0.834941690700405,
  "smpl_oracle_mean_px": 0.0,
  "collapsed_over_no": 98.93809317535592,
  "viewwise_over_no": 0.7322617712293558,
  "best_conf_over_no": 0.9494768411539276,
  "smpl_oracle_over_no": 0.0,
  "viewwise_over_collapsed": 0.0074012116842752315
}
```
