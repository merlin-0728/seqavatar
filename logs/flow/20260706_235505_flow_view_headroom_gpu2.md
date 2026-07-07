# Flow View-wise Token Headroom

- Sequences: `0051_09`
- Source train views: `0,4,8,12,16,20,24,28,32,36,40,44`
- Held-out validation train views: `2,6,10,14,18,22,26,30,34,38,42,46`
- Pose range/step: `1-99/5`
- Max vertices: `2048`

| Seq | No px | Collapsed px | Viewwise px | Best-conf px | SMPL oracle px | Viewwise/Collapsed | Viewwise/No | Source views |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0051_09 | 1.3197 | 143.4352 | 0.8485 | 1.0389 | 0.0000 | 0.0059 | 0.6429 | 12.0000 |

## Aggregate

```json
{
  "count": 419040,
  "no_motion_mean_px": 1.3196509298948833,
  "collapsed_mean_px": 143.4351827098457,
  "viewwise_mean_px": 0.8484522137022033,
  "best_conf_mean_px": 1.0388778789882531,
  "smpl_oracle_mean_px": 0.0,
  "collapsed_over_no": 108.69176041975813,
  "viewwise_over_no": 0.6429368513155116,
  "best_conf_over_no": 0.7872368786729115,
  "smpl_oracle_over_no": 0.0,
  "viewwise_over_collapsed": 0.0059152308218446865
}
```
