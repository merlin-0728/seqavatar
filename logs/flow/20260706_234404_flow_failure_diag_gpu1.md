# Flow Failure Mode Diagnostics

- Sequences: `0044_11`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Flow run for metric response / PLY: `20260706_010043`

## Direction / Farneback / Boundary

| Sequence | Flow/No | Better No | FB err | High-FB ratio | Boundary Flow/No | NonBoundary Flow/No | dPSNR flow_v2-zero | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 0.9361 | 0.6281 | 0.0494 | 0.0043 | 0.8383 | 1.0450 | 0.0028 | 0.0033 |

## Multi-view and Gaussian KNN Attenuation

| Sequence | Multi-view consistency mean | MV p25 | MV low<0.5 | Gaussian KNN consistency | KNN p25 | KNN low<0.5 | Gaussian count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 0.3237 | 0.1530 | 0.7916 | 0.3202 | 0.1706 | 0.8269 | 59582 |

## Aggregate

```json
{
  "observation_count": 8296992,
  "flow_endpoint_mean_px": 0.4206340713523185,
  "no_flow_mean_px": 0.4493692653571032,
  "fb_error_mean_px": 0.049375280580787675,
  "flow_mag_mean_px": 0.39205537088616704,
  "flow_better_than_no_ratio": 0.6281182385134275,
  "high_fb_ratio": 0.0042883011096069516,
  "boundary_flow_error_px": 0.34614278442095553,
  "boundary_no_flow_error_px": 0.412890477847599,
  "nonboundary_flow_error_px": 0.5209320972189564,
  "nonboundary_no_flow_error_px": 0.49848575298850856,
  "multiview_consistency_mean": 0.32365789335070205,
  "gaussian_knn_consistency_mean": 0.3201627676795419,
  "flow_over_no_ratio": 0.9360543850680364,
  "boundary_flow_over_no_ratio": 0.838340438911065,
  "nonboundary_flow_over_no_ratio": 1.0450290586960171,
  "flow_v2_delta_PSNR": 0.00284387270609443,
  "flow_v2_delta_SSIM": 8.172293504138395e-06,
  "flow_v2_delta_LPIPS": 3.324495628474755e-06
}
```
