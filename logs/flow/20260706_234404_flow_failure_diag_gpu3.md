# Flow Failure Mode Diagnostics

- Sequences: `0206_04, 0813_05`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Flow run for metric response / PLY: `20260706_010043`

## Direction / Farneback / Boundary

| Sequence | Flow/No | Better No | FB err | High-FB ratio | Boundary Flow/No | NonBoundary Flow/No | dPSNR flow_v2-zero | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0206_04 | 0.6902 | 0.6740 | 0.0579 | 0.0077 | 0.5866 | 0.9393 | -0.0101 | 0.0320 |
| 0813_05 | 1.0078 | 0.5033 | 0.1903 | 0.0414 | 0.9104 | 1.2982 | 0.0438 | -0.3379 |

## Multi-view and Gaussian KNN Attenuation

| Sequence | Multi-view consistency mean | MV p25 | MV low<0.5 | Gaussian KNN consistency | KNN p25 | KNN low<0.5 | Gaussian count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0206_04 | 0.4106 | 0.2271 | 0.6361 | 0.4332 | 0.2568 | 0.6120 | 41396 |
| 0813_05 | 0.5328 | 0.2905 | 0.4792 | 0.5548 | 0.3337 | 0.4358 | 39790 |

## Aggregate

```json
{
  "observation_count": 16593984,
  "flow_endpoint_mean_px": 0.8378830714358003,
  "no_flow_mean_px": 0.9587055840688286,
  "fb_error_mean_px": 0.12412414622256052,
  "flow_mag_mean_px": 0.9704288860905513,
  "flow_better_than_no_ratio": 0.5886392321458186,
  "high_fb_ratio": 0.024531721857752785,
  "boundary_flow_error_px": 0.8444398372165455,
  "boundary_no_flow_error_px": 1.0800221677633415,
  "nonboundary_flow_error_px": 0.8325476586836712,
  "nonboundary_no_flow_error_px": 0.7393531288950672,
  "multiview_consistency_mean": 0.47172796358205005,
  "gaussian_knn_consistency_mean": 0.4940114661701284,
  "flow_over_no_ratio": 0.8739732878990366,
  "boundary_flow_over_no_ratio": 0.7818726896738867,
  "nonboundary_flow_over_no_ratio": 1.1260487392917096,
  "flow_v2_delta_PSNR": 0.016880194346109434,
  "flow_v2_delta_SSIM": -2.6707102855016984e-05,
  "flow_v2_delta_LPIPS": -0.0001529980256843068
}
```
