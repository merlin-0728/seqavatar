# Flow Failure Mode Diagnostics

- Sequences: `0051_09`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Flow run for metric response / PLY: `20260706_010043`

## Direction / Farneback / Boundary

| Sequence | Flow/No | Better No | FB err | High-FB ratio | Boundary Flow/No | NonBoundary Flow/No | dPSNR flow_v2-zero | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0051_09 | 0.7568 | 0.6427 | 0.1059 | 0.0126 | 0.7110 | 0.8372 | 0.0073 | -0.1302 |

## Multi-view and Gaussian KNN Attenuation

| Sequence | Multi-view consistency mean | MV p25 | MV low<0.5 | Gaussian KNN consistency | KNN p25 | KNN low<0.5 | Gaussian count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0051_09 | 0.6849 | 0.4492 | 0.2930 | 0.6723 | 0.4377 | 0.2976 | 50569 |

## Aggregate

```json
{
  "observation_count": 8296992,
  "flow_endpoint_mean_px": 0.9745332687980289,
  "no_flow_mean_px": 1.2876533401685128,
  "fb_error_mean_px": 0.10594814698942391,
  "flow_mag_mean_px": 1.1827667287979684,
  "flow_better_than_no_ratio": 0.6426995470165573,
  "high_fb_ratio": 0.012569133488377475,
  "boundary_flow_error_px": 0.9718955884013197,
  "boundary_no_flow_error_px": 1.3668658852098856,
  "nonboundary_flow_error_px": 0.9784926136254903,
  "nonboundary_no_flow_error_px": 1.168749707692501,
  "multiview_consistency_mean": 0.6848554580893934,
  "gaussian_knn_consistency_mean": 0.6722748240335702,
  "flow_over_no_ratio": 0.7568289060396438,
  "boundary_flow_over_no_ratio": 0.7110394654791481,
  "nonboundary_flow_over_no_ratio": 0.8372131408335141,
  "flow_v2_delta_PSNR": 0.0073341846466092875,
  "flow_v2_delta_SSIM": 1.623282829921191e-05,
  "flow_v2_delta_LPIPS": -0.00013022345956414938
}
```
