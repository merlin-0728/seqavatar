# Flow Failure Mode Diagnostics

- Sequences: `0007_04`
- Views: train views only `0,2`
- Pose range: `1-2`
- Scale: `0.25`
- Flow run for metric response / PLY: `20260706_010043`

## Direction / Farneback / Boundary

| Sequence | Flow/No | Better No | FB err | High-FB ratio | Boundary Flow/No | NonBoundary Flow/No | dPSNR flow_v2-zero | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.9893 | 0.4793 | 0.0035 | 0.0000 | 0.9026 | 1.0337 | 0.0140 | 0.1800 |

## Multi-view and Gaussian KNN Attenuation

| Sequence | Multi-view consistency mean | MV p25 | MV low<0.5 | Gaussian KNN consistency | KNN p25 | KNN low<0.5 | Gaussian count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.9283 | 0.9068 | 0.0152 | 0.9042 | 0.8970 | 0.0581 | 27973 |

## Aggregate

```json
{
  "observation_count": 3812,
  "flow_endpoint_mean_px": 0.08735799116504174,
  "no_flow_mean_px": 0.08830123659609211,
  "fb_error_mean_px": 0.0035288445878136642,
  "flow_mag_mean_px": 0.045608670096217774,
  "flow_better_than_no_ratio": 0.4792759706190976,
  "high_fb_ratio": 0.0,
  "boundary_flow_error_px": 0.05812461839983566,
  "boundary_no_flow_error_px": 0.06439665143389918,
  "nonboundary_flow_error_px": 0.11269739850804604,
  "nonboundary_no_flow_error_px": 0.10902166545852184,
  "multiview_consistency_mean": 0.9283456798928231,
  "gaussian_knn_consistency_mean": 0.9042295159083161,
  "flow_over_no_ratio": 0.9893178683854116,
  "boundary_flow_over_no_ratio": 0.9026031184167778,
  "nonboundary_flow_over_no_ratio": 1.0337156200473077,
  "flow_v2_delta_PSNR": 0.01398468017578125,
  "flow_v2_delta_SSIM": 0.00023841112852096558,
  "flow_v2_delta_LPIPS": 0.00018001659773290296
}
```
