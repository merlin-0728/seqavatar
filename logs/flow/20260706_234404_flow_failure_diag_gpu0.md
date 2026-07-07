# Flow Failure Mode Diagnostics

- Sequences: `0007_04, 0019_10`
- Views: train views only `0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46`
- Pose range: `1-99`
- Scale: `0.25`
- Flow run for metric response / PLY: `20260706_010043`

## Direction / Farneback / Boundary

| Sequence | Flow/No | Better No | FB err | High-FB ratio | Boundary Flow/No | NonBoundary Flow/No | dPSNR flow_v2-zero | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.6173 | 0.6311 | 0.0280 | 0.0007 | 0.5235 | 0.7333 | 0.0140 | 0.1800 |
| 0019_10 | 0.6505 | 0.7862 | 0.4904 | 0.1316 | 0.6563 | 0.6196 | -0.0378 | 0.1258 |

## Multi-view and Gaussian KNN Attenuation

| Sequence | Multi-view consistency mean | MV p25 | MV low<0.5 | Gaussian KNN consistency | KNN p25 | KNN low<0.5 | Gaussian count |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.5987 | 0.3840 | 0.3488 | 0.5863 | 0.3720 | 0.3715 | 27973 |
| 0019_10 | 0.7307 | 0.4981 | 0.2509 | 0.7364 | 0.5146 | 0.2418 | 35105 |

## Aggregate

```json
{
  "observation_count": 16593984,
  "flow_endpoint_mean_px": 1.133494153478534,
  "no_flow_mean_px": 1.7577255604900586,
  "fb_error_mean_px": 0.2591908935726694,
  "flow_mag_mean_px": 1.5617209356136874,
  "flow_better_than_no_ratio": 0.708657908793934,
  "high_fb_ratio": 0.06616885975061805,
  "boundary_flow_error_px": 1.2071187183823255,
  "boundary_no_flow_error_px": 1.9057203596508323,
  "nonboundary_flow_error_px": 0.8472119682876172,
  "nonboundary_no_flow_error_px": 1.3187265636102437,
  "multiview_consistency_mean": 0.6646986592589221,
  "gaussian_knn_consistency_mean": 0.6613805897989624,
  "flow_over_no_ratio": 0.6448641238183468,
  "boundary_flow_over_no_ratio": 0.6334185979959278,
  "nonboundary_flow_over_no_ratio": 0.6424470331197598,
  "flow_v2_delta_PSNR": -0.011910057067872515,
  "flow_v2_delta_SSIM": 5.7039409875836444e-05,
  "flow_v2_delta_LPIPS": 0.00015289285608256938
}
```
