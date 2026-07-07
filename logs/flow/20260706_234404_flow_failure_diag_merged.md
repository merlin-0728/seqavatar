# Flow Failure Mode Diagnostics Merged

- Run: `20260706_234404`
- Source: four GPU split diagnostics

| Seq | Flow/No | Better No | FB px | High-FB | Boundary F/No | NonBoundary F/No | MV mean | MV low<0.5 | KNN mean | KNN low<0.5 | dPSNR | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 0.6173 | 0.6311 | 0.0280 | 0.0007 | 0.5235 | 0.7333 | 0.5987 | 0.3488 | 0.5863 | 0.3715 | 0.0140 | 0.1800 |
| 0019_10 | 0.6505 | 0.7862 | 0.4904 | 0.1316 | 0.6563 | 0.6196 | 0.7307 | 0.2509 | 0.7364 | 0.2418 | -0.0378 | 0.1258 |
| 0044_11 | 0.9361 | 0.6281 | 0.0494 | 0.0043 | 0.8383 | 1.0450 | 0.3237 | 0.7916 | 0.3202 | 0.8269 | 0.0028 | 0.0033 |
| 0051_09 | 0.7568 | 0.6427 | 0.1059 | 0.0126 | 0.7110 | 0.8372 | 0.6849 | 0.2930 | 0.6723 | 0.2976 | 0.0073 | -0.1302 |
| 0206_04 | 0.6902 | 0.6740 | 0.0579 | 0.0077 | 0.5866 | 0.9393 | 0.4106 | 0.6361 | 0.4332 | 0.6120 | -0.0101 | 0.0320 |
| 0813_05 | 1.0078 | 0.5033 | 0.1903 | 0.0414 | 0.9104 | 1.2982 | 0.5328 | 0.4792 | 0.5548 | 0.4358 | 0.0438 | -0.3379 |

## Aggregate

```json
{
  "observation_count": 49781952,
  "flow_endpoint_mean_px": 0.8896536316631695,
  "no_flow_mean_px": 1.1949808157738986,
  "flow_better_than_no_ratio": 0.6442353445682484,
  "fb_error_mean_px": 0.1536589178601119,
  "flow_mag_mean_px": 1.1065202905154354,
  "high_fb_ratio": 0.033043099635787686,
  "boundary_flow_error_px": 0.9035259140033364,
  "boundary_no_flow_error_px": 1.2918735696476387,
  "nonboundary_flow_error_px": 0.8098239941311707,
  "nonboundary_no_flow_error_px": 0.9638991409486051,
  "multiview_consistency_mean": 0.5468944328536733,
  "multiview_consistency_low_ratio": 0.46660090789529507,
  "gaussian_knn_consistency_mean": 0.5505369506085489,
  "gaussian_knn_low_ratio": 0.46426665335858874,
  "flow_over_no_ratio": 0.7444919783812665,
  "boundary_flow_over_no_ratio": 0.6993919027616418,
  "nonboundary_flow_over_no_ratio": 0.8401542855761818,
  "flow_v2_delta_PSNR": 0.0033530553181962595,
  "flow_v2_delta_SSIM": 1.4178289307498204e-05,
  "flow_v2_delta_LPIPS": -2.1184883856524908e-05
}
```
