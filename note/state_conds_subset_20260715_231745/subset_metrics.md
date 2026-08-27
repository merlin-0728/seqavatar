# state_conds subset evaluation

state_experiment: `state_conds`
state_run: `20260715_231745`
high_motion_frac: `0.25`
boundary_width: `8`
crop_size: `128`

Delta is `state_conds - baseline`; negative L1 / LPIPS is better, positive PSNR / SSIM is better.

## high_motion overall

| Method | L1 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| baseline | 0.005441 | 31.831970 | 0.971860 | 31.528705 |
| state_conds | 0.005427 | 31.879028 | 0.972055 | 31.438498 |
| delta | -0.000014 | 0.047058 | 0.000195 | -0.090207 |

## boundary overall

| Method | L1 | PSNR |
|---|---:|---:|
| baseline | 0.051762 | 20.897225 |
| state_conds | 0.051750 | 20.896754 |
| delta | -0.000011 | -0.000471 |

## high_error_crop overall

| Method | L1 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| baseline | 0.082639 | 19.644206 | 0.670262 | 277.129885 |
| state_conds | 0.082447 | 19.671799 | 0.671942 | 274.973602 |
| delta | -0.000192 | 0.027593 | 0.001680 | -2.156282 |

## Per-sequence delta

| Subset | Sequence | dL1 | dPSNR | dSSIM | dLPIPS x1000 |
|---|---|---:|---:|---:|---:|
| high_motion | 0044_11 | 0.000025 | -0.050955 | 0.000151 | 0.114981 |
| high_motion | 0051_09 | -0.000011 | 0.008307 | 0.000134 | -0.011428 |
| high_motion | 0206_04 | -0.000043 | 0.062711 | 0.000249 | -0.022402 |
| high_motion | 0007_04 | -0.000011 | 0.054240 | -0.000032 | -0.499353 |
| high_motion | 0813_05 | -0.000019 | 0.087388 | 0.000331 | -0.329387 |
| high_motion | 0019_10 | -0.000025 | 0.120659 | 0.000338 | 0.206345 |
| boundary | 0044_11 | 0.000161 | -0.048276 | NA | NA |
| boundary | 0051_09 | 0.000474 | -0.108387 | NA | NA |
| boundary | 0206_04 | -0.000117 | 0.022779 | NA | NA |
| boundary | 0007_04 | -0.000263 | 0.074151 | NA | NA |
| boundary | 0813_05 | -0.000082 | 0.016408 | NA | NA |
| boundary | 0019_10 | -0.000239 | 0.040499 | NA | NA |
| high_error_crop | 0044_11 | 0.000206 | -0.039225 | -0.000112 | 0.528345 |
| high_error_crop | 0051_09 | -0.000564 | 0.029004 | 0.000787 | -4.548871 |
| high_error_crop | 0206_04 | -0.000466 | 0.058361 | 0.000926 | -2.462235 |
| high_error_crop | 0007_04 | 0.000190 | -0.014547 | -0.000283 | -1.410451 |
| high_error_crop | 0813_05 | -0.000038 | 0.011944 | 0.001310 | -0.562543 |
| high_error_crop | 0019_10 | -0.000481 | 0.120021 | 0.007454 | -4.481939 |

