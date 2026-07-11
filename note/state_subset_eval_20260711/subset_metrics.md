# state_warm_a04 subset evaluation

state_run: `20260710_220456`
high_motion_frac: `0.25`
boundary_width: `8`
crop_size: `128`

Delta is `state_warm_a04 - baseline`; negative L1 / LPIPS is better, positive PSNR / SSIM is better.

## high_motion overall

| Method | L1 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| baseline | 0.005441 | 31.831970 | 0.971860 | 31.528705 |
| state_warm_a04 | 0.005453 | 31.851846 | 0.971923 | 31.490971 |
| delta | 0.000012 | 0.019877 | 0.000063 | -0.037734 |

## boundary overall

| Method | L1 | PSNR |
|---|---:|---:|
| baseline | 0.051762 | 20.897225 |
| state_warm_a04 | 0.051596 | 20.927096 |
| delta | -0.000166 | 0.029871 |

## high_error_crop overall

| Method | L1 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| baseline | 0.082639 | 19.644206 | 0.670262 | 277.129885 |
| state_warm_a04 | 0.082616 | 19.667364 | 0.671457 | 275.862750 |
| delta | -0.000023 | 0.023158 | 0.001195 | -1.267135 |

## Per-sequence delta

| Subset | Sequence | dL1 | dPSNR | dSSIM | dLPIPS x1000 |
|---|---|---:|---:|---:|---:|
| high_motion | 0044_11 | -0.000020 | 0.019326 | 0.000068 | 0.153326 |
| high_motion | 0051_09 | 0.000018 | 0.018672 | 0.000224 | -0.095728 |
| high_motion | 0206_04 | 0.000059 | -0.016685 | -0.000206 | 0.445842 |
| high_motion | 0007_04 | 0.000026 | -0.036397 | -0.000267 | -0.422726 |
| high_motion | 0813_05 | -0.000006 | 0.064787 | 0.000181 | -0.077287 |
| high_motion | 0019_10 | -0.000005 | 0.069557 | 0.000381 | -0.229831 |
| boundary | 0044_11 | -0.000204 | 0.014723 | NA | NA |
| boundary | 0051_09 | -0.000376 | 0.049433 | NA | NA |
| boundary | 0206_04 | -0.000151 | 0.033522 | NA | NA |
| boundary | 0007_04 | -0.000111 | 0.027764 | NA | NA |
| boundary | 0813_05 | 0.000134 | -0.003979 | NA | NA |
| boundary | 0019_10 | -0.000286 | 0.057761 | NA | NA |
| high_error_crop | 0044_11 | -0.000056 | 0.019764 | 0.000196 | -1.293078 |
| high_error_crop | 0051_09 | -0.000186 | 0.035149 | 0.000200 | 1.472194 |
| high_error_crop | 0206_04 | -0.000138 | 0.039515 | 0.002963 | -0.713700 |
| high_error_crop | 0007_04 | 0.000350 | -0.048915 | -0.003591 | 0.439328 |
| high_error_crop | 0813_05 | 0.000236 | 0.017254 | 0.000736 | -0.972434 |
| high_error_crop | 0019_10 | -0.000345 | 0.076179 | 0.006663 | -6.535121 |

