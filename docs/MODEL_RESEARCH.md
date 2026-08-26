# Depth backbone research and measurements

Last updated: 2026-08-26. All numbers produced on this machine; reproduce with
the commands at the bottom.

## Summary

1. **No remote-sensing height model with released weights exists.** The
   specialist literature is papers without checkpoints.
2. **Metric-depth variants are a trap on aerial imagery** — roughly 50% worse
   RMSE than the current default.
3. `v2-large` is marginally better than `v2-base` but the margin is within
   single-scene noise and costs about 6x the runtime on large images.
4. **Real accuracy headroom is fine-tuning**, not model swapping.

## Accuracy against real LiDAR ground truth

Scene: `data/sample/dc-urban/` — DC 2021 3-inch orthophoto against the 2020 1 m
LiDAR DSM. Benchmark affine calibration, guided refinement disabled, so these
measure the raw prediction. Lower is better.

| Model | MAE | RMSE | RMSE vs default |
| --- | ---: | ---: | ---: |
| `Depth-Anything-V2-Base-hf` (current default) | 6.380 | 8.848 | — |
| `Depth-Anything-V2-Large-hf` | 6.388 | **8.596** | **+2.9%** |
| `Depth-Anything-V2-Metric-Outdoor-Base-hf` | 10.931 | 13.427 | −51.7% |
| `Depth-Anything-V2-Metric-Outdoor-Large-hf` | 10.560 | 13.079 | −47.8% |

The metric variants are anchored to an absolute depth range learned from
ground-level outdoor scenes. That prior does not transfer to nadir imagery and
actively degrades the height field.

Caveat: this is **one tile**, and the RGB is 2021 while the DSM is 2020, so some
disagreement is genuine change. The 2.9% gap between base and large is not a
result — treat base and large as equivalent in accuracy until measured across
several scenes.

## Edge sharpness proxy — and why not to trust it

`scripts/compare_depth_backbones.py` scores backbones without ground truth by
correlating the depth gradient with the image gradient. On the San Diego scene:

| Model | edge alignment | edge contrast | seconds |
| --- | ---: | ---: | ---: |
| `v2-base` | 0.046 | 3.711 | 10.5 |
| `v2-large` | 0.068 | 3.726 | 67.1 |
| `Metric-Outdoor-Large` | 0.075 | 3.674 | 25.0 |

**This proxy ranked Metric-Outdoor first, and ground truth ranked it last.** It
measures whether depth edges sit on image edges, which is a visual-fidelity
property, not an accuracy one. Use it for rendering decisions only; never
promote a model on the strength of it.

All alignment values are low in absolute terms. That is the domain gap: these
backbones were trained on natural ground-level photographs.

`apple/DepthPro-hf` requires input of at least 1536 px and our inference bound
resizes to 768, so it has not been evaluated. Raising
`DEPTHWIZARD_MAX_INPUT_SIZE` would allow a test.

## Guided refinement: display only

An RGB-guided filter (`ml/refine.py`) raises edge alignment from 0.046 to 0.326
and edge contrast from 3.7 to 7.4. Against the DC LiDAR DSM it makes accuracy
**worse**:

| Refinement radius | MAE | RMSE | RMSE change |
| --- | ---: | ---: | ---: |
| off | 6.380 | 8.848 | — |
| 8 | 6.560 | 9.044 | −2.2% |
| 16 | 6.890 | 9.398 | −6.2% |
| 32 | 7.652 | 10.160 | −14.8% |

The filter transfers image albedo into geometry: dark roads read as low, bright
roofs as high. It therefore applies to the render surface only. `depth.npy`,
the calibrated DSM GeoTIFF, and all metrics use the unrefined prediction, which
is verified by MAE and RMSE being bit-identical with refinement on and off.

## Specialist remote-sensing models

None of these ship usable weights.

| Model | Repository | Weights | License |
| --- | --- | --- | --- |
| HTC-DC Net | [zhu-xlab/HTC-DC-Net](https://github.com/zhu-xlab/HTC-DC-Net) | **None** — 0 releases, README covers training only | **None declared** |
| IM2HEIGHT | paper only | **None released** | n/a |
| TSE-Net | [arXiv 2511.13552](https://arxiv.org/pdf/2511.13552) | **None released** | n/a |
| CMT | [ISPRS J. 2025](https://www.sciencedirect.com/science/article/abs/pii/S0924271625002710) | **None released** | n/a |

A repository with no declared license is not safe to use even if weights appear
later.

## The one path with real headroom

[arXiv 2507.09681](https://arxiv.org/pdf/2507.09681) reports that fine-tuning
Depth Anything V2 on remote-sensing data yields up to **24% RMSE reduction** on
urban height reconstruction — an order of magnitude more than any swap measured
above.

Requirements: aligned RGB/nDSM tiles (ISPRS Potsdam is already documented in
`docs/DATASETS.md`, 38 tiles at 5 cm GSD with float32 DSMs), a regression head
on the V2 encoder, and held-out tiles kept strictly separate from any tile used
for calibration.

This is a project, not an afternoon, but it is the only option that moves the
50% accuracy criterion meaningfully.

## Reproduce

```powershell
# Sharpness proxy, no ground truth needed
python scripts/compare_depth_backbones.py --image <scene.jpg>

# Accuracy against the DC LiDAR pair
python scripts/evaluate_depth.py `
  --image data/sample/dc-urban/rgb_2021.tif `
  --ground-truth data/sample/dc-urban/dsm_2020.tif `
  --output-dir outputs/dc-check
```

Select a backbone with `DEPTHWIZARD_MODEL_ID`. Control refinement with
`DEPTHWIZARD_REFINE`, `DEPTHWIZARD_REFINE_RADIUS`, `DEPTHWIZARD_REFINE_EPSILON`.
