---
title: DepthWizard Backend
emoji: 🛰️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DepthWizard Backend

FastAPI service for SIH26175 monocular depth estimation on aerial imagery.
Runs Depth Anything V2 on free CPU Basic hardware.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Service, model, and device status |
| `GET /api/demo` | Precomputed synthetic fixture |
| `POST /api/analyze` | Multipart image upload; returns depth grid and artifact URLs |
| `GET /artifacts/{job_id}/{file}` | Generated artifacts |
| `GET /docs` | OpenAPI reference |

Inference runs on CPU, so a single analyze call takes roughly 20–60 seconds.
`DEPTHWIZARD_MAX_INPUT_SIZE` is set to 518, the model's native training
resolution, which keeps latency bounded without resampling loss.

Source: https://github.com/Harsh1t-S/DepthWizard
