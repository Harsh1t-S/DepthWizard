# Dataset guide

DepthWizard keeps dataset handling generic: provide an RGB image or GeoTIFF,
then optionally provide either benchmark DSM ground truth, a coarse reference
DEM, or sparse GCP elevations. Dataset files and downloaded model artifacts are
not committed to this repository.

## 1. Official ISRO/SAC reference repository

Highest priority: [IMG-PROCESS-SAC/SIH-DepthWizard-2026](https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026).
The older `IMG-PROCESS-SAC/SIH2026` URL redirects there.

Checked on **2026-08-25** at commit
`feb4dc63596fdf8c801d1a4f07ef8f2ff4e107be`: the repository contained only
`README.md`; no imagery, DSM, DEM, LiDAR, metadata, archive, or evaluation format
had been published. Recheck it before every SIH test because it is authoritative
and can change:

```powershell
python scripts/check_official_sac_data.py
```

Any later SAC sample can be passed into the existing CLI/API without changing
the model architecture. Record its original filename, CRS, transform, band
order, NoData value, horizontal/vertical datum, units, and license.

## 2. ISPRS Potsdam — primary development benchmark

Official description: [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx).
Official download page: [ISPRS benchmark downloads](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx).

- 38 tiles, each 6000 x 6000 pixels.
- 5 cm GSD orthophotos and aligned float32 DSMs.
- RGB, IRRG, and RGBIR orthophoto variants are available.
- Orthophoto and DSM share a UTM/WGS84 grid and include world files.
- The official Potsdam download is currently a password-protected 13.3 GB
  archive; the public password is displayed on the download page.

Use Potsdam to measure relative Pearson correlation and same-scene affine-fit
MAE/RMSE. It is a **development benchmark**, not the official ISRO evaluation
dataset. Keep calibration tiles and held-out evaluation tiles separate for any
accuracy claim.

Suggested placement:

```text
data/isprs/potsdam/images/
data/isprs/potsdam/dsm/
```

## 3. ISPRS Vaihingen — secondary development benchmark

Official description: [ISPRS Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx).

- 33 variably sized tiles at 9 cm GSD with aligned float32 DSMs.
- The three image bands are near-infrared, red, and green, not ordinary RGB.
- The official archive is currently 16 GB.

This is useful for domain-shift experiments, but do not silently treat the NIR
band as blue or compare its metrics directly with Potsdam without documenting
preprocessing.

## 4. Compact real urban smoke test

For a quick real-data pipeline check without a multi-gigabyte archive, the DC
Government publishes a [2021 3-inch orthophoto](https://catalog.data.gov/dataset/aerial-photography-orthophoto-2021)
and a [2020 1-m LiDAR DSM](https://catalog.data.gov/dataset/2020-lidar-digital-surface-model)
through public ArcGIS image services. Download a small urban subset resampled
onto one requested grid:

```powershell
python scripts/download_dc_sample.py
python scripts/evaluate_depth.py `
  --image data/sample/dc-urban/rgb_2021.tif `
  --ground-truth data/sample/dc-urban/dsm_2020.tif `
  --output-dir outputs/dc-smoke-test
```

The script writes source URLs, license/attribution, grid metadata, and the
temporal caveat to `SOURCE.json`. The RGB is from 2021 and the DSM from 2020, so
construction, vegetation, and other changes can be genuine disagreement. Use
this pair to verify I/O, inference, calibration, evaluation, and 3D rendering;
do **not** present its numbers as a clean model-accuracy benchmark.

## 5. Coarse DEMs for deployment-style metric calibration

Coarse terrain DEMs provide reference scale/offset and broad terrain trend;
they do not contain roof/tree surface detail and are not interchangeable with
a high-resolution DSM.

### NASA SRTM 30 m

[NASA LP DAAC](https://www.earthdata.nasa.gov/centers/lp-daac) lists
`SRTMGL1.003` at 30 m resolution. Select the tile covering the image in
[Earthdata Search](https://search.earthdata.nasa.gov/search?q=SRTMGL1+V003).
An Earthdata login is required for protected cloud downloads. Prefer the
void-filled SRTMGL1 V003 product and preserve its NoData/vertical-datum metadata.

### NRSC/Bhuvan CartoDEM for India

[NRSC Open EO Data Archive](https://bhuvan-app3.nrsc.gov.in/data) provides
CartoDEM 1 arc-second (approximately 30 m) for India with a Bhuvan account.
The [Bhuvan FAQ](https://bhuvan.nrsc.gov.in/wiki/index.php/Frequently_Asked_Questions)
states an approximate vertical accuracy of 8 m at 90% confidence. Treat that as
a coarse calibration reference and cite the product metadata for the selected
tile; it cannot validate fine roof-level DSM structure.

## Input roles

| Input | Role | Valid claim |
| --- | --- | --- |
| RGB/JPG/PNG only | Monocular inference | Relative depth / Relative DSM |
| RGB GeoTIFF + coarse DEM/SRTM/CartoDEM | Deployment-style affine reference | Estimated metric DSM, limited by reference and registration |
| RGB + at least 3 elevation GCPs | Sparse affine reference | Estimated metric DSM, limited by GCP coverage and quality |
| RGB + full aligned DSM | Benchmark calibration/evaluation | Same-scene feasibility metrics; not deployable inference |
| RGB + calibration reference + separate DSM | Calibrate then evaluate | Holdout-style metrics, if the datasets are genuinely independent |

Always inspect registration, CRS, vertical datum, units, NoData, acquisition
date, band order, and spatial resolution before interpreting a metric result.
