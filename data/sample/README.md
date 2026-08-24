# Local sample-data staging

No imagery, DSM, benchmark result, or model output is bundled in this folder.
Place temporary local inputs here if convenient; `.gitignore` keeps them out of
version control while retaining this guide.

For a live relative-depth check, add an image such as `IMAGE.tif` and run from
the repository root:

```powershell
python scripts/test_depth.py data/sample/IMAGE.tif --output-dir outputs/feasibility
```

For benchmark-only evaluation, add the image's pixel-aligned ground-truth DSM as
`GROUND_TRUTH_DSM.tif`:

```powershell
python scripts/evaluate_depth.py data/sample/IMAGE.tif data/sample/GROUND_TRUTH_DSM.tif --output-dir outputs/evaluation
```

The second command uses the supplied ground truth to fit scale and shift before
reporting error. Its output is a same-scene feasibility diagnostic, not a claim
of deployable metric-depth accuracy. Do not commit licensed ISPRS data, private
imagery, credentials, generated artifacts, or model weights.
