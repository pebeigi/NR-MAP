# Lebanon_Jounieh — calibration-ready outputs

## Files
- `prepared/trajectories_calibration.csv` — highway-compatible schema, subsampled to ≈0.1 s
- `prepared/lane_code_map.csv` — string `lane_kf` (e.g. `6-2`) → integer code
- `Jounieh_Road_Boundaries.csv` — **site boundaries** (provided outer + island polygons)

## Regenerate trajectories
```bat
python data\Lebanon_Jounieh\prepare_for_calibration.py
```

## Plot boundary + trajectories
```bat
python data\_plot_site_boundaries.py
```
Output: `data/_qa_plots/jounieh_boundaries_with_traj.png`

## Notes
Do not use PCA-derived corridor envelopes for this site; the provided road polygons are the correct boundaries. Vehicle lengths in this file look unusually small (~1 m); inspect before trusting footprint-sensitive terms.
