# TGSIM Foggy Bottom — calibration-ready outputs

## Files
- `prepared/trajectories_calibration.csv` — highway-compatible schema (`speed_kf`/`acceleration_kf` from x/y, `run_id=1`, `class=type_most_common`)
- `prepared/type_code_note.csv` — counts/sizes per class
- `Foggy_Bottom_boundaries.txt` — provided lane polygons (pixels)
- `derived_boundaries/street_boundaries.csv` — **site curb** (union exterior + holes, meters)
- `derived_boundaries/street_boundaries_meta.csv`

## Regenerate trajectories
```bat
python "data\TGSIM FB\prepare_for_calibration.py"
```

## Regenerate curb boundaries
```bat
python "data\TGSIM FB\build_street_boundaries.py"
```

## Plot boundary + trajectories
```bat
python data\_plot_site_boundaries.py
```
Output: `data/_qa_plots/tgsim_boundaries_with_traj.png`

## Notes
Passenger cars ≈ `class=3`. Do not use PCA-derived per-lane envelopes for this site; use the curb union from the provided lane polygons.
