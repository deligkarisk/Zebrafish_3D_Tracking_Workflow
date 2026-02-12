# Zebrafish Tracking Workflow

A codebase for tracking and analyzing zebrafish interactions in health and disease.

Deligkaris, K., Neiman, R., Hiroi, M., O'Shaughnessy, L., Carretero, L., Masai, I., & Stephens, G. (2025). A dataset of fine-grained zebrafish interactions in health and disease [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.17190142](https://doi.org/10.5281/zenodo.17190142)

## Overview

This repository contains code for:
- Camera calibration for 3D tracking
- Preprocessing SLEAP and idTracker data
- Tracking zebrafish in 3D space
- Analyzing zebrafish interactions and behaviors

## Project Structure

- `base_workflow/`: Main scripts for running the workflow
  - `make_calibration_models.py`: Creates calibration models from input data
  - `preprocess_sleap_idtracker_data.py`: Prepares tracking data for analysis
  - `track_experiment.py`: Main tracking script
- `lib/`: Library modules
  - `calibration/`: Camera calibration utilities
  - `idtracker/`: idTracker processing functions
  - `registration/`: Registration methods for aligning data
  - `sleap_idtracker_merge/`: Functions for merging tracking data
  - `tracking/`: Core tracking functionality
  - `tracking_quality/`: Quality assessment tools
  - `various/`: Miscellaneous utility functions

## Requirements
- SLEAP inference results for each of the three cameras
- idTracker inference results for the XY camera
- Python environment with dependencies listed in `environment.yml`

## Usage

### Calibration

```python
# Create calibration models from input data
python base_workflow/make_calibration_models.py
```

### Data Preprocessing

```python
# Preprocess SLEAP and idTracker data
python base_workflow/preprocess_sleap_idtracker_data.py $exp $start_frame $end_frame $num_splits_inference $idtracker_session
```

#### Command-Line Arguments

| Argument | Description |
|---|---|
| `$exp`| Label of the experiment |
| `$start_frame` | Start frame of the tracking |
| `$end_frame` | End frame of the tracking |
| `$num_splits_inference` | Number of splits for SLEAP results (per a single experiment) |
| `$id_tracker_session` | Label of the idTracker session |

### Tracking

```python
# Run tracking on preprocessed data
python base_workflow/track_experiment.py $exp $frame_registration_threshold $id_tracker_sleap_registration_threshold $start_frame_in_video
```

#### Command-Line Arguments

| Argument | Description |
|---|---|
| `$exp`| Label of the experiment |
| `$frame_registration_threshold` | Threshold for cross-camera registration cost |
| `$id_tracker_sleap_registration_threshold` | Threshold for idTracker-SLEAP registration cost |
| `$start_frame_in_video` | Start frame for tracking |

## License

[MIT](https://choosealicense.com/licenses/mit/)
