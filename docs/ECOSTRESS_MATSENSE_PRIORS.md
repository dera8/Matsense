# ECOSTRESS Priors for MatSense

Date frozen: 2026-08-08

This document defines the practical ECOSTRESS integration plan for the current MatSense pipeline.

The target sensor is:

- Velodyne `VLP-32C` (the roof-centre LiDAR of the Fortuna recording vehicle)
- near-infrared wavelength: approximately `0.903 um`

The `905nm` names of the CSV columns and templates are kept; the 2 nm difference is negligible for these priors.

The goal is not to claim that MatSense pseudo-reflectance is a direct radiometric estimate of laboratory reflectance. The goal is narrower:

1. use ECOSTRESS as a physics-informed nominal prior;
2. compare the resulting prior with same-road nominal bag medians;
3. build an `ECOSTRESS-informed` ablation profile while keeping weather ratios empirical.

## 1. Why ECOSTRESS is only a prior here

The current MatSense pseudo-reflectance is an empirical quantity derived from:

- LiDAR intensity;
- distance correction;
- material-aware scaling.

It is therefore sensor- and pipeline-dependent.

By contrast, ECOSTRESS provides laboratory spectral reflectance / emissivity profiles. These can be used to define plausible nominal ordering and ranges for materials at the VLP-32C wavelength, but should not be treated as a direct one-to-one ground truth for point-level pseudo-reflectance.

## 2. Material family mapping

The initial class-to-family mapping is stored in:

- [ecostress_material_family_mapping.csv](ecostress_material_family_mapping.csv)

This mapping is intentionally explicit because the four MatSense classes are coarser than the material categories typically available in spectral libraries.

## 3. Prior table

The ECOSTRESS prior template is stored in:

- [ecostress_905nm_priors_template.csv](ecostress_905nm_priors_template.csv)

The file is designed to be filled from manually selected ECOSTRESS spectra at approximately `0.905 um`.

Expected workflow:

1. choose a small number of ECOSTRESS spectra for each class family;
2. extract reflectance values at, or near, `0.905 um`;
3. aggregate them into:
   - `prior_reflectance_905nm`
   - `prior_reflectance_low`
   - `prior_reflectance_high`
4. record the selected spectral IDs in the CSV.

## 4. Same-road nominal summary

The current nominal bag summary is produced from the two same-road nominal runs:

- `nominal_114118`
- `nominal_115904`

The aggregation script is:

- [aggregate_nominal_material_stats.py](tools/aggregate_nominal_material_stats.py)

The resulting output is intended to represent the empirical nominal reference for the current paper setup.

## 5. Comparison script

The comparison script is:

- [compare_ecostress_priors.py](tools/compare_ecostress_priors.py)

It compares:

- ECOSTRESS nominal priors at `0.905 um`
- same-road nominal MatSense pseudo-reflectance medians

The comparison is done at material level, not point level.

Outputs include:

- material-level joined table
- min-max normalized comparison
- linear fit from nominal pseudo-reflectance to ECOSTRESS prior values
- simple plausibility checks against prior low/high ranges, if provided

## 6. Ablation profile builder

The profile builder is:

- [build_ecostress_informed_profile.py](tools/build_ecostress_informed_profile.py)

It creates a separate profile snippet named, by default:

- `ecostress_informed_v1`

Design choice:

- `nominal_base` comes from ECOSTRESS priors, normalized into the MatSense profile scale;
- `weather_ratio` is copied from `realbag_empirical_v1`;
- this keeps the weather component empirical while changing only the nominal material prior.

This is the correct ablation for the current project:

- `empirical-only nominal base`
- versus
- `ECOSTRESS-informed nominal base`

with the same downstream weather ratios.

## 7. Recommended paper wording

Use wording of this kind:

> ECOSTRESS priors were used as physics-informed nominal material priors at the VLP-32C wavelength, while weather-conditioned ratios remained empirically estimated from the real-bag pipeline.

Avoid wording of this kind:

> MatSense pseudo-reflectance was validated as a direct estimate of ECOSTRESS reflectance.

The second statement is stronger than the current pipeline supports.
