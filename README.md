# Devasthal Site Characterization – Image Quality (Seeing) Pipeline

This repository contains the code I developed for **Devasthal site characterization**, focused on estimating **Image Quality (seeing)** from raw telescope imaging data.

## What this code does
These scripts are designed to run automatically once you provide the **year-wise folder directory**. The pipeline:

- Scans the raw-data directory (organized year-wise)
- Automatically selects valid science frames from the raw dataset
- Filters frames based on quality criteria, such as:
  - Frames that are **well-focused**
  - Frames that contain an **adequate number of detected sources**
- Measures **Image Quality (seeing)** using source FWHM values
- Computes the **median Image Quality** per frame (or per selected set)
- Reports seeing as **FWHM in arcseconds**
- Supports FITS files with **different CCD binning modes**

## Output
The main output is the **median seeing (FWHM in arcseconds)** computed from selected sources in each accepted frame, which can be used for site-quality and performance analysis.

## Notes
- The scripts are intended for raw observational FITS data.
- Different binning settings are handled consistently during the seeing (arcsec) calculation.

The pipeline uses DAOStarFinder for source detection and applies 1D/2D Gaussian fitting for FWHM estimation where required; some utility functions were developed for flexibility and future use and may not be invoked in the current main workflow. 
And some Malayalam (my language ) comments for me 🙂):
