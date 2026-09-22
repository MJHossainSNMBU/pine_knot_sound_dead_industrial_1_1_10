# Package changes

## Initial repository package

* Collected the supplied cropper, hybrid resampler, patch index builder, physical scale viewer, training and evaluation code.
* Replaced personal absolute paths with command line options and WAIKNOT_PROJECT_ROOT.
* Changed the cropper default range from trees 23 through 24 to all 24 trees. Subset runs remain selectable.
* Added new output folder protection to the cropper entry point. Existing data are not overwritten.
* Preserved the network, optimisation defaults, class balancing, label construction and boundary search algorithms.
* Corrected provenance wording so classifier choices are not incorrectly attributed to the study segmentation settings.
* Added portable Slurm jobs, synthetic tests, method notes, an Overleaf results section and the supplied aggregate reference results.
* Simplified repository, script, dataset and result names by removing the former experiment suffix.

The archived experiment was not rerun on the actual Pine data during packaging. Packaging tests use synthetic inputs.
