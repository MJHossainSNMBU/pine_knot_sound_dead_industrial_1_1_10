"""Portable defaults. Command line arguments take precedence over these paths."""
import os
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('WAIKNOT_PROJECT_ROOT', 'work')).expanduser().resolve()
SOURCE_ROOT = PROJECT_ROOT / 'source'
PITH_ROOT = PROJECT_ROOT / 'pith'
CROP_ROOT = PROJECT_ROOT / 'Individual_Knot_Crops'
BLOCK_ROOT = PROJECT_ROOT / 'Individual_Knot_Crops_160x80x80_thesis_fit'
DATASET_ROOT = PROJECT_ROOT / 'Sound_Dead_Dataset_11x80x80_thesis_fit'
PREVIOUS_ROOT = PROJECT_ROOT / 'Individual_Knot_Crops_1x1x10_VolumeAware'
RESULTS_ROOT = PROJECT_ROOT / 'results_wet'
INDEX_CSV = DATASET_ROOT / 'patch_index_11x80x80_thesis_fit.csv'


def require_input_directory(path):
    if not Path(path).is_dir():
        raise FileNotFoundError(f'Input directory does not exist: {path}')


def require_new_output(path):
    path = Path(path)
    if path.name in {'', '.', '..'}:
        raise ValueError('Choose a specific output folder.')
    if path.exists():
        raise FileExistsError(f'Output already exists. Choose a new folder: {path}')
