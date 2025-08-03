import nibabel as nib
import numpy as np
import os
from pprint import pprint
import time
import argparse
from collections import defaultdict
from utils import load_events, PARTICIPANT_IDS

from glmsingle.glmsingle import GLM_single # https://github.com/cvnlab/GLMsingle
from tseriesinterp import tseriesinterp # https://github.com/Charestlab/pyslicetime/blob/main/slicetime/tseriesinterp.py

'''
This script (included for reference) fits a GLM to the fMRI data for a given participant and paradigm.
Adapted from: https://github.com/cvnlab/GLMsingle/blob/main/examples/example1.ipynb
'''

parser = argparse.ArgumentParser()

parser.add_argument('--paradigm', required=True, type=str, choices=['word_clouds', 'pictures', 'sentences'],
                    help='Paradigm to fit the GLM for')
parser.add_argument('--id', required=True, type=str, choices=PARTICIPANT_IDS, 
                    help='Participant ID to fit the GLM for')
parser.add_argument('--output_dir', required=True, type=str, help='Output directory for the results')
parser.add_argument('--input_nii_path_template', required=True, type=str, 
                    help='Template for the path to the preprocessed brain image NIfTI file.' \
                    'It needs to have a slot for the run number, e.g. "/path/to/nii/{}.nii"')
args = parser.parse_args()

events_by_run = load_events(args.id, args.paradigm)
n_conditions = sum([len(e) for e in events_by_run.values()])

stimulus2idx = defaultdict(lambda: len(stimulus2idx))

design_matrix_list = []
brain_images_list = []

os.makedirs(args.output_dir, exist_ok=True)

for run in sorted(list(events_by_run.keys())):
    events = [(e.stimulus, e.onset) for e in events_by_run[run]]

    print(f"Loading brain image for run {run}: {args.input_nii_path_template.format(run)}")
    images = nib.load(args.input_nii_path_template.format(run)) # type: ignore
    xyzt = images.shape # type: ignore
    print(f"Brain image object dimensions: {xyzt}")

    number_of_trs = 2 * xyzt[3]
    design_matrix = np.zeros((number_of_trs, n_conditions), dtype=np.int8)

    for stimulus, onset in events:
        stimulus_idx = stimulus2idx[stimulus]

        if onset < number_of_trs:
            design_matrix[int(onset)][stimulus_idx] = 1
            
        else:
            print(f"IPS mismatch, skipping stimuli from {onset} onwards")
            break

    design_matrix_list.append(design_matrix)

    # Interpolation of brain imaging time series
    # Interpolation function taken from https://github.com/Charestlab/pyslicetime/blob/main/slicetime/tseriesinterp.py
    interpolated = np.zeros([91, 109, 91, number_of_trs], dtype=np.float32)
    for z in range(91):
        horizontal_slice = images.dataobj[:,:,z,:] # type: ignore
        interpolated_slice = tseriesinterp(horizontal_slice, 2, 1)
        interpolated[:, :, z, :] = interpolated_slice
    brain_images_list.append(interpolated)

print(f"Total stimuli found: {len(set(stimulus2idx))}")
assert len(set(stimulus2idx)) == n_conditions

stimdur = 3
tr = 1

opt = dict()

# Using the canonical HRF
opt['wantlibrary'] = 0

opt['wantglmdenoise'] = 1
opt['wantfracridge'] = 1
opt['wantfileoutputs'] = [1,1,1,1]
opt['wantmemoryoutputs'] = [0,0,0,0]
opt['chunklen'] = 50000

# Since trials are never repeated for the same participant, cross-validation is not possible
# Therefore, we set the below parameters (number of PCs, regularization level) manually
opt['pcstop'] = -5
opt['fracs'] = 0.05

glmsingle_obj = GLM_single(opt)

# Visualize all the hyperparameters
pprint(glmsingle_obj.params)

start_time = time.time()

print(f'Running GLMsingle...')
# Run GLMsingle
results_glmsingle = glmsingle_obj.fit(
design_matrix_list,
brain_images_list,
stimdur,
tr,
outputdir=args.output_dir,)

elapsed_time = time.time() - start_time

print(
    '\tElapsed time: ',
    f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}'
)