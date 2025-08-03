import os
import pandas as pd
import nibabel as nib
from nilearn import image
import numpy as np
from collections import namedtuple
from typing import Optional, List, Dict, Tuple, Any
from ast import literal_eval
from tqdm import tqdm

from _paths import *

######################### GLOBAL VARIABLES #########################

CONCEPTS = ['Ability', 'Accomplished', 'Angry', 'Apartment', 'Applause', 'Argument', 'Argumentatively', 'Art', 
    'Attitude', 'Bag', 'Ball', 'Bar', 'Bear', 'Beat', 'Bed', 'Beer', 'Big', 'Bird', 'Blood', 'Body', 'Brain', 
    'Broken', 'Building', 'Burn', 'Business', 'Camera', 'Carefully', 'Challenge', 'Charity', 'Charming', 
    'Clothes', 'Cockroach', 'Code', 'Collection', 'Computer', 'Construction', 'Cook', 'Counting', 'Crazy', 
    'Damage', 'Dance', 'Dangerous', 'Deceive', 'Dedication', 'Deliberately', 'Delivery', 'Dessert', 'Device', 
    'Dig', 'Dinner', 'Disease', 'Dissolve', 'Disturb', 'Do', 'Doctor', 'Dog', 'Dressing', 'Driver', 'Economy', 
    'Election', 'Electron', 'Elegance', 'Emotion', 'Emotionally', 'Engine', 'Event', 'Experiment', 'Extremely', 
    'Feeling', 'Fight', 'Fish', 'Flow', 'Food', 'Garbage', 'Gold', 'Great', 'Gun', 'Hair', 'Help', 'Hurting', 
    'Ignorance', 'Illness', 'Impress', 'Invention', 'Investigation', 'Invisible', 'Job', 'Jungle', 'Kindness', 
    'King', 'Lady', 'Land', 'Laugh', 'Law', 'Left', 'Level', 'Liar', 'Light', 'Magic', 'Marriage', 'Material', 
    'Mathematical', 'Mechanism', 'Medication', 'Money', 'Mountain', 'Movement', 'Movie', 'Music', 'Nation', 
    'News', 'Noise', 'Obligation', 'Pain', 'Personality', 'Philosophy', 'Picture', 'Pig', 'Plan', 'Plant', 
    'Play', 'Pleasure', 'Poor', 'Prison', 'Professional', 'Protection', 'Quality', 'Reaction', 'Read', 
    'Relationship', 'Religious', 'Residence', 'Road', 'Sad', 'Science', 'Seafood', 'Sell', 'Sew', 'Sexy', 
    'Shape', 'Ship', 'Show', 'Sign', 'Silly', 'Sin', 'Skin', 'Smart', 'Smiling', 'Solution', 'Soul', 'Sound', 
    'Spoke', 'Star', 'Student', 'Stupid', 'Successful', 'Sugar', 'Suspect', 'Table', 'Taste', 'Team', 'Texture', 
    'Time', 'Tool', 'Toy', 'Tree', 'Trial', 'Tried', 'Typical', 'Unaware', 'Usable', 'Useless', 'Vacation', 
    'War', 'Wash', 'Weak', 'Wear', 'Weather', 'Willingly', 'Word']

PARTICIPANT_IDS = [f"M{i:02d}" for i in range(1, 18)]

######################### DATA LOADING #########################

def load_glasser_parcellation() -> Tuple[np.ndarray, Dict[int, str]]:
    glasser_map = image.resample_to_img(GLASSER_PARCELS_NII_PATH, 
                                 target_img=BRAIN_MASK_NII_PATH, 
                                 interpolation='nearest', copy_header=True, force_resample=True).get_fdata() # type: ignore
    # Creating separate parcels IDs for left and right hemispheres
    glasser_map[:46,:,:] += 200
    glasser_map[glasser_map == 200] = 0
    glasser_map = np.round(glasser_map)

    glasser_df_lh = pd.read_csv(GLASSER_PARCEL_NAMES_PATH, sep=' ', names=["parcel_id", "parcel_name"])
    glasser_df_rh = glasser_df_lh.copy()
    glasser_df_rh["parcel_id"] = glasser_df_rh["parcel_id"] + 200
    glasser_df_rh["parcel_name"] = glasser_df_rh["parcel_name"].str.replace("L_", "R_")
    glasser_df = pd.concat([glasser_df_lh, glasser_df_rh], ignore_index=True)

    return glasser_map, glasser_df.set_index("parcel_id")["parcel_name"].to_dict()

Event = namedtuple('Event', ['stimulus', 'concept', 'stimulus_idx', 'onset', 'split'])

def load_events(participant_id: str, paradigm: str) -> Dict[int, List[Event]]:
    events_by_run = {}
    df_stimuli = pd.read_csv(f"data/stimuli/stimuli_order_{participant_id}_{paradigm}.csv")
    df_stimuli = pd.merge(left=df_stimuli, 
                          right=pd.read_csv(f"data/stimuli/split_{paradigm}.csv"),
                          on=["concept", "stimulus_idx"], how='left')
    for run in set(df_stimuli["run"]):
        df_run = df_stimuli[df_stimuli['run'] == run].copy()
        df_run.drop(columns=["run"], inplace=True)
        events_by_run[run] = [Event(*row) for _, row in df_run.iterrows()]
    return events_by_run

######################### BRAIN MASKING #########################

def get_brain_mask() -> np.ndarray:
    mask = nib.load(BRAIN_MASK_NII_PATH).get_fdata() # type: ignore
    mask[mask == 0] = np.nan
    return mask

def get_intersected_participant_map(participant_id: str) -> np.ndarray:
    # All voxels that are consistent w/ p<0.05 in both splits
    split_maps = []
    for split in 1, 2:
        split_map = np.load(f"outputs/semantic_consistency/{participant_id}_split{split}.npy")
        split_map[np.isnan(split_map)] = 0
        split_maps.append(split_map)

    intersected_participant_map = split_maps[0] * split_maps[1]
    intersected_participant_map[intersected_participant_map == 0] = np.nan
    return intersected_participant_map

############################# EXPORTING DATA #########################

def export_probabilistic_consistency_map():
    reference_nii = nib.load(BRAIN_MASK_NII_PATH)  # type: ignore
    subject_maps = []

    for subject_id in tqdm(PARTICIPANT_IDS):
        subject_map = get_intersected_participant_map(subject_id)
        subject_map[np.isnan(subject_map)] = 0
        subject_maps.append(subject_map)

    ni_img = nib.Nifti1Image(np.nanmean(subject_maps, axis=0), # type: ignore
        affine=reference_nii.affine.copy(), # type: ignore
        header=reference_nii.header.copy())
    
    nib.save(ni_img, f"outputs/semantic_consistency/probabilistic_consistency_map.nii") # type: ignore

def export_roi_map():
    glasser_map, glasser_dict = load_glasser_parcellation()
    glasser_lookup = {v: k for k, v in glasser_dict.items()}

    rois = {
        "ROI 1": ["A5", "STSdp", "TPOJ1", "TPOJ2"],
        "ROI 2": ["IFSa", "45", "FOP5"],
        "ROI 3": ["TE2p", "PH"],
    }
    roi_map = np.full(glasser_map.shape, np.nan)
    for idx in range(1, 4):
        glasser_area_indices = [glasser_lookup[f"L_{area}_ROI"] for area in rois[f"ROI {idx}"]]
        roi_map[np.isin(glasser_map, glasser_area_indices)] = idx

    np.save("outputs/roi_map.npy", roi_map)
    # Saving as NIfTI
    reference_nii = nib.load(BRAIN_MASK_NII_PATH)  # type: ignore
    nib.save(nib.Nifti1Image(roi_map, # type: ignore
        affine=reference_nii.affine.copy(), # type: ignore
        header=reference_nii.header.copy()),
        f"outputs/roi_map.nii") # type: ignore

  
######################### PANDAS #########################

def filter_df_by_level(df: pd.DataFrame, level_name: str, value: Any) -> pd.DataFrame:
    return df[np.isin(df.index.get_level_values(level_name), value)]