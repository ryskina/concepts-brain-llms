from utils import *
from analysis_utils import *

def get_language_selectivity_semantic_consistency(participant_id: str, 
                                                  parcels: List[int], 
                                                  parcel_map: npt.NDArray, 
                                                  parcel_dict: dict,
                                                  out_fpath: str):
    """    
    Computes language selectivity and semantic consistency for a given participant
    and saves the results to a CSV file.    

    Args:
        participant_id (str): The ID of the participant (M{01-17}).
        parcels (List[int]): List of parcel indices to consider in the provided parcellation.
        parcel_map (npt.NDArray): 3D array representing the parcellation to use 
            (in our case, the semantically consistent ROI map).
        parcel_dict (dict): Dictionary mapping parcel indices to names.
        out_fpath (str): Path to save the results CSV file.
    """
    
    participant_data = {"voxel": [], "semantic_consistency": [], 
                        "language_selectivity": [], "parcel": []}
    betas_SN_contrast = nib.load(f"data/language_selectivity_nii/{participant_id}_S-N_contrast.nii").get_fdata() # type: ignore
    mask = (np.isin(parcel_map, parcels)).astype(float)
    mask[mask == 0] = np.nan

    print(f"Mask contains {int(np.nansum(mask))} voxels")

    betas_SN_contrast = np.multiply(betas_SN_contrast, mask)

    events_df = load_events_and_responses(participant_id, mask,
                            ["sentences", "word_clouds", "pictures"],
                            parcel_dict=parcel_dict, 
                            parcel_map=parcel_map, 
                            avg_by_parcel=False, stim_split=None)

    for voxel_coords in tqdm(np.argwhere(~np.isnan(mask))):
        voxel_coords = voxel_coords.tolist()
        cons = compute_consistency_in_voxel(voxel_coords, events_df)
        participant_data["voxel"].append(str(tuple(voxel_coords)))
        participant_data["parcel"].append(parcel_dict.get(int(parcel_map[tuple(voxel_coords)]), "None")) # type: ignore
        participant_data["semantic_consistency"].append(cons)
        participant_data["language_selectivity"].append(betas_SN_contrast[tuple(voxel_coords)])

    participant_df = pd.DataFrame.from_dict(participant_data)
    participant_df["id"] = participant_id
    participant_df.to_csv(out_fpath, index=False)
