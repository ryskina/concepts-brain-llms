from utils import *
from analysis_utils import *
import argparse

def save_consistency_map(participant_id: str, output_dir: str,
                         voxel_mask: Optional[npt.ArrayLike] = None,
                         permutation: Optional[bool] = False, 
                         stim_split: Optional[int] = None) -> npt.NDArray:
    """
    Saves the consistency map for a given participant ID into a .npy file.

    Args:
        participant_id (str): The ID of the participant (M{01-17}).
        output_dir (str): The directory where the consistency map will be saved.
        voxel_mask (Optional[npt.ArrayLike]): A mask to restrict the voxels used in the analysis.
        permutation (Optional[bool]): If True, performs a permutation test to determine which voxels have significant consistency (saves binary map).
            If False, computes the raw value of the consistency metric C (saves map of floats).
        stim_split (Optional[int]): If provided, saves the map for a specific half of the stimuli (1 or 2).
    Returns:
        consistency_map (npt.NDArray): The computed consistency map.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    if stim_split is not None:
        output_path = f"{output_dir}/{participant_id}_split{stim_split}.npy"
    else:
        output_path = f"{output_dir}/{participant_id}.npy"

    if voxel_mask is None:
        voxel_mask = get_brain_mask()
        print(f"Subject ID: {participant_id}")
        print(f"Using {int(np.nansum(voxel_mask))} voxels within reference brain mask (SPM TPN)")
    else:
        print(f"Using {int(np.nansum(voxel_mask))} voxels for subject {participant_id} within mask")

    consistency_map = np.zeros((91, 109, 91))
    consistency_map[np.isnan(voxel_mask)] = np.nan

    events_df = load_events_and_responses(participant_id, voxel_mask=voxel_mask, 
                                        paradigms=["sentences", "word_clouds", "pictures"],
                                        stim_split=stim_split)

    for voxel_coords in tqdm(np.argwhere(~np.isnan(voxel_mask))):
        voxel_coords = voxel_coords.tolist()
        if not permutation:
            cons = compute_consistency_in_voxel(voxel_coords, events_df)
        else:
            cons = voxel_consistency_permutation_test(voxel_coords, events_df, p_threshold=0.05)
        consistency_map[tuple(voxel_coords)] = cons

    np.save(output_path, consistency_map)
    return consistency_map

# ------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Saves the consistency map for a given participant ID into a .npy file.")
    parser.add_argument("--id", type=str, choices=PARTICIPANT_IDS, required=True, 
                        help="The ID of the participant (M{01-17})")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="The directory where the consistency map will be saved.")
    parser.add_argument("--permutation", action='store_true', 
                        help="If True, performs a permutation test to determine which voxels have significant consistency (saves binary map)."
                             "If False, computes the raw value of the consistency metric C (saves map of floats).")
    parser.add_argument("--stim_split", type=int, default=None, choices=[1, 2],
                        help="If provided, saves the map for a specific half of the stimuli (1 or 2).")

    args = parser.parse_args()
    
    save_consistency_map(args.id, args.output_dir, 
                         permutation=args.permutation, 
                         stim_split=args.stim_split)
