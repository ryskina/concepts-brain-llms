from utils import *
from collections import Counter
import random
from tqdm import tqdm
import numpy.typing as npt


def load_events_and_responses(participant_id: str, 
                              voxel_mask: npt.ArrayLike,
                              paradigms: list[str], 
                              parcel_dict: Optional[dict] = None, 
                              parcel_map: Optional[npt.NDArray] = None, 
                              avg_by_parcel: Optional[bool] = False,
                              stim_split: Optional[int] = None) -> pd.DataFrame:
    
    """
    Loads events (stimuli shown) and responses (brain activations) for a given participant, 
    optionally filtering by voxel mask and averaging by parcel.
    
    Args:   
        participant_id (str): The ID of the participant (M{01-17}).
        voxel_mask (npt.ArrayLike): A mask to restrict the voxels used in the analysis.
        paradigms (list[str]): List of paradigms to load events for.
        parcel_dict (Optional[dict]): Dictionary mapping parcel indices to names.
        parcel_map (Optional[npt.NDArray]): 3D array representing the parcellation to use 
            (in our case, either the Glasser parcellation or the semantically consistent ROI map).
        avg_by_parcel (Optional[bool]): If True, averages betas within parcel for each stimulus. 
            If False, saves betas per voxel.
            If True, the `parcel_dict` and `parcel_map` must be provided.
            If False but `parcel_dict` and `parcel_map` are provided, a parcel label is saved for each voxel.
        stim_split (Optional[int]): If provided, saves the map for a specific half of the stimuli (1 or 2).

    Returns:
        pd.DataFrame: A DataFrame containing the loaded events and responses.
    """

    voxel_coords_list = np.argwhere(~np.isnan(voxel_mask))
    save_parcel_info = (parcel_dict and (parcel_map is not None))
    events_dict = {"stimulus": [], "concept": [], "stimulus_idx": [],
                   "beta": [], "paradigm": []}
    if save_parcel_info: events_dict["parcel"] = []
    if not avg_by_parcel: events_dict["voxel"] = []

    for paradigm in paradigms:
        print(f"Loading responses to {paradigm} in {len(voxel_coords_list)} voxels...")
        events_by_run = load_events(participant_id, paradigm)

        stimulus_list = []
        stim_indices_list = []
        concept_list = []
        split_mask = []
        for run in sorted(list(events_by_run.keys())):
            if stim_split is not None:
                stimulus_list += [e.stimulus for e in events_by_run[run] if e.split == stim_split]
                stim_indices_list += [e.stimulus_idx for e in events_by_run[run] if e.split == stim_split]
                concept_list += [e.concept for e in events_by_run[run] if e.split == stim_split]
                split_mask += [e.split == stim_split for e in events_by_run[run]]
            else:
                stimulus_list += [e.stimulus for e in events_by_run[run]]
                concept_list += [e.concept for e in events_by_run[run]]
                stim_indices_list += [e.stimulus_idx for e in events_by_run[run]]

        results_glmsingle = np.load(f"data/GLMsingle_outputs/{participant_id}_{paradigm}_TYPED_FITHRF_GLMDENOISE_RR.npy",
            allow_pickle=True).item()
        
        if not avg_by_parcel:

            for voxel_coords in tqdm(voxel_coords_list):
                voxel_coords = voxel_coords.tolist()
                all_betas = results_glmsingle["betasmd"][tuple(voxel_coords)]
                if stim_split is not None:
                    betas = [beta for beta, include in zip(all_betas, split_mask) if include]
                else:
                    betas = list(all_betas)
                events_dict["stimulus"] += stimulus_list
                events_dict["concept"] += concept_list
                events_dict["stimulus_idx"] += stim_indices_list
                events_dict["beta"] += betas
                events_dict["paradigm"] += [paradigm] * len(betas)
                events_dict["voxel"] += [str(tuple(voxel_coords))] * len(betas)
                if (parcel_dict and (parcel_map is not None)):
                    events_dict["parcel"] += \
                        [parcel_dict.get(int(parcel_map[tuple(voxel_coords)]), "None")] * len(betas)

        else:
            assert parcel_dict and (parcel_map is not None)
            # Averaging betas within each parcel
            for parcel_idx, parcel_name in parcel_dict.items():
                # Check if parcel is present in voxel_mask
                parcel_mask = np.where(np.multiply(parcel_map, voxel_mask) == parcel_idx)
                if len(parcel_mask[0]) > 0:
                    all_betas = results_glmsingle["betasmd"][parcel_mask]
                    all_betas = np.nanmean(all_betas, axis=0)
                    if stim_split is not None:
                        betas = [beta for beta, include in zip(all_betas, split_mask) if include]
                    else:
                        betas = list(all_betas)
                    events_dict["stimulus"] += stimulus_list
                    events_dict["concept"] += concept_list
                    events_dict["stimulus_idx"] += stim_indices_list
                    events_dict["beta"] += list(betas)
                    events_dict["paradigm"] += [paradigm] * len(betas)
                    events_dict["parcel"] += [parcel_name] * len(betas)

        events_df = pd.DataFrame.from_dict(events_dict)
        events_df = events_df.pivot_table(index=[c for c in events_df.columns if c!="beta"], 
                                          values=["beta"])
    return events_df


def compute_consistency_in_voxel(voxel_coords: tuple, events_df: pd.DataFrame, 
                                 concepts: Optional[List[str]] = None) -> float:
    """
    Computes the semantic consistency metric C for a given voxel, 
    i.e. the consistency of responses in a given voxel across different paradigms.

    Args:
        voxel_coords (tuple): Coordinates of the voxel in the format (x, y, z).
        events_df (pd.DataFrame): DataFrame returned by load_events_and_responses.
        concepts (Optional[List[str]]): Subset of concepts to consider for the consistency calculation.
    Returns:
        float: The mean correlation of responses in the voxel across paradigms.
    """

    events_in_voxel_df = filter_df_by_level(events_df, "voxel", str(tuple(voxel_coords)))
    events_in_voxel_df = events_in_voxel_df.groupby(["concept", "paradigm"]).mean(numeric_only=True) \
        .reset_index()
    events_in_voxel_df = events_in_voxel_df.pivot(index="concept", values="beta", columns="paradigm")
    if len(events_in_voxel_df) != 180:
        print(f"ERROR: voxel {tuple(voxel_coords)} -- found {len(events_in_voxel_df)} concepts")
        print(events_in_voxel_df)
        return np.nan
    if not set(events_in_voxel_df.columns) == set(events_df.index.get_level_values("paradigm")):
        print(f"ERROR: voxel {tuple(voxel_coords)} -- found responses only to {list(events_in_voxel_df.columns)}")
        return np.nan

    if concepts: 
        corrs = events_in_voxel_df.loc[concepts].corr().values
    else:
        corrs = events_in_voxel_df.corr().values
    return corrs[np.triu_indices_from(corrs, 1)].mean()


def compute_consistency_in_parcel(parcel_name: str, events_df: pd.DataFrame, 
                                 concepts: Optional[List[str]] = None) -> float:
    """
    Computes the semantic consistency metric C for a given parcel,
    i.e. the consistency of responses in a given parcel across different paradigms.
    Args:
        parcel_name (str): Name of the parcel to compute consistency for.
        events_df (pd.DataFrame): DataFrame returned by load_events_and_responses.
        concepts (Optional[List[str]]): Subset of concepts to consider for the consistency calculation.
        
    Returns:
        float: The mean correlation of responses in the parcel across paradigms.
    """

    events_in_parcel_df = filter_df_by_level(events_df, "parcel", parcel_name)
    events_in_parcel_df = events_in_parcel_df.groupby(["concept", "paradigm"]).mean(numeric_only=True) \
        .reset_index()
    events_in_parcel_df = events_in_parcel_df.pivot(index="concept", values="beta", columns="paradigm")
    assert len(events_in_parcel_df) == 180
    assert set(events_in_parcel_df.columns) == set(events_df.index.get_level_values("paradigm"))

    if concepts: 
        corrs = events_in_parcel_df.loc[concepts].corr().values
    else:
        corrs = events_in_parcel_df.corr().values
    
    return corrs[np.triu_indices_from(corrs, 1)].mean()


def voxel_consistency_permutation_test(voxel_coords: tuple, events_df: pd.DataFrame,
                                       p_threshold: float,
                                       n_permutations: int = 1000) -> float:
    """
    Performs a permutation test to assess the significance of semantic consistency in a given voxel.

    Args:
        voxel_coords (tuple): Coordinates of the voxel in the format (x, y, z).
        events_df (pd.DataFrame): DataFrame returned by load_events_and_responses.
        p_threshold (float): p-value threshold for significance.
            Once the p-value exceeds this threshold (i.e. p_threshold * n_permutations trials were negative), 
            the voxel is rejected.
        n_permutations (int): Number of permutations to perform.

    Returns:
        float: 1 if the voxel passes the significance test, 0 otherwise.
    """

    vc = tuple(voxel_coords)
    seed = int(f"{vc[0]:0>2}{vc[1]:0>3}{vc[2]:0>2}")

    events_in_voxel_df = filter_df_by_level(events_df, "voxel", str(tuple(voxel_coords)))
    events_in_voxel_df = events_in_voxel_df.groupby(["concept", "paradigm"]).mean(numeric_only=True) \
        .reset_index()
    events_in_voxel_df = events_in_voxel_df.pivot(index="concept", values="beta", columns="paradigm")
    if len(events_in_voxel_df) != 180:
        print(f"ERROR: voxel {tuple(voxel_coords)} -- found {len(events_in_voxel_df)} concepts")
        print(events_in_voxel_df)
        return np.nan
    if not set(events_in_voxel_df.columns) == set(["sentences", "pictures", "word_clouds"]):
        print(f"ERROR: voxel {tuple(voxel_coords)} -- found responses only to {list(events_in_voxel_df.columns)}")
        return np.nan

    corrs = events_in_voxel_df.corr().values
    true_corr = corrs[np.triu_indices_from(corrs, 1)].mean()

    permuted_df = events_in_voxel_df.copy()
    comparison = Counter()
    random.seed(seed)
    # np.random.seed(seed)  # this seed wasn't set in the original implementation; 
                            # setting it makes the output reproducible, but slightly different from the original

    for _ in range(n_permutations):
        for col in events_in_voxel_df.columns:
            permuted_df[col] = np.random.permutation(events_in_voxel_df[col].values) # type: ignore
        corrs = permuted_df.corr().values
        permuted_corr = corrs[np.triu_indices_from(corrs, 1)].mean()
        comparison[permuted_corr < true_corr] += 1
        # Rejecting voxels that have p > threshold
        if comparison[False] == np.round(p_threshold * n_permutations):
            return 0

    return 1


def NSD_noiseceiling(data: np.ndarray,
                     NC_n: Optional[int] = None):
    """
    Compute the noise ceiling as in Allen et al., 2021 (NSD), but instead of assuming variance of the data = 1,
    compute the actual "data" variance.
    Implementation from Tuckute et al., 2024: https://github.com/gretatuckute/drive_suppress_brains/blob/main/src/run_analyses/utils.py#L632

    Args
        data (np.ndarray): (n_items, n_UIDs)
        NC_n (int): The n (normalization in the noise ceiling computation (NC = (ncsnr^2 / (ncsnr^2 + 1/NC_n))

    Returns
        noiseceiling (float): noise ceiling in Pearson correlation units
        ncsnr (float): noise ceiling SNR
        sd_signal (float): standard deviation of the signal
        sd_noise (float): standard deviation of the noise

    """
    n_UIDs = data.shape[1]

    if NC_n is None:
        NC_n = n_UIDs # If a manual NC_n is not specified, use the number of UIDs ("trials")

    ### 1. Estimation of the noise standard deviation (sd_noise) ###

    # First compute std across "trials" (participants) and square it
    std_across_trials_sq = np.square(
        np.nanstd(data, axis=1, ddof=1))  # (n_items,) std: estimating the population standard deviation (across the population of people-trials)

    # Mean across items (sentences)
    mean_across_stimuli = np.nanmean(std_across_trials_sq)  # (1,) # OBS: If we run *two* participants, and one of them has just a single nan,
                                                                # then the std_across_trials_sq will be nan, and the mean_across_stimuli will be nan as well.

    # Obtain sd (take sqrt)
    sd_noise = np.sqrt(mean_across_stimuli)  # (1,)
    if sd_noise == 0: # if there is no variance across trials
        print(f'WARNING: sd_noise is {sd_noise} == 0')

    ### 2. Estimation of the signal standard deviation (sd_signal) ###
    # Take mean across trials (participants)
    mean_across_trials = np.nanmean(data, axis=1)  # (n_items,)

    # Take variance across items (sentences)
    var_across_stimuli = np.var(mean_across_trials, ddof=1)  # (1,) # std: population across "items"

    sd_signal_temp = var_across_stimuli - (np.square(sd_noise) / n_UIDs) # pre sqrt. the n is how many "trials" we used to compute sd_noise
    if sd_signal_temp < 0:
        print(f'WARNING: sd_signal_temp is {sd_signal_temp} < 0')
        sd_signal_temp = 0

    sd_signal = np.sqrt(sd_signal_temp)  # (1,) #

    ### 3. Estimation of signal-to-noise and noise ceiling ###
    # Noise ceiling SNR
    ncsnr = sd_signal / sd_noise

    # Fraction variance noise ceiling
    noiseceiling_temp = np.square(ncsnr) / (np.square(ncsnr) + (1 / NC_n))

    # convert to Pearson correlation units
    noiseceiling = np.sqrt(noiseceiling_temp)

    return noiseceiling, ncsnr, sd_signal, sd_noise


def noise_ceiling():
    lst_across_glasser_parsels = [] # Store noise ceiling across ROIs
    glasser_map, glasser_labels_dict = load_glasser_parcellation()

    mask = (glasser_map > 0).astype(float)
    mask[mask == 0] = np.nan

    df_betas = pd.DataFrame()
    for participant_id in tqdm(PARTICIPANT_IDS):
        df_participant = load_events_and_responses(participant_id, mask,
                    ["sentences", "pictures", "word_clouds"], 
                    glasser_labels_dict, glasser_map,
                    avg_by_parcel=True, stim_split=None).reset_index()
        df_participant["id"] = participant_id
        df_betas = pd.concat([df_betas, df_participant], ignore_index=True)

    for parcel_idx in tqdm(glasser_labels_dict.keys()):
        parcel_name = glasser_labels_dict[parcel_idx]
        for paradigm in ["sentences", "pictures", "word_clouds"]:
            df_select = df_betas.query(f"parcel == @parcel_name and paradigm == @paradigm")
            df_across_participants = df_select.pivot(index=['concept', 'stimulus_idx'], columns='id', values='beta')
                    
            n_items = df_across_participants.shape[0]
            n_participants = df_across_participants.shape[1]
            NC_n = n_participants

            temp = df_across_participants.values  # (1080, n_participants)
            assert temp.shape == (1080, n_participants)

            # Compute noise ceiling
            noiseceiling, ncsnr, sd_signal, sd_noise = NSD_noiseceiling(data=temp, NC_n=n_participants)
            
            # Package into df
            df = pd.DataFrame({'noiseceiling': noiseceiling,
                                'ncsnr': ncsnr,
                                'sd_signal': sd_signal,
                                'sd_noise': sd_noise,
                                'parcel': parcel_name,
                                'n_items': n_items,
                                'n_UIDs': n_participants,
                                'NC_n': NC_n,
                                'paradigm': paradigm
                            }, index=[0])

            df.set_index(["parcel", "paradigm"], inplace=True)
            lst_across_glasser_parsels.append(df)

    df_across_glasser_parsels = pd.concat(lst_across_glasser_parsels, axis=0)     
    print(df_across_glasser_parsels)
    df_across_glasser_parsels.to_csv("outputs/noise_ceiling_per_glasser_parcel.csv")   
