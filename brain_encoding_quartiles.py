from lm_utils import *
from utils import *
from analysis_utils import *
from ast import literal_eval
from lang_selectivity_vs_sem_consistency import get_language_selectivity_semantic_consistency

def quantilize(selectivity_vs_consistency_df: pd.DataFrame, 
               n: int = 4) -> pd.DataFrame:
    """    
    Splits the voxels in the DataFrame into quantiles based on their 
    language selectivity and semantic consistency values.

    Args:
        selectivity_vs_consistency_df (pd.DataFrame): DataFrame containing language selectivity 
            and semantic consistency values per voxel.
        n (int): Number of quantiles to create.
            Defaults to 4 (quartiles).
    Returns:
        pd.DataFrame: DataFrame with additional columns for quantiles 
            over language selectivity and semantic consistency.
    """
    selectivity_vs_consistency_df = selectivity_vs_consistency_df.dropna()
    selectivity_vs_consistency_df["cons_quantile"] = 0
    selectivity_vs_consistency_df["lang_quantile"] = 0
    selectivity_vs_consistency_df.set_index("voxel", inplace=True)

    for parcel in selectivity_vs_consistency_df["parcel"].unique():
        df_parcel = selectivity_vs_consistency_df.query("parcel == @parcel")
        for quant in np.linspace(0, 1, n+1)[:-1]:
            cons_threshold = df_parcel.quantile(quant, numeric_only=True)["semantic_consistency"]
            sn_threshold = df_parcel.quantile(quant, numeric_only=True)["language_selectivity"]
            voxels_above_cons_threshold = df_parcel[df_parcel["semantic_consistency"] >= cons_threshold].index
            voxels_above_sn_threshold = df_parcel[df_parcel["language_selectivity"] >= sn_threshold].index
            selectivity_vs_consistency_df.loc[voxels_above_cons_threshold, "cons_quantile"] += 1 # type: ignore
            selectivity_vs_consistency_df.loc[voxels_above_sn_threshold, "lang_quantile"] += 1 # type: ignore
    
    selectivity_vs_consistency_df["quantile"] = "(" + selectivity_vs_consistency_df["cons_quantile"].astype(str) + \
        ", " + selectivity_vs_consistency_df["lang_quantile"].astype(str) + ")"

    return selectivity_vs_consistency_df[["cons_quantile", "lang_quantile", "parcel", "quantile"]].reset_index()


def choose_best_layer_and_pooling_per_bin(model_name: str, 
                                          paradigm: str, 
                                          betas_df: pd.DataFrame, 
                                          cv_fname: str):
    choose_layer_and_pooling(model_name, paradigm, betas_df, cv_fname)


def predict_per_bin(participant_id: str, 
                    model_name: str, 
                    paradigm: str, 
                    parcel_names: list[str], 
                    parcel_map: npt.NDArray, 
                    parcel_dict: dict,
                    params_by_parcel: dict[str, tuple[int, str]],
                    n: int = 4, 
                    out_fname: Optional[str] = None):
    """
    Predicts brain activations for each quantile of language selectivity and semantic consistency
    for a given participant and paradigm, using the specified model and parameters.

    Args:
        participant_id (str): ID of the participant.
        model_name (str): Name of the model to use for prediction (one of the keys in MODEL_CONFIGS).
        paradigm (str): Paradigm for which to run the prediction.
        parcel_names (list[str]): List of parcel names to consider.
        parcel_map (npt.NDArray): 3D parcel map.
        parcel_dict (dict): Dictionary mapping parcel indices to names.
        params_by_parcel (dict[str, tuple[int, str]]): Dictionary mapping parcels 
            to their best layer and pooling settings.
        n (int): Number of quantiles to create. Defaults to 4 (quartiles).
        out_fname (Optional[str]): Output filename for saving predictions. 
            If None, predictions are not saved.
    """

    selectivity_vs_consistency_df = pd.read_csv(f"{output_dirname}/selectivity_vs_consistency_{participant_id}.csv")
    df_quantile = quantilize(selectivity_vs_consistency_df, n=n)

    combined_masks_by_quantile = {}
    params_by_parcel_and_quantile = {p: {} for p in parcel_names}

    for quantile in df_quantile["quantile"].unique():
        mask = np.zeros((91, 109, 91))
        for voxel in df_quantile.query(f"quantile == @quantile")["voxel"].unique():
            mask[literal_eval(voxel)] = 1
        mask[mask == 0] = np.nan
        combined_masks_by_quantile[quantile] = mask
        for parcel in parcel_names:
            params_by_parcel_and_quantile[parcel][quantile] = params_by_parcel[parcel]
        
    predict(model_name, participant_id, combined_masks_by_quantile, params_by_parcel_and_quantile, 
            parcel_map, parcel_dict, paradigm, out_fname)

# ----------------------------------------------------------
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run one step of the quartlie-based brain encoding pipeline for a given paradigm")

    parser.add_argument("--step", type=str, required=True, choices=["save_metrics", "save_betas", "choose_best_layer", "predict"])
    parser.add_argument("--paradigm", type=str, choices=["sentences", "pictures", "word_clouds"], 
                        help="Paradigm to run the experiment for; required for 'save_betas', 'choose_best_layer', and 'predict' steps")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), 
                        help="Model to use for the experiment; required for 'choose_best_layer' and 'predict' steps")
    parser.add_argument("--id", type=str, choices=PARTICIPANT_IDS, 
                        help="Participant ID; required for 'save_metrics', 'save_betas', 'predict' steps")
    args = parser.parse_args()  

    roi_map = np.load("outputs/roi_map.npy")
    roi_dict = {1: "ROI 1", 2: "ROI 2", 3: "ROI 3"}

    output_dirname = f"outputs/brain_encoding_quartiles/"
    os.makedirs(output_dirname, exist_ok=True)

    # Saving the values of semantic consistency and language selectivity
    # for all voxels of the semantically consistent ROIs for a given participant
    if args.step == "save_metrics":
        if args.id is None:
            raise ValueError("Participant ID must be provided for this step")
        get_language_selectivity_semantic_consistency(args.id, parcels=[1, 2, 3], 
                                                      parcel_map=roi_map, parcel_dict=roi_dict,
                                                      out_fpath=f"{output_dirname}/selectivity_vs_consistency_{args.id}.csv")

    # Saving the activations in all voxels of the semantically consistent ROIs
    # for a given participant and paradigm 
    if args.step == "save_betas":
        if args.id is None:
            raise ValueError("Participant ID must be provided for this step") 
        if args.paradigm is None:
            raise ValueError("Paradigm must be provided for this step")
        mask = (roi_map > 0).astype(float)
        mask[mask == 0] = np.nan
        cache_betas_in_mask(args.id, {"all": mask}, roi_map, roi_dict, args.paradigm,
                            out_fpath=f"{output_dirname}/betas_{args.paradigm}_{args.id}.csv", 
                            avg_by_parcel=False)
        
    # Running cross-validation to find the best layer and pooling method for a given model and paradigm
    # for each semantically consistent ROI
    if args.step == "choose_best_layer":
        if args.paradigm is None:
            raise ValueError("Paradigm must be provided for this step")
        if args.model is None:
            raise ValueError("Model must be provided for this step")
        betas_fnames = {participant_id: f"{output_dirname}/betas_{args.paradigm}_{participant_id}.csv" for participant_id in PARTICIPANT_IDS}
        betas_df = pd.DataFrame()
        for participant_id in tqdm(PARTICIPANT_IDS):
            df_uid = pd.read_csv(betas_fnames[participant_id])
            betas_df = pd.concat([betas_df, df_uid], ignore_index=True)
        betas_df["voxel_population"] = "all"
        cv_fname = f"{output_dirname}/cv_scores_{args.paradigm}_{args.model}.csv"
        choose_best_layer_and_pooling_per_bin(args.model, args.paradigm, betas_df, cv_fname)

    # Predicting brain activations in each voxel bin 
    # (intersection of language selectivity quartile and semantic consistency quartile)
    # for a given participant, model, and paradigm
    if args.step == "predict":
        if args.id is None:
            raise ValueError("Participant ID must be provided for this step")
        if args.paradigm is None:
            raise ValueError("Paradigm must be provided for this step")
        if args.model is None:
            raise ValueError("Model must be provided for this step")
        cv_fname = f"{output_dirname}/cv_scores_{args.paradigm}_{args.model}.csv"
        params_by_parcel_all = get_best_setting(cv_fname)
        params_by_parcel = {k: v["all"] for k, v in params_by_parcel_all.items()}
        predict_per_bin(args.id, args.model, args.paradigm, list(roi_dict.values()), 
                    roi_map, roi_dict, params_by_parcel,
                    n=4, out_fname=f"{output_dirname}/predict_{args.paradigm}_{args.model}_{args.id}.csv")
