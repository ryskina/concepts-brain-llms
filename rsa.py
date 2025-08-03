from brain_encoding_quartiles import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

def rsa(participant_id: str, model_name: str, 
        betas_fnames_by_paradigm: Dict[str, str],
        out_fpath: Optional[str] = None,
        permuted: bool = False, 
        text_only: bool = False,
        sig_voxels_only: bool = False):
    """
    Runs RSA for a given participant and model.
    Args:
        participant_id (str): ID of the participant.
        model_name (str): Name of the model to use for prediction (one of the keys in MODEL_CONFIGS).
        betas_fnames_by_paradigm (Dict[str, str]): Dictionary mapping paradigms to paths to files 
            storing the corresponding brain activation DataFrames.
        out_fpath (Optional[str]): Output filename for saving RSA results.
            If None, results are not saved.
        permuted (bool): If True, runs RSA on a random permutation of the data.
        text_only (bool): If True, uses only sentence and word cloud embeddings (for VLMs).
        sig_voxels_only (bool): If True, uses only voxels with significant semantic consistency (p<0.05) 
            in both halves of the data.
    """

    n_layers = MODEL_CONFIGS[model_name].n_layers
    pooling_methods = MODEL_CONFIGS[model_name].pooling_methods
    is_vlm = MODEL_CONFIGS[model_name].is_vlm

    df_betas = pd.DataFrame()
    for paradigm in "sentences", "word_clouds", "pictures":
        assert betas_fnames_by_paradigm[paradigm].endswith(f"{participant_id}.csv") # double-check
        df_paradigm = pd.read_csv(betas_fnames_by_paradigm[paradigm])
        df_betas = pd.concat([df_betas, df_paradigm], ignore_index=True)

    if sig_voxels_only:
        mask = get_intersected_participant_map(participant_id)
        sig_voxels = [str(tuple(v)) for v in np.argwhere(~np.isnan(mask))]
        df_betas = df_betas.query("voxel in @sig_voxels")

    concepts_paradigms_and_indices = df_betas.set_index(["concept", "paradigm", "stimulus_idx"]).\
        sort_index().index.drop_duplicates()

    df_betas = df_betas.groupby(["voxel", "parcel", "concept"]).mean(numeric_only=True).reset_index()

    embeddings_by_pooling_and_layer = defaultdict(list)
    rdms_llm = {}
    for pooling_method in pooling_methods:
        for concept in concepts_paradigms_and_indices.get_level_values("concept").unique():
            embeddings_per_concept_by_layer = defaultdict(list)
            for paradigm in "sentences", "word_clouds", "pictures":
                # Not using "pictures" embeddings for text-only condition
                if paradigm == "pictures" and (not is_vlm or text_only):
                    continue
                stim_filtered = filter_df_by_level(concepts_paradigms_and_indices.to_frame(), \
                                                "concept", concept)
                stim_filtered = filter_df_by_level(stim_filtered, "paradigm", paradigm)
                stim_indices = stim_filtered["stimulus_idx"]
                with open(f"outputs/lm_embeddings/{paradigm}/{model_name}/{pooling_method}_pooling/{concept}.pkl", "rb") as f:
                    concept_embedding_dict = pickle.load(f)
                    for layer in range(n_layers+1):
                        if paradigm == "word_clouds":
                            embeddings_per_concept_by_layer[layer].append(concept_embedding_dict[layer])
                        else:
                            embeddings_per_concept_by_layer[layer] += [concept_embedding_dict[idx][layer] for idx in stim_indices]
            for layer in range(n_layers+1):
                embeddings_by_pooling_and_layer[(pooling_method, layer)].append(
                    np.array(embeddings_per_concept_by_layer[layer]).mean(axis=0)
                    )
    for pooling_method in pooling_methods:
        for layer in range(n_layers+1):
            rdms_llm[(pooling_method, layer)] = pdist(np.array(embeddings_by_pooling_and_layer[(pooling_method, layer)]), metric="correlation")

    rsa_scores = {"pooling": [], "layer": [], "parcel": [], "spearmanr": []}
    for parcel in df_betas["parcel"].unique():
        df_parcel = df_betas.query("parcel == @parcel")
        print(f"RSA in parcel={parcel}")
        df_parcel = df_parcel.pivot(index=["concept"], columns="voxel", values="beta")
        print(f"Found {len(df_parcel.columns)} voxels")
        if len(df_parcel.columns) <= 1:
            continue
        df_parcel = df_parcel.sort_index()
        assert (df_parcel.index == CONCEPTS).all()

        if permuted:
            random.seed(0)
            df_parcel = df_parcel.sample(frac=1, random_state=0)

        rdm_brain = 1 - df_parcel.T.corr().to_numpy()
        rdm_brain = squareform(rdm_brain)

        for pooling_method in pooling_methods:
            for layer in range(n_layers+1):
                rdm_llm = rdms_llm[(pooling_method, layer)]
                rsa_scores["pooling"].append(pooling_method)
                rsa_scores["layer"].append(layer)
                rsa_scores["parcel"].append(parcel)
                rsa_scores["spearmanr"].append(spearmanr(rdm_brain, rdm_llm).correlation) # type: ignore
        
    df_rsa = pd.DataFrame.from_dict(rsa_scores)
    if out_fpath:
        print(f"Saving to {out_fpath}")
        df_rsa.to_csv(out_fpath, index=False)

# ----------------------------------------------------------
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run one step of the RSA pipeline for a given paradigm")

    parser.add_argument("--step", type=str, required=True, choices=["save_betas", "rsa"],)
    parser.add_argument("--paradigm", type=str, choices=["sentences", "pictures", "word_clouds"], 
                        help="Paradigm to run the experiment for; required for 'save_betas', 'choose_best_layer', and 'predict' steps")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), 
                        help="Model to use for the experiment; required for 'choose_best_layer' and 'predict' steps")
    parser.add_argument("--text_only", action='store_true', 
                        help="Use only sentence and word_cloud embeddings (for VLMs)")
    parser.add_argument("--permuted", action='store_true', 
                        help="Permuted RSA baseline")
    parser.add_argument("--id", type=str, choices=PARTICIPANT_IDS, required=True,
                        help="Participant ID; required for 'save_metrics', 'save_betas', 'predict' steps")
    parser.add_argument("--sig_voxels_only", action='store_true', 
                        help="Use only voxels with significant semantic consistency (p<0.05) in both data splits")
    args = parser.parse_args()  

    roi_map = np.load("outputs/roi_map.npy")
    roi_dict = {1: "ROI 1", 2: "ROI 2", 3: "ROI 3"}

    output_dirname = f"outputs/rsa"
    os.makedirs(output_dirname, exist_ok=True)

    # Saving the activations in all voxels of the semantically consistent ROIs
    # for a given participant and paradigm 
    if args.step == "save_betas":
        if args.paradigm is None:
            raise ValueError("Paradigm must be specified for this step")
        if args.id is None:
            raise ValueError("Participant ID must be specified for this step")
        mask = (roi_map > 0).astype(float)
        mask[mask == 0] = np.nan
        cache_betas_in_mask(args.id, {"all": mask}, roi_map, roi_dict, args.paradigm,
                            out_fpath=f"{output_dirname}/betas_{args.paradigm}_{args.id}.csv", avg_by_parcel=False)

    # Running RSA on 180 concepts in each semantically consistent ROI 
    # for a given participant and model,
    # using either all paradigms or only text-based paradigms
    if args.step == "rsa":
        if args.id is None:
            raise ValueError("Participant ID must be specified for this step")
        if args.model is None:
            raise ValueError("Model must be specified for this step")
        if args.permuted:
            out_fpath = f"{output_dirname}/permuted_rsa180_{args.model}_text{'only' if args.text_only else 'image'}{'_sig_only' if args.sig_voxels_only else ''}_{args.id}.csv"
        else:
            out_fpath = f"{output_dirname}/rsa180_{args.model}_text"
            out_fpath += f"{'only' if (args.text_only or not MODEL_CONFIGS[args.model].is_vlm) else 'image'}"
            out_fpath += f"{'_sig_only' if args.sig_voxels_only else ''}_{args.id}.csv"
        betas_fnames = {paradigm: f"{output_dirname}/betas_{paradigm}"
                        f"_{args.id}.csv" for paradigm in ["sentences", "word_clouds", "pictures"]}
        
        rsa(args.id, args.model, betas_fnames,
            out_fpath=out_fpath, permuted=args.permuted,
            text_only=args.text_only, 
            sig_voxels_only=args.sig_voxels_only)