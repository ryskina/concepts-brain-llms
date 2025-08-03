from lm_utils import *
import argparse

output_dirname = f"outputs/brain_encoding_whole_brain/"
os.makedirs(output_dirname, exist_ok=True)

parser = argparse.ArgumentParser(description="Run one step of the whole-brain encoding pipeline for a given paradigm")

parser.add_argument("--step", type=str, required=True, choices=["save_betas", "choose_best_layer", "predict"])
parser.add_argument("--paradigm", type=str, choices=["sentences", "pictures", "word_clouds"], 
                    required=True, help="Paradigm to run the experiment for")
parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()), 
                    help="Model to use for the experiment")
parser.add_argument("--id", type=str, choices=PARTICIPANT_IDS, 
                    help="Participant ID; required for saving betas and predicting")
args = parser.parse_args()  


glasser_map, glasser_labels_dict = load_glasser_parcellation()
glasser_label_lookup = {v: k for k, v in glasser_labels_dict.items()}
mask = (glasser_map > 0).astype(float)
mask[mask == 0] = np.nan
betas_fnames = {participant_id: f"{output_dirname}/betas_{args.paradigm}_{participant_id}.csv" for participant_id in PARTICIPANT_IDS}

# Saving the activations in all Glasser parcels for a given participant
if args.step == "save_betas":
    if args.id is None:
        raise ValueError("Participant ID must be provided for this step")
    cache_betas_in_mask(args.id, {"all": mask}, glasser_map, glasser_labels_dict, args.paradigm,
                        out_fpath=betas_fnames[args.id], avg_by_parcel=True)

# Running cross-validation to find the best layer and pooling method for a given model and paradigm
# for each Glasser parcel
if args.step == "choose_best_layer":
    if args.model is None:
        raise ValueError("Model must be provided for this step")
    df_betas = pd.DataFrame()
    for participant_id in tqdm(PARTICIPANT_IDS):
        df_betas = pd.concat([df_betas, pd.read_csv(betas_fnames[participant_id])], ignore_index=True)
    df_betas["voxel_population"] = "all"
    cv_fname = f"{output_dirname}/cv_scores_{args.paradigm}_{args.model}.csv"
    choose_layer_and_pooling(args.model, args.paradigm, df_betas, cv_fname)

# Predicting brain activations in each Glasser area for a given participant, model, and paradigm
if args.step == "predict":
    if args.id is None:
        raise ValueError("Participant ID must be provided for this step")
    if args.model is None:
        raise ValueError("Model must be provided for this step")
    cv_fname = f"{output_dirname}/cv_scores_{args.paradigm}_{args.model}.csv"
    params_by_parcel_and_voxel_population = get_best_setting(cv_fname)
    predict_fname = f"{output_dirname}/predict_{args.paradigm}_{args.model}_{args.id}.csv" 
    predict(args.model, args.id, {"all": mask},
            params_by_parcel_and_voxel_population,
            glasser_map, glasser_labels_dict, args.paradigm,
            out_fpath=predict_fname)
