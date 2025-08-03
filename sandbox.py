from utils import *
from analysis_utils import *
from scipy.stats import sem
from lm_utils import *

for participant_id in PARTICIPANT_IDS:
    for paradigm in ["sentences", "word_clouds", "pictures"]:
        betas_fname = f"outputs/rsa_copy/betas_{paradigm}_{participant_id}.csv"
        betas_df = pd.read_csv(betas_fname)
        betas_df["roi"] = "all"
        betas_df.rename(columns={"roi": "voxel_population"}, inplace=True)
        betas_df.to_csv(betas_fname, index=False)

exit()

for participant_id in PARTICIPANT_IDS:
    for paradigm in ["sentences", "word_clouds", "pictures"]:
        betas_fname = f"outputs/brain_encoding_quartiles/betas_{paradigm}_{participant_id}.csv"
        betas_df = pd.read_csv(betas_fname)
        betas_df["roi"] = "all"
        betas_df.rename(columns={"roi": "voxel_population"}, inplace=True)
        betas_df.to_csv(betas_fname, index=False)

for paradigm in ["sentences", "word_clouds", "pictures"]:
    for model_name in MODEL_CONFIGS.keys():
        cv_fname = f"outputs/brain_encoding_quartiles/cv_scores_{paradigm}_{model_name}.csv"
        if not os.path.exists(cv_fname):
            continue
        df_cv = pd.read_csv(cv_fname)
        df_cv["roi"] = "all"
        df_cv.rename(columns={"roi": "voxel_population"}, inplace=True)
        df_cv.to_csv(cv_fname, index=False)

        for participant_id in PARTICIPANT_IDS:
            predict_fname = f"outputs/brain_encoding_quartiles/predict_{paradigm}_{model_name}_{participant_id}.csv"
            if not os.path.exists(predict_fname):
                continue
            df_predict = pd.read_csv(predict_fname)
            df_predict.rename(columns={"roi": "voxel_population"}, inplace=True)
            df_predict.to_csv(predict_fname, index=False)

exit()

for participant_id in PARTICIPANT_IDS:
    for paradigm in ["sentences", "word_clouds", "pictures"]:
        betas_fname = f"outputs/brain_encoding_whole_brain/betas_{paradigm}_{participant_id}.csv"
        betas_df = pd.read_csv(betas_fname)
        betas_df["roi"] = "all"
        betas_df.rename(columns={"roi": "voxel_population"}, inplace=True)
        betas_df.to_csv(betas_fname, index=False)

for paradigm in ["sentences", "word_clouds", "pictures"]:
    for model_name in MODEL_CONFIGS.keys():
        cv_fname = f"outputs/brain_encoding_whole_brain/cv_scores_{paradigm}_{model_name}.csv"
        if not os.path.exists(cv_fname):
            continue
        df_cv = pd.read_csv(cv_fname)
        df_cv["roi"] = "all"
        df_cv.rename(columns={"roi": "voxel_population"}, inplace=True)
        df_cv.to_csv(cv_fname, index=False)

        for participant_id in PARTICIPANT_IDS:
            predict_fname = f"outputs/brain_encoding_whole_brain/predict_{paradigm}_{model_name}_{participant_id}.csv"
            if not os.path.exists(predict_fname):
                continue
            df_predict = pd.read_csv(predict_fname)
            df_predict["roi"] = "all"
            df_predict.rename(columns={"roi": "voxel_population"}, inplace=True)
            df_predict.to_csv(predict_fname, index=False)

exit()

results_glmsingle = np.load(f"data/GLMsingle_outputs/M01_word_clouds_TYPED_FITHRF_GLMDENOISE_RR.npy",
    allow_pickle=True).item()

print(results_glmsingle["betasmd"].shape)

exit()

df = pd.read_csv("tmp.csv")
corrs = []
for cons_q in range(1, 5):
    df_q = df.query("parcel == 'ROI 3' and cons_q == @cons_q")
    print(df_q)
    exit()
    corr = df_q[["pearsonr", "lang_q"]].corr().values[0,1]
    corrs.append(corr)
print(f"r_L = {np.mean(corrs):.2f} ± {sem(corrs):.2f}")

corrs = []
for lang_q in range(1, 5):
    df_q = df.query("parcel == 'ROI 3' and lang_q == @lang_q")
    corr = df_q[["pearsonr", "cons_q"]].corr().values[0,1]
    corrs.append(corr)
print(f"r_C = {np.mean(corrs):.2f} ± {sem(corrs):.2f}")

exit()

df_cons = pd.read_csv("outputs/semantic_consistency/consistency_by_glasser_area.csv")
glasser_map, glasser_labels_dict = load_glasser_parcellation()

# raw consistency values
vals = {"parcel": [], "consistency": [], "id": []}
for participant_id in tqdm(PARTICIPANT_IDS):
    betas_df = pd.DataFrame()
    for paradigm in "sentences", "word_clouds", "pictures":
        betas_fname = f"outputs/brain_encoding_whole_brain/betas_{paradigm}_{participant_id}.csv"
        betas_df = pd.concat([betas_df, pd.read_csv(betas_fname)], ignore_index=True)

    betas_df = betas_df.drop(columns=["roi", "id"])
    betas_df.set_index(["stimulus", "concept", "stimulus_idx", "paradigm", "parcel"], inplace=True)

    for parcel_idx, parcel_name in glasser_labels_dict.items():
        vals["parcel"].append(parcel_name)
        vals["consistency"].append(compute_consistency_in_parcel(parcel_name, betas_df))
        vals["id"].append(participant_id)

    df_cons2 = pd.DataFrame.from_dict(vals).drop(columns=["id"])
    df_cons2 = df_cons2.groupby("parcel").mean().reset_index()
    df_cons2.rename(columns={"consistency": "mean_consistency"}, inplace=True)

df_cons = df_cons.merge(df_cons2, on="parcel", how="outer")
df_cons.set_index("parcel", inplace=True)
print(df_cons)
print(df_cons.corr())

exit()

evlab_roi = nib.load("data/brain_parcels/allParcels-language-SN220.nii").get_fdata() # type: ignore

glasser_map, glasser_labels_dict = load_glasser_parcellation()

for lang_parcel_idx in range(1, 7):
    print(lang_parcel_idx)
    parcels, counts = np.unique(glasser_map[evlab_roi == lang_parcel_idx], return_counts=True)
    print({glasser_labels_dict[k]: v / df.loc[glasser_labels_dict[k], "size"] for k, v in zip(parcels, counts) if k})
    print("====================")
exit()


mnums = {18: "M07", 
         50: "M16", 
         199: "M13", 
         215: "M03", 
         252: "M17",
         280: "M06", 
         288: "M01", 
         289: "M02",
         296: "M04", 
         298: "M05", 
         343: "M08", 
         361: "M10", 
         366: "M09", 
         407: "M14", 
         408: "M11", 
         411: "M12", 
         426: "M15"}

from lm_utils import *

for paradigm in ["sentences", "word_clouds", "pictures"]:
    para2 = paradigm
    models = MODEL_CONFIGS.keys()
    if paradigm == "pictures":
        para2 = "images"
        models = [model for model in MODEL_CONFIGS.keys() if MODEL_CONFIGS[model].is_vlm]
    for model in models:
        os.makedirs(f"outputs/lm_embeddings/{paradigm}/{model}", exist_ok=True)
        for pooling in "last_tok", "mean", "first_tok":
            if os.path.exists(f"outputs/lm_embeddings/{paradigm}/{model}/{pooling}_pooling"):
                print(f"Skipping {paradigm}, {model}, {pooling} (already exists)")
                continue
            if not os.path.exists(f"/nese/mit/group/evlab/u/ryskina/glmsingle/files/embeddings/{para2[:-1]}/{model}/{pooling}"):
                continue
            os.symlink(f"/nese/mit/group/evlab/u/ryskina/glmsingle/files/embeddings/{para2[:-1]}/{model}/{pooling}", \
                       f"outputs/lm_embeddings/{paradigm}/{model}/{pooling}_pooling", \
                       target_is_directory=True)

exit()
# glasser_map = load_glasser_parcellation()[0]
# mask = (glasser_map == 82).astype(float) 
# mask[mask == 0] = np.nan

# from consistency_map import save_consistency_map
# save_consistency_map(subject_id=mnums[18],
#                      output_dir="tmp",
#                      voxel_mask=mask, permutation=True,
#                      stim_split=1)

# export_roi_map()

# import pickle

# a = pickle.load(open("data/Ability.pkl", "rb"))
# b = pickle.load(open("outputs/lm_embeddings/sentences/gpt2/last_tok_pooling/Ability.pkl", "rb"))

# for k in a.keys():
#     for kk in a[k].keys():
#         # assert np.all(a[k][kk] == b[k][kk]), f"Mismatch in {k}, {kk}"
#         print(k, kk)
#         print(a[k][kk] - b[k][kk])
#     exit()

# from lm_utils import *

# model_name = "gpt2"
# paradigm = "sentences"
# df_betas = pd.DataFrame()
# UIDS = [18, 50, 199, 215, 252, 280, 288, 289, 296, 298, 343, 361, 366, 407, 408, 411, 426]
# for uid in tqdm(UIDS):
#     df_betas = pd.concat([df_betas, pd.read_csv(f"data/ref_0322-all-glasser/0322-all-glasser/betas_sentences_uid{uid}.csv")], ignore_index=True)
# df_betas["roi"] = "all"

# layers = list(range(MODEL_CONFIGS[model_name].n_layers + 1))
# pooling_methods = MODEL_CONFIGS[model_name].pooling_methods
# is_vlm = MODEL_CONFIGS[model_name].is_vlm
# single_embedding_per_concept = (paradigm == "word_clouds" or (paradigm == "pictures" and not is_vlm))

# cv_scores = {"layer": [], "pooling": [], "fold": [], "pearsonr": [], "roi": [], 
#                 "alpha": [], "parcel": []}

# for roi in df_betas["roi"].unique():
#     y_per_parcel = {}
#     df_betas_roi = df_betas[df_betas["roi"] == roi]
#     df_betas_roi = df_betas_roi.drop(columns=["roi", "uid"])

#     if single_embedding_per_concept:
#         df_betas_roi = df_betas_roi.groupby(["concept", "parcel"]).mean(numeric_only=True)
#     else:    
#         df_betas_roi = df_betas_roi.groupby(["concept", "stimulus_idx", "parcel"]).mean(numeric_only=True)

#     concepts_and_indices = df_betas_roi.groupby(["concept", "stimulus_idx"]). \
#         mean(numeric_only=True).index

#     parcels = df_betas_roi.index.get_level_values(1 if single_embedding_per_concept else 2).\
#         drop_duplicates()

#     for parcel in parcels:
#         df_parcel = filter_df_by_level(df_betas_roi, "parcel", parcel)
#         if len(df_parcel.dropna()) > 0:
#             y_per_parcel[parcel] = df_parcel["beta"].to_numpy()

#     parcels = list(y_per_parcel.keys())
#     for pooling_method in pooling_methods:
#         print(model_name, pooling_method, roi)
#         for layer in tqdm(layers):
#             embedding_dict = {}
#             for concept in df_betas_roi.index.get_level_values(0).drop_duplicates():
#                 with open(f"data/ref_sent_gpt2_embedding/{pooling_method}/{concept}.pkl", "rb") as f:
#                 # with open(f"outputs/lm_embeddings/{paradigm}/{model_name}/{pooling_method}_pooling/{concept}.pkl", "rb") as f:
#                     concept_embedding_dict = pickle.load(f)
#                     if single_embedding_per_concept:
#                         embedding_dict[concept] = concept_embedding_dict[layer]
#                     else:
#                         for idx in range(1, 7):
#                             embedding_dict[(concept, idx)] = concept_embedding_dict[idx][layer]

#             y = np.vstack([y_per_parcel[p] for p in parcels]).T
#             if single_embedding_per_concept:
#                 X = np.vstack([embedding_dict[concept] for concept in concepts_and_indices.get_level_values(0).drop_duplicates()])
#             else:
#                 X = np.vstack([embedding_dict[concept_and_idx] for concept_and_idx in concepts_and_indices])

#             kf = KFold(n_splits=5, shuffle=True, random_state=0)
#             for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
#                 model = RidgeCV(alphas=np.asarray([10 ** x for x in range(-30, 30)]), alpha_per_target=True)
#                 model.fit(X[train_index, :], y[train_index])
#                 y_pred = model.predict(X[test_index, :])

#                 for i, parcel in enumerate(parcels):
#                     if len(y_pred.shape) == 1:
#                         y_pred = y_pred[:, np.newaxis]
#                     r = pearsonr(y_pred[:, i], y[test_index, i]).statistic # type: ignore
#                     cv_scores["fold"].append(fold_idx+1)
#                     if type(model.alpha_) in [float, int]:
#                         cv_scores["alpha"].append(model.alpha_)
#                     else:
#                         cv_scores["alpha"].append(model.alpha_[i]) # type: ignore
#                     cv_scores["pearsonr"].append(r)
#                     cv_scores["layer"].append(layer)
#                     cv_scores["pooling"].append(pooling_method)
#                     cv_scores["roi"].append(roi)
#                     cv_scores["parcel"].append(parcel)

#             df_cv = pd.DataFrame.from_dict(cv_scores)
#             print(df_cv)
#             # df_cv.to_csv(out_fpath, index=False)

# paradigm = "sentences"
# para2 = paradigm
# if paradigm == "pictures":
#     para2 = "images"
# model_name = "gpt2"
# uid = 288

# df_a = pd.read_csv(f"outputs/brain_encoding_whole_brain/predict_{paradigm}_{model_name}_{mnums[uid]}.csv")
# df_a = (df_a.groupby(["id", "parcel", "roi", "best_layer", "best_pooling"]).mean(numeric_only=True).sort_values("pearsonr", ascending=False))
# df_a = df_a.reset_index().set_index("parcel")

# df_b = pd.read_csv(f"data/ref_0322-all-glasser/predict_{para2}_{model_name}_uid{uid}.csv")
# df_b = (df_b.groupby(["uid", "parcel", "roi", "best_layer", "best_aggregation"]).mean(numeric_only=True).sort_values("pearsonr", ascending=False))
# df_b = df_b.reset_index().set_index("parcel")

# print(df_a)
# print(df_b)

# df = pd.merge(df_a, df_b, left_index=True, right_index=True, suffixes=("_a", "_b"))
# df["diff"] = abs(df["pearsonr_a"] - df["pearsonr_b"])
# df.drop(columns=["id", "uid", "roi_a", "roi_b"], inplace=True)
# df.sort_values("diff", ascending=False, inplace=True)

# df.to_csv("tmp.csv")

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     if paradigm == "pictures":
#         para2 = "images"
#     for uid in mnums.keys():
#         fname = f"data/ref_0322-all-glasser/betas_{para2}_uid{uid}.csv"
#         df = pd.read_csv(fname)
#         df["id"] = mnums[uid]
#         df = df.drop(columns=["uid"])
#         df["roi"] = "whole_brain"
#         df["paradigm"] = paradigm
#         df.to_csv(f"outputs/brain_encoding_whole_brain/betas_{paradigm}_{mnums[uid]}.csv", index=False)

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     if paradigm == "pictures":
#         para2 = "images"
#     for uid in mnums.keys():
#         fname = f"data/ref_0315-glasser-quartiles/betas_{para2}_uid{uid}.csv"
#         df = pd.read_csv(fname)
#         df["id"] = mnums[uid]
#         df = df.drop(columns=["uid"])
#         df["roi"] = "all_sem_cons_rois"
#         df["paradigm"] = paradigm
#         roi_mapping = {"posttemp": "ROI 1", "frontal": "ROI 2", "ventral": "ROI 3"}
#         df["parcel"] = df["parcel"].apply(lambda x: roi_mapping[x] if x in roi_mapping else None)
#         df = df[df["parcel"].notnull()]
#         df.to_csv(f"outputs/brain_encoding_quartiles/betas_{paradigm}_{mnums[uid]}.csv", index=False)

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     if paradigm == "pictures":
#         para2 = "images"
#     for uid in tqdm(mnums.keys()):
#         fname = f"data/ref_0327-rsa-extrafair/betas_{para2}_uid{uid}.csv"
#         df = pd.read_csv(fname)
#         df["id"] = mnums[uid]
#         df = df.drop(columns=["uid"])
#         df["roi"] = "all_sem_cons_rois"
#         df["paradigm"] = paradigm
#         roi_mapping = {"posttemp": "ROI 1", "frontal": "ROI 2", "ventral": "ROI 3"}
#         df["parcel"] = df["parcel"].apply(lambda x: roi_mapping[x] if x in roi_mapping else None)
#         df = df[df["parcel"].notnull()]
#         df.to_csv(f"outputs/rsa/betas_{paradigm}_{mnums[uid]}.csv", index=False)


# for uid in mnums.keys():
#     fname = f"data/ref_0315-glasser-quartiles/spm_beta_vs_consistency_uid{uid}_all.csv"
#     df = pd.read_csv(fname)
#     df["id"] = mnums[uid]
#     df = df.drop(columns=["uid"])
#     roi_mapping = {"posttemp": "ROI 1", "frontal": "ROI 2", "ventral": "ROI 3"}
#     df["parcel"] = df["parcel"].apply(lambda x: roi_mapping[x] if x in roi_mapping else None)
#     df = df.rename(columns={"spm_beta_003": "language_selectivity", "consistency": "semantic_consistency"})
#     df = df[df["parcel"].notnull()]
#     df.to_csv(f"outputs/brain_encoding_quartiles/selectivity_vs_consistency_{mnums[uid]}.csv", index=False)

from lm_utils import *

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     if paradigm == "pictures":
#         para2 = "images"
#     for model in MODEL_CONFIGS.keys():
#         fname = f"data/ref_0322-all-glasser/cv_scores_{para2}_{model}.csv"
#         if not os.path.isfile(fname):
#             print("Missing", fname)
#             continue
#         df = pd.read_csv(fname)
#         df.rename(columns={"aggregation": "pooling"}, inplace=True)
#         df["roi"] = "whole_brain"
#         df.to_csv(f"outputs/brain_encoding_whole_brain/cv_scores_{paradigm}_{model}.csv", index=False)

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     models = MODEL_CONFIGS.keys()
#     if paradigm == "pictures":
#         para2 = "images"
#         models = [model for model in MODEL_CONFIGS.keys() if MODEL_CONFIGS[model].is_vlm]
#     for model in models:
#         fname = f"data/ref_0315-glasser-quartiles/cv_scores_{para2}_{model}.csv"
#         if not os.path.isfile(fname):
#             print("Missing", fname)
#             continue
#         df = pd.read_csv(fname)
#         df.rename(columns={"aggregation": "pooling"}, inplace=True)
#         roi_mapping = {"posttemp": "ROI 1", "frontal": "ROI 2", "ventral": "ROI 3"}
#         df["parcel"] = df["parcel"].apply(lambda x: roi_mapping[x] if x in roi_mapping else None)
#         df = df[df["parcel"].notnull()]
#         df.to_csv(f"outputs/brain_encoding_quartiles/cv_scores_{paradigm}_{model}.csv", index=False)

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     if paradigm == "pictures":
#         para2 = "images"
#     for model in MODEL_CONFIGS.keys():
#         for uid in mnums.keys():
#             fname = f"data/ref_0322-all-glasser/predict_{para2}_{model}_uid{uid}.csv"
#             if not os.path.isfile(fname):
#                 print("Missing", fname)
#                 continue
#             df = pd.read_csv(fname)
#             df.rename(columns={"best_aggregation": "best_pooling"}, inplace=True)
#             df["roi"] = "whole_brain"
#             df["id"] = mnums[uid]
#             df = df.drop(columns=["uid"])
#             df.to_csv(f"outputs/brain_encoding_whole_brain/predict_{paradigm}_{model}_{mnums[uid]}.csv", index=False)

# for paradigm in ["sentences", "word_clouds", "pictures"]:
#     para2 = paradigm
#     models = MODEL_CONFIGS.keys()
#     if paradigm == "pictures":
#         para2 = "images"
#         models = [model for model in MODEL_CONFIGS.keys() if MODEL_CONFIGS[model].is_vlm]
#     for model in models:
#         for uid in mnums.keys():
#             fname = f"data/ref_0315-glasser-quartiles/predict_{para2}_{model}_uid{uid}.csv"
#             if not os.path.isfile(fname):
#                 print("Missing", fname)
#                 continue
#             df = pd.read_csv(fname)
#             df.rename(columns={"best_aggregation": "best_pooling"}, inplace=True)
#             roi_mapping = {"posttemp": "ROI 1", "frontal": "ROI 2", "ventral": "ROI 3"}
#             df["parcel"] = df["parcel"].apply(lambda x: roi_mapping[x] if x in roi_mapping else None)
#             df = df[df["parcel"].notnull()]
#             df["id"] = mnums[uid]
#             df = df.drop(columns=["uid"])
#             df.to_csv(f"outputs/brain_encoding_quartiles/predict_{paradigm}_{model}_{mnums[uid]}.csv", index=False)

models = MODEL_CONFIGS.keys()
for model in models:
    types = ["textonly"]
    if MODEL_CONFIGS[model].is_vlm:
        types += ["textimage"]
    for pref in ["", "permuted_"]:
        for ttype in types:
            for uid in tqdm(mnums.keys()):
                fname = f"data/ref_0327-rsa-extrafair/{pref}rsa180_{model}_{ttype}_uid{uid}.csv"
                if not os.path.isfile(fname):
                    print("Missing", fname)
                    continue
                df = pd.read_csv(fname)
                df.rename(columns={"agg": "pooling"}, inplace=True)
                roi_mapping = {"posttemp": "ROI 1", "frontal": "ROI 2", "ventral": "ROI 3"}
                df["parcel"] = df["parcel"].apply(lambda x: roi_mapping[x] if x in roi_mapping else None)
                df = df[df["parcel"].notnull()]
                df.to_csv(f"outputs/rsa/{pref}rsa180_{model}_{ttype}_{mnums[uid]}.csv", index=False)