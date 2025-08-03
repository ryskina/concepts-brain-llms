from utils import *
from lm_utils import *
from tqdm import tqdm
import nibabel as nib
from nilearn import plotting, surface, datasets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem

MODEL_FULL_NAMES = {"gpt2": "GPT-2", 
                    "gpt2-medium": "GPT-2 Medium", 
                    "gpt2-large": "GPT-2 Large", 
                    "gpt2-xl": "GPT-2 XL", 
                    "qwen-1.5b": "Qwen2.5-1.5B",
                    "qwen-1.5b-instruct": "Qwen2.5-1.5B-Instruct",
                    "qwen-3b": "Qwen2.5-3B",
                    "qwen-3b-instruct": "Qwen2.5-3B-Instruct",
                    "qwen-7b": "Qwen2.5-7B",
                    "qwen-7b-instruct": "Qwen2.5-7B-Instruct",
                    "vicuna-7b": "Vicuna-1.5-7B",
                    "flava": "FLAVA", 
                    "qwen-3b-vl-instruct": "Qwen2.5-VL-3B-Instruct",
                    "qwen-7b-vl-instruct": "Qwen2.5-VL-7B-Instruct",
                    "llava-7b": "LLaVA-1.5-7B"}

def probabilistic_map_semantic_consistency(): 
    # Fig. 2b, 7
    fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
    ni_img = nib.load(f"outputs/semantic_consistency/probabilistic_consistency_map.nii") # type: ignore

    for hemi in ["left", "right"]:

        surf = surface.vol_to_surf(img=ni_img,
                surf_mesh=fsaverage[f'pial_{hemi}'],
                interpolation='nearest',
                radius=0)
        
        for view in "lateral", "medial":

            fig = plotting.plot_surf_stat_map(surf_mesh=fsaverage[f'infl_{hemi}'],
                                            stat_map=surf,
                                            hemi=hemi,
                                            vmax=0.3,
                                            vmin=0,
                                            symmetric_cbar=False, # type: ignore
                                            colorbar=False,
                                            bg_map=fsaverage[f'sulc_{hemi}'],
                                            cmap="jet",
                                            view=view,
                                            engine="matplotlib"
                                            )
            
            plt.savefig(f"probabilistic_map_semantic_consistency_{hemi}_{view}.png", dpi=300, bbox_inches='tight')


def export_consistency_by_glasser_area():
    # Fig. 8
    # The actual figure is generated in R using ggseg
    glasser_map, glasser_labels_dict = load_glasser_parcellation()
    mean_map = nib.load(f"outputs/semantic_consistency/probabilistic_consistency_map.nii").get_fdata() # type: ignore
    vals = {"parcel": [], "mean_consistency": []}
    for parcel_idx, parcel_name in glasser_labels_dict.items():
        vals["parcel"].append(parcel_name)
        vals["mean_consistency"].append(np.nanmean(mean_map[glasser_map == parcel_idx]))
    df_cons = pd.DataFrame.from_dict(vals).sort_values(by="mean_consistency", ascending=False)
    df_cons.to_csv("outputs/semantic_consistency/consistency_by_glasser_area.csv", index=False)


def plot_brain_encoding_whole_brain(use_raw_consistency_metric: bool = False, 
                                    divide_by_noise_ceiling: bool = False):
    # Fig. 3, 11
    sns.set_style("ticks", {"font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})

    glasser_map, glasser_labels_dict = load_glasser_parcellation()

    if not use_raw_consistency_metric:
        vals = {"parcel": [], "mean_consistency": []}
        mean_map = nib.load(f"outputs/semantic_consistency/probabilistic_consistency_map.nii").get_fdata() # type: ignore
        for parcel_idx, parcel_name in glasser_labels_dict.items():
            vals["parcel"].append(parcel_name)
            vals["mean_consistency"].append(mean_map[glasser_map == parcel_idx].mean())
        df_cons = pd.DataFrame.from_dict(vals)
    else:
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

        df_cons = pd.DataFrame.from_dict(vals).drop(columns=["id"])
        df_cons = df_cons.groupby("parcel").mean().reset_index()
        df_cons.rename(columns={"consistency": "mean_consistency"}, inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3), sharey=True, 
                             gridspec_kw={'width_ratios': [1, 1, 1]})
    for paradigm, ax in zip(["sentences", "pictures", "word_clouds"], axes):
        df_predict = pd.DataFrame()
        if paradigm == "pictures":
            models = ["flava", "llava-7b",
                        "qwen-3b-vl-instruct",
                        "qwen-7b-vl-instruct"]
        else:
            models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                        "flava", "vicuna-7b", "llava-7b",
                        "qwen-1.5b", "qwen-1.5b-instruct",
                        "qwen-3b", "qwen-3b-instruct", "qwen-3b-vl-instruct",
                        "qwen-7b", "qwen-7b-instruct", "qwen-7b-vl-instruct"]
        for model in models:
            for participant_id in tqdm(PARTICIPANT_IDS):
                fname = f"outputs/brain_encoding_whole_brain/predict_{paradigm}_{model}_{participant_id}.csv"
                if not os.path.isfile(fname):
                    print("Missing", fname)
                    continue
                df_uid = pd.read_csv(fname)
                df_predict = pd.concat([df_predict, df_uid], ignore_index=True)
        df_predict = df_predict.drop(columns=["roi"])

        df_all = pd.DataFrame()
        for parcel in df_predict["parcel"].unique():
            df_filtered = df_predict.query("parcel == @parcel")
            df_filtered = df_filtered.drop(columns=["fold", "alpha"])
            df_filtered = df_filtered.groupby(["id", "parcel", "best_layer", "best_pooling"]).mean().reset_index()
            df_all = pd.concat([df_all, df_filtered], ignore_index=True)

        df = df_cons.merge(df_all, left_on="parcel", right_on="parcel")
        df = df.drop(columns=["id", "best_layer", "best_pooling"])

        if divide_by_noise_ceiling:
            noise_ceiling = pd.read_csv("outputs/noise_ceiling_per_glasser_parcel.csv")
            noise_ceiling = noise_ceiling.query("paradigm == @paradigm")
            df = df.merge(noise_ceiling, on=["parcel"])
            df["pearsonr"] /= df["noiseceiling"]
            df.drop(columns=["noiseceiling", "paradigm", "ncsnr", "sd_signal", "sd_noise", "n_items", "n_UIDs", "NC_n"], 
                    inplace=True)
        df = df.groupby(["parcel", "mean_consistency"]).agg(["mean", "sem"]).reset_index()
        df.columns = ["parcel", "mean_consistency", "pearsonr", "pearsonr SEM"]
        
        df.set_index("parcel", inplace=True)
        df["Glasser areas"] = "other"
        df.loc[[f"L_{p}_ROI" for p in ["A5", "STSdp", "TPOJ1", "TPOJ2"]], "Glasser areas"] = "in ROI 1"
        df.loc[[f"L_{p}_ROI" for p in ["TE2p", "PH"]], "Glasser areas"] = "in ROI 2"
        df.loc[[f"L_{p}_ROI" for p in ["IFSa", "45", "FOP5"]], "Glasser areas"] = "in ROI 3"
        print(f"Correlation on {paradigm}: ", df[["mean_consistency", "pearsonr"]].corr().round(2).values[0, 1])

        ax.errorbar(data=df, x="mean_consistency", y="pearsonr", yerr="pearsonr SEM", 
                    fmt='none', elinewidth=0.5, alpha=0.5, ecolor='k', 
                    label=None)
        sns.regplot(df, x="mean_consistency", y="pearsonr", scatter=False,
                    line_kws={'color': 'darkorange', 'linewidth': 1}, ax=ax)

        roi_colors = ['#EE1289', '#00bfff', '#90ee90', 'gray']
        hue_order = ['in ROI 1', 'in ROI 2', 'in ROI 3', 'other']

        sns.scatterplot(data=df, x="mean_consistency", y="pearsonr", hue="Glasser areas", 
                        edgecolor=None,
                        size="Glasser areas", sizes={"other": 22, "in ROI 1": 22, "in ROI 2": 22, "in ROI 3": 22},
                        linewidth=0.5, style="Glasser areas", palette=roi_colors, hue_order=hue_order,
                        markers=['.', 'd', 'o', 's'],
                        ax=ax)
        ax.set_title(f"{paradigm[:-1].capitalize().replace('_', ' ')} paradigm", fontsize=18)
        ax.set_xlabel("Semantic consistency", fontsize=15)
        ax.set_ylabel("Mean LM predictivity", fontsize=15)
        ax.yaxis.set_tick_params(labelbottom=True)
        ax.tick_params(axis='y', labelsize=12)
        ax.tick_params(axis='x', labelsize=12)
        if paradigm != "word_clouds":
            ax.legend_.remove()
        else:
            sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1, 1))
        ax.spines[['right', 'top']].set_visible(False)

    plt.setp(ax.get_legend().get_title(), fontsize=15)
    plt.setp(ax.get_legend().get_texts(), fontsize=15)

    plt.savefig(f"brain_encoding_whole_brain{'_raw_c' if use_raw_consistency_metric else ''}"
                f"{'_nc' if divide_by_noise_ceiling else ''}.pdf", 
        format="pdf", bbox_inches='tight', dpi=300)

plot_brain_encoding_whole_brain(divide_by_noise_ceiling=True)
exit()

def plot_brain_encoding_quartiles_single_model(model: str):
    # Fig. 13-27
    sns.set_style("ticks", {"font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    parcel_names = ["ROI 1", "ROI 2", "ROI 3"]
    is_vlm = MODEL_CONFIGS[model].is_vlm
    if is_vlm:
        paradigms = ["sentences", "pictures", "word_clouds"]
    else:
        paradigms = ["sentences", "word_clouds"]

    fig, axes = plt.subplots(len(parcel_names), 2*len(paradigms),
                            figsize=(28 if is_vlm else 19, 3*len(parcel_names)), 
                            sharex=True, sharey=True)

    corrs_red = {p: [] for p in parcel_names}
    corrs_blue = {p: [] for p in parcel_names}
    for j, paradigm in enumerate(paradigms):

        df = pd.DataFrame()
        for participant_id in PARTICIPANT_IDS:
            df_participant = pd.read_csv(f"outputs/brain_encoding_quartiles/predict_{paradigm}_{model}_{participant_id}.csv")
            df_participant["id"] = participant_id
            df = pd.concat([df, df_participant], ignore_index=True)

        df = df.drop(columns=["best_layer", "best_pooling", "alpha"])
        df = df.groupby(["id", "parcel", "roi"]).mean().reset_index()
        df["cons_q"] = df["roi"].apply(lambda x: literal_eval(x)[0])
        df["lang_q"] = df["roi"].apply(lambda x: literal_eval(x)[1])

        for parcel in parcel_names:
            for lang_q in range(1, 5):
                df_q = df.query("parcel == @parcel and lang_q == @lang_q")
                corr = df_q[["pearsonr", "cons_q"]].corr().values[0,1]
                corrs_red[parcel].append(corr)
            for cons_q in range(1, 5):
                df_q = df.query("parcel == @parcel and cons_q == @cons_q")
                corr = df_q[["pearsonr", "lang_q"]].corr().values[0,1]
                corrs_blue[parcel].append(corr)

        for i, parcel in enumerate(parcel_names):
            ax = axes[i, 2*j]
            sns.lineplot(data=df.query("parcel == @parcel"), x="cons_q", y="pearsonr", 
                        size="lang_q", color="indianred",
                        errorbar='se', ax=ax, 
                        marker='o')
            ax.get_legend().remove()
            ax.set_xlabel("Quartile (cons.)", fontsize=26)
            ax.set_ylabel("Predictivity", fontsize=26)
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='x', labelsize=20)
        
            ax = axes[i, 2*j+1]
            sns.lineplot(data=df.query("parcel == @parcel"), x="lang_q", y="pearsonr", 
                        size="cons_q", color="royalblue",
                        errorbar='se', ax=ax, marker='o')
            ax.set_xlabel("Quartile (lang.)", fontsize=26)
            ax.set_ylabel("Predictivity",  fontsize=26)
            ax.get_legend().remove()
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='x', labelsize=20)

    print(f"Model: {MODEL_FULL_NAMES[model]}")
    for parcel in parcel_names:
        print(f"{parcel}: " +
            "$r_{\\redc}=" +
            f"{np.mean(corrs_red[parcel]):.2f} \\pm {sem(corrs_red[parcel]):.2f}, " +
            "r_{\\bluel}=" +
            f"{np.mean(corrs_blue[parcel]):.2f} \\pm {sem(corrs_blue[parcel]):.2f}$."
            )

    # Adapted from https://stackoverflow.com/a/25814386
    cols = ['Controlled for\n language', 'Controlled for\n consistency'] * len(paradigms)
    rows = ['ROI 1', 'ROI 2', 'ROI 3']

    axes[0,0].annotate("Sentence paradigm", xy=(1.055, 1), xytext=(0, 65),
            xycoords='axes fraction', textcoords='offset points',
            size=34, ha='center', va='baseline')
    if is_vlm:
        axes[0,2].annotate("Picture paradigm", xy=(1.055, 1), xytext=(0, 65),
                xycoords='axes fraction', textcoords='offset points',
                size=34, ha='center', va='baseline')
        axes[0,4].annotate("Word cloud paradigm", xy=(1.055, 1), xytext=(0, 65),
                xycoords='axes fraction', textcoords='offset points',
                size=34, ha='center', va='baseline')
    else:
        axes[0,2].annotate("Word cloud paradigm", xy=(1.055, 1), xytext=(0, 65),
        xycoords='axes fraction', textcoords='offset points',
        size=34, ha='center', va='baseline')

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 8),
                    xycoords='axes fraction', textcoords='offset points',
                    size=27, ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=30, ha='right', va='center')
    
    plt.subplots_adjust(hspace=0.1,wspace=0.07)
    if not is_vlm:
        plt.suptitle("Model: " + MODEL_FULL_NAMES[model], fontsize=40, y=1.1)
    else:
        plt.suptitle("Model: " + MODEL_FULL_NAMES[model], fontsize=40, y=1.1, x=0.515)

    plt.savefig(f"brain_encoding_quartiles_{model}.pdf", format="pdf", bbox_inches="tight", dpi=300)


def plot_brain_encoding_quartiles():
    # Fig. 4
    sns.set_style("ticks", {"font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    parcel_names = ["ROI 1", "ROI 2", "ROI 3"]
    fig, axes = plt.subplots(len(parcel_names), 6, figsize=(28, 3*len(parcel_names)), 
                             sharex=True, sharey=True)
    for j, paradigm in enumerate(["sentences", "pictures", "word_clouds"]):
        if paradigm == "pictures":
            models = ["flava", "llava-7b", "qwen-3b-vl-instruct", "qwen-7b-vl-instruct"]
        else:
            models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                        "flava", "vicuna-7b", "llava-7b",
                        "qwen-1.5b", "qwen-1.5b-instruct",
                        "qwen-3b", "qwen-3b-instruct", "qwen-3b-vl-instruct",
                        "qwen-7b", "qwen-7b-instruct", "qwen-7b-vl-instruct"]

        df = pd.DataFrame()
        for model in models:
            for participant_id in PARTICIPANT_IDS:
                df_participant = pd.read_csv(f"outputs/brain_encoding_quartiles/predict_{paradigm}_{model}_{participant_id}.csv")
                df_participant["id"] = participant_id
                df = pd.concat([df, df_participant], ignore_index=True)

        df = df.drop(columns=["best_layer", "best_pooling", "alpha"])
        df = df.groupby(["id", "parcel", "roi"]).mean().reset_index()
        df["cons_q"] = df["roi"].apply(lambda x: literal_eval(x)[0])
        df["lang_q"] = df["roi"].apply(lambda x: literal_eval(x)[1])
        
        for i, parcel in enumerate(parcel_names):
            ax = axes[i, 2*j]
            sns.lineplot(data=df.query("parcel == @parcel"), x="cons_q", y="pearsonr", 
                        size="lang_q", color="indianred",
                        errorbar='se', ax=ax, 
                        marker='o')
            ax.get_legend().remove()
            ax.set_xlabel("Quartile (cons.)", fontsize=26)
            ax.set_ylabel("Predictivity", fontsize=26)
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='x', labelsize=20)
        
            ax = axes[i, 2*j+1]
            sns.lineplot(data=df.query("parcel == @parcel"), x="lang_q", y="pearsonr", 
                        size="cons_q", color="royalblue",
                        errorbar='se', ax=ax, marker='o')
            ax.set_xlabel("Quartile (lang.)", fontsize=26)
            ax.set_ylabel("Predictivity",  fontsize=26)
            ax.get_legend().remove()
            ax.tick_params(axis='y', labelsize=20)
            ax.tick_params(axis='x', labelsize=20)

            corrs = []
            for lang_q in range(1, 5):
                df_q = df.query("parcel == @parcel and lang_q == @lang_q")
                corr = df_q[["pearsonr", "cons_q"]].corr().values[0,1]
                corrs.append(corr)
            print(parcel, paradigm, "r_CONS=", f"{np.mean(corrs):.2f} ± {sem(corrs):.2f}")

            corrs = []
            for cons_q in range(1, 5):
                df_q = df.query("parcel == @parcel and cons_q == @cons_q")
                corr = df_q[["pearsonr", "lang_q"]].corr().values[0,1]
                corrs.append(corr)
            print(parcel, paradigm, "r_LANG=", f"{np.mean(corrs):.2f} ± {sem(corrs):.2f}")

    # Adapted from https://stackoverflow.com/a/25814386
    cols = ['Controlled for\n language', 'Controlled for\n consistency'] * 3
    rows = ['ROI 1', 'ROI 2', 'ROI 3']

    axes[0,0].annotate("Sentence paradigm", xy=(1.055, 1), xytext=(0, 65),
            xycoords='axes fraction', textcoords='offset points',
            size=34, ha='center', va='baseline')
    axes[0,2].annotate("Picture paradigm", xy=(1.055, 1), xytext=(0, 65),
            xycoords='axes fraction', textcoords='offset points',
            size=34, ha='center', va='baseline')
    axes[0,4].annotate("Word cloud paradigm", xy=(1.055, 1), xytext=(0, 65),
            xycoords='axes fraction', textcoords='offset points',
            size=34, ha='center', va='baseline')

    for ax, col in zip(axes[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 8),
                    xycoords='axes fraction', textcoords='offset points',
                    size=27, ha='center', va='baseline')

    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=30, ha='right', va='center')
    
    plt.subplots_adjust(hspace=0.1,wspace=0.07)

    plt.savefig(f"brain_encoding_quartiles.pdf", format="pdf", bbox_inches="tight", dpi=300)


def plot_brain_encoding_quartiles_heatmap():
    sns.set_style("ticks", {"font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    parcel_names = ["ROI 1", "ROI 2", "ROI 3"]
    annot = True
    fig, axes = plt.subplots(len(parcel_names), 3, figsize=(17, 5*len(parcel_names)), 
                             gridspec_kw={'width_ratios': [1, 0.95, 1.15]})
    for j, paradigm in enumerate(["sentences", "pictures", "word_clouds"]):
        df = pd.DataFrame()
        if paradigm != "pictures": 
            model_names = MODEL_CONFIGS.keys()
        else:
            model_names = [m for m in MODEL_CONFIGS.keys() if MODEL_CONFIGS[m].is_vlm]
        for model in model_names:
            for participant_id in PARTICIPANT_IDS:
                df_participant = pd.read_csv(f"outputs/brain_encoding_quartiles/predict_{paradigm}_{model}_{participant_id}.csv")
                df_participant["id"] = participant_id
                df_participant["model"] = model
                df = pd.concat([df, df_participant], ignore_index=True)

        df["cons_q"] = df["roi"].apply(lambda x: int(literal_eval(x)[0]))
        df["lang_q"] = df["roi"].apply(lambda x: int(literal_eval(x)[1]))
    
        for i, parcel_name in enumerate(parcel_names):
            ax = axes[i, j]
            df_parcel = df.query(f"parcel == @parcel_name")
            df_parcel = df_parcel.groupby(["roi"]).mean(numeric_only=True).reset_index()
            df_parcel["cons_q"] = df_parcel["cons_q"].astype(int)
            df_parcel["lang_q"] = df_parcel["lang_q"].astype(int)
            df_parcel = df_parcel.pivot_table(index="cons_q", columns="lang_q", values="pearsonr")
            sns.heatmap(df_parcel, annot=annot, vmin=0, vmax=0.25, ax=ax, cbar=(j == 2), 
                        annot_kws={"size":15})
            if j == 2:
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(labelsize=15) 
            ax.set_title(f"{parcel_name}, {paradigm[:-1].capitalize().replace('_', ' ')} paradigm", 
                         fontsize=24)
            ax.set_xlabel("Quartile by language", fontsize=20)
            ax.set_ylabel("Quartile by consistency", fontsize=20)
            ax.tick_params(axis='y', labelsize=15)
            ax.tick_params(axis='x', labelsize=15)
            plt.tight_layout()
            plt.savefig(f"brain_encoding_quartiles_heatmap.pdf", format="pdf", bbox_inches="tight", dpi=300)


def plot_brain_encoding_by_layer(pooling_method: str):
    # Fig. 9-10
    sns.set_style("whitegrid", {"axes.edgecolor": ".0", "axes.facecolor":"none", 
                                "font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharey=True, sharex=False)

    df_all = pd.DataFrame()
    for model_name in MODEL_CONFIGS:
        for paradigm in ["sentences", "pictures", "word_clouds"]:
            if paradigm == "pictures" and not MODEL_CONFIGS[model_name].is_vlm:
                continue
            df_cv = pd.read_csv(f"outputs/brain_encoding_quartiles/cv_scores_{paradigm}_{model_name}.csv")
            df_filtered = df_cv.query(f"pooling == '{pooling_method}'").\
                drop(columns=['roi', 'pooling', 'alpha'])
            df_filtered["paradigm"] = paradigm
            df_filtered["model"] = MODEL_FULL_NAMES[model_name]
            df_filtered["style"] = "Qwen2.5"
            if MODEL_FULL_NAMES[model_name].startswith("GPT"):
                df_filtered["style"] = "GPT"
            elif MODEL_FULL_NAMES[model_name].startswith("Qwen2.5-VL"):
                df_filtered["style"] = "Qwen2.5-VL"
            elif MODEL_FULL_NAMES[model_name] == "Vicuna-1.5-7B":
                df_filtered["style"] = "Vicuna"
            elif MODEL_FULL_NAMES[model_name] == "LLaVA-1.5-7B":
                df_filtered["style"] = "LLaVA"
            elif MODEL_FULL_NAMES[model_name] == "FLAVA":
                df_filtered["style"] = "FLAVA"

            df_all = pd.concat([df_all, df_filtered], ignore_index=True).dropna()

    hue_order = ["GPT-2", "GPT-2 Medium", "GPT-2 Large", "GPT-2 XL", 
                "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct", "Qwen2.5-3B",
                "Qwen2.5-3B-Instruct", "Qwen2.5-7B", "Qwen2.5-7B-Instruct",
                "Vicuna-1.5-7B", "LLaVA-1.5-7B", "FLAVA", 
                "Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct"]

    for i, parcel in enumerate(["ROI 1", "ROI 2", "ROI 3"]):
        for j, paradigm in enumerate(["sentences", "pictures", "word_clouds"]):
            df_filtered = df_all.query(f"parcel == @parcel & paradigm == @paradigm")
            if paradigm == "pictures":
                df_filtered = df_filtered.query("model == 'FLAVA' or model == 'Qwen2.5-VL-3B-Instruct' "
                                                "or model == 'LLaVA-1.5-7B' or model == 'Qwen2.5-VL-7B-Instruct'")

            ax = axes[i, j]
            sns.lineplot(data=df_filtered, 
                    x="layer", y="pearsonr", hue="model", style="model",
                    hue_order = hue_order,
                    style_order=hue_order,
                    errorbar='se', 
                    markers=['.'] * 4 + ['d'] * 6 + ['o', 's', '^'] + ['X'] * 2,
                    err_style='bars', ax=ax, linewidth=1, 
                    err_kws={'elinewidth': 0.5})
            plt.ylim([0, 0.7])
            ax.set_xlim([-3, 50])
            if i != 0 or j != 1:
                ax.get_legend().remove()
            else:
                ax.legend(loc='upper center', bbox_to_anchor=(0.4, 1.79), fontsize=17, ncol=4)
            ax.set_title(f"{parcel}, {paradigm[:-1].capitalize().replace('_', ' ')} paradigm", fontsize=20)
            ax.set_ylabel("Cross-validated\n LM predictivity", fontsize=17)
            if i == 2:
                ax.set_xlabel("Layer", fontsize=17)
            else:
                ax.set_xlabel(None)
            ax.tick_params(axis='y', labelsize=13)
            ax.tick_params(axis='x', labelsize=13)

    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f"brain_encoding_by_layer_{pooling_method}.pdf", format="pdf", bbox_inches="tight", dpi=300)


def plot_rsa(sig_sem_cons_voxels_only: bool = False):
    # Fig. 5, 29
    sns.set_style("ticks", {"font.family": "sans-serif", "font.sans-serif": ["Helvetica"]})
    df_scores = pd.DataFrame()
    for model in MODEL_CONFIGS:
        is_vlm = MODEL_CONFIGS[model].is_vlm
        dirname = f"outputs/rsa"
        df_model = pd.DataFrame()
        for participant_id in tqdm(PARTICIPANT_IDS):
            df_participant = pd.read_csv(f"{dirname}/rsa180_{model}_textonly{'_sig_only' if sig_sem_cons_voxels_only else ''}_{participant_id}.csv")
            df_participant["type"] = "Result (text only)"
            df_participant["id"] = participant_id
            df_model = pd.concat([df_model, df_participant], ignore_index=True)
            if is_vlm:
                df_participant = pd.read_csv(f"{dirname}/rsa180_{model}_textimage{'_sig_only' if sig_sem_cons_voxels_only else ''}_{participant_id}.csv")
                df_participant["type"] = "Result (text + images)"
                df_participant["id"] = participant_id
                df_model = pd.concat([df_model, df_participant], ignore_index=True)
                df_participant = pd.read_csv(f"{dirname}/permuted_rsa180_{model}_textimage{'_sig_only' if sig_sem_cons_voxels_only else ''}_{participant_id}.csv")
                df_participant["type"] = "Permuted baseline"
                df_participant["id"] = participant_id
                df_model = pd.concat([df_model, df_participant], ignore_index=True)
            else:
                df_participant = pd.read_csv(f"{dirname}/permuted_rsa180_{model}_textonly{'_sig_only' if sig_sem_cons_voxels_only else ''}_{participant_id}.csv")
                df_participant["type"] = "Permuted baseline"
                df_participant["id"] = participant_id
                df_model = pd.concat([df_model, df_participant], ignore_index=True)
        df_model["model"] = model

        df_mean = df_model.groupby(["model", "pooling", "layer", "parcel", "type"]). \
            agg({"spearmanr": "mean"}).reset_index()
        for parcel in ["ROI 1", "ROI 2", "ROI 3"]:
            if is_vlm:
                ttypes = ["Result (text + images)", "Result (text only)", "Permuted baseline"]
            else:
                ttypes = ["Result (text only)", "Permuted baseline"]
            for ttype in ttypes:
                df_type = df_mean.query("parcel == @parcel and type == @ttype")
                df_type = df_type.drop(columns=["model", "parcel",  "type"])
                df_type = df_type.set_index(["pooling",  "layer"])
                best_pooling, best_layer = df_type.idxmax().values[0]
                df_scores = pd.concat([df_scores, df_model.query("parcel==@parcel and type==@ttype"
                                                                 " and pooling==@best_pooling and layer==@best_layer")], 
                                                                 ignore_index=True)
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), 
                            sharex="col", 
                            sharey=True, gridspec_kw={'width_ratios': [1, 0.55]})
    for i, parcel in enumerate(["ROI 1", "ROI 2", "ROI 3"]):
        ax = axes[i, 0]
        df = df_scores.query("parcel == @parcel")
        q_vl = " or ".join(f"model == '{m}'" for m in ["flava", "llava-7b", 
                                                     "qwen-3b-vl-instruct", "qwen-7b-vl-instruct"])
        q_l = " and ".join(f"model != '{m}'" for m in ["flava", "llava-7b", 
                                                "qwen-3b-vl-instruct", "qwen-7b-vl-instruct"])
        colors = ["lightseagreen", "silver"]
        colors2 = ["paleturquoise","lightseagreen", "silver"]
        sns.barplot(data=df.query(q_l), x="model", y="spearmanr", ax=ax, hue="type", 
            palette=colors, errorbar='se',
            order=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "qwen-1.5b", "qwen-1.5b-instruct", 
                    "qwen-3b", "qwen-3b-instruct", "qwen-7b", "qwen-7b-instruct", "vicuna-7b"])
        ax.legend_.remove()
        ax.set_ylabel("Correlation", fontsize=17)
        ax.set_xlabel(None)
        ax.tick_params(axis='x', labelrotation=90, labelsize=17)
        ax.tick_params(axis='y', labelsize=13)
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xticklabels(["GPT-2", "GPT-2 Medium", "GPT-2 Large", "GPT-2 XL", 
            "Qwen2.5-1.5B", "Qwen2.5-1.5B-\nInstruct",
            "Qwen2.5-3B", "Qwen2.5-3B-\nInstruct",
            "Qwen2.5-7B", "Qwen2.5-7B-\nInstruct",
            "Vicuna-1.5-7B"])

        ax = axes[i, 1]
        sns.barplot(data=df.query(q_vl), x="model", y="spearmanr", ax=ax, hue="type", 
                    hue_order = ["Result (text + images)", "Result (text only)", "Permuted baseline"],
                    palette=colors2, errorbar='se',order=["flava", 
                                                "qwen-3b-vl-instruct", "qwen-7b-vl-instruct", 
                                                "llava-7b"])
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:])
        if i == 0:
            bbox_to_anchor = (-0.17, 1.0)
        elif i == 1:
            if sig_sem_cons_voxels_only:
                bbox_to_anchor = (-0.17, 1.0)
            else:
                bbox_to_anchor = (-0.17, 1.09)
        else:
            bbox_to_anchor = (-0.17, 1.0)
        sns.move_legend(ax, loc="upper right", bbox_to_anchor=bbox_to_anchor)
        plt.setp(ax.get_legend().get_texts(), fontsize=15)
        ax.set_xlabel(None)
        ax.tick_params(axis='x', labelrotation=90, labelsize=17)
        ax.tick_params(axis='y', labelsize=13)
        ax.spines[['right', 'top']].set_visible(False)
        ax.yaxis.set_tick_params(labelbottom=True)
        ax.set_xticklabels(["FLAVA", "Qwen2.5-VL-\n3B-Instruct",
            "Qwen2.5-VL-\n7B-Instruct",
            "LLaVA-1.5-7B"])

    rows = ['ROI 1', 'ROI 2', 'ROI 3']
    cols = ["Language-only models", "Vision-language models"]
    pad = 5
    for ax, row in zip(axes[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size=20, ha='right', va='center')
    for ax, col in zip(axes[0], cols):
        ax.annotate(col, 
                    xy=(0.5, 1),  # Center the text
                    xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size=20, ha='center', va='baseline')
        
    plt.subplots_adjust(hspace=0.08)
    plt.savefig(f"rsa{'_sig_voxels_only' if sig_sem_cons_voxels_only else ''}.pdf", format='pdf', bbox_inches='tight', dpi=300)


def lang_network_vs_sem_cons_rois():
    # Fig. 28
    fs = 10
    sns.set_theme(style="ticks", rc={"font.size": fs, "axes.titlesize": fs, 
                                     "axes.labelsize": fs+2,
                                "xtick.labelsize": fs, "ytick.labelsize": fs, "figure.titlesize": fs})
    fig, axes = plt.subplots(3, 1, figsize=(8, 8))

    mean_map = nib.load(f"outputs/semantic_consistency/probabilistic_consistency_map.nii").get_fdata() # type: ignore

    vals = {"parcel": [], "mean_consistency": []}
    glasser_map, glasser_labels_dict = load_glasser_parcellation()
    glasser_lookup = {v: k for k, v in glasser_labels_dict.items()}
    
    for parcel_idx, glasser_parcel_name in glasser_labels_dict.items():
        vals["parcel"].append(glasser_parcel_name)
        vals["mean_consistency"].append(mean_map[glasser_map == parcel_idx].mean())
    df_cons = pd.DataFrame.from_dict(vals)

    lang_parcel_map = nib.load("data/brain_parcels/allParcels-language-SN220.nii").get_fdata() # type: ignore
    glasser_parcels_to_lang_parcels = {}
    for lang_parcel_idx, lang_parcel_name in zip(range(1, 7), ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp", "AngG"]):
        lang_parcel_mask = lang_parcel_map == lang_parcel_idx
        glasser_parcels_in_mask, overlap_size = np.unique(glasser_map[lang_parcel_mask], return_counts=True)
        for glasser_parcel_idx, overlap_size in zip(glasser_parcels_in_mask, overlap_size):
            if glasser_parcel_idx == 0:
                continue
            glasser_parcel_name = glasser_labels_dict[glasser_parcel_idx]
            glasser_parcel_size = np.sum(glasser_map == glasser_parcel_idx)
            if overlap_size / glasser_parcel_size > 0.25:  # More than 25% overlap
                if glasser_parcel_name in glasser_parcels_to_lang_parcels:
                    # Each Glasser parcel can only be assigned to one language parcel
                    # If it overlaps with multiple, keep the one with the largest overlap
                    existing_overlap = glasser_parcels_to_lang_parcels[glasser_parcel_name][1]
                    if overlap_size / glasser_parcel_size < existing_overlap:
                        continue
                glasser_parcels_to_lang_parcels[glasser_parcel_name] = (lang_parcel_name, overlap_size / glasser_parcel_size)

    for paradigm, ax in zip(["sentences", "pictures", "word_clouds"], axes):

        df_predict = pd.DataFrame()
        if paradigm == "pictures":
            models = [model for model in MODEL_CONFIGS.keys() if MODEL_CONFIGS[model].is_vlm]
        else:
            models = MODEL_CONFIGS.keys()
        for model in models:
            for participant_id in tqdm(PARTICIPANT_IDS):
                fname = f"outputs/brain_encoding_whole_brain/predict_{paradigm}_{model}_{participant_id}.csv"
                if not os.path.isfile(fname):
                    print("Missing", fname)
                    continue
                df_uid = pd.read_csv(fname)
                df_predict = pd.concat([df_predict, df_uid], ignore_index=True)
        df_predict = df_predict.drop(columns=["roi"])

        df_all = pd.DataFrame()

        for parcel in df_predict["parcel"].unique():
            df_filtered = df_predict.query("parcel == @parcel")
            df_filtered = df_filtered.drop(columns=["fold", "alpha"])
            df_filtered = df_filtered.groupby(["id", "parcel", "best_layer", "best_pooling"]).mean().reset_index()
            df_all = pd.concat([df_all, df_filtered], ignore_index=True)

        df = df_cons.merge(df_all, left_on="parcel", right_on="parcel")
        df = df.drop(columns=["best_layer", "best_pooling", "id"])
        df = df.groupby(["parcel", "mean_consistency"]).mean().reset_index()
        
        df.set_index("parcel", inplace=True)

        df_sem_cons_roi_copy = df.copy()
        df_sem_cons_roi_copy["location"] = None
        df_sem_cons_roi_copy.loc[[f"L_{p}_ROI" for p in ["A5", "STSdp", "TPOJ1", "TPOJ2"]], "location"] = "ROI 1"
        df_sem_cons_roi_copy.loc[[f"L_{p}_ROI" for p in ["TE2p", "PH"]], "location"] = "ROI 2"
        df_sem_cons_roi_copy.loc[[f"L_{p}_ROI" for p in ["IFSa", "45", "FOP5"]], "location"] = "ROI 3"
        df_sem_cons_roi_copy.reset_index(inplace=True)

        for glasser_parcel, (lang_parcel_name, _) in glasser_parcels_to_lang_parcels.items():
            df.loc[[glasser_parcel], "location"] = lang_parcel_name

        df.reset_index(inplace=True)

        df = pd.concat([df, df_sem_cons_roi_copy], ignore_index=True)
        df.dropna(subset=["location"], inplace=True)

        # Reordering to know the order of the stripplot points
        bar_order = ["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp", "AngG", "ROI 1", "ROI 2", "ROI 3"]
        bar_lookup = {v: i for i, v in enumerate(bar_order)}
        df.sort_values(by=["mean_consistency"], ascending=False, inplace=True)
        df.sort_values(by=['location'], key=lambda x: x.map(bar_lookup), inplace=True)

        sns.stripplot(df, x="location", y="pearsonr", hue="location", 
                    order=["IFGorb", "IFG", "MFG", "AntTemp", "PostTemp", "AngG", "ROI 1", "ROI 2", "ROI 3"],
                    ax=ax, edgecolor='k', linewidth=1.5, dodge=False, jitter=False,
                    palette="Dark2", legend=False, marker='_')
        coordinates = []
        for collection in ax.collections:
            coordinates.extend(collection.get_offsets())
            print("Coordinates:", coordinates)

        # Label the points with their y-offset (y_var)
        for i, (x, y) in enumerate(coordinates):
            col = df.iloc[i]['location']
            label= df.iloc[i]['parcel'][2:-4]  # Remove the "_ROI" suffix
            ax.text(x-0.05, y-0.005, label, ha='right', va='bottom', size=9)
        xmin, xmax=ax.get_xlim()
        ax.set_xlabel(None)
        ax.set_ylim(top=0.3, bottom=0)
        ax.set_title(f"{paradigm[:-1].capitalize().replace('_', ' ')} paradigm", fontsize=12)
        ax.vlines(x=5.3, ymin=0, ymax=0.3, color='k', linestyle='--', linewidth=0.5)
        ax.set_ylabel("Mean LM predictivity", fontsize=10)

    plt.tight_layout()
    plt.savefig(f"lang_network_vs_sem_cons_rois.pdf", format="pdf", bbox_inches='tight', dpi=300)
