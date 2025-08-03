from transformers import GPT2Tokenizer, GPT2Model, FlavaModel, BertTokenizer, FlavaProcessor, \
                LlavaForConditionalGeneration, LlamaTokenizer, Qwen2_5_VLForConditionalGeneration, \
                AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor
from utils import *
from analysis_utils import *
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
import pickle
from tqdm import tqdm
from PIL import Image
from scipy.stats import pearsonr
import torch
from collections import defaultdict

ModelConfig = namedtuple("ModelConfig", ["model_id", "pooling_methods", "hidden_size", "n_layers", "is_vlm"])
MODEL_CONFIGS = {
    "gpt2": ModelConfig("gpt2", ['last_tok', 'mean'], 768, 12, False),
    "gpt2-medium": ModelConfig("gpt2-medium", ['last_tok', 'mean'], 1024, 24, False),
    "gpt2-large": ModelConfig("gpt2-large", ['last_tok', 'mean'], 1280, 36, False),
    "gpt2-xl": ModelConfig("gpt2-xl", ['last_tok', 'mean'], 1600, 48, False),
    "flava": ModelConfig("facebook/flava-full", ['last_tok', 'first_tok', 'mean'], 768, 14, True),
    "llava-7b": ModelConfig("llava-hf/llava-1.5-7b-hf", ['last_tok', 'mean'], 4096, 32, True),
    "vicuna-7b": ModelConfig("lmsys/vicuna-7b-v1.5", ['last_tok', 'mean'], 4096, 32, False),
    "qwen-7b": ModelConfig("Qwen/Qwen2.5-7B", ['last_tok', 'mean'], 3584, 28, False),
    "qwen-7b-instruct": ModelConfig("Qwen/Qwen2.5-7B-Instruct", ['last_tok', 'mean'], 3584, 28, False),
    "qwen-7b-vl-instruct": ModelConfig("Qwen/Qwen2.5-VL-7B-Instruct", ['last_tok', 'mean'], 3584, 28, True),
    "qwen-1.5b": ModelConfig("Qwen/Qwen2.5-1.5B", ['last_tok', 'mean'], 1536, 28, False),
    "qwen-1.5b-instruct": ModelConfig("Qwen/Qwen2.5-1.5B-Instruct", ['last_tok', 'mean'], 1536, 28, False),
    "qwen-3b": ModelConfig("Qwen/Qwen2.5-3B", ['last_tok', 'mean'], 2048, 36, False),
    "qwen-3b-instruct": ModelConfig("Qwen/Qwen2.5-3B-Instruct", ['last_tok', 'mean'], 2048, 36, False),
    "qwen-3b-vl-instruct": ModelConfig("Qwen/Qwen2.5-VL-3B-Instruct", ['last_tok', 'mean'], 2048, 36, True),
}

DF_STIM = pd.read_excel(f"data/stimuli/complang_paradigms_stims.xlsx").set_index('a. wordSent')

###################### INTERNAL FUNCTIONS ######################

def load_model(model_name: str, 
               use_gpu: bool = False) -> Dict[str, Any]:
    """
    Loads the specified model and its tokenizer/processor.
    
    Args:
        model_name (str): Name of the model to load (one of MODEL_CONFIG keys).
        use_gpu (bool): Whether to load the model on GPU.
    
    Returns:
        Dict[str, Any]: A dictionary containing the model, tokenizer/processor, and model name.
    """

    model_id = MODEL_CONFIGS[model_name].model_id
    processor = None
    tokenizer = None

    if model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        model = GPT2Model.from_pretrained(model_id)
    elif model_name == 'flava':
        tokenizer = BertTokenizer.from_pretrained(model_id)
        model = FlavaModel.from_pretrained(model_id)
        processor = FlavaProcessor.from_pretrained(model_id)
    elif model_name in ['vicuna-7b', 'alpaca-7b']:
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    elif model_name in ["qwen-7b", "qwen-7b-instruct", 
                        "qwen-1.5b", "qwen-1.5b-instruct",
                        "qwen-3b", "qwen-3b-instruct"]:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True).eval()
    # VL models
    elif model_name == 'llava-7b':
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id)
        processor = AutoProcessor.from_pretrained(model_id)
    elif model_name in ['qwen-7b-vl-instruct', 'qwen-1.5b-vl-instruct', 'qwen-3b-vl-instruct']:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

    if use_gpu:
        model = model.to(0) # type: ignore

    return_dict = {"model_name": model_name, "model": model}
    if processor:
        return_dict["processor"] = processor
    if tokenizer:
        return_dict["tokenizer"] = tokenizer
    return return_dict


def get_hidden_states(model_dict: Dict[str, Any], 
                      text: str, 
                      img_path: Optional[str] = None, 
                      use_gpu: bool = False) -> List[torch.Tensor]:
    # TODO: is this actually saving tensors for FLAVA? Or is it tuples?
    """
    Extracts the model's hidden states for the given text and/or image input.

    Args:
        model_dict (Dict[str, Any]): Dictionary containing the model and tokenizer/processor,
            returned by load_model.
        text (str): The text input for the model.
        img_path (Optional[str]): The image path for the model input (if applicable).
        use_gpu (bool): Whether to use GPU for model inference.

    Returns:
        List[torch.Tensor]: A list of hidden states from the model, with one tensor per layer.
    """

    model = model_dict['model']
    model_name = model_dict['model_name']
    n_layers = MODEL_CONFIGS[model_name].n_layers
    is_vlm = MODEL_CONFIGS[model_name].is_vlm

    print(f"Stimulus: text={text}, img={img_path}")

    if "tokenizer" in model_dict and (img_path is None or not is_vlm):
        # text-only input via tokenizer
        tokenizer = model_dict['tokenizer']
        encoded_input = tokenizer([text], return_tensors='pt', padding=False)
    elif "processor" in model_dict:
        # any input via processor
        processor = model_dict['processor']        
        if img_path:
            image = Image.open(img_path).convert('RGB')
            encoded_input = processor(text=text, images=image, return_tensors='pt')
        else:
            encoded_input = processor(text=text, return_tensors='pt')
    else:
        exit(f"Stimulus processing not defined: {model_name}")

    if use_gpu:
        encoded_input = encoded_input.to(0) # type: ignore
    output = model(**encoded_input, output_hidden_states=True) # type: ignore

    def detach_plus(tensor):
        if use_gpu:
            return tensor.detach().cpu()
        else:
            return tensor.detach()

    if model_name != "flava":
        assert len(output['hidden_states']) == n_layers + 1
        return [detach_plus(output['hidden_states'][l]) for l in range(n_layers + 1)]
    else:
        # special handling for FLAVA, which has separate text and image encoders
        if not img_path:
            # if only text, returning:
            # text_output.hidden_states [0-12] + 
            # text_embeddings [13] (checking that it's always the same as text_output.last_hidden_state)
            # get_text_features [14] (pooled and projected text_embeddings?) 
            assert (output["text_embeddings"] == output["text_output"]["last_hidden_state"]).all()
            assert len(output['text_output']['hidden_states']) == n_layers - 1
            text_features = model.get_text_features(**encoded_input)
            assert text_features.shape == output["text_embeddings"].shape
            result = [detach_plus(output['text_output']['hidden_states'][l])
                    for l in range(n_layers - 1)] + \
                [detach_plus(output["text_embeddings"])] + \
                [detach_plus(text_features)]
        else:
            # if text+image, returning:
            # tuple of (text_output.hidden_states, image_output.hidden_states) [0-12] + 
            # tuple of (text_embeddings, image_embeddings) [13]
            # multimodal_embedding [14] (checking that it's always the same as multimodal_output.last_hidden_state)
            assert len(output['text_output']['hidden_states']) == n_layers - 1
            assert len(output['image_output']['hidden_states']) == n_layers - 1
            assert output['text_embeddings'].shape[1] + output['image_embeddings'].shape[1] == \
                output['multimodal_embeddings'].shape[1] - 1
            assert (output["multimodal_output"]["last_hidden_state"] == output["multimodal_embeddings"]).all()

            result = []
            for l in range(n_layers - 1):
                result.append((detach_plus(output['text_output']['hidden_states'][l]),
                               detach_plus(output['image_output']['hidden_states'][l])))
            result.append((detach_plus(output['text_embeddings']), 
                           detach_plus(output['image_embeddings'])))
            result.append(detach_plus(output['multimodal_embeddings']))
        assert len(result) == n_layers + 1
        return result


def get_stim_text(paradigm: str, concept: str, idx: int, model_name: Optional[str] = None) -> str:
    """
    Gets the stimulus text for a given paradigm, concept, and stimulus index.

    Args:
        paradigm (str): The paradigm to use (e.g., "sentences", "pictures", "word_clouds").
        concept (str): The concept word.
        idx (int): The index of the stimulus (1-6).
        model_name (Optional[str]): The name of the model (one of MODEL_CONFIG keys);
            only needed for the picture paradigm to know which special image tokens to use.

    Returns:
        str: The stimulus text for the specified paradigm and concept.
    """
    # Special case: "Counting" used in word clouds and pictures, but "Count" in sentences
    concept_lookup = concept
    if concept == "Counting":
        concept_lookup = "Count"
    row = DF_STIM.loc[concept_lookup]
    if paradigm == "pictures":
        assert model_name is not None
        if model_name == "llava-7b":
            return f"<image> {row['f. wordImg']}"
        elif model_name in ["qwen-3b-vl-instruct", "qwen-7b-vl-instruct"]:
            return f"<|vision_start|><|image_pad|><|vision_end|> {row['f. wordImg']}"
        else:
            return str(row['f. wordImg'])
    elif paradigm == "sentences":
        return row[f'b{idx}. sent{idx}'].strip()
    elif paradigm == "word_clouds":
        return " ".join([row[f'e{idx}. cloud{idx}'] for idx in range(1, 7)]) # type: ignore
    else:
        raise ValueError("Invalid paradigm: " + paradigm)


def pool_hidden_states(hidden_states: List[torch.Tensor], pooling_method: str) -> Dict[int, np.ndarray]:

    """
    Pools the hidden states according to the specified pooling method.
    
    Args:
        hidden_states (List[torch.Tensor]): List of hidden states from the model, with one tensor per layer.
        pooling_method (str): The pooling method to use ('last_tok', 'mean', 'first_tok').
    
    Returns:
        Dict[int, np.ndarray]: A dictionary with layer indices as keys and pooled hidden states as values.
    """

    n_layers = len(hidden_states) - 1
    result = {}
    if pooling_method == "last_tok":
        for layer in range(0, n_layers + 1):
            if type(hidden_states[layer]) == torch.Tensor:
                result[layer] = hidden_states[layer][0, -1].numpy()
            elif type(hidden_states[layer]) == tuple:
                hidden_states_t, hidden_states_i = hidden_states[layer]
                # multimodal flava: hidden states 0-12 are tuples of (h_t, h_i)
                result[layer] = 0.5 * (hidden_states_t[0, -1].numpy() + hidden_states_i[0, -1].numpy())
            else:
                raise ValueError(f"Unexpected hidden state type for layer {layer}: {type(hidden_states[layer])}")
        return result
    elif pooling_method == 'mean':
        # case when the same embedding is used for all stimuli
        if type(hidden_states[0]) == torch.Tensor and hidden_states[0].shape[1] == 1:
            return {layer: hidden_states[layer].squeeze().numpy() 
                    for layer in range(0, n_layers + 1)}
        else:
            for layer in range(0, n_layers + 1):
                if type(hidden_states[layer]) == torch.Tensor:
                    result[layer] = hidden_states[layer].squeeze().numpy().mean(axis=0) 
                elif type(hidden_states[layer]) == tuple:
                    hidden_states_t, hidden_states_i = hidden_states[layer]
                    # multimodal flava: hidden states 0-12 are tuples of (h_t, h_i)
                    result[layer] = 0.5 * (hidden_states_t.squeeze().numpy().mean(axis=0) + \
                                        hidden_states_i.squeeze().numpy().mean(axis=0))
                else:
                    raise ValueError(f"Unexpected hidden state type for layer {layer}: {type(hidden_states[layer])}")
        return result
    elif pooling_method == 'first_tok':
        for layer in range(0, n_layers + 1):
            if type(hidden_states[layer]) == torch.Tensor:
                result[layer] = hidden_states[layer][0, 0].numpy()
            elif type(hidden_states[layer]) == tuple:
                hidden_states_t, hidden_states_i = hidden_states[layer]
                # multimodal flava: hidden states 0-12 are tuples of (h_t, h_i)
                result[layer] = 0.5 * (hidden_states_t[0, 0].numpy() + \
                                       hidden_states_i[0, 0].numpy())
            else:
                raise ValueError(f"Unexpected hidden state type for layer {layer}: {type(hidden_states[layer])}")
        return result
    else:
        raise ValueError("Invalid pooling method: " + pooling_method)


###################### PIPELINE STEPS ######################

def cache_lm_embeddings(model_name: str, 
                       paradigm: str, 
                       use_gpu: bool = False):
    """
    Caches the LM embeddings for the specified model and paradigm.

    Args:
        model_name (str): Name of the model to use (one of MODEL_CONFIG keys).
        paradigm (str): The paradigm to use (e.g., "sentences", "pictures", "word_clouds").
        use_gpu (bool): Whether to use GPU for model inference.
    """
    pooling_methods = MODEL_CONFIGS[model_name].pooling_methods
    is_vlm = MODEL_CONFIGS[model_name].is_vlm
    model_dict = load_model(model_name, use_gpu)

    for pooling_method in pooling_methods:
        os.makedirs(f"outputs/lm_embeddings/{paradigm}/{model_name}/{pooling_method}_pooling/", 
                    exist_ok=True)

    for concept in tqdm(CONCEPTS):
        
        # Text is the same for all word cloud indices
        max_idx = 6
        if paradigm == "word_clouds" or (paradigm == "pictures" and not is_vlm):
            max_idx = 1
        lm_embed_dict = {pooling_method: {} for pooling_method in pooling_methods}
        for idx in range(1, max_idx + 1):
            img_path = None
            if paradigm == "pictures":
                img_path =f"data/stimuli/images/" \
                        f"{concept}/{concept.lower()}_{idx}.jpg"
            
            text = get_stim_text(paradigm, concept, idx, model_name)
            if text is None:
                exit("Text retrieving error")
            with torch.no_grad():
                hidden_states = get_hidden_states(model_dict, text, img_path, use_gpu) # type: ignore
            for pooling_method in pooling_methods:
                layer_dict = pool_hidden_states(hidden_states, pooling_method)
                assert len(layer_dict[0]) == MODEL_CONFIGS[model_name].hidden_size
                if max_idx == 1:
                    lm_embed_dict[pooling_method] = layer_dict
                else:
                    lm_embed_dict[pooling_method][idx] = layer_dict

        for pooling_method in pooling_methods:
            fname = f"outputs/lm_embeddings/{paradigm}/{model_name}/{pooling_method}_pooling/{concept}.pkl"
            print(f"Writing to {fname}")
            with open(fname, 'wb') as f:
                pickle.dump(lm_embed_dict[pooling_method], f)


def cache_betas_in_mask(participant_id: str, 
                        combined_masks_by_voxel_population: dict[str, npt.ArrayLike],
                        parcel_map: npt.NDArray, 
                        parcel_dict: Dict[int, str], 
                        paradigm: str,
                        out_fpath: Optional[str] = None,
                        avg_by_parcel: bool = False) -> pd.DataFrame:
    """
    Caches the betas for the specified participant and paradigms in the provided masks.

    Args:
        participant_id (str): The ID of the participant (M{01-17}).
        combined_masks_by_voxel_population (dict[str, npt.ArrayLike]): Dictionary mapping 
            voxel populations to their corresponding masks (union over this type of voxels across 
            all parcels).
            This function was originally written to work with different voxel populations within 
            anatomical parcels; 
            if using entire parcels (e.g. Glasser parcels or semantic consistency ROIs), 
            pass a dict with "all" as the key and the union of all parcels as the value.
            For quartile-based encoding, the quartiles are passed as keys and 
            the union of the appropriate quartile voxels across all ROIs as the value.
        parcel_map (npt.NDArray): The 3D parcel map 
            (in our case, either the Glasser parcellation or the map of semantically consistent ROIs).
        parcel_dict (Dict[int, str]): Dictionary mapping parcel indices to their names.
        paradigm (str): The paradigm to use (e.g., "sentences", "pictures", "word_clouds").
        out_fpath (Optional[str]): Optional output file path to save the betas DataFrame.
        avg_by_parcel (bool): Whether to average the betas by parcel.

    Returns:
        pd.DataFrame: A DataFrame containing the betas for the specified participant and paradigms
            in the provided masks.
    """
    df_betas = pd.DataFrame()

    for voxel_population, mask in combined_masks_by_voxel_population.items():
        df_betas_mask = load_events_and_responses(participant_id, voxel_mask=mask, paradigms=[paradigm], 
                        parcel_map=parcel_map, parcel_dict=parcel_dict, avg_by_parcel=avg_by_parcel)
        df_betas_mask["voxel_population"] = voxel_population
        df_betas_mask["id"] = participant_id
        df_betas = pd.concat([df_betas, df_betas_mask.reset_index()])

    df_betas["stimulus_idx"] = df_betas["stimulus_idx"].astype(int)
    if out_fpath:
        df_betas.to_csv(out_fpath, index=False)
    return df_betas


def choose_layer_and_pooling(model_name: str, 
                             paradigm: str, 
                             df_betas: pd.DataFrame, 
                             out_fpath: str) -> pd.DataFrame:
    """
    Performs cross-validation to choose the layer and pooling method 
    that yield the best brain encoding performance for the specified model and paradigm.
    
    Args:
        model_name (str): Name of the model to use (one of MODEL_CONFIG keys).
        paradigm (str): The paradigm to use (e.g., "sentences", "pictures", "word_clouds").
        df_betas (pd.DataFrame): DataFrame containing the betas for the specified participant and paradigm
            in the chosen brain region(s).
        out_fpath (str): Output file path to save the cross-validation results.
    Returns:
        pd.DataFrame: A DataFrame containing the cross-validation scores for different layers and pooling methods
            for the specified model and paradigm.
    """

    layers = list(range(MODEL_CONFIGS[model_name].n_layers + 1))
    pooling_methods = MODEL_CONFIGS[model_name].pooling_methods
    is_vlm = MODEL_CONFIGS[model_name].is_vlm
    single_embedding_per_concept = (paradigm == "word_clouds" or (paradigm == "pictures" and not is_vlm))

    cv_scores = {"layer": [], "pooling": [], "fold": [], "pearsonr": [], "voxel_population": [], 
                 "alpha": [], "parcel": []}

    # voxel_population is either "all" (for whole-parcel/ROI analyses) 
    # or one of the bins for the quartile analyses (e.g., (1, 1), (1, 2), ... (4, 4))
    for voxel_population in df_betas["voxel_population"].unique():
        y_per_parcel = {}
        df_betas_voxel_population = df_betas[df_betas["voxel_population"] == voxel_population]
        df_betas_voxel_population = df_betas_voxel_population.drop(columns=["voxel_population", "id"])

        if single_embedding_per_concept:
            df_betas_voxel_population = df_betas_voxel_population.groupby(["concept", "parcel"]).mean(numeric_only=True)
        else:    
            df_betas_voxel_population = df_betas_voxel_population.groupby(["concept", "stimulus_idx", "parcel"]).mean(numeric_only=True)

        concepts_and_indices = df_betas_voxel_population.groupby(["concept", "stimulus_idx"]). \
            mean(numeric_only=True).index

        parcels = df_betas_voxel_population.index.get_level_values(1 if single_embedding_per_concept else 2).\
            drop_duplicates()

        for parcel in parcels:
            df_parcel = filter_df_by_level(df_betas_voxel_population, "parcel", parcel)
            if len(df_parcel.dropna()) > 0:
                y_per_parcel[parcel] = df_parcel["beta"].to_numpy()

        parcels = list(y_per_parcel.keys())
        for pooling_method in pooling_methods:
            print(model_name, pooling_method, voxel_population)
            for layer in tqdm(layers):
                embedding_dict = {}
                for concept in df_betas_voxel_population.index.get_level_values(0).drop_duplicates():
                    with open(f"outputs/lm_embeddings/{paradigm}/{model_name}/{pooling_method}_pooling/{concept}.pkl", "rb") as f:
                        concept_embedding_dict = pickle.load(f)
                        if single_embedding_per_concept:
                            embedding_dict[concept] = concept_embedding_dict[layer]
                        else:
                            for idx in range(1, 7):
                                embedding_dict[(concept, idx)] = concept_embedding_dict[idx][layer]

                y = np.vstack([y_per_parcel[p] for p in parcels]).T
                if single_embedding_per_concept:
                    X = np.vstack([embedding_dict[concept] for concept in concepts_and_indices.get_level_values(0).drop_duplicates()])
                else:
                    X = np.vstack([embedding_dict[concept_and_idx] for concept_and_idx in concepts_and_indices])

                kf = KFold(n_splits=5, shuffle=True, random_state=0)
                for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
                    model = RidgeCV(alphas=np.asarray([10 ** x for x in range(-30, 30)]), alpha_per_target=True)
                    model.fit(X[train_index, :], y[train_index])
                    y_pred = model.predict(X[test_index, :])

                    for i, parcel in enumerate(parcels):
                        if len(y_pred.shape) == 1:
                            y_pred = y_pred[:, np.newaxis]
                        r = pearsonr(y_pred[:, i], y[test_index, i]).statistic # type: ignore
                        cv_scores["fold"].append(fold_idx+1)
                        if type(model.alpha_) in [float, int]:
                            cv_scores["alpha"].append(model.alpha_)
                        else:
                            cv_scores["alpha"].append(model.alpha_[i]) # type: ignore
                        cv_scores["pearsonr"].append(r)
                        cv_scores["layer"].append(layer)
                        cv_scores["pooling"].append(pooling_method)
                        cv_scores["voxel_population"].append(voxel_population)
                        cv_scores["parcel"].append(parcel)

                df_cv = pd.DataFrame.from_dict(cv_scores)
                df_cv.to_csv(out_fpath, index=False)
    return df_cv

def get_best_setting(mean_cv_fname: str) -> dict:
    """
    Reads the cross-validation results from the specified file and returns the best layer and pooling method
    for each parcel and voxel population.
    Args:
        mean_cv_fname (str): Path to the CSV file containing the cross-validation results.
    Returns:
        dict: A dictionary mapping each parcel to a dictionary of ROIs and their best layer and pooling method.
            For more details on the parcel/ROI terminology here, see the `cache_betas_in_mask` function.   
    """
    df_cv = pd.read_csv(mean_cv_fname)
    params_by_parcel_and_voxel_population = defaultdict(dict)
    for voxel_population in df_cv["voxel_population"].unique():
        for parcel in df_cv.query(f"voxel_population == '{voxel_population}'")["parcel"].unique():
            df_filtered = df_cv.query(f"parcel == '{parcel}' & voxel_population == '{voxel_population}'")
            df_filtered = df_filtered.drop(columns=["parcel", "voxel_population", "fold", "alpha"])
            df_filtered = df_filtered.groupby(["layer", "pooling"]).mean()
            params = df_filtered.idxmax().to_list()[0]
            params_by_parcel_and_voxel_population[parcel][voxel_population] = params
    return params_by_parcel_and_voxel_population


def predict(model_name: str, 
            participant_id: str, 
            combined_masks_by_voxel_population: dict[str,npt.ArrayLike],
            params_by_parcel_and_voxel_population: dict, 
            parcel_map: npt.NDArray, 
            parcel_dict: dict, 
            paradigm: str,
            out_fpath: Optional[str] = None) -> pd.DataFrame:
    """
    Predicts the betas for the specified participant and paradigms within the given voxel mask
    using the cached LM embeddings and the best layer and pooling method for each parcel and voxel population.
    Args:
        model_name (str): Name of the model to use (one of MODEL_CONFIG keys).
        participant_id (str): The ID of the participant (M{01-17}).
        combined_masks_by_voxel_population (dict[str, npt.ArrayLike]): Dictionary mapping voxel populations to
            their corresponding masks (union over all parcels).
            For more details, see the `cache_betas_in_mask` function.
        params_by_parcel_and_voxel_population (dict): Dictionary mapping each parcel to a dictionary of
            voxel populations and their best-performing layers and pooling methods.
            For more details, see the `cache_betas_in_mask` function.
        parcel_map (npt.NDArray): The 3D parcel map 
            (in our case, either the Glasser parcellation or the map of semantically consistent ROIs).
        parcel_dict (dict): Dictionary mapping parcel indices to their names.
        paradigm (str): The paradigm to use (e.g., "sentences", "pictures", "word_clouds").
        out_fpath (Optional[str]): Optional output file path to save the prediction results.
    Returns:
        pd.DataFrame: A DataFrame containing the predicted betas for the specified participant and paradigm
            in the provided voxels.
    """

    data = {"id": [], "parcel": [], "voxel_population": [], "fold": [], "pearsonr": [],
            "best_layer": [], "best_pooling": [], "alpha": []}

    is_vlm = MODEL_CONFIGS[model_name].is_vlm
    single_embedding_per_concept = (paradigm == "word_clouds" or (paradigm == "pictures" and not is_vlm))

    for voxel_population, mask in combined_masks_by_voxel_population.items():
        print(f"Predicting in voxel_population={voxel_population}")
        df_betas_roi = load_events_and_responses(participant_id, mask, [paradigm], 
                        parcel_map=parcel_map, parcel_dict=parcel_dict, avg_by_parcel=True)
        
        parcels = df_betas_roi.reset_index()["parcel"].unique()
        for parcel in parcels:
            best_layer, best_pooling = params_by_parcel_and_voxel_population[parcel][voxel_population]
            print(f"Parcel={parcel}, best layer={best_layer}, best pooling={best_pooling}")

            df_betas = df_betas_roi.query(f"parcel == @parcel")
            embeddings = []
            if single_embedding_per_concept:
                df_betas = df_betas.reset_index()
                df_betas = df_betas.groupby(["concept", "paradigm", "parcel"]).mean(numeric_only=True)
                for concept, _, _ in df_betas.index:
                    with open(f"outputs/lm_embeddings/{paradigm}/{model_name}/" \
                                f"{best_pooling}_pooling/{concept}.pkl", "rb") as f:
                        concept_embedding_dict = pickle.load(f)
                        embeddings.append(concept_embedding_dict[best_layer])
            else:
                for _, concept, stim_idx, _, _ in df_betas.index:
                    with open(f"outputs/lm_embeddings/{paradigm}/{model_name}/" \
                                f"{best_pooling}_pooling/{concept}.pkl", "rb") as f:
                        concept_embedding_dict = pickle.load(f)
                        embeddings.append(concept_embedding_dict[stim_idx][best_layer])
            
            X = np.vstack(embeddings)
            print("Feature matrix:", X.shape)
            y = df_betas["beta"].to_numpy()
            print("Outputs:", y.shape)
                        
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
            for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
                model = RidgeCV(alphas=np.asarray([10 ** x for x in range(-30, 30)]), alpha_per_target=True)
                model.fit(X[train_index, :], y[train_index])
                y_pred = model.predict(X[test_index, :])
                r = pearsonr(y_pred, y[test_index]).statistic # type: ignore
                data["parcel"].append(parcel)
                data["voxel_population"].append(voxel_population)
                data["alpha"].append(model.alpha_)
                data["pearsonr"].append(r)
                data["fold"].append(fold_idx+1)
                data["id"].append(participant_id)
                data["best_layer"].append(best_layer)
                data["best_pooling"].append(best_pooling)

    df = pd.DataFrame.from_dict(data)
    print(df) 

    if out_fpath:
        df.to_csv(out_fpath, index=False)
    return df

# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cache LM embeddings")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()),
                        required=True, help="Name of the model to use")
    parser.add_argument("--paradigm", type=str, choices=["sentences", "pictures", "word_clouds"], 
                        required=True, help="Paradigm to use (e.g., 'sentences', 'pictures')")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for computations")
    args = parser.parse_args()

    cache_lm_embeddings(args.model, args.paradigm, args.use_gpu)
