Code and data for the paper ["Language models align with brain regions that represent concepts across modalities"](https://arxiv.org/abs/2508.11536) (COLM 2025)

🛠️ Under construction, more info coming soon!

## Downloading data

* The per-participant fMRI data (processed with [GLMsingle](https://github.com/cvnlab/GLMsingle/tree/main)) can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1td7k_5UbkQ4jsNtt5yqLOB8Cm50GBzLd?usp=sharing) _(warning: large files!)_ <br>
For the code to run, the `GLMsingle_outputs_M*.tgz` files should be downloaded into `data/` and unzipped: `tar -xvf GLMsingle_outputs_M*.tgz`

* External files required by this code can be downloaded from the following sources:

| Location to store files (required by the code) | File description | Download link |
| --- | --- | --- |
| `data/brain_parcels/allParcels-language-SN220.nii` | [Fedorenko et al.](https://journals.physiology.org/doi/prev/20100421-aop/pdf/10.1152/jn.00032.2010)'s language network parcels | [Link](https://evlab.squarespace.com/s/allParcels-language-SN220.nii) |
| `data/brain_parcels/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz` & `data/brain_parcels/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt`  | [HCP-MMP1.0](https://www.nature.com/articles/nature18933) atlas (MNI volumetric projection NIfTI file & area labels text file) | [Link](https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_MNI2009a_GM_volumetric_in_NIfTI_format/3501911) |
| `data/brain_parcels/mask.volume.brainmask.nii` | [CONN](https://web.conn-toolbox.org/) whole-brain mask | [Link](https://github.com/alfnie/conn/blob/master/utils/surf/mask.volume.brainmask.nii) |
| `data/stimuli/images/*/*.jpg`  | [Pereira et al.](https://www.nature.com/articles/s41467-018-03068-4)'s picture paradigm stimuli images | [Link](https://osf.io/crwz7/) (files inside `IARPA_expt1_stim_images.zip`) |

## fMRI processing (GLMsingle)

We use the fMRI data originally collected by [Pereira et al. (2018)](https://www.nature.com/articles/s41467-018-03068-4) (participants M01-M17), but apply a _different processing pipeline_ than the original paper. Ours uses the [GLMsingle](https://github.com/cvnlab/GLMsingle/tree/main) library (v. 0.0.1) to estimate the brain responses to experimental stimuli. For reference, we provide the code for this step in `fit_glm.py`. 

`fit_glm.py` takes in the preprocessed fMRI data (in NIfTI format) for each participant and saves a GLMsingle output object (`TYPED_FITHRF_GLMDENOISE_RR.npy`). The resulting GLMsingle outputs for each participant can be found at [this Google Drive link](https://drive.google.com/drive/folders/1td7k_5UbkQ4jsNtt5yqLOB8Cm50GBzLd?usp=sharing). 

The `"betasmd"` field of the GLMsingle output contains a 4D tensor of the participant's brain activations for each stimulus (`X x Y x Z x TRIAL`, where `(X, Y, Z)` are the voxel corrdinates and `TRIAL` is the chronological index of the stimulus presentation). The order in which the stimuli were presented to each participant for each paradigm is documented in `data/stimuli/stimuli_order_M*_{paradigm}.csv`.

## Semantic consistency

`analysis_uitls.py` contains the code for: 
 * computing the semantic consistency metric $C$ for a single voxel: `compute_consistency_in_voxel()`
 * computing $C$ for set of voxels: `compute_consistency_in_parcel()`
 * performing a permutation test to determine if a voxel's consistency is statistically significant: `voxel_consistency_permutation_test()`

`consistency_map.py` saves a whole-brain consistency map for a given participant. If the `--permutation` flag is set, the significance test outcome (1 if $p<0.05$, 0 otherwise) is recorded for each voxel instead of the raw $C$ value.

The individual participants' significance maps obtained independently for each half of the data are stored at `outputs/semantic_consistency/M*_split*.npy`. 

The probabilistic map aggregated over participants (shown below) is saved at `outputs/semantic_consistency/probabilistic_consistency_map.nii`
<img width="1787" height="1004" alt="Probabilistic map aggregated over participants and the two halves of the data" src="https://github.com/user-attachments/assets/950d6762-86c6-4749-b779-07730013ca0b" />

The resulting semantic consistency ROI map (below) used in this work is stored at `outputs/roi_map.nii` and `outputs/roi_map.npy`.

<img width="618" height="346" alt="Semantic consistency ROI map" src="https://github.com/user-attachments/assets/ebffed39-df17-403d-8296-9d208ddf16b1" />

## Running experiments: whole-brain encoding

1. Save whole-brain activations for each participant (e.g., `M01`) and paradigm (e.g., `sentences`):
```
python brain_encoding_whole_brain.py --step save_betas --paradigm sentences --id M01
```
2. After Step 1 is done for all participants, perform cross-validation to choose the best performing model layer and pooling for each paradigm (e.g., `sentences`) and model (e.g., `gpt2-xl`):
```
python brain_encoding_whole_brain.py --step choose_best_layer --paradigm sentences --model gpt2-xl
```
3. For the best layer and pooling, predict whole-brain activations for each participant (e.g., `M01`), model (e.g., `gpt2-xl`), and paradigm (e.g., `sentences`):
```
python brain_encoding_whole_brain.py --step predict --paradigm sentences --model gpt2-xl --id M01
```

## Citation

If you use our code or data, please cite:

```
@inproceedings{ryskina2025language,
  title={Language models align with brain regions that represent concepts across modalities},
  author={Maria Ryskina and Greta Tuckute and Alexander Fung and Ashley Malkin and Evelina Fedorenko},
  year={2025},
  booktitle={Second Conference on Language Modeling},
  url={https://arxiv.org/abs/2508.11536}
}
```
