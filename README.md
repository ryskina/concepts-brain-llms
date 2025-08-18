# concepts-brain-llms
Code and data for the paper ["Language models align with brain regions that represent concepts across modalities"](https://arxiv.org/abs/2508.11536) (COLM 2025)

🛠️ Under construction, more info coming soon!

## Downloading data

* The per-participant fMRI data (preprocessed with [GLMsingle](https://github.com/cvnlab/GLMsingle/tree/main)) can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/1td7k_5UbkQ4jsNtt5yqLOB8Cm50GBzLd?usp=sharing) _(warning: large files!)_ <br>
For the code to run, the `GLMsingle_outputs_M*.tgz` files should be downloaded into `data/` and unzipped: `tar -xvf GLMsingle_outputs_M*.tgz`

* External files required by this code can be downloaded from the following sources:

| Location to store files (required by the code) | File description | Download link |
| --- | --- | --- |
| `data/brain_parcels/allParcels-language-SN220.nii` | [Fedorenko et al.](https://journals.physiology.org/doi/prev/20100421-aop/pdf/10.1152/jn.00032.2010)'s language network parcels | [Link](https://evlab.squarespace.com/s/allParcels-language-SN220.nii) |
| `data/brain_parcels/HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii.gz` & `data/brain_parcels/HCP-MMP1_on_MNI152_ICBM2009a_nlin.txt`  | [HCP-MMP1.0](https://www.nature.com/articles/nature18933) atlas (MNI volumetric projection NIfTI file & area labels text file) | [Link](https://figshare.com/articles/dataset/HCP-MMP1_0_projected_on_MNI2009a_GM_volumetric_in_NIfTI_format/3501911) |
| `data/brain_parcels/mask.volume.brainmask.nii` | [CONN](https://web.conn-toolbox.org/) whole-brain mask | [Link](https://github.com/alfnie/conn/blob/master/utils/surf/mask.volume.brainmask.nii) |
| `data/stimuli/images/*/*.jpg`  | [Pereira et al.](https://www.nature.com/articles/s41467-018-03068-4)'s picture paradigm stimuli images | [Link](https://osf.io/crwz7/) (files inside `IARPA_expt1_stim_images.zip`) |

## Reference

```
@inproceedings{ryskina2025language,
  title={Language models align with brain regions that represent concepts across modalities},
  author={Maria Ryskina and Greta Tuckute and Alexander Fung and Ashley Malkin and Evelina Fedorenko},
  year={2025},
  booktitle={Second Conference on Language Modeling},
  url={https://arxiv.org/abs/2508.11536}
}
```
