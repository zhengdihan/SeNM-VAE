# Dataset Preparation

## Paired dataset
We provide the paired dataset used in our experiment at https://drive.google.com/drive/folders/175oMG_plGSS-lHJVghhoD-e2K3Wj_ykM?usp=sharing.

## Unpaired Dataset:
The unpaired dataset can be constructed in the following steps:

1. Download the SIDD-Medium dataset from https://abdokamel.github.io/sidd/.
2. Use `./set_unpaired_dataset.py` to generate image patches:
```
python set_unpaired_dataset.py --src_dir PATH/TO/SIDD-Medium/dataset \
                               --tar_dir PATH/TO/OUTPUT/FOLDER
```
3. The unpaired dataset will be located at `tar_dir/sub_clean` and `tar_dir/sub_noisy`.

# Pretrained Models

The pretrained SeNM-VAE models can be found in https://drive.google.com/drive/folders/1l4lMg6x6S6Km7P8M4WY9AIALYugzZkRg?usp=sharing.

## Validating the Noise Generation Model:

The validation set of SIDD dataset can be found in https://drive.google.com/drive/folders/175oMG_plGSS-lHJVghhoD-e2K3Wj_ykM.

1. Put the checkpoint file in the `./model_zoo` directory.
2. In the ./main_test_translate.py script, modify the following variables: `model_name`: Name of the noise generation model.`H_path`: Path to the clean images. `L_path`: Path to the noisy images.
3. Run `python main_test_translate.py`

# Generate Synthetic Datasets

To generate a synthetic noisy dataset, run:
```
python main_translate.py --model_path PATH/TO/MODEL.pth \
                         --H_path PATH/TO/CLEAN/IMAGES \
                         --G_path PATH/TO/OUTPUT/NOISY/IMAGES \
                         --image_size 256 \
                         --normalize  # Normalize the generated noisy images
```

# Downstream Models

The pretrained DRUNet models can be found in https://drive.google.com/drive/folders/1l4lMg6x6S6Km7P8M4WY9AIALYugzZkRg?usp=sharing.

## Testing the DRUNet Model

1. Use the code provided in the DPIR repository: https://github.com/cszn/DPIR to test the model.
2. Modify the code to handle images with 3 channels (RGB).

# Training SeNM-VAE

1. In the file `./options/train_xx_paired_data.json`, update the following paths: `dataroots_pair`: Path to the paired dataset.`dataroots_unpair`: Path to the unpaired dataset.
2. Run the training script: `python main_train.py -opt ./options/train_xx_paired_data.json`
