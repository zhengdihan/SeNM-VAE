# Dataset Preparation

1. Download the SIDD-Medium dataset from https://abdokamel.github.io/sidd/.
2. Use `./set_dataset.py` to create the training dataset:
```
python set_dataset.py --src_dir PATH/TO/SIDD-Medium/dataset \
                      --tar_dir PATH/TO/OUTPUT/FOLDER
```
3. The validation set of SIDD dataset can be found at https://drive.google.com/drive/folders/175oMG_plGSS-lHJVghhoD-e2K3Wj_ykM

# Pretrained Models

The pretrained SeNM-VAE model `SeNM-VAE_full_paired.pth` can be found at https://drive.google.com/drive/folders/1l4lMg6x6S6Km7P8M4WY9AIALYugzZkRg?usp=sharing.

## Validating the Noise Generation Model:

To valiate the generation ablity, run:

```
python main_test_translate.py --model_path PATH/TO/MODEL.pth \
                              --H_path PATH/TO/CLEAN/IMAGES \
                              --L_path PATH/TO/NOISY/IMAGES \
                              --G_path PATH/TO/OUTPUT/NOISY/IMAGES \
```

# Generate Synthetic Datasets

To generate synthetic noisy dataset, run:
```
python main_translate.py --model_path PATH/TO/MODEL.pth \
                         --H_path PATH/TO/CLEAN/IMAGES \
                         --G_path PATH/TO/OUTPUT/NOISY/IMAGES \
```

# Downstream Models

The finetuned NAFNet model can be found at https://drive.google.com/drive/folders/1l4lMg6x6S6Km7P8M4WY9AIALYugzZkRg?usp=sharing.

Use the code provided in the NAFNet repository: https://github.com/megvii-research/NAFNet to test the model.

# Training SeNM-VAE

Update the data path in the `./options/train.json` file, and run
```
python main_train.py
```
