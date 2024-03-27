# Dataset Preparation

1. Download the SIDD-Medium dataset from https://abdokamel.github.io/sidd/.
2. Use `./set_dataset.py` to create the training dataset:
```
python set_dataset.py --src_dir PATH/TO/SIDD-Medium/dataset \
                      --tar_dir PATH/TO/OUTPUT/FOLDER
```

# Pretrained Models

The pretrained SeNM-VAE model can be found at https://drive.google.com/drive/folders/1l4lMg6x6S6Km7P8M4WY9AIALYugzZkRg?usp=sharing.

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
