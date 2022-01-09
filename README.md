# Carvana-Image-Masking-Challenge
In this Repo, A #UNET in #Pytorch is presented for Image segmentation of #Carvana challenge.
To solve the carvana challenge for image segmentation in this repo a UNET shape CNN algoritem is 
Presented for image segmentation.
The dataset and the challenge is available on Kaggle in the following link:
https://www.kaggle.com/c/carvana-image-masking-challenge/data

The proposed algorithm could achieve score 0.995 in 6 epoches, and possible to provide better results under more training epoches.
The first part of the code makes it possible to set all parametters, and address directories to training, and test folders.
Models will be saved in MODEL_DIR, and the predicted results to show how the training process is going on will be saved in PREDICTED_IMG_DIR. If AUGMENT = True, the model would be trained under trainin augmentation, and it provides better results.
