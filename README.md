# Applications of Transfer Learning in Pneumonia Diagnosis
Joseph McCombs, Jakob Schuppan, Chris Cowell

This repository contains files that are configured to perform Transfer Learning with a GPU.

There are 4 separate models that can be run with varying success:

gleason_transfer.py
resnet_transfer.py
vgg16_transfer.py
xception_transfer.py

The gleason_transfer.py is configured to use the model weights of a pretrained model built to classify Gleason Scores, which will need to be downloaded from https://github.com/eiriniar/gleason_CNN with a MobileNet Convolutional base.

The other 3 files used the ImageNet weights with their convolutional bases in the file names.

There are also 4 separate log files which contain results of 1 run of each model. In order to visualize the Accuracy, Validation Accuracy, Loss and Validation Loss, the plot.py script can be utilized. 

The VGG16 model with ImageNet weights performed the best acheived a validation accuracy of 93.75% and a validation loss of 0.2 before overfitting.
 

