
# Visual-XAI-project

## CNN vs ScatNet
Implementation of a 2D Convolutional Neural Network (CNN) and a ScatNet for image classification, comparison of the parameters for understanding which is the best for the task under analysis, and display and comparison the filters extracted from the CNN (first convolutional layer) and the ScatNet. 
Reduce the number of images until ScatNet performs better than CNN. 
Finally, application of XAI algorithms on the two networks implemented and a statistical analysis on the final attributions. 





### Requirements

Install conda env:

```console
  $ conda create --name <env> --file requirements.txt
```

[Download Dataset](https://www.kaggle.com/datasets/jorgebuenoperez/datacleaningglassesnoglasses) and extract it in the root folder with archive_488 named folder.

```console
  root
   |-- img
   |-- model
   |-- archive_488
   |   |-- Images
   |   |   |-- Images
   |   |   |   |-- glasses
   |   |   |   |   |-- ...
   |   |   |   |-- no_glasses
   |   |   |   |   |-- ...
   |   |-- glasses
   |   |-- no_clear
   |   |-- no_glasses
   |-- ...
   |-- ...
```



    
  

## Dataset



We select a dataset that contains 4920 RGB labelled images of people with glasses (2769) or without glasses (2151) with shape 1024x1024.[1] The dataset is divided into 3936 train images and 984 test images. We resize all the images in 128x128 and we apply data augmentation (RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ColorJitter) on the train dataset to increase the image’s diversity. 

![Dataset](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/dataset.png?raw=true)

## Confusion Matrix CNN
We load the best CNN model that has a 98% of accuracy and make inference on it with the test set. We computed f1 score, accuracy and the confusion matrix and the returned values are:
- F1 score:  0.99
- Accuracy:  0.99

![Confusion Matrix CNN](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/confusion%20matrix%20cnn.png?raw=true)


## Confusion Matrix ScatNet
We load the best Scatnet model that has a 89.59% of accuracy and make inference on it with the test set. We computed f1 score, accuracy and the confusion matrix and the returned values are:
- F1 score: 0.94
- Accuracy: 0.92

![Confusion Matrix ScatNet](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/confusion%20matrix%20scatnet.png?raw=true)

## When Scatnet Works Better
We take the initial dataset and we select only 
- 46 images of people with glasses 
- 32 images of people with no glasses
We found that CNN's best accuracy on fold on train is 66.7% and on test is 59.3%.\
On the other hand, Scatnet’s best accuracy on fold on train is 73.33% and on test is 92.4%.

![When Scatnet Works Better](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/CM_less_img.png?raw=true)

# XAI algorithms

## Integrated gradients by scratch
![Integrated gradients by scratch](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/IG_scratch.png?raw=true)
*Example of the results of integrated gradients by scratch CNN (sx) and Scatnet (dx), in the first case with the zero baseline and in the second case with the random normalized baseline* 

##  Integrated gradients Captum
![Captum](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/IG_captum.png?raw=true)
*Example of the results of Captum’s integrated gradients CNN (sx) and Scatnet (dx), in the first case with the zero baseline and in the second case with the random normalized baseline*
## LIME
![LIME](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/LIME.png?raw=true)
*CNN (sx) and Scatnet (dx) Results of Lime in three different images*
## SHAP
![SHAP](https://github.com/enricozorzi/Visual-XAI-project/blob/main/img/SHAP.png?raw=true)
*CNN (first line) and Scatnet (second line) examples of shap on two different images, in the first case with glasses and in the second one without glasses*
## Authors

- [@albertocaporusso](https://github.com/albertocaporusso)
- [@SerenaDeAntoni](https://github.com/SerenaDeAntoni)
- [@enricozorzi](https://github.com/enricozorzi)


