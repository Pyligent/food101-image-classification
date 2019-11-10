## Food 101 Image Classification Challenge Problem

![img](https://www.vision.ee.ethz.ch/datasets_extra/food-101/static/img/food-101.jpg)

### Results Summary

- **Model** : **ResNet50**
- **Training Epochs** : **16**
- **Random Transform with TTA**
- `top_1_accuracy:`  **89.63%**      
  `top_5_accuracy:`  **98.04%**
  
### 0. Overview

### Problem background
- The recent advent of deep learning technologies has achieved successes incomputer vision areas. But the status quo of computer vision and pattern recognition is still far from matching human capabilities, especially when it comes to classifying an image whose intra-category appearance might present more differences than its intercategory counterparts. 
- How to use the deep learning algorithm to classify the Food Images is a challengen problem. Also with high accuracy results, the solution of this probelm can been applied a lot of areas, such as marketing promotions, social media applications. It has the hugh business potential values.

### Data EDA and Data Augmentation
- In this notebook, we will build the pipeline from the raw dataset to model training by using Fastai data block API
- Data exploration is the key for understanding the problems. we will plot the data images throught different ways to understand the dataset
- Data augmentation is the key to improve the result in computer vision projects. In this notebook, we will use the simple and random transform to explore and train the model. Also will apply the augmentation in the test time to increase the accuracy

### Model training and SoTA results
- Deep Convolution Neural Network model have achieved remarkable results in image classification problems. For food 101 data the current SoTA results are:
    -  **InceptionV3** : 88.28% / 96.88% (Top 1/Top 5)
    -  **ResNet200** : 90.14% (Top 1)
    -  **WISeR** :  90.27% / 98.71%  (Top 1/Top 5)

- **My Results**: By using the pre-trained ResNet50 model, started by training the network with an image size of 224x224 for 16 epochs , training on image size of 512x512 for additional 16 epochs.   
   `top_1_accuracy:`  **89.63%**      
   `top_5_accuracy:`  **98.04%**
   
### Development Enviroment
- The whole training and development environment is on Google Cloud Platform
- The Deep Learning frame work is based on Fastai/PyTorch



### 1. Data Set

- This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels. The detailed information can be found this website https://www.vision.ee.ethz.ch/datasets_extra/food-101/

- Data Set Structure
    - All images can be found in the "images" folder and are organized per class. All image ids are unique and correspond to the foodspotting.com review ids. Thus the original articles can retrieved trough http://www.foodspotting.com/reviews/<image_id> or through the foodspotting api (http://www.foodspotting.com/api).
    
- Train and Test Data Set
   - In meta directory, the dataset provides the data train/test files list. Dataset has not imbalance problem.
   - For each category, 250 images are in the test set and 750 for the training set. The training set is split into 20% for the validation set and 80% for the training set.
   - Total Training set: 60600 images
   - Total Validation set: 15150 images
   - Total Test set: 25250 images

### 2. Data Exploration

#### In the Data exploration part, this notebook will use Fasiai Data Block API to build the `DataBunch` from the raw Data Set to feed the Model. 

#### 2.1 Datablock API

- The data block API will customize the creation of a DataBunch by isolating the underlying parts of that process in separate blocks. For the food-101 data set, this notebook will create the data from data frame. 
- The steps are as following:
    - Define the function `build_data_frame(path_name, file_name, img_format = 'jpg')` to create the data frame
    - Build the databunch via `ImageList.from_df`
    - Add the split the data into a training and validation sets?
    - Add the label the inputs?
    - Add data augment transforms
    - add the test set?
    - Wrap in dataloaders and create the DataBunch?
    
#### 2.1 Explore the Data set images

- Exploring and showing the images are important for the image classfication problems.This part will show the image from the three ways:
    - show images from the Databunch
    - show images from the same class /w Data Augmentation
        - define the functions `show_class_images` and `show_class_images_with_tfms`
    - show single /w Data Augmentation 
        - define the functions `show_img_augmentation`
        
### 3. Data Augmentation

- Data augmentation is a strategy that enables deep learning algorithm to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks. However, most approaches used in training neural networks only use basic types of augmentation. While neural network architectures have been investigated in depth, less focus has been put into discovering strong types of data augmentation and data augmentation policies that capture data invariances.So in this notebook will use the random transforms which apply flip, warp, rotate,zoom,lighting and contrast randomly

- Explore the basic Data augmentation transform in show-image functions
    - the basic the transform parameters will be defined as:
        - `max_rotate=15, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,p_affine=1., p_lighting=1.`
        
- Random Data Augmentation Transform for model training
    - Random transform will increase the variety of image samples and prevents overfitting. Randomly applying the rotate, zoom,lighting,warp and etc. 
    - Transform on training set: 
        - `Crop_pads, Affine_flip,Coord_warp,Affine_rotate,Affine_zoom,Lighting_brightness,Lighting_contrast`
    - Transform on validation set:
        - `Crops`

### 4. Model Training

- The results of the SoTA models for food classification are as following:
    - **InceptionV3** : by Hassannejad, Hamid, et al. ["Food Image Recognition Using Very Deep Convolutional Networks." Proceedings of the 2nd International Workshop on Multimedia Assisted Dietary Management. ACM, 2016.](http://dl.acm.org/citation.cfm?id=2986042)
        - Top-1 Accuracy : 88.28% 
        - Top-5 Accuracy : 96.88%
        - Augmentation: Flip, Rotation, Color and Zoom
        - Crops: 10 crops for validation
      
    - **ResNet200**: by Keun-dong Lee, et al.[NVIDIA DEEP LEARNING CONTEST 2016.](https://www.gputechconf.co.kr/assets/files/presentations/2-1650-1710_DL_Contest_%EC%A7%80%EC%A0%95%EC%A3%BC%EC%A0%9C_%EB%8C%80%EC%83%81.pdf)
        - Top-1 Accuracy : 90.14% (78 epoch)
        - Crops: Multi-Crops evaluation
    - **WISeR** : by . Martinel,et al. [Wide-Slice Residual Networks for Food Recognition.](https://arxiv.org/pdf/1612.06543.pdf)
        - Top-1 Accuracy : 90.27% 
        - Top-5 Accuracy : 98.71%
        - Augmentation: Flip, Rotation, Color and Zoom
        - Crops: 10 crops for validation
 
- From the SoTA models, the best archtecture is based on ResNet-ish model. I will use the standard **ResNet50** pre-trained model for training
    - First Step: Train images size 224x224 until the *Loss* started to converge **16 epoch**
    - Second Step: Train images size 512x512,  **16 epoch**

### 5. Evaluate and Test
- Test-Time Augmentation is an application of data augmentation to the test dataset. Applied the TTA into the test cases has impoved the accuracy. Specifically, it involves creating multiple augmented copies of each image in the test set, having the model make a prediction for each, then returning an ensemble of those predictions.
- Without TTA prediction:
    -  `top_1_accuracy:`  **88.36%**   
    -  `top_5_accuracy:`  **97.76%**   
    
- Applied TTA prediction:
    - `top_1_accuracy:`  **89.63%**   
    - `top_5_accuracy:`  **98.04%**

### 6. Results Analysis

#### Apply ClassificationInterpretation Interpretation methods for analyzing the results, furthermore, from the top losses cases, it is more easier to tune the data augmenetation parameters or focus on the specific class to train again.

- Investigate the Top losses by using `plot_top_losses` method
- `most_confused` method will give us the sorted descending list of largest non-diagonal entries of confusion matrix, presented as actual, predicted, number of occurrences.


### 7. Future Work

- For other cases, based on the most confused matrix and top losses plot images, we could adjust the augmentation hyper-parameters 
- For some cases, we could specially only train the some classes and improve their accuracy
- Use more deeper network or newer structure, such as **ResNet200** or **EfficientNet** to train the model





