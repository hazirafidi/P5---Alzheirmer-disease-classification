# P5 - Alzheirmer disease classification
 Determine the Alzheimer's disease based on MRI brain image using Transfer Learning Method

## 1. Project Summary

This project is carried out to implement Deep Learning approach on image classification using Convolutional Neural Network (CNN) and Transfer Learning. The focus of the model training is to predict the Alzheimer Disease based on the MRI scan brain image. The dataset is obtained via [this link](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset). The dataset consist of 4 classes which belongs to class 1 (Mild Demented), Class 2 (Moderate Demented), Class 3 (Non Demented) and Class 4 (Very Mild Demented).

## 2. IDE and Framework

The project is built with Jupyter Notebook as the main IDE and the model is trained via Google Colab Notebook. The main frameworks used in this project are TensorFlow, Numpy, Matplotlib, OpenCV and Scikit-learn.

## 3. Methodology

The methodolgy of this project is inpired by Tensorflow Image Segmentation Tutorial. The documentation can be found via this [link](https://www.tensorflow.org/tutorials/images/classification).

### 3.1 Input Pipeline

The dataset files contains a 4 classes of Alzheimer Disease in the format of images. The dataset is split into train data and validation data and the validation data is further split into test data for prediction. No data augmentation is applied for the dataset.

### 3.2 Model Pipeline

The model architecture can be illustrate as figure below.

![model_architecture](https://user-images.githubusercontent.com/100177902/164650914-3da3b62e-fdd2-4fe3-9c18-bdcc795f1cc8.png)

Since transfer learning approach is applied for this project, the feature extraction layer from base model VGG16 is freezed and classifier, dropout and output layer are added to train the model with new dataset with the pre-trained model weights with 100 initial epochs. Early stopping is added with patience of 2 and monitor by 'loss'. The model is evaluated before training and can be obsreved as in figure below. 

![before_training_loss_accuracy.](https://user-images.githubusercontent.com/100177902/164651950-2b211b49-b039-4703-8ae3-6980327e2611.png)

Then, fine-tune is applied afterwards by fine-tuning at 10 Feature Extraction layers ahead to enhance the model by making the weights updating during backpropagation. This will ensure model adapted and more relevant to the current input image. The model is evaluated and can be observed as in figure below.

![after_training_loss_accuracy](https://user-images.githubusercontent.com/100177902/164653941-ee77e663-c7a9-4721-9b9e-19e46ab2cf5f.png)

## 4. Result

The model training process can be evaluated via its training loss/accuracy vs validation loss/accuracy. During the feature extraction model training, the training accuracy (orange) and validation accuracy (blue) is slowly converging but the accuracy percentage is only in the range of 60-65% and same goes to graph of training and validation loss. The raining stop at epoch 26. Refer to figure below.

Graph of Training and Validation Accuracy at Feature Extraction training stage. Both training and validation line converging.

![accuracy_at_feature_extraction_graph](https://user-images.githubusercontent.com/100177902/164656933-2e1feb69-ac5d-439e-b15e-52a5a5122329.png)

Graph of Training and Validation Loss at Feature Extraction training stage. Both training and validation line converging.

![loss_at_feature_extraction_graph](https://user-images.githubusercontent.com/100177902/164663576-1497c151-6e89-46f6-9f91-b1425eaedaf0.png)

At fine-tune stage, the model accuracy shows spike of increasing to 97% accuracy. The graph can be illustrated as in figure below.

Graph of Training and Validation Accuracy at Fine-tune stage.

![accuracy_at_fine_tune_graph](https://user-images.githubusercontent.com/100177902/164675518-906c5cc9-9e6c-4d09-bf00-a4b199a119e2.png)

Graph of Training and Validation Loss at Fine-tune stage.

![loss_at_fine_tune_graph](https://user-images.githubusercontent.com/100177902/164675595-48b66132-f875-4c4f-bbb6-b0f04c324e5a.png)

The model is test with test data and the result is obtained as in figure below.

![result-image.](https://user-images.githubusercontent.com/100177902/164676145-c86116f7-93e7-4fa6-9739-560ec3c737e2.png)

From the figure above, the test data image is used to make prediction. From the figure above we can say that the model predict the image pretty well.

## 5. Conclusion

The performance of the model is very good and satisfying. The model managed to achieved 97% accuracy and predicted the test data in accurate manner.



