# CNN Plant Classification Project

The primary objective of this project is to develop a Convolutional Neural Network (CNN) model that can accurately classify plants into different species based on images. By leveraging deep learning techniques and deploying the model using a web application, users can upload plant images and receive predictions in real-time. The dataset used for this project includes images of various plant species such as basil, mint, tulip, and others. This project demonstrates the power of machine learning in automating plant species identification and aims to provide a robust solution for botanical and agricultural research.

## Overview

1. **Dataset:** The dataset consists of 4,498 plant images divided into three sets: 3,691 for training, 539 for validation, and 268 for testing. The images represent 14 plant species, including basil, tulip, rose, and more. The dataset was sourced from Hugging Face and contains both images and corresponding labels.

2. **Task:** Build a CNN model using TensorFlow to classify the plant species from images and deploy the model using Streamlit for real-time predictions.

## 🛠 Tech Used
1. Python
2. TensorFlow
3. NumPy
4. Pandas
5. Hugging Face Datasets
6. Scikit-learn
7. Streamlit

## Workflow

### **1. Data Preprocessing and Augmentation:**
The first step involves preprocessing the images to prepare them for the CNN model. Images are resized to a uniform size of (224, 224) and normalized to a scale of 0 to 1. Additionally, data augmentation techniques such as rotation, zooming, shifting, and flipping are applied to enhance model generalization. The `ImageDataGenerator` is used for augmenting images dynamically during the training process.

### **2. CNN Model Architecture:**
The model consists of multiple convolutional layers followed by pooling layers. Key layers include:
- **Conv2D layers** with ReLU activation for feature extraction.
- **MaxPooling2D layers** to reduce spatial dimensions.
- **Dense layers** for classification with dropout layers to prevent overfitting.
- The final layer uses a **softmax activation** function, providing probabilities for each of the 14 classes.

### **3. Model Training:**
The model is trained using the categorical cross-entropy loss function, optimized with Adam, and monitored with accuracy as the primary metric. The training is done over multiple epochs, using a batch size of 32. The validation data is used to tune the model during training.

### **4. Model Evaluation:**
After training, the model's performance is evaluated using the test set. The accuracy of the model is calculated using `accuracy_score` from Scikit-learn. Predictions are generated, and the highest probability class is selected as the predicted label for each image. The model achieves a notable accuracy in plant species classification.

### **5. Deployment with Streamlit:**
The model is deployed as a web application using Streamlit. Users can upload plant images, which are passed through the CNN model to predict the plant species. The web interface is built using HTML and CSS, providing a simple and user-friendly experience. The prediction results are displayed instantly after the image is uploaded and processed.

## Screenshots

### Screenshot 1
<img src="https://github.com/HarishwarTG/CNN_Plant_Classification/blob/main/artifacts/screencapture1.png" width="600">

### Screenshot 2
<img src="https://github.com/HarishwarTG/CNN_Plant_Classification/blob/main/artifacts/screencapture2.png" width="600">

## Authors

- [@HarishwarTG](https://github.com/HarishwarTG)

This project showcases the use of CNN for plant classification and the integration of machine learning with real-time web applications using Streamlit.