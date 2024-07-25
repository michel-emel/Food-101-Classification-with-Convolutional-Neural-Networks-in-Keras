# Food-101 Classification with Convolutional Neural Networks in Keras

## Project Overview
This project focuses on classifying food images using Convolutional Neural Networks (CNNs) implemented in Keras. The CNN model is trained on the Food-101 dataset, which includes a wide range of food categories. The goal is to build a model that can accurately classify images into their respective food categories.

## Project Structure
- **`Food_101_Classification.ipynb`**: Jupyter Notebook containing the code for data preparation, model training, and evaluation.
- **`README.md`**: This file.

## Data Preparation
1. **Dataset Download**:
   - Uncomment and run the following commands to download and unzip the dataset:
     ```bash
     wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
     tar xzvf food-101.tar.gz
     ```

2. **Dataset Splitting**:
   - The notebook includes code to organize the dataset into training and testing directories based on metadata.

3. **Image Augmentation**:
   - The `ImageDataGenerator` is used to apply various augmentations such as flipping, rotation, and brightness adjustments to increase the diversity of the training data.

## Model Architecture
The CNN model includes:
- **Convolutional Layers**: Several layers with increasing filter sizes (32, 64, 128, 512) and ReLU activation.
- **Pooling Layers**: MaxPooling layers to reduce spatial dimensions.
- **Dropout Layers**: Applied after convolutional layers to prevent overfitting.
- **Fully-Connected Layers**: A Dense layer with 1024 units followed by a final Dense layer with softmax activation for classification.

## Training
- **Optimizer**: RMSprop with a learning rate of 0.0001.
- **Loss Function**: Categorical crossentropy.
- **Metrics**: Accuracy.
- **Epochs**: 100, with early stopping and model checkpoint callbacks to monitor performance and avoid overfitting.

## Evaluation
- **Testing**: The model's performance is evaluated using precision, recall, F-score, and accuracy metrics.
- **Prediction Visualization**: A subset of images is displayed with predictions to visualize model performance.

## Results
- The model achieved approximately 58.8% accuracy on both training and validation sets before early stopping was triggered to prevent overfitting.

