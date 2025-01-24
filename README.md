Animal Pictures Classification Using CNN


A convolutional neural network (CNN) project for classifying animal images from the Animal-10 dataset. This project demonstrates deep learning concepts using TensorFlow/Keras and provides a complete pipeline for training, evaluation, and testing.

Project Overview

This project aims to classify animal pictures into predefined categories using CNNs. The key steps include:

Data preprocessing (image resizing, normalization).
Model building using TensorFlow/Keras.
Training and evaluating the model.
Visualizing the results.
Dataset
Name: Animal-10 Dataset
Description: Contains 10 classes of animal images (e.g., dog, cat, lion, horse, etc.).
Preprocessing: Images are resized to 128x128 and normalized for training.
Project Structure

Animal-Pictures-Classification-CNN/
├── data/
│   ├── train/
│   ├── test/
├── models/
│   ├── cnn_model.h5
├── README.md
├── requirements.txt
└── animal_classification.py
Installation
Clone the repository:


git clone https://github.com/Mohamediaaraben/Animal-Pictures-Classification-CNN.git
cd Animal-Pictures-Classification-CNN
Install dependencies:

pip install -r requirements.txt
How to Use
1. Train the Model
Run the script to preprocess data, build the CNN model, and start training:

python animal_classification.py --mode train

2. Evaluate the Model
Evaluate the trained model on the test dataset:


python animal_classification.py --mode evaluate
3. Predict with the Model
Use the trained model to predict the class of a single image:

python animal_classification.py --mode predict --image_path path_to_image.jpg


----------------------------------------------------------------------------------------------------------
## Objectifs

- Préparer les données pour l'entraînement.
- Construire un modèle CNN performant.
- Entraîner et évaluer le modèle sur des ensembles de données de validation et de test.

---

## Prérequis

Installez les bibliothèques nécessaires avant de commencer :

```bash
pip install torch torchvision numpy matplotlib opencv-python

## Dataset

- **Name:** [Animal-10 Dataset](https://www.kaggle.com/alessiocorrado99/animals10)
- **Description:** Contains 10 classes of animal images (e.g., dog, cat, lion, horse, etc.).
- **Preprocessing:** Images are resized to 128x128 and normalized for training.

---

## Project Structure
Animal-Pictures-Classification-CNN/ ├── data/ │ ├── train/ │ ├── test/ ├── models/ │ ├── cnn_model.h5 ├── README.md ├── requirements.txt └── animal_classification.py


---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Mohamediaaraben/Animal-Pictures-Classification-CNN.git
   cd Animal-Pictures-Classification-CNN
2 .Install dependencies:
pip install -r requirements.txt

How to Use
1. Train the Model
Run the script to preprocess data, build the CNN model, and start training:
python animal_classification.py --mode train

2. Evaluate the Model
Evaluate the trained model on the test dataset:
python animal_classification.py --mode evaluate

3. Predict with the Model
Use the trained model to predict the class of a single image:
python animal_classification.py --mode predict --image_path path_to_image.jpg

Code Examples
Below is the core Python code for building and training the CNN model:
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(128, 128, 3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

Results
Training Accuracy: 95%
Validation Accuracy: 92%
Test Accuracy: 91%
Here is an example of the training loss and accuracy curves:


Future Improvements
Implement data augmentation to improve generalization.
Experiment with transfer learning using pre-trained models like ResNet or VGG.
Deploy the model using Flask or FastAPI for real-time predictions.
Contributing
Feel free to fork the repository, submit issues, or make pull requests to enhance the project.

