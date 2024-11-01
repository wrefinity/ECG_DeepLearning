# ECG_DeepLearning


ECG Classification using CNN, LSTM, and Hybrid CNN-LSTM Models
This project implements and evaluates Convolutional Neural Networks (CNN), Long Short-Term Memory networks (LSTM), and a hybrid CNN-LSTM model to classify ECG (electrocardiogram) signals into multiple classes. The goal is to compare the performance of these architectures on the task of identifying arrhythmias and other heart-related abnormalities based on ECG data.

### Dataset
The dataset used is sourced from the MIT-BIH Arrhythmia Database and made available on Kaggle. It contains labeled ECG data across multiple classes, where each sample has 188 features representing the sequential ECG measurements.

Train Data: mitbih_train.csv
Test Data: mitbih_test.csv
Each record contains 188 features with a label that classifies the heartbeat type. Preprocessing includes feature selection using Information Gain to select the 60 most relevant features, optimizing the performance and reducing training time.

### Project Structure
Data Preprocessing:

Data normalization and selection of top 60 features based on Information Gain.
Model Architectures:

CNN Model: Utilizes convolutional layers to extract spatial features from ECG sequences.
LSTM Model: Employs LSTM layers to capture temporal dependencies in the data.
Hybrid CNN-LSTM Model: Combines CNN for feature extraction and LSTM for sequence learning, aiming to leverage the strengths of both architectures.
Evaluation Metrics:

Accuracy
Confusion Matrix
Classification Report (Precision, Recall, F1-Score for each class)
ROC-AUC Score (One-vs-Rest for each class)


## Installation
Clone the repository and install dependencies:
```
git clone https://github.com/username/ECG-Classification-CNN-LSTM
cd ECG-Classification-CNN-LSTM
pip install -r requirements.txt

```


### Acknowledgements
MIT-BIH Arrhythmia Database: The dataset used in this project is from the MIT-BIH Arrhythmia Database, available on Kaggle.
TensorFlow and Keras: For building and training deep learning models.
Scikit-Learn: For evaluation metrics and feature selection.
