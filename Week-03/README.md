# Academic Performance Prediction using TensorFlow DNN

## Objective
Develop a predictive model that predicts whether a student will **Pass or Fail** using:
- Study Hours
- Attendance Percentage

The project uses a TensorFlow/Keras Deep Neural Network (DNN) for binary classification.

## Project Structure

academic_performance_dnn/
│
├── student_performance.csv       # Dataset
├── train.py                      # Train and evaluate the DNN
├── predict.py                    # Predict result for a new student
├── requirements.txt              # Python dependencies
├── README.md                     # Instructions
└── results/                      # Created after training
    ├── academic_performance_dnn.keras
    ├── accuracy.png
    ├── loss.png
    └── test_predictions.csv

## Requirements
- Python 3.10 or 3.11 recommended
- pip
- Internet connection for installing packages

## Windows Setup

Open Command Prompt/PowerShell inside this project folder.

### 1. Create a virtual environment

    python -m venv venv

### 2. Activate it

PowerShell:

    .\venv\Scripts\Activate.ps1

Command Prompt:

    venv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Train the model

    python train.py

This will:
- Load the student dataset
- Split it into training and testing data
- Standardize the two features
- Build the TensorFlow DNN
- Train for 100 epochs
- Evaluate test accuracy
- Print classification report and confusion matrix
- Save the trained model
- Generate accuracy and loss graphs

### 5. Predict a new student's result

After training:

    python predict.py

Example:

    Enter study hours: 5.5
    Enter attendance percentage: 82

The program will display the pass probability and predicted result.

## DNN Architecture

Input Layer
    ↓
Dense Layer: 16 neurons, ReLU
    ↓
Dense Layer: 8 neurons, ReLU
    ↓
Output Layer: 1 neuron, Sigmoid
    ↓
Pass / Fail

## Why these components?

- ReLU: introduces non-linearity into hidden layers.
- Sigmoid: produces a probability between 0 and 1 for binary classification.
- Binary Cross-Entropy: suitable loss function for Pass/Fail classification.
- Adam: optimizer used to update model weights.
- StandardScaler: puts study hours and attendance on comparable scales.

## Important
The dataset is a small demonstration dataset created for the assignment. Therefore, the resulting accuracy should be interpreted as a demonstration of the model pipeline, not as a real-world measure of student performance.

## Expected Output
After training, the terminal prints:
- Test loss
- Test accuracy
- Classification report
- Confusion matrix

The `results` folder contains the trained model and graphs.
