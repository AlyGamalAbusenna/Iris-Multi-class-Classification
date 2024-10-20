# Iris-Multi-class-Classification
Multi-class Classification using PyTorch on Iris Dataset

## -> Objective
The goal of this project is to build a multi-class classification model using PyTorch to classify different species of Iris flowers based on four features: sepal length, sepal width, petal length, and petal width. The species are one of three classes:

1.Iris-setosa
2.Iris-versicolor
3.Iris-virginica
The model is trained using a neural network to accurately classify these species, aiming for an accuracy above 95%.

->Dataset Description
The dataset used in this project is the famous Iris dataset. The dataset contains 150 samples, each with:

4 features: sepal length, sepal width, petal length, and petal width.
1 target variable: species of the Iris flower, which can be one of three categories (Iris-setosa, Iris-versicolor, Iris-virginica).
In this project, the dataset was provided as a CSV file.

Dataset Columns:
1.Sepal Length: Continuous feature
2.Sepal Width: Continuous feature
3.Petal Length: Continuous feature
4.Petal Width: Continuous feature
Species: Target variable with three classes (Setosa, Versicolor, Virginica)

->Steps to Run the Code in Jupyter (Google Colab)
To run the project in Jupyter Notebook or Google Colab, follow these steps:

1.Upload the Dataset: Upload the provided CSV dataset (iris.csv) to your Google Colab environment.

1.Install Required Dependencies: Before running the code, ensure you have all the dependencies installed. You can do this by running the following command in your notebook:

!pip install torch scikit-learn pandas matplotlib seaborn

2.Load the Dataset: In the notebook, load the uploaded CSV file using pandas. The dataset is then split into features (X) and target (y).

3.Build and Train the Model: Follow the step-by-step cells in the notebook, which include:

Loading the dataset
Preprocessing and normalization
Defining the neural network model
Training the model on the training set
Evaluating the model using the test set
Visualizing the training process (loss, confusion matrix, ROC-AUC)
Evaluate the Model: After training, the notebook evaluates the model using:

Accuracy
Confusion Matrix
Precision, Recall, F1-Score
ROC-AUC curve (One-vs-Rest approach for multi-class)
Visualization: The notebook provides visualizations of the loss during training and ROC curves for each class.

->Dependencies
Ensure you have the following dependencies installed in your environment:

-Python 3.7+
-PyTorch: Install PyTorch using the official installation guide from PyTorch.

pip install torch torchvision torchaudio

-scikit-learn: For dataset splitting, scaling, and evaluation metrics.

pip install scikit-learn

-pandas: For loading and handling the dataset.

pip install pandas

-matplotlib: For plotting loss curves and visualizing ROC curves.

pip install matplotlib

-seaborn: For creating confusion matrix heatmaps.

pip install seaborn
