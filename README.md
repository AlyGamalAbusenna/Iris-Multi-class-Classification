# Iris-Multi-class-Classification
Multi-class Classification using PyTorch on Iris Dataset

## Objective
The objective of this project is to develop a deep learning model using Keras to address a specific problem utilizing the provided dataset. This project also showcases how to visualize the model's training process using TensorBoard, evaluate its performance using ROC-AUC curves, and analyze both training and validation losses to gain insights into the model's learning behavior.

## Dataset Description
The dataset used in this project is focused on [insert type of data here, e.g., image classification, sentiment analysis, etc.]. The dataset is divided into training and validation subsets to facilitate model training and to evaluate its performance. It is important to provide detailed information about the dataset, including its source and format, to help users understand its structure and purpose.

## Steps to Run the Code in Jupyter
1. **Clone the Repository**: First, clone the GitHub repository to your local machine:
   ```bash
   git clone <repository_link>
   cd <repository_name>
   ```

2. **Open Jupyter Notebook**: Launch Jupyter Notebook to open the project notebook (`Deep2_keras.ipynb`):
   ```bash
   jupyter notebook
   ```

3. **Install Dependencies**: Ensure all necessary dependencies are installed by running the following command in a terminal:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**: Open the Jupyter Notebook file (`Deep2_keras.ipynb`) and execute the cells sequentially:
   - The notebook includes code for loading the dataset, preprocessing, model training, and performance evaluation.
   - Execute each cell step-by-step, following the logical flow from data loading to model training.

## Dependencies and Installation Instructions
- **Python 3.7+**: Ensure you are using Python version 3.7 or higher.
- **TensorFlow**: Required for deep learning model training and evaluation. Install TensorFlow using the following command:
  ```bash
  pip install tensorflow
  ```
- **Keras**: The model is built using the Keras API, which is included with TensorFlow.
- **Matplotlib**: Required for visualizing ROC-AUC curves and other plots:
  ```bash
  pip install matplotlib
  ```
- **Scikit-Learn**: Used for calculating the ROC-AUC score and other evaluation metrics:
  ```bash
  pip install scikit-learn
  ```
- **Pandas**: Used for loading and handling the dataset:
  ```bash
  pip install pandas
  ```
- **Jupyter Notebook**: Required to run the code interactively:
  ```bash
  pip install notebook
  ```
- **TensorBoard**: For visualizing training logs and monitoring the model's learning progress:
  ```bash
  pip install tensorboard
  ```

## Visualizations and Performance Evaluation

### TensorBoard Visualizations
- **Training and Validation Loss**: During the model training process, the loss for both training and validation datasets is recorded and can be visualized using TensorBoard. To start TensorBoard and view these graphs, run the following command in the terminal:
  ```bash
  tensorboard --logdir=logs/fit
  ```
- **Histograms and Metrics**: TensorBoard also provides histograms and metric visualizations that are useful in understanding model performance during training.

### ROC-AUC Curves
- After training, the model's performance is evaluated using the ROC-AUC metric, which is commonly used for evaluating classification models. The ROC curve provides insights into the model's ability to distinguish between classes, while the AUC score summarizes the overall performance.
- The ROC-AUC curve is plotted using Matplotlib, and this plot can be found in the notebook under the performance evaluation section.

## Notes
- Update the paths to the dataset or other relevant files as per your local setup.
- The TensorBoard logs are saved in the `logs/fit` directory. You may modify this path as per your requirements.

Feel free to explore the notebook, modify the hyperparameters, and observe how these changes affect the model's performance!


# Important plots 
![image](https://github.com/user-attachments/assets/82c35d07-6686-4778-8791-edefafe53981)

(Scatter plot) 

![image](https://github.com/user-attachments/assets/bc034ddb-0a09-4f76-b124-6be86799bc0a)

(ROC curve)
