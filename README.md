# Alphabet Soup Funding Analysis

## Overview of the Analysis

The purpose of this analysis is to develop a binary classifier using a neural network model that can predict whether applicants funded by the nonprofit foundation Alphabet Soup will be successful in their ventures. This model will help the foundation select applicants with the best chances of success, thereby optimizing their funding decisions and ensuring resources are allocated effectively.

## Results

### Data Preprocessing

- <strong>Target Variable(s):</strong>
  - The target variable for the model is `IS_SUCCESSFUL`, which indicates whether the funding provided to an organization was used effectively.

- <strong>Feature Variable(s):</strong>
  - The feature variables used in the model include:
    - `APPLICATION_TYPE`
    - `AFFILIATION`
    - `CLASSIFICATION`
    - `USE_CASE`
    - `ORGANIZATION`
    - `STATUS`
    - `INCOME_AMT`
    - `SPECIAL_CONSIDERATIONS`
    - `ASK_AMT`

- <strong>Removed Variable(s):</strong>
  - The columns `EIN` and `NAME` were removed from the input data as they are identifiers and do not provide useful information for the prediction task.

## Compiling, Training, and Evaluating the Model

### Neurons, Layers, and Activation Functions:

- The initial neural network model was designed with the following architecture:
  - <strong>Input Layer:</strong> The number of neurons in the input layer corresponds to the number of features in the dataset after preprocessing.
  - <strong>Hidden Layer 1:</strong> 80 neurons with the `ReLU` (Rectified Linear Unit) activation function, chosen for its effectiveness in handling non-linear relationships.
  - <strong>Hidden Layer 2:</strong> 30 neurons with the `ReLU` activation function to further capture complex patterns in the data.
  - <strong>Output Layer:</strong> 1 neuron with a `sigmoid`activation function to produce a probability output for binary classification.

### Model Performance:

    - The initial model did not achieve the target accuracy of 75%. It was evaluated using the test dataset, and the accuracy was below the desired threshold.

### Steps Taken to Increase Model Performance:

- <strong>Attempt 1:</strong>
    - In the first model, I used 3 hidden layers with 80, 30, and 20 neurons each, utilizing the `ReLU` activation function. `ReLU` was chosen for its ability to effectively handle non-linear relationships by introducing non-linearity to the model, which helps in learning complex patterns. For the output layer, the `Sigmoid` activation function was used to produce a probability output for the binary classification task.

- <strong>Attempt 2:</strong>
    - In the second model, I experimented with the `Tanh` activation function. `Tanh` is a non-linear activation function that outputs values between -1 and 1, which can center the data and make the model's learning process more stable. I used 3 hidden layers with 80, 50, and 30 neurons, respectively. This was an exploratory attempt to see if `Tanh` would increase the model's accuracy, although I was unsure if it would outperform `ReLU`.

- <strong>Attempt 3:</strong>

    - In my third and final model, I used auto-optimization to determine the best model architecture and hyperparameters to achieve an accuracy higher than 75%. I employed Keras Tuner's Hyperband approach to optimize the model. This process allowed the tuner to select the activation functions (either `ReLU` or `Tanh`) and decide the number of neurons in each layer. The model could have anywhere from 1 to 6 hidden layers, with neurons ranging between 1 and 10 in each layer. The output layer used the `Sigmoid `activation function to handle the binary classification task. This approach was taken to systematically explore different architectures and hyperparameters to find the optimal configuration for the model.

## Summary

The deep learning model built for Alphabet Soup provides a foundation for predicting the success of funded applicants. Despite several optimization attempts, the model struggled to consistently achieve the desired accuracy of 75%. This suggests that while the neural network model captured some patterns, there is still room for improvement.

## Recommendation:

To enhance the prediction accuracy, it is recommended to explore other machine learning models such as Random Forest or Gradient Boosting Machines. These ensemble methods are often more effective at handling complex datasets with mixed data types and can provide better performance for classification tasks. Additionally, leveraging techniques like feature engineering and hyperparameter tuning could further improve model accuracy.
