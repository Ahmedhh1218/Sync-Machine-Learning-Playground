# Boston Housing Price Prediction Project

**Author:** Ahmed Hisham Fathy Hassabou

## Project Overview

This project is an end-to-end implementation of a machine learning model for predicting housing prices in the Boston area. It utilizes the Boston Housing dataset and follows a structured workflow to achieve the following objectives:

- **Load and Prepare the Dataset**: The project starts by importing essential libraries and loading the Boston Housing dataset. The dataset includes features related to housing and the corresponding target variable, which is the price of houses.

- **Data Splitting**: The data is divided into training and testing sets. The training set is used for model training, while the testing set is reserved for model evaluation.

- **Model Building**: A Decision Tree Regressor model is employed to predict housing prices. This machine learning algorithm is suitable for regression tasks and will be fine-tuned for optimal performance.

- **Cross-Validation**: Cross-validation is performed to assess the model's performance, providing a robust way to measure its predictive accuracy.

- **Hyperparameter Tuning**: Grid search is used to find the best hyperparameters for the Decision Tree Regressor. Hyperparameters can significantly impact the model's behavior, and this process aims to optimize its performance.

- **Model Visualization**: The project includes data visualization using pair plots to examine the relationships between features and the target variable.

- **Residual Analysis**: Residuals, the differences between actual and predicted values, are visualized to understand how well the model fits the data.

- **Price Prediction**: The trained model is used to make predictions for a test house's features, demonstrating how the model can be applied to real-world scenarios.

## How to Use This Project

To run this project on your own machine or Jupyter Notebook environment, follow these steps:

1. **Prerequisites**: Ensure you have the necessary libraries installed. You can do this by running the import statements provided in the Jupyter Notebook.

2. **Dataset**: The project loads the Boston Housing dataset from an online source. If needed, you can replace the `data_url` with your own data source.

3. **Explore and Experiment**: Feel free to explore the project and experiment with different aspects, such as hyperparameter tuning, model evaluation, and more.

## Results

The project provides insights into the Boston Housing dataset and demonstrates the development of a machine learning model for housing price prediction. Here are some key results:

- **Cross-Validation RMSE Scores**: The model's performance is assessed through cross-validation, and the Root Mean Squared Error (RMSE) scores are reported.

- **Best Hyperparameters**: Grid search is used to find the best hyperparameters for the Decision Tree Regressor.

- **Test RMSE**: The final model's performance is evaluated on the test set, and the test RMSE is reported.

- **Data Visualization**: Pair plots are created to visualize the relationships between features and the target variable.

- **Residual Analysis**: Residuals are visualized to understand how well the model fits the data.

- **Price Prediction**: The model is used to predict the price of a test house based on its features.

## Conclusion

This project serves as a comprehensive guide to building, evaluating, and optimizing a machine learning model for housing price prediction. It can be a valuable resource for individuals interested in data analysis and machine learning in real estate and housing market applications.
