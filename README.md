Aim:

The aim of this project is to develop a sales prediction model for advertising campaigns based on budgets allocated to TV, radio, and newspaper advertisements. 
By analyzing the relationship between advertising budgets and sales, the model will help businesses make data-driven decisions to maximize their return on investment (ROI) in advertising.

Process:

1.Data Loading and Exploration:

The dataset is loaded using pandas from a CSV file named "Advertising.csv".
Initial exploratory data analysis (EDA) is conducted to understand the distribution, statistical summary, 
and relationships between variables. Tools like sns.pairplot and correlation matrix are used to visualize the data and explore the relationships between features.

2. Data Preprocessing:
   
Features (TV, radio, newspaper) are extracted as X, and the target variable (sales) is extracted as y.
The dataset is split into training and testing sets using train_test_split to evaluate the model's performance on unseen data.
Polynomial features are generated using PolynomialFeatures to capture non-linear relationships between advertising budgets and sales.

3.Model Training:

Linear Regression models are trained for different polynomial degrees (1 to 9) to find the best polynomial degree for prediction.
The R² scores for training and testing sets are stored and plotted to observe model performance across different polynomial degrees.

4.Model Evaluation:

The best polynomial degree is determined based on the R² scores from the test set.
Cross-validation (cross_val_score) is performed to ensure the model's robustness and prevent overfitting.
Finally, predictions are made for new advertising budget data, and the trained model is saved using joblib for future use.

5.Web Application Deployment:

A simple Flask web application is built to allow users to input advertising budgets and get predictions in real time.
The web app loads the pre-trained model and polynomial transformer to make predictions on new data provided by users via an HTML form.

Tools and Technologies:

1.Python Libraries:

pandas: For data loading, preprocessing, and manipulation.
numpy: For numerical operations.
matplotlib & seaborn: For data visualization and plotting.
scikit-learn: For machine learning model training, polynomial feature transformation, and cross-validation.
joblib: For saving and loading machine learning models.

2.Machine Learning:

PolynomialFeatures: For creating polynomial features from the input data to capture non-linear relationships.
Linear Regression: For modeling the relationship between advertising budgets and sales.
Cross-validation: To ensure the model generalizes well to unseen data.

Web Technologies:

Flask: For building the web application backend that serves predictions based on user inputs.
HTML: For creating the frontend user interface that collects user inputs and displays predictions.

Output:
The main output of the project is a predictive web application that allows users to input advertising budgets for TV, radio, and newspaper and receive a sales prediction. The model predicts sales with high accuracy based on the learned relationships from the dataset.


Cross-validation score:
The cross-validation result indicates how well the model generalizes, with an average score of around 0.98, signifying strong performance across different subsets of the data.

