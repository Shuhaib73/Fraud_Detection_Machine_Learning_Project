# **Credit Card fraud detection system using Machine Learning**

### The objective of this project is to develop machine learning models that can effectively identify fraudulent credit card transactions. To achieve this, we will conduct an in-depth analysis of customer-level data. 

## **Problem Statement**

### In this project, our primary objective is to develop an advanced fraud detection system for credit card transactions using machine learning techniques. 
### Our dataset comprises almost 1.3 million fully anonymized transactions, each clearly labeled as either fraudulent or non-fraudulent. It's important to note that the prevalence of fraudulent transactions in this dataset is remarkably low, accounting for just 0.4% of all transactions. This means that a simplistic system that labels every transaction as normal could achieve an accuracy exceeding 99.6%, even without detecting any fraudulent transactions. Consequently, we need to employ sophisticated adjustment techniques to address this class imbalance in the dataset.

## **Model Building on Imbalanced Data**
### When dealing with heavily imbalanced data, such as in this case where only 0.4% of the transactions are labeled as fraudulent (class 1) and 99.6% are non-fraudulent (class 0), it's important to select appropriate metrics for model evaluation.

## The project pipeline can be summarized in the following steps: 
#### **Data Understanding and Exploration** : This phase involves loading the data and Explore the characteristics of the available features. Understanding the data helps us select the relevant features for our final model.  
#### **Exploratory Data Analysis (EDA)**: Conduct in-depth exploratory data analysis, including univariate and bivariate analyses. Address data skewness if present, as it can impact model development.
#### **Data Preprocessing**: Handle missing values, outliers, and any data cleansing tasks. Consider feature engineering or transformation to enhance model performance.
#### **Feature Selection and Engineering**: Refine feature selection based on insights from EDA. Experiment with feature engineering techniques to improve model predictability.
#### **Model Building and Hyperparameter Tuning**: Explore a variety of machine learning models and fine-tune hyperparameters. Consider using different sampling techniques to address class imbalance.
#### **Model Evaluation**: Assess model performance using suitable metrics, emphasizing the accurate identification of fraudulent transactions. 
#### **Deployment**: Deploy the finalized model, potentially as a web application using Streamlit or other suitable technology.

### Machine Learning Models used in the project:
#### The project compares the results of different techniques :
##### - Decision Tree
##### - Random Forest
##### - Logistic Regression
##### - XGB Boost

## **Conclusion** :
### After running different models on Oversampled data: The selection of the Random Forest classifier with Random Oversampling is well-founded due to its ability to achieve exceptional accuracy, strong precision and recall values, and a balanced trade-off between the two. It demonstrates proficiency in addressing the challenges posed by imbalanced data and holds promise for accurate fraud detection in real-world scenarios.

### **The model's performance in classifying the data is quite promising:**
#### 1. **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve): With an AUC-ROC of 98.6, the model demonstrates an outstanding ability to distinguish between the positive and negative classes. It exhibits a high true positive rate while maintaining a low false positive rate
#### 2. **AUC-PR** (Area Under the Precision-Recall Curve): The AUC-PR score of 88.44 signifies that the model achieves a favorable balance between precision and recall, particularly for the positive class. It demonstrates a capacity to make accurate positive predictions while minimizing the risk of missing positive cases.
#### 3. Accuracy Score on Testing Data is 99%

### **In conclusion, the model exhibits strong classification performance by effectively distinguishing between classes and striking a favorable balance between precision and recall. **
