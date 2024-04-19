## **Credit Card fraud detection system using Machine Learning**

<p align="center">
    <img src="https://github.com/Shuhaib73/Fraud_Detection_Machine_Learning_Project/blob/project_branch/hack-fraud-card-code.jpg" width="450" height="250" />
</p>

### **You can access the web application by following this link: https://frauddetectionsystem.streamlit.app/**
#### The objective of this project is to develop machine learning models that can effectively identify fraudulent credit card transactions. To achieve this, we will conduct an in-depth analysis of customer-level data. 

## **Problem Statement**

### In this project, our primary objective is to develop an advanced fraud detection system for credit card transactions using machine learning techniques. 
``` The dataset comprises almost 1.3 million fully anonymized transactions, each clearly labeled as either fraudulent or non-fraudulent. It's important to note that the prevalence of fraudulent transactions in this dataset is remarkably low, accounting for just 0.4% of all transactions. This means that a simplistic system that labels every transaction as normal could achieve an accuracy exceeding 99.6%, even without detecting any fraudulent transactions. Consequently, we need to employ sophisticated adjustment techniques to address this class imbalance in the dataset.```

## **Model Building on Imbalanced Data**
    When dealing with heavily imbalanced data, such as in this case where only 0.4% of the transactions are labeled as fraudulent (class 1) and 99.6% are non-fraudulent (class 0), it's important to select appropriate metrics for model evaluation.

## The project pipeline can be summarized in the following steps: 
#### **Data Understanding and Exploration** : 
```In the initial phase of our project, we focus on understanding and exploring the dataset. This involves loading the data and delving into the characteristics of the available features. By performing exploratory data analysis (EDA), we gain insights into the distribution of variables, identify potential patterns, and understand the relationships between different features. This understanding guides us in selecting relevant features for our final model, laying the foundation for subsequent phases.```

#### **Exploratory Data Analysis (EDA)**: 
```Conducting an in-depth EDA is the next step, involving both univariate and bivariate analyses. This process provides valuable insights into the dataset, allowing us to address data skewness and make informed decisions that will impact the model development stage.```

#### <strong>Data Preprocessing</strong>: 
```The data preprocessing phase is crucial for ensuring the quality and reliability of our model. We address missing values, handle outliers, and perform any necessary data cleansing tasks. This step contributes to the overall data integrity and prepares the dataset for model training. Additionally, we consider feature engineering and transformations to enhance the model's performance by creating new meaningful features or transforming existing ones.```

#### <strong>Feature Selection and Engineering</strong>: 
```Building on insights gained from EDA, we refine our feature selection strategy to focus on the most influential variables. Feature engineering techniques are explored to further improve the predictability of our model. This phase aims to enhance the model's ability to capture relevant patterns in the data, contributing to better overall performance.```

#### <strong>Model Building and Hyperparameter Tuning</strong>: 
```The heart of the project lies in building and fine-tuning our model. We explore various machine learning models and fine-tuning hyperparameters. I carefully consider different sampling techniques to address class imbalance, ensuring that the model is robust and performs well across various scenarios.```

#### <strong>Model Evaluation</strong>: 
```Assessing the model's performance is critical. I have used appropriate metrics, emphasizing the accurate identification of fraudulent transactions. Rigorous evaluation ensures that the model meets the project objectives and provides reliable results.``` 

#### <strong>Deployment</strong>: Deployed the finalized model as a web application using Streamlit Python framework.
``` To deploy the fraud detection model, I have utilized Streamlit.``` 
**You can access the web application by following this link:** https://frauddetectionsystem.streamlit.app/

### Machine Learning Models used in the project:
#### The project compares the results of different techniques :
##### - Decision Tree
##### - Random Forest
##### - Logistic Regression
##### - XGB Boost

## **Conclusion** :
``` After running different models on Oversampled data: The selection of the Random Forest classifier with Random Oversampling is well-founded due to its ability to achieve exceptional accuracy, strong precision and recall values, and a balanced trade-off between the two. It demonstrates proficiency in addressing the challenges posed by imbalanced data and holds promise for accurate fraud detection in real-world scenarios.```

### **The model's performance in classifying the data is quite promising:**
#### 1. **AUC-ROC** (Area Under the Receiver Operating Characteristic Curve): With an AUC-ROC of 98%, the model demonstrates an outstanding ability to distinguish between the positive and negative classes. It exhibits a high true positive rate while maintaining a low false positive rate
#### 2. **AUC-PR** (Area Under the Precision-Recall Curve): The AUC-PR score of 90% signifies that the model achieves a favorable balance between precision and recall, particularly for the positive class. It demonstrates a capacity to make accurate positive predictions while minimizing the risk of missing positive cases.
#### 3. Accuracy Score on Testing & Training set is 97%

#### *In summary, the developed model demonstrates robust classification performance by adeptly distinguishing between classes and achieving a harmonious balance between precision and recall. The successful outcome underscores the model's efficacy in accurately categorizing instances, validating its potential as a reliable tool for the intended purpose. The culmination of meticulous data analysis, feature engineering, and model tuning has resulted in a solution that meets the project's objectives and holds promise for practical applications.*
