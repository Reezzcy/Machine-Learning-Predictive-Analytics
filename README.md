# Machine Learning Project Report - Nicolas Debrito

## Project Overview

In today's digital age, the concept of smart cities has become a key goal for many cities around the world. Smart cities integrate advanced technologies to improve the quality of life of their citizens, optimise the use of resources, and support sustainable development. One of the key elements of smart cities is the increasingly common use of smart devices, such as smartphones, tablets, laptops, and Internet of Things (IoT) devices. These devices are used in various sectors, from households to industries, to improve operational efficiency and provide better services to citizens.

According to (Metallidou, 2020) in the journal Energy efficiency in smart buildings: IoT approaches to achieve the goals of smart cities and energy efficiency, more than just smart reconstruction of new buildings and transformation of existing buildings into Nearly Zero Energy Buildings (NZEB) is needed, but also increased transparency of Energy Performance Certificates. According to the European Commission, current Energy Performance System inspections are inefficient because they cannot guarantee the initial and ongoing performance of a building's technical systems. Energy Performance Certificates should ensure that the performance of technical systems installed, replaced or upgraded is well documented and that all parameters required to measure energy consumption are checked and meet minimum energy performance requirements.

In light of the Smart City Readiness indicators and European legislation objectives for the gradual transformation of buildings into smart buildings, as well as to improve Building Certification and compliance checks related to energy performance, a management system was developed that will check energy consumption. The inspection and certification process starts with an inspection of the smart devices in the building and measures technical parameters such as energy consumption, daily usage hours, incidence of malfunctions, and lifespan of the devices.

This project aims to develop a machine learning model that can classify whether a smart device is using energy efficiently or not. By utilising machine learning techniques, we hope to produce a model that is able to identify energy consumption patterns and provide recommendations for more efficient use, contributing to the smart city goal of optimising resource use, and improving the quality of life for citizens.
  
Reference: Metallidou, C. K., Psannis, K. E., & Egyptiadou, E. A. (2020). [Energy efficiency in smart buildings: IoT approaches.](https://ieeexplore.ieee.org/abstract/document/9050775) IEEE Access, 8, 63679-63699.

## Business Understanding

### Problem Statements

- How to ensure the devices used in buildings actually meet the set energy efficiency standards?
- How can the inspection and certification of Energy Performance Systems be effective?
- How can the implementation of machine learning models help classify devices as efficient?

### Goals

- Identify the methods, procedures, and tools needed to inspect devices used in buildings according to set energy efficiency standards.
- Generate ways to improve the effectiveness of the Energy Performance System inspection and certification process through machine learning models, thereby speeding up the certification process.
- Describe how machine learning models can be applied in practice to classify the energy efficiency of smart devices, as well as the expected benefits of applying this technology in smart cities. 

### Solution statements

- Using Random Forest Classifier, Bagging Classifier, and Gradient Boosting to classify smart devices based on energy efficiency, with high accuracy and sufficient values of other evaluation metrics.
- Performed hyperparameter tuning with the help of GridSearchCV to find suitable parameters to be used in each classification model based on the highest accuracy. 

## Data Understanding

The dataset used is a dataset containing technical parameters such as energy consumption, daily usage hours, incidence of malfunctions, age of the device, etc. totalling 5403 rows. [Kaggle: Predict Smart Home Device Efficiency Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-smart-home-device-efficiency-dataset). 

### The variables in the dataset are as follows:

- UserID: A unique identifier for each user.
- DeviceType: Type of smart home device (e.g., Lamp, Thermostat).
- UsageHoursPerDay: Average hours per day the device is used.
- EnergyConsumption: Daily energy consumption of the device (kWh).
- UserPreferences: User preferences for device usage (0 - Low, 1 - High).
- MalfunctionIncidents: Number of reported malfunction incidents.
- DeviceAgeMonths: The age of the device in months.
- SmartHomeEfficiency (Target Variable): The efficiency status of smart home devices (0 - Inefficient, 1 - Efficient).

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Boxplot.png)

Boxplot Numerical Columns

Visualisation with boxplots is used to identify outliers in numerical columns by graphically displaying the distribution of data. The boxplot shows the quartiles of the data (Q1, Q2/Median, Q3) as well as the upper and lower bounds that help identify extreme values outside the interquartile range (IQR). Points outside these boundaries are considered potential outliers.

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Univariate_Categorical.png)

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Univariate_Numerical.png)

EDA with Univariate Analysis 

Exploratory Data Analysis (EDA) with Univariate Analysis is used to analyse data by examining one variable at a time with the aim of understanding the distribution, descriptive statistics, and other characteristics of that variable individually.

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Pairplot.png)

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Heatmap.png)

EDA with Multivariate Analysis

Multivariate Analysis in Exploratory Data Analysis (EDA) involves examining the relationships between two or more variables simultaneously which makes it possible to explore complex patterns between different variables in a dataset. 

EDA results:
- The distribution in the boxplot indicates that there are no outliers in the data.
- From the univariate EDA results, the distribution of values in the variable tends to be evenly distributed along the range of values, but the efficiency of the device tends to be 0 (inefficient).
- From the multivariate EDA results, each feature tends to be statistically independent or has no clear correlation pattern.

## Data Preparation

**Upsampling**

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Unbalance_Data.png)

Upsampling is used to handle class imbalance in the dataset, in the visualisation above class 0 (inefficient) is larger than class 1 (efficient). This technique increases the number of samples in class 1 (efficient) by adding duplicates or creating synthetic samples of the class. The goal is to increase the class minority so that the machine learning model can learn better from the data without bias towards the more dominant class 0 (inefficient).

**One Hot Encoding**

One hot encoding is used to convert categorical variables to numerical. This technique converts each value in the categorical variable into a binary vector where only one bit is set to 1 (hot), while the others become 0 (cold). It is used so that machine learning algorithms can process categorical data, so that by using one hot encoding, we can incorporate information about categorical variables into the model effectively.

**Principal Component Analysis (PCA)**

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/PCA.png)

Principal Component Analysis (PCA) is used to reduce the dimensionality of complex datasets by transforming the original data into a new, lower feature space. The main goal is to retain as much of the information present in the original data as possible, while reducing the number of dimensions required to represent the data. PCA is used to retrieve features with the number of components that represent 90% of the variation in the data in this case the n_components used are 2.

**Train Test Split**

Train Test Split is used to split the dataset into two subsets which are the training subset (trainset) and the testing subset (testset). The training subset is used to train the machine learning model, while the test subset is used to test the performance of the trained model on data that has never been seen before. In this case, the split is done by dividing the data 80% for the training data set and 20% for the testset.

**Standardisation with StandardScaler**

Standard Scaler is used to scale each feature (column) of the dataset so that it has a mean (average) of zero and a variance (standard deviation) of one. This process helps in balancing the various scales of variables in the data, thus preventing some features from having a greater influence simply because of their larger numerical scale compared to other features.

## Modeling

1. **Random Forest Classifier**: Random Forest Classifier is an ensemble method that uses multiple decision trees to make predictions. Each decision tree is generated from a subsample of the dataset and uses splitting based on randomly selected features. 

Advantages: 
- Resistant to overfitting due to averaging multiple trees. 
- Able to handle imbalanced data.

Disadvantages: 
- Tends to be slow in the training and prediction phases on large datasets.

Parameters used in GridSearchCV:
- ‘n_estimators’: Number of decision trees in the ensemble. [50, 100, 200]
- ‘max_depth’: The maximum depth of each decision tree in the ensemble. [None, 10, 20]
- ‘min_samples_split’: The minimum number of samples required to split the internal nodes. [2, 5] 
- ‘min_samples_leaf’: The minimum number of samples required to be a leaf node. [1, 2] 
- ‘criterion’: A function to measure the split quality. ‘gini’ for Gini impurity and ‘entropy’ for Information Gain. [‘gini’, ‘entropy’] 
- ‘bootstrap’: Determines whether bootstrap samples are used when building trees. [True, False]

2. **Gradient Boosting**: Gradient Boosting Classifier is an ensemble method that builds a model incrementally by minimising the error (loss function) at each iteration using gradient descent. 

Advantages: 
- Able to handle imbalanced data well. 
- Can provide very accurate prediction results.

Disadvantages: 
- Prone to overfitting if hyperparameters are not set properly. 
- The training process can take a long time.

Parameters used in GridSearchCV:
- ‘n_estimators’: Number of learning iterations. [10, 20, 30] 
- ‘max_samples’: The proportion of the dataset used to train each estimator. [0.5, 0.7, 1.0]
- ‘max_features’: The maximum proportion of features considered when finding the best split for each estimator. [0.5, 0.7, 1.0] 
- ‘bootstrap’: Specifies whether subsampling is performed when building each estimator. [True, False]

3. **Bagging Classifier**: Bagging Classifier is an ensemble technique that takes a subsample of the dataset and trains multiple base estimators independently. Prediction is done by taking the average or majority vote of the predictions of each estimator. 

Advantages: 
- Reduces variance, improves model stabilisation. 
- Can be used with multiple base estimators.

Disadvantages: 
- Does not improve model bias. 
- Sensitive to noise in training data.

Parameters used in GridSearchCV:
- ‘learning_rate’: A learning rate that controls how fast the model learns from errors. [0.01, 0.1, 0.2] 
- ‘n_estimators’: The number of base estimators (e.g., decision trees) trained in the ensemble. [50, 100, 150]
- ‘max_depth’: The maximum depth of each base estimator in the ensemble. [3, 4, 5]
- ‘subsample’: The proportion of the dataset used to train each base estimator. [0.8, 0.9, 1.0]
- ‘min_samples_split’: The minimum number of samples required to split the internal nodes. [2, 5]
- ‘min_samples_leaf’: The minimum number of samples required to be a leaf node. [1, 2]
- ‘max_features’:  The maximum number of features considered when finding the best split. [None, ‘sqrt’, ‘log2’]

After performing the evaluation using GridSearchCV and considering various parameters that affect the performance of the model, the Random Forest Classifier was shown to be the best model compared to the Gradient Boosting Classifier and Bagging Classifier. 

**Random Forest Classifier**

Random Forest Training Time:  0.314 Seconds

Accuracy Random Forest = 0.9028189910979229

Random Forest - Best Params: {'bootstrap': False, 'criterion': 'gini', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 50}

**Gradient Boosting**

Boosting Training Time:  0.633 Seconds

Accuracy Boosting = 0.8931750741839762

Boosting - Best Params: {'learning_rate': 0.2, 'max_depth': 5, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 150, 'subsample': 1.0}

**Bagging Classifier**

Bagging Training Time:  0.328 Seconds

Accuracy Bagging = 0.8968842729970327

Bagging - Best Params: {'bootstrap': False, 'max_features': 0.7, 'max_samples': 1.0, 'n_estimators': 20}

From the above results, Random Forest has the fastest training time among the three models, which is 0.314 seconds. This is very important in real applications, where a shorter training time can speed up the iteration process and model refinement. In addition, Random Forest also has the highest accuracy of 0.9028189910979229.

## Evaluation

The evaluation metrics used at this stage are:

1. Accuracy, Accuracy measures the proportion of correct predictions to overall predictions. This is the most intuitive metric, but may be less informative in the case of unbalanced datasets.

Formula: TP+TN / TP+TN+FP+FN

2. Precision, Precision measures the proportion of positive predictions that are actually positive. This is important when the cost of false positive predictions is high.

Formula: TP/TP+FP

3. Recall, Recall measures the proportion of positive data that is actually identified as positive by the model. It is important when the cost of false negative predictions is high.

Formula: TP/TP+FN

4. F1 Score, F1 Score is the harmonic mean of precision and recall. It provides a balanced picture when both metrics are important and provides a balance between dealing with False Positive and False Negative.

Formula: 2 x (Precision x Recall) / (Precision + Recall)

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/Barplot_Accuracy.png)

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/CM_RF.png)

| | precision | recall | f1-score |
|-|-|-|-|
|0| 0.92 | 0.89 | 0.90 |
|1| 0.89 | 0.92 | 0.90 |

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/CM_Boosting.png)

| | precision | recall | f1-score |
|-|-|-|-|
|0| 0.93 | 0.86 | 0.89 |
|1| 0.86 | 0.93 | 0.90 |

![image](https://raw.githubusercontent.com/Reezzcy/Machine-Learning-Predictive-Analytics/main/assets/CM_Bagging.png)

| | precision | recall | f1-score |
|-|-|-|-|
|0| 0.91 | 0.89 | 0.90 |
|1| 0.89 | 0.90 | 0.90 |

Based on the accuracy evaluation and other evaluation metrics (precision, recall, and F1-score), Random Forest Classifier was selected as the best model for this classification task. Although the Gradient Boosting Classifier showed a slightly higher precision for class 0, the Random Forest Classifier provided a better balance between precision, recall, and F1-score for both classes. 

To answer the business question, determining whether a device has met the set efficiency standards requires methods, procedures, and tools to be used. Machine learning models can be a solution to help determine whether a device is efficient through classification based on given technical parameters. The machine learning model will be used to classify a device as efficient or not through the given technical parameters. This will help the process of determining whether or not a device has met the set energy efficiency standards. This project successfully developed a classification model to identify the energy efficiency of smart devices using Random Forest Classifier, Gradient Boosting Classifier, and Bagging Classifier algorithms. After evaluation, the Random Forest Classifier model was selected as the best model with an accuracy of 90.3%. The contribution of this model is that decision-making related to certification and assessment of device energy efficiency can be done more quickly and accurately, reducing the need for time-consuming and costly manual evaluations, and the use of devices that are efficient in energy consumption which can help reduce operational costs and environmental impacts.
