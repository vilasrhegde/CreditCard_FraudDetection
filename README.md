# A Credit Card Fraud Detection Model 
> A logistic regression model to predict the fraud happened in credit card transactions.

## How it is done?
1. Loading the [dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download)
2. This dataset has nearly **3 Lakh** entries!
3. Checking for null values ```data.isnull().sum()```
4. Counting the values for label 0 and 1 using ```data['Class'].value_counts()```
5. Here we found highly unbalanced datapoints.   
    > label 0 has 99% more values than label 1
6. Separate the data from analysis as legit and fraud
7. Compare the values for both transactions using ```data.groupby('Class').mean()```
8. Build a **Sample dataset** consisting similar distribution of legit & fraud
9. Concat two dataframes by pandas as ```pd.concat([legit_sample,fraud],axis=0)```
10. Observe mean of new dataframe by using ```new_df.groupby('Class').mean()```
11. Here we need to make sure the nature of the data unchanged. Which means mean difference in any column remain same in old & new dataframes
12. Split the data into features and targets
13. Split the data into Training and Testing 
      - Here we used 20% for test data and
      - Stratified target to maintain distribution of 0 and 1 in both training and testing data
14. Creating the model by instantiating the ```LogisticRegression()```
15. Training the model by calling fit() function
16. Model evaluation
    - Training data accuracy score
    - Testinng data accuracy score
17. We got 92% for training data and 90% for testing data
18. One thing to note here is why we calculated accuracy for training data is, to check overfitting and underfitting for the model
19. If the training data accuracy came 50% then it means we **overfit** the data
20. If we get low train data accuracy and very high test data accuracy then model is **underfitted**

## Packages needed
- numpy
- pandas
- train_test_split from sklearn.model_selection
- accuracy_score from sklearn.metrics
- LogisticRegression from sklearn.linear_model
