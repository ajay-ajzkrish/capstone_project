# TRANSACTION DATA (CREDIT FRAUD)

“Data Transactions” (transactions.csv) - Fixed dataset used for use case fraud and clustering.

Sumber dataset : https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection/data

Machine Learning Classification Process for transaction.csv data.

The EDA notebook contains the data exploration stage, but will also include preprocessing. So that each feature of the data being explored will be clean of Null Values, and in accordance with the definition of each feature. The purpose of this data exploration is to find out more about the patterns of customers who commit fraud.

Data exploration is carried out in several stages and is carried out based on data groups. Exploration to target features - isFraud, then exploration to numerical, categorical groups and finally exploration to datetime features.

The next notebook in the series of Machine Learning processes, consists of HYPERPARAMETER Tuning and MACHINE LEARNING itself. So in practice, this process only focuses on building a prediction model. The models used in testing are Logistic Regression and K - Nearest Neighbors. Model creation is done using the pipelining method, so the results tend to be better. The model development also goes through several feature engineering stages, where the data will be tested for normality and outliers checking, feature selection, and cross-validation. Because the data used is very imbalanced, oversampling is done using the SMOTE method.


# FEATURE DESCRIPTION :

- accountNumber             = The account number of The customer
- customerId    =  The ID of The customer
- creditLimit               =  The amount of money that can be charged to The debit card
- availableMoney        = The amount of money in The debit card before adjusting for pending charges      
- transactionDateTime       = The transaction timestamp when it happened
- transactionAmount        = The amount of transaction 
- merchantName          = The merchant name of The particular transaction
- acqCountry             = The country where The merchant is located
- merchantCountryCode    = The country code for The specific merchant
- posEntryMode        = a code that tells The processor how The transaction was captured
- posConditionCode   = a code identifying transaction conditions at The point-of-sale or point-of-service
- merchantCategoryCode    = The merchant category/types
- currentExpDate       =  The expiry date of The credit card
- accountOpenDate      = The date when The customer open The credit card
- dateOfLastAddressChange    = The last date when The customer change The credit card address
- cardCVV = The actual card verification value
- enteredCVV       = The entered card verification value
- cardLast4Digits     = The last 4 digits of The debit card
- transactionType        = The types of transactions
- isFraud      = The status of The fraud transaction
- echoBuffer    = number of delayed response transactions
- currentBalance =  The current balance of The debit card
- merchantCity = The location for The specific merchant (City)
- merchantState   = The location for The specific merchant (State)
- merchantZip = The location for The specific merchant (Zip Code)           
- cardPresent =     The physical presence of The debit card in The transaction
- posOnPremises = The location of The point of sales          
- recurringAuthInd  = wheTher The auThentication recurred or not
- expirationDateKeyInMatch = The match between The expiration date in The system and what was inputted

# SUMMARY - EDA 


Even though it is quite difficult to find a pattern of fraudulent customers, we know that there are around 10,892 customers or around 2% of the total customers who are registered as fraudulent. We also know that fraud occurs almost evenly every day, and every year the number continues to increase. And if we look at the portion of calculations per month, we will find that there was a slight increase in March and October, although it was not very significant.

Fraudulent customers often shop at merchants who have online retail, fast food or other food categories, most of these merchants come from the US. The type of transaction that is often carried out is PURCHASE or purchase. And most of those who commit fraud do not carry a debit card, and do not have the same expiration date recorded in the system and the one entered.

Through the pair plot on the numerical features, it can be seen that there are features that have quite a patterned distribution. Even though the distribution pattern is difficult to see, if we look carefully, we will find a distribution pattern among fraudulent customers. Most credit limits are still in the range of 20,000 dollars or lower, while those without fraud are in the range of 30,000 to 50,000 dollars. It can also be seen that in customers who are fraudulent, they tend to have less available money than those who are not fraudulent.


# SUMMARY - MACHINE LEARNING

After oversampling, it is clear that the model is getting better than before. And in the end, the Logistic Regression model with scaling using Robust Scaler and the oversampling technique with SMOTE is the best model to use in predicting this dataset.

In essence, this model has 81% accuracy and 99% precision in the non-fraud category. Moreover, this model also has a fairly high recall, namely 82% in the no fraud category, meaning this model can correctly predict around 82% in the no fraud category. Even though the fraud category is still only around 42%, however, the results of this model version are better and more balanced compared to the results of the model version before balancing was carried out.

# CHALLENGES 

- Relatively large dataset size, more than 600,000 data. So the data processing takes a long time.
- Hyperparameter tuning takes about 5 to 6 hours. Fitting the model to training data takes about 30 to 40 minutes.
- The data is very unbalanced with a ratio of 98 : 2. Of course this makes it difficult to find patterns in EDA and modeling in MACHINE LEARNING.
- Description of the definition in some features is not clear.
