# Income-Level-Prediction

1. Abstract

   This is a predictive modeling project to determine whether a person makes over 50K a year

   ​

2. Data Set Information

   I got this data set from UCI Machine Learning Repository

   "Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))" 

   ​

3. Attribute Information

   age: continuous. 
   workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
   fnlwgt: continuous. 
   education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
   education-num: continuous. 
   marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
   occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
   relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
   race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black. 
   sex: Female, Male. 
   capital-gain: continuous. 
   capital-loss: continuous. 
   hours-per-week: continuous. 
   native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

   ​

4. Modeling Summary

   1. Overview
     To find the best model for this classification issue, I will try three different models: Logistic Regression, Decision Tree and Random Forest. To compare the performance of these three models, we should consider following factors: accuracy (both from testset and cross-validation), confusion matrix (precision, recall and f-measure)

     ​

   2. Pre-processing techniques and optimization for different models

     1) Logistic Regression
     For logistic regression, it’s unnecessary to normalize data. Thus we didn’t apply normalization in this case. We split data into train and test set and applied cross-validation.

     2) Decision Tree
     Highlight: 
     •	Tree pruning 
     For Decision tree model, we don’t need to normalize data or select features. However, we need to apply tree pruning to avoid over-fitting. 

     3) Random Forest
     Highlight:
     •	Parameter tuning 
     Random forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision tree at training time and outputting the class that is the mode of classes (classification) of the individual trees. Random decision forests correct for decision tree’s habit of overfitting to their training set. For Random Forest, we also applied cross validation to deliver more robust accuracy.

   ​

   ​