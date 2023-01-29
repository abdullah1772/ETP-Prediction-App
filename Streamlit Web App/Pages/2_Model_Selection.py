import streamlit as st
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
import seaborn as sns

#Read data
data = pd.read_csv(r"C:\Users\Abdullah\Desktop\zip\fivver\order_3\app\DataET0.csv" , sep=';' , decimal=',')

#Dropping unnecessary columns 
data.drop('Date/heure', inplace=True, axis=1)
data.drop('moy_WindDirection[deg]', inplace=True, axis=1)
data.drop('dernier_WindDirection[deg]', inplace=True, axis=1)

#Filiing null values with the median value
data.fillna(data.median(numeric_only=True).round(1), inplace=True)

#Extracting the ground truth column as labels 
Lables = data["ETP quotidien [mm]"]
data.drop("ETP quotidien [mm]", inplace=True, axis=1)

#Splitting training testing data
X_train, X_test, Y_train, Y_test = train_test_split(data, Lables, test_size=0.2, random_state=42)

#Encoding the labels for training and testing
label_encoder = preprocessing.LabelEncoder()
Y_Train_transformed = label_encoder.fit_transform(Y_train)
Y_Test_transformed = label_encoder.fit_transform(Y_test)

#Applying Random Forest classifier on the data 
st.header('Model Training')

st.subheader('Model 1. Random Forest')

# Create Random forest classifer object
Random_forest = RandomForestClassifier(n_estimators=100)

# Train Random forest Classifer
Random_forest.fit(X_train,Y_Train_transformed)

#Predict the response for test dataset
Random_forest_preds = Random_forest.predict(X_test)


# Model Accuracy, how often is the classifier correct?
Random_forest_accuracy = metrics.accuracy_score(Y_Test_transformed, Random_forest_preds)
st.write("Accuracy for random forest classifier :" , Random_forest_accuracy)
st.write("F1 score for random forest classifier :" , f1_score(Y_Test_transformed, Random_forest_preds, average='macro'))
Random_forest_F1_score_macro = f1_score(Y_Test_transformed, Random_forest_preds, average='macro')
st.write("F1 score with macro avg for Random_forest classifier :" , Random_forest_F1_score_macro)
Random_forest_F1_score_micro = f1_score(Y_Test_transformed, Random_forest_preds, average='micro')
st.write("F1 score with micro avg for Random_forest classifier :" , Random_forest_F1_score_micro)

st.subheader('Random_forest Model Evaluation')
#Plotting Random_forest acurracy metrics
Y_axis = [Random_forest_accuracy , Random_forest_F1_score_macro , Random_forest_F1_score_micro]
X_axis = ["Accuracy" , "F1_score_macro", "F1_score_micro"]

fig = plt.figure(figsize =(5, 5))

# Horizontal Bar Plot
plt.bar(X_axis, Y_axis,color = ['red', 'green','blue'])
 
# Show Plot
st.write(fig)


st.subheader('Model 2. Decision Tree')

# Create Decision Tree classifer object
Decision_Tree = DecisionTreeClassifier()

# Train Decision Tree Classifer
Decision_Tree.fit(X_train,Y_Train_transformed)

#Predict the response for test dataset
Decision_Tree_preds = Decision_Tree.predict(X_test)

# Model Accuracy, how often is the classifier correct?
Decision_tree_accuracy = metrics.accuracy_score(Y_Test_transformed, Decision_Tree_preds)
st.write("Accuracy for Decisoion Tree classifier :",Decision_tree_accuracy)
Decision_tree_F1_score_macro = f1_score(Y_Test_transformed, Decision_Tree_preds, average='macro')
st.write("F1 score with macro avg for Decisoion Tree classifier :" , Decision_tree_F1_score_macro)
Decision_tree_F1_score_micro = f1_score(Y_Test_transformed, Decision_Tree_preds, average='micro')
st.write("F1 score with micro avg for Decisoion Tree classifier :" , Decision_tree_F1_score_micro)

st.subheader('Decision Tree Model Evaluation')
#Plotting Decision tree acurracy metrics
Y_axis = [Decision_tree_accuracy , Decision_tree_F1_score_macro , Decision_tree_F1_score_micro]
X_axis = ["Accuracy" , "F1_score_macro", "F1_score_micro"]

fig = plt.figure(figsize =(5, 5))
 
# Horizontal Bar Plot
plt.bar(X_axis, Y_axis,color = ['red', 'green','blue'])
 
# Show Plot
st.write(fig)


st.subheader('Model 3. SVR')

# Create SVR regression object
SVR_regressor = SVR(C=1.0, epsilon=0.2)

# Train SVR regression
SVR_regressor.fit(X_train,Y_Train_transformed)

#Predict the response for test dataset
SVR_Regressor_preds = SVR_regressor.predict(X_test)

# Model Accuracy, how often is the regressor correct?
SVR_Accuracy = r2_score(Y_Test_transformed, SVR_Regressor_preds)
st.write("R2 score for SVR regression :",SVR_Accuracy)
SVR_mean_squared_log_error = mean_squared_log_error(Y_Test_transformed, SVR_Regressor_preds)
st.write("Mean_squared_log_error for SVR regression :",SVR_mean_squared_log_error)
SVR_mean_absolute_error = mean_absolute_error(Y_Test_transformed, SVR_Regressor_preds)
st.write("Mean_absolute_error for SVR regression :",SVR_mean_absolute_error)

st.subheader('SVR Model Evaluation')

Y_axis = [SVR_Accuracy , SVR_mean_squared_log_error , SVR_mean_absolute_error]
X_axis = ["R2 Score" , "MSLE", "MAE"]

fig = plt.figure(figsize =(5, 5))
 
# Horizontal Bar Plot
plt.bar(X_axis, Y_axis,color = ['red', 'green','blue'])
 
# Show Plot
st.write(fig)


st.subheader('Model Selection')

Accuracies = [Decision_tree_accuracy , Random_forest_accuracy , SVR_Accuracy]
Models = ["Decison_Tree" , "Random_Forest", "SVR"]

fig = plt.figure(figsize =(5, 5))
 
# Horizontal Bar Plot
plt.bar(Models, Accuracies,color = ['red', 'green','blue'])
 
# Show Plot
st.write(fig)

st.subheader('Model with best performance is SVR.')