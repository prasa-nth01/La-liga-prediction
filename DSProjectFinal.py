#!/usr/bin/env python
# coding: utf-8

# In[91]:


#data preprocessing
import pandas as pd
#produces a prediction model in the form of an ensemble of weak prediction models, typically decision tree
import xgboost as xgb
#the outcome (dependent variable) has only a limited number of possible values. 
#Logistic Regression is used when response variable is categorical in nature.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
#displayd dat
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[92]:


# Read data and drop redundant column.
loc = '/Users/prashanth/Downloads/DS Project/'

data = pd.read_csv(loc + 'final_dataset.csv')

# Preview data.
display(data.head())


#Full Time Result (H=Home Win, D=Draw, A=Away Win)
#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

#Input - 12 other features (fouls, shots, goals, misses,corners, red card, yellow cards)
#Output - Full Time Result (H=Home Win, D=Draw, A=Away Win) 


# In[93]:


data.keys()


# In[94]:


# Correlation Matrix for dataset
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,10)) 
sns.heatmap(data.corr(), annot= True)


# In[95]:


# Remove few column
data2 = data.copy().drop(columns =['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG',
       'HTGS', 'ATGS', 'HTGC', 'ATGC',
       'HM4', 'HM5','AM4', 'AM5', 'MW', 'HTFormPtsStr',
       'ATFormPtsStr', 'HTFormPts', 'ATFormPts', 'HTWinStreak3',
       'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5', 'ATWinStreak3',
       'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
       'DiffPts'] )


# In[96]:


data2.keys()


# In[97]:


data2.head(10)


# In[98]:


#what is the win rate for the home team?

# Total number of matches.
n_matches = data2.shape[0]

# Calculate number of features. -1 because we are saving one as the target variable (win/lose/draw)
n_features = data2.shape[1] - 1

# Calculate matches won by home team.
n_homewins = len(data2[data2.FTR == 'H'])

# Calculate win rate for home team.
win_rate = (float(n_homewins) / (n_matches)) * 100

# Print the results
print("Total number of matches: {}".format(n_matches))
print ("Number of features: {}".format(n_features))
print( "Number of matches won by home team: {}".format(n_homewins))
print ("Win rate of home team: {:.2f}%".format(win_rate))


# In[99]:


# Visualising distribution of data
from pandas.plotting import scatter_matrix

#the scatter matrix is plotting each of the columns specified against each other column.
#You would have observed that the diagonal graph is defined as a histogram, which means that in the 
#section of the plot matrix where the variable is against itself, a histogram is plotted.

#Scatter plots show how much one variable is affected by another. 
#The relationship between two variables is called their correlation
#negative vs positive correlation

#HTGD - Home team goal difference
#ATGD - away team goal difference
#HTP - Home team points
#ATP - Away team points
#DiffFormPts Diff in points
#DiffLP - Differnece in last years prediction

scatter_matrix(data2[['HTGD','ATGD','HTP','ATP','DiffFormPts']], figsize=(15,15))


# PREPARING THE DATA

# In[119]:


# Separate into feature set and target variable
#FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
X_all = data2.drop(['FTR'],1)
y_all = data2['FTR']

# Standardising the data.
from sklearn.preprocessing import scale

#Center to the mean and component wise scale to unit variance.
cols = [['HTGD','ATGD','HTP','ATP']]
for col in cols:
    X_all[col] = scale(X_all[col])
    


# In[101]:


#last 3 wins for both sides
X_all.HM1 = X_all.HM1.astype('str')
X_all.HM2 = X_all.HM2.astype('str')
X_all.HM3 = X_all.HM3.astype('str')
X_all.AM1 = X_all.AM1.astype('str')
X_all.AM2 = X_all.AM2.astype('str')
X_all.AM3 = X_all.AM3.astype('str')

#we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
   ''' Preprocesses the football data and converts catagorical variables into dummy variables. '''
   
   # Initialize new output DataFrame
   output = pd.DataFrame(index = X.index)

   # Investigate each feature column for the data
   for col, col_data in X.iteritems():

       # If data type is categorical, convert to dummy variables
       if col_data.dtype == object:
           col_data = pd.get_dummies(col_data, prefix = col)
                   
       # Collect the revised columns
       output = output.join(col_data)
   
   return output

X_all = preprocess_features(X_all)
print ("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))




# In[102]:


# Show the feature information by printing the first five rows
print("\nFeature values:")
display(X_all.head())


# Spliting the dataset

# In[103]:


from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.3,
                                                    random_state = 2,
                                                    stratify = y_all)


# Applying the Logistic Regression

# In[104]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# In[105]:


Y_pred = classifier.predict(X_test)


# In[106]:


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, Y_pred)


# In[107]:


sns.heatmap(cm, annot=True,fmt='d')


# In[108]:


print(classification_report(y_test, Y_pred))


# In[109]:


from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_all)

# Split and train with polynomial features
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_all, test_size=0.3, random_state=2, stratify=y_all)
classifier.fit(X_train, y_train)
Y_pred_poly = classifier.predict(X_test)

# Evaluate
cm_poly = confusion_matrix(y_test, Y_pred_poly)
print("Confusion Matrix with polynomial features:\n", cm_poly)

# Plotting the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix with Polynomial Features')
plt.show()


# Applying the SVM

# In[110]:


#fitting the SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier.fit(X_train, y_train)


# In[111]:


#predicting result
Y_pred = classifier.predict(X_test)


# In[112]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred)


# In[113]:


sns.heatmap(cm, annot=True, fmt='d')


# In[114]:


print(classification_report(y_test, Y_pred))


# Applying the XGBoost

# In[115]:


from sklearn.preprocessing import LabelEncoder

# Initialize the label encoder
le = LabelEncoder()

# Fit and transform the labels to numeric values
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Now fit the XGBoost model with the encoded labels
from xgboost import XGBClassifier
classifier = XGBClassifier(seed=82)
classifier.fit(X_train, y_train_encoded)

# Predict the results
y_pred = classifier.predict(X_test)

# Optionally, you can inverse transform the predictions back to original labels
y_pred_labels = le.inverse_transform(y_pred)

# Print the classification report with original labels
print(classification_report(y_test, y_pred_labels))


# In[116]:


sns.heatmap(cm, annot=True,fmt='d')


# Clearly XGBoost seems like the best model as it has the highest F1 score and accuracy score on the test set.

# Tuning the parameters of XGBoost.

# In[117]:


# Import 'GridSearchCV' and 'make_scorer'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb

# Parameters to tune
parameters = {
    'learning_rate': [0.1],
    'n_estimators': [40],
    'max_depth': [3],
    'min_child_weight': [3],
    'gamma': [0.4],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'scale_pos_weight': [1],
    'reg_alpha': [1e-5]
}

# Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# Make an f1 scoring function using 'make_scorer'
f1_scorer = make_scorer(f1_score, pos_label=1)  # Change pos_label to 1

# Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, scoring=f1_scorer, param_grid=parameters, cv=5)

# Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train_encoded)

# Get the best estimator
clf = grid_obj.best_estimator_
print(clf)

# Define predict_labels function
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score and accuracy. '''
    y_pred = clf.predict(features)
    return f1_score(target, y_pred, pos_label=1), sum(target == y_pred) / float(len(y_pred))

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, X_train, y_train_encoded)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1, acc))

f1, acc = predict_labels(clf, X_test, y_test_encoded)
print("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1, acc))


# In[118]:


from time import time
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))

def predict_labels(clf, features, target_encoded, pos_label):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target_encoded, y_pred, pos_label=pos_label), sum(target_encoded == y_pred) / float(len(y_pred))

def train_predict(clf, X_train, y_train, X_test, y_test, pos_label):
    ''' Train and predict using a classifier based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}...".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train, pos_label)
    print("Training F1 score and accuracy: {:.4f}, {:.4f}.".format(f1, acc))
    
    f1, acc = predict_labels(clf, X_test, y_test, pos_label)
    print("Testing F1 score and accuracy: {:.4f}, {:.4f}.".format(f1, acc))

# Sample data setup (replace with your actual data loading)
# X_train, X_test, y_train, y_test = ...

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
pos_label_encoded = label_encoder.transform(['H'])[0]  # Get the encoded value for 'H'

# Initialize the classifiers
clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=912, kernel='rbf')
clf_C = xgb.XGBClassifier(seed=82)

# Train and evaluate models
train_predict(clf_A, X_train, y_train_encoded, X_test, y_test_encoded, pos_label_encoded)
print('')
train_predict(clf_B, X_train, y_train_encoded, X_test, y_test_encoded, pos_label_encoded)
print('')
train_predict(clf_C, X_train, y_train_encoded, X_test, y_test_encoded, pos_label_encoded)
print('')


# In[ ]:




