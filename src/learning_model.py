from sklearn.preprocessing import StandardScaler
from Signalverarbeitung import Signals,bearing_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay

##my data frame

df = pd.read_csv(r"C:\Users\manis\Documents\Masters\self project\Data\data_features\bearing_feature.csv")

y = df['Label']
x= df.drop('Label',axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape)
print(Y_train.value_counts())

model  = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))
ConfusionMatrixDisplay.from_predictions(Y_test,y_pred)

importances = model.feature_importances_
feature_names = ['RMS', 'Kurtosis', 'Crest_factor', 'Skewness', 'Peak_to_Peak']
plt.figure()
plt.bar(feature_names, importances)
plt.title('Feature Importance')
plt.show()