import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
import joblib
from data_prep import X_train,X_test,Y_train,Y_test,scaler,x,y
from sklearn.model_selection import cross_val_score

##my data frame
model  = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

scores= cross_val_score(model,x,y,cv=5)
print(scores)
print(scores.mean())
print(scores.std())

print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

ConfusionMatrixDisplay.from_predictions(Y_test,y_pred)

importances = model.feature_importances_
feature_names = ['RMS', 'Kurtosis', 'Crest_factor', 'Skewness', 'Peak_to_Peak']
plt.figure()
plt.bar(feature_names, importances)
plt.title('Feature Importance')
plt.show()
