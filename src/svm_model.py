import matplotlib.pyplot as plt
from data_prep import X_train,X_test,Y_train,Y_test,scaler,x,y
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import  GridSearchCV,cross_val_score
from sklearn.inspection import permutation_importance

param_grid = {
    'C': [0.1,1,10,100],
    'gamma': [1,0.1,0.01,0.001],
    'kernel': ['rbf'],
}

grid = GridSearchCV(
    SVC(),
    param_grid=param_grid,
    refit = True,
    cv = 5)


grid.fit(X_train, Y_train)

y_pred = grid.predict(X_test)
result = permutation_importance(grid, X_test, Y_test, n_repeats=10, random_state=42)

scores= cross_val_score(grid,x,y,cv=5)
print(scores)
print(scores.mean())
print(scores.std())

"""
ConfusionMatrixDisplay.from_predictions(Y_test,y_pred)
feature_names = ['RMS', 'Kurtosis', 'Crest_factor', 'Skewness', 'Peak_to_Peak']
plt.figure()
plt.bar(feature_names, result.importances_mean)
plt.title('SVM Feature Importance')
plt.show()


print(accuracy_score(Y_test,y_pred))
print(classification_report(Y_test,y_pred))

joblib.dump(grid, 'bearing_model_svm.pkl')
joblib.dump(scaler, 'bearing_scaler_svm.pkl')
"""