from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split

##my data frame

df = pd.read_csv(r"C:\Users\manis\Documents\Masters\self project\Data\data_features\bearing_feature.csv")

y = df['Label']
x= df.drop('Label',axis=1)

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)