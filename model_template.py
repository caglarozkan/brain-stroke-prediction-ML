import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

""" Dataset Address:
https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset
"""

df=pd.read_csv("full_data.csv") #target -> stroke
print(df["work_type"].value_counts())
print(df.head())
print(df.info())
print(df.columns)
print(df.isnull().sum())


print(df["stroke"].value_counts())
print(df["stroke"].value_counts(normalize=True))
# The dataset is highly imbalanced (~5% stroke cases)
# Therefore recall will be prioritized and class_weight will be used.

sns.countplot(x="stroke",data=df)
plt.title("Stroke Distribution")
plt.show()

sns.scatterplot(data=df,x="age",y="bmi",hue="stroke")
plt.show()

#numeric and categorical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
num_cols.remove("stroke") #target col

cat_cols = df.select_dtypes(include="object").columns

print("Numerical:", num_cols)
print("Categorical:", cat_cols)

for col in ["age", "avg_glucose_level", "bmi"]:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="stroke", y=col, data=df)
    plt.title(f"{col} vs Stroke")
    plt.show()


X=df.drop("stroke",axis=1)
y=df["stroke"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

#scaling for logistic reg
preprocessor_scaled= ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), num_cols),
        ("categorical", OneHotEncoder(drop="first",handle_unknown="ignore"), cat_cols)
    ]
)
X_train_scaled = preprocessor_scaled.fit_transform(X_train)
X_test_scaled = preprocessor_scaled.transform(X_test)

#NOT SCALED (Random Forest)
preprocesser_no_scaled = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", num_cols),
        ("categorical", OneHotEncoder(handle_unknown="ignore",drop="first"), cat_cols)
    ]
)

X_train_no_scaled = preprocesser_no_scaled.fit_transform(X_train)
X_test_no_scaled = preprocesser_no_scaled.transform(X_test)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(class_weight="balanced")
logreg.fit(X_train_scaled, y_train)
y_pred = logreg.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix,recall_score,classification_report
print("-- Logistic Regression -- ")
print("Recall Score",recall_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))


from sklearn.svm import SVC
svc = SVC(kernel="linear",class_weight="balanced")
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)
print("-- SVC -- ")
print("Recall Score",recall_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))



from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100,class_weight="balanced_subsample")
rfc.fit(X_train_no_scaled, y_train)
y_pred = rfc.predict(X_test_no_scaled)

print("-- Random Forest -- ")
print("Recall Score",recall_score(y_test,y_pred))
print("Classification Report",classification_report(y_test,y_pred))
print("Confusion Matrix",confusion_matrix(y_test,y_pred))


# Threshold tuning to increase Recall (minimize False Negatives)
# Important for medical risk prediction problems
y_proba_rf = rfc.predict_proba(X_test_no_scaled)[:,1]

for t in [0.05,0.1, 0.15,0.2,0.3]:
    y_pred_custom = (y_proba_rf > t).astype(int)
    print("-- Random Forest -- ")
    print(f"Threshold: {t}")
    print("Recall:", recall_score(y_test, y_pred_custom))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))
    print("-"*40)


y_proba_log = logreg.predict_proba(X_test_scaled)[:,1]
for t in [0.15,0.2,0.3, 0.4, 0.5]:
    y_pred_custom = (y_proba_log > t).astype(int)
    print("Logistic")
    print("Threshold:", t)
    print("Recall:", recall_score(y_test, y_pred_custom))
    print(confusion_matrix(y_test, y_pred_custom))

pd.DataFrame(X_test_scaled).to_csv("(scaled_test).csv", index=False)
# Final Decision ->LOGISTIC REG THRESHOLD -> 0.4

import pickle
with open("model_complete.pkl", "wb") as f:
    pickle.dump({
        "model": logreg,
        "scaler":preprocessor_scaled,
        "threshold":0.4,
    },f)

# Compare train and test recall scores to check overfitting
y_train_pred = logreg.predict(X_train_scaled)
y_test_pred = logreg.predict(X_test_scaled)

print("\n--- RECALL SKORLARI KIYASLAMASI ---") #overfitting için
print(f"TRAIN Recall Score: {recall_score(y_train, y_train_pred):.4f}")
print(f"TEST  Recall Score: {recall_score(y_test, y_test_pred):.4f}")
