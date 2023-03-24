import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

X = df.drop(["Personal Loan", "ID"], axis = 1)
y = df["Personal Loan"]


##Data Pre-Processing
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(12, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.fit(X_train, y_train, epochs=50, verbose=2)

model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer= "Adam", metrics=['accuracy'])

y_pred_tf = model.predict(X_test)
y_pred_tf = y_pred_tf > 0.5

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_tf))