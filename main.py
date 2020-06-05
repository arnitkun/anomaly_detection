import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from pyod.models.knn import KNN
from numpy import savetxt
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

data = pd.read_csv("pump_data.csv", index_col=0)

#removing instances where ON/OFF status of pump is unknown
data = data[data.BFPT_B_LP_SPEED_INPUT__1__ !=0]

#dropping columns where more than 49% values are missing
missing_thresh = len(data)*0.49
data = data.dropna(thresh=missing_thresh, axis = 1)

#creating imputer that will place median where values are missing
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(data.iloc[:, :])
data.iloc[:, :] = imputer.transform(data.iloc[:, :])

#dropping empty rows
data.dropna()

#finding % of data missing
print(data.isnull().sum()) 
print(data.shape)

data.to_csv('out.csv', index=False)

#splitting the filtered data into train and test sets
# train_set, test_set = train_test_split(data, test_size = 0.20, random_state=10)

trainer_name = 'KNN'
trainer = KNN(n_neighbors=50, method='mean')

trainer.fit(data)

scores = trainer.decision_function(data)

savetxt('scores.csv', scores, delimiter=",")
plt.plot(scores)
plt.show()




