import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("pump_data.csv")

#removing instances where ON/OFF status of pump is unknown
data = data[data.BFPT_B_LP_SPEED_INPUT__1__ !=0]

#finding % of data missing
# print(data.isnull().mean()) 

#dropping columns where more than 49% values are missing

missing_thresh = len(data)*0.49
data = data.dropna(thresh=missing_thresh, axis = 1)


#splitting the filtered data into train and test sets
train_set, test_set = train_test_split(data, test_size = 0.20, random_state=10)



print(round(train_set.isnull().mean(),2)) 
print(round(test_set.isnull().mean(), 2))
# print(train_set.shape)
# print(train_set.isnull().sum())
# print(data.head(5))