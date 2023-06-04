EXP-10 Data Science Process on Complex Dataset

#AIM

To Perform Data Science Process on a complex dataset and save the data to a file.

#ALGORITHM

Step 1 Read the given Data

Step 2 Clean the Data Set using Data Cleaning Process

Step 3 Apply Feature Generation/Feature Selection Techniques on the data set

Step 4 Apply EDA /Data visualization techniques to all the features of the data set

CODE
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size")

plt.show()

sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

#OUTPUt

![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/30d549cb-02a7-48d2-a8ca-9f66d098ad31)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/453f9959-a615-4d09-aa4a-e08a6a03d37d)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/30f580cc-468e-4634-87f0-52190b3ff02f)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/f30cf26e-27e5-4a57-a31a-0aa038ae9cf1)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/0e0b2d77-79bf-4d2e-88b5-6e66de0114ff)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/7fc2730f-1d87-46a2-a4de-58ad06bd9c88)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/12815d72-4259-48a9-8847-0041078a292d)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/42e67284-4319-4bd8-a4d9-538b562911fb)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/9f1acb6a-1ec4-46f5-9d0f-9d263c849ee1)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/384bdbd7-22ba-43a9-9e6c-10b5031811fc)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/8f91aab8-aaba-48f1-8025-cb216831ff57)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/01c696be-5d51-4d97-aa6d-05d5f38e59fb)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/f9976279-887a-4e5f-b54b-03eab9b832b6)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/cea2a65f-e08c-4b5b-9a6b-33cefab6f5f6)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/82d46ebe-7893-416b-aa9e-bbb2188f33c7)


![image](https://github.com/nivetharajaa/Ex-10/assets/120543388/d8f2756d-5bd0-421a-b1c0-1e73b711df14)


#RESULT

Thus Data Science Process on a complex dataset was performed successfully.
