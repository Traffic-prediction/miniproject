import pandas 
import numpy as np



data=pandas.read_csv(r"F:/Metro Traffic volume\Metro_Interstate_Traffic_Volume.csv")
print(data.head(5))



from sklearn.preprocessing import LabelEncoder  
le=LabelEncoder()  
data['holiday']=le.fit_transform(data['holiday'])  
data['weather_main']=le.fit_transform(data['weather_main'])  
data['weather_description']=le.fit_transform(data['weather_description']) 



data['date_time']=pandas.to_datetime(data.date_time,errors='coerce')


data['year'] = data['date_time'].dt.year 
data['month'] = data['date_time'].dt.month 
data['day'] = data['date_time'].dt.day 
data['hour'] = data['date_time'].dt.hour 
data['minute'] = data['date_time'].dt.minute


data=data.drop(['date_time'],axis=1)
data=data.drop(['weather_description'],axis=1)




x=data.iloc[:,[0,1,2,3,4,5,7,8,9,10,11]].values
y=data.iloc[:,[6]].values




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=9)


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)



from sklearn import metrics
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)*100))
print(metrics.mean_absolute_error(y_test,y_pred)*100)

'''import pickle
filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))'''

print(model.predict([[	7,	289.36,	0.0,	0.0,	75,	1,	2012,	2,	10,	10,	0]]))



