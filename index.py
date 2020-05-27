from flask import *
import pandas 
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

app=Flask(__name__)

@app.route("/")
@app.route("/home",methods=["GET","POST"])
def home():
	if request.method=="POST":
		Holiday=int(request.form['Holiday'])
		temp=float(request.form['Temp'])
		rain_1h=float(request.form['rain_1h'])
		snow_1h=float(request.form['Snow_1h'])
		clouds_all=int(request.form['Clouds_all'])
		weather_main=int(request.form['weather_main'])
		Day=int(request.form['Day'])
		Month=int(request.form['Month'])
		Year=int(request.form['Year'])
		Hour=int(request.form['Hour'])
		Minutes=int(request.form['Minutes'])
		
		data=pandas.read_csv(r"F:/Metro Traffic volume\Metro_Interstate_Traffic_Volume.csv")
		#print(data.head(5))



		
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




		
		x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.22,random_state=9)


		
		model=LinearRegression()
		model.fit(x_train,y_train)
		y_pred=model.predict(x_test)



		
		M_error=np.sqrt(metrics.mean_squared_error(y_test,y_pred)*100)
		A_error=metrics.mean_absolute_error(y_test,y_pred)*100
        
		
		Y=model.predict([[Holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,Day,Month,Year,Hour,Minutes]])
		y=list(Y[0])
		print(M_error,A_error,y)
		return render_template("home.html",M_error=M_error,A_error=A_error,Y=Y)
        




		       

		




				




		
	else:
		Days=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
		Months=[1,2,3,4,5,6,7,8,9,10,11,12]
		Years=[2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]
		return render_template("home.html",Days=Days,Months=Months,Years=Years)
	
	


        
if __name__=="__main__":
	app.run(debug=True)
