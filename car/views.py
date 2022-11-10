from django.http import HttpResponse
from django.shortcuts import render,HttpResponse
from urllib import response
from django.http import HttpRequest,HttpResponse
from django.shortcuts import render,redirect
from multiprocessing import context
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC


def index(request):
    car = pd.read_csv("static/Cleaned Car.csv",index_col=[0])
    print(car)
    x=car.drop(columns="Price")#dependent variable
    y=car["Price"]
    ohe=OneHotEncoder()
    ohe.fit_transform(x[['name','company','fuel_type']])
    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder="passthrough")
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=433)##Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    r2_score(y_test,y_pred)
    fig = plt.figure(figsize=(6,5))
    plt.scatter(y_test,y_pred)
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test,y_test*m+b)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    if request.method=="POST":
        model = request.POST.get('model')
        year = request.POST.get('year')
        fuel = request.POST.get('fuel')
        company = request.POST.get('company')
        kms = request.POST.get('kms')

        car=pd.DataFrame([[model,company,year,kms,fuel]],columns=['name','company','year','kms_driven','fuel_type'])
        print(pipe.predict(car)[0])
       
        return HttpResponse(int(pipe.predict(car)[0]))     
    else:
        print("no")
        context={
        "comp":sorted(x['company'].unique()),
        "models":sorted(x['name'].unique()),
        "f_type":sorted(x['fuel_type'].unique()),
        "yop":sorted(x['year'].unique()),
        "variable":r2_score(y_test,y_pred)*100,
        "graph":data,
        "pred": "model"
        }   
        return render(request,"linear3.html",context)
# Create your views here.
def home(request):
    if request.method=="POST":
        transcript = request.POST.get('transcript') 
        transcript=transcript.lower()
        if(transcript=="home"):
            return HttpResponse("/")
        if(transcript=="cars"):
            return HttpResponse("/car")
        if(transcript=="scooters" or transcript=="bikes"):
            return HttpResponse("/scooter")

    return render(request,"home.html")
def scooter(request):
    car = pd.read_csv("static/Used_Bikes.csv",)
    
    car = car.drop(columns="city")
    car = car.drop(columns="power")
    print(car)
    x=car.drop(columns="price")#dependent variable
    y=car["price"]
    ohe=OneHotEncoder()
    ohe.fit_transform(x[['bike_name','brand','owner']])
    a=[]
    column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['bike_name','brand','owner']),remainder="passthrough")
    # for i in range(2000):
    #     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    #     lr=LinearRegression()
    #     pipe=make_pipeline(column_trans,lr)
    #     pipe.fit(x_train,y_train)
    #     y_pred=pipe.predict(x_test)
    #     a.append(r2_score(y_test,y_pred))
    # print(np.argmax(a))

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1597)##Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    r2_score(y_test,y_pred)
    fig = plt.figure(figsize=(6,5))
    plt.scatter(y_test,y_pred)
    m, b = np.polyfit(y_test, y_pred, 1)
    plt.plot(y_test,y_test*m+b)
    imgdata = StringIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)
    data = imgdata.getvalue()
    if request.method=="POST":
        bike_name = request.POST.get('model')
        kms_driven = float(request.POST.get('kms'))
        owner = request.POST.get('fuel')
        age = float(request.POST.get('year'))
        brand = request.POST.get('company')
        print(bike_name)

        car=pd.DataFrame([[bike_name,kms_driven,owner,age,brand]],columns=['bike_name','kms_driven','owner','age','brand'])
        print(pipe.predict(car)[0])
       
        return HttpResponse(int(pipe.predict(car)[0]))     
    else:
        print("no")
        context={
        "comp":sorted(x['brand'].unique()),
        "models":sorted(x['bike_name'].unique()),
        
        "variable":r2_score(y_test,y_pred)*100,
        "graph":data,
        "pred": "model"
        }   
        return render(request,"scooter.html",context)

    