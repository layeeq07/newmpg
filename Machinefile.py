import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df = pd.read_csv('mpgnew.data')
rfc = RandomForestRegressor()
x = df[['cylinders','displacement','HP','weight','accelartion','modelY','origin']]
y = df['mgp']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
rfc.fit(xtrain,ytrain)
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def machine(a:float,b:float,c:float,d:float,e:float,f:float,g:float):
    pred=rfc.predict([[a,b,c,d,e,f,g]])
    return str(pred)
    