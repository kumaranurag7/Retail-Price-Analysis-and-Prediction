from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler as ss

ss1 = pickle.load(open('ss1.pkl', 'rb'))
rfecv_2 = pickle.load(open('rfecv_2.pkl', 'rb'))
xgb = pickle.load(open('xgb.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    InvoiceNo = request.form['InvoiceNo']
    StockCode = request.form['StockCode']
    Description = request.form['Description']
    Quantity = request.form['Quantity']
    CustomerID = request.form['CustomerID']
    Country = request.form['Country']
    
    Date = (request.form['Date'])
    year = int(Date.split('-')[0])
    month = int(Date.split('-')[1])
    day = int(Date.split('-')[2])
    
    Time = request.form['Time']
    hour = int(Time.split('-')[0])
    minute = int(Time.split('-')[1])
    
    month_sin = np.sin((month-1)*(2.*np.pi/12))
    month_cos = np.cos((month-1)*(2.*np.pi/12))
    day_sin = np.sin((day-1)*(2.*np.pi/12))
    day_cos = np.cos((day-1)*(2.*np.pi/12))
    hour_sin = np.sin((hour-1)*(2.*np.pi/12))
    hour_cos = np.cos((hour-1)*(2.*np.pi/12))
    minute_sin = np.sin((minute-1)*(2.*np.pi/12))
    minute_cos = np.cos((minute-1)*(2.*np.pi/12))
    
    
    # prepare a df
    pred_df=pd.DataFrame({'InvoiceNo':InvoiceNo, 'StockCode':StockCode,'Description':Description,
                            'Quantity':Quantity, 'CustomerID':CustomerID, 'Country':Country,
                            'year':year, 'month_sin':month_sin,'month_cos':month_cos,
                            'day_sin':day_sin,'day_cos':day_cos,'hour_sin':hour_sin,'hour_cos':hour_cos,
                            'minute_sin':minute_sin,'minute_cos':minute_cos}, index = [0])
    
    
    # Standardize
    pred_df = ss1.transform(pred_df)
    
    
    # rfecv
    pred_df = rfecv_2.transform(pred_df)

    pred = round(float(xgb.predict(pred_df)),2)
    print(pred)
    return render_template('after.html', pred=(pred)) #, place = place.lower()

if __name__ == "__main__":
    app.run(debug=False)















