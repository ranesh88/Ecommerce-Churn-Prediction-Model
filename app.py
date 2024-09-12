from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
from sklearn.preprocessing import StandardScaler
import sklearn
import pickle
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('rf_classifier.pkl','rb'))
preprocessor = pickle.load(open('preprocessor.pkl','rb'))
Ecomm=pd.read_csv('Ecommerce_Churn_Dataset_Final.csv')


def predict(Tenure,PreferredLoginDevice,CityTier,WarehouseToHome,PreferredPaymentMode,Gender,HourSpendOnApp,NumberOfDeviceRegistered,PreferedOrderCat,SatisfactionScore,MaritalStatus,NumberOfAddress,Complain,OrderAmountHikeFromlastYear,CouponUsed,OrderCount,DaySinceLastOrder,CashbackAmount):
    # Prepare features array
    features = np.array([[Tenure,PreferredLoginDevice,CityTier,WarehouseToHome,PreferredPaymentMode,Gender,HourSpendOnApp,NumberOfDeviceRegistered,PreferedOrderCat,SatisfactionScore,MaritalStatus,NumberOfAddress,Complain,OrderAmountHikeFromlastYear,CouponUsed,OrderCount,DaySinceLastOrder,CashbackAmount]],dtype = 'object')

    # transformed featured
    transformed_features = preprocessor.transform(features)

    # predict by model
    result = model.predict(transformed_features).reshape(1, -1)

    return result[0]

@app.route('/',methods=['GET','POST'])
def index():
    PreferredLoginDevice=sorted(Ecomm['PreferredLoginDevice'].unique())
    PreferredPaymentMode = sorted(Ecomm['PreferredPaymentMode'].unique())
    Gender=sorted(Ecomm['Gender'].unique())
    PreferedOrderCat=sorted(Ecomm['PreferedOrderCat'].unique())
    MaritalStatus = sorted(Ecomm['MaritalStatus'].unique())


    PreferredLoginDevice.insert(0,'Select your Preferred Login Device!')
    PreferredPaymentMode.insert(0, 'Select your Preferred Payment Mode! ')
    Gender.insert(0, 'Confirm your Gender ?')
    PreferedOrderCat.insert(0, 'Select your Prefered Order Category ?')
    MaritalStatus.insert(0, 'Select your Marital Status !')

    return render_template('index.html' , PreferredLoginDevice = PreferredLoginDevice,PreferredPaymentMode = PreferredPaymentMode,Gender = Gender , PreferedOrderCat = PreferedOrderCat ,MaritalStatus = MaritalStatus )

@app.route('/predict_route',methods=['GET','POST'])
@cross_origin()

def predict_route():
    if request.method == 'POST':
        Tenure  = int(request.form['Tenure'])
        PreferredLoginDevice = request.form['PreferredLoginDevice']
        CityTier = int(request.form['CityTier'])
        WarehouseToHome = float(request.form['WarehouseToHome'])
        PreferredPaymentMode = request.form['PreferredPaymentMode']
        Gender = request.form['Gender']
        HourSpendOnApp = float(request.form['HourSpendOnApp'])
        NumberOfDeviceRegistered =int(request.form['NumberOfDeviceRegistered'])
        PreferedOrderCat = request.form['PreferedOrderCat']
        SatisfactionScore = int(request.form['SatisfactionScore'])
        MaritalStatus = request.form['MaritalStatus']
        NumberOfAddress = int(request.form['NumberOfAddress'])
        Complain = int(request.form['Complain'])
        OrderAmountHikeFromlastYear = int(request.form['OrderAmountHikeFromlastYear'])
        CouponUsed = int(request.form['CouponUsed'])
        OrderCount = int(request.form['OrderCount'])
        DaySinceLastOrder = int(request.form['DaySinceLastOrder'])
        CashbackAmount =float(request.form['CashbackAmount'])



    prediction = predict(Tenure,PreferredLoginDevice,CityTier,WarehouseToHome,PreferredPaymentMode,Gender,HourSpendOnApp,NumberOfDeviceRegistered,PreferedOrderCat,SatisfactionScore,MaritalStatus,NumberOfAddress,Complain,OrderAmountHikeFromlastYear,CouponUsed,OrderCount,DaySinceLastOrder,CashbackAmount)
    prediction_text = "The Customer will Churn" if prediction == 1 else "The Customer will not Churn"

    return render_template('index.html', prediction=prediction_text)


if __name__=='__main__':
    app.run(debug=True)


