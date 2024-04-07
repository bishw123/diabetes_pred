from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict():
    
    load_model = pickle.load(open(r"model_diabetes.pkl", 'rb'))

    Pregnancies=3
    Glucose=235
    BloodPressure=450
    SkinThickness=4
    Insulin=40
    BMI=12
    DiabetesPedigreeFunction=45
    Age=45

    result= load_model.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(f"Prediction = result[0]")

    return f"Prediction : {result[0]}"


@app.route('/get_data',methods = ['POST'])
def get_data():
    data = request.form
    print(f"Data = {data}")
    
    Pregnancies=int(data['Pregnancies'])
    Glucose=float(data['Glucose'])
    BloodPressure=float(data['BloodPressure'])
    SkinThickness=float(data['SkinThickness'])
    Insulin=float(data['Insulin'])
    BMI=float(data['BMI'])
    DiabetesPedigreeFunction=float(data['DiabetesPedigreeFunction'])
    Age=float(data['Age'])

    user_input = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
    
    
    load_model = pickle.load(open(r"C:\Users\agraw\Documents\bishwa programs jupyter\GIT&FLASK&AWS\SHRI\L1\model_diabetes.pkl", 'rb'))
    result = load_model.predict(user_input)

    return {"Prediction": result[0]}


if __name__=="__main__":
    app.run(debug=False, host='0.0.0.0',post=8080)
