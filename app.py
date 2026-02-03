from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('./Student_Score_Prediction_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Student Score Prediction API!"

@app.route('/predict',methods = ['POST'])
def predict():
    try:
        data = request.get_json()
        hours = data.get("hours")
        
        if hours is None:
            return jsonify({"error" : "Missing 'hours' in request data"}), 400
        
        prediction = model.predict([[hours]])
        return jsonify({"Studied hours": hours, "Marks Obtained": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug = True)
