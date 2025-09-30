from flask import Flask, render_template, request
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Predictor Class
class AIBDGPredictor:
    def __init__(self, past_results):
        self.past_results = past_results[-100:]
        self.model_number = None
        self.model_type = None
        self.prepare_data()
    
    def prepare_data(self):
        X_num, y_num, X_type, y_type = [], [], [], []
        for i in range(3, len(self.past_results)):
            feature = self.past_results[i-3:i]
            X_num.append(feature)
            y_num.append(self.past_results[i])
            X_type.append(feature)
            y_type.append('Big' if self.past_results[i]>=5 else 'Small')
        self.X_num, self.y_num = np.array(X_num), np.array(y_num)
        self.X_type, self.y_type = np.array(X_type), np.array(y_type)
    
    def train_models(self):
        self.model_number = RandomForestClassifier(n_estimators=200)
        self.model_number.fit(self.X_num, self.y_num)
        self.model_type = RandomForestClassifier(n_estimators=200)
        self.model_type.fit(self.X_type, self.y_type)
    
    def predict_next(self):
        last_features = np.array(self.past_results[-3:]).reshape(1,-1)
        next_number = self.model_number.predict(last_features)[0]
        next_type = self.model_type.predict(last_features)[0]
        prob_number = np.max(self.model_number.predict_proba(last_features))*100
        prob_type = np.max(self.model_type.predict_proba(last_features))*100
        return next_number, next_type, round(prob_number,2), round(prob_type,2)

# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        numbers = request.form['numbers']
        past_results = [int(x) for x in numbers.split(',') if x.strip().isdigit()]
        if len(past_results) < 10:
            prediction = "Please enter at least 10 numbers."
        else:
            predictor = AIBDGPredictor(past_results)
            predictor.train_models()
            next_guess = predictor.predict_next()
            prediction = f"Next Number: {next_guess[0]}, Big/Small: {next_guess[1]}, Number Prob: {next_guess[2]}%, Type Prob: {next_guess[3]}%"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
