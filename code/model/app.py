import numpy as np
import pickle
from flask import Flask, request, render_template

# Create application
app = Flask(__name__)

# Load machine learning model
model_sf = pickle.load(open('models/sf.pkl', 'rb'))
model_mv = pickle.load(open('models/mv.pkl', 'rb'))
model_sj = pickle.load(open('models/sj.pkl', 'rb'))
model_re = pickle.load(open('models/re.pkl', 'rb'))


# Bind home function to URL
@app.route('/')
def home():
    return render_template('app_new.html')


# Bind predict function to URL
@app.route('/predict', methods=['POST'])
def predict():

    # Put all form entries values in a list
    features = [float(i) for i in request.form.values()]

    # Remove first element from features and store it in a variable
    city = features.pop(0)

    # If length of features is not 30, then make it 30
    if len(features) != 30:
        features = features + [0] * (30 - len(features))

    if city == 0:
        prediction = model_sf.predict([features])
    elif city == 1:
        features.pop(-1)
        prediction = model_sj.predict([features])
    elif city == 2:
        features.pop(-1)
        prediction = model_re.predict([features])
    else:
        features.pop(-1)
        prediction = model_mv.predict([features])

    # Check the output values and retrieve the result with html tag based on the value
    return render_template('app_new.html', result=prediction)


app.run()
