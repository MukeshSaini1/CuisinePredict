from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load your trained model and feature DataFrame
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('feature_df.pkl', 'rb') as feature_file:
    feature_df = pickle.load(feature_file)

app = Flask(__name__)

# Extract the ingredients (feature names) from the DataFrame
ingredients = list(feature_df.columns)

def predict_cuisine(user_ingredients):
    # Create an empty DataFrame with the same columns as the feature_df
    user_input_vector = pd.DataFrame(columns=feature_df.columns, index=[0]).fillna(0)

    # Update the input vector with the selected ingredients
    for ingredient in user_ingredients:
        if ingredient in user_input_vector.columns:
            user_input_vector[ingredient] = 1

    # Predict probabilities for the input vector
    proba = model.predict_proba(user_input_vector)
    classes = model.classes_
    result_df = pd.DataFrame(data=proba, columns=classes)

    return result_df

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_ingredients = [
            request.form.get('ingredient1'),
            request.form.get('ingredient2'),
            request.form.get('ingredient3'),
            request.form.get('ingredient4'),
            request.form.get('ingredient5')
        ]

        result_df = predict_cuisine(selected_ingredients)
        result_dict = result_df.T.to_dict()[0]
        return render_template('result.html', ingredients=ingredients, result=result_dict)

    return render_template('index.html', ingredients=ingredients)

@app.route('/document')
def document():
    return render_template('document.html')

@app.route('/about')
def about():
    return render_template('about.html')

