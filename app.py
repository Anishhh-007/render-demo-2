from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('spam_classifier_model_1.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']  # Get the input message from the form
        prediction = model.predict([message])

        if prediction == 1:
            result = "This is a spam email."
        else:
            result = "This is a ham (non-spam) email."

        return render_template('index.html', prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
