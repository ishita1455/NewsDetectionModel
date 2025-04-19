from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    output = predict_news(input_text)
    return render_template('index.html',
                           prediction=output['result'],
                           confidence=output['confidence'],
                           articles=output['related_articles'],
                           news=input_text)

if __name__ == '__main__':
    app.run(debug=True)
