from flask import Flask, render_template, request
from predict import predict_news

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("news")
        prediction = predict_news(input_text)
    return render_template("index.html", prediction=prediction, news=input_text)

if __name__ == "__main__":
    app.run(debug=True)
