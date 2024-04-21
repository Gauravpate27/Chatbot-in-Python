from flask import Flask, render_template, request, jsonify
from chat import get_response

app = Flask(__name__)

# Route for rendering the index page
@app.get("/")
def index_get():
    return render_template("base.html")

# Route for receiving POST requests with user messages and returning responses
@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # Generate a response based on the user message
    response = get_response(text)
    # Prepare the response in JSON format
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
