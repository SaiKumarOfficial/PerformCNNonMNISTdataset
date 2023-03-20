from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

model = tf.keras.models.load_model("mnistmodel.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    imgDataArray = request.get_json()
    imgDataArray = np.array(imgDataArray)
    # Load the input array from the numpy array
    input_array = imgDataArray.astype('float32')
    # Resize the array to 28x28 pixels using OpenCV
    input_array = cv2.resize(input_array, (28, 28))
    # Normalize the array
    input_array = input_array / 255.0
    # Reshape the array
    input_array = input_array.reshape((1, 28, 28, 1))
    # Make prediction
    prediction = model.predict(input_array)
    predictedDigit = np.argmax(prediction)
    return jsonify({"prediction": predictedDigit})

if __name__ == "__main__":
    app.run(debug=True)
