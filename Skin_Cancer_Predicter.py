from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import base64
import io
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load pre-trained Xception model
model = load_model("skin_model_2_sig.h5")

# Define classes
classes = {1: "benign", 0: "malignant"}


@app.route("/")
def upload_file():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Parse image content and make predictions
        content = request.files["file"].read()
        image = Image.open(io.BytesIO(content))
        image = image.resize((224, 224))  # Resize image to match model input size
        image = np.expand_dims(np.array(image), axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(image)[0]
        predicted_class = np.argmax(prediction)
        confidence = round(prediction[predicted_class] * 100, 2)
        result = f"Prediction: {classes[predicted_class]} ({confidence}% confidence)"
        return render_template(
            "predict.html",
            image=base64.b64encode(content).decode(),
            result=result,
        )


if __name__ == "__main__":
    app.run(debug=True)
