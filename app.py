import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


app = Flask(__name__)

# --------------------------- LOAD MODELS ---------------------------
# Linear Regression
with open("models/Linear_Regression.pkl", "rb") as f:
    linear_model = pickle.load(f)

# Gradient Boosting Regressor
with open("models/Gradient_Boosting_Regressor.pkl", "rb") as f:
    boosting_model = pickle.load(f)

# Cloud Detection (Neural Network)
cloud_model = load_model("models/cloud_model.keras")

# --------------------------- ROUTES FOR HTML ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/neural_network")
def neural_network():
    return render_template("neural_network.html")

@app.route("/predict_house_form")
def predict_house_form():
    # หน้าเว็บ มี 2 ฟอร์ม แยก LR กับ GB
    return render_template("predict_house.html")

@app.route("/predict_cloud_form")
def predict_cloud_form():
    return render_template("predict_cloud.html")

# --------------------------- ENDPOINT: LINEAR REGRESSION ---------------------------
@app.route("/predict_house_lr", methods=["POST"])
def predict_house_lr():
    """
    รับ 12 ฟีเจอร์ -> ทำนายด้วย Linear Regression
    ถ้าติดลบ => 0
    """
    try:
        # ดึง 12 ฟีเจอร์
        features = [
            float(request.form["price"]),
            float(request.form["quantity"]),
            float(request.form["customer_rate"]),
            float(request.form["longitude"]),
            float(request.form["latitude"]),
            float(request.form["housing_median_age"]),
            float(request.form["total_rooms"]),
            float(request.form["total_bedrooms"]),
            float(request.form["population"]),
            float(request.form["households"]),
            float(request.form["median_income"]),
            float(request.form["ocean_proximity_INLAND"])
        ]
        X = np.array(features).reshape(1, -1)

        # Predict
        pred_lr = linear_model.predict(X)[0]
        # Clamp negative => 0
        if pred_lr < 0:
            pred_lr = 0

        return jsonify({
            "lr_prediction": float(pred_lr)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------- ENDPOINT: GRADIENT BOOSTING ---------------------------
@app.route("/predict_house_gb", methods=["POST"])
def predict_house_gb():
    """
    รับ 12 ฟีเจอร์ -> ทำนายด้วย Gradient Boosting
    ถ้าติดลบ => 0
    """
    try:
        features = [
            float(request.form["price"]),
            float(request.form["quantity"]),
            float(request.form["customer_rate"]),
            float(request.form["longitude"]),
            float(request.form["latitude"]),
            float(request.form["housing_median_age"]),
            float(request.form["total_rooms"]),
            float(request.form["total_bedrooms"]),
            float(request.form["population"]),
            float(request.form["households"]),
            float(request.form["median_income"]),
            float(request.form["ocean_proximity_INLAND"])
        ]
        X = np.array(features).reshape(1, -1)

        # Predict
        pred_gb = boosting_model.predict(X)[0]
        # Clamp negative => 0
        if pred_gb < 0:
            pred_gb = 0

        return jsonify({
            "gb_prediction": float(pred_gb)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------- ENDPOINT: CLOUD ---------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict_cloud", methods=["POST"])
def predict_cloud():
    try:
        if "image" not in request.files:
            return jsonify({"error": "ไม่มีไฟล์ภาพที่อัปโหลด"})

        file = request.files["image"]
        
        if not allowed_file(file.filename):
            return jsonify({"error": "ไฟล์ที่อัปโหลดต้องเป็น .jpg หรือ .png เท่านั้น"})

        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        img = load_img(file_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cloud_model.predict(img_array)[0][0]
        label = "Cloud" if prediction <= 0.5 else "Not Cloud"
        
        return jsonify({
            "prediction": label,
            "image_url": file_path
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    app.run(debug=True)

