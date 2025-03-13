import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ปิดการใช้งาน GPU เพื่อลดการใช้ทรัพยากร
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# --------------------------- LOAD MODELS ONCE ---------------------------
print("[INFO] Loading models...")
try:
    with open("models/Linear_Regression.pkl", "rb") as f:
        linear_model = pickle.load(f)
    
    with open("models/Gradient_Boosting_Regressor.pkl", "rb") as f:
        boosting_model = pickle.load(f)
    
    cloud_model = load_model("models/cloud_model.keras")
    
    print("[INFO] Models loaded successfully!")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")
    linear_model, boosting_model, cloud_model = None, None, None

# --------------------------- ROUTES FOR HTML ---------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/neural_network")
def neural_network():
    return render_template("neural_network.html")

@app.route("/predict_house_form")
def predict_house_form():
    return render_template("predict_house.html")

@app.route("/predict_cloud_form")
def predict_cloud_form():
    return render_template("predict_cloud.html")

# --------------------------- ENDPOINT: LINEAR REGRESSION ---------------------------
@app.route("/predict_house_lr", methods=["POST"])
def predict_house_lr():
    if linear_model is None:
        return jsonify({"error": "Model not loaded"})
    
    try:
        features = [float(request.form[key]) for key in request.form]
        X = np.array(features).reshape(1, -1)
        pred_lr = max(0, linear_model.predict(X)[0])
        return jsonify({"lr_prediction": float(pred_lr)})
    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------- ENDPOINT: GRADIENT BOOSTING ---------------------------
@app.route("/predict_house_gb", methods=["POST"])
def predict_house_gb():
    if boosting_model is None:
        return jsonify({"error": "Model not loaded"})
    
    try:
        features = [float(request.form[key]) for key in request.form]
        X = np.array(features).reshape(1, -1)
        pred_gb = max(0, boosting_model.predict(X)[0])
        return jsonify({"gb_prediction": float(pred_gb)})
    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------- ENDPOINT: CLOUD DETECTION ---------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict_cloud", methods=["POST"])
def predict_cloud():
    if cloud_model is None:
        return jsonify({"error": "Model not loaded"})
    
    try:
        file = request.files.get("image")
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file. Please upload a .jpg or .png image."})
        
        file_path = os.path.join("static/uploads", file.filename)
        file.save(file_path)

        img = load_img(file_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cloud_model.predict(img_array)[0][0]
        label = "Cloud" if prediction <= 0.5 else "Not Cloud"
        
        return jsonify({"prediction": label, "image_url": file_path})
    except Exception as e:
        return jsonify({"error": str(e)})

# --------------------------- MAIN ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
