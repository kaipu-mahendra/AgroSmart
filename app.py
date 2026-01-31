import traceback
import os
import pickle
import sqlite3
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import google.generativeai as genai

# --- 1. CONFIGURATION ---
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# ABSOLUTE PATHS (Fixes frontend not found)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DB_PATH = os.path.join(BASE_DIR, 'agro.db')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS farmers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS crops (id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, crop_name TEXT, quantity REAL, expected_price REAL, season TEXT)''')
        conn.commit()
        conn.close()
        print("✅ Database Initialized")
    except Exception as e:
        print(f"❌ Database Setup Error: {e}")

init_db()

# --- 3. CHATBOT SETUP (Smart Auto-Detect) ---
chat_model = None
def configure_chatbot():
    global chat_model
    try:
        api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCCT-8Txzfpk4M0wilgT1sHx8KZh1CLKDc") 
        genai.configure(api_key=api_key)
        
        # Try Flash first, then Pro
        models_to_try = ["gemini-1.5-flash", "gemini-pro"]
        
        for model_name in models_to_try:
            try:
                print(f"🔄 Connecting to {model_name}...")
                model = genai.GenerativeModel(model_name)
                model.generate_content("Hi")
                chat_model = model
                print(f"✅ Chatbot ready: {model_name}")
                return
            except:
                continue
        print("❌ Chatbot failed to load models.")

    except Exception as e:
        print(f"❌ Chatbot Setup Error: {e}")

configure_chatbot()

# --- 4. LOAD MODELS ---
try:
    # Load using absolute paths
    model = pickle.load(open(os.path.join(BASE_DIR, "crop_model.pkl"), "rb"))
    label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb"))
except:
    model = None
    label_encoders = {}

# TFLITE SETUP
interpreter = None
input_details = None
output_details = None
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

try:
    tflite_path = os.path.join(BASE_DIR, "plant_disease.tflite")
    if os.path.exists(tflite_path):
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("✅ TFLite Model Loaded")
except Exception as e:
    print(f"❌ Error loading TFLite: {e}")

# --- 5. ROUTES ---

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        if chat_model:
            response = chat_model.generate_content(f"You are AgroBot. Answer concisely: {user_message}")
            return jsonify({"reply": response.text})
        return jsonify({"reply": "Chatbot is temporarily unavailable."})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

# --- YIELD PREDICTION (FIXED TYPE ERROR) ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None: return jsonify({"error": "Model not loaded"}), 500
        data = request.get_json()
        
        # Pre-process inputs (Fixing 'Yes/No' and casing)
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                val = value.strip().lower()
                if val == "yes": processed_data[key] = 1
                elif val == "no": processed_data[key] = 0
                else: processed_data[key] = value.title() # Keep original casing for categorical
            else:
                processed_data[key] = value

        features = []
        for col in model.feature_names_in_:
            val = processed_data.get(col)
            
            # If it's a categorical column that needs encoding
            if col in label_encoders:
                # If value is somehow missing, use a placeholder
                if val is None: val = "Unknown"
                val = str(val) # Ensure string for encoder
                encoded_val = label_encoders[col].transform([val])[0]
                features.append(encoded_val)
            else:
                # It is a number (Rainfall, Temp, etc.) -> Convert to Float
                try:
                    features.append(float(val))
                except:
                    features.append(0.0) # Safe fallback

        # Convert to numpy array to fix "Unicode-2" error
        final_features = np.array([features])
        prediction = round(float(model.predict(final_features)[0]), 2)
        
        return jsonify({"predicted_crop_yield": prediction})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- OTHER ROUTES ---
@app.route("/register-farmer", methods=["POST"])
@app.route("/register_farmer", methods=["POST"])
def register_farmer():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO farmers (name, contact, location) VALUES (?, ?, ?)", 
                       (data.get('name'), data.get('contact'), data.get('location')))
        farmer_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"message": "Registered!", "farmer_id": farmer_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/submit-crop", methods=["POST"])
def submit_crop():
    try:
        data = request.json
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO crops (farmer_id, crop_name, quantity, expected_price, season) VALUES (?, ?, ?, ?, ?)",
                       (data.get('farmer_id'), data.get('crop_name'), data.get('quantity'), data.get('expected_price'), data.get('season')))
        conn.commit()
        conn.close()
        return jsonify({"message": "Crop submitted!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/find_matches", methods=["POST"])
@app.route("/match_crops", methods=["POST"])
@app.route("/match-crops", methods=["POST"])
def find_matches():
    try:
        data = request.json
        crop_name = (data.get('preferred_crop') or data.get('crop') or data.get('crop_name') or '').strip().lower()
        location_filter = (data.get('location') or data.get('city') or '').strip().lower()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = '''SELECT f.name, f.contact, f.location, c.crop_name, c.quantity, c.expected_price, c.id 
                   FROM farmers f JOIN crops c ON f.id = c.farmer_id WHERE LOWER(c.crop_name) LIKE ?'''
        params = [f'%{crop_name}%']
        if location_filter:
            query += " AND LOWER(f.location) LIKE ?"
            params.append(f'%{location_filter}%')
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        results = [{"farmer_name": r[0], "contact": r[1], "location": r[2], "crop_name": r[3], "quantity": r[4], "expected_price": r[5], "crop_id": r[6]} for r in rows]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if interpreter is None: return jsonify({"error": "Disease model not loaded."}), 500
        img = Image.open(file_path).resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]
        
        predicted_class = class_names[np.argmax(preds)] if len(class_names) > np.argmax(preds) else f"Class {np.argmax(preds)}"
        severity = round(float(np.max(preds) * 100), 2)
        return jsonify({"prediction": predicted_class, "confidence": severity, "severity": severity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- FRONTEND SERVING ---
@app.route('/')
def serve_index():
    public_path = os.path.join(BASE_DIR, 'public')
    temp_path = os.path.join(BASE_DIR, 'temp')
    if os.path.exists(os.path.join(public_path, 'index.html')): return send_from_directory(public_path, 'index.html')
    if os.path.exists(os.path.join(temp_path, 'index.html')): return send_from_directory(temp_path, 'index.html')
    if os.path.exists(os.path.join(BASE_DIR, 'index.html')): return send_from_directory(BASE_DIR, 'index.html')
    return "Error: index.html not found"

@app.route('/<path:filename>')
def serve_static(filename):
    public_path = os.path.join(BASE_DIR, 'public')
    temp_path = os.path.join(BASE_DIR, 'temp')
    if os.path.exists(os.path.join(public_path, filename)): return send_from_directory(public_path, filename)
    if os.path.exists(os.path.join(temp_path, filename)): return send_from_directory(temp_path, filename)
    return send_from_directory(BASE_DIR, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)