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
import tensorflow as tf  # Using TensorFlow for TFLite
from tensorflow.keras.preprocessing import image
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import google.generativeai as genai

# --- 1. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# --- 2. CONFIGURATION ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Database File Name
DB_NAME = "agro.db"

# Image Upload Config
IMG_SIZE = (128, 128)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 3. DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        tables = [
            '''CREATE TABLE IF NOT EXISTS farmers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT)''',
            '''CREATE TABLE IF NOT EXISTS crops (id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, crop_name TEXT, quantity REAL, expected_price REAL, season TEXT)''',
            '''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT, soil_type TEXT, crop TEXT, rainfall_mm REAL, temperature_celsius REAL, fertilizer_used INTEGER, irrigation_used INTEGER, weather_condition TEXT, days_to_harvest INTEGER, predicted_yield REAL)''',
            '''CREATE TABLE IF NOT EXISTS investors (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT, preferred_crop TEXT, investment_amount REAL)''',
            '''CREATE TABLE IF NOT EXISTS funding (id INTEGER PRIMARY KEY AUTOINCREMENT, investor_id INTEGER, crop_id INTEGER, funded_amount REAL)'''
        ]
        for table in tables:
            cursor.execute(table)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Database Setup Error: {e}")

init_db()

# --- 4. CHATBOT SETUP ---
chat_model = None
def configure_chatbot():
    global chat_model
    try:
        # Using the key from your snippet
        api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCCT-8Txzfpk4M0wilgT1sHx8KZh1CLKDc") 
        genai.configure(api_key=api_key)
        chat_model = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Chatbot Model Loaded")
    except Exception as e:
        print(f"❌ Chatbot Setup Error: {e}")

configure_chatbot()

# --- 5. FUZZY LOGIC SETUP ---
confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 50])
confidence['medium'] = fuzz.trimf(confidence.universe, [30, 50, 70])
confidence['high'] = fuzz.trimf(confidence.universe, [60, 100, 100])
temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['moderate'] = fuzz.trimf(temperature.universe, [15, 25, 35])
temperature['high'] = fuzz.trimf(temperature.universe, [30, 50, 50])
humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 40])
humidity['medium'] = fuzz.trimf(humidity.universe, [30, 50, 70])
humidity['high'] = fuzz.trimf(humidity.universe, [60, 100, 100])
severity['low'] = fuzz.trimf(severity.universe, [0, 0, 50])
severity['medium'] = fuzz.trimf(severity.universe, [30, 50, 70])
severity['high'] = fuzz.trimf(severity.universe, [60, 100, 100])

rule1 = ctrl.Rule(confidence['high'] & humidity['high'], severity['high'])
rule2 = ctrl.Rule(confidence['medium'] & temperature['moderate'], severity['medium'])
rule3 = ctrl.Rule(confidence['low'], severity['low'])
severity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
severity_system = ctrl.ControlSystemSimulation(severity_ctrl)

# --- 6. LOAD MODELS ---

# A. Crop Yield Models
try:
    model = pickle.load(open("crop_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except:
    model = None
    label_encoders = {}

# B. Fertilizer Data
try:
    fertilizer_df = pd.read_csv("updated_crop_disease_fertilizer.csv")
    fertilizer_df.columns = fertilizer_df.columns.str.strip().str.lower()
    fertilizer_df['crop_name'] = fertilizer_df['crop_name'].str.strip().str.lower()
    fertilizer_df['disease_name'] = fertilizer_df['disease_name'].str.strip().str.lower()
except:
    fertilizer_df = pd.DataFrame()

# C. TFLite Disease Model
interpreter = None
input_details = None
output_details = None
class_names = []

try:
    if os.path.exists("plant_disease.tflite"):
        interpreter = tf.lite.Interpreter(model_path="plant_disease.tflite")
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Load class names if folder exists
        if os.path.exists("PlantVillage/train"):
            class_names = sorted(os.listdir("PlantVillage/train"))
        else:
            # Fallback list 
            class_names = ["Disease_1", "Disease_2", "Healthy"] 
            
        print("✅ TFLite Model Loaded Successfully")
    else:
        print("❌ TFLite model file not found in directory.")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# --- 7. ROUTES ---

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        if not user_message: return jsonify({"reply": "Please type a message."})
        if chat_model:
            response = chat_model.generate_content(f"You are AgroBot. Answer concisely: {user_message}")
            return jsonify({"reply": response.text})
        return jsonify({"reply": "I'm thinking..."})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None: return jsonify({"error": "Model not loaded"}), 500
        data = request.get_json()
        raw_data = data.copy()
        
        for key, value in data.items():
            if isinstance(value, str):
                val = value.strip().lower()
                if val == "yes": data[key] = 1
                elif val == "no": data[key] = 0
                else: data[key] = val.title()

        features = [data[col] if col not in label_encoders else label_encoders[col].transform([data[col]])[0] for col in model.feature_names_in_]
        prediction = round(float(model.predict([features])[0]), 2)

        # Save to SQLite
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (region, soil_type, crop, rainfall_mm, temperature_celsius, fertilizer_used, irrigation_used, weather_condition, days_to_harvest, predicted_yield) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                       (raw_data.get('Region'), raw_data.get('Soil_Type'), raw_data.get('Crop'), raw_data.get('Rainfall_mm'), raw_data.get('Temperature_Celsius'), 1 if raw_data.get('Fertilizer_Used')=='Yes' else 0, 1 if raw_data.get('Irrigation_Used')=='Yes' else 0, raw_data.get('Weather_Condition'), raw_data.get('Days_to_Harvest'), prediction))
        conn.commit()
        conn.close()
        return jsonify({"predicted_crop_yield": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-fertilizer", methods=["POST"])
def recommend_fertilizer():
    try:
        if fertilizer_df.empty: return jsonify({"error": "Dataset not loaded"}), 500
        data = request.get_json()
        crop = data.get('crop', '').strip().lower()
        disease = data.get('disease', '').strip().lower()
        result = fertilizer_df[(fertilizer_df['crop_name'] == crop) & (fertilizer_df['disease_name'] == disease)]
        if not result.empty:
            return jsonify({"fertilizer": result['fertilizer_name'].values[0], "quantity": result['quantity_to_use'].values[0]})
        return jsonify({"error": "No recommendation found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        if 'file' not in request.files: return jsonify({"error": "No file"}), 400
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        if interpreter is None: 
            return jsonify({"error": "Disease model (TFLite) not loaded."}), 500

        # Preprocess Image
        img = Image.open(file_path).resize(IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32) # Cast to float32 for TFLite

        # Run Inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        confidence_val = float(np.max(preds) * 100)
        # Safe class name retrieval
        if class_names and len(class_names) > np.argmax(preds):
            predicted_class = class_names[np.argmax(preds)]
        else:
            predicted_class = f"Class {np.argmax(preds)}"

        # Fuzzy Logic
        severity_system.input['confidence'] = confidence_val
        severity_system.input['temperature'] = 28
        severity_system.input['humidity'] = 65
        severity_system.compute()
        severity_val = round(severity_system.output['severity'], 2)

        return jsonify({"prediction": predicted_class, "confidence": round(confidence_val, 2), "severity": severity_val})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- 8. SMART FRONTEND ROUTING ---
@app.route('/')
def serve_index():
    # Priority 1: Check 'index.html' in main folder (current directory)
    if os.path.exists('index.html'):
        return send_from_directory('.', 'index.html')
    # Priority 2: Check 'index.html' inside 'temp' folder
    elif os.path.exists('temp/index.html'):
        return send_from_directory('temp', 'index.html')
    # Priority 3: Check old name 'mod3.html'
    elif os.path.exists('mod3.html'):
        return send_from_directory('.', 'mod3.html')
    
    return "Error: Could not find index.html. Please ensure it is in the main folder or 'temp' folder."

@app.route('/<path:filename>')
def serve_static(filename):
    # Try serving from main folder first
    if os.path.exists(filename):
        return send_from_directory('.', filename)
    # If not found, try serving from 'temp' folder
    return send_from_directory('temp', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)