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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Folders
DB_NAME = "agro.db"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 2. DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS farmers 
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS crops 
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, crop_name TEXT, quantity REAL, expected_price REAL, season TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions 
                          (id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT, soil_type TEXT, crop TEXT, rainfall_mm REAL, temperature_celsius REAL, fertilizer_used INTEGER, irrigation_used INTEGER, weather_condition TEXT, days_to_harvest INTEGER, predicted_yield REAL)''')
        
        conn.commit()
        conn.close()
        print("✅ Database Initialized")
    except Exception as e:
        print(f"❌ Database Setup Error: {e}")

init_db()

# --- 3. CHATBOT SETUP ---
chat_model = None
def configure_chatbot():
    global chat_model
    try:
        api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyDWravHPJcdoI8ijiz2L-sArfwdjupFHJg") 
        genai.configure(api_key=api_key)
        chat_model = genai.GenerativeModel("gemini-pro") 
        print("✅ Chatbot Model Loaded (gemini-pro)")
    except Exception as e:
        print(f"❌ Chatbot Setup Error: {e}")

configure_chatbot()

# --- 4. FUZZY LOGIC SETUP ---
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

# --- 5. LOAD MODELS ---
try:
    model = pickle.load(open("crop_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except:
    model = None
    label_encoders = {}

try:
    fertilizer_df = pd.read_csv("updated_crop_disease_fertilizer.csv")
    fertilizer_df.columns = fertilizer_df.columns.str.strip().str.lower()
    fertilizer_df['crop_name'] = fertilizer_df['crop_name'].str.strip().str.lower()
    fertilizer_df['disease_name'] = fertilizer_df['disease_name'].str.strip().str.lower()
except:
    fertilizer_df = pd.DataFrame()

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
        if os.path.exists("PlantVillage/train"):
            class_names = sorted(os.listdir("PlantVillage/train"))
        else:
            class_names = ["Disease_1", "Disease_2", "Healthy"]
        print("✅ TFLite Model Loaded")
    else:
        print("❌ TFLite model file not found.")
except Exception as e:
    print(f"❌ Error loading TFLite model: {e}")

# --- 6. ROUTES ---

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        if not user_message: return jsonify({"reply": "Please type a message."})
        if chat_model:
            response = chat_model.generate_content(f"You are AgroBot. Answer concisely: {user_message}")
            return jsonify({"reply": response.text})
        return jsonify({"reply": "Chatbot is initializing..."})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

# --- 1. FARMER REGISTRATION (Matches your HTML /register-farmer) ---
@app.route("/register-farmer", methods=["POST"])  # Hyphenated route to match HTML
@app.route("/register_farmer", methods=["POST"])
def register_farmer():
    try:
        data = request.json
        print("📥 Registering Farmer:", data)
        
        name = data.get('name')
        contact = data.get('contact')
        location = data.get('location')
        
        if not name or not contact:
             return jsonify({"error": "Name and Contact are required"}), 400

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO farmers (name, contact, location) VALUES (?, ?, ?)", (name, contact, location))
        farmer_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({"message": "Farmer registered successfully!", "farmer_id": farmer_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- 2. CROP SUBMISSION (Matches your HTML /submit-crop) ---
@app.route("/submit-crop", methods=["POST"])
def submit_crop():
    try:
        data = request.json
        print("📥 Submitting Crop:", data)
        
        farmer_id = data.get('farmer_id')
        crop_name = data.get('crop_name')
        quantity = data.get('quantity')
        price = data.get('expected_price')
        season = data.get('season')

        if not farmer_id or not crop_name:
            return jsonify({"error": "Farmer ID and Crop Name are required"}), 400

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO crops (farmer_id, crop_name, quantity, expected_price, season) VALUES (?, ?, ?, ?, ?)",
                       (farmer_id, crop_name, quantity, price, season))
        conn.commit()
        conn.close()

        return jsonify({"message": "Crop submitted successfully!"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# --- 3. CROP MATCHING (Matches Find_Crop.html /find_matches) ---
@app.route("/find_matches", methods=["POST"])
@app.route("/match_crops", methods=["POST"])
def find_matches():
    try:
        data = request.json
        crop_name = (data.get('crop') or data.get('crop_name') or '').strip().lower()
        location_filter = (data.get('location') or data.get('city') or '').strip().lower()

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        query = '''
            SELECT f.name, f.contact, f.location, c.crop_name, c.quantity, c.expected_price 
            FROM farmers f
            JOIN crops c ON f.id = c.farmer_id
            WHERE LOWER(c.crop_name) LIKE ? 
        '''
        params = [f'%{crop_name}%']
        
        if location_filter:
            query += " AND LOWER(f.location) LIKE ?"
            params.append(f'%{location_filter}%')
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "farmer_name": row[0],
                "contact": row[1],
                "location": row[2],
                "crop": row[3],
                "quantity": row[4],
                "price": row[5]
            })
            
        return jsonify({"matches": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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

        img = Image.open(file_path).resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        confidence_val = float(np.max(preds) * 100)
        predicted_class = class_names[np.argmax(preds)] if len(class_names) > np.argmax(preds) else f"Class {np.argmax(preds)}"

        severity_system.input['confidence'] = confidence_val
        severity_system.input['temperature'] = 28
        severity_system.input['humidity'] = 65
        severity_system.compute()
        severity_val = round(severity_system.output['severity'], 2)

        return jsonify({"prediction": predicted_class, "confidence": round(confidence_val, 2), "severity": severity_val})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 7. FILE SERVING ---
@app.route('/')
def serve_index():
    if os.path.exists('public/index.html'): return send_from_directory('public', 'index.html')
    if os.path.exists('temp/index.html'): return send_from_directory('temp', 'index.html')
    if os.path.exists('index.html'): return send_from_directory('.', 'index.html')
    return "Error: Could not find index.html."

@app.route('/<path:filename>')
def serve_static(filename):
    if os.path.exists(os.path.join('public', filename)): return send_from_directory('public', filename)
    if os.path.exists(os.path.join('temp', filename)): return send_from_directory('temp', filename)
    return send_from_directory('.', filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)