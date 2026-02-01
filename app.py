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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
DB_PATH = os.path.join(BASE_DIR, 'agro.db')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- 2. DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS farmers (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS crops (id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, crop_name TEXT, quantity REAL, expected_price REAL, season TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT, soil_type TEXT, crop TEXT, rainfall_mm REAL, temperature_celsius REAL, fertilizer_used INTEGER, irrigation_used INTEGER, weather_condition TEXT, days_to_harvest INTEGER, predicted_yield REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS investors (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT, preferred_crop TEXT, investment_amount REAL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS funding (id INTEGER PRIMARY KEY AUTOINCREMENT, investor_id INTEGER, crop_id INTEGER, funded_amount REAL)''')
        conn.commit()
        conn.close()
        print("✅ Database Initialized")
    except Exception as e:
        print(f"❌ Database Setup Error: {e}")

init_db()

# --- 3. CHATBOT SETUP (FIXED) ---
chat_model = None

def configure_chatbot():
    global chat_model
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not set")
            return

        genai.configure(api_key=api_key)

        models_to_try = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-pro"
        ]

        for model_name in models_to_try:
            try:
                print(f"🔄 Connecting to {model_name}")
                model = genai.GenerativeModel(model_name)
                model.generate_content("Hello")  # test
                chat_model = model
                print(f"✅ Chatbot ready: {model_name}")
                return
            except:
                continue

        print("❌ Gemini models unavailable")

    except Exception as e:
        print(f"❌ Chatbot setup error: {e}")

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
    model = pickle.load(open(os.path.join(BASE_DIR, "crop_model.pkl"), "rb"))
    label_encoders = pickle.load(open(os.path.join(BASE_DIR, "label_encoders.pkl"), "rb"))
except:
    model = None
    label_encoders = {}

try:
    fertilizer_df = pd.read_csv(os.path.join(BASE_DIR, "updated_crop_disease_fertilizer.csv"))
    fertilizer_df.columns = fertilizer_df.columns.str.strip().str.lower()
    fertilizer_df['crop_name'] = fertilizer_df['crop_name'].str.strip().str.lower()
    fertilizer_df['disease_name'] = fertilizer_df['disease_name'].str.strip().str.lower()
except:
    fertilizer_df = pd.DataFrame()

# --- TFLITE MODEL ---
interpreter = None
input_details = None
output_details = None

class_names = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
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
    print(f"❌ TFLite error: {e}")

# --- CHAT ROUTE ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        if not chat_model:
            return jsonify({"reply": "Chatbot unavailable. Server updating..."})

        data = request.get_json()
        user_message = (data.get("message") or "").strip()

        if not user_message:
            return jsonify({"reply": "Please ask a crop-related question 🌱"})

        response = chat_model.generate_content(
            f"You are AgroBot AI for farmers. Answer clearly and briefly.\nUser: {user_message}"
        )
        return jsonify({"reply": response.text})

    except Exception:
        print(traceback.format_exc())
        return jsonify({"reply": "Chatbot error. Please try again."})

# --- FRONTEND ---
@app.route('/')
def serve_index():
    for folder in ['public', 'temp', BASE_DIR]:
        path = os.path.join(BASE_DIR, folder, 'index.html')
        if os.path.exists(path):
            return send_from_directory(os.path.dirname(path), 'index.html')
    return "Frontend not found"

@app.route('/<path:filename>')
def serve_static(filename):
    for folder in ['public', 'temp', BASE_DIR]:
        path = os.path.join(BASE_DIR, folder, filename)
        if os.path.exists(path):
            return send_from_directory(os.path.dirname(path), filename)
    return "File not found", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)