import traceback
import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import warnings

# --- 1. SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

# --- 2. IMPORT LIBRARIES ---
import google.generativeai as genai 
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# --- 3. CONFIGURATION ---
app = Flask(__name__)
# Enable CORS for all routes so your frontend can talk to the backend
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Database File Name
DB_NAME = "agro.db"

# Image Upload Config
IMG_SIZE = (128, 128)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- 4. DATABASE SETUP ---
def init_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        # Create Tables
        cursor.execute('''CREATE TABLE IF NOT EXISTS farmers (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, contact TEXT, location TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS crops (
            id INTEGER PRIMARY KEY AUTOINCREMENT, farmer_id INTEGER, crop_name TEXT, 
            quantity REAL, expected_price REAL, season TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, region TEXT, soil_type TEXT, crop TEXT, 
            rainfall_mm REAL, temperature_celsius REAL, fertilizer_used INTEGER, 
            irrigation_used INTEGER, weather_condition TEXT, days_to_harvest INTEGER, 
            predicted_yield REAL)''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database {DB_NAME} initialized.")
    except Exception as e:
        print(f"❌ Database Setup Error: {e}")

init_db()

# --- 5. CHATBOT SETUP ---
chat_model = None

def configure_chatbot():
    global chat_model
    try:
        # REPLACE WITH YOUR API KEY
        api_key = "AIzaSyCpKQcHrYUzHODj2Zb71sp7noarvN7SuXA"
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        
        # Try finding a valid model (Flash or Pro)
        model_name = "gemini-1.5-flash" 
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    if 'flash' in m.name:
                        model_name = m.name
                        break
        except:
            pass

        chat_model = genai.GenerativeModel(model_name)
        print(f"✅ Chatbot Model Loaded: {model_name}")
        
    except Exception as e:
        print(f"❌ Chatbot Setup Error: {e}")
        try:
            chat_model = genai.GenerativeModel('gemini-pro')
        except:
            pass

configure_chatbot()

# --- 6. MODEL LOADING ---
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

try:
    disease_model = load_model("plant_disease_finetuned.keras") 
    if os.path.exists("PlantVillage/train"):
        class_names = sorted(os.listdir("PlantVillage/train"))
    else:
        class_names = ["Disease_1", "Disease_2", "Healthy"] 
except:
    disease_model = None

# --- 7. ROUTES ---

# THIS IS THE MISSING ROUTE CAUSING YOUR 404 ERROR
@app.route('/chat', methods=['POST'])
def chat():
    try:
        print("📩 Received Chat Request") # Debug print
        data = request.json
        user_message = data.get('message')
        if not user_message:
            return jsonify({"reply": "Please type a message."})

        full_prompt = (
            "You are AgroBot, an expert agriculture assistant. "
            "Keep answers concise and helpful.\n\n"
            f"User: {user_message}\nAgroBot:"
        )
        
        if chat_model:
            response = chat_model.generate_content(full_prompt)
            if response.text:
                return jsonify({"reply": response.text})
        
        return jsonify({"reply": "I'm thinking, but I couldn't come up with an answer."})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"reply": f"System Error: {str(e)}"})

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

        for col, le in label_encoders.items():
            if col in data:
                try: data[col] = le.transform([data[col]])[0]
                except: return jsonify({"error": f"Invalid value for '{col}'"}), 400

        features = [data[col] for col in model.feature_names_in_]
        prediction = round(float(model.predict([features])[0]), 2)

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (region, soil_type, crop, rainfall_mm, temperature_celsius,
            fertilizer_used, irrigation_used, weather_condition, days_to_harvest, predicted_yield) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (raw_data.get('Region'), raw_data.get('Soil_Type'), raw_data.get('Crop'), 
              raw_data.get('Rainfall_mm'), raw_data.get('Temperature_Celsius'),
              1 if raw_data.get('Fertilizer_Used')=='Yes' else 0,
              1 if raw_data.get('Irrigation_Used')=='Yes' else 0,
              raw_data.get('Weather_Condition'), raw_data.get('Days_to_Harvest'), prediction))
        conn.commit()
        conn.close()

        return jsonify({"predicted_crop_yield": prediction})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/recommend-fertilizer", methods=["POST"])
def recommend_fertilizer():
    try:
        if fertilizer_df.empty: return jsonify({"error": "Dataset not loaded"}), 500
        data = request.get_json()
        crop = data.get('crop', '').strip().lower()
        disease = data.get('disease', '').strip().lower()

        result = fertilizer_df[
            (fertilizer_df['crop_name'] == crop) & 
            (fertilizer_df['disease_name'] == disease)
        ]

        if not result.empty:
            return jsonify({
                "fertilizer": result['fertilizer_name'].values[0],
                "quantity": result['quantity_to_use'].values[0]
            })
        else:
            return jsonify({"error": "No recommendation found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/register-farmer", methods=["POST"])
def register_farmer():
    try:
        data = request.get_json()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO farmers (name, contact, location) VALUES (?, ?, ?)", 
                       (data.get("name"), data.get("contact"), data.get("location")))
        conn.commit()
        fid = cursor.lastrowid
        conn.close()
        return jsonify({"message": "Registered", "farmer_id": fid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/submit-crop", methods=["POST"])
def submit_crop():
    try:
        data = request.get_json()
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO crops (farmer_id, crop_name, quantity, expected_price, season)
            VALUES (?, ?, ?, ?, ?)
        """, (data.get("farmer_id"), data.get("crop_name"), data.get("quantity"), 
              data.get("expected_price"), data.get("season")))
        conn.commit()
        conn.close()
        return jsonify({"message": "Crop submitted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/match-crops", methods=["POST"])
def match_crops():
    try:
        data = request.get_json()
        conn = sqlite3.connect(DB_NAME)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT crops.id AS crop_id, crops.crop_name, crops.quantity, crops.expected_price,
                   farmers.name AS farmer_name, farmers.contact, farmers.location
            FROM crops
            JOIN farmers ON crops.farmer_id = farmers.id
            WHERE crops.crop_name = ? AND farmers.location = ?
        """, (data.get("preferred_crop"), data.get("location")))
        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
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

        if disease_model is None: return jsonify({"error": "Disease model not loaded."}), 500

        img = Image.open(file_path).resize(IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = disease_model.predict(img_array)[0]
        confidence_val = float(np.max(preds) * 100)
        predicted_class = class_names[np.argmax(preds)] if class_names else "Unknown"

        return jsonify({"prediction": predicted_class, "confidence": round(confidence_val, 2)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)