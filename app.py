
import traceback
from flask import Flask, request, jsonify,send_from_directory
import numpy as np
import pickle
import mysql.connector
import pandas as pd
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image 

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

confidence = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
temperature = ctrl.Antecedent(np.arange(0, 51, 1), 'temperature')
humidity = ctrl.Antecedent(np.arange(0, 101, 1), 'humidity')
severity = ctrl.Consequent(np.arange(0, 101, 1), 'severity')

# Membership functions
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


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Vamsi@123',
    'database': 'my_flask_db'
}

# Load model and label encoders
try:
    model = pickle.load(open("crop_model.pkl", "rb"))
    label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    model = None
    label_encoders = {}

# Load fertilizer recommendation dataset
try:
    fertilizer_df = pd.read_csv(r"C:\summer project\updated_crop_disease_fertilizer.csv")
    fertilizer_df.columns = fertilizer_df.columns.str.strip().str.lower()
    fertilizer_df['crop_name'] = fertilizer_df['crop_name'].str.strip().str.lower()
    fertilizer_df['disease_name'] = fertilizer_df['disease_name'].str.strip().str.lower()
except Exception as e:
    print(f"Error loading fertilizer dataset: {e}")
    fertilizer_df = pd.DataFrame()
#image upload config
IMG_SIZE = (128, 128)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    disease_model = load_model("plant_disease_finetuned.keras")
    class_names = sorted(os.listdir(r"C:\summer project\PlantVillage\train"))
  
 # Replace with actual path
except Exception as e:
    print(f"Error loading disease model: {e}")
    disease_model = None
    class_names = []

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided."}), 400

        raw_data = data.copy()

        # Normalize yes/no to 1/0, title-case other strings
        for key, value in data.items():
            if isinstance(value, str):
                val = value.strip().lower()
                if val == "yes":
                    data[key] = 1
                elif val == "no":
                    data[key] = 0
                else:
                    data[key] = val.title()

        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in data:
                try:
                    data[col] = le.transform([data[col]])[0]
                except Exception:
                    return jsonify({"error": f"Invalid value for '{col}': {data[col]}"}), 400

        # Check for missing fields
        missing_fields = [f for f in model.feature_names_in_ if f not in data]
        if missing_fields:
            return jsonify({"error": f"Missing input fields: {missing_fields}"}), 400

        features = [data[col] for col in model.feature_names_in_]
        features_array = np.array([features], dtype=np.float32)

        prediction = float(model.predict(features_array)[0])
        prediction_rounded = round(prediction, 2)

        def yes_no_to_int(key):
            val = (raw_data.get(key) or raw_data.get(key.lower()) or "").strip().lower()
            return 1 if val == "yes" else 0

        fertilizer_used = yes_no_to_int('Fertilizer_Used')
        irrigation_used = yes_no_to_int('Irrigation_Used')

        # Insert into database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO predictions (
                region, soil_type, crop, rainfall_mm, temperature_celsius,
                fertilizer_used, irrigation_used, weather_condition, days_to_harvest,
                predicted_yield
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(insert_query, (
            raw_data.get('Region') or raw_data.get('region'),
            raw_data.get('Soil_Type') or raw_data.get('soil_type'),
            raw_data.get('Crop') or raw_data.get('crop'),
            raw_data.get('Rainfall_mm') or raw_data.get('rainfall_mm'),
            raw_data.get('Temperature_Celsius') or raw_data.get('temperature_celsius'),
            fertilizer_used,
            irrigation_used,
            raw_data.get('Weather_Condition') or raw_data.get('weather_condition'),
            raw_data.get('Days_to_Harvest') or raw_data.get('days_to_harvest'),
            prediction_rounded
        ))

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"predicted_crop_yield": prediction_rounded})

    except mysql.connector.Error as err:
        traceback.print_exc()
        return jsonify({"error": f"MySQL error: {str(err)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500




@app.route("/recommend-fertilizer", methods=["POST"])
def recommend_fertilizer():
    try:
        if fertilizer_df.empty:
            return jsonify({"error": "Fertilizer dataset not loaded"}), 500

        data = request.get_json()
        crop = data.get('crop')
        disease = data.get('disease')

        if not crop or not disease:
            return jsonify({"error": "Both 'crop' and 'disease' must be provided."}), 400

        crop = crop.strip().lower()
        disease = disease.strip().lower()

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
            return jsonify({"error": f"No recommendation found for crop '{crop}' and disease '{disease}'."}), 404

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
# ------------------ Farmer Registration ------------------

@app.route("/register-farmer", methods=["POST"])
def register_farmer():
    try:
        data = request.get_json()
        name = data.get("name")
        contact = data.get("contact")
        location = data.get("location")

        if not all([name, contact, location]):
            return jsonify({"error": "Missing fields"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO farmers (name, contact, location) 
            VALUES (%s, %s, %s)
        """, (name, contact, location))
        conn.commit()

        farmer_id = cursor.lastrowid
        cursor.close()
        conn.close()

        return jsonify({"message": "Farmer registered", "farmer_id": farmer_id})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/submit-crop", methods=["POST"])
def submit_crop():
    try:
        data = request.get_json()
        farmer_id = data.get("farmer_id")
        crop_name = data.get("crop_name")
        quantity = data.get("quantity")
        expected_price = data.get("expected_price")
        season = data.get("season")

        if not all([farmer_id, crop_name, quantity, expected_price, season]):
            return jsonify({"error": "Missing fields"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO crops (farmer_id, crop_name, quantity, expected_price, season)
            VALUES (%s, %s, %s, %s, %s)
        """, (farmer_id, crop_name, quantity, expected_price, season))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Crop details submitted successfully"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
#investor module
@app.route("/investor-register", methods=["POST"])
def investor_register():
    try:
        data = request.get_json()
        name = data.get("name")
        contact = data.get("contact")
        location = data.get("location")
        preferred_crop = data.get("preferred_crop")
        investment_amount = data.get("investment_amount")

        if not all([name, contact, location, preferred_crop, investment_amount]):
            return jsonify({"error": "Missing fields"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO investors (name, contact, location, preferred_crop, investment_amount)
            VALUES (%s, %s, %s, %s, %s)
        """, (name, contact, location, preferred_crop, investment_amount))
        conn.commit()
        investor_id = cursor.lastrowid

        cursor.close()
        conn.close()
        return jsonify({"message": "Investor registered", "investor_id": investor_id})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/match-crops", methods=["POST"])
def match_crops():
    try:
        data = request.get_json()
        preferred_crop = data.get("preferred_crop")
        location = data.get("location")

        if not all([preferred_crop, location]):
            return jsonify({"error": "Missing criteria"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT crops.id AS crop_id, crops.crop_name, crops.quantity, crops.expected_price,
                   farmers.name AS farmer_name, farmers.contact, farmers.location
            FROM crops
            JOIN farmers ON crops.farmer_id = farmers.id
            WHERE crops.crop_name = %s AND farmers.location = %s
        """, (preferred_crop, location))
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
@app.route("/fund-crop", methods=["POST"])
def fund_crop():
    try:
        data = request.get_json()
        investor_id = data.get("investor_id")
        crop_id = data.get("crop_id")
        funded_amount = data.get("funded_amount")

        if not all([investor_id, crop_id, funded_amount]):
            return jsonify({"error": "Missing fields"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO funding (investor_id, crop_id, funded_amount)
            VALUES (%s, %s, %s)
        """, (investor_id, crop_id, funded_amount))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Funding recorded successfully"})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500
        
 
   # ------------------ Plant Disease Detection ------------------
 # Add this at the top if not already present

@app.route("/predict-disease", methods=["POST"])
def predict_disease():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess image
        img = Image.open(file_path).resize(IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict disease
        preds = disease_model.predict(img_array)[0]
        confidence_val = np.max(preds) * 100
        predicted_class = class_names[np.argmax(preds)] if class_names else "Unknown"

        if confidence_val < 45:
            return jsonify({
                "warning": "Low confidence. Upload a clearer leaf image.",
                "confidence": round(confidence_val, 2)
            })

        # ===== Fuzzy logic integration =====
        temp = 28   # Replace with actual temperature input
        hum = 65    # Replace with actual humidity input

        severity_system.input['confidence'] = confidence_val
        severity_system.input['temperature'] = temp
        severity_system.input['humidity'] = hum
        severity_system.compute()

        severity_val = round(severity_system.output['severity'], 2)

        # Print values for debugging/logging
        print(f" Prediction: {predicted_class}")
        print(f" Confidence: {round(confidence_val, 2)}%")
        print(f" Severity (Fuzzy Logic): {severity_val}%")

        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence_val, 2),
            "severity": severity_val
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500



# Serve HTML frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'mod3.html')

if __name__ == "__main__":
    app.run(debug=True)
