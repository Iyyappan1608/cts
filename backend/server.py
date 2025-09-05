import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_bcrypt import Bcrypt 
import os
import json
import mysql.connector
import joblib
from datetime import datetime
from datetime import timedelta
from care_plan_generator import generate_care_plan_from_report as llm_generate_care_plan
from care_plan_generator import _format_output as format_care_text  # reuse formatter
import re

# --- 1. SETUP ---
app = Flask(__name__)
bcrypt = Bcrypt(app)
CORS(app, resources={r"/*": {"origins": "*"}})

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'health_app_db'
}

# --- Initialize Database Tables ---
def init_database():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Create comprehensive predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                prediction_type ENUM('chronic_disease', 'diabetes_subtype', 'hypertension', 'vitals', 'general_health'),
                page_source VARCHAR(100),
                input_data JSON,
                output_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                INDEX idx_patient_id (patient_id),
                INDEX idx_prediction_type (prediction_type),
                INDEX idx_created_at (created_at)
            )
        """)
        
        # Create user sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                session_token VARCHAR(255) UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            )
        """)
        
        # Create user activity log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_activity_log (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                activity_type VARCHAR(100),
                page_visited VARCHAR(100),
                details JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
            )
        """)

        # Create diabetes entries table (stores raw inputs and model result)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diabetes_entries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                input_data JSON,
                output_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                INDEX idx_patient_id (patient_id),
                INDEX idx_created_at (created_at)
            )
        """)

        # Create hypertension entries table (stores raw inputs and model result)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypertension_entries (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT,
                input_data JSON,
                output_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                INDEX idx_patient_id (patient_id),
                INDEX idx_created_at (created_at)
            )
        """)

        # Care plan master table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS care_plans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                patient_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                INDEX idx_patient_id (patient_id)
            )
        """)

        # Per-day care plan items
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS care_plan_days (
                id INT AUTO_INCREMENT PRIMARY KEY,
                care_plan_id INT NOT NULL,
                day_number TINYINT NOT NULL,
                content MEDIUMTEXT NOT NULL,
                unlocked_at DATETIME NOT NULL,
                is_completed TINYINT(1) DEFAULT 0,
                completed_choice ENUM('yes','no') NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (care_plan_id) REFERENCES care_plans(id) ON DELETE CASCADE,
                UNIQUE KEY unique_day (care_plan_id, day_number)
            )
        """)
        
        conn.commit()
        print("Database tables initialized successfully!")
        
    except mysql.connector.Error as err:
        print(f"Database initialization error: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# Initialize database on startup
init_database()

# --- Add HypertensionPipeline class ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

class HypertensionPipeline(BaseEstimator):
    def __init__(self):
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        numerical_features = ['age', 'bmi', 'creatinine', 'systolic_bp', 'diastolic_bp']
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features)
            ])
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        return pipeline
    
    def fit(self, X, y):
        return self.pipeline.fit(X, y)
    
    def predict(self, X):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
    
    def get_params(self, deep=True):
        return self.pipeline.get_params(deep=deep)
    
    def set_params(self, **params):
        return self.pipeline.set_params(**params)

# --- 2. MODEL LOADING ---
models = {}
try:
    # Load Chronic Disease Models
    with open(os.path.join('pkl', 'classification_model.pkl'), 'rb') as f:
        models['chronic_classifier'] = pickle.load(f)
    with open(os.path.join('pkl', 'regression_models.pkl'), 'rb') as f:
        models['chronic_regressors'] = pickle.load(f)
    with open(os.path.join('pkl', 'mlb.pkl'), 'rb') as f:
        models['chronic_mlb'] = pickle.load(f)
    print("--- Chronic disease models loaded successfully! ---")

    # Load Diabetes Subtype Models
    diabetes_pkl_path = r"D:\SEN PRJ\cts\backend\diabetes_class"
    if os.path.exists(os.path.join(diabetes_pkl_path, 'lgbm_classifier.pkl')):
        models['diabetes_subtype_classifier'] = joblib.load(os.path.join(diabetes_pkl_path, 'lgbm_classifier.pkl'))
        models['diabetes_risk_model'] = joblib.load(os.path.join(diabetes_pkl_path, 'kmeans_model.pkl'))
        models['diabetes_preprocessor'] = joblib.load(os.path.join(diabetes_pkl_path, 'preprocessor.pkl'))
        models['diabetes_cluster_map'] = joblib.load(os.path.join(diabetes_pkl_path, 'cluster_risk_map.pkl'))
        print("--- Diabetes subtype models loaded successfully! ---")
    else:
        print("--- Diabetes subtype models not found ---")
    
    # Load Hypertension Model from the new location (FIXED - removed unsafe parameter)
    hypertension_pkl_path = r"D:\SEN PRJ\cts\backend\hypertension_class"
    print(f"Looking for hypertension models in: {hypertension_pkl_path}")

    if os.path.exists(hypertension_pkl_path):
        hypertension_files = os.listdir(hypertension_pkl_path)
        print(f"Files found in hypertension directory: {hypertension_files}")
        
        try:
            model_loaded = False
            
            # Try loading the main pipeline without unsafe parameter
            try:
                models['hypertension_pipeline'] = joblib.load(
                    os.path.join(hypertension_pkl_path, 'complete_pipeline.pkl')
                )
                print("--- Hypertension main pipeline loaded successfully! ---")
                model_loaded = True
            except Exception as e1:
                print(f"Loading failed: {e1}")
            
            # Try loading other model files without unsafe parameter
            if not model_loaded:
                other_model_files = [f for f in hypertension_files if f.endswith('.pkl') and f != 'complete_pipeline.pkl']
                for model_file in other_model_files:
                    try:
                        models['hypertension_pipeline'] = joblib.load(
                            os.path.join(hypertension_pkl_path, model_file)
                        )
                        print(f"--- Loaded from alternative file: {model_file} ---")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load {model_file}: {e}")
                        continue
            
            # Load additional metadata files without unsafe parameter
            for metadata_file in ['risk_metadata.pkl', 'hypertension_stages.pkl', 'hypertension_subtypes.pkl']:
                metadata_path = os.path.join(hypertension_pkl_path, metadata_file)
                if os.path.exists(metadata_path):
                    try:
                        models[metadata_file.split('.')[0]] = joblib.load(metadata_path)
                        print(f"--- {metadata_file} loaded successfully! ---")
                    except Exception as e:
                        print(f"!!! Error loading {metadata_file}: {e} !!!")
                else:
                    print(f"--- {metadata_file} not found ---")
            
            if not model_loaded:
                print("--- Could not load hypertension model, will use calculations ---")
            else:
                print("--- Hypertension models loaded successfully! ---")
                    
        except Exception as e:
            print(f"!!! Error loading hypertension models: {e} !!!")
    else:
        print("--- Hypertension directory not found ---")
        
except Exception as e:
    print(f"!!! Error loading models: {e} !!!")

# --- 3. ML HELPER FUNCTIONS ---
def get_risk_level(score):
    if score < 35: return "Low"
    elif score < 70: return "Medium"
    else: return "High"

def get_disease_explanation(patient_data, disease):
    reasons = []
    if disease == 'CKidney Disease':
        if patient_data.get('eGFR', 100) < 60: reasons.append(f"low eGFR ({patient_data.get('eGFR')})")
    elif disease == 'Diabetes':
        if patient_data.get('HbA1c', 0) >= 6.5: reasons.append(f"very high HbA1c ({patient_data.get('HbA1c')}%)")
    elif disease == 'Hypertension':
        if patient_data.get('Systolic_BP', 0) >= 140 or patient_data.get('Diastolic_BP', 0) >= 90: reasons.append(f"high blood pressure (Stage 2)")
    elif disease == 'Heart Disease':
        if patient_data.get('LDL_Cholesterol', 0) > 130: reasons.append(f"high LDL cholesterol")
    elif disease == 'Stroke':
        if patient_data.get('History_of_Stroke') == 1: reasons.append("a prior history of stroke")
    elif disease == 'Asthma':
        if patient_data.get('FEV1_FVC_Ratio', 1) < 0.7: reasons.append(f"low FEV1/FVC ratio")
            
    return ", ".join(reasons) if reasons else "a combination of sub-clinical factors."

def generate_hypertension_explanation(prediction, patient_data, probability, stage, subtype):
    explanation_parts = []
    
    if prediction:
        explanation_parts.append(f"The model predicts {subtype} hypertension ({stage}) with {probability:.1%} confidence.")
        explanation_parts.append("Primary contributing factors:")
        
        if patient_data.get('systolic_bp', 0) >= 140:
            explanation_parts.append(f"• Elevated systolic BP ({patient_data.get('systolic_bp')} mmHg)")
        if patient_data.get('diastolic_bp', 0) >= 90:
            explanation_parts.append(f"• Elevated diastolic BP ({patient_data.get('diastolic_bp')} mmHg)")
        if patient_data.get('age', 0) > 50:
            explanation_parts.append(f"• Age ({patient_data.get('age')} years)")
        if patient_data.get('bmi', 0) >= 25:
            explanation_parts.append(f"• BMI ({patient_data.get('bmi')})")
        if patient_data.get('family_history', 0) == 1:
            explanation_parts.append("• Family history")
        if patient_data.get('creatinine', 0) > 1.2:
            explanation_parts.append(f"• Creatinine level ({patient_data.get('creatinine')} mg/dL)")
            
    else:
        explanation_parts.append("No hypertension detected. Blood pressure levels appear normal.")
        if patient_data.get('systolic_bp', 0) < 120 and patient_data.get('diastolic_bp', 0) < 80:
            explanation_parts.append("• Optimal blood pressure range")
        
    return "\n".join(explanation_parts)

def calculate_hypertension_risks(systolic_bp, diastolic_bp, age, creatinine):
    """Calculate 1-year risks based on blood pressure and other factors"""
    bp_risk_factor = max(0, (systolic_bp - 120) / 40 + (diastolic_bp - 80) / 20)
    age_factor = max(0, (age - 40) / 30)
    creatinine_factor = max(0, (creatinine - 0.8) / 0.4)
    
    kidney_risk = min(95, 30 + bp_risk_factor * 25 + age_factor * 15 + creatinine_factor * 20)
    stroke_risk = min(95, 25 + bp_risk_factor * 30 + age_factor * 20)
    heart_risk = min(95, 20 + bp_risk_factor * 28 + age_factor * 18)
    
    return kidney_risk, stroke_risk, heart_risk

def determine_hypertension_stage(systolic_bp, diastolic_bp):
    """Determine hypertension stage based on blood pressure"""
    if systolic_bp >= 180 or diastolic_bp >= 120:
        return "Stage_3", "Hypertensive_Crisis"
    elif systolic_bp >= 160 or diastolic_bp >= 100:
        return "Stage_2", "Established_Hypertension"
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        return "Stage_1", "Primary_Hypertension"
    elif systolic_bp >= 130 or diastolic_bp >= 80:
        return "Elevated", "Prehypertension"
    else:
        return "Normal", "Optimal"

def determine_hypertension_subtype(patient_data):
    """Determine hypertension subtype based on patient characteristics"""
    if patient_data.get('family_history', 0) == 1:
        return "Familial"
    elif patient_data.get('creatinine', 1.0) > 1.3:
        return "Secondary_Renal"
    elif patient_data.get('bmi', 25) > 30:
        return "Obesity_Related"
    else:
        return "Primary_Essential"

# --- 4. DATABASE HELPER FUNCTIONS ---
def get_current_patient_id_from_token():
    """Get patient ID from session token"""
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        session_token = auth_header.split(' ')[1]
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT patient_id FROM user_sessions WHERE session_token = %s AND expires_at > NOW()", (session_token,))
        session = cursor.fetchone()
        return session['patient_id'] if session else None
    except mysql.connector.Error as err:
        print(f"Error getting current patient from token: {err}")
        return None
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def save_user_activity(patient_id, activity_type, page_visited, details=None):
    """Save user activity to database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
            INSERT INTO user_activity_log (patient_id, activity_type, page_visited, details)
            VALUES (%s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            patient_id,
            activity_type,
            page_visited,
            json.dumps(details) if details else None
        ))
        
        conn.commit()
        print(f"User activity saved for patient {patient_id}")
        
    except mysql.connector.Error as err:
        print(f"Error saving user activity: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def save_prediction_to_db(patient_id, prediction_type, page_source, input_data, output_data):
    """Save prediction input and output to database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
            INSERT INTO user_predictions (patient_id, prediction_type, page_source, input_data, output_data)
            VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            patient_id,
            prediction_type,
            page_source,
            json.dumps(input_data),
            json.dumps(output_data)
        ))
        
        conn.commit()
        print(f"Prediction saved to database for patient {patient_id}")
        
    except mysql.connector.Error as err:
        print(f"Error saving prediction to database: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def save_diabetes_entry(patient_id, input_data, output_data):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO diabetes_entries (patient_id, input_data, output_data)
            VALUES (%s, %s, %s)
            """,
            (patient_id, json.dumps(input_data), json.dumps(output_data))
        )
        conn.commit()
        print(f"Diabetes entry saved for patient {patient_id}")
    except mysql.connector.Error as err:
        print(f"Error saving diabetes entry: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def save_hypertension_entry(patient_id, input_data, output_data):
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO hypertension_entries (patient_id, input_data, output_data)
            VALUES (%s, %s, %s)
            """,
            (patient_id, json.dumps(input_data), json.dumps(output_data))
        )
        conn.commit()
        print(f"Hypertension entry saved for patient {patient_id}")
    except mysql.connector.Error as err:
        print(f"Error saving hypertension entry: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def save_vitals_data(patient_id, vitals_data):
    """Save vitals data to database"""
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        query = """
            INSERT INTO user_predictions (patient_id, prediction_type, page_source, input_data, output_data)
            VALUES (%s, 'vitals', 'add-vitals', %s, %s)
        """
        
        cursor.execute(query, (
            patient_id,
            json.dumps(vitals_data),
            json.dumps({"status": "vitals_recorded", "timestamp": datetime.now().isoformat()})
        ))
        
        conn.commit()
        print(f"Vitals saved for patient {patient_id}")
        
    except mysql.connector.Error as err:
        print(f"Error saving vitals: {err}")
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def build_patient_report_text(patient_id: int) -> str:
    """Aggregate patient data from DB into a single text blob to feed the LLM.
    Excludes login/session information.
    """
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        sections = []

        # Predictions summary
        cursor.execute(
            """
            SELECT prediction_type, input_data, output_data, created_at
            FROM user_predictions
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (patient_id,)
        )
        for row in cursor.fetchall():
            sections.append(f"[Prediction: {row['prediction_type']} at {row['created_at']}]\nInput: {row['input_data']}\nOutput: {row['output_data']}")

        # Diabetes entries
        cursor.execute(
            """
            SELECT input_data, output_data, created_at
            FROM diabetes_entries
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (patient_id,)
        )
        for row in cursor.fetchall():
            sections.append(f"[Diabetes Entry at {row['created_at']}]\nInput: {row['input_data']}\nOutput: {row['output_data']}")

        # Hypertension entries
        cursor.execute(
            """
            SELECT input_data, output_data, created_at
            FROM hypertension_entries
            WHERE patient_id = %s
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (patient_id,)
        )
        for row in cursor.fetchall():
            sections.append(f"[Hypertension Entry at {row['created_at']}]\nInput: {row['input_data']}\nOutput: {row['output_data']}")

        report_text = "\n\n".join(sections) if sections else "No prior records."
        return report_text
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def split_days(plan_text: str):
    days = {}
    parts = re.split(r"\n(?=Day \d+)", plan_text.strip())
    for p in parts:
        if p.strip():
            label = p.split("\n")[0].strip()
            days[label] = p.strip()
    return days

def generate_and_store_care_plan(patient_id: int) -> int:
    """Generate a 7-day care plan for patient and store it. Returns care_plan_id."""
    report_text = build_patient_report_text(patient_id)
    care_text = llm_generate_care_plan(report_text)
    care_text = format_care_text(care_text)

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO care_plans (patient_id) VALUES (%s)", (patient_id,))
        care_plan_id = cursor.lastrowid

        days_map = split_days(care_text)
        created_at = datetime.now()
        for i in range(1, 8):
            label = f"Day {i}"
            content = days_map.get(label, "")
            unlock_time = created_at + timedelta(days=i-1)
            cursor.execute(
                """
                INSERT INTO care_plan_days (care_plan_id, day_number, content, unlocked_at)
                VALUES (%s, %s, %s, %s)
                """,
                (care_plan_id, i, content, unlock_time)
            )
        conn.commit()
        return care_plan_id
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# --- 5. API ENDPOINTS ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "Python server is running!"})

# -- PATIENT AUTHENTICATION --
@app.route('/patients/signup', methods=['POST'])
def patient_signup():
    data = request.get_json()
    name, email, password = data.get('name'), data.get('email'), data.get('password')
    if not all([name, email, password]): return jsonify({"message": "Missing required fields"}), 400
    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        query = "INSERT INTO patients (name, email, password_hash) VALUES (%s, %s, %s)"
        cursor.execute(query, (name, email, pw_hash))
        conn.commit()
        
        # Get the newly created patient ID
        patient_id = cursor.lastrowid
        
        # Create session
        session_token = bcrypt.generate_password_hash(f"{email}{datetime.now()}").decode('utf-8')
        cursor.execute(
            "INSERT INTO user_sessions (patient_id, session_token, expires_at) VALUES (%s, %s, DATE_ADD(NOW(), INTERVAL 7 DAY))",
            (patient_id, session_token)
        )
        conn.commit()
        
        # Log signup activity
        save_user_activity(patient_id, "signup", "signup", {"email": email})
        
        return jsonify({
            "message": "Patient account created successfully",
            "patient_id": patient_id,
            "session_token": session_token,
            "name": name
        }), 201
        
    except mysql.connector.Error as err:
        if err.errno == 1062: return jsonify({"message": "An account with this email already exists."}), 409
        return jsonify({"message": f"Database error: {err}"}), 500
    finally:
        if 'conn' in locals() and conn.is_connected(): cursor.close(); conn.close()

@app.route('/patients/login', methods=['POST'])
def patient_login():
    data = request.get_json()
    email, password = data.get('email'), data.get('password')
    if not email or not password: return jsonify({"message": "Missing email or password"}), 400
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True) 
        query = "SELECT * FROM patients WHERE email = %s"
        cursor.execute(query, (email,))
        patient = cursor.fetchone()
        
        if patient and bcrypt.check_password_hash(patient['password_hash'], password):
            # Create new session
            session_token = bcrypt.generate_password_hash(f"{email}{datetime.now()}").decode('utf-8')
            cursor.execute(
                "INSERT INTO user_sessions (patient_id, session_token, expires_at) VALUES (%s, %s, DATE_ADD(NOW(), INTERVAL 7 DAY))",
                (patient['id'], session_token)
            )
            conn.commit()
            
            # Log login activity
            save_user_activity(patient['id'], "login", "login", {"email": email})
            
            return jsonify({ 
                "message": "Login successful",
                "patient_id": patient['id'],
                "session_token": session_token,
                "name": patient['name']
            }), 200
        else:
            return jsonify({"message": "Invalid email or password"}), 401
            
    except mysql.connector.Error as err:
        return jsonify({"message": f"Database error: {err}"}), 500
    finally:
        if 'conn' in locals() and conn.is_connected(): cursor.close(); conn.close()

# --- 6. ML PREDICTION ENDPOINTS ---
@app.route('/generate_report', methods=['POST'])
def generate_report():
    if 'chronic_classifier' not in models: 
        return jsonify({"error": "Chronic disease models are not loaded."}), 500
    
    patient_input_data = request.get_json()
    if not patient_input_data: 
        return jsonify({"error": "No input data provided."}), 400
    
    try:
        patient_id = get_current_patient_id_from_token()
        input_df = pd.DataFrame([patient_input_data])
        final_report = { "predicted_conditions": [], "risk_assessment": [] }
        
        # Chronic disease prediction
        predicted_list = models['chronic_mlb'].inverse_transform(models['chronic_classifier'].predict(input_df))[0]
        
        if not predicted_list:
            final_report["predicted_conditions"].append({ 
                "disease": "Healthy", 
                "explanation": "The model predicts the patient is Healthy."
            })
        else:
            for disease in predicted_list:
                explanation = get_disease_explanation(patient_input_data, disease)
                final_report["predicted_conditions"].append({ 
                    "disease": disease, 
                    "explanation": f"Detected based on {explanation}."
                })
                
                if disease in models['chronic_regressors']:
                    reg_model = models['chronic_regressors'][disease]
                    risk_score = reg_model.predict(input_df)[0]
                    risk_score = max(0, min(100, risk_score))
                    final_report["risk_assessment"].append({ 
                        "disease": disease, 
                        "risk_score": round(risk_score, 1), 
                        "risk_level": get_risk_level(risk_score), 
                        "primary_drivers": f"This risk level is driven by {explanation}." 
                    })
        
        # Save to database
        if patient_id:
            save_prediction_to_db(patient_id, 'chronic_disease', 'generate-report', patient_input_data, final_report)
            save_user_activity(patient_id, 'prediction', 'generate-report', {
                'conditions': predicted_list if predicted_list else ['Healthy']
            })
            # Auto-generate a new care plan immediately after analysis
            try:
                care_plan_id = generate_and_store_care_plan(patient_id)
                print(f"Care plan generated for patient {patient_id}: {care_plan_id}")
            except Exception as ce:
                print(f"Care plan generation failed for patient {patient_id}: {ce}")
        
        return jsonify(final_report)
    except Exception as e:
        return jsonify({"error": "Care plan server busy, try again"}), 503

# --- DIABETES SUBTYPE PREDICTION ---
@app.route('/predict_diabetes_subtype', methods=['POST'])
def predict_diabetes_subtype():
    if 'diabetes_subtype_classifier' not in models:
        return jsonify({"error": "Diabetes subtype models are not loaded."}), 500
    
    try:
        data = request.get_json()
        patient_id = get_current_patient_id_from_token()
        
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Is_Pregnant': data.get('Is_Pregnant', 0),
            'Age_at_Diagnosis': data.get('Age_at_Diagnosis', 45),
            'BMI_at_Diagnosis': data.get('BMI_at_Diagnosis', 25),
            'HbA1c': data.get('HbA1c', 6.0),
            'C_Peptide_Level': data.get('C_Peptide_Level', 1.0),
            'Family_History': data.get('Family_History', 'None'),
            'Autoantibodies_Status': data.get('Autoantibodies_Status', 'Negative'),
            'Genetic_Test_Result': data.get('Genetic_Test_Result', 'Negative')
        }])
        
        # Make prediction
        prediction = models['diabetes_subtype_classifier'].predict(input_data)[0]
        prediction_proba = models['diabetes_subtype_classifier'].predict_proba(input_data)[0]
        
        # Get risk level
        processed_data = models['diabetes_preprocessor'].transform(input_data)
        risk_cluster = models['diabetes_risk_model'].predict(processed_data)[0]
        risk_level = models['diabetes_cluster_map'].get(risk_cluster, 'Unknown')
        
        # Generate explanation
        explanation = f"The model predicts {prediction} with {max(prediction_proba)*100:.1f}% confidence. Risk assessment: {risk_level} risk level."
        
        response = {
            'predicted_type': str(prediction),
            'confidence_score': float(max(prediction_proba)) * 100,
            'risk_level': risk_level,
            'explanation': explanation
        }
        
        # Save to database
        if patient_id:
            save_prediction_to_db(patient_id, 'diabetes_subtype', 'diabetes-check', data, response)
            save_diabetes_entry(patient_id, data, response)
            save_user_activity(patient_id, 'prediction', 'diabetes-check', {
                'type': 'diabetes',
                'predicted_type': str(prediction)
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": "Care plan server busy, try again"}), 503

# --- HYPERTENSION PREDICTION ---
@app.route('/predict_hypertension', methods=['POST'])
def predict_hypertension():
    try:
        data = request.get_json()
        patient_id = get_current_patient_id_from_token()
        
        # Extract patient data
        systolic_bp = data.get('systolic_bp', 120)
        diastolic_bp = data.get('diastolic_bp', 80)
        age = data.get('age', 50)
        creatinine = data.get('creatinine', 1.0)
        
        # Determine hypertension stage and subtype
        stage, stage_description = determine_hypertension_stage(systolic_bp, diastolic_bp)
        subtype = determine_hypertension_subtype(data)
        
        # Calculate 1-year risks
        kidney_risk, stroke_risk, heart_risk = calculate_hypertension_risks(
            systolic_bp, diastolic_bp, age, creatinine
        )
        
        # Determine overall hypertension risk
        hypertension_risk = stage not in ["Normal", "Elevated"]
        probability = min(0.95, (kidney_risk + stroke_risk + heart_risk) / 300)
        
        # Determine risk level
        if probability > 0.7:
            risk_level = 'High'
        elif probability > 0.3:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        # Generate explanation
        explanation = generate_hypertension_explanation(
            hypertension_risk, 
            data, 
            probability,
            stage_description,
            subtype
        )
        
        response = {
            'hypertension_risk': hypertension_risk,
            'probability': probability,
            'risk_level': risk_level,
            'stage': stage,
            'subtype': subtype,
            'kidney_risk_1yr': round(kidney_risk, 2),
            'stroke_risk_1yr': round(stroke_risk, 2),
            'heart_risk_1yr': round(heart_risk, 2),
            'explanation': explanation
        }
        
        # Save to database
        if patient_id:
            save_prediction_to_db(patient_id, 'hypertension', 'hypertension-check', data, response)
            save_hypertension_entry(patient_id, data, response)
            save_user_activity(patient_id, 'prediction', 'hypertension-check', {
                'type': 'hypertension',
                'risk_level': risk_level,
                'stage': stage
            })
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": "Care plan server busy, try again"}), 503

# --- VITALS RECORDING ---
@app.route('/record_vitals', methods=['POST'])
def record_vitals():
    try:
        data = request.get_json()
        patient_id = get_current_patient_id_from_token()
        
        if not patient_id:
            return jsonify({"error": "Not authenticated"}), 401
        
        # Save vitals to database
        save_vitals_data(patient_id, data)
        save_user_activity(patient_id, 'vitals_recording', 'add-vitals', data)
        
        return jsonify({
            "message": "Vitals recorded successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- CARE PLAN GENERATION & RETRIEVAL ---
@app.route('/care_plan/generate', methods=['POST'])
def generate_care_plan():
    try:
        patient_id = get_current_patient_id_from_token()
        if not patient_id:
            return jsonify({"error": "Not authenticated"}), 401

        # Build consolidated patient report from DB
        report_text = build_patient_report_text(patient_id)

        # Call LLM
        care_text = llm_generate_care_plan(report_text)
        care_text = format_care_text(care_text)

        # Persist care plan and per-day items with daily unlock policy
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO care_plans (patient_id) VALUES (%s)", (patient_id,))
        care_plan_id = cursor.lastrowid

        days_map = split_days(care_text)
        created_at = datetime.now()
        for i in range(1, 8):
            label = f"Day {i}"
            content = days_map.get(label, "")
            unlock_time = created_at + timedelta(days=i-1)
            cursor.execute(
                """
                INSERT INTO care_plan_days (care_plan_id, day_number, content, unlocked_at)
                VALUES (%s, %s, %s, %s)
                """,
                (care_plan_id, i, content, unlock_time)
            )

        conn.commit()
        return jsonify({"message": "Care plan generated", "care_plan_id": care_plan_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/care_plan/today', methods=['GET'])
def get_today_care_plan():
    try:
        patient_id = get_current_patient_id_from_token()
        if not patient_id:
            return jsonify({"error": "Not authenticated"}), 401

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # Latest plan
        cursor.execute("SELECT id, created_at FROM care_plans WHERE patient_id = %s ORDER BY created_at DESC LIMIT 1", (patient_id,))
        plan = cursor.fetchone()
        if not plan:
            return jsonify({"error": "No care plan found"}), 404

        # Find the latest unlocked day
        cursor.execute(
            """
            SELECT day_number, content, unlocked_at, is_completed, completed_choice
            FROM care_plan_days
            WHERE care_plan_id = %s AND unlocked_at <= NOW()
            ORDER BY day_number DESC
            LIMIT 1
            """,
            (plan['id'],)
        )
        day = cursor.fetchone()
        if not day:
            return jsonify({"error": "No unlocked day yet"}), 404

        return jsonify({"day_number": day['day_number'], "content": day['content'], "unlocked_at": day['unlocked_at'].isoformat(), "is_completed": bool(day['is_completed']), "completed_choice": day['completed_choice']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/care_plan/complete', methods=['POST'])
def complete_today_plan():
    try:
        patient_id = get_current_patient_id_from_token()
        if not patient_id:
            return jsonify({"error": "Not authenticated"}), 401

        data = request.get_json() or {}
        choice = data.get('choice')  # 'yes' or 'no'
        if choice not in ['yes', 'no']:
            return jsonify({"error": "choice must be 'yes' or 'no'"}), 400

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT id FROM care_plans WHERE patient_id = %s ORDER BY created_at DESC LIMIT 1", (patient_id,))
        plan = cursor.fetchone()
        if not plan:
            return jsonify({"error": "No care plan found"}), 404

        # Find latest unlocked day
        cursor.execute(
            """
            SELECT id, day_number FROM care_plan_days
            WHERE care_plan_id = %s AND unlocked_at <= NOW()
            ORDER BY day_number DESC LIMIT 1
            """,
            (plan['id'],)
        )
        day = cursor.fetchone()
        if not day:
            return jsonify({"error": "No unlocked day to complete"}), 400

        # Mark as completed with choice
        cursor2 = conn.cursor()
        cursor2.execute(
            "UPDATE care_plan_days SET is_completed = 1, completed_choice = %s WHERE id = %s",
            (choice, day['id'])
        )
        conn.commit()
        return jsonify({"message": "Day marked completed", "day_number": day['day_number'], "choice": choice})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# --- 7. DATA RETRIEVAL ENDPOINTS ---
@app.route('/user/predictions', methods=['GET'])
def get_user_predictions():
    try:
        patient_id = get_current_patient_id_from_token()
        if not patient_id:
            return jsonify({"error": "Not authenticated"}), 401
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT prediction_type, page_source, input_data, output_data, created_at 
            FROM user_predictions 
            WHERE patient_id = %s 
            ORDER BY created_at DESC
        """
        
        cursor.execute(query, (patient_id,))
        predictions = cursor.fetchall()
        
        return jsonify({"predictions": predictions})
        
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

@app.route('/user/activity', methods=['GET'])
def get_user_activity():
    try:
        patient_id = get_current_patient_id_from_token()
        if not patient_id:
            return jsonify({"error": "Not authenticated"}), 401
        
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        query = """
            SELECT activity_type, page_visited, details, created_at 
            FROM user_activity_log 
            WHERE patient_id = %s 
            ORDER BY created_at DESC
            LIMIT 100
        """
        
        cursor.execute(query, (patient_id,))
        activities = cursor.fetchall()
        
        return jsonify({"activities": activities})
        
    except mysql.connector.Error as err:
        return jsonify({"error": f"Database error: {err}"}), 500
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

# --- 8. RUN SERVER ---
if __name__ == '__main__':
    try:
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    except Exception as e:
        print(f"Error starting server: {e}")