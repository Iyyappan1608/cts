-- Database setup script for health_app_db
-- Run this script in your MySQL client

-- Create database
CREATE DATABASE IF NOT EXISTS health_app_db;
USE health_app_db;

-- Create patients table (required for foreign key references)
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create user_predictions table
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
);

-- Create user_sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    session_token VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Create user_activity_log table
CREATE TABLE IF NOT EXISTS user_activity_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    activity_type VARCHAR(100),
    page_visited VARCHAR(100),
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Show all tables
SHOW TABLES;

-- Display success message
SELECT 'Database health_app_db setup completed successfully!' as status;
