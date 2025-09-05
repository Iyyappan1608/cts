import mysql.connector

# Test database connection
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'admin',
    'database': 'health_app_db'
}

try:
    print("Testing database connection...")
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT 'Connection successful!' as message")
    result = cursor.fetchone()
    print(f"Result: {result[0]}")
    cursor.close()
    conn.close()
    print("Database connection test passed!")
except Exception as e:
    print(f"Database connection failed: {e}")
