
import random
import mysql.connector

# Connect to the project database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password=input("Enter your MySQL root password: "),
    database="index_advisor_test"
)

cursor = connection.cursor()

departments = ["CSE", "ECE", "IT", "EEE", "MECH"]
cities = ["Hyderabad", "Warangal", "Vijayawada", "Chennai", "Bengaluru"]

# Generate 10,000 sample students
records = []

for i in range(10000):
    records.append((
        f"Student_{i + 1}",
        random.choice(departments),
        random.choice(cities),
        round(random.uniform(5.0, 10.0), 2)
    ))

# Insert records efficiently in batches
query = """
INSERT INTO students (name, department, city, cgpa)
VALUES (%s, %s, %s, %s)
"""

batch_size = 1000

for start in range(0, len(records), batch_size):
    cursor.executemany(query, records[start:start + batch_size])
    connection.commit()
    print(f"Inserted {min(start + batch_size, len(records))} records")

cursor.close()
connection.close()

print("Data generation completed successfully!")
