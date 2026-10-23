-- Workload-Aware MySQL Index Advisor
-- Schema Setup Script for index_advisor_test
-- Target Engine: MySQL 8.4+

CREATE DATABASE IF NOT EXISTS index_advisor_test;
USE index_advisor_test;

CREATE TABLE IF NOT EXISTS students (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    city VARCHAR(50) NOT NULL,
    cgpa DECIMAL(3, 2) NOT NULL,
    INDEX idx_students_department (department)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- If the table already exists without the baseline index, add it via:
-- CREATE INDEX idx_students_department ON students (department);
