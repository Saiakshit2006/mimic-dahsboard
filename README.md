# 🏥 AI-Powered ICU Analysis & Medical Assistant

A full-stack medical data analytics platform built on Azure + Databricks + AI.

## 🚀 Live Demo
👉 [View Live App](https://mimic-dahsboard-ynvmjcjzg5wanpef77ugw7.streamlit.app/)

## 📌 Project Overview
This project analyzes the MIMIC-III clinical database from Beth Israel 
Deaconess Medical Center using modern cloud and AI technologies.

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Cloud Storage | Azure Blob Storage |
| Data Engineering | Azure Databricks |
| Data Catalog | Databricks Unity Catalog |
| Delta Lake | Delta Tables (Raw Layer) |
| AI/GenAI | Groq LLaMA 3.3 70B |
| Frontend | Streamlit |
| Language | Python |

## 📊 Features
- 📈 ICU Dashboard with real-time charts
- 🔍 Patient Search & Medical History
- ⚠️ Patient Risk Score & Mortality Predictor
- 🧪 Lab Results Trend Analysis
- 🔄 ICU Readmission Analysis
- 🔬 Interactive Filter & Explore
- 🤖 AI Medical Assistant (Chat with ICU Data)

## 🗄️ Dataset
MIMIC-III Clinical Database Demo
- 100 Patients
- 26 Tables
- 758,355 Chart Events
- Stored in Databricks Unity Catalog (mimic_db)

## 📁 Project Structure
mimic-dahsboard/
├── app.py              # Main Streamlit app
├── mimic_data.json     # MIMIC-III processed data
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## 🔧 Setup & Installation
pip install -r requirements.txt
streamlit run app.py

## 👨‍💻 Author
Akshit — Data Engineering Intern
