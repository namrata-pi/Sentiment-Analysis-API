# Sentiment-Analysis-API
A FastAPI-based REST API that predicts sentiment (positive, neutral, negative) from text using a trained Logistic Regression model.


1. Clone/Download the Project
2. Install Dependencies
   pip install -r requirements.txt
3. Prepare Your Model
    Make sure you have these files in your project directory:

   preprocessed.xlsx -  training dataset
   main.py - The FastAPI application
   requirements.txt - Dependencies
4.Train and Save the Model
     python log_model.py
This will create sentiment_model.pkl file.

5.Start the API server
  python main.py

6.Test the API
   Open your browser and go to:
   API Documentation: http://localhost:8000/docs
   API Usage 
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Looking forward to the demo!"}'

  

