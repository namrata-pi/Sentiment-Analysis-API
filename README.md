# Sentiment-Analysis-API
A FastAPI-based REST API that predicts sentiment (positive, neutral, negative) from text using a trained Logistic Regression model.


1. Clone/Download the Project
2. Install Dependencies
   ```bash
   pip install -r requirements.txt
5. Prepare Your Model
    Make sure you have these files in your project directory:

  * preprocessed.xlsx -  training dataset
  * main.py - The FastAPI application
  *  requirements.txt - Dependencies
4.Train and Save the Model
     ```bash
     python log_model.py
This will create sentiment_model.pkl file.

5.Start the API server
  ```bash
python main.py
```
 
6.Test the API
   Open your browser and go to:
   API Documentation: http://localhost:8000/docs
   API Usage 
   
       curl -X POST "http://localhost:8000/predict" \
            -H "Content-Type: application/json" \
            -d '{"text": "This is amazing let's proceed"}'
   Response:
   ```bash
   {
  "label": "positive",
  "confidence": 0.886,
  "probabilities": {
    "negative": 0.05,
    "neutral": 0.08,
    "positive": 0.886
  }
}
```

 Project Structure
```bash
   sentiment-api
   main.py                # FastAPI application
   log_model.py           # Script to train and save model
   test_api.py            # Test script
   requirements.txt       # Python dependencies
   README.md              # This file
   preprocessed.xlsx      # Training dataset
   sentiment_model.pkl    # Trained model (generated)
```

Model Parameters
The model uses:

*Algorithm: Logistic Regression
*Vectorization: TF-IDF with 1000 features
*N-grams: Unigrams and bigrams (1,2)
*Classes: negative (-1), neutral (0), positive (1)


Testing
  ``` bash
python test_api.py
```

   Manual Testing
```bash
   1.Start the API server
   2.Go to http://localhost:8000/docs
   3.Try the /predict endpoint with different texts:
       "This is amazing!" (should predict positive)
        "Not interested" (should predict negative)

```

      

       

  

