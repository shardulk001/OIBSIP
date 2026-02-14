# Email Spam Detection 📧

## Internship
Oasis Infobyte – Data Science / Machine Learning Internship (OIBSIP)

## Objective
To build a machine learning model that classifies messages as **Spam** or **Not Spam (Ham)** using Python.

## Dataset
The project uses a spam dataset (`spam.csv`) containing labeled messages:
- `spam` → Unwanted / promotional messages
- `ham` → Normal messages

Each record contains:
- Label (spam/ham)
- Message text

## Algorithm Used
- **TF-IDF Vectorizer** for converting text into numerical features
- **Naive Bayes Classifier (MultinomialNB)** for classification

## Steps Performed
1. Loaded the dataset using pandas
2. Selected relevant columns (label and message)
3. Converted labels into numeric form (spam = 1, ham = 0)
4. Cleaned the data and converted text to string format
5. Converted text into numerical features using TF-IDF
6. Split the data into training and testing sets
7. Trained a Naive Bayes model
8. Evaluated the model using accuracy score
9. Tested the model with a custom message

## How to Run the Project
1. Install the required libraries:
2. Run the Python file:


## Output
The program prints:
- Model accuracy (around 97% on this dataset)
- Prediction for a sample message (Spam or Not Spam)

## Tools & Technologies
- Python
- Pandas
- Scikit-learn

## Conclusion
This project demonstrates a simple and effective approach to building an email spam detection system using machine learning and text processing techniques.
