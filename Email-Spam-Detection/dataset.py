import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# STEP 1: LOAD THE DATASET (spam.csv)
# Make sure spam.csv is in the SAME folder as this file
data = pd.read_csv("spam.csv", encoding="latin-1")

# Most spam datasets have columns like: v1 (label), v2 (message)
# Keep only first two columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# STEP 2: MAP LABELS TO NUMBERS
# spam = 1, ham = 0
data["label"] = data["label"].map({"spam": 1, "ham": 0})

# Remove any bad rows
data = data.dropna(subset=["label", "message"])
data["message"] = data["message"].astype(str)

# STEP 3: FEATURES AND TARGET
X = data["message"]
y = data["label"]

# STEP 4: TEXT VECTORIZATION
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# STEP 5: TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 6: TRAIN MODEL
model = MultinomialNB()
model.fit(X_train, y_train)

# STEP 7: EVALUATE MODEL
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# STEP 8: TEST WITH A CUSTOM MESSAGE
test_msg = ["Congratulations you won a free prize"]
result = model.predict(vectorizer.transform(test_msg))

if result[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
