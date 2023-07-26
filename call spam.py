# Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load the dataset
data = pd.read_csv('spam_call_data.csv')

# Split the dataset into features and target
X = data.iloc[:, :-1]
print(X)
y = data.iloc[:, -1]  # select only the last column
# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Predict the class labels for all data points
predicted_labels = []
for i in range(len(X)):
    row = X[i]
    predicted_label = rf.predict([row])[0]
    if predicted_label == 0:
        predicted_labels.append('CALL IS  A SPAM')
    else:
        predicted_labels.append('CALL IS  NOT SPAM')

# Print the predicted labels for all data points
for i in range(len(data)):
    row = data.iloc[i]
    print(f"Call {i+1}: {row['Is Spam']} is {predicted_labels[i]}")


