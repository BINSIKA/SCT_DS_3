import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Load the Data ---
print("--- 1. Loading and Initial Data Inspection ---")
try:
    # Explicitly force the semicolon (;) delimiter, which is typical for this dataset.
    df = pd.read_csv(r'bank.csv')
    
    # Final check for successful loading before proceeding
    if 'deposit' not in df.columns:
        raise ValueError("Could not find the 'deposit' column. Please check the CSV file content.")
        
except FileNotFoundError:
    print("Error: 'bank.csv' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    raise Exception(f"An error occurred during CSV loading: {e}") from e


# Display the first few rows and check data types
print("Shape of data:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData Types:")
# Use df.info(verbose=True) to show all columns clearly
df.info()

# --- 2. Data Pre-processing: Feature Engineering and Encoding ---

# RE-INCLUDING: The target variable is 'deposit'. Convert 'yes'/'no' to 1/0.
# This enables the script to train a classifier to predict customer purchase.
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

# Separate features (X) and target (y)
X = df.drop('deposit', axis=1)
y = df['deposit']

# Identify categorical features for one-hot encoding (get_dummies)
# Exclude the duration feature to prevent data leakage, as it's known only after the call.
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Drop 'duration' feature and convert remaining categorical features to numerical
X_processed = X.drop('duration', axis=1)
X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)

# Handle potential residual text/binary columns if any
for col in X_processed.columns:
    if X_processed[col].dtype == 'bool':
        X_processed[col] = X_processed[col].astype(int)

print("\n--- 2. Pre-processing Complete ---")
print(f"Number of features after encoding: {X_processed.shape[1]}")
print(X_processed.head())

# --- 3. Split the Data ---
# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

print("\n--- 3. Data Split Complete ---")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# --- 4. Train the Decision Tree Classifier ---
# Initialize the Decision Tree Classifier
# Using Gini impurity and setting max_depth to control overfitting
dt_classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,       # Limit depth for interpretability and to prevent extreme overfitting
    random_state=42
)

print("\n--- 4. Training Decision Tree Classifier (Max Depth: 5) ---")
dt_classifier.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluate the Model ---
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n--- 5. Model Evaluation ---")
print(f"Accuracy on the Test Set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Feature Importance Visualization ---

# Extract feature importance scores
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X_processed.columns)

# Get the top 10 most important features
top_10_features = feature_importances.nlargest(10)





# --- 7. Decision Tree Structure Visualization (New Section) ---
print("\n--- 7. Decision Tree Structure Visualization ---")

# Setup figure size for better viewing of the tree structure
plt.figure(figsize=(20, 10))
plot_tree(
    dt_classifier, 
    filled=True, 
    rounded=True, 
    class_names=['No Deposit (0)', 'Deposit (1)'], 
    feature_names=X_processed.columns,
    max_depth=3, # Limiting depth to 3 for readability
    fontsize=8
)
plt.title(f'Decision Tree Structure (Max Depth: {dt_classifier.max_depth})')
plt.show()


print("\nAnalysis Summary:")
print("The Decision Tree Classifier was trained to predict customer deposit (purchase).")
print(f"It achieved an accuracy of {accuracy:.4f} on the test set.")
print("The feature importance scores highlight which demographic and behavioral factors were most predictive of a purchase.")
print("Note: The 'duration' feature was intentionally dropped to ensure the model is based on features available *before* the customer call.")
