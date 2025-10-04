 Bank Marketing Prediction (Decision Tree Classifier)

A Decision Tree model that predicts whether a bank customer will subscribe to a term deposit  based on demographic and marketing data.

---

 Overview

Steps:
1. Load and inspect `bank.csv`  
2. Preprocess data (encode categorical features, drop `duration`)  
3. Train a Decision Tree Classifier
4. Evaluate model (accuracy, classification report)  
5. Visualize the decision tree
   
Goal:
Predict deposit subscription (`yes`/`no`) before customer contact ends — avoiding data leakage from `duration`.

⚙️ Requirements

bash
pip install pandas scikit-learn matplotlib numpy

▶️ Run

1. Place `bank.csv` in the project folder.  
2. Run the script:

bash
python bank_decision_tree.py

📊 Output

Test accuracy and classification report  
 Decision tree visualization  

