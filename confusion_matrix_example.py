# Python code to calculate confusion matrix and metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Hypothetical predictions and true labels
# 200 positives (readmitted), 800 negatives (not readmitted)
y_true = [1]*200 + [0]*800
# Predictions: TP=150, FP=50, FN=50, TN=750
y_pred = [1]*150 + [0]*50 + [1]*50 + [0]*750

cm = confusion_matrix(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print("Confusion Matrix:")
print(cm)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
