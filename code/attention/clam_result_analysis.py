import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
with open('/home/minkyoon/first/CLAM/results/remission_stratified_for_sysam_recommend_setting_7,2,1/task_1_tumor_vs_normal_CLAM_50_s1/split_0_results.pkl', 'rb') as file:
    results = pickle.load(file)

# Extract true labels and predicted probabilities
true_labels = np.array([v['label'] for v in results.values()])
predicted_probs = np.array([v['prob'][0, 1] for v in results.values()])  # Probability of positive class


threshold = 0.5  
predicted_labels = (predicted_probs >= threshold).astype(int)


#  confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)



# Compute AUROC
fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
roc_auc = auc(fpr, tpr)

# Compute Sensitivity, Specificity, F1-Score, Recall
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
f1 = f1_score(true_labels, predicted_labels)
recall = sensitivity  # in binary classification, recall is the same as sensitivity

# Plotting the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"verticalalignment": 'center', 'size': 15})
plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], labels=['Negative', 'Positive'])
plt.yticks([0.5, 1.5], labels=['Negative', 'Positive'], va='center')
plt.show()


# ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# metrics
print(f'Sensitivity: {sensitivity:.2f}')
print(f'Specificity: {specificity:.2f}')
print(f'F1-Score: {f1:.2f}')
print(f'Recall: {recall:.2f}')








#
cm=np.array([[28 ,17],
            [29 ,16]])
cm=np.array([[95,5],
             [14,13]])

# find tp tn fp fn

# Initialize counters for tp, tn, fp, fn
tp = []
tn = []
fp = []
fn = []

# Iterate over the dictionary items
for slide_id, info in results.items():
    # Determine the predicted label (class with highest probability)
    pred_label = np.argmax(info['prob'])
    
    # Compare the predicted and true labels
    if pred_label == info['label']:
        # If they're the same, it's either a true positive or true negative
        if pred_label == 1:
            tp.append(int(slide_id))  # True positive
        else:
            tn.append(int(slide_id))  # True negative
    else:
        # If they're different, it's either a false positive or false negative
        if pred_label == 1:
            fp.append(int(slide_id))  # False positive
        else:
            fn.append(int(slide_id))  # False negative

print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
