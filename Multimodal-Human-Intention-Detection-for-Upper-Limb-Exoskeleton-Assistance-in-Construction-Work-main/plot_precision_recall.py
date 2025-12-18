import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np

SAVE_DIR = "Pre Trained Model"
with open(f"{SAVE_DIR}/test_results.pkl", "rb") as f:    
    data = pickle.load(f)

all_labels = data["labels"]
all_scores = data["scores"]
class_names = data["class_names"]


print(f"Labels shape: {all_labels.shape}")
print(f"Scores shape: {all_scores.shape}")
print(f"Class names: {class_names}")


plt.figure(figsize=(8,6))
for i, class_name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(all_labels == i, all_scores[:, i])
    plt.plot(recall, precision, lw=2, label=f"{class_name}")

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
