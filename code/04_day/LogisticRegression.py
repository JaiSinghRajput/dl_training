import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)

def run(show_plot=True, verbose=True, random_state=42):
    data = load_breast_cancer()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)  

    model = LogisticRegression(max_iter=1000, solver='lbfgs', C=1.0)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),  
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=4)
    }

    if verbose:
        print('----- Logistic Regression -----')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}") 
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"Classification Report:\n{metrics['classification_report']}")
        print(f"ROC & AUC: {metrics['auc']:.4f}")

    if show_plot:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("LR - ROC Curve")
        plt.legend()
        plt.show()

    return metrics

if __name__ == "_main_":
    run(show_plot=True, verbose=True)