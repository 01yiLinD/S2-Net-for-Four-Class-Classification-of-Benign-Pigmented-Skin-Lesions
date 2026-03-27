import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def calculate_metrics(true_labels, predictions):
    """
    计算评估指标
    """
    accuracy = accuracy_score(true_labels, predictions) * 100
    report = classification_report(true_labels, predictions, zero_division=0)
    cm = confusion_matrix(true_labels, predictions)

    return accuracy, report, cm


def save_confusion_matrix(cm, class_names, save_path):
    """
    绘制并保存混淆矩阵
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()