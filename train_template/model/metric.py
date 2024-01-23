""" setup accuracy in metric file """
import torch
from sklearn.metrics import f1_score

""" implementing F1 score """

def setup_accuracy(y_true, y_pred):
    f1_score(y_true, y_pred, average="macro")
