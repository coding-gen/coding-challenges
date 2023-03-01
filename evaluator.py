#! /usr/bin/python3

"""
Author: Genevieve LaLonde

Evaluation of accuracy, precision, and recall in multiclass classification.
Created as part of an interview coding challenge.
"""


from sklearn import metrics


def evaluate(gold, predicted):
	print("Confusion Matrix:")
	print(metrics.confusion_matrix(gold, predicted))
	print("\nClassification Report:")
	print("\t*Ignore F1-Score since this is multiclass.")
	print(metrics.classification_report(gold, predicted, digits=3))

	print("\nIndividualized metrics:")
	accuracy = metrics.accuracy_score(gold, predicted)
	precision = metrics.precision_score(gold, predicted, average=None)
	recall = metrics.recall_score(gold, predicted, average=None)
	print(f"accuracy: {accuracy}")
	print(f"precision: {precision}")
	print(f"recall: {recall}")
	class_counts = predicted.value_counts()

	return (accuracy, precision, recall, class_counts)