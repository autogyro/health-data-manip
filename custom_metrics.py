# Collection of utilities to define custom metrics and scores Capstone Project
# Udacity's Machie Learning Nanodegree Certification
# @Juan E. Rolon
# https://github.com/juanerolon


def get_pred_array(probas):
    """Returns an array of prediction vectors given a set of probabilities.
    The probabilities are taking a threshold values. The prediction vectors
    are cast as either booleans or binary values"""

    scores = np.sort(probas)
    pred_array = []
    for val in scores:
        pred_thr = probas >= val
        pred_thr = pred_thr.astype(int)
        pred_array.append(pred_thr)

    return pred_array

def conf_matrix(y_true, y_pred):
    """Returns the false positve rate (FPR) and true positive rate (TPR) given
    a set of labels (y_true) and predictions(y_pred)"""

    TP, FP, TN, FN = 0,0,0,0
    for i, true_val in enumerate(y_true):
        if (y_pred[i] == 1 and true_val ==1 ):
            TP +=1
        elif (y_pred[i] == 1 and true_val ==0):
            FP +=1
        elif (y_pred[i] == 0 and true_val ==0):
            TN +=1
        elif (y_pred[i] == 0 and true_val ==1 ):
            FN +=1
        else:
            raise Exception("conf_matrix() error. "
                            "Invalid confusion matrix conditions.\n "
                            "y_true, y_pred must be integers or booleans.")

    TPR = float(TP / (TP + FN))
    FPR = float(FP / (FP + TN))

    return FPR, TPR

def auc_score(labels, probas):
    """Returns the area under the ROC curve given a set of labels and the probabilities
    of observing them according to the output of a Statistical or ML model"""
    pred_array = get_pred_array(probas)
    ROC_POINTS = []
    for pred_vec in pred_array:
        ROC_POINTS.append(conf_matrix(labels, pred_vec))
    L = []
    for k in reversed(ROC_POINTS):
        L.append(k)
    ROC_POINTS = L
    sum = 0.0
    for k in range(1, len(ROC_POINTS)):
        sum += 0.5 * (ROC_POINTS[k][1] + ROC_POINTS[k - 1][1])*(ROC_POINTS[k][0] - ROC_POINTS[k-1][0])

    return sum
