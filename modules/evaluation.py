from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score
)



def evaluate_model(
    y_test,
    prediction
):


    matrix = confusion_matrix(
        y_test,
        prediction
    )


    return {


        "confusion_matrix":
        matrix.tolist()

    }