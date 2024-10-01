from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, jaccard_score, cohen_kappa_score
from sklearn.metrics import confusion_matrix
import json
from tqdm import tqdm

def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_mortality_sonnet_base_zeroshot_test_results_multithreaded.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_test_ulti.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_test_ulti_no_shuffled.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_test_ulti_no_shuffled_x.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_mortality_ehr_coagent_results_round_2.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_ulti_shuffled_x_new_wogt_results.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_mortality_ulti_shuffled_x_new_wogt_results.json"
# result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_ulti_shuffled_x_new_wogt_results.json"
result_path = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_mortality_straight_ft.json"

result = json.load(open(result_path, "r"))

if "zeroshot" in result_path or "fewshot" in result_path:
    y_true_all = []
    y_pred_all = []

    for key, patient in tqdm(result.items()):
        ground_truth = patient["ground_truth"]
        prediction = patient["reasoning_and_prediction"]
        if "\n# Prediction #  \n" in prediction:
            # y_true_all.append(ground_truth[0])
            y_true_all.append(str(ground_truth))
            y_pred_all.append(prediction.split("\n# Prediction #  \n")[1][0])
        else:
            y_true_all.append(str(ground_truth))
            y_pred_all.append("0")
    
    print(f"Number of samples: {len(y_true_all)}")
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    sensitivity, specificity = calculate_sensitivity_specificity(y_true_all, y_pred_all)

    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    
elif "coagent" in result_path:
    y_true_all = []
    y_pred_all = []

    for key, patient in tqdm(result.items()):
        ground_truth = patient["ground_truth"]
        prediction = patient["prediction"]
        if "\n# Prediction # \n" in prediction:
            # y_true_all.append(ground_truth[0])
            y_true_all.append(str(ground_truth))
            y_pred_all.append(prediction.split("\n# Prediction # \n")[1][0])
        else:
            y_true_all.append(str(ground_truth))
            y_pred_all.append("0")
    
    print(f"Number of samples: {len(y_true_all)}")
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    sensitivity, specificity = calculate_sensitivity_specificity(y_true_all, y_pred_all)

    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    
elif "straight" in result_path or "multitask" in result_path:
    y_true_all = []
    y_pred_all = []

    for key, patient in tqdm(result.items()):
        ground_truth = patient["ground_truth"]
        prediction = patient["reasoning_and_prediction"]
        if len(patient["reasoning_and_prediction"]) == 1:
            y_true_all.append(ground_truth)
            y_pred_all.append(prediction)
        else:
            y_true_all.append(ground_truth)
            y_pred_all.append("0" if ground_truth == "1" else "1")
    
    print(f"Number of samples: {len(y_true_all)}")
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    sensitivity, specificity = calculate_sensitivity_specificity(y_true_all, y_pred_all)

    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    
else:
    y_true_all = []
    y_pred_all = []
    cnt = 0
    for key, patient in tqdm(result.items()):
        ground_truth = patient["ground_truth"]
        prediction = patient["reasoning_and_prediction"]
        if "\n# Prediction #\n" in prediction:
            y_true_all.append(ground_truth.split("\n# Prediction #\n")[-1].split(" ")[0])
            y_pred_all.append(prediction.split("\n# Prediction #\n")[-1].split(" ")[0])
        else:
            cnt += 1
            gt = ground_truth.split("\n# Prediction #\n")[-1].split(" ")[0]
            y_true_all.append(gt)
            y_pred_all.append("0")
            # y_pred_all.append("1" if gt == "0" else "0")
        
    acc = accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=1)
    sensitivity, specificity = calculate_sensitivity_specificity(y_true_all, y_pred_all)

    print(f"Accuracy: {acc}")
    print(f"F1: {f1}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")
    print(f"Number of samples with no prediction: {cnt}")