from flask import Flask, jsonify, request
import joblib
from functions import *

app = Flask(__name__)

loan_app_status_classifier = joblib.load("models/loan_app_status_classifier.pkl")
grade_classifier = joblib.load("models/grade_classifier.pkl")
subgrade_classifiers = {
    'A': joblib.load("models/A_subgrades_classifier.pkl"),
    'B': joblib.load("models/B_subgrades_classifier.pkl"),
    'C': joblib.load("models/C_subgrades_classifier.pkl"),
    'D': joblib.load("models/D_subgrades_classifier.pkl"),
    'E': joblib.load("models/E_subgrades_classifier.pkl"),
    'F': joblib.load("models/F_subgrades_classifier.pkl"),
    'G': joblib.load("models/G_subgrades_classifier.pkl"),
}

interest_rate_regressor = joblib.load("models/int_rate_regressor.pkl")

grade_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
subgrade_map = {}

additive = 0
for k in grade_map:
    for i in range(5):
        subgrade_map[additive + i] = grade_map[k] + str(i + 1)
    additive += 5


@app.route('/loan/status', methods=['POST'])
def loan_status():
    content = request.json
    inputs = pd.DataFrame(columns=['amount_requested', 'date', 'loan_purpose', 'fico_score', 'dti_ratio',
                                   'state', 'emp_length'])
    inputs = inputs.append(content, ignore_index=True)
    res = loan_app_status_classifier.predict(inputs)
    result = 'Accepted'
    if res[0] == 0:
        result = 'Rejected'

    return jsonify({'result': result})


@app.route('/loan/ranking', methods=['POST'])
def loan_ranking():
    grade_prediction_columns = ['fico_range_low', 'fico_range_high', 'term', 'all_util',
                                'bc_open_to_buy', 'bc_util', 'revol_util', 'percent_bc_gt_75',
                                'total_bc_limit', 'total_rev_hi_lim', 'acc_open_past_24mths',
                                'inq_last_12m', 'il_util', 'inq_last_6mths', 'num_tl_op_past_12m',
                                'open_il_24m', 'inq_fi', 'verification_status', 'open_il_12m',
                                'tot_hi_cred_lim', 'loan_amnt', 'dti', 'earliest_cr_line',
                                'pct_tl_nvr_dlq', 'issue_d', 'annual_inc', 'emp_length', 'purpose']

    content = request.json

    grade_prediction_inputs = pd.DataFrame(columns=grade_prediction_columns)
    grade_prediction_inputs = grade_prediction_inputs.append({key: content[key] for key in grade_prediction_columns},
                                                             ignore_index=True)

    grade_res = grade_classifier.predict(grade_prediction_inputs)
    grade = grade_map[grade_res[0]]

    subgrade_prediction_columns = ['term', 'fico_range_low', 'fico_range_high',
                                   'percent_bc_gt_75', 'all_util', 'bc_util', 'revol_util',
                                   'verification_status', 'bc_open_to_buy', 'acc_open_past_24mths',
                                   'inq_last_12m', 'inq_last_6mths', 'num_tl_op_past_12m', 'open_il_24m',
                                   'open_rv_24m', 'open_il_12m', 'inq_fi', 'total_bc_limit', 'il_util',
                                   'total_rev_hi_lim', 'loan_amnt', 'issue_d', 'earliest_cr_line', 'dti',
                                   'tot_hi_cred_lim', 'pct_tl_nvr_dlq', 'annual_inc',
                                   'mths_since_last_delinq', 'emp_length', 'purpose']

    subgrade_prediction_inputs = pd.DataFrame(columns=subgrade_prediction_columns)
    subgrade_prediction_inputs = subgrade_prediction_inputs.append(
        {key: content[key] for key in subgrade_prediction_columns}, ignore_index=True)

    subgrade_res = subgrade_classifiers[grade].predict(subgrade_prediction_inputs)
    subgrade = subgrade_map[subgrade_res[0]]

    interest_rate_prediction_columns = ['fico_range_low', 'fico_range_high', 'term', 'all_util',
                                        'bc_open_to_buy', 'bc_util', 'revol_util', 'percent_bc_gt_75',
                                        'total_bc_limit', 'total_rev_hi_lim', 'acc_open_past_24mths',
                                        'inq_last_12m', 'inq_fi', 'verification_status', 'il_util',
                                        'open_il_24m', 'inq_last_6mths', 'num_tl_op_past_12m', 'open_il_12m',
                                        'issue_d', 'loan_amnt', 'dti', 'mo_sin_old_rev_tl_op',
                                        'tot_hi_cred_lim', 'mths_since_last_delinq', 'earliest_cr_line',
                                        'pct_tl_nvr_dlq', 'annual_inc', 'emp_length', 'purpose']

    interest_rate_prediction_inputs = pd.DataFrame(columns=interest_rate_prediction_columns)
    interest_rate_prediction_inputs = interest_rate_prediction_inputs.append(
        {key: content[key] for key in interest_rate_prediction_columns}, ignore_index=True)

    interest_rate_res = interest_rate_regressor.predict(interest_rate_prediction_inputs)
    interest_rate = round(interest_rate_res[0], 2)

    return jsonify({
        'grade': grade,
        'subgrade': subgrade,
        'interest_rate': interest_rate,
    })


if __name__ == '__main__':
    app.run(debug=False, port=80, host="0.0.0.0")
