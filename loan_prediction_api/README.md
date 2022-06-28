# Loan prediction API

The API exposes the following endpoints:

## POST /loan/status

Returns the loan application status (accepted/rejected).

Accepts JSON, expected payload follows the following format:

```json
{
    "amount_requested": 30000.0,
    "date": "2018-12-01",
    "loan_purpose": "credit_card_refinancing",
    "fico_score": 722.0,
    "dti_ratio": 8.21,
    "emp_length": "10+ years",
    "state": "NJ"
}
```

Example response:

```json 
{
    "result": "Accepted"
}
```

## POST /loan/ranking

Return the loan grade, sub-grade and interest rate.

Accepts JSON, expected payload follows the following format:

```json
{
    "fico_range_low": 700.0,
    "fico_range_high": 704.0,
    "term": "36 months",
    "all_util": 50.0,
    "bc_open_to_buy": 12313.0,
    "bc_util": 63.9,
    "revol_util": 58.3,
    "percent_bc_gt_75": 50.0,
    "total_bc_limit": 34100.0,
    "total_rev_hi_lim": 42600.0,
    "acc_open_past_24mths": 0.0,
    "inq_last_12m": 1.0,
    "il_util": 28.0,
    "inq_last_6mths": 1.0,
    "num_tl_op_past_12m": 0.0,
    "open_il_24m": 0.0,
    "inq_fi": 0.0,
    "verification_status": "Source Verified",
    "open_il_12m": 0.0,
    "tot_hi_cred_lim": 263533.0,
    "loan_amnt":40000.0,
    "dti": 13.22,
    "earliest_cr_line": "Dec-2001",
    "pct_tl_nvr_dlq": 92.3,
    "issue_d": "2018-10-01",
    "annual_inc": 100000.0,
    "emp_length":"6 years",
    "purpose": "debt_consolidation",
    "open_rv_24m": 6.0,
    "mths_since_last_delinq": 40.0,
    "mo_sin_old_rev_tl_op": 90.0
}
```

Example response:

```json
{
    "grade": "B",
    "interest_rate": 12.69,
    "subgrade": "B2"
}
```

## Docker Image

Docker image available at: obolinyte/loan-prediction-api:latest