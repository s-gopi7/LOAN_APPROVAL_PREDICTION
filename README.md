# RBI-Compliant Loan Approval

An RBI-aligned loan approval prediction project that combines:
- A machine learning training pipeline (`model.py`)
- A Flask prediction API (`App.py`)
- Frontend dashboards (`frontend.html`, `Model_dash.html`)

The model uses domain-driven feature engineering and a stacking ensemble to estimate approval probability and risk level.

## Project Structure

- `/model.py` — trains and evaluates the ML model, then saves `loan_model.pkl`
- `/App.py` — Flask backend exposing prediction and health endpoints
- `/rbi_loan_dataset.csv` — dataset used for training
- `/loan_model.pkl` — serialized trained model artifact used by backend
- `/loan_model_evaluation.png` — model evaluation chart output
- `/optimized_ml_results/` — additional model comparison and optimization artifacts

## Requirements

- Python 3.9+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

From the repository root:

```bash
python model.py
```

This generates/updates:
- `loan_model.pkl`
- `loan_model_evaluation.png`

## Run the Backend API

From the repository root:

```bash
python App.py
```

Server runs at: `http://localhost:5000`

## API Endpoints

### Health Check

- `GET /health`
- Example:

```bash
curl http://localhost:5000/health
```

### Loan Prediction

- `POST /predict`
- Content-Type: `application/json`
- Example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "employment_type": "Salaried",
    "years_employed": 8,
    "annual_income_lakhs": 18,
    "dependents": 2,
    "state": "Maharashtra",
    "property_area": "Urban",
    "cibil_score": 780,
    "credit_history_years": 9,
    "no_of_existing_loans": 1,
    "existing_emi_lakhs": 0.10,
    "previous_defaults": 0,
    "bankruptcies": 0,
    "loan_type": "Home Loan",
    "loan_amount_lakhs": 40,
    "tenure_months": 240,
    "interest_rate_pct": 8.7,
    "ltv_ratio": 75,
    "collateral_value_lakhs": 55
  }'
```

The API returns approval decision, probability, computed indicators (EMI/FOIR/etc.), and risk factors.

## Notes

- Ensure `loan_model.pkl` exists before running the backend.
- If missing, run `python model.py` first.
