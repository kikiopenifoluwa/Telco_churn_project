import pandas as pd
def data_preprocessor(df):

    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Drop unnecessary columns
    X = df.drop(
        columns=[
            "customerID",
            "PaymentMethod",
            "PaperlessBilling",
            "Churn"
        ],
        errors="ignore"
    )

    # Binary encoding
    X["gender"] = X["gender"].apply(
        lambda x: 1 if x == "Male" else 0
    )

    yes_no_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Partner",
        "Dependents",
        "PhoneService"
    ]

    for col in yes_no_cols:
        X[col] = X[col].apply(
            lambda x: 1 if x == "Yes" else 0
        )

    # Categorical encoding
    X["InternetService"] = X["InternetService"].apply(
        lambda x:
            2 if x == "Fiber optic"
            else 1 if x == "DSL"
            else 0
    )

    X["Contract"] = X["Contract"].apply(
        lambda x:
            1 if x == "Month-to-month"
            else 2 if x == "One year"
            else 3
    )

    return X