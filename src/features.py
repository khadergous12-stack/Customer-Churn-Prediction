def add_features(df):
    df = df.copy()
    df["avg_charge_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["high_spender"] = (df["MonthlyCharges"] > 70).astype(int)
    return df