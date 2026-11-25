
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "results" / "tables"

INPUT_FILES = {
    "Healthy": "fault_isolation_healthy_baseline.csv",
    "Sensor Fault": "fault_isolation_sensor_signature.csv",
    "Actuator Fault": "fault_isolation_actuator_signature.csv",
    "Model Mismatch": "fault_isolation_model_mismatch_signature.csv",
}

RESIDUALS = ["r_T", "r_Tj", "r_C", "r_u"]

THRESHOLDS = {
    "sensor_temperature": 0.50,
    "state_mismatch": 0.30,
    "actuator_flow": 0.10,
}


def load_summary(filename):
    path = TABLE_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if "residual" not in df.columns:
        raise ValueError(
            f"Missing 'residual' column in {filename}"
        )

    if df["residual"].duplicated().any():
        raise ValueError(
            f"Duplicate residual names in {filename}"
        )

    return df.set_index("residual")


def get_value(df, residual, column):
    if residual not in df.index:
        raise ValueError(f"Missing residual: {residual}")

    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

    return float(df.loc[residual, column])


def build_signature(class_name):
    filename = INPUT_FILES[class_name]
    df = load_summary(filename)

    signature = {}

    for residual in RESIDUALS:
        if class_name == "Healthy":
            column = "mean"
        elif class_name == "Sensor Fault":
            column = "faulty_mean"
        elif class_name == "Actuator Fault":
            column = "faulty_mean"
        else:
            column = "mean"

        signature[residual] = get_value(
            df, residual, column
        )

    return signature


def classify_fault(signature):
    r_t = abs(signature["r_T"])
    r_tj = abs(signature["r_Tj"])
    r_c = abs(signature["r_C"])
    r_u = abs(signature["r_u"])

    if r_u > THRESHOLDS["actuator_flow"]:
        return "Actuator Fault"

    if (
        r_t > THRESHOLDS["sensor_temperature"]
        and r_tj <= THRESHOLDS["state_mismatch"]
        and r_c <= THRESHOLDS["state_mismatch"]
    ):
        return "Sensor Fault"

    if max(r_tj, r_c) > THRESHOLDS["state_mismatch"]:
        return "Model Mismatch"

    return "Healthy"


def main():
    print("\n========== FAULT ISOLATION SIGNATURE VALIDATION ==========\n")

    results = []

    for expected_class in INPUT_FILES:
        signature = build_signature(expected_class)
        predicted_class = classify_fault(signature)

        results.append({
            "expected_class": expected_class,
            "predicted_class": predicted_class,
            "correct": expected_class == predicted_class,
            **signature,
        })

    results_df = pd.DataFrame(results)

    print("========== RESIDUAL SIGNATURES ==========\n")
    print(
        results_df[
            ["expected_class", *RESIDUALS]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\n========== CLASSIFICATION RESULTS ==========\n")
    print(
        results_df[
            ["expected_class", "predicted_class", "correct"]
        ].to_string(index=False)
    )

    correct_count = int(results_df["correct"].sum())
    total_count = len(results_df)

    print("\n========== SIGNATURE VALIDATION ==========\n")
    print(f"Correct signatures: {correct_count}/{total_count}")
    print(f"Signature consistency: {100 * correct_count / total_count:.2f}%")

    output_path = (
        TABLE_DIR / "fault_isolation_signature_validation.csv"
    )

    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
