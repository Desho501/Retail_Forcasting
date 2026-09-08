import pandas as pd
import numpy as np

df = pd.read_csv("../outputs/test_predictions.csv") 

df["absolute_error"] = (
    df["actual_units_sold"] - df["predicted_units_sold"]
).abs()

def segmented_error(data, group_column):
    result = (
        data.groupby(group_column, as_index=False)
        .agg(
            observations=("actual_units_sold", "size"),
            actual_units=("actual_units_sold", "sum"),
            predicted_units=("predicted_units_sold", "sum"),
            MAE=("absolute_error", "mean"),
            total_absolute_error=("absolute_error", "sum")
        )
    )

    result["WAPE_percent"] = np.where(
        result["actual_units"].abs() > 0,
        result["total_absolute_error"] /
        result["actual_units"].abs() * 100,
        np.nan
    )

    result["MAE"] = result["MAE"].round(4)
    result["WAPE_percent"] = result["WAPE_percent"].round(4)

    return result

category_error = segmented_error(df, "cat_id")
store_error = segmented_error(df, "store_id")

category_error.to_csv("../outputs/category_error_analysis.csv", index=False)
store_error.to_csv("../outputs/store_error_analysis.csv", index=False)

print("\nError by Category")
print(category_error.to_string(index=False))

print("\nError by Store")
print(store_error.to_string(index=False))