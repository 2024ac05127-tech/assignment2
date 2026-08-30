import pandas as pd

df = pd.read_csv("monitoring/performance_data.csv")

total = len(df)
correct = (df["predicted_label"] == df["true_label"]).sum()
incorrect = total - correct
accuracy = correct / total

print("======================================")
print("POST-DEPLOYMENT MODEL PERFORMANCE")
print("======================================")
print(f"Total predictions : {total}")
print(f"Correct           : {correct}")
print(f"Incorrect         : {incorrect}")
print(f"Accuracy          : {accuracy:.2%}")
print("======================================")

if accuracy < 0.70:
    raise SystemExit("Performance test FAILED: accuracy below 70%")

print("Performance test PASSED")