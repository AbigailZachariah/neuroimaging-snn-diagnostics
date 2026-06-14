from src.anomaly import anomaly_score, is_anomaly

probs1 = [0.90, 0.05, 0.03, 0.02]

print("Case 1")
print("Score:", anomaly_score(probs1))
print("Anomaly:", is_anomaly(probs1))

print()

probs2 = [0.30, 0.25, 0.20, 0.25]

print("Case 2")
print("Score:", anomaly_score(probs2))
print("Anomaly:", is_anomaly(probs2))