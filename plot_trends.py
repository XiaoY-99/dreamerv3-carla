import json
import matplotlib.pyplot as plt

# Replace with your actual log file path
log_file = "/home/meihan/dreamerv3_carla/logs_carla/metrics.jsonl/metrics.jsonl"

steps = []
scores = []
returns = []

with open(log_file, "r") as f:
    for line in f:
        try:
            data = json.loads(line)
            if "episode/score" in data:
                steps.append(data["step"])
                scores.append(data["episode/score"])
            elif "train/ret" in data:
                returns.append((data["step"], data["train/ret"]))
        except json.JSONDecodeError:
            continue

# Plot
plt.figure(figsize=(10, 5))

if scores:
    plt.plot(steps, scores, label="Episode Score", alpha=0.7)
if returns:
    r_steps, r_vals = zip(*returns)
    plt.plot(r_steps, r_vals, label="Train Return", alpha=0.7)

plt.xlabel("Step")
plt.ylabel("Score / Return")
plt.title("Training Performance")
plt.legend()
plt.grid(True)
plt.show()
