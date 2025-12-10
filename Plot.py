import re
import os
import matplotlib.pyplot as plt

LOG_FILE = "/workspace/ese-3060-project/Part_2_logs/Actual Logs with 8x PU/Baseline 1/21e49b78-d6ac-44d7-b708-99bfea090222.txt"

# Extract directory so images go in the same folder
OUTPUT_DIR = os.path.dirname(LOG_FILE)

# Patterns for parsing
train_pattern = re.compile(r"step:(\d+)/\d+.*?train_loss:([0-9.]+).*?train_time:(\d+)ms")
val_pattern   = re.compile(r"step:(\d+)/\d+.*?val_loss:([0-9.]+).*?train_time:(\d+)ms")

train_steps = []
train_losses = []
train_times = []

val_steps = []
val_losses = []
val_times = []

with open(LOG_FILE, "r") as f:
    for line in f:
        # training loss
        m = train_pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            t_ms = int(m.group(3))

            train_steps.append(step)
            train_losses.append(loss)
            train_times.append(t_ms)

        # validation loss
        m = val_pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            t_ms = int(m.group(3))

            val_steps.append(step)
            val_losses.append(loss)
            val_times.append(t_ms)

print("Train points:", len(train_losses))
print("Val points:", len(val_losses))

TARGET_LOSS = 3.28

# ----------------------------------------------------------------------
# 1. LOSS VS STEP
# ----------------------------------------------------------------------

plt.figure(figsize=(12,7))
plt.plot(train_steps, train_losses, label="Train Loss", linewidth=1.5)
plt.plot(val_steps, val_losses, 'o-', label="Validation Loss", markersize=6)

plt.axhline(TARGET_LOSS, color='gray', linestyle='--', linewidth=1, label="Target Loss = 3.28")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Step-Based)")
plt.grid(True)
plt.legend()
plt.ylim(3, 6)
plt.tight_layout()

step_plot_path = os.path.join(OUTPUT_DIR, "training_curve.png")
plt.savefig(step_plot_path, dpi=300)
print("Saved step plot to:", step_plot_path)

# ----------------------------------------------------------------------
# 2. LOSS VS TIME
# ----------------------------------------------------------------------

train_times_s = [t / 1000 for t in train_times]
val_times_s = [t / 1000 for t in val_times]

plt.figure(figsize=(12,7))
plt.plot(train_times_s, train_losses, label="Train Loss", linewidth=1.5)
plt.plot(val_times_s, val_losses, 'o-', label="Validation Loss", markersize=6)

plt.axhline(TARGET_LOSS, color='gray', linestyle='--', linewidth=1, label="Target Loss = 3.28")

plt.xlabel("Time (seconds)")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss (Time-Based)")
plt.grid(True)
plt.legend()
plt.ylim(3, 6)
plt.xlim(0, 900)
plt.tight_layout()

time_plot_path = os.path.join(OUTPUT_DIR, "training_curve_time.png")
plt.savefig(time_plot_path, dpi=300)
print("Saved time plot to:", time_plot_path)