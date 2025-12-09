import re
import matplotlib.pyplot as plt

LOG_FILE = "/workspace/ese-3060-project/first SXM 4x logs/dcb72c7e-0fd4-4837-bd5e-d5577351460c.txt"   # change if needed

# Patterns for parsing
train_pattern = re.compile(r"step:(\d+)/\d+.*?train_loss:([0-9.]+)")
val_pattern   = re.compile(r"step:(\d+)/\d+.*?val_loss:([0-9.]+)")

train_steps = []
train_losses = []
val_steps = []
val_losses = []

with open(LOG_FILE, "r") as f:
    for line in f:
        # training loss
        m = train_pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            train_steps.append(step)
            train_losses.append(loss)

        # validation loss
        m = val_pattern.search(line)
        if m:
            step = int(m.group(1))
            loss = float(m.group(2))
            val_steps.append(step)
            val_losses.append(loss)

print("Train points:", len(train_losses))
print("Val points:", len(val_losses))

plt.figure(figsize=(12,7))

# TRAINING CURVE
plt.plot(train_steps, train_losses, label="Train Loss", linewidth=1.5)

# VALIDATION CURVE
plt.plot(val_steps, val_losses, 'o-', label="Validation Loss", markersize=6)

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.legend()

# >>> SET Y-AXIS RANGE HERE <<<
plt.ylim(3, 8)

plt.tight_layout()
plt.savefig("training_curve.png", dpi=300)