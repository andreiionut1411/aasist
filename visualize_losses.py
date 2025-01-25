import matplotlib.pyplot as plt

losses_file = './losses/losses_multihead.txt'

with open(losses_file) as file:
    lines = file.readlines()

training_losses = list(map(float, lines[0].strip('[]\n').split(', ')))
dev_losses = list(map(float, lines[1].strip('[]\n').split(', ')))

# X-axis values for training and dev losses
training_epochs = list(range(1, len(training_losses) + 1))
dev_epochs = [1] + [i for i in range(5, 51, 5)]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot training losses as a line
plt.plot(training_epochs, training_losses, label='Training Loss', color='blue', linestyle='-', marker=None)

# Plot dev losses as points with a connecting line for visibility
plt.plot(dev_epochs, dev_losses, label='Dev Loss', color='red', linestyle='-', marker=None)

# Add labels, title, and legend
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Dev Loss Over Epochs')
plt.legend()

# Show the grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Display the plot
plt.show()
