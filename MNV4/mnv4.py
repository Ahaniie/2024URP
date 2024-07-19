import torch
import matplotlib.pyplot as plt
import numpy as np
from mobilenetv4 import MobileNetV4

# Support ['MobileNetV4ConvSmall', 'MobileNetV4ConvMedium', 'MobileNetV4ConvLarge']
# Also supported ['MobileNetV4HybridMedium', 'MobileNetV4HybridLarge']
model = MobileNetV4("MobileNetV4ConvSmall")

# Check the trainable params
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {total_params}")

# Check the model's output shape
print("Check output shape ...")
x = torch.rand(1, 3, 224, 224)
y = model(x)
for i in y:
    print(i.shape)

print(y[4])

# Create a tensor with shape [1, 1280, 1, 1]
tensor = y[4]

# Reshape the tensor to [1280]
tensor_reshaped = tensor.detach().view(-1).numpy()

# Visualize the tensor values as a bar plot
plt.figure(figsize=(20, 5))
plt.bar(np.arange(len(tensor_reshaped)), tensor_reshaped)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Visualization of 3D Tensor with Shape [1, 1280, 1, 1]')
plt.show()