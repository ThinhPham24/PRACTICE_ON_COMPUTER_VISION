import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# Hyper-parameters
num_epochs = 50
learning_rate = 0.001
# Set initialize parameter y = ax + b
a = 0.5
b = 1

# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09]], dtype=np.float32)

# Linear regression model
model = nn.Linear(1, 1)
# Initialize parameter
model.weight.data.fill_(a)
model.bias.data.fill_(b)

# Define Loss 
criterion = nn.MSELoss()
# Define optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
print('Initial Function: y = {:.2f}x + {:.2f}'.format(a, b))
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
	# Get the model parameters (a:slope, b:bias)
    [a, b] = model.parameters()
    #print('Target:{}'.format(y_train))
    #print('Predict:{}'.format(outputs))
    print ('Epoch [{}/{}], Loss: {:.4f}, New_a:{:.2f}, New_b:{:.2f}, a_gradient:{}, b_gradient:{}'.format(epoch+1, num_epochs, loss.item(),a.data[0][0], b.data[0],model.weight.grad,model.bias.grad))
	# Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Plot the graph
    predicted = model(torch.from_numpy(x_train)).detach().numpy()
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predicted, label='Fitted line')
    plt.axis([min(x_train)-1, max(x_train)+1, min(y_train)-3, max(y_train)+3])
    plt.draw()
    if epoch+1 == num_epochs:
        [a, b] = model.parameters()
        print('Final Function: y = {:.2f}x + {:.2f}'.format(a.data[0][0], b.data[0]))
        plt.ioff()
        plt.show()
    plt.pause(0.05)
    plt.clf()