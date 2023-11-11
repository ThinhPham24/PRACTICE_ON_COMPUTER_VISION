import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
num_epochs = 1000
learning_rate = 0.1

# Toy dataset
x_train = torch.FloatTensor([[3.3, 1.7], [4.4, 2.76], [5.5, 2.09], [6.71, 3.19], [6.93, 1.694], [4.168, 1.573], 
                    [9.779, 3.366], [6.182, 2.596], [7.59, 2.53], [2.167, 1.221], [7.042, 2.827], 
                    [10.791, 3.465], [5.313, 1.65], [7.997, 2.904], [3.1, 1.3]])

y_train = torch.LongTensor([0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0])

# Linear regression model
class Feedforward(torch.nn.Module):
        def __init__(self):
            super(Feedforward, self).__init__()
            self.input_size = 2
            self.hidden_size  = 1
            self.output_size  = 2
            self.input = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu1 = torch.nn.ReLU()
            self.fc1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu2 = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
            self.relu3 = torch.nn.ReLU()
            self.output = torch.nn.Linear(self.hidden_size, self.output_size)

        def forward(self, x):
            input = self.input(x)
            relu1 = self.relu1(input)
            hidden1 = self.fc1(relu1)
            relu2 = self.relu2(hidden1)
            hidden2 = self.fc2(relu2)
            output = self.relu3(hidden2)
            output = self.output(output)
            return output

def test(model, data, target):
    model.eval()
    correct = 0
    with torch.no_grad():
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    print('Accuracy: {}/{} - {}%'.format(correct, len(pred), (correct/len(pred))*100))

# Loss and optimizer
model=Feedforward()
print(model)
#assert()

criterion = torch.nn.MSELoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = x_train
    targets = y_train

    # if MSE lose, the ground-truth need to be modified into one-hot code.
    one_hot = torch.sparse.torch.eye(2).index_select(dim=0, index=targets)
    
    # Forward pass
    outputs = model(inputs)
    
    loss = criterion(outputs, one_hot)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    test(model, inputs, targets)

# Plot the graph
predicted = model(x_train).argmax(dim=1, keepdim=True).detach().numpy().flatten()
plt.subplot(1,2,1)
plt.title('Ground Truth')
plt.scatter(x_train[:,0], x_train[:,1], c=y_train)
plt.subplot(1,2,2)
plt.title('Predict')
plt.scatter(x_train[:,0], x_train[:,1], c=predicted)
plt.show()