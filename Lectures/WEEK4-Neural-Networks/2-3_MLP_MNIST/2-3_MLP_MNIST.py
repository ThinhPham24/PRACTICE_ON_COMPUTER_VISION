import torch 
import numpy 
import numpy as np
from torchvision.datasets import mnist
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


def data_transform(x):
    x = np.array(x, dtype = 'float32') / 255
    x = (x - 0.5) /0.5
    x = x.reshape((-1, ))
    x = torch.from_numpy(x)
    return x

# default input size is 28x28.
trainset = mnist.MNIST('./dataset/mnist', train=True, transform=data_transform, download=True)
testset = mnist.MNIST('./dataset/mnist', train = False, transform=data_transform, download=True)

# hyper-paramter setting
Epoch_Flag = 20
Train_Batch_Size_Flag = 128
Test_Batch_Size_Flag = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



train_data = DataLoader(trainset, batch_size=Train_Batch_Size_Flag, shuffle=True)
test_data = DataLoader(testset, batch_size=Test_Batch_Size_Flag, shuffle=False)

class MLP(nn.Module):
     def __init__(self):
         super(MLP, self).__init__()
         self.fc1 = nn.Linear(28*28, 500)
         self.fc2 = nn.Linear(500, 250)
         self.fc3 = nn.Linear(250, 125)
         self.fc4 = nn.Linear(125, 10)
         
     def forward(self, x):
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = F.relu(self.fc3(x))
         x = self.fc4(x)
         return x
mlp = MLP()
mlp.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(mlp.parameters(), 1e-3)


for epoch in range(Epoch_Flag):
    train_loss = 0  
    train_acc = 0
    mlp.train()
    for step, (batch_img, label) in enumerate(train_data):
        batch_img = batch_img.to(device)
        label = label.to(device)
        
        predict = mlp(batch_img)
        loss = criterion(predict, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step%100==0:
            print('Epoch:{}, Steps:{}, Losses:{}'.format(epoch, step, loss))

GroundTruth = []
Predict_Test = []
for index, (test_img, test_label) in enumerate(test_data):

    GroundTruth += list(test_label.cpu().detach().numpy())

    test_img = test_img.to(device)
    predict_test = mlp(test_img)
    predict_test_probability = predict_test.cpu().detach().numpy()
    predict_test_label = np.argmax(predict_test_probability, axis=1)

    Predict_Test += list(predict_test_label)


Compare = np.array(GroundTruth) - np.array(Predict_Test)
print('Performance={:.2f}%'.format((len(np.where(Compare==0)[0])/len(GroundTruth))*100))





