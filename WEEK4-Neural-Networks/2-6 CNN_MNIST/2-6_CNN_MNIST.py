import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch.nn.functional as F
import numpy as np
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # the default parameters: padding=0, and stride=1, please check the pytroch tutorial https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2, stride=1)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0, stride=1) 
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        
        # flatten the dimension to fit fc layer's input dimension
        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

criterion = nn.CrossEntropyLoss().to(device)

def load_data(train_batch_size, test_batch_size):
    # Fetch training data: total 60000 samples
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((32, 32)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True)

    # Fetch test data: total 10000 samples
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)

    return (train_loader, test_loader)


model = LeNet()
model = model.to(device)

lr = 0.01
momentum=0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

train_batch_size = 128
test_batch_size = 16
train_loader, test_loader = load_data(train_batch_size, test_batch_size)

epochs = 10
log_interval = 100


def train(model, optimizer, epoch, train_loader, log_interval):
    # State that you are training the model
    model.train()

    # define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Iterate over batches of data
    for batch_idx, (data, target) in enumerate(train_loader):
        # Wrap the input and target output in the `Variable` wrapper
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        # Clear the gradients, since PyTorch accumulates them
        optimizer.zero_grad()

        # Forward propagation
        output = model(data)

        loss = loss_fn(output, target)

        # Backward propagation
        loss.backward()

        # Update the parameters(weight,bias)
        optimizer.step()

        # print log
        if batch_idx % log_interval == 0:
            print('Train set, Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                loss.item()))

for epoch in range(1, epochs + 1):
    train(model, optimizer, epoch, train_loader, log_interval=log_interval)

GroundTruth = []
Predict_Test = []
for index, (test_img, test_label) in enumerate(test_loader):

    GroundTruth += list(test_label.cpu().detach().numpy())

    test_img = test_img.to(device)
    predict_test = model(test_img)
    predict_test_probability = predict_test.cpu().detach().numpy()
    predict_test_label = np.argmax(predict_test_probability, axis=1)
   

    Predict_Test += list(predict_test_label)

Compare = np.array(GroundTruth) - np.array(Predict_Test)
print('Performance={:.2f}%'.format((len(np.where(Compare==0)[0])/len(GroundTruth))*100))

Confusion_Collect = {'Predict/GroundTruth':[]}
for idx in range(10): Confusion_Collect.setdefault('GT_{}'.format(idx+1), [])

for idx in range(10):
    pred_result_indices = np.where(np.array(Predict_Test)==idx)[0]

    # Compute the number of preditct<->GT data
    Confusion_Collect['Predict/GroundTruth'].append('Pred_{}'.format(idx+1))
    for GT_idx in range(10):
        Confusion_Collect['GT_{}'.format(GT_idx+1)].append(len(np.where(np.array(GroundTruth)[pred_result_indices]==GT_idx)[0]))

Confusion_Collect['Predict/GroundTruth'].append('Total Performance: {}%'.format((len(np.where(Compare==0)[0])/len(GroundTruth))*100))

LIST = ['Predict/GroundTruth']
for idx in range(10): LIST.append('GT_{}'.format(idx+1))
Confusion_Collect = pd.DataFrame.from_dict(Confusion_Collect, orient='index').transpose()
Confusion_Collect = Confusion_Collect.reindex(columns=LIST)
Confusion_Collect.to_csv('MultiClass_Confusion_Matrix.csv', index=False)