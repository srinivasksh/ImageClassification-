%matplotlib inline

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Training settings
batch_size =  1024
test_batch_size = 1000
epochs = 10
lr = 1
m_iter = 5
h_size = 5
seed = 1
log_interval = 10

torch.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()

optimizer = optim.LBFGS(model.parameters(),lr=lr,max_iter=m_iter,history_size=h_size)


def train(epoch):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		def closure():
			optimizer.zero_grad()
			output = model(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			return loss
		optimizer.step(closure)
        #if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss))


def test(epoch):
	total_train = len(train_loader.dataset)
	total_test = len(test_loader.dataset)
	correct_train = 0
	correct_test = 0
	train_loss = 0
	test_loss = 0
    # TODO: Test the model on the test-set and report the loss and accuracy.
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = Variable(data), Variable(target)
		pred_target_tmp = model(data).data
		pred_target = pred_target_tmp.argmax(dim=1, keepdim=True)
		correct_train += pred_target.eq(target.view_as(pred_target)).sum().item()
		train_loss += F.nll_loss(pred_target_tmp, target, reduction='sum').item()
	print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,train_loss/total_train))
	print('Train Epoch: {} \tClassification Error: {:.6f}'.format(epoch,1-correct_train/total_train))
	
    # TODO: Test the model on the test-set and report the loss and accuracy.
	for batch_idx, (data, target) in enumerate(test_loader):
		data, target = Variable(data), Variable(target)
		pred_target_tmp = model(data).data
		pred_target = pred_target_tmp.argmax(dim=1, keepdim=True)
		correct_test += pred_target.eq(target.view_as(pred_target)).sum().item()
		test_loss += F.nll_loss(pred_target_tmp, target, reduction='sum').item()
	print('Test Epoch: {} \tLoss: {:.6f}'.format(epoch,test_loss/total_test))
	print('Test Epoch: {} \tClassification Error: {:.6f}'.format(epoch,1-correct_test/total_test))
	
	return(train_loss/total_train,test_loss/total_test,1-correct_train/total_train,1-correct_test/total_test)

train_err_vec = []
test_err_vec = []
train_loss_vec = []
test_loss_vec = []
for epoch in range(1, epochs + 1):
	train(epoch)
	train_loss,test_loss,train_err,test_err = test(epoch)
	train_loss_vec.append(train_loss)
	test_loss_vec.append(test_loss)
	train_err_vec.append(train_err)
	test_err_vec.append(test_err)

## Error plot
plt.plot(epoch_vec,train_err_vec,'r-',label="Train Error")
plt.plot(epoch_vec,test_err_vec,'b-',label="Test Error")
title_name = "Train Error and Test Error vs Epochs"
plt.title(title_name)
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show()

## Loss plot
plt.plot(epoch_vec,train_loss_vec,'r-',label="Train Loss")
plt.plot(epoch_vec,test_loss_vec,'b-',label="Test Loss")
title_name = "Train Loss and Test Loss vs Epochs"
plt.title(title_name)
plt.legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show()