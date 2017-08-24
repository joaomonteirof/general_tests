from __future__ import print_function
import argparse
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from Utils import data_loader, batch_generator

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

(x_train, y_train), (x_valid, y_valid) = data_loader('mnist')
print(x_train.size())
print(y_train.size())
print(x_valid.size())
print(y_valid.size())

class MLP_MNIST(nn.Module):
	def __init__(self):
		super(MLP_MNIST, self).__init__()
		self.den1 = nn.Linear(784, 128)
		self.den2 = nn.Linear(128, 10)

	def forward(self, x):
		x = x.view(-1, 784)
		x = self.den1(x)
		x = F.dropout(x)
		x = F.relu(x)
		x = self.den2(x)
		return F.softmax(x)

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
		return F.softmax(x)

#model = MLP_MNIST()
model = Net()
if args.cuda:
	model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
	model.train()
	train_loader = batch_generator(x_train, y_train, args.batch_size)

	for batch_idx, (data, target) in enumerate(train_loader):
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model(data)
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), x_train.size()[0], 100. * batch_idx * len(data) / x_train.size()[0], loss.data[0]))

def test(epoch):
	model.eval()
	test_loss = 0
	correct = 0
	test_loader = batch_generator(x_valid, y_valid, args.batch_size)
	for data, target in test_loader:
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	test_loss = test_loss
	test_loss /= len(test_loader) # loss function already averages over batch size
	len(test_loader)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, x_valid.size()[0], 100. * correct / x_valid.size()[0]))

def countParams(myModel):
	counter = 0
	for param in myModel.parameters():
		acc =1
		for value in param.size():
			acc*=value
		counter+=acc
	print(counter)
	return counter

def tensorSize(myTensor):
	acc =1

	for value in myTensor.size():
		acc*=value

	return acc

def setParams(myModel):
	totalNumPar = countParams(myModel)

	paramsToInclude = numpy.ones(totalNumPar)

	paramsCopy = torch.from_numpy(paramsToInclude)

	for param in myModel.parameters():

		numPar = tensorSize(param)
		parSize = param.size()
		paramsubset = paramsCopy[0:numPar]
		param_size = param.size()
		param.data = paramsubset.view(param_size)
		try:
			paramsCopy = paramsCopy[numPar:]
		except ValueError:
			break	

def parameters(myModel):
	for param in myModel.parameters():
		print (param)

#parameters(model)
#countParams(model)
#setParams(model)
#parameters(model)
#countParams(model)

#train(1)

for epoch in range(1, args.epochs + 1):
	train(epoch)
test(epoch)
