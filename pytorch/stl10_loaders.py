import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter

def imshow(img):
	img = img / 2. + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

#testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)

#testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

unlabeledset=torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)

unlabeledloader = torch.utils.data.DataLoader(unlabeledset, batch_size=4, shuffle=False, num_workers=2)

print(type(unlabeledset.__getitem__(10)[0]))
print(len(unlabeledset.__getitem__(10)[0]))

print(unlabeledset.__getitem__(10)[1])

conv = transforms.ToPILImage()
unconv = transforms.ToTensor()

a = unlabeledset.__getitem__(10)[0] / 2. + 0.5
a = conv(a)
a = a.filter(ImageFilter.GaussianBlur(radius=0.1))

b = unconv(a)

print(type(a))
print(type(b))

a.show()
imshow(b)
