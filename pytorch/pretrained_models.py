import torchvision

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

#model.fc = nn.Linear(512, 100)

# Optimize only the classifier
#optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

print(model)
