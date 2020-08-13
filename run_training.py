import torch
import torchvision
import torchvision.transforms as transforms

# setup imageNet dataset
#imagenet = datasets.ImageNet(root="~/Pictures/ImageNet", split='train')
transform = transforms.Compose([transforms.Grayscale()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# setup model and optimization
model = models.deepconv()