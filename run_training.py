import models
import image_loader
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# setup dataset
#imagenet = datasets.ImageNet(root="~/Pictures/ImageNet", split='train')
#transform = transforms.Compose([transforms.Grayscale()])
#trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = image_loader.Cifar10Loader('./data/cifar-10-python/cifar-10-batches-py/data_batch_1')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# setup model and optimization
net = models.DeepColorSimple().double()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.MSELoss()

# setup tensorboard
writer = SummaryWriter()

i = 0
for epoch in range(100):  # loop over the dataset multiple times

    running_loss = 0.0
    for tr_data in trainloader:
        i += 1
        # get the inputs; data is a list of [inputs, labels]
        input = tr_data['input']
        target = tr_data['target']

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        prediction = net(input)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        # output images for tensorboard
        if i % 300 == 0:    # print every 500 mini-batches
            grid = torchvision.utils.make_grid([torch.cat(3*[input[0]]), prediction[0], target[0]])
            writer.add_image('Cifar10-TrainingSamples', grid, i)

        writer.add_scalar('Loss/train', loss, i)

print('Finished Training')

