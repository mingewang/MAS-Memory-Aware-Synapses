from MAS import *
from MNIST_split import *
import os

# prepare dir
os.system("mkdir -p data/Pytorch_MNIST_dataset")

# download MNIST to data/processed, data/raw
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
batch_size=100
kwargs = {'num_workers': 1, 'pin_memory': True} 
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

# we split the full MNIST training
# data set into 5 subsets of consecutive digits. The 5 tasks
# correspond to learning to distinguish between two consecutive digits from 0 to 10.
for digits in [[1,2],[3,4],[5,6],[7,8],[9,0]]:
    dsets={}
    dsets['train']=    MNIST_Split('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),digits=digits)
    dsets['val']=  MNIST_Split('data', train=False, download=True, 
                      transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]),digits=digits)
    dlabel=''
    for i in digits:
        dlabel=dlabel+str(i)
    torch.save(dsets,'data/Pytorch_MNIST_dataset//split'+dlabel+'_dataset.pth.tar')

