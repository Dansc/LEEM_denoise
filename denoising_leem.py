import cv2
import numpy as np
import os

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from nets import UNet

torch.cuda.empty_cache()
norm_mean = 0.0
norm_std = 1.0

class LEEMdataset(Dataset):
    def __init__(self, datalst, transforms=None):
        """
        datalst is list of tuple with leem image path as first, 
        background image as second argument
        transforms is a torchvision transforms object
        """
        super(LEEMdataset, self).__init__()
 
        self.datalst = datalst
        self.transforms = transforms

    def __len__(self):
        return len(self.datalst)

    def __getitem__(self, idx):
        img = cv2.imread(self.datalst[idx])
        img = img[:,:,0]
        if self.transforms:
            img = PIL.Image.fromarray(img)
            img = self.transforms(img)
        
        else:
            img = torch.from_numpy(img).float()
        return img

def add_noise(images, noise_level):
    n = noise_level*(torch.randn(*images.size()) - norm_mean) / norm_std
    n = n.to(device)
    images = images + n
    images = torch.clamp(images, 0, 1)
    return images

# List all LEEM images
files = os.listdir("D:\\repos\\LEEM_imgs")
files = [os.path.join("D:\\repos\\LEEM_imgs", f) for f in files if f.endswith('png')]


trans = transforms.Compose([transforms.RandomCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize((norm_mean,), (norm_std, ))
                            ])

                      
# Variables for validation split
batch_size = 16
validation_split = .2
shuffle_dataset = True
random_seed= 55
amount_use = 0.4

# Variables for training
num_epochs = 50
noise_level = 0.1
lr = 0.0005

# Create random dataset
dataset = LEEMdataset(files, transforms=trans)
dataset_size = len(dataset)
indices = list(range(dataset_size))
dataset_size = int(amount_use*len(dataset))

print("Dataset size: {}".format(dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
     
indices = indices[:dataset_size] 
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

# Create the neural network
model = UNet(1, 1)
print(model)
model.to(device)

# specify loss function
criterion = nn.BCELoss() # BCE loss superior to MSEloss here

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Length train loader: {}".format(len(train_loader)))
print("Batchsize: {}".format(batch_size))
print("Batchsize*train_loader: {}".format(len(train_loader)*batch_size))

d = iter(validation_loader)
test_data = next(d)
test_data = test_data.to(device)
test_noisy = add_noise(test_data, noise_level)
grid = torchvision.utils.make_grid((test_data+2*norm_mean)*norm_std)
writer = SummaryWriter()
writer.add_image('images/testsample', grid, 0)
grid = torchvision.utils.make_grid((test_noisy+2*norm_mean)*norm_std)
writer.add_image('images/testsamplenoisy', grid, 0)

test_noisy = test_noisy.to(device)

train_loss = 0.0
for n in range(num_epochs):
    model.train()
    print("Epoch: {0} of {1}".format(n+1, num_epochs))
    print(train_loss/(len(train_loader)*batch_size))
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        noisy_imgs = add_noise(data, noise_level).to(device)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)

        # clear gradients of all optimized variables
        optimizer.zero_grad()
        outputs = model(noisy_imgs)

        # calculate the loss, target is data, not noisy
        loss = criterion(outputs, data)

        # backward pass: compute gradient of the loss with respect
        # to model parameters
        loss.backward()
        optimizer.step()

        # update running training loss
        train_loss += loss.item()*data.size(0)

    with torch.no_grad():
        model.eval()  # Turn off dropout, batch norm to use running mean
        test = model(test_noisy)
        grid = torchvision.utils.make_grid((test+2*norm_mean)*norm_std)
        writer.add_scalar("Loss/train", train_loss, n+1)
        writer.add_image('images/test', grid, n+1)

torch.save(model, 'model.pth')
writer.close()