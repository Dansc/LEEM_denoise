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
from nets import UNet, _NetD
import time
import random

torch.cuda.empty_cache()
norm_mean = 0.0
norm_std = 1.0

class SEMdataset(Dataset):
    def __init__(self, datalst, transforms=None):
        """
        datalst is list of tuple with leem image path as first, 
        background image as second argument
        transforms is a torchvision transforms object
        """
        super(SEMdataset, self).__init__()
 
        self.datalst = datalst
        self.transforms = transforms

    def __len__(self):
        return len(self.datalst)


    def loadimg(self, fn):
        img = cv2.imread(fn)
        img = img[:,:,0]
        if self.transforms:
            img = PIL.Image.fromarray(img)
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img).float()
        return img


    def __getitem__(self, idx):
        seed =  random.randint(1, 1E6)
        random.seed(seed)
        img_noisy = self.loadimg(self.datalst[idx][0])
        random.seed(seed)
        img_notnoisy = self.loadimg(self.datalst[idx][1])
        
        return (img_notnoisy, img_noisy)

def add_noise(images, noise_level):
    n = noise_level*np.random.randn()*(torch.randn(*images.size()) - norm_mean) / norm_std
    n = n.to(device)
    images = images + n
    images = torch.clamp(images, 0, 1)
    return images

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)

    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    error_real = criterion(prediction_real, ones_target(N))
    # error_real.backward()
    
    prediction_fake = discriminator(fake_data)
    error_fake = criterion(prediction_fake, zeros_target(N))
    # error_fake.backward()

    error = error_real/2 + error_fake/2
    error.backward()

    optimizer.step()

    return error, prediction_real, prediction_fake


# List all LEEM images
files = os.listdir("E:\\Dschwa\\datasets\\noisyimgs")
files = [f for f in files if int(f.split('_')[0]) > 6999]
files = [os.path.join("E:\\Dschwa\\datasets\\noisyimgs", f) for f in files if f.endswith('tif')]

filesnoisy = files[::2]
filesnotnoisy = files[1::2]

files = zip(filesnoisy, filesnotnoisy)
files = list(files)

#transforms.CenterCrop((768, 900)), 
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
num_epochs = 400
lr = 0.0001

# Create random dataset
dataset = SEMdataset(files, transforms=trans)
dataset_size = len(dataset)
indices = list(range(dataset_size))
dataset_size = int(amount_use*len(dataset))

print("Dataset size: {}".format(dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)

random.seed(random_seed)
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

discriminator =  _NetD()
# specify loss function
criterion = nn.BCELoss() # BCE loss superior to MSEloss here
mseloss = nn.MSELoss()
criterion = nn.MSELoss()

discloss = nn.BCEWithLogitsLoss()

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
d_optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print("Length train loader: {}".format(len(train_loader)))
print("Batchsize: {}".format(batch_size))
print("Batchsize*train_loader: {}".format(len(train_loader)*batch_size))


# specify optimizer
t = str(int(time.time()))
logdir = os.path.join('runs-denoise', t)
os.mkdir(logdir)

d = iter(validation_loader)
data = next(d)
test_data = data[0].to(device)
test_noisy = data[1]
#test_noisy = add_noise(test_data, noise_level)

grid = torchvision.utils.make_grid((test_data+2*norm_mean)*norm_std)
writer = SummaryWriter(log_dir=logdir)
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
    for d in train_loader:
        data = d[1].to(device)
        noisy_imgs = d[0].to(device)
       # noisy_imgs = add_noise(data, noise_level).to(device)
        #noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)

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
        mselosses = 0.
        for i in validation_loader:
            img = i[1].to(device)
            img_noisy = i[0].to(device)

            outputs = model(img_noisy)
            mselosses += mseloss(outputs, img).item()


        model.eval()  # Turn off dropout, batch norm to use running mean
        test = model(test_noisy)
        grid = torchvision.utils.make_grid((test+2*norm_mean)*norm_std)
        writer.add_scalar("Loss/train", train_loss, n+1)
        writer.add_image('images/test', grid, n+1)
        writer.add_scalar('PSNR/valid', 10*np.log10(len(validation_loader)/mselosses), n+1)
    if ((n+1) % 5) == 0:
        print("Saving generator and optimizer state dicts.")
        torch.save(model.state_dict(), os.path.join(logdir, 'model.pth'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, 'optimizer_state.pth'))
torch.save(model, 'model_'+t+'.pth')
writer.close()