import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import pickle
import bm3d
import time

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SequentialSampler
from torch import nn
from nets import UNet
import pytorch_ssim

import pandas as pd
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
    images = images + n
    images = torch.clamp(images, 0, 1)
    return images

batch_size = 32
random_seed = 1346


image = "D:\\LEEM000896.png"
old_run = "runs-denoise\\1580244059\\"
old_run = "runs-denoise\\1580841091\\"

files = os.listdir("D:\\repos\\LEEM_imgs")
files = [os.path.join("D:\\repos\\LEEM_imgs", f) for f in files if f.endswith('png')]

np.random.seed(random_seed)
np.random.shuffle(files)

trans = transforms.Compose([transforms.CenterCrop(256),
                            transforms.ToTensor(),
                            transforms.Normalize((norm_mean,), (norm_std, ))
                            ])

dataset = LEEMdataset(files, transforms=trans)

indices = list(range(len(dataset)))

val_indices = indices[:100]

valid_sampler = SequentialSampler(val_indices)

validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

criterion = nn.BCELoss()
mseloss = nn.MSELoss()
ssim = pytorch_ssim.SSIM()


d = iter(validation_loader)

test_data_input = next(d)
test_data_target = test_data_input


img = cv2.imread(image)[:,:,0]
img = PIL.Image.fromarray(img)
img = trans(img)
img = img.unsqueeze(0)

model = UNet(1, 1)
model.load_state_dict(torch.load(old_run+'model.pth'))


print(test_data_input.shape)

metrics = pd.DataFrame(index=range(batch_size))


with torch.no_grad():

    for noise_level, name in [(0.05, "005"), (0.1, "01"), (0.2, "02"), (0.4, "04")]:
        test_data_input = add_noise(test_data_target.clone(), noise_level)

        t1 = time.time()
        out = model(test_data_input)
        t2 = time.time()



        t_nn = t2-t1
        ssimscores = []
        msescores = []
        print(name)
        print(t_nn)
        for i in range(batch_size):
            cv2.imwrite('evaluation{0}\\target\\{1}.png'.format(name, i), 255*test_data_target[i,0, :, :].numpy())
            cv2.imwrite('evaluation{0}\\out\\{1}.png'.format(name, i), 255*out[i,0, :, :].numpy())
            cv2.imwrite('evaluation{0}\\noisy\\{1}.png'.format(name, i), 255*test_data_input[i, 0, :, :].numpy())
            msescore = mseloss(out[i,:,:,:].unsqueeze(0), test_data_target[i,:,:,:].unsqueeze(0)).item()
            ssimscore = ssim(out[i,:,:,:].unsqueeze(0), test_data_target[i,:,:,:].unsqueeze(0)).item()
            
            msescores.append(10*np.log10(1/(msescore)))
            ssimscores.append(ssimscore)

        metrics['out_ssim'] = ssimscores
        metrics['out_psnr'] = msescores
        
        #print(metrics)
        cv2.destroyAllWindows()

        metrics.to_csv('evaluation{0}\\metrics.csv'.format(name))