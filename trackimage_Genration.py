import argparse
import os
import numpy as np
import math
import pandas as pd

import requests
import time
import PIL
from PIL import Image

from tensorboardX import SummaryWriter

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt

x = 1
data = np.load('/content/gdrive/MyDrive/spd_train.npy', allow_pickle=True).item()
while x != 0:
  x = 0
  for idx, img in enumerate(data['data']):
    if img.shape != (64, 64, 3):
      x +=1
      del data['data'][idx]
      del data['names'][idx]
      del data['labels'][idx]
  print(x)
data['data'] = data['data'][0:500000]
data['names'] = data['names'][0:500000]
data['labels'] = data['labels'][0:500000]

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100000, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=250, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--save_dir", type=str, default='TrainWB', help="directory to save logging information")
parser.add_argument("--name", type=str, default='GAN1', help="name of this training run")
opt = parser.parse_args(args=[])
print(opt)

tbx = SummaryWriter(opt.save_dir)
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

class GaussianNoise(nn.Module):
    def __init__(self, stdev):
        super().__init__()
        self.stdev = stdev

    def forward(self, x):
        if self.training:
            return x + torch.autograd.Variable(torch.randn(x.size()) * self.stdev)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model_noise = nn.Sequential(
          nn.Linear(in_features=opt.latent_dim, out_features=4*4*1024),
          nn.LeakyReLU(),
          nn.BatchNorm1d(num_features=4*4*1024),
          nn.Linear(in_features=4*4*1024, out_features=4*4*1024),
          nn.LeakyReLU(),
          nn.BatchNorm1d(num_features=4*4*1024),
          nn.Unflatten(1, (1024, 4, 4))
        )

        self.model_label = nn.Sequential(
          nn.Linear(in_features=opt.n_classes, out_features=4*4*1024),
          nn.LeakyReLU(),
          nn.BatchNorm1d(num_features=4*4*1024),
          nn.Linear(in_features=4*4*1024, out_features=4*4*1),
          nn.LeakyReLU(),
          nn.BatchNorm1d(num_features=4*4*1),
          nn.Unflatten(1, (1, 4, 4)),
        )

        self.model_img = nn.Sequential(
          nn.ConvTranspose2d(in_channels=1025, out_channels=512, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(num_features=512),
          nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(num_features=256),
          nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(num_features=128),
          nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=4, stride=2, padding=1),
          nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        model_n = self.model_noise(noise)
        model_l = self.model_label(labels)
        gen_input = torch.cat((model_n, model_l), dim=1)
        img = self.model_img(gen_input)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=6),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Flatten(),
            nn.Linear(in_features=8*8*128, out_features=4*4*64),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.img_out = nn.Linear(in_features=4*4*64, out_features=1)
        self.label_out = nn.Linear(in_features=4*4*64, out_features=10)


    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        #if len(labels.shape) > 2:
        #    new_labels = torch.stack([labels.permute(1,3,0,2).squeeze() for _ in range(opt.batch_size)])
        #else:
        #  new_labels = torch.stack([torch.stack([labels for a in range(opt.img_size)]) for b in range(opt.img_size)]).permute((2,3,0,1)).type(FloatTensor)
        #d_in = torch.cat((img, new_labels), 1) #self.label_embedding(labels)

        out_state = self.model(img)
        validity = self.img_out(out_state)
        label_guess = self.label_out(out_state)
        return validity, label_guess

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def label_generator(num_labels):
  #idxs = np.random.randint(0, len(data['labels']), num_labels)
  #entries = [data['labels'][i] for i in idxs]
  entries=[]
  for i in range(num_labels):
    entries.append((np.random.random(size=10) * 2) - 1)
    '''entry = []
    entry.append(np.random.randint(0, high=100)) #popularity
    entry.append(np.random.random() * 2 - 1) #acousticness
    entry.append(np.random.random()) #danceability
    entry.append(np.random.random()) #energy
    entry.append(np.random.random()) #instrumentalness
    entry.append(np.random.random()) #liveleness
    entry.append(np.random.randint(-60, high=0)) # loudness
    entry.append(np.random.random()) # speechiness
    entry.append(np.random.randint(50, high=200)) # tempo
    entry.append(np.random.random()) # valence
    entries.append(entry)'''
  return Variable(FloatTensor(np.array(entries)), requires_grad=True)


def sample_image(n_row, batches_done, tbx):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    n_row = 5
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = label_generator(n_row**2)
    gen_imgs = ((generator(z, labels) + 1)/2) * 255
    tbx.add_images("images/%d.png" % batches_done, gen_imgs, batches_done)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

def compute_gradient_penalty(D, real_samples, fake_samples, real_labels, fake_labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    try:
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device='cuda')
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).type(FloatTensor)
        interpolates_labels = (alpha * real_labels + ((1 - alpha) * fake_labels)).requires_grad_(True).type(FloatTensor)
        d_interpolates = D(interpolates, interpolates_labels)
        fake = Variable(torch.zeros((opt.batch_size,1)).fill_(1.0), requires_grad=False).to(device='cuda')
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    except:
        gradient_penalty = 0
    return gradient_penalty

def lat_opt_ngd(G,D,z,labels, batch_size, alpha=500, beta=0.1, norm=1000):
    x_hat = G(z, labels)
    f_z, l_z = D(x_hat.view(batch_size, 3, 64, 64), labels)

    fz_dz = torch.autograd.grad(outputs=f_z,
                                inputs= z,
                                grad_outputs=torch.ones_like(f_z),
                                retain_graph=True,
                                create_graph= True
                                   )[0]
    lz_dz = torch.autograd.grad(outputs=l_z,
                                inputs= labels,
                                grad_outputs=torch.ones_like(l_z),
                                retain_graph=True,
                                create_graph= True
                                   )[0]
    
    delta_z = torch.ones_like(fz_dz)
    delta_z2 = torch.ones_like(lz_dz)
    delta_z = (alpha * fz_dz) / (beta +  torch.norm(delta_z, p=2, dim=0) / norm).to(device='cuda')
    delta_z2 = (alpha * lz_dz) / (beta +  torch.norm(delta_z2, p=2, dim=0) / norm).to(device='cuda')
    with torch.no_grad():
        z_prime = torch.clamp(z + delta_z, min=-1, max=1) 
        l_prime = torch.clamp(labels + delta_z2, min =-1, max =1)
        
    return z_prime, l_prime

def sample_noise(batch_size, dim):
    return Variable(2 * torch.rand([batch_size, dim]) - 1, requires_grad=True)

def diversity(imgs):
  score = 0 
  for target in imgs:
    for sample in imgs:
      #score += torch.mean(torch.abs(target - sample))
      score += torch.sum((target == sample) * 1)/64
  return score/(64**2)

if cuda:
    generator.cuda()
    discriminator.cuda()

# ----------
#  Training
# ----------

batch_size = opt.batch_size
for epoch in range(opt.n_epochs):
    for i in range(0, len(data['names']), batch_size):

        # Get further data
        imgs = []
        labels = []
        for idx, img in enumerate(data['data'][i: i+batch_size]):
          imgs.append(torch.from_numpy((((img.transpose(-1, 0, 1))/255)-0.5) * 2))
          label = np.array(data['labels'][idx])
          label[0] = ((label[0]/100)*2)-1
          label[1:6] = label[1:6] * 2 - 1
          label[6] = (label[6] / -60) * 2 - 1
          label[7] = label[7] * 2 - 1
          label[8] = ((label[8] - 50)/200) * 2 - 1
          label[9] = label[9] * 2 - 1
          labels.append(torch.from_numpy(np.array(label)))
        imgs = torch.stack(imgs)
        labels = torch.stack(labels)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))

        # -----------------
        #  Train D
        # -----------------

        optimizer_D.zero_grad()
        z = Variable(sample_noise(batch_size, opt.latent_dim), requires_grad=True).to(device='cuda')
        gen_labels = label_generator(batch_size).to(device='cuda')
        z, gen_labels = lat_opt_ngd(generator, discriminator, z, gen_labels, batch_size)

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels).detach().to(device='cuda')

        # Loss measures generator's ability to fool the discriminator
        real_img_score, real_label_score = discriminator(real_imgs, labels)
        fake_image_score, fake_label_score = discriminator(gen_imgs, gen_labels)
        fl = torch.abs(fake_label_score - gen_labels)
        rl = torch.abs(real_label_score - labels)
        d_loss = (-(torch.mean(real_img_score) + torch.mean(rl)) + (torch.mean(fake_image_score) + torch.mean(fl))).to(device='cuda') #+ compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, labels, gen_labels) * 10

        d_loss.backward()
        optimizer_D.step()

        for p in discriminator.parameters():
          p.data.clamp_(-opt.clip_value, opt.clip_value)

        # ---------------------
        #  Train G
        # ---------------------
        if i % opt.n_critic == 0:

          optimizer_G.zero_grad()

          # Generate images and labels
          gen_labels = label_generator(batch_size).to(device='cuda')
          gen_imgs = generator(z, gen_labels).to(device='cuda')

          #TEST
          #to_pil = transforms.ToPILImage()
          #display(to_pil(real_imgs[0]))
          #display(to_pil(gen_imgs[0]))

          # Loss for generator
          #div = diversity(gen_imgs)
          im_score, l_score = discriminator(gen_imgs, gen_labels)
          g_loss = -(torch.mean(im_score) + torch.mean(torch.abs(l_score-gen_labels))).to(device='cuda') #+ div 

          g_loss.backward()
          optimizer_G.step()

        if i % opt.sample_interval == 0:

          print(
              "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
              % (epoch, opt.n_epochs, i, len(data['names']), d_loss.item(), g_loss.item()) #dataloader
          )

        batches_done = epoch * len(data['names']) + i
        #Log info 
        tbx.add_scalars('Loss/train' + opt.name, {"Generator" : g_loss, "Discriminator": d_loss}, batches_done)

        if batches_done % opt.sample_interval == 0:
            torch.save(generator.state_dict(), '/content/gdrive/MyDrive/ti_generator.tar')
            torch.save(discriminator.state_dict(), '/content/gdrive/MyDrive/ti_disriminator.tar')
            sample_image(n_row=10, batches_done=batches_done/opt.sample_interval, tbx=tbx)
