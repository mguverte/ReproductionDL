"""
Date:   25-03-2022
Author: Austin, Mustafa, Richard, Dimme
Based on: [1], [2], [3], [4], [5]
Descr:  Part of Code for Table 1 of the paper " Variational Autoencoders: A Harmonic Perspective"
        Stores figures for verification purposes.
        
Sources:
    [1] examples from https://github.com/didriknielsen/survae_flow
    [2] https://avandekleut.github.io/vae/
    [3] The paper: Variational Autoencoders: A Harmonic Perspective
    [4] Lab 8 of CS4240-Deep Learning course
    [5] https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
    
Notes: 
    (a) you need to have trained the models before using this file, and stored at the proper location.
    (b) you need to have a folder Figures inside the directory where this .py file is stored.
"""

import torch
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt 
import numpy as np

from torch import nn
from survae.data.loaders.image import CelebA
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from skimage import color

#User settings
select = [2]                    #select which models, 0: CelebA, 1: Cifar, 2: Sinc


plt.rc('font', size=14) #controls default text size

#Define model architecture
###############################################################################
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, nChan, imSizeH, imSizeV):
        super(VariationalEncoder, self).__init__()
        self.input = nn.Flatten(1)
        self.dense1 =  nn.Linear(imSizeH*imSizeV*nChan, 256)
        self.dense2 =  nn.Linear(256, 256)
        self.dense3 =  nn.Linear(256, 256)
        self.outputMu =  nn.Linear(256, latent_dims)
        #self.outputSig = nn.Linear(256, latent_dims)       #this layer is not needed as sigma is fixed
        
        self.N = torch.distributions.Normal(0, 1)   
        self.kl = 0
        
    def forward(self, x, sigma_en):
        x = self.input(x)
        x = torch.sigmoid(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        x = torch.sigmoid(self.dense3(x))
        mu =  self.outputMu(x)
        #sigma = torch.exp(self.outputSig(x))               #this layer is not needed as sigma is fixed
        
        z = mu + (sigma_en**2)*self.N.sample(mu.shape)      #combination of encoders output of layer2, layer3
        self.kl = (sigma_en**2 + mu**2 - torch.log(sigma_en) - 1/2).sum()
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims, nChan, imSizeH, imSizeV):
        super(Decoder, self).__init__()
        self.dense1 = nn.Linear(latent_dims, 256)
        self.dense2 = nn.Linear(256, 256)
        self.dense3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, nChan*imSizeH*imSizeV)
        self.outputShape = nn.Unflatten(dim=1, unflattened_size = (nChan, imSizeH, imSizeV))

    def forward(self, z):
        z = torch.sigmoid(self.dense1(z))
        z = torch.sigmoid(self.dense2(z))
        z = torch.sigmoid(self.dense3(z))
        z = self.output(z)
        z = self.outputShape(z)
        return z

class fnc_get_model(nn.Module):
    def __init__(self, latent_dims, nChan, imSizeH, imSizeV):
        super(fnc_get_model, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, nChan, imSizeH, imSizeV)
        self.decoder = Decoder(latent_dims, nChan, imSizeH, imSizeV)

    def forward(self, x, sigma_en):
        z = self.encoder(x, sigma_en)
        return self.decoder(z)
###############################################################################

###############################################################################
def fnc_getStore_and_Load(selector):
    if selector==0: #CelebA
        #Save/load paths for CelebA, left column
        path_ll = ["Models/modelL1-sigma-1_0", "Models/modelL1-sigma-0_5", "Models/modelL1-sigma-0_1"]                          #Load locations
        path_sfl = ["Figures/FigL1-sigma-1_0-ev.png", "Figures/FigL1-sigma-0_5-ev.png", "Figures/FigL1-sigma-0_1-ev.png"]       #Store locations evaluation
        path_sdl = ["Figures/FigL1-sigma-1_0-data.png", "Figures/FigL1-sigma-0_5-data.png", "Figures/FigL1-sigma-0_1-data.png"] #Store locations data
        
        #Save/load paths for CelebA, right column
        path_lr = ["Models/modelR1-sigma-1_0", "Models/modelR1-sigma-0_5", "Models/modelR1-sigma-0_0"]              #Load locations
        path_sfr = ["Figures/FigR1-sigma-1_0-ev.png", "Figures/FigR1-sigma-0_5-ev.png", "Figures/FigR1-sigma-0_0-ev.png"]       #Store locations evaluation
        path_sdr = ["Figures/FigR1-sigma-1_0-data.png", "Figures/FigR1-sigma-0_5-data.png", "Figures/FigR1-sigma-0_0-data.png"] #Store locations data
       
    elif selector == 1: #Cifar
        #Save/load paths for CIFAR, left column
        path_ll = ["Models/modelL2-sigma-1_0", "Models/modelL2-sigma-0_5", "Models/modelL2-sigma-0_1"]                          #Load locations
        path_sfl = ["Figures/FigL2-sigma-1_0-ev.png", "Figures/FigL2-sigma-0_5-ev.png", "Figures/FigL2-sigma-0_1-ev.png"]       #Store locations evaluation
        path_sdl = ["Figures/FigL2-sigma-1_0-data.png", "Figures/FigL2-sigma-0_5-data.png", "Figures/FigL2-sigma-0_1-data.png"] #Store locations data
        
        #Save/load paths for CIFAR, right column
        path_lr = ["Models/modelR2-sigma-1_0", "Models/modelR2-sigma-0_5", "Models/modelR2-sigma-0_0"]                          #Load locations
        path_sfr = ["Figures/FigR2-sigma-1_0-ev.png", "Figures/FigR2-sigma-0_5-ev.png", "Figures/FigR2-sigma-0_0-ev.png"]       #Store locations evaluation
        path_sdr = ["Figures/FigR2-sigma-1_0-data.png", "Figures/FigR2-sigma-0_5-data.png", "Figures/FigR2-sigma-0_0-data.png"] #Store locations data
  
    else: #Sinc
        #Save/load paths for SINC, left column
        path_ll = ["Models/modelL3-sigma-1_0", "Models/modelL3-sigma-0_5", "Models/modelL3-sigma-0_1"]                          #Load locations
        path_sfl = ["Figures/FigL3-sigma-1_0-ev.png", "Figures/FigL3-sigma-0_5-ev.png", "Figures/FigL3-sigma-0_1-ev.png"]       #Store locations evaluation
        path_sdl = ["not used", "not used", "not used"] #Store locations data
   
        #Save/load paths for SINC, right column
        path_lr = ["Models/modelR3-sigma-1_0", "Models/modelR3-sigma-0_5", "Models/modelR3-sigma-0_0"]                          #Load locations
        path_sfr = ["Figures/FigR3-sigma-1_0-ev.png", "Figures/FigR3-sigma-0_5-ev.png", "Figures/FigR3-sigma-0_0-ev.png"]       #Store locations evaluation
        path_sdr = ["not used", "not used", "not used"] #Store locations data

    return path_ll, path_lr, path_sfl, path_sfr, path_sdl, path_sdr
###############################################################################

###############################################################################
def fnc_getValid_loader(selector):    
    if selector == 0:
        data = CelebA()
        valid_loader = DataLoader(dataset=data.valid, batch_size=256, shuffle=False, num_workers=8, drop_last=True)
    elif selector == 1:
        cifar_trainset = datasets.CIFAR10(root='../DATA', train=False, download=True, transform=ToTensor())
        valid_loader = DataLoader(dataset=cifar_trainset, batch_size=256, shuffle=False, num_workers=8, drop_last=True) 
    else:
        w, t_s, t_e, N, M = 5, -1, 1, 172, 2        #frequency, start time, end time, number of samples
        data_p = sinc(w, t_s, t_e, N, M)            #get raw data
        data, offset, scale = preprocess(data_p)    #preprocess raw data. Using scale and offset og data can be returned
        valid_loader = data
    return valid_loader

#function to get sinc data
def sinc(w, t_s, t_e, N, M):
    t_ax = torch.linspace(t_s, t_e, N)      
    t_ax = t_ax.unsqueeze(1)
    t_ax = t_ax.unsqueeze(0)
    t_ax = t_ax.unsqueeze(0)  
    f_eval = torch.sinc(w*t_ax) 
    out = torch.cat((f_eval, t_ax),-1)
    return out

#function to preprocess sinc data
def preprocess(data):
    offset = torch.min(data, dim=2)
    data = data - offset.values         #make minimum zero for both t and f(t)
    scale = torch.max(data, dim=2)
    data = data/scale.values            #make max one for both t and f(t)
    data = 2*data - 1                   #from t, f(t) in [0,1] to t, f(t) in [-1, 1]
    return data, offset, scale          #return data, offset and scale. Note that the 2* and -1 are not returned
###############################################################################

###############################################################################
def fnc_plot(model, data_name, result_name, valid_loader, selector):
    N = 4
    fft = 0
    if selector == 0: #CELEBA
        img = next(iter(valid_loader))[:N]
        img = (2*img/255.0)-1 #Normalise between [-1, 1]
        samples = torch.zeros(N,3,64,64)
        
        #some extra for when FFT+converting to gray scale is needed
        samples2 = np.zeros((N, 64, 64, 3))
        img2 = np.zeros((N,64, 64, 3))
        samplesGray = np.zeros((N, 64, 64))   #store gray scale
        imgGray = np.zeros((N, 64, 64))       #store gray scale
        SAMPLES = np.zeros((N, 64, 64))       #store freq domain
        IMG = np.zeros((N,64,64))             #store freq domain
    
        for i in range(N):
            samples[i] = model(img[i].unsqueeze(dim=0), torch.tensor(0))
        
        img = (img+1)/2         #Normalise between [0,1]
        samples = (samples+1)/2 #Normalise between [0,1]
        
        if fft==1:
            samples = samples.detach().numpy()
            img = img.detach().numpy()
            
            for i in range(N): #permute submatrices
                samples2[i] = np.transpose(samples[i], (1, 2, 0))
                img2[i] = np.transpose(img[i], (1, 2, 0))
            
            for i in range(N): #convert to gray scale
                samplesGray[i] = color.rgb2gray(samples2[i])
                imgGray[i] = color.rgb2gray(img2[i])
                SAMPLES[i] = 20*np.log10(abs(np.fft.fft2(samplesGray[i])))
                IMG[i] = 20*np.log10(abs(np.fft.fft2(imgGray[i])))
                
            plt.subplot(2, 2, 1)
            plt.imshow(SAMPLES[0], vmin=-65, vmax=65, cmap='jet', aspect='auto') 
            plt.colorbar()
            plt.subplot(2, 2, 2)
            plt.imshow(SAMPLES[1], vmin=-65, vmax=65, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.subplot(2, 2, 3)
            plt.imshow(SAMPLES[2], vmin=-65, vmax=65, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.subplot(2, 2, 4)
            plt.imshow(SAMPLES[3], vmin=-65, vmax=65, cmap='jet', aspect='auto')
            plt.colorbar()
            plt.show()
            
        else:
            vutils.save_image(img.cpu().float(), fp=data_name, nrow=2)
            vutils.save_image(samples.cpu().float(), fp=result_name, nrow=2) 
        return img, samples
    
    elif selector == 1: #CIFAR
        img = next(iter(valid_loader))[:N]
        img = img[0]
        img = (2*img)-1 #Normalise between [-1, 1]
        samples = torch.zeros(N,3,32, 32)
        img2 = torch.zeros(N,3,32, 32)
    
        for i in range(N):
            samples[i] = model(img[i].unsqueeze(dim=0), torch.tensor(0))
            img2[i] = img[i]
            
        img2 = (img2+1)/2         #Normalise between [0,1]
        samples = (samples+1)/2 #Normalise between [0,1]
        vutils.save_image(img2.cpu().float(), fp=data_name, nrow=2)
        vutils.save_image(samples.cpu().float(), fp=result_name, nrow=2) 
        return img2, samples
    
    else: #sinc
        x = next(iter(valid_loader))[:172]
        sample = torch.zeros(1, 1, 172,2)
        sample = model(x, torch.tensor(0))
        x = torch.squeeze(x)
        sample = torch.squeeze(sample)
        x = x.detach().numpy()
        sample = sample.detach().numpy()
        fig = plt.figure( )
        plt.plot(x[:,1], x[:, 0], label='input')
        plt.plot(sample[:, 1], sample[:, 0], label='reconstruction')
        plt.xlabel("Time [-]")
        plt.ylabel("sinc(5t), normalised to [-1, 1]")
        plt.legend(loc='upper right')
        fig.savefig(result_name)
        return x, sample
###############################################################################

latent_dims = [64, 64, 1]       #latent dimensions
imSizeH = [64, 32, 172]         #image size 1 (horizontal)
imSizeV = [64, 32, 2]           #image size 2 (vertical)
nChan = [3, 3, 1]               #number of channels

for k in select:
    print("the selected model is {}. Note: 0=celeba, 1=cifar, 2=sinc".format(k))
    print(" ")
    print("Step 1: Loading data")
    valid_loader = fnc_getValid_loader(k)
    path_ll, path_lr, path_sfl, path_sfr, path_sdl, path_sdr = fnc_getStore_and_Load(k)     #get location to load models, store figures
    
    print("Saving figures for left column")
    for j in range(3):
        model = fnc_get_model(latent_dims=latent_dims[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        model.load_state_dict(torch.load(path_ll[j]))
        model.eval()
        data, samples = fnc_plot(model, path_sdl[j], path_sfl[j], valid_loader, k)
        
    print("Saving figures for right column")
    
    for j in range(3):
        model = fnc_get_model(latent_dims=latent_dims[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        model.load_state_dict(torch.load(path_lr[j]))
        model.eval()
        
        data, samples = fnc_plot(model, path_sdr[j], path_sfr[j], valid_loader, k)
    print(" ")        
