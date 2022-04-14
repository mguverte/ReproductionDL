"""
Date:       21-03-2022
Last edit:  24-03-2022
Author:     Austin, Richard, Mustafa, Dimme

Based on:   [1], [2], [3], [4], [5]
Trains:     3x6 models on celeba, cifar, sinc
Descr:      Part of Code for Table 1 of the paper " Variational Autoencoders: A Harmonic Perspective"
            decoder is deterministic (given a latent space)
            
            Where the first three models have no noise injections on the input, and fixed encoder standard deviation (!) of 
                [1.0, 0.5, 0.1]
            Where the last three models have fixed encoder standard deviation (!) 0.5 and noise injections of 
                a standard normal distribution with zero mean and standard deviation of [1.0, 0.5, 0.0]
                
Some notes: 
    (a) inside the folder where you put this file, make the following two folders:
            (1) Model (used for storing model state)
            (2) Optim (used for storing optimiser state)
            
Sources:
    [1] examples from https://github.com/didriknielsen/survae_flow
    [2] https://avandekleut.github.io/vae/
    [3] The paper: Variational Autoencoders: A Harmonic Perspective
    [4] Lab 8 of CS4240-Deep Learning course
    [5] https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
"""

import torch
import torchvision.datasets as datasets

from torch import nn
from survae.data.loaders.image import CelebA
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from pytictoc import TicToc

#use one of three seeds for reproducable results+calculating standard deviation
#seeds obtained using torch.seed() three times.

torch.manual_seed(14609714069366804553) 
#torch.manual_seed(10209623728859046282)
#torch.manual_seed(13978253786968215756) 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
t = TicToc()

#User settings
select = [2]                                #select which models, 0: CelebA, 1: Cifar, 2: Sinc; [0, 1, 2] runs all
epochs, lr = [200, 200 , 100], [1e-3, 1e-3, 1e-3]  #number of epochs, learning rate; epochs[0] corresponds to CelebA, etc.
load_old = [0, 0, 1]                        #set to one if you want to continue training. load_old[0] corresponds to CelebA, etc.
norm = [0, 1, 2]                            #set type of normalisation: 0: 8 bit integer to float [-1, 1]; 1: float to float [-1, 1]; 2: no normalisation


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

###############################################################################3
#function to train model (left column)
def fnc_train(epochs, sigma_en, sigma_in, norm, nChan, imSizeH, imSizeV):
    if norm == 0:
        scale, div, offset = 2, 255, 1
    elif norm == 1:
        scale, div, offset = 2, 1, 1
    else:
        scale, div, offset = 1, 1, 0
        
    if sigma_in == 0:
        for epoch in range(epochs):
            l = 0.0
            for i, x in enumerate(train_loader):
                if isinstance(x, list):
                    x = x[0]
                x = scale*(x/div)-offset   #normalise between [-1,1]
                x = x.to(device)  
                optimizer.zero_grad()
                x_hat = model(x, sigma_en)
                loss = ((x - x_hat)**2).sum() + model.encoder.kl
                loss.backward()
                optimizer.step()
                l += loss.detach().cpu().item()
            print('Epoch: {}/{}, Loss: {:.3f}'.format(epoch+1, epochs, l/(i+1), end='\r'))
    else:
        for epoch in range(epochs): #this part is based on [3,4]
            l = 0.0
            for i, x in enumerate(train_loader):
                if isinstance(x, list):
                    x = x[0]
                noise = torch.normal(0,sigma_in, (nChan,imSizeH,imSizeV))
                x = scale*(x/div)-offset   #normalise between [-1,1]
                x = x+noise
                x = x.to(device)  
                optimizer.zero_grad()
                x_hat = model(x, sigma_en)
                loss = ((x - x_hat)**2).sum() + model.encoder.kl #b
                loss.backward()
                optimizer.step()
                l += loss.detach().cpu().item()
            print('Epoch: {}/{}, Loss: {:.3f}'.format(epoch+1, epochs, l/(i+1), end='\r'))
            
###############################################################################

########## Functions to get training data #####################################
def fnc_getTrain_loader(selector):    
    if selector == 0:
        data = CelebA()
        train_loader = DataLoader(dataset=data.train, batch_size=256, shuffle=True, num_workers=8, drop_last=True)
        t.toc()
    elif selector == 1:
        cifar_trainset = datasets.CIFAR10(root='../DATA', train=True, download=True, transform=ToTensor())
        train_loader = DataLoader(dataset=cifar_trainset, batch_size=256, shuffle=True, num_workers=8, drop_last=True) 
        t.toc()
    else:
        w, t_s, t_e, N, M = 5, -1, 1, 172, 8*512            #frequency, start time, end time, data of track, number of tracks
        ft, t_ax = sinc(w, t_s, t_e, N, M)                  #get raw data
        data, offset, scale = preprocess(ft, t_ax)          #preprocess raw data. Max of data in [-1, 1]
        train_loader = DataLoader(dataset=data, batch_size=256, shuffle=True, num_workers=8, drop_last=True) 
        t.toc()
    return train_loader

#function to get sinc data
def sinc(w, t_s, t_e, N, M):
    t_ax = 2*torch.rand(M, 1, N, 1) - 1     
    t_ax = torch.sort(t_ax, dim=2)
    t_ax = t_ax[0]
    f_eval = torch.sinc(w*t_ax) 
    return f_eval, t_ax

#function to preprocess sinc data
def preprocess(ft, t):
    offset = torch.min(ft)
    ft = ft - offset                #make minimum zero for both t and f(t)
    scale = torch.max(ft)
    ft = ft/scale                   #make max one for both t and f(t)
    ft = 2*ft - 1                   #from t, f(t) in [0,1] to t, f(t) in [-1, 1]
    out = torch.cat((ft, t),-1)
    return out, offset, scale          #return data, offset and scale. Note that the 2* and -1 are not returned
###############################################################################

###############################################################################
def fnc_getStore_and_Load(selector):
    if selector==0: #CelebA
        #Save/load paths for CelebA, left column
        path_ll = ["Models/modelL1-sigma-1_0", "Models/modelL1-sigma-0_5", "Models/modelL1-sigma-0_1"]   #Load locations
        path_sl = ["Models/modelL1-sigma-1_0", "Models/modelL1-sigma-0_5", "Models/modelL1-sigma-0_1"]   #Store locations
        path_lol = ["Optim/modelL1-sigma-1_0", "Optim/modelL1-sigma-0_5", "Optim/modelL1-sigma-0_1" ]    #Load locations  
        path_sol = ["Optim/modelL1-sigma-1_0", "Optim/modelL1-sigma-0_5", "Optim/modelL1-sigma-0_1" ]    #Store locations

        #Save/load paths for CelebA, right column
        path_lr = ["Models/modelR1-sigma-1_0", "Models/modelR1-sigma-0_5", "Models/modelR1-sigma-0_0"]   #Load locations
        path_sr = ["Models/modelR1-sigma-1_0", "Models/modelR1-sigma-0_5", "Models/modelR1-sigma-0_0"]   #Store locations
        path_lor = ["Optim/modelR1-sigma-1_0", "Optim/modelR1-sigma-0_5", "Optim/modelR1-sigma-0_0" ]    #Load locations  
        path_sor = ["Optim/modelR1-sigma-1_0", "Optim/modelR1-sigma-0_5", "Optim/modelR1-sigma-0_0" ]    #Store locations
    
    elif selector == 1: #Cifar
        #Save/load paths for CIFAR, left column
        path_ll = ["Models/modelL2-sigma-1_0", "Models/modelL2-sigma-0_5", "Models/modelL2-sigma-0_1"]   #Load locations
        path_sl = ["Models/modelL2-sigma-1_0", "Models/modelL2-sigma-0_5", "Models/modelL2-sigma-0_1"]   #Store locations
        path_lol = ["Optim/modelL2-sigma-1_0", "Optim/modelL2-sigma-0_5", "Optim/modelL2-sigma-0_1" ]    #Load locations  
        path_sol = ["Optim/modelL2-sigma-1_0", "Optim/modelL2-sigma-0_5", "Optim/modelL2-sigma-0_1" ]    #Store locations
        
        #Save/load paths for CIFAR, right column
        path_lr = ["Models/modelR2-sigma-1_0", "Models/modelR2-sigma-0_5", "Models/modelR2-sigma-0_0"]   #Load locations
        path_sr = ["Models/modelR2-sigma-1_0", "Models/modelR2-sigma-0_5", "Models/modelR2-sigma-0_0"]   #Store locations
        path_lor = ["Optim/modelR2-sigma-1_0", "Optim/modelR2-sigma-0_5", "Optim/modelR2-sigma-0_0" ]    #Load locations  
        path_sor = ["Optim/modelR2-sigma-1_0", "Optim/modelR2-sigma-0_5", "Optim/modelR2-sigma-0_0" ]    #Store locations
    
    else: #Sinc
        #Save/load paths for SINC, left column
        path_ll = ["Models/modelL3-sigma-1_0", "Models/modelL3-sigma-0_5", "Models/modelL3-sigma-0_1"]   #Load locations
        path_sl = ["Models/modelL3-sigma-1_0", "Models/modelL3-sigma-0_5", "Models/modelL3-sigma-0_1"]   #Store locations    
        path_lol = ["Optim/modelL3-sigma-1_0", "Optim/modelL3-sigma-0_5", "Optim/modelL3-sigma-0_1" ]    #Load locations  
        path_sol = ["Optim/modelL3-sigma-1_0", "Optim/modelL3-sigma-0_5", "Optim/modelL3-sigma-0_1" ]    #Store locations
    
        #Save/load paths for SINC, right column
        path_lr = ["Models/modelR3-sigma-1_0", "Models/modelR3-sigma-0_5", "Models/modelR3-sigma-0_0"]   #Load locations
        path_sr = ["Models/modelR3-sigma-1_0", "Models/modelR3-sigma-0_5", "Models/modelR3-sigma-0_0"]   #Store locations

        path_lor = ["Optim/modelR3-sigma-1_0", "Optim/modelR3-sigma-0_5", "Optim/modelR3-sigma-0_0" ]    #Load locations  
        path_sor = ["Optim/modelR3-sigma-1_0", "Optim/modelR3-sigma-0_5", "Optim/modelR3-sigma-0_0" ]    #Store locations
    return path_ll, path_sl, path_lol, path_sol, path_lr, path_sr, path_lor, path_sor

###############################################################################

###############################################################################
#some general model settings
sigma_in1 = torch.tensor(0)                 #input standard deviation, left column
sigma_enc1 = torch.tensor([1.0, 0.5, 0.1])  #encoder standard deviation, left column
sigma_in2 = torch.tensor([1.0, 0.5, 0.0])   #input standard deviation, right column
sigma_enc2 = torch.tensor(0.5)              #encoder standard deviation, right column

latent_dims = [64, 64, 1]                   #latent dimensions
imSizeH = [64, 32, 172]                     #image size 1 (horizontal)
imSizeV = [64, 32, 2]                       #image size 2 (vertical)
nChan = [3, 3, 1]                           #number of channels

#start timer
t.tic()

for k in select:    #k denotes the data 
    print("the selected model is {}. Note: 0=celeba, 1=cifar, 2=sinc".format(k))
    print("Step 1: Loading data")
    train_loader = fnc_getTrain_loader(k)
    path_ll, path_sl, path_lol, path_sol, path_lr, path_sr, path_lor, path_sor = fnc_getStore_and_Load(k)
    
    print(" ")
    print("training for left column")
    for i in range(3): #i denotes the settings for sigma_enc
        model = fnc_get_model(latent_dims=latent_dims[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        optimizer = Adam(model.parameters(), lr=lr[k])
        if load_old[k] == 1:
            model.load_state_dict(torch.load(path_ll[i]))
            optimizer.load_state_dict(torch.load(path_lol[i]))  
        
        model.train()
        fnc_train(epochs=epochs[k], sigma_en=sigma_enc1[i], sigma_in=sigma_in1, norm=norm[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        
        torch.save(model.state_dict(), path_sl[i])           #Save current state of model 
        torch.save(optimizer.state_dict(), path_sol[i])      #Save current state of optimiser
        print("Saved model for data {}: sigma_enc={}, sigma_in={}".format(k, sigma_enc1[i], sigma_in1))
        print(" ")
    print("Done training for left column")

    print(" ")
    
    print("Training for right column")
    for i in range(3): #i denotes the settings for sigma_in
        model = fnc_get_model(latent_dims=latent_dims[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        optimizer = Adam(model.parameters(), lr=lr[k])
        if load_old[k] == 1:
            model.load_state_dict(torch.load(path_lr[i]))
            optimizer.load_state_dict(torch.load(path_lor[i]))  
        
        model.train()
        fnc_train(epochs=epochs[k], sigma_en=sigma_enc2, sigma_in=sigma_in2[i], norm=norm[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        
        torch.save(model.state_dict(), path_sr[i])           #Save current state of model
        torch.save(optimizer.state_dict(), path_sor[i])      #Save current state of optimiser
        
        print("Saved model for data {}: sigma_enc={}, sigma_in={}".format(k, sigma_enc2, sigma_in2[i]))
        print(" ")
    print("Done training for right column")    
    
    print(" ")
    