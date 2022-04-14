"""
Date:   25-03-2022
Author: Austin, Mustafa, Richard, Dimme
Based on: [1], [2], [3], [4], [5], [6]
Descr:      Part of Code for Table 1 of the paper " Variational Autoencoders: A Harmonic Perspective"

            Store weights from (1) decoders and (2) encoders from models made using Step1-model_training.py
                as a .mat file. This .mat file can be used to estimate Lipschitz constant L using code from [4]
            
            note: does not store biases. For the left column of table 1, decoders are stored (L). For the right column, encoders are stored (R)
                    
Sources:
    [1] examples from https://github.com/didriknielsen/survae_flow
    [2] https://avandekleut.github.io/vae/
    [3] The paper: Variational Autoencoders: A Harmonic Perspective
    [4] Lab 8 of CS4240-Deep Learning course
    [5] https://medium.com/dataseries/variational-autoencoder-with-pytorch-2d359cbf027b
    [6] LipSDP: https://github.com/arobey1/LipSDP
Notes: 
    (a) you need to have trained the models before using this file, and stored at the proper location.
    (b) you need to have a folder Weights inside the directory where this .py file is stored.
"""

import torch 
from torch import nn  
from scipy.io import savemat
import numpy as np

#User settings
select = [2]   #select which models, 0: CelebA, 1: Cifar, 2: Sinc           

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
#Edited version of code presented in [4] 
def extract_weights(net):
    weights = []
    bias = []
    for param_tensor in net.state_dict():
        tensor = net.state_dict()[param_tensor].detach().numpy().astype(np.float64)
        if 'weight' in param_tensor:
            weights.append(tensor)
        if 'bias' in param_tensor:
            bias.append(tensor)
    return weights, bias
###############################################################################

###############################################################################
def fnc_getStore_and_Load(selector):
    if selector==0: #CelebA
        #Save/load paths for CelebA, left column
        path_ll = ["Models/modelL1-sigma-1_0", "Models/modelL1-sigma-0_5", "Models/modelL1-sigma-0_1"]                          #Load locations
        path_swl = ["Weights/WeightsL1-sigma-1_0.mat", "Weights/WeightsL1-sigma-0_5.mat", "Weights/WeightsL1-sigma-0_1.mat"]    #Store locations
        
        #Save/load paths for CelebA, right column
        path_lr = ["Models/modelR1-sigma-1_0", "Models/modelR1-sigma-0_5", "Models/modelR1-sigma-0_0"]                          #Load locations
        path_swr = ["Weights/WeightsR1-sigma-1_0.mat", "Weights/WeightsR1-sigma-0_5.mat", "Weights/WeightsR1-sigma-0_0.mat"]    #Store locations
    
    elif selector == 1: #Cifar
        #Save/load paths for CIFAR, left column
        path_ll = ["Models/modelL2-sigma-1_0", "Models/modelL2-sigma-0_5", "Models/modelL2-sigma-0_1"]                          #Load locations
        path_swl = ["Weights/WeightsL2-sigma-1_0.mat", "Weights/WeightsL2-sigma-0_5.mat", "Weights/WeightsL2-sigma-0_1.mat"]    #Store locations
 
        #Save/load paths for CIFAR, right column
        path_lr = ["Models/modelR2-sigma-1_0", "Models/modelR2-sigma-0_5", "Models/modelR2-sigma-0_0"]                          #Load locations
        path_swr = ["Weights/WeightsR2-sigma-1_0.mat", "Weights/WeightsR2-sigma-0_5.mat", "Weights/WeightsR2-sigma-0_0.mat"]    #Store locations
    
    else: #Sinc
        #Save/load paths for SINC, left column
        path_ll = ["Models/modelL3-sigma-1_0", "Models/modelL3-sigma-0_5", "Models/modelL3-sigma-0_1"]                          #Load locations
        path_swl = ["Weights/WeightsL3-sigma-1_0.mat", "Weights/WeightsL3-sigma-0_5.mat", "Weights/WeightsL3-sigma-0_1.mat"]    #Store locations
  
        #Save/load paths for SINC, right column
        path_lr = ["Models/modelR3-sigma-1_0", "Models/modelR3-sigma-0_5", "Models/modelR3-sigma-0_0"]                          #Load locations
        path_swr = ["Weights/WeightsR3-sigma-1_0.mat", "Weights/WeightsR3-sigma-0_5.mat", "Weights/WeightsR3-sigma-0_0.mat"]    #Store locations
    return path_ll, path_lr, path_swl, path_swr
###############################################################################
       
latent_dims = [64, 64, 1]                   #latent dimensions
imSizeH = [64, 32, 172]                     #image size 1 (horizontal)
imSizeV = [64, 32, 2]                       #image size 2 (vertical)
nChan = [3, 3, 1]                           #number of channels

#Loop through all six models
for k in select:
    print("the selected model is {}. Note: 0=celeba, 1=cifar, 2=sinc".format(k))
    print(" ")
    path_ll, path_lr, path_swl, path_swr = fnc_getStore_and_Load(k)

    print("storing weights for left column")    
    for j in range(3):
        model = fnc_get_model(latent_dims=latent_dims[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        model.load_state_dict(torch.load(path_ll[j]))
        model.eval()
        
        weights2, bias = extract_weights(model.decoder)     
        data = {'weights': np.array(weights2, dtype=object)}
        savemat(path_swl[j], data)
        print("Stored weights from model {} at location {}".format(path_ll[j], path_swl[j]))
        print(" ")
    
    print("storing weights for right column")
    for j in range(3):
        model = fnc_get_model(latent_dims=latent_dims[k], nChan=nChan[k], imSizeH=imSizeH[k], imSizeV=imSizeV[k])
        model.load_state_dict(torch.load(path_lr[j]))
        model.eval()
        
        weights, _ = extract_weights(model.encoder) 
        data = {'weights': np.array(weights, dtype=object)}
        savemat(path_swr[j], data)
        print("Stored weights from model {} at location {}".format(path_lr[j], path_swr[j]))
        print(" ")
    
    