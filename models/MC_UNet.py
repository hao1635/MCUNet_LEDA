import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import copy
from torch.nn.parameter import Parameter
import numbers
from einops import rearrange
from torch.nn import init
import os
import util.util as util
import ipdb
from .mamba import SS2D

    

class Mamba_Block(nn.Module):
    def __init__(self,input_channel,output_channel,num_heads=8):
        super(Mamba_Block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        base_d_state=4
        d_state=int(base_d_state * num_heads)
        self.attention_s=SS2D(d_model=input_channel, d_state=d_state,expand=2,dropout=0, **kwargs)

    def forward(self, inputs):
        attn_s=self.attention_s(inputs)
        inputs_attn=inputs+attn_s

        return inputs_attn

class ResConv_block(nn.Module):
    def __init__(self,input_channel,middle_channel,output_channel,res=True):
        super(ResConv_block,self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.conv_1=nn.Conv2d(input_channel,middle_channel,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_2=nn.Conv2d(middle_channel,output_channel,kernel_size=3,stride=1,padding=1,bias=False)
        if self.input_channel != self.output_channel:
            self.shortcut=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=1,padding=0,stride=1,groups=1,bias=False)
        self.res=res
        self.act=nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        conv_S=self.act(self.conv_1(inputs))
        conv_S=self.act(self.conv_2(conv_S))

        if self.input_channel == self.output_channel:
            identity_out=inputs
        else:
            identity_out=self.shortcut(inputs)

        if self.res:
            output=conv_S+identity_out
        else:
            output=conv_S

        return output


class ME_Block(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=8,res=True):
        super(ME_Block,self).__init__()
        self.esaublock=nn.Sequential(
            Mamba_Block(in_channels,in_channels,num_heads=num_heads),
            ResConv_block(in_channels,in_channels,out_channels,res=res),
        )
    def forward(self,x):
        return self.esaublock(x)
      
               
class Down(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(Down,self).__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d((2,2), (2,2)),
            ME_Block(in_channels,out_channels,num_heads=num_heads,res=res)
        )
            
    def forward(self, x):
        return self.encoder(x)

    
class LastDown(nn.Module):

    def __init__(self, in_channels, out_channels,num_heads=8,res=True):
        super(LastDown,self).__init__()

        self.encoder = nn.Sequential(
            nn.MaxPool2d((2,2), (2,2)),
            Mamba_Block(in_channels,in_channels,num_heads=num_heads),
            ResConv_block(in_channels,2*in_channels,out_channels,res=res),
            )
    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels,res_unet=True,trilinear=True, num_heads=8,res=True):
        super(Up,self).__init__()
        self.res_unet=res_unet
        if trilinear:
            self.up = nn.Upsample(scale_factor=(2,2), mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels , kernel_size=2, stride=2)
        
        self.conv = ME_Block(in_channels, out_channels, num_heads=num_heads,res=res)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        
        if self.res_unet:
            x=x1+x2
        else:
            x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels,decouple=None,bn=True,res=True,activation=False):
        super(SingleConv,self).__init__()
        self.act=activation
        self.conv =nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.activation = nn.LeakyReLU(inplace=True)
        

    def forward(self, x):
        x=self.conv(x)
        if self.act==True:
            x=self.activation(x)
        return x
        


class ME_UNet(nn.Module):
    def __init__(self,opt,in_channels=1,out_channels=1,n_channels=64,num_heads=[1,2,4,8],res=True):
        super(ME_UNet,self).__init__()
        #ipdb.set_trace()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels
        
        self.firstconv=SingleConv(in_channels, n_channels//2,res=res,activation=True)
        self.enc1 = ME_Block(n_channels//2, n_channels,num_heads=num_heads[0],res=res) 
        
        self.enc2 = Down(n_channels, 2 * n_channels,num_heads=num_heads[1],res=res)
        
        self.enc3 = Down(2 * n_channels, 4 * n_channels,num_heads=num_heads[2],res=res)
        
        self.enc4 = LastDown(4 * n_channels, 4 * n_channels,num_heads=num_heads[3],res=res)
        
        self.dec1 = Up(4 * n_channels, 2 * n_channels,num_heads=num_heads[2],res=res)
        
        self.dec2 = Up(2 * n_channels, 1 * n_channels,num_heads=num_heads[1],res=res)
        
        self.dec3 = Up(1 * n_channels, n_channels//2,num_heads=num_heads[0],res=res)

        self.out1 = SingleConv(n_channels//2,n_channels//2,res=res,activation=True)
        
        self.out2 = SingleConv(n_channels//2,out_channels,res=res,activation=False)


    
    def forward(self, x):
        b, c, h, w = x.size()

        x =self.firstconv(x)
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        output = self.dec1(x4, x3)
        output = self.dec2(output, x2)
        output = self.dec3(output, x1)
        output = self.out1(output)

        output = self.out2(output)

        return output