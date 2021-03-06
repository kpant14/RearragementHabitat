'''The models for building the Transformer
This definitions are inspired from :
    * https://github.com/jadore801120/attention-is-all-you-need-pytorch
'''
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformer.Layers import EncoderLayer, DecoderLayer
import torchvision
from torchvision import datasets, models, transforms
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import math
from typing import Tuple
import torch
from torch import nn, Tensor, prelu, relu
from torchsummary import summary

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncoderMap(nn.Module):
    ''' The encoder of the planner.
    '''

    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, dropout,):
        '''
        Intialize the encoder.
        :param n_layers: Number of layers of attention and fully connected layer.
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of encoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN
        :param dropout: The value to the dropout argument.
        '''
        super().__init__()
        # Convert the image to and input embedding.
        # NOTE: This is one place where we can add convolution networks.
        # Convert the image to linear model

        # NOTE: Padding of 3 is added to the final layer to ensure that 
        # the output of the network has receptive field across the entire map.
        # NOTE: pytorch doesn't have a good way to ensure automatic padding. This
        # allows only for a select few map sizes to be solved using this method.
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(16, d_model, kernel_size=5, stride=5, padding=3),
            # nn.MaxPool2d(kernel_size=2),
            # nn.ReLU(),
            # nn.Conv2d(d_model, d_model, kernel_size=3),
        )
        
        self.reorder_dims = Rearrange('b c h w -> b (h w) c')
        self.position_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, input_map):
        '''
        The input of the Encoder should be of dim (b, c, h, w).
        :param input_map: The input map for planning.
        '''
        enc_output = self.to_patch_embedding(input_map)
        enc_output = self.reorder_dims(enc_output)
        enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, slf_attn_mask=None)
        return enc_output



class Transformer(nn.Module):
    ''' A Transformer module
    '''
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, dropout):
        '''
        Initialize the Transformer model.
        :param n_layers: Number of layers of attention and fully connected layers
        :param n_heads: Number of self attention modules.
        :param d_k: Dimension of each Key.
        :param d_v: Dimension of each Value.
        :param d_model: Dimension of input/output of decoder layer.
        :param d_inner: Dimension of the hidden layers of position wise FFN1
        :param dropout: The value of the dropout argument.
        '''
        super().__init__()

        self.encoder_map = EncoderMap(
            n_layers=n_layers, 
            n_heads=n_heads, 
            d_k=d_k, 
            d_v=d_v, 
            d_model=d_model, 
            d_inner=d_inner, 
            dropout=dropout, 
        )
        # self.depth_model = nn.Sequential(
        #     nn.Conv2d(1, 64, kernel_size=8, stride=3),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=7, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=6, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=5, stride=2),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(1152,512),
        #     nn.ReLU(),

        # )

        # resnet18 = models.resnet18(pretrained=True)
        # #Taking features from the second last layer of the Resnet18 Network
        # self.rgb_model = torch.nn.Sequential(*(list(resnet18.children())[:-1]))
        # Last linear layer for prediction
        self.class_pred = nn.Sequential(
            Rearrange('b c d_model -> (b c) d_model 1 1'),
            nn.Conv2d(512, 2, kernel_size=1),
            
            #nn.Conv2d(512, 1, kernel_size=1),

            Rearrange('bc d 1 1 -> bc d')
        )

    def forward(self, input_map, rgb, depth):
        '''
        The callback function.
        :param input_map:
        :param goal: A 2D torch array representing the goal.
        :param start: A 2D torch array representing the start.
        :param cur_index: The current anchor point of patch.
        '''
        enc_output = self.encoder_map(input_map)
        #print(summary(self.encoder_map,(3,480,480))) 
        # rgb_output  = self.rgb_model(rgb)
        # rgb_output = rearrange(rgb_output,'b c 1 1 -> b 1 c')
        # depth_output = self.depth_model(depth)
        # depth_output = rearrange(depth_output,'b c-> b 1 c')
        # combined_output = torch.cat((enc_output,rgb_output,depth_output), dim = 1)
        # seq_logit = self.class_pred(combined_output)
        seq_logit = self.class_pred(enc_output)
        batch_size = input_map.shape[0]
        return rearrange(seq_logit, '(b c) d -> b c d', b=batch_size)
        #return rearrange(seq_logit, '(b c) d -> b d c', b=batch_size)






        