
import torch
import torch.nn as nn
import torch.nn.functional as F
from SwinTransformer import SwinTransformerBlock

class spikyUNet(nn.Module):
    def __init__(self, IMG_CHANNELS=3, IMG_HEIGHT=512, IMG_WIDTH=512):
        super(spikyUNet, self).__init__()
        
        # Contraction path (same number of filters as TensorFlow)
        self.conv1 = self.contracting_block(IMG_CHANNELS, 64)

        self.swin_block1 = SwinTransformerBlock(
            dim=64,  # Number of channels in c1
            input_resolution=(IMG_HEIGHT, IMG_WIDTH),  # Height and width after conv1
            num_heads=8,  # Adjust based on your model complexity
            window_size=5,  # Typical window size
            mlp_ratio=4.0
        )
        self.conv2 = self.contracting_block(64, 128)
        self.swin_block2 = SwinTransformerBlock(
            dim=128,  # Number of channels in c1
            input_resolution=(IMG_HEIGHT, IMG_WIDTH),  # Height and width after conv1
            num_heads=8,  # Adjust based on your model complexity
            window_size=5,  # Typical window size
            mlp_ratio=4.0
        )
        self.conv3 = self.contracting_block(128, 256)
        self.swin_block3 = SwinTransformerBlock(
            dim=256,  # Number of channels in c1
            input_resolution=(IMG_HEIGHT, IMG_WIDTH),  # Height and width after conv1
            num_heads=8,  # Adjust based on your model complexity
            window_size=5,  # Typical window size
            mlp_ratio=4.0
        )
        self.conv4 = self.contracting_block(256, 512)
        self.swin_block4 = SwinTransformerBlock(
            dim=512,  # Number of channels in c1
            input_resolution=(IMG_HEIGHT, IMG_WIDTH),  # Height and width after conv1
            num_heads=8,  # Adjust based on your model complexity
            window_size=5,  # Typical window size
            mlp_ratio=4.0
        )
        self.conv5 = self.contracting_block(512, 1024)
        self.swin_block5 = SwinTransformerBlock(
            dim=1024,  # Number of channels in c1
            input_resolution=(IMG_HEIGHT, IMG_WIDTH),  # Height and width after conv1
            num_heads=8,  # Adjust based on your model complexity
            window_size=5,  # Typical window size
            mlp_ratio=4.0
        )
        
        # Expansive path
        self.upconv6 = self.expansive_block(1024, 512)
        self.upconv6_2 = self.expansive_block2(1024, 512)
        self.upconv6_3 = self.expansive_block2(512, 512)        
        self.upconv7 = self.expansive_block(512, 256)
        self.upconv7_2 = self.expansive_block2(512, 256)
        self.upconv7_3 = self.expansive_block2(256, 256)
        self.upconv8 = self.expansive_block(256, 128)
        self.upconv8_2= self.expansive_block2(256, 128)
        self.upconv8_3= self.expansive_block2(128, 128)        
        self.upconv9 = self.expansive_block(128, 64)
        self.upconv9_2 = self.expansive_block2(128, 64)
        self.upconv9_3 = self.expansive_block2(64, 64)

        
        # Final layer to produce the output
        self.final_layer = nn.Conv2d(64, 3, kernel_size=1)
        self.dropout_1 = nn.Dropout(p=0.1)  # For smaller layers
        self.dropout_2 = nn.Dropout(p=0.2)

    def contracting_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),  # Dropout depending on the layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # Dropout depending on the layer
        )
        return block
    def expansive_block2(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        return block

    def forward(self, x):
        # Contraction path 
        
        c1 = self.conv1(x)
        ct1 = self.swin_block1(c1)
        p1 = F.max_pool2d(ct1, kernel_size=2, stride=2)

        c2 = self.conv2(p1)
        ct2 = self.swin_block2(c2)
        p2 = F.max_pool2d(ct2, kernel_size=2, stride=2)

        c3 = self.conv3(p2)
        ct3 = self.swin_block3(c3)
        p3 = F.max_pool2d(ct3, kernel_size=2, stride=2)

        c4 = self.conv4(p3)
        ct4 = self.swin_block4(c4)
        p4 = F.max_pool2d(ct4, kernel_size=2, stride=2)

        c5 = self.conv5(p4)
        ct5 = self.swin_block5(c5)

        # Expansive path
        u6 = self.upconv6(ct5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.upconv6_2(u6)
        c6 =   self.dropout_1(c6)
        c6 = self.upconv6_3(c6)
        ct6 = self.swin_block4(c6)

        u7 = self.upconv7(ct6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.upconv7_2(u7)
        c7 =   self.dropout_1(c7)
        c7 = self.upconv7_3(c7)
        ct7 = self.swin_block3(c7)
        
        u8 = self.upconv8(ct7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.upconv8_2(u8)
        c8 =   self.dropout_1(c8)
        c8 = self.upconv8_3(c8)
        ct8 = self.swin_block2(c8)
        
        u9 = self.upconv9(ct8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.upconv9_2(u9)
        c9 =   self.dropout_1(c9)
        c9 = self.upconv9_3(c9)
        
       
        final = self.final_layer(c9)  # Final layer with 1 channel output (like TensorFlow)
        
        output= torch.tanh(final)
      

        return output
