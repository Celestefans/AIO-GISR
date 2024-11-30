import torch
import torch.nn as nn
# from Component import *
from .Component import *
import torch.nn.functional as F
import numbers
from einops import rearrange 
from torch.distributions.normal import Normal
import numpy as np

class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 42,

        num_blocks = [4,6,6,8],  
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        ):

        super(Restormer, self).__init__()
        self.upMode = 'bilinear'  


        self.patch_embed1 = OverlapPatchEmbed(4, dim) # depth
        self.patch_embed2 = OverlapPatchEmbed(2, dim) # mri
        self.patch_embed3 = OverlapPatchEmbed(5, dim) # pan

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        ###########################
            
        self.output1 = nn.Conv2d(int(dim*2**1), 1, kernel_size=3, stride=1, padding=1, bias=bias) # depth
        self.output2 = nn.Conv2d(int(dim*2**1), 1, kernel_size=3, stride=1, padding=1, bias=bias) # mri
        self.output3 = nn.Conv2d(int(dim*2**1), 4, kernel_size=3, stride=1, padding=1, bias=bias) # pan



    # def forward(self, ms, pan):
    def forward(self, inp_ms, inp_pan):
        _, C1, _, _ = inp_ms.shape
        _, C2, H, W = inp_pan.shape
        inp_ms = F.interpolate(inp_ms, size = (H,W), mode = self.upMode)
        # inp_tar = F.interpolate(inp_tar, size = (H,W), mode = 'bilinear')
        inp_img = torch.concat([inp_ms, inp_pan],dim=1)
        if C1+C2 == 4:  # depth
            inp_enc_level1 = self.patch_embed1(inp_img)
        elif C1+C2 ==2: # mri
            inp_enc_level1 = self.patch_embed2(inp_img)
        elif C1+C2 == 5:          # pan
            inp_enc_level1 = self.patch_embed3(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)       
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        if C1+C2 == 4:  # depth
            out_dec_level1 = self.output1(out_dec_level1) + inp_ms
        elif C1+C2 ==2: # mri
            out_dec_level1 = self.output2(out_dec_level1) + inp_ms
        else:           # pan
            out_dec_level1 = self.output3(out_dec_level1) + inp_ms

        return out_dec_level1


    

if __name__== "__main__":
    # lr1,lr2,lr3 = torch.rand(4,1,64,64),torch.rand(4,1,60,60),torch.rand(4,4,32,32)
    # gt1,gt2,gt3 = torch.rand(4,1,256,256),torch.rand(4,1,240,240),torch.rand(4,4,128,128)
    # gi1,gi2,gi3 = torch.rand(4,3,256,256),torch.rand(4,1,240,240),torch.rand(4,1,128,128)

    model = Restormer(inp_channels=9, out_channels=8, dim = 22, num_blocks=[3, 4, 4, 6])
    # model = Restormer(inp_channels=9, out_channels=8, dim = 16, num_blocks=[2, 3, 3, 5])
    # gen1 = model(lr1,gi1)
    # gen2 = model(lr2,gi2)
    # gen3 = model(lr3,gi3)

    # print(gen1.shape,gen2.shape,gen3.shape)
    # model = PAN_PromptBlock(prompt_size=512)
    # model = Restormer_pan(inp_channels=9, out_channels=8, dim = 16,num_blocks=[2, 2, 2, 3])
    # x = torch.rand(4, 8, 64, 64)
    # prompt_gen = MS_PromptBlock() 
    # prompt = prompt_gen(x)
    # print(prompt.shape)
    
    # inp_ms = torch.rand(4, 8, 128, 128)
    # inp_pan = torch.rand(4, 1, 512, 512)
    # inp_ms = torch.rand(4, 4, 32, 32)
    # inp_pan = torch.rand(4, 1, 128, 128)


    # B,C,H,W = inp_ms.shape
    # ms_prompt_generate = RIN(8)
    # pan_prompt_generate = PAN_PromptBlock_multiscale(prompt_size=128)
    # pan_in = Spatial_prompt_in()

    # ms_prompt = ms_prompt_generate(inp_ms)
    # pan_prompt_list = pan_prompt_generate(inp_pan)

    # SCgate = SCgate(dim=16, channel_prompt_dim=256)
    # feature = torch.rand(4, 16, 128, 128)

    # feature = SCgate(feature, pan_prompt_list[0], ms_prompt)

    # print(pan_prompt_list[0].shape)
    # print(pan_prompt_list[1].shape)
    # print(pan_prompt_list[2].shape)
    # print(pan_prompt_list[3].shape)



    # output = model(inp_ms, inp_pan)
    # print(output.shape)
    print(sum(p.numel() for p in model.parameters() )/1e6)