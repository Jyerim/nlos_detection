import torch
import torch.nn as nn

class MaxPool4d(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=1):
        super(MaxPool4d, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
    def forward(self, x):
        
        bsz, inch, ha, wa, hb, wb = x.size()
        out1 = x.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.maxpool(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()
        
        bsz, inch, ha, wa, hb, wb = out1.size()
        out2 = out1.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.maxpool(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        return out2
        
class Conv4d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Conv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels//2, kernel_size=(3,3), stride=stride[:2],
                                bias=bias, padding=(1,1))
            
        self.conv2 = nn.Conv2d(out_channels//2, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

    def forward(self, x):

        bsz, inch, ha, wa, hb, wb = x.size()
        out1 = x.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = out1.size()
        out2 = out1.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        return out2