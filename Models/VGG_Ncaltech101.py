from unicodedata import numeric
import torch.nn as nn
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
import torch
torch.backends.cudnn.enabled = False
class VGGNet(nn.Module):
    def __init__(self,num_classes=10):
        super(VGGNet,self).__init__()
        conv = []
        #conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
        conv.extend(VGGNet.conv3x3(2, 64,3,1,1))
        conv.extend(VGGNet.conv3x3(64, 128,3,1,1))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        conv.extend(VGGNet.conv3x3(128, 256,3,1,1))
        conv.extend(VGGNet.conv3x3(256, 256,3,1,1))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        conv.extend(VGGNet.conv3x3(256, 512,3,1,1))
        conv.extend(VGGNet.conv3x3(512, 512,3,1,1))
        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        conv.extend(VGGNet.conv3x3(512, 512,3,1,1))
        conv.extend(VGGNet.conv3x3(512, 512,3,1,1))
        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        self.conv = nn.Sequential(*conv)
        w=int(48/2/2/2/2)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * w*w, num_classes)),

        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        ratio=int(x.shape[0]/2)
        #x=x[5:6,...]
        #x=x[::2,...]
        out_spikes = self.fc(self.conv(x)) # shape = [T, N, 110]
        return out_spikes

    @staticmethod
    def conv3x3(in_plane,out_plane,kernel_size,stride,padding):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
                nn.BatchNorm2d(out_plane)
            ),
            nn.ReLU(),
        ]

class VGGSNNNet(nn.Module):
    def __init__(self,num_classes=101):
        super(VGGSNNNet,self).__init__()
        conv = []
        #conv.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d(48)))
        conv.extend(VGGSNNNet.conv3x3(2, 64,3,1,1))
        conv.extend(VGGSNNNet.conv3x3(64, 128,3,1,1))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        conv.extend(VGGSNNNet.conv3x3(128, 256,3,1,1))
        conv.extend(VGGSNNNet.conv3x3(256, 256,3,1,1))

        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        conv.extend(VGGSNNNet.conv3x3(256, 512,3,1,1))
        conv.extend(VGGSNNNet.conv3x3(512, 512,3,1,1))
        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        conv.extend(VGGSNNNet.conv3x3(512, 512,3,1,1))
        conv.extend(VGGSNNNet.conv3x3(512, 512,3,1,1))
        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))

        self.conv = nn.Sequential(*conv)
        w=int(48/2/2/2/2)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.SeqToANNContainer(nn.Linear(512 * w*w, num_classes)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        ratio=int(x.shape[0]/2)
        x=x[5:6,...]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
        return out_spikes

    @staticmethod
    def conv3x3(in_plane,out_plane,kernel_size,stride,padding):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_plane,out_plane,kernel_size,stride,padding),
                nn.BatchNorm2d(out_plane)
            ),
            #neuron.MultiStepLIFNode(surrogate_function=surrogate.Trangle(),backend='cupy', decay_input=False)
            #nn.ReLU()
            IFSpike()
        ]

class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.5, gama=1.0):
        super(LIFSpike, self).__init__()
        self.act = ZIF.apply
        # self.k = 10
        # self.act = F.sigmoid
        self.thresh = thresh
        self.tau = tau
        self.gama = gama

    def forward(self, x):
        #print(x.shape)
        mem = 0
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * self.tau + x[t, ...]
            spike = self.act(mem - self.thresh, self.gama)
            # spike = self.act((mem - self.thresh)*self.k)
            mem = (1 - spike) * mem
            spike_pot.append(spike)
            #print(spike_pot)
        return torch.stack(spike_pot, dim=0)

class IFSpike(nn.Module):
    def __init__(self, scale=1.0,thresh=1.0,gama=1.0):
        super(IFSpike, self).__init__()
        self.act = ZIF.apply
        self.scale = scale
        self.gama = gama
        self.thresh=thresh

    def forward(self, x):
        spike_pot = []
        T = x.shape[0]
        mem = 0.5
        for t in range(T):
            mem = mem + x[t ,...]
            #print('mem',mem)
            spike = self.act(mem -self.thresh, self.gama)
            #print(spike)
            #mem = (1 - spike) * mem
            mem=mem-spike*self.thresh
            spike_pot.append(spike)
        return torch.stack(spike_pot,dim=0)
