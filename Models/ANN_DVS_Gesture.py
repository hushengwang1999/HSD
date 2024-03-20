from unicodedata import numeric
import torch.nn as nn
from spikingjelly.clock_driven import functional, surrogate, layer, neuron
import torch
torch.backends.cudnn.enabled = False

class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class VGG(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(VGG.conv3x3(2, channels))
        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        for i in range(4):
            conv.extend(VGG.conv3x3(channels, channels))
            conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(channels * 4 * 4, channels * 2 * 2)),
            nn.ReLU(),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(channels * 2 * 2, 110)),
            nn.ReLU(),
            layer.SeqToANNContainer(nn.Linear(110, 11)),
        )
        #self.vote = VotingLayer(10)

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
        #ratio=int(x.shape[0]/2)
       	x=x[5:15,...]
        out_spikes = self.fc(self.conv(x)) # shape = [T, N, 110]
        return out_spikes

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
            ),
            nn.ReLU()
        ]


class VGGSNNNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(VGG.conv3x3(2, channels))
        conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        for i in range(4):
            conv.extend(VGG.conv3x3(channels, channels))
            conv.append(layer.SeqToANNContainer(nn.AvgPool2d(2, 2)))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(2),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(channels * 4 * 4, channels * 2 * 2)),
            LIFSpike(),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(channels * 2 * 2, 110)),
            LIFSpike(),
            layer.SeqToANNContainer(nn.Linear(110, 11)),
        )
        #self.vote = VotingLayer(10)

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
        x=x[5:15,...]
        out_spikes = self.fc(self.conv(x))  # shape = [T, N, 110]
        #return self.vote(out_spikes.mean(0))
        return out_spikes

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
            ),
            LIFSpike()
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