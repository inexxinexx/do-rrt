import torch
import torch.nn as nn

def Linears(in_dim, hidden_dims, out_dim):
    blocks = []

    in_d = in_dim
    for hidden_d in hidden_dims:
        block = nn.Sequential(
            nn.Linear(in_d, hidden_d),
            # nn.LayerNorm(hidden_d),
            nn.ReLU(True)
        )
        blocks.append(block)
        in_d = hidden_d

    # append the last layer
    blocks.append(nn.Linear(in_d, out_dim))

    return nn.Sequential(*blocks)

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

# (s, g) -> s'
class Actor(BaseNetwork):
    def __init__(self, state_dim, goal_dim, pos_min, pos_max, hidden_dims=[256, 256], init_weights=True):
        super(Actor, self).__init__()
        
        self.pos_min = pos_min
        self.pos_max = pos_max

        self.net = Linears(state_dim + goal_dim, hidden_dims, state_dim)

        if init_weights:
            self.init_weights(init_type='kaiming')

    def forward(self, state, goal):
        x = torch.cat([state, goal], 1)
        x = self.net(x)
        x = torch.tanh(x)
        x = (self.pos_max - self.pos_min) / 2 * x + (self.pos_max + self.pos_min) / 2 # scale (-1, 1) to (self.pos_min, self.pos_max)
        return x                                            # next state