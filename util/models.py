from util.utils import *
from util.module import *

def aug_feature_dropout(input_feat, drop_rate=0.2):
    """
    dropout features for augmentation.
    args:
        input_feat: input features
        drop_rate: dropout rate
    returns:
        aug_input_feat: augmented features
    """
    aug_input_feat = copy.deepcopy(input_feat).squeeze(0)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    aug_input_feat = aug_input_feat.unsqueeze(0)

    return aug_input_feat


def aug_feature_shuffle(input_feat):
    """
    shuffle the features for fake samples.
    args:
        input_feat: input features
    returns:
        aug_input_feat: augmented features
    """
    fake_input_feat = input_feat[:, np.random.permutation(input_feat.shape[1]), :]
    return fake_input_feat

class Net(torch.nn.Module):
    def __init__(self, indim, K=2):
        super().__init__()
        self.conv1 = SGConv(indim, 64, K=K, cached=True)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        _ = self.conv1(x, edge_index, edge_weight)
        return self.conv1._cached_x.detach()

class EigenMLP(nn.Module):
    def __init__(self, hidden_dim, output_dim, u_dim, period=20):
        super(EigenMLP, self).__init__()

        self.period = period

        self.phi = nn.Sequential(nn.Linear(u_dim, 16), nn.ReLU(), nn.Linear(16, 16))
        self.psi = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, u_dim))

        self.mlp1 = nn.Linear(2*period, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        self.SSL_head = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.classifier = nn.Linear(hidden_dim, output_dim, bias=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, data):
        H = self.embedding(data)
        H = self.classifier(H)
        return F.log_softmax(H, dim=1)


    def embedding(self, data):
        e, u = data.e, data.u
        e = e * 100

        # u = self.psi(self.phi(u) + self.phi(-u))

        period_term = torch.arange(0, self.period, device=u.device).float()
        period_e = e.unsqueeze(1) * torch.pow(2, period_term) 
        fourier_e = torch.cat([torch.sin(period_e), torch.cos(period_e)], dim=-1)

        h = u @ fourier_e
        h = self.mlp1(h)
        h = F.relu(h)
        h = self.mlp2(h)

        return h

    def SSL_dis(self, e, u):
        a = e[np.random.permutation(e.shape[0])]

        z_1 = self.SSL_head(self.embedding(e, u).squeeze(0)).sum(1)
        z_2 = self.SSL_head(self.embedding(a, u).squeeze(0)).sum(1)
        logit = torch.cat((z_1, z_2), 0)
        return logit


class GCN(torch.nn.Module):
    def __init__(self, nin, nhid, nout, nlayers, dropout=0.5):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

        if nlayers == 1:
            self.layers.append(GCNConv(nin, nout))
        else:
            self.layers.append(GCNConv(nin, nhid)) 
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv(nhid, nhid)) 
            self.layers.append(GCNConv(nhid, nhid))  
        self.dropout = dropout
        self.classifier = nn.Linear(nhid, nout, bias=True)
        self.linkpredictor = nn.Linear(nhid, nhid, bias=True)
        self.SSL_head = nn.Linear(nhid, nhid, bias=True)
        self.proj = nn.Sequential(nn.Linear(nhid, nhid), nn.PReLU(), nn.Linear(nhid, nhid))

        self.initialize()

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_weight)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
    
    def embedding(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv(x, edge_index, edge_weight)
        return x
    
    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.linkpredictor(x)
        return x
    
    def decode(self, x, edge_label_index):
        return (x[edge_label_index[0]] * x[edge_label_index[1]]).sum(dim=-1)


    def conv(self, x, edge_index, edge_weight=None):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_weight)
        return x
    
    def SSL_dis(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x_aug = aug_feature_dropout(x)
        x_shuffle = aug_feature_shuffle(x_aug)

        z_1 = self.SSL_head(self.conv(x_aug, edge_index, edge_weight).squeeze(0)).sum(1)
        z_2 = self.SSL_head(self.conv(x_shuffle, edge_index, edge_weight).squeeze(0)).sum(1)
        logit = torch.cat((z_1, z_2), 0)
        return logit
    
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes, bias = True)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

