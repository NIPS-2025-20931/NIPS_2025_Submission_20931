from utils import *
from torch_geometric.nn import global_max_pool, GCNConv, TopKPooling, BatchNorm

class ResidualGCN(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False):
        super().__init__()
        self.features = features
        self.gcn1 = GCNConv(features, features, bias=bias)
        self.gcn2 = GCNConv(features, features, bias=bias)
        self.bn1 = BatchNorm(features)
        self.bn2 = BatchNorm(features)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        gcn_1 = self.relu(self.bn1(self.gcn1(x=self.weight_1*x, edge_index=edge_index)))
        gcn_2 = self.relu(self.bn2(self.gcn2(x=gcn_1, edge_index=edge_index)))
        return self.weight_2*(x+gcn_2)


class GraphConvEncoder(nn.Module):
    def __init__(self, in_channels, res_layers, out_channels, graph_nodes, pooling="MaxPooling"):
        super().__init__()
        
        hidden_features = out_channels
        hidden_layers = res_layers
        self.net = []
        self.net.append(GCNConv(in_channels, hidden_features, bias=False))
        self.pooling_method = pooling
        for i in range(hidden_layers):
            self.net.append(ResidualGCN(hidden_features, bias=False, ave_first=False, ave_second=True))
        if self.pooling_method == "MaxPooling":
            self.net.append(GCNConv(hidden_features, hidden_features, bias=False))
        elif self.pooling_method == "TopKPooling":
            self.net.append(GCNConv(hidden_features, 1, bias=False))
        else:
            raise ValueError("Latent Fuse Method Error")

        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        
        self.pool = TopKPooling(1, ratio=hidden_features/graph_nodes)
        self.out_channels = out_channels

        self.net = nn.Sequential(*self.net)

    def forward(self, x: torch.Tensor, edge_idx: torch.Tensor) -> torch.Tensor:
        # forward through network
        x = self.relu(self.net[0](x=x, edge_index=edge_idx))
        for i in range(1, len(self.net)):
            x = self.net[i](x=x, edge_index=edge_idx)
        # max pooling
        if self.pooling_method == "MaxPooling":
            latent = global_max_pool(x, batch=None)
        # TopK
        elif self.pooling_method == "TopKPooling":
            predict, edge_idx, _, batch, _, _ = self.pool(x, edge_idx)
            predict = torch.flatten(predict)
            latent = predict[None, :]
        assert latent.shape[1] == self.out_channels
        return latent

class GCN_SR(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, out_features, g_channel, g_layers, g_features, graph_nodes, pos_freq_const=10, pos_freq_num=30, pooling="MaxPooling"):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = GraphConvEncoder(g_channel, g_layers, g_features, graph_nodes, pooling=pooling)
        self.pos_features = in_features*pos_freq_num*2
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + g_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, g_node, g_edge):
        assert coords.shape[0] == g_node.shape[0]
        assert g_node.shape[0] == g_edge.shape[0]
        batch_size = coords.shape[0]
        for b in range(batch_size):
            g_latent = self.lowres_encoding(g_node[b], g_edge[b])
            pos_latent = self.pos_encoding(coords[b])
            g_latent = g_latent[0].repeat(pos_latent.shape[0], 1)
            latent = torch.cat([pos_latent, g_latent], dim=-1)
            latent = latent[None, ...]
            try:
                latents = torch.vstack((latents, latent))
            except:
                latents = latent
        output = self.net(latents)
        return output
    

class GCN_SR_3D(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, out_features, g_channel, g_layers, g_features, graph_nodes, pos_freq_const=10, pos_freq_num=30, pooling="MaxPooling"):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = GraphConvEncoder(g_channel, g_layers, g_features, graph_nodes, pooling=pooling)
        self.pos_features = in_features*pos_freq_num*3
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + g_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, g_node, g_edge):
        assert coords.shape[0] == g_node.shape[0]
        assert g_node.shape[0] == g_edge.shape[0]
        batch_size = coords.shape[0]
        for b in range(batch_size):
            g_latent = self.lowres_encoding(g_node[b], g_edge[b])
            pos_latent = self.pos_encoding(coords[b])
            g_latent = g_latent[0].repeat(pos_latent.shape[0], 1)
            latent = torch.cat([pos_latent, g_latent], dim=-1)
            latent = latent[None, ...]
            try:
                latents = torch.vstack((latents, latent))
            except:
                latents = latent
        output = self.net(latents)
        return output

if __name__ == '__main__':
    print("test")