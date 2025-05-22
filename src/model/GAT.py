from utils import *
from torch_geometric.nn.models import GAT
from torch_geometric.nn import global_max_pool, TopKPooling

class GATEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, graph_nodes, pooling="MaxPooling"):
        super().__init__()
        self.pooling_method = pooling
        if self.pooling_method == "MaxPooling":
            self.gat = GAT(in_channels, hidden_channels, num_layers)
        elif self.pooling_method == "TopKPooling":
            self.gat = GAT(in_channels, hidden_channels, num_layers, 1)
        else:
            raise ValueError("Latent Fuse Method Error")
        self.pool = TopKPooling(1, ratio=hidden_channels/graph_nodes)
    
    def forward(self, edge_idx, node_features):
        predict = self.gat(node_features, edge_idx)
        if self.pooling_method == "MaxPooling":
            latent = global_max_pool(predict, batch=None)
        elif self.pooling_method == "TopKPooling":
            predict, edge_idx, _, batch, _, _ = self.pool(predict, edge_idx)
            predict = torch.flatten(predict)
            latent = predict[None, :]
        return latent

class GAT_SR(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, node_dim, latent_features, msg_passing_steps, graph_nodes, pos_freq_const=10, pos_freq_num=30, pooling="MaxPooling"):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = GATEncoder(in_channels=node_dim, hidden_channels=latent_features, num_layers=msg_passing_steps, graph_nodes=graph_nodes, pooling=pooling)
        self.pos_features = in_features*pos_freq_num*2
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + latent_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, 1, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, edge_idx, g_node):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        assert coords.shape[0] == edge_idx.shape[0]
        assert edge_idx.shape[0] == g_node.shape[0]
        batch_size = coords.shape[0]
        for b in range(batch_size):
            g_latent = self.lowres_encoding(edge_idx[b], g_node[b])
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
    

class GAT_SR_3D(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, node_dim, latent_features, msg_passing_steps, graph_nodes, pos_freq_const=10, pos_freq_num=30, pooling="MaxPooling"):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = GATEncoder(in_channels=node_dim, hidden_channels=latent_features, num_layers=msg_passing_steps, graph_nodes=graph_nodes, pooling=pooling)
        self.pos_features = in_features*pos_freq_num*3
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + latent_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, 1, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, edge_idx, g_node):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        assert coords.shape[0] == edge_idx.shape[0]
        assert edge_idx.shape[0] == g_node.shape[0]
        batch_size = coords.shape[0]
        for b in range(batch_size):
            g_latent = self.lowres_encoding(edge_idx[b], g_node[b])
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