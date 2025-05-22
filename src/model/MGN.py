from utils import *
from torch_geometric.nn import global_max_pool, TopKPooling
from torch_scatter import scatter_add


class GraphEncoder(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, num_layers, bias=True, lastact=False):
        super().__init__()
        self.node_mlp = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers, bias=bias, activate_final=lastact)
        self.edge_mlp = MLP(input_dim=input_dim_edge, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers, bias=bias, activate_final=lastact)
        
    def forward(self, node_features, edge_features):
        node_latents = self.node_mlp(node_features)
        edge_latents = self.edge_mlp(edge_features)
        
        return node_latents, edge_latents

class GraphNetBlock(nn.Module):
    def __init__(self, hidden_dim, num_layers, bias=True, lastact=False):
        super().__init__()
        # 2*hidden_dim: [nodes, accumulated_edges]
        self.mlp_node = MLP(input_dim=2*hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers, bias=bias, activate_final=lastact)
        # 3*hidden_dim: [sender, edge, receiver]
        self.mlp_edge = MLP(input_dim=3*hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers, bias=bias, activate_final=lastact)
    
    def _update_edges(self, edge_idx, node_features, edge_features):
        senders, receivers = edge_idx[0], edge_idx[1]
        sender_features = node_features[senders]
        receiver_features = node_features[receivers]
        features = torch.cat([sender_features, receiver_features, edge_features], dim=-1)
        return self.mlp_edge(features)
    
    def _update_nodes(self, edge_idx, node_features, edge_features):
        receivers = edge_idx[1]
        accumulate_edges = scatter_add(edge_features, receivers, dim=0)   # ~ tf.math.unsorted_segment_sum
        features = torch.cat([node_features, accumulate_edges], dim=-1)
        return self.mlp_node(features)
        
    def forward(self, edge_idx, node_features, edge_features):
        new_edge_features = self._update_edges(edge_idx, node_features, edge_features)
        new_node_features = self._update_nodes(edge_idx, node_features, new_edge_features)
        
        #add residual connections
        new_node_features += node_features
        new_edge_features += edge_features
        
        return new_node_features, new_edge_features

class GraphProcess(nn.Module):
    def __init__(self, hidden_dim, num_layers, message_passing_steps, bias=False, lastact=False):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(message_passing_steps):
            self.blocks.append(GraphNetBlock(hidden_dim, num_layers, bias, lastact))
            
    def forward(self, edge_idx, node_features, edge_features):
        for graphnetblock in self.blocks:
            node_features, edge_features = graphnetblock(edge_idx, node_features, edge_features)
            
        return node_features, edge_features

class GraphDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, bias=True, lastact=False):
        super().__init__()
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, bias=bias, activate_final=lastact)
    
    def forward(self, node_features):
        return self.mlp(node_features)

class MGN(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, num_enc_layers, num_proc_layers, num_dec_layers, message_passing_steps, \
                 enc_lastact=False, enc_bias=True, proc_lastact=False, proc_bias=True, dec_lastact=False, dec_bias=True, graph_nodes=2880, pooling="MaxPooling"):
        super().__init__()
        self.encoder = GraphEncoder(input_dim_node, input_dim_edge, hidden_dim, num_enc_layers)
        self.process = GraphProcess(hidden_dim, num_proc_layers, message_passing_steps)
        self.pooling_method = pooling
        if self.pooling_method == "MaxPooling":
            self.decoder = GraphDecoder(hidden_dim, hidden_dim, num_dec_layers)
        else:
            raise ValueError("Latent Fuse Method Error")
        self.pool = TopKPooling(1, ratio=hidden_dim/graph_nodes)

    def forward(self, edge_idx, node_features, edge_features):
        # Encode node/edge feature to latent space
        node_features, edge_features = self.encoder(node_features, edge_features)
        # Process message passing
        node_features, edge_features = self.process(edge_idx, node_features, edge_features)
        # Decode to output space
        predict = self.decoder(node_features)
        # max pooling
        if self.pooling_method == "MaxPooling":
            latent = global_max_pool(predict, batch=None)
        # TopK
        elif self.pooling_method == "TopKPooling":
            predict, edge_idx, _, batch, _, _ = self.pool(predict, edge_idx)
            predict = torch.flatten(predict)
            latent = predict[None, :]
        return latent

class MGN_SR(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, out_features, node_dim, edge_dim, g_features, latent_features, enc_layers, proc_layers, dec_layers, msg_passing_steps, pos_freq_const=10, pos_freq_num=30, enc_lastact=False, enc_bias=True, proc_lastact=False, proc_bias=True, dec_lastact=False, dec_bias=True, graph_nodes=2880, pooling="MaxPooling"):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = MGN(node_dim, edge_dim, g_features, latent_features, enc_layers, proc_layers, dec_layers, msg_passing_steps, \
                                    enc_lastact=enc_lastact, enc_bias=enc_bias, proc_lastact=proc_lastact, proc_bias=proc_bias, dec_lastact=dec_lastact, dec_bias=dec_bias,
                                    graph_nodes=graph_nodes, pooling=pooling)
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
    
    def forward(self, coords, edge_idx, g_node, g_edge):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        assert coords.shape[0] == edge_idx.shape[0]
        assert edge_idx.shape[0] == g_node.shape[0]
        assert g_node.shape[0] == g_edge.shape[0]
        batch_size = coords.shape[0]
        pos_latent = self.pos_encoding(coords)
        for b in range(batch_size):
            g_latent = self.lowres_encoding(edge_idx[b], g_node[b], g_edge[b])
            g_latent = g_latent[0].repeat(pos_latent.shape[1], 1)
            g_latent = g_latent[None, ...]
            try:
                g_latents = torch.vstack((g_latents, g_latent))
            except:
                g_latents = g_latent
        latent = torch.cat([pos_latent, g_latents], dim=-1)
        output = self.net(latent)
        return output

class MGN_SR_3D(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, out_features, node_dim, edge_dim, g_features, latent_features, enc_layers, proc_layers, dec_layers, msg_passing_steps, pos_freq_const=10, pos_freq_num=30, enc_lastact=False, enc_bias=True, proc_lastact=False, proc_bias=True, dec_lastact=False, dec_bias=True, graph_nodes=2880, pooling="MaxPooling"):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = MGN(node_dim, edge_dim, g_features, latent_features, enc_layers, proc_layers, dec_layers, msg_passing_steps, \
                                    enc_lastact=enc_lastact, enc_bias=enc_bias, proc_lastact=proc_lastact, proc_bias=proc_bias, dec_lastact=dec_lastact, dec_bias=dec_bias,
                                    graph_nodes=graph_nodes, pooling=pooling)
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
    
    def forward(self, coords, edge_idx, g_node, g_edge):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        assert coords.shape[0] == edge_idx.shape[0]
        assert edge_idx.shape[0] == g_node.shape[0]
        assert g_node.shape[0] == g_edge.shape[0]
        batch_size = coords.shape[0]
        pos_latent = self.pos_encoding(coords)
        for b in range(batch_size):
            g_latent = self.lowres_encoding(edge_idx[b], g_node[b], g_edge[b])
            g_latent = g_latent[0].repeat(pos_latent.shape[1], 1)
            g_latent = g_latent[None, ...]
            try:
                g_latents = torch.vstack((g_latents, g_latent))
            except:
                g_latents = g_latent
        latent = torch.cat([pos_latent, g_latents], dim=-1)
        output = self.net(latent)
        return output
    
    def get_latents(self, edge_idx, g_node, g_edge):
        batch_size = edge_idx.shape[0]
        for b in range(batch_size):
            g_latent = self.lowres_encoding(edge_idx[b], g_node[b], g_edge[b])
            try:
                g_latents = torch.vstack((g_latents, g_latent))
            except:
                g_latents = g_latent
        return g_latents

if __name__ == '__main__':
    print('test')