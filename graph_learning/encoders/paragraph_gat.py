import torch.nn as nn # type: ignore
from torch_geometric.nn import GATv2Conv # type: ignore

class ParagraphGAT(nn.Module):
    def __init__(self, hidden_dim, heads=8, final_heads=4):
        super().__init__()
        
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim, heads=final_heads, concat=False)
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.unusable = 0

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        
        if edge_index is None or edge_index.numel() == 0 or edge_index.dim() != 2 or edge_index.size(1) == 0:
            print("issue encountered: edge_index is empty or malformed")
            data.x = x
            self.unusable += 1
            return data
        
        x_residual = x 
        x = self.conv1(x, edge_index)
        x = x + x_residual 
        x = self.activation(x)
        x = self.dropout(x)
        
        
        x_residual = x
        x = self.conv2(x, edge_index)
        x = x + x_residual
        x = self.activation(x)
        x = self.dropout(x)
        
        
        x_residual = x
        x = self.conv3(x, edge_index)
        x = x + x_residual
        
        data.x = x
        return data
