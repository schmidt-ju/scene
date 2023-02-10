import math

import torch
from dgl.nn.pytorch.utils import Identity
from dgl.nn import GATConv

from nn.edgegatconv import EdgeGATConv


class ResidualEdgeGATConv(torch.nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            num_heads="auto",
            edge_feats=None,
            bias=True,
            residual=True,
            residual_type="add",
            force_residual_trafo=True,
            **kwargs):
        """Wrapper class for the GAT graph convolution.
            If multiple heads are used, the results are concatenated to one final embedding.

        Args:
            in_feats (Union[int, (int, int)]): Input feature size.
            out_feats (int): Output feature size.
            num_heads (Union[str, int], optional): Number of attention heads. Defaults to "auto",
                which means that there will be one head per 32 features.
            edge_feats (int, optional): Number of edge features. Defaults to None,
                which means that there are no edge features.
            bias (bool, optional): Use bias. Defaults to True.
            residual (bool, optional): Use residual. Defaults to True.
            residual_type (str, optional): Residual type, "add" and "concat" are possible. Defaults to "add".
            force_residual_trafo (bool, optional): Force transformation for residual connection
                independent of matching input and output sizes. Defaults to True.
        """
        super(ResidualEdgeGATConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.edge_feats = edge_feats
        self.residual = residual
        self.residual_type = residual_type
        self.force_residual_trafo = force_residual_trafo

        # Automatically determine number of attention heads
        if num_heads == "auto":
            if isinstance(in_feats, tuple):
                in_feat_size = in_feats[1]
            else:
                in_feat_size = in_feats
            num_heads = int(math.ceil(in_feat_size / 32))
            self.num_heads = num_heads

        # Use default GATConv if there are no edge features and EdgeGATConv if there are edge features
        # Residual and bias are set to false, because they are handled in this wrapper class separately
        if self.edge_feats is None:
            self.conv = GATConv(in_feats, int(out_feats / num_heads), num_heads,
                                allow_zero_in_degree=True, residual=False, bias=False)
        else:
            self.conv = EdgeGATConv(in_feats, int(out_feats / num_heads), num_heads, edge_feats,
                                    allow_zero_in_degree=True, residual=False, bias=False)

        if residual and residual_type == "concat":
            # Take destination feature size as the input size for the residual connection
            if isinstance(in_feats, tuple):
                in_feats = in_feats[1]

            if force_residual_trafo:
                self.res = torch.nn.Linear(
                    in_feats + out_feats, out_feats, bias=False)
            else:
                if in_feats != out_feats:
                    self.res = torch.nn.Linear(
                        in_feats + out_feats, out_feats, bias=False)
                else:
                    self.res = Identity()

        if residual and residual_type == "add":
            # Take destination feature size as the input size for the residual connection
            if isinstance(in_feats, tuple):
                in_feats = in_feats[1]

            if force_residual_trafo:
                self.res = torch.nn.Linear(in_feats, out_feats, bias=False)
            else:
                if in_feats != out_feats:
                    self.res = torch.nn.Linear(in_feats, out_feats, bias=False)
                else:
                    self.res = Identity()

        if bias:
            self.bias = torch.nn.parameter.Parameter(torch.zeros(out_feats))
        else:
            self.register_buffer("bias", None)

    def forward(self, graph, feat, edge_feat=None, get_attention=False):
        """Forward pass of the GAT graph convolution.

        Args:
            graph (dgl.heterograph.DGLHeteroGraph): Input graph.
            feat ((torch.Tensor, torch.Tensor)): Tuple containing source
                and destination node features.
            edge_feat (torch.Tensor, optional): Edge features. Defaults to None.
            get_attention (bool, optional): Return attention weights. Defaults to False.

        Returns:
            torch.Tensor: Output features of shape :math:`(N, out_feats)`
                where :math:`N` corresponds to the number of destination nodes.
            torch.Tensor, optional: Attention weights of shape :math:`(E, 1)`
                where :math:`E` corresponds to the number of edges.
        """        
        if get_attention and edge_feat is None:
            x, a = self.conv(graph, feat, get_attention=True)
        elif not get_attention and edge_feat is None:
            x = self.conv(graph, feat, get_attention=False)
        elif get_attention and edge_feat is not None:
            x, a = self.conv(graph, feat, edge_feat, get_attention=True)
        elif not get_attention and edge_feat is not None:
            x = self.conv(graph, feat, edge_feat, get_attention=False)

        # Concat heads, e.g., [d, 4, 16] -> [d, 64]
        x = torch.flatten(x, start_dim=1)

        if self.residual and self.residual_type == "concat":
            # Take destination features as the input for the residual connection
            if isinstance(feat, tuple):
                feat = feat[1]

            x = self.res(torch.cat((x, feat), dim=-1))

        if self.residual and self.residual_type == "add":
            # Take destination features as the input for the residual connection
            if isinstance(feat, tuple):
                feat = feat[1]

            x = x + self.res(feat)

        # Add bias
        if self.bias is not None:
            x = x + self.bias

        if get_attention:
            a = torch.flatten(a, start_dim=1)
            return x, a
        else:
            return x
