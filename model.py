import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from nn.res_edgegatconv import ResidualEdgeGATConv


class SCENE(pl.LightningModule):
    def __init__(
        self,
        in_nodes,
        hidden_size,
        out_sizes,
        num_heads,
        canonical_etypes,
        learning_rate=0.001,
        weight_decay=0.0,
        dropout=0.0,
    ):
        """Graph convolution model from `SCENE <https://arxiv.org/pdf/2301.03512.pdf>`.
            The model cascades multiple layers of graph convolution to aggregate information
            into the nodes to be classified.

        Args:
            in_nodes (dict): Dictionary containing node types and number of nodes per node type
                of the knowledge graph to be trained on.
            hidden_size (int): Hidden size used during graph convolution.
            out_sizes (dict): Dictionary containing the node type to be classified and the number
                of possible classes of this node type.
            num_heads (int): Number of attention heads of the EdgeGAT operator.
            canonical_etypes (list[(str, str, str)]): List of the canonical edge types of the knowledge
                graph to be trained on.
            learning_rate (float, optional): Learning rate for Adam optimizer. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for Adam optimizer. Defaults to 0.0.
            dropout (float, optional): Dropout applied during decoding. Defaults to 0.0.
        """
        super().__init__()
        self.in_nodes = in_nodes
        self.hidden_size = hidden_size
        self.out_sizes = out_sizes
        self.num_heads = num_heads
        self.canonical_etypes = canonical_etypes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_p = dropout

        # Extract training objective
        self.category = list(out_sizes.keys())[0]
        self.out_size = out_sizes[self.category]

        # Create node embeddings
        self.node_embeddings = torch.nn.ParameterDict()
        for key in in_nodes:
            embed = torch.nn.Parameter(
                torch.Tensor(in_nodes[key], hidden_size))
            torch.nn.init.xavier_uniform_(
                embed, gain=torch.nn.init.calculate_gain('relu'))
            self.node_embeddings[key] = embed

        # Initialize graph convolutions (cascaded style)
        self.full_graph_conv = torch.nn.ModuleDict()
        self.full_graph_conv["conv_1"] = torch.nn.ModuleDict()
        self.full_graph_conv["conv_2"] = torch.nn.ModuleDict()
        self.full_graph_conv["conv_3"] = torch.nn.ModuleDict()
        self.full_graph_conv["conv_4"] = torch.nn.ModuleDict()
        for edge in canonical_etypes:
            which_graph_conv = None
            if (edge[0] == edge[2]) and (edge[0] != self.category):
                # Self-conv excluding the target node
                which_graph_conv = "conv_1"
            elif (edge[0] != edge[2]) and (edge[2] != self.category):
                # Conv from all nodes to others but the target node
                which_graph_conv = "conv_2"
            elif (edge[2] == self.category) and (edge[0] != self.category):
                # Conv to the target node
                which_graph_conv = "conv_3"
            elif (edge[2] == self.category) and (edge[0] == self.category):
                # Self update
                which_graph_conv = "conv_4"
            else:
                NotImplementedError(
                    f"Undefined graph convolution for edge {edge}")

            if which_graph_conv is not None:
                self.full_graph_conv[which_graph_conv][str(edge)] = ResidualEdgeGATConv(
                    in_feats=hidden_size, out_feats=hidden_size, num_heads=num_heads)

        # Initialize decoder
        self.decoder_1 = torch.nn.Linear(
            in_features=hidden_size*3, out_features=hidden_size)
        self.dropout_1 = torch.nn.Dropout(p=dropout)
        self.decoder_2 = torch.nn.Linear(
            in_features=hidden_size, out_features=self.out_size)
        self.dropout_2 = torch.nn.Dropout(p=dropout)

        self.save_hyperparameters()

    def forward(self, graph):
        """Forward pass of the model.

        Args:
            graph (dgl.heterograph.DGLHeteroGraph): Input graph.

        Returns:
            torch.Tensor: Predicted labels.
        """        
        # Node embedding update
        for ntype in graph.ntypes:
            graph.nodes[ntype].data["x"] = F.relu(self.node_embeddings[ntype])

        # Iterate over the cascaded layers
        for conv_key, conv_dict in self.full_graph_conv.items():
            # Collect the node types that are possible targets during this layer of graph convolution
            conv_dict_key_tuples = [
                tuple(map(str, string[2:-2].split("', '"))) for string in conv_dict.keys()]

            targets = [x[2] for x in conv_dict_key_tuples]
            targets = list(set(targets))

            embeddings = {x: 0.0 for x in targets}
            # Do graph convolution
            for curr_conv_key, curr_conv in conv_dict.items():
                # Get key of target node
                curr_tuple = tuple(map(str, curr_conv_key[2:-2].split("', '")))
                target_ntype = curr_tuple[2]

                # Extract subgraph
                curr_subgraph = graph.edge_type_subgraph([curr_tuple])
                src_feats = graph.nodes[curr_tuple[0]].data["x"]
                dst_feats = graph.nodes[curr_tuple[2]].data["x"]

                embeddings[target_ntype] += curr_conv(
                    curr_subgraph, (src_feats, dst_feats))

            # Update each node simultaneously
            for node_key, embedding in embeddings.items():
                graph.nodes[node_key].data["x"] = F.relu(embedding)

            # Residual values
            if conv_key == "conv_2":
                x_res1 = graph.nodes[self.category].data["x"]
            if conv_key == "conv_3":
                x_res2 = graph.nodes[self.category].data["x"]

        # Decoder
        x = graph.nodes[self.category].data["x"]
        x = torch.cat([x, x_res1, x_res2], dim=-1)
        x = self.dropout_1(x)
        x = self.decoder_1(x)
        x = F.relu(x)
        x = self.dropout_2(x)
        x = self.decoder_2(x)

        # No softmax, because nn.CrossEntropy already does that
        return x

    def configure_optimizers(self):
        """ Configuration of the optimizer.

        Returns:
            torch.optim.adam.Adam: Optimizer used during training.
        """        
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        """Training step of the model.

        Args:
            batch (dgl.heterograph.DGLHeteroGraph): Input graph.
            batch_idx (int): Index of current batch.

        Returns:
            torch.Tensor: Loss.
        """ 
        out = self(batch)
        y = batch.nodes[self.category].data["labels"].squeeze()

        mask = batch.nodes[self.category].data["train_mask"].to(torch.bool)

        out_labeled = out[mask]
        y_labeled = y[mask]

        loss = F.cross_entropy(out_labeled, y_labeled)

        # Logging to TensorBoard (disabled by the trainer)
        self.log("loss_train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step of the model.

        Args:
            batch (dgl.heterograph.DGLHeteroGraph): Input graph.
            batch_idx (int): Index of current batch.

        Returns:
            (torch.Tensor, torch.Tensor): Tuple containing predictions and labels.
        """        
        out = self(batch)
        y = batch.nodes[self.category].data["labels"].squeeze()
        mask = batch.nodes[self.category].data["test_mask"].to(torch.bool)

        out_labeled = out[mask]
        y_labeled = y[mask]

        loss = F.cross_entropy(out_labeled, y_labeled)
        # Logging to TensorBoard (disabled by the trainer)
        self.log("loss_test", loss, prog_bar=True)

        y_pred_softmax = torch.log_softmax(out_labeled, dim=-1)
        _, y_pred = torch.max(y_pred_softmax, dim=-1)
        return y_pred, y_labeled

    def validation_epoch_end(self, validation_step_outputs):
        """Evaluation at the end of validation.

        Args:
            validation_step_outputs ([(torch.Tensor, [torch.Tensor)]): List of tuples
                containing predictions and labels.
        """    
        pred = [out[0] for out in validation_step_outputs]
        y = [out[1] for out in validation_step_outputs]
        pred = torch.cat(pred)
        y = torch.cat(y)

        self.end_log_dict = dict()
        if len(pred) and len(y):  # Avoid throwing error in f1 if all are masked
            acc_test = torchmetrics.functional.accuracy(pred, y.to(int))
            self.end_log_dict["acc_test"] = acc_test
            f1_score_micro = torchmetrics.functional.f1(
                pred, y.to(int), average="micro")
            self.end_log_dict["f1_test_micro"] = f1_score_micro
            f1_score_macro = torchmetrics.functional.f1(
                pred, y.to(int), average="macro", num_classes=self.out_size)
            self.end_log_dict["f1_test_macro"] = f1_score_macro

        self.log_dict(self.end_log_dict, prog_bar=True)
