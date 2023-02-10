import argparse
import yaml
import os
import pandas as pd

import torch
import dgl
import pytorch_lightning as pl
from dgl.dataloading import GraphDataLoader as DataLoader

from model import SCENE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["aifb", "mutag", "bgs", "am"], default="aifb")
parser.add_argument("--gpu", type=bool, default=True)
parser.add_argument("--loader_workers", type=int, default=0)
parser.add_argument("--seeds", type=list, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

args = parser.parse_args()


def train(config):
    """Traing of the model from `SCENE <https://arxiv.org/pdf/2301.03512.pdf>`
        on node classification in knowledge graphs.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        float: Final accuracy on the test nodes.
    """
    #####################
    # Config
    #####################

    if args.dataset == "aifb":
        dataset = dgl.data.rdf.AIFBDataset(
            insert_reverse=True, force_reload=False)
    elif args.dataset == "mutag":
        dataset = dgl.data.rdf.MUTAGDataset(
            insert_reverse=True, force_reload=False)
    elif args.dataset == "bgs":
        dataset = dgl.data.rdf.BGSDataset(
            insert_reverse=True, force_reload=False)
    elif args.dataset == "am":
        dataset = dgl.data.rdf.AMDataset(
            insert_reverse=True, force_reload=False)
    else:
        NotImplementedError(
            f"{args.dataset} is currently not implemented as a dataset")

    epochs = config["epochs"]
    dropout = config["dropout"]
    weight_decay = float(config["weight_decay"])
    learning_rate = float(config["learning_rate"])
    degree_cutting = config["degree_cutting"]
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]

    graph = dataset[0]

    category = dataset.predict_category
    out_sizes = {}
    out_sizes[category] = dataset.num_classes

    #####################
    # Graph Preprocessing
    #####################

    # Add artificial node features, in order to make the conversion to a homogeneous
    # graph work
    for ntype in graph.ntypes:
        if ntype != category:
            graph.nodes[ntype].data["labels"] = torch.zeros(
                graph.num_nodes(ntype), dtype=torch.int64)
            graph.nodes[ntype].data["test_mask"] = torch.zeros(
                graph.num_nodes(ntype), dtype=torch.uint8)
            graph.nodes[ntype].data["train_mask"] = torch.zeros(
                graph.num_nodes(ntype), dtype=torch.uint8)

    # Convert to homogeneous graph and perform degree cutting
    hom_graph = dgl.to_homogeneous(
        graph, ndata=["labels", "test_mask", "train_mask"])
    degrees = hom_graph.in_degrees() + hom_graph.out_degrees()
    degree_mask = degrees <= degree_cutting
    category_index = graph.ntypes.index(category)
    category_mask = hom_graph.ndata["_TYPE"] != category_index

    # All nodes below or equal the defined degree that do not belong to the category type are cut
    combined_masks = torch.logical_and(degree_mask, category_mask)
    hom_graph.remove_nodes(combined_masks.nonzero(as_tuple=True)[0])

    # Go back to heterogeneous graph
    graph = dgl.to_heterogeneous(hom_graph, graph.ntypes, graph.etypes)

    # Get number of final nodes for each node type
    in_nodes = dict()
    for ntype in graph.ntypes:
        in_nodes[ntype] = graph.num_nodes(ntype)

    dataset = SceneGraphDataset(graph)

    #####################
    # Data Setup
    #####################

    # Batch size does not matter at all here, because we perform full batch training and inference
    # Train data loader
    train_loader = DataLoader(dataset, batch_size=4,
                              shuffle=True, num_workers=args.loader_workers)
    # Validation data loader
    val_loader = DataLoader(dataset, batch_size=4,
                            num_workers=args.loader_workers)

    #####################
    # Model
    #####################

    model = SCENE(in_nodes, hidden_size, out_sizes, num_heads, graph.canonical_etypes,
                  learning_rate=learning_rate, dropout=dropout, weight_decay=weight_decay)

    #####################
    # Training
    #####################

    # Most basic trainer, we disable checkpointing and logging
    trainer = pl.Trainer(
        max_epochs=epochs,
        checkpoint_callback=False,
        logger=False,
        gpus=int(args.gpu and torch.cuda.is_available()),
        deterministic=True,
        weights_save_path=None,
        auto_scale_batch_size=False,
    )
    trainer.fit(model, train_loader, val_loader)

    return trainer.progress_bar_metrics["acc_test"]


class SceneGraphDataset(dgl.data.DGLDataset):
    def __init__(self, in_graph):
        """Wrapper class for dataloading.
            This dataset contains only one graph,
            because training is done full-batch.

        Args:
            in_graph (dgl.heterograph.DGLHeteroGraph): Input graph.
        """        
        super().__init__(name='scene graph')
        self.graph = in_graph

    def process(self):
        """Process the dataset. In this case do nothing.
        """        
        pass

    def __getitem__(self, i):
        """Get the single graph of the dataset.

        Args:
            i (int): Index of the requested graph.
                In this case ignored.

        Returns:
            dgl.heterograph.DGLHeteroGraph: Requested graph.
        """        
        return self.graph

    def __len__(self):
        """Get the dataset length.

        Returns:
            int: Always return "1", because
                the dataset only contains one graph.
        """        
        return 1


if __name__ == "__main__":
    # Load dataset config
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{args.dataset}.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Run multiple runs with different seeds
    metrics = []
    for seed in args.seeds:
        pl.seed_everything(seed)

        metric = train(config)
        metrics.append(metric)

    # Print and save results
    df = pd.DataFrame({"seed": args.seeds, "acc": metrics})
    print(df.describe())
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              "results", f"graph_benchmark_results_{args.dataset}.csv"))
