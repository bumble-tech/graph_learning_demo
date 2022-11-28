from typing import Dict, Tuple, Union
import dgl
import dgl.nn as dglnn
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


def preprocess_features(nodes:pd.DataFrame):
    from datetime import datetime
    created_at = pd.to_datetime(nodes.pop('created_at'))
    nodes['created_year'] = created_at.dt.year
    nodes['created_month'] = created_at.dt.month
    updated_at = pd.to_datetime(nodes.pop('updated_at'))
    nodes['update_year'] = updated_at.dt.year
    nodes['update_month'] = updated_at.dt.month

    nodes = nodes.sort_values(by='numeric_id')
    nodes = nodes.drop('numeric_id', axis=1)

    from sklearn.preprocessing import OneHotEncoder
    language = nodes.pop('language')
    encoder = OneHotEncoder().fit(language.unique().reshape(-1, 1))
    nodes[[f'language_{x}' for x in encoder.get_feature_names_out()]] = encoder.transform(language.values.reshape(-1, 1)).toarray()

    print(f'nodes shape, {nodes.shape}')
    return nodes


def _return_reverse_eids(g):
    nsrc = g.num_edges()
    reverse_eids = torch.cat([torch.arange(nsrc, 2 * nsrc), torch.arange(0, nsrc)])
    return reverse_eids


def create_graph(edges:pd.DataFrame, nodes:pd.DataFrame):
    print("Creating network graph...")
    u = edges['numeric_id_1'].values
    v = edges['numeric_id_2'].values

    g = dgl.graph((u, v)).to('cpu')
    print('number of nodes', g.num_nodes())
    print('number of edges', g.num_edges())
    # TODO: how to use this
    # train_edges, val_edges, test_edges = split_dataset(g.edges(), frac_list=[0.8, 0.1, 0.1], shuffle=True, relabel_nodes=False)
    # train_subgraph = dgl.edge_subgraph(g, train_edges)
    # val_subgraph = dgl.edge_subgraph(g, val_edges)
    # test_subgraph = dgl.edge_subgraph(g, test_edges)
    reverse_eids = _return_reverse_eids(g)
    # making graph bi-directional, ie. undirected
    print("making the graph bi-directional...")
    g = dgl.to_bidirected(g)
    print('number of edges', g.num_edges())

    g.ndata["feat"] = torch.from_numpy(nodes.values).float()

    return g, reverse_eids

def load_graph(filepath: Union[str, Path]) -> dgl.DGLGraph:
    """A wrapper to load graphs.

    Arguments:
        filepath {Union[str, Path]} -- filepath of saved graph

    Returns:
        dgl.DGLGraph
    """
    from dgl.data.utils import load_graphs
    glist, _ = load_graphs(str(filepath))
    graph = glist[0]

    return graph


import torch.nn as nn
import dgl.nn as dglnn

class Model(nn.Module):
    def __init__(self, in_feats, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.conv1 = dglnn.SAGEConv(in_feats, self.n_hidden, 'mean')
        self.conv2 = dglnn.SAGEConv(self.n_hidden, self.n_hidden, 'mean')
        self.conv3 = dglnn.SAGEConv(self.n_hidden, self.n_hidden, 'mean')
        self.predictor = nn.Sequential(
                nn.Linear(n_hidden*2, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, 1),
                # nn.Softmax(dim=1)
            )
        
    def predict(self, h_src, h_dst):
        return self.predictor(torch.cat([h_src, h_dst], dim=1)) # binary output

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = self.conv1(blocks[0], x)
        h = nn.ReLU()(h)
        h = self.conv2(blocks[1], h)
        
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])

        return h_pos, h_neg


    def inference(self, g, device, batch_size=1280):
        # Use all neighbours in the first layer in inference,
        # while sample from 3 layers during training
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"]
        )
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # first-layer
        feat = g.ndata["feat"].to(device)
        print(feat.shape)
        y = torch.zeros(
            g.num_nodes(), self.n_hidden, device='cpu', pin_memory=True
        )
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            x = feat[input_nodes]
            h = self.conv1(blocks[0], x)
            h = nn.ReLU()(h)
            y[output_nodes] = h.to("cpu")
        
        # second-layer
        feat = y.to(device)
        print(feat.shape)
        y = torch.zeros(
            g.num_nodes(), self.n_hidden, device='cpu', pin_memory=True
        )
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            x = feat[input_nodes]
            h = self.conv2(blocks[0], x)
            y[output_nodes] = h.to("cpu")
            
        # third-layer
        feat = y.to(device)
        print(feat.shape)
        y = torch.zeros(
            g.num_nodes(), self.n_hidden, device='cpu', pin_memory=True
        )
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            x = feat[input_nodes]
            h = self.conv3(blocks[0], x)
            y[output_nodes] = h.to("cpu")

        return y



def _initialise_training(
    graph,
    reverse_eids,
    device: str,
    n_hidden: int,
    learning_rate: float,
    graph_sampling_size:Tuple[int],
    negative_sample_size: int,
    weight_decay: float,
    data_batch_size: int,
):

    # Move data to GPU
    graph = graph.to(device)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(graph.num_edges()).to(device)

    # Initialise model
    n_in_feats = graph.ndata["feat"].shape[1]
    model = Model(
        in_feats=n_in_feats, n_hidden=n_hidden
    ).to(device)

    # Initialise optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Initiate sampler for traversing the graph
    sampler = dgl.dataloading.NeighborSampler(
        list(graph_sampling_size), prefetch_node_feats=["feat"]
    )
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(negative_sample_size),
    )

    # Initialise data loader
    dataloader = dgl.dataloading.DataLoader(
        graph,
        seed_edges,
        sampler,
        device=device,
        batch_size=data_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    return model, dataloader, optimizer


def _run_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    dataloader: dgl.dataloading.DataLoader,
    n_optimizer_steps = None,
    verbose = 100,
):
    for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
        if n_optimizer_steps and (it + 1) == n_optimizer_steps:
            print(f"Stop training this epoch after {n_optimizer_steps} steps.")
            
            # TODO: report ndcg, roc-auc
            
            break
            
        x = blocks[0].srcdata["feat"]
        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)

        # softmax output
        # pos_label = torch.zeros_like(pos_score)
        # pos_label[:, 1] = True
        # neg_label = torch.zeros_like(neg_score)
        # neg_label[:, 0] = True
        # pred = torch.cat([pos_score, neg_score])
        # labels = torch.cat([pos_label, neg_label])
        # loss = nn.BCELoss()(pred, labels)

        # linear output
        pos_label = torch.ones_like(pos_score)
        neg_label = torch.zeros_like(neg_score)
        pred = torch.cat([pos_score, neg_score])
        labels = torch.cat([pos_label, neg_label])
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = loss.item()

        if (it + 1) % verbose == 0:
            mem = torch.cuda.max_memory_allocated() / 1000000
            print(f"Loss {loss_value}, GPU Mem {mem}MB")

        

    return model, optimizer, loss_value


def train(
    graph: dgl.DGLGraph,
    reverse_eids,
    n_epoch: int,
    learning_rate:float,
    device = 'cuda',
    graph_sampling_size = (10, 5),
    negative_sample_size = 10,
    inference_batch_size: int = 2000,
    evaluate_batch_size: int = 500,
):
    model, dataloader, optimizer = _initialise_training(
        graph, reverse_eids, device, n_hidden=128, learning_rate=learning_rate, graph_sampling_size=graph_sampling_size,
        negative_sample_size=negative_sample_size, weight_decay=1e-5, data_batch_size=512
    )
    test_mrr_list = []
    train_loss_list = []

    for epoch in range(n_epoch):
        print('epoch', epoch)
        model.train()
        model, optimizer, loss = _run_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader
        )

        # TODO
        # model.eval()
        # test_metrics = evaluate(
        #     model=model,
        #     graph=graph,
        #     edge_split=eval_dict,
        #     device=device,
        #     num_workers=0,
        #     inference_batch_size=inference_batch_size,
        #     evaluate_batch_size=evaluate_batch_size,
        # )
        # print(f"Test MRR: {test_metrics['average MRR']}")

        # test_mrr_list.append(test_metrics["average MRR"])
        train_loss_list.append(loss)

    return (
        model,
        {"mrr": test_mrr_list, "train_loss": train_loss_list},
    )