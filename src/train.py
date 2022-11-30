import logging
import pickle
from typing import Tuple

import dgl
import torch
import torch.nn.functional as F
from dgl.data.utils import save_graphs

from .model import Model

logger = logging.getLogger(__name__)


def initialise_training(
    graph,
    reverse_eids,
    device: str,
    n_hidden: int,
    learning_rate: float,
    graph_sampling_size: Tuple[int],
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
    model = Model(in_feats=n_in_feats, n_hidden=n_hidden).to(device)

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


def run_one_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    dataloader: dgl.dataloading.DataLoader,
    n_optimizer_steps=None,
    verbose=100,
):
    """Run one epoch

    Arguments:
        model {[type]} -- Initialised model
        optimizer {torch.optim.Optimizer} -- optimiser
        dataloader {dgl.dataloading.DataLoader} -- dataloader for training data

    Keyword Arguments:
        n_optimizer_steps {[type]} -- number of steps to learn from batch of data
        (default: {None}, iterate through all batches of training dataset.)
        verbose {int} -- report loss per verbose steps (default: {100})

    Returns:
        model, optimiser, loss value
    """
    for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(dataloader):
        if n_optimizer_steps and (it + 1) == n_optimizer_steps:
            print(f"Stop training this epoch after {n_optimizer_steps} steps.")
            break

        x = blocks[0].srcdata["feat"]
        pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)

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
    g, reverse_eids, nepoch: int = 10, verbose=2000, n_optimizer_steps: int = None
):
    model, dataloader, optimizer = initialise_training(
        g,
        reverse_eids,
        "cuda",
        n_hidden=128,
        learning_rate=1e-5,
        graph_sampling_size=[15, 10],
        negative_sample_size=5,
        weight_decay=1e-5,
        data_batch_size=512,
    )

    train_losses = []
    for epoch in range(nepoch):
        logger.info(f"epoch - {epoch}")
        model.train()
        model, optimizer, loss = run_one_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            verbose=verbose,
            n_optimizer_steps=n_optimizer_steps,
        )
        train_losses.append(loss)
        logger.info(f"loss - {loss}")

    save_graphs("data/graph.bin", [g])
    with open("data/model.pkl", "wb") as stream:
        pickle.dump(model, stream)

    with torch.no_grad():
        node_emb = model.inference(g, device="cuda").numpy()

    with open("data/embeddings.pkl", "wb") as f:
        pickle.dump(node_emb, f)

    return train_losses
