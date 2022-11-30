import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import tqdm


class Model(nn.Module):
    def __init__(self, in_feats, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.conv1 = dglnn.SAGEConv(in_feats, self.n_hidden, "mean")
        self.conv2 = dglnn.SAGEConv(self.n_hidden, self.n_hidden, "mean")
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden * 2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def predict(self, h_src, h_dst):
        return self.predictor(torch.cat([h_src, h_dst], dim=1))

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
        y = torch.zeros(g.num_nodes(), self.n_hidden, device="cpu", pin_memory=True)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            x = feat[input_nodes]
            h = self.conv1(blocks[0], x)
            h = nn.ReLU()(h)
            y[output_nodes] = h.to("cpu")

        # second-layer
        feat = y.to(device)
        print(feat.shape)
        y = torch.zeros(g.num_nodes(), self.n_hidden, device="cpu", pin_memory=True)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            x = feat[input_nodes]
            h = self.conv2(blocks[0], x)
            y[output_nodes] = h.to("cpu")

        return y
