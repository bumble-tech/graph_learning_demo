# Copyright 2022 Bumble Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

    def predict(self, src_embedding, dst_embedding):
        return self.predictor(torch.cat([src_embedding, dst_embedding], dim=1))

    def forward(self, pair_graph, negative_pair_graph, blocks, input_feat):
        h = self.conv1(blocks[0], input_feat)
        h = nn.ReLU()(h)
        h = self.conv2(blocks[1], h)

        positive_src, positive_dst = pair_graph.edges()
        negative_src, negative_dst = negative_pair_graph.edges()
        h_pos = self.predict(h[positive_src], h[positive_dst])
        h_neg = self.predict(h[negative_src], h[negative_dst])
        return h_pos, h_neg

    def inference(self, graph, device, batch_size=1280):
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["feat"]
        )
        dataloader = dgl.dataloading.NodeDataLoader(
            graph,
            torch.arange(graph.num_nodes()).to(graph.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # first-layer
        feat = graph.ndata["feat"].to(device)
        y = torch.zeros(graph.num_nodes(), self.n_hidden, device="cpu", pin_memory=True)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            input_feat = feat[input_nodes]
            h = self.conv1(blocks[0], input_feat)
            h = nn.ReLU()(h)
            y[output_nodes] = h.to("cpu")

        # second-layer
        feat = y.to(device)
        y = torch.zeros(graph.num_nodes(), self.n_hidden, device="cpu", pin_memory=True)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            input_feat = feat[input_nodes]
            h = self.conv2(blocks[0], input_feat)
            y[output_nodes] = h.to("cpu")

        return y
