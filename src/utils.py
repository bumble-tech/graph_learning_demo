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
import logging
from pathlib import Path
from typing import Union

import dgl
import pandas as pd
import torch
from numpy import random
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def preprocess_features(nodes: pd.DataFrame) -> pd.DataFrame:
    """Treat node freatures
    treat date features, remove index, encode language.

    Arguments:
        nodes {pd.DataFrame} -- input dataframe with raw features

    Returns:
        [d.DataFrame] -- transformed node features
    """
    created_at = pd.to_datetime(nodes.pop("created_at"))
    nodes["created_year"] = created_at.dt.year
    nodes["created_month"] = created_at.dt.month
    updated_at = pd.to_datetime(nodes.pop("updated_at"))
    nodes["update_year"] = updated_at.dt.year
    nodes["update_month"] = updated_at.dt.month

    nodes = nodes.sort_values(by="numeric_id")
    nodes = nodes.drop("numeric_id", axis=1)

    language = nodes.pop("language")
    encoder = OneHotEncoder().fit(language.unique().reshape(-1, 1))
    nodes[
        [f"language_{x}" for x in encoder.get_feature_names_out()]
    ] = encoder.transform(language.values.reshape(-1, 1)).toarray()

    logger.info(f"nodes shape, {nodes.shape}")
    return nodes


def _return_reverse_eids(g) -> torch.Tensor:
    """Create reverse edge ids for bi-directed, ie. undirected graph.
    The reverse ids are used during sampling

    Arguments:
        g {[type]} -- graph

    Returns:
        [torch.Tensor] -- reverse. ids
    """
    nsrc = g.num_edges()
    reverse_eids = torch.cat([torch.arange(nsrc, 2 * nsrc), torch.arange(0, nsrc)])
    return reverse_eids


def create_graph(edges: pd.DataFrame, nodes: pd.DataFrame):
    """Create a bi-directed graph from edges and node features

    Arguments:
        edges {pd.DataFrame} -- edges
        nodes {pd.DataFrame} -- node features

    Returns:
        -- graph
        -- reverse edge ids, used for sampling during training
    """
    logger.info("Creating network graph...")
    u = edges["numeric_id_1"].values
    v = edges["numeric_id_2"].values

    g = dgl.graph((u, v)).to("cpu")
    logger.info(f"number of nodes - {g.num_nodes()}")
    logger.info(f"number of edges - {g.num_edges()}")
    reverse_eids = _return_reverse_eids(g)
    # making graph bi-directional, ie. undirected
    logger.info("making the graph bi-directional...")
    g = dgl.to_bidirected(g)
    logger.info(f"number of edges - {g.num_edges()}")

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


def create_evaluation_dataset(test_edges, unique_users):
    """Generate evaluation adtaset
    a sequence includes 5 positive samples + 5 negative samples

    Arguments:
        test_edges {pd.DataFrame} -- Test edges
        unique_users {list} -- all unique users, used for negative sampling

    Returns:
        pd.DataFrame
    """
    groupby = test_edges.groupby("numeric_id_1")
    unique_users = set(unique_users)

    def _sample_users(sub_df, unique_users):
        positives = sub_df.loc[:, "numeric_id_2"].values
        negatives = list(unique_users.difference(positives))
        users = list(random.choice(positives, 5)) + list(random.choice(negatives, 5))

        return users

    from pandarallel import pandarallel

    pandarallel.initialize()
    sampled_users = groupby.parallel_apply(
        lambda sub_df: _sample_users(sub_df, unique_users)
    )
    return sampled_users


def train_test_split(train_frac: float = 0.9):
    """Split edgest into train/test.

    Save all data to disk.

    Keyword Arguments:
        train_frac {float} -- proportion of train size (default: {0.9})
    """
    features = pd.read_csv("data/large_twitch_features.csv")
    features = preprocess_features(features)
    edges = pd.read_csv("data/large_twitch_edges.csv")
    train_index = edges.index.to_series().sample(frac=train_frac)
    test_mask = ~edges.index.isin(train_index)
    train_edges = edges.loc[train_index]
    test_edges = edges.loc[test_mask]
    logger.info(
        f"train_edges.shape:{train_edges.shape}, test_edges.shape: {test_edges.shape}"
    )
    features.to_csv("data/processed_features.csv")
    train_edges.to_csv("data/train_edges.csv")
    test_edges.to_csv("data/test_edges.csv")
