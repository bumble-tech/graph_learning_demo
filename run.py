import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ndcg_score

from src.train import train
from src.utils import create_evaluation_dataset, create_graph, train_test_split

logging.basicConfig(
    format="%(asctime)s: %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

DEVICE = "cuda"


def predict(model, node_emb, eval_dataset: pd.DataFrame):
    first_player_embeddings = np.vstack(
        [np.repeat(node_emb[x].reshape(1, -1), 10, axis=0) for x in eval_dataset.index]
    )
    second_player_embeddings = np.vstack([node_emb[x] for x in eval_dataset])

    predictions = model.predict(
        torch.from_numpy(first_player_embeddings).to(DEVICE),
        torch.from_numpy(second_player_embeddings).to(DEVICE),
    )
    predictions = predictions.view(-1, 10).detach().cpu().numpy()
    labels = np.hstack(
        [np.ones((eval_dataset.shape[0], 5)), np.zeros((eval_dataset.shape[0], 5))]
    )

    return predictions, labels


def _predict_and_sort_for_all(g, target_user, targets, node_emb, model):
    first_player_embeddings = np.vstack(
        [np.repeat(node_emb[target_user].reshape(1, -1), len(targets), axis=0)]
    )
    second_player_embeddings = np.vstack([node_emb[x] for x in targets])

    predictions = model.predict(
        torch.from_numpy(first_player_embeddings).to(DEVICE),
        torch.from_numpy(second_player_embeddings).to(DEVICE),
    )
    predictions = predictions.squeeze().detach().cpu().numpy()
    sorted_idx = np.argsort(predictions)

    return sorted_idx


def return_most_likely(target_user: int, g, model, node_emb, n: int = 5):
    targets = np.delete(g.nodes().numpy(), target_user)
    sorted_idx = _predict_and_sort_for_all(g, target_user, targets, node_emb, model)
    raw_features = pd.read_csv("data/large_twitch_features.csv")
    print("target_user", raw_features.iloc[target_user])
    print(raw_features.iloc[targets[sorted_idx[-n:]]])


def return_least_likely(target_user: int, g, model, node_emb, n: int = 5):
    targets = np.delete(g.nodes().numpy(), target_user)
    sorted_idx = _predict_and_sort_for_all(g, target_user, targets, node_emb, model)
    raw_features = pd.read_csv("data/large_twitch_features.csv")
    print("target_user", raw_features.iloc[target_user])
    print(raw_features.iloc[targets[sorted_idx[:n]]])


def run():
    # preprocess data
    train_test_split()
    train_edges = pd.read_csv("data/train_edges.csv")
    test_edges = pd.read_csv("data/test_edges.csv")
    features = pd.read_csv("data/processed_features.csv")
    # create graph
    g, reverse_eids = create_graph(edges=train_edges, nodes=features)
    train_loss_list = train(g, reverse_eids, n_optimizer_steps=None, nepoch=20)

    # Evaluation
    eval_fp = Path("data/eval_dataset.pkl")
    if eval_fp.exists():
        eval_dataset = pickle.load(open(eval_fp, "rb"))
    else:
        eval_dataset = create_evaluation_dataset(
            test_edges, train_edges["numeric_id_2"].unique()
        )
        pickle.dump(eval_dataset, open("data/eval_dataset.pkl", "wb"))

    node_emb = pickle.load(open("data/embeddings.pkl", "rb"))
    model = pickle.load(open("data/model.pkl", "rb"))

    predictions, labels = predict(model, node_emb, eval_dataset)

    ndcg = ndcg_score(labels, predictions)
    logger.info(f"ndcg_score - {ndcg}")

    # Sanity check for one user:
    print('Top 5 most likely to connect')
    return_most_likely(24516, g, model, node_emb)
    print('Top 5 least likely to connect')
    return_least_likely(24516, g, model, node_emb)
    

if __name__ == "__main__":
    run()
