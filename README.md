# Graph Learning Example
This example is used as a demo in the lightening talk about graphs at Bumble X Data science festival taken place on Dec. 1st 2022.

The example uses [Twitch Gamer Social Network data](https://snap.stanford.edu/data/twitch_gamers.html). It contains Twitch users as nodes and their mutual following relationships as edges. 

*_Task_*
- Predict how likely users form connection
*_Evaluation Dataset_*
For each player, create a sequence of 5 Positive sample: connected edges, and 5 Negative sampling from unconnected edges.
*_Evaluation Metric_*
NDCG


The example builds a simple GNN with GaphSage layers can be found in `src/model.py`
Data preprocessing and other utils can be found in `src/utils.py`
Model training code can be found in `src/train.py`

### You will need the following packages
```
pip install scikit-learn numpy pandas torch tqdm pandarallel
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
```
[NOTE]: install dgl that's compatible with your cuda version. [reference](https://www.dgl.ai/pages/start.html)


### To run the code
```
cd graph_learning_demo
```
Download data first.
```
wget https://snap.stanford.edu/data/twitch_gamers.zip
mkdir data
cd data
unzip ../twitch_gamers.zip
```

Then run the following
```
python run.py
```

You can modify example_user by updaing the input argument for `run.run()`
It will print the top mostly likely users to form a connection with the example user, as well as the least likely users to form a connection with the example user. 