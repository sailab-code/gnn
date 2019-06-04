import numpy as np
import pandas as pd
import scipy.io as sio
import os
from scipy.sparse import coo_matrix


def GetInput(mat, lab, batch=1, grafi=None):
    # numero di batch
    batch_number = grafi.max() // batch

    dmat = pd.DataFrame(mat, columns=["id_1", "id_2"])
    dlab = pd.DataFrame(lab, columns=["lab" + str(i) for i in range(0, lab.shape[1])])
    # darch=pd.DataFrame(arc, columns=["arch"+str(i) for i in range(0,arc.shape[1])])
    dgr = pd.DataFrame(grafi, columns=["graph"])

    dresult = dmat
    dresult = pd.merge(dresult, dlab, left_on="id_1", right_index=True, how='left')
    dresult = pd.merge(dresult, dlab, left_on="id_2", right_index=True, how='left')
    # dresult=pd.concat([dresult, darch], axis=1)
    dresult = pd.merge(dresult, dgr, left_on="id_1", right_index=True, how='left')

    data_batch = []
    arcnode_batch = []
    nodegraph_batch = []
    node_in = []
    for i in range(0, batch_number + 1):

        # prendo i dati
        grafo_indexMin = (i * batch)
        grafo_indexMax = (i * batch) + batch

        adj = dresult.loc[(dresult["graph"] >= grafo_indexMin) & (dresult["graph"] < grafo_indexMax)]
        min_id = adj[["id_1", "id_2"]].min(axis=0).min()

        adj["id_1"] = adj["id_1"] - min_id
        adj["id_2"] = adj["id_2"] - min_id

        min_gr = adj["graph"].min()
        adj["graph"] = adj["graph"] - min_gr

        data_batch.append(adj.values[:, 1:-1])

        # arcMat
        max_id = int(adj[["id_1", "id_2"]].max(axis=0).max())

        max_gr = int(adj["graph"].max())

        mt = adj[["id_1", "id_2"]].values

        arcnode = np.zeros((mt.shape[0], max_id + 1))
        for j in range(0, mt.shape[0]):
            arcnode[j][mt[j][0]] = 1

        arcnode_batch.append(arcnode)

        # nodegraph
        nodegraph = np.zeros((max_id + 1, max_gr + 1))

        for t in range(0, max_id + 1):
            val = adj[["graph"]].loc[(adj["id_1"] == t) | (adj["id_2"] == t)].values[0]
            nodegraph[t][val] = 1

        nodegraph_batch.append(nodegraph)
        # node number in each graph
        grbtc = dgr.loc[(dgr["graph"] >= grafo_indexMin) & (dgr["graph"] < grafo_indexMax)]
        node_in.append(grbtc.groupby(["graph"]).size().values)

    return data_batch, arcnode_batch, nodegraph_batch, node_in


def set_load_subgraph(data_path, set_type):
    # load adjacency list
    types = ["train", "valid", "test"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

        # load adjacency list
        mat = sio.loadmat(os.path.join(data_path, 'conmat{}.mat'.format(set_type)))
        adj = coo_matrix(mat["conmat_{}set".format(set_type)])
        adj = np.array([adj.row, adj.col]).T

        # load node label
        mat = sio.loadmat(os.path.join(data_path, "nodelab{}.mat".format(set_type)))
        lab = np.asarray(mat["nodelab_{}set".format(set_type)]).T

        # load target and convert to one-hot encoding
        mat = sio.loadmat(os.path.join(data_path, "tar{}.mat".format(set_type)))
        target = np.asarray(mat["target_{}set".format(set_type)]).T
        labels = pd.get_dummies(pd.Series(target.reshape(-1)))
        labels = labels.values
        # compute inputs and arcnode
        inp, arcnode, nodegraph, nodein = GetInput(adj, lab, 1, np.zeros(700, dtype=int))
        return inp, arcnode, nodegraph, nodein, labels

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)

def set_load_clique(data_path, set_type):
    import load as ld
    # load adjacency list
    types = ["train", "validation", "test"]
    train = ld.loadmat(os.path.join(data_path, "cliquedataset.mat"))
    train = train["dataSet"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

        # load adjacency list
        # take adjacency list
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'])
        adj = np.array([adj.row, adj.col]).T

        # take node labels
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T

        # take targets and convert to one-hot encoding
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        labels = pd.get_dummies(pd.Series(target))
        labels = labels.values

        # compute inputs and arcnode
        get_lab = lab.reshape(lab.shape[0], 1) if set_type == "train" else lab.reshape(700, 1)
        inp, arcnode, nodegraph, nodein = GetInput(adj, get_lab, 1,
                                                           np.zeros(700, dtype=int))
        return inp, arcnode, nodegraph, nodein, labels

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)


def set_load_mutag(set_type, train):
    # load adjacency list
    types = ["train", "validation", "test"]
    try:
        if set_type not in types:
            raise NameError('Wrong set name!')

            ############ training set #############

            # take adjacency list
        adj = coo_matrix(train['{}Set'.format(set_type)]['connMatrix'])
        adj = np.array([adj.row, adj.col]).T

        # take node labels
        lab = np.asarray(train['{}Set'.format(set_type)]['nodeLabels']).T
        mask = coo_matrix(train['{}Set'.format(set_type)]["maskMatrix"])

        # take target, generate output for each graph, and convert to one-hot encoding
        target = np.asarray(train['{}Set'.format(set_type)]['targets']).T
        v = mask.col
        target = np.asarray([target[x] for x in v])
        # target = target[target != 0] # equivalent code
        labels = pd.get_dummies(pd.Series(target))
        labels = labels.values

        # build graph indices
        gr = np.array(mask.col)
        indicator = []
        for j in range(0, len(gr) - 1):
            for i in range(gr[j], gr[j + 1]):
                indicator.append(j)
        for i in range(gr[-1], adj.max() + 1):
            indicator.append(len(gr) - 1)
        indicator = np.asarray(indicator)

        # take input, arcnode matrix, nodegraph matrix
        inp, arcnode, nodegraph, nodein = GetInput(adj, lab, indicator.max() + 1, indicator)

        return inp, arcnode, nodegraph, nodein, labels

    except Exception as e:
        print("Caught exception: ", e)
        exit(1)


