import tensorflow as tf
import numpy as np
import gnn.gnn_utils as gnn_utils
import gnn.GNN as GNN
import Net_Subgraph as n
from scipy.sparse import coo_matrix

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
data_path = "./data"
#data_path = "./Clique"
set_name = "sub_15_7_200"
############# training set ################


#inp, arcnode, nodegraph, nodein, labels = Library.set_load_subgraph(data_path, "train")
inp, arcnode, nodegraph, nodein, labels, _ = gnn_utils.set_load_general(data_path, "train", set_name=set_name)
############ test set ####################

#inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test = Library.set_load_subgraph(data_path, "test")
inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test, _ = gnn_utils.set_load_general(data_path, "test", set_name=set_name)

############ validation set #############

#inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val = Library.set_load_subgraph(data_path, "valid")
inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val, _ = gnn_utils.set_load_general(data_path, "validation", set_name=set_name)

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.01
learning_rate = 0.01
state_dim = 5
tf.reset_default_graph()
input_dim = len(inp[0][0])
output_dim = 2
max_it = 50
num_epoch = 10000
optimizer = tf.train.AdamOptimizer

# initialize state and output network
net = n.Net(input_dim, state_dim, output_dim)

# initialize GNN
param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
print(param)

tensorboard = False

g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, optimizer, learning_rate, threshold, graph_based=False, param=param, config=config,
            tensorboard=tensorboard)

# train the model
count = 0

######

for j in range(0, num_epoch):
    _, it = g.Train(inputs=inp[0], ArcNode=arcnode[0], target=labels, step=count)

    if count % 30 == 0:
        print("Epoch ", count)
        print("Validation: ", g.Validate(inp_val[0], arcnode_val[0], labels_val, count))

        # end = time.time()
        # print("Epoch {} at time {}".format(j, end-start))
        # start = time.time()

    count = count + 1

# evaluate on the test set
print("\nEvaluate: \n")
print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0])[0])
