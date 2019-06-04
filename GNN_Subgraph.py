import tensorflow as tf
import gnn.Library as Library
import gnn.GNN as GNN
import Net_Subgraph as n

##### GPU & stuff config
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

data_path = "./Data/Subgraph"

#############DATA LOADING##################################################
############# training set ################

inp, arcnode, nodegraph, nodein, labels = Library.set_load_subgraph(data_path, "train")
############ test set ####################

inp_test, arcnode_test, nodegraph_test, nodein_test, labels_test = Library.set_load_subgraph(data_path, "test")

############ validation set #############

inp_val, arcnode_val, nodegraph_val, nodein_val, labels_val = Library.set_load_subgraph(data_path, "valid")

# set input and output dim, the maximum number of iterations, the number of epochs and the optimizer
threshold = 0.01
learning_rate = 0.0001
state_dim = 5
tf.reset_default_graph()
input_dim = len(inp[0][0])
output_dim = 2
max_it = 50
num_epoch = 10
optimizer = tf.train.AdamOptimizer

# initialize state and output network
net = n.Net(input_dim, state_dim, output_dim)

# initialize GNN
#param = "st_d" + str(state_dim) + "_th" + str(threshold) + "_lr" + str(learning_rate)
#print(param)

tensorboard = True

# g = GNN.GNN(net, input_dim, output_dim, state_dim,  max_it, optimizer, learning_rate, threshold, False, param, config,
#             tensorboard)
g = GNN.GNN(net, input_dim, output_dim, state_dim)

# train the model
count = 0


for j in range(0, num_epoch):
    g.Train(inp[0], arcnode[0], labels, j)
    if j % 30 == 0:
        print(g.Validate(inp_val[0], arcnode_val[0], labels_val, j, nodegraph_val[0]))

# evaluate on the test set
print(g.Evaluate(inp_test[0], arcnode_test[0], labels_test, nodegraph_test[0]))
