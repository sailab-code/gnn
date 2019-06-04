Graph Neural Network Model
========


This repo contains a Tensorflow implementation of the Graph Neural Network model.


- **Website (including documentation):** https://sailab.diism.unisi.it/gnn/index.html

Install
-------

Install the latest version of NetworkX:

    $ pip install gnn

For additional details, please see `INSTALL.rst`.

Simple usage example
--------------------



        import GNN
        import Net as n
        
        # Provide your own functions to generate input data
        inp, arcnode, nodegraph, labels = set_load()

        # Create the state transition function, output function, loss function and  metrics 
        net = n.Net(input_dim, state_dim, output_dim)

        # Create the graph neural network model
        g = GNN.GNN(net, input_dim, output_dim, state_dim)
        
        #Training
                
        for j in range(0, num_epoch):
            g.Train(inp, arcnode, labels, count, nodegraph)
            
            # Validate            
            print(g.Validate(inp_val, arcnode_val, labels_val, count, nodegraph_val))


License
-------

Released under the 3-Clause BSD license (see `LICENSE`)

Copyright (C) 2004-2019 Matteo Tiezzi
Matteo Tiezzi <mtiezzi@diism.unisi.it>
Alberto Rossi <alrossi@unifi.it>