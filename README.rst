Graph Neural Network Model
========
..
    .. image:: https://travis-ci.org/networkx/networkx.svg?branch=master
       :target: https://travis-ci.org/networkx/networkx

    .. image:: https://ci.appveyor.com/api/projects/status/github/networkx/networkx?branch=master&svg=true
       :target: https://ci.appveyor.com/project/dschult/networkx-pqott

    .. image:: https://codecov.io/gh/networkx/networkx/branch/master/graph/badge.svg
      :target: https://codecov.io/gh/networkx/networkx


This repo contains a Tensorflow implementation of the Graph Neural Network model.


- **Website (including documentation):** https://sailab.diism.unisi.it/gnn/index.html
..
    - **Mailing list:** https://groups.google.com/forum/#!forum/networkx-discuss
    - **Source:** https://github.com/networkx/networkx
    - **Bug reports:** https://github.com/networkx/networkx/issues

Install
-------

Install the latest version of NetworkX::

    $ pip install gnn


For additional details, please see `Install <https://sailab.diism.unisi.it/gnn/install.html>`_.

Simple usage example
--------------------

::

        import gnn.GNN
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


..
    Bugs
    ----

    Please report any bugs that you find `here <https://github.com/networkx/networkx/issues>`_.
    Or, even better, fork the repository on `GitHub <https://github.com/networkx/networkx>`_
    and create a pull request (PR). We welcome all changes, big or small, and we
    will help you make the PR if you are new to `git` (just ask on the issue and/or
    see `CONTRIBUTING.rst`).

License
-------

Released under the 3-Clause BSD license (see `LICENSE.txt`)::

   Copyright (C) 2004-2019 Matteo Tiezzi
   Matteo Tiezzi <mtiezzi@diism.unisi.it>
   Alberto Rossi <alrossi@unifi.it>