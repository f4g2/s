#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.


def get_torchbiggraph_config():
    """
    config = dict(  # noqa
        # I/O data
        entity_path="data/s2orc",
        edge_paths=[            
            "data/s2orc/train_partitioned", 
            #"data/test_partitioned"
        ],
        checkpoint_path="model/s2orc",
        
        # Graph structure
        entities={"paper_id": {"num_partitions": 1}},
        relations=[
            {"name": "citation", "lhs": "paper_id", "rhs": "paper_id", "operator": "none"}
        ],
        # Scoring model
        dimension=300,  # 1024?
        global_emb=False,
        
        # Training
        num_epochs=2,  # 50? checkpoints?
        lr=0.001,
        
        # Misc
        #hogwild_delay=2,
        
        eval_fraction=0,  # to reproduce results, we need to use all training data
        
        # GPU
        num_gpus=2,
        batch_size=10_000,
    )
    """
    
    # Twitter config: https://github.com/facebookresearch/PyTorch-BigGraph/issues/86
    # (Twitter as roughly the same size as S2ORC citation graph) 
    config = {
        # I/O data
        "entity_path": "./data/biggraph/s2orc_100k",
        "edge_paths": [
            "./data/biggraph/s2orc_100k/train_partitioned",
        ],
        "checkpoint_path": "./data/biggraph/models/s2orc_100k",
        "checkpoint_preservation_interval": 5,

        # Graph structure
        "entities": {
            "paper_id": {"num_partitions": 1},
        },
        "relations": [{"name": "citation", "lhs": "paper_id", "rhs": "paper_id", "operator": "none"}],

        # Scoring model
        "dimension": 500,
        "max_norm": 1.0,
        "global_emb": False,
        "comparator": "dot",

        # Training
        "num_epochs": 50,
        "num_edge_chunks": 10,
        "batch_size": 10_000,
        "num_uniform_negs": 0,
        "margin": 0.15,
        "lr": 0.1,

        # GPU
        #"num_gpus": 2,
        
        # Evaluation during training
        "eval_fraction": 0,  # to reproduce results, we need to use all training data
    }
    
    return config
