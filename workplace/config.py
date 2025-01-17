import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec


HYPERPARAMETERS = {
    "batch_size": [64, 128, 256],
    #"batch_size": [64],
    "learning_rate": [0.01, 0.02, 0.03],
    #"threshold": [0.3, 0.5, 0.7, 0.8], 
    "model_embedding_size": [16, 32, 64],
    "model_layers": [2, 3, 4, 5],
    "model_dense_neurons": [32, 64, 128, 256], 
    #"learning_rate": [0.001, 0.0015, 0.002, 0.0025, 0.003],
    "weight_decay": [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9, 0.8, 0.5],
    "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
    #"scheduler_gamma": [0.8],
    #"pos_weight" : [1.0, 1.3, 0.8, 0.9],  
    "model_attention_heads": [1, 2, 3, 4],
    #"model_attention_heads": [2, 3],
    "model_dropout_rate": [0.2, 0.5, 0.9],
    #"model_dropout_rate": [0.2],
    "model_top_k_ratio": [0.2, 0.5, 0.8, 0.9],
    #"model_top_k_ratio": [0.5, 0.8, 0.9],
    "model_top_k_every_n": [1, 2, 3],
    #"model_top_k_every_n": [1]
}


BEST_PARAMETERS = {
    "batch_size": [128],
    "learning_rate": [0.01],
    "weight_decay": [0.0001],
    "sgd_momentum": [0.8],
    "scheduler_gamma": [0.8],
    "pos_weight": [1.3],
    "model_embedding_size": [64],
    "model_attention_heads": [3],
    "model_layers": [4],
    "model_dropout_rate": [0.2],
    "model_top_k_ratio": [0.5],
    "model_top_k_every_n": [1],
    "model_dense_neurons": [256]
}

input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 30), name="x"), 
                       TensorSpec(np.dtype(np.float32), (-1, 11), name="edge_attr"), 
                       TensorSpec(np.dtype(np.int32), (2, -1), name="edge_index"), 
                       TensorSpec(np.dtype(np.int32), (-1, 1), name="batch_index")])

output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)