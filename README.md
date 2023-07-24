# Soccer Ball Outcome Prediction Project
This project aims to utilize Graph Neural Network to predict the results of soccer ball games based on the individual abilities of soccer players and the formation of all players. This model hopes to assist commercial event evaluation/guessing and team formation analysis, player scheduling, and ability training to achieve the goal of using technology to assist sports events.
Inspired by: https://deepfindr.github.io/

- [Key words](#Key-words)
- [Purpose of project](#Purpose-of-project)
- [Overview and comparison of existing related research](#Overview-and-comparison-of-existing-related-research)
- [Research methods](#Research-methods)
- [Implementation](#Implementation)
- [Experiment](#Experiment)
- [Performance and Result](#Performance-and-Result)
- [Conclusion](#Conclusion)
- [Reference](#Reference)
 
## Key words
Graph Neural Network, Transformer, Ensemble

## Purpose of project
The original idea of this project came from my enthusiasm for sports events, and the approaching World Cup at that time. I thought it would be interesting to have a machine learning model that could predict the outcome of the game from the team formations announced before the game and FIFA's player ability ratings. The ultimate goal of this project is to use this model to assist commercial event evaluation and team formation analysis, player scheduling, and ability training to achieve the goal of using technology to assist sports events.

## Overview and comparison of existing related research
Compared with other machine learning fields, the research of graph neural networks are relatively unpopular, but the application of graph neural network is quite extensive nowadays. At present, it is mostly used in chemistry, transportation, and recommendation systems, and there is also related research on images, text, etc. [1]. Recently, there are related papers on the application of graph neural networks to the prediction of sports events [2], but the main research objects are American football and e-sports, and the research focuses on adding representations to the input graphs to improve the performance of the model, which is a little different from this project.

## Research methods
![Alt text](https://imgur.com/a/CpJPdDL)
I divided the overall project architecture into three parts:
1. Graph conversion: For GNN, the input data must be in the form of a Graph, so before training, the unstructured data  (formations & player statistics) must be converted into graphs. Therefore, I convert each match into a graph, each of them will have 22 nodes, representing all players on the field (11 from the home team and 11 from the away team). More, there are edges connecting nodes to form the graph. The graph data processing is therefore divided into two parts:
    1. Node: A Node represents a player on the court, and each node will have 40 Features (i.e. X, Y, Height, Ball Skills, Defense…)
    2. Edge: To present the formation of the team, there will be edge connections between nodes. Edges will connect two adjacent nodes (side-to-side and front-to-back), and the node in the front will also be connected to the opponent's front node, completing the graph conversion of a single match.
    ![Alt text](https://imgur.com/a/6HgPsLd)
2. Model training: After processing the conversion of graphs, model training will be carried out next. During this part, I imported hyperparameter search, adjusting parameters with different initializations like model depth, number of neurons, learning rate, etc.
3. Ensemble: After training multiple models, I selected models with better performance for ensembling, and used ensembles to predict outcomes of matches (away win/tie/home win) for evaluation.

## Implementation 
1. GNN: In order to make graph-level predictions, I utilized Transformers Convolution Layer and Top-K Pooling in the model design:
    1. Transformers Convolution Layer: When training the model, I refer to the attention concept in Transformers. Since how graphs are connected is arbitrarily designed, and there are no differences in the weights between edges, the introduction of Attention allows the model to determine the importance of an edge between a node and another node.
    2. Top-K Graph Pooling: Different from node-level prediction, for graph-level prediction, models need to reduce all feature Data (node embedding) of the graph to an embedding vector as the representation of that graph. I am utilizing Top-K Pooling as the hierarchical pooling. The model will select which K nodes will stay according to a learnable vector, and iteratively reduce the number of nodes in the graph. After message passing, the feature data of the removed nodes will be distributed to neighboring nodes. The final embedding vector composition is the average and maximum value of all node vectors at each stage at that time.
![Alt text](https://imgur.com/a/lbrFWuZ)
The overall design of the model is shown below. It consists of several blocks of Transformer Convolution Layer, Linear Transformation Layer, and Batch Normalization Layer, and then the input graph will be downsized through the Top-K Pooling Layer. The average and maximum values of Node Embedding are used for the composition of the final graph representation every time the pooling layer is passed.
![Alt text](https://imgur.com/a/4bYVYvM)

2. Ensemble: In order to optimize the prediction performance, I took the top 10 models (Top10 Highest Accuracy/F1 Score) with the best performance among the 176 different initialization models and stacked them in the form of stacking ensemble for the final performance evaluation. In addition, I also added the concept of threshold to simulate the situation of not predicting unsure events during actual commercial outcome prediction. The implementation of the threshold is to make the model only predict the most confident top N% of all events.

## Experiment
1. GNN: I adjusted the following different hyperparameters for experiments:
    1. Batch Size: 64, 128, 256
    2. Learning Rate: 0.01, 0.02, 0.03
    3. Embedding Size: 16, 32, 64
    4. Model Block Layers: 2, 3, 4, 5
    5. Model Dense Neurons: 32, 64, 128, 256
    6. Model Transformer Attention Heads: 1, 2, 3, 4
    7. Model Transformer Dropout Rate: 0.2, 0.5, 0.9
    8. Model Top K Ratio: 0.2, 0.5, 0.8, 0.9
    9. Model Top K Every N Blocks: 1, 2, 3
    10. Weight Decay: 0.00001, 0.0001, 0.001
    11. SGD Momentum: 0.5, 0.8, 0.9
    12. Scheduler Gamma: 0.5, 0.8, 0.9, 0.995, 1
2. Ensemble: 
    1. Model Selection: Highest Validation Accuracy, Highest Validation F1 Score
    2. Ensemble Output: Voting, Average Output
    3. Threshold: 30%, 50%, 70%, 90%, 100%
    4. Threshold Target: Smallest Vote Variance, Highest Average Output

## Performance and Result
![Alt text](https://imgur.com/a/B7czjX1)
The results are shown in the table above. Using the target with the largest proportion (home victory) among the three labels in the Dataset (away win/tie/home win) as the baseline, it can be seen that all 8 models have successfully improved the accuracy by about 10%. In addition, the two combinations of Voting-Avg Output and Avg-Vote Var composed of F1 Score at a threshold of 30% both achieved an Accuracy of nearly 80%, while Avg-Avg Output composed of Accuracy stood out for the rest of the thresholds.

## Conclusion
It can be seen from the results that although the GNN model has reached a certain level and can assist in predicting the results of soccer matches to a certain extent, its overall performance still needs to be improved. More, because there are some known "stronger" teams, the model still requires more evaluation to see if there is a significantly more prominent performance in prediction compared with the experienced soccer experts. In addition, due to the randomness of sports events, there will be considerable restrictions on prediction, and the performance of the model will also be affected by draws or upsets. Even so, the capability of this model can still provide people with relatively fast, automated and convenient predictions for the results of soccer matches, and achieve the goals of assisting commercial event evaluation/guessing and team formation analysis, player scheduling, and ability training.
Future research goals may be towards adding Normalizing Flow, which is capable of dealing with the randomness of data, as an improvement.

## Reference
[1] Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C., & Sun, M. (2020). Graph neural networks: A review of methods and applications. AI Open, 1, 57-81. https://doi.org/10.1016/j.aiopen.2021.01.001
[2] Xenopoulos, P., & Silva, C. (2022). Graph Neural Networks to Predict Sports Outcomes. arXiv. https://doi.org/10.1109/BigData52589.2021.9671833
