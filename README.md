# Task-Agnostic Graph Neural Network Evaluation via Adversarial Collaboration

A CST Part III research project

Project supervisor: Professor Pietro Liò \
Project co-supervisors: Dr Dominique Beaini, Hannes Stärk

### Abstract

Graph Neural Networks (GNNs) have experienced rapid growth over the last decade, and have been successful in many real-world applications. In order to cope with the rapid growth of this field, it is increasingly demanding to develop reliable GNN evaluation methods to facilitate GNN research and quantify their progress. Current GNN benchmarking methods all focus on comparing the GNNs with respect to their training performances on some node/graph classification/regression tasks in certain datasets, but there has not been any principled, task-agnostic method to directly compare the two GNNs.

Furthermore, learning informative representations of graph-structured data using self-supervised learning (SSL) is becoming crucial in many real-world tasks nowadays, when labelled data are expensive and limited. Most of the existing graph SSL works incorporate handcrafted augmentations to the graph, which has several severe difficulties due to the unique characteristics of graph-structured data. Therefore, it is highly needed to develop a principled SSL framework across various types of graphs, that does not require handcrafted augmentations.

In this project, I tackled both questions above, and developed GraphAC (Graph Adversarial Collaboration), a conceptually novel, principled, task-agnostic, and stable framework for evaluating GNNs through contrastive self-supervision. It consists of two different GNNs directly competing against each other, with the more expressive GNN wins by producing more informative graph representations. I built two frameworks for GraphAC, and designed a novel objective function that enables stable and effective training of two different GNNs, inspired by Barlow Twins. 

The experimental results show that GraphAC succeeds in distinguishing GNNs of different expressivity across various aspects including the number of layers, hidden dimensionality, aggregators, GNN architecture and edge features, and always allow more expressive GNNs to win with statistically significant difference. GraphAC proved to be a principled and reliable GNN evaluation method, and enables stable SSL without needing handcrafted augmentations.
