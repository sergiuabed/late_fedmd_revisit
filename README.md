# FedMD

Contributors:

- Sergiu Abed
- Riccardo Musumarra

## 1. Introduction

Federated learning is a machine learning technique in which multiple models collaborate by sharing their knowledge so that each model can improve its performance when executing a certain task.

The need for a framework like federated learning arises from the low availability of training data of a client and the privacy issues that can be caused by collecting sensitive data from multiple clients in a centralized fashion. Instead of sharing their own data, each client updates the global model received from the coordinating server on its own data and sends the updated model to the server which then averages the received models from all clients to obtain a new global model [1].

There are 3 main challanges that federated learning faces:

- statistical heterogeneity
- system heterogeneity
- privacy threats

FedMD addresses the first two.

The initial federated learning implementation (as described above) works only when all clients have the same network architecture for their models, since model parameters averaging doesn't work otherwise. It makes perfect sense to think that each client would want to have their own implementation of their model based on their computational resources. Some institutions may have more computational power at their disposal which they can use to train better performing models, while others may have limited resources and so they would have to resort to more computational efficient architectures. One solution to this problem (i.e. system heterogeneity) is FedMD (Federated Model Distillation) [2].

## 2. Related work

The goal of this work is reproducing the experiments presented in the paper "FedMD: Heterogeneous Federated Learning via Model Distillation" [2]. However, our problem setting is a bit different than what the authors simulated. We address the difference in the "Experiment" section of this text.

The paper presents a federated setting consisting of 10 clients under i.i.d. (independent and identically distributed) and non-i.i.d. datasets conditions and each client can choose its own model architecture. For one of the experiments, they used CIFAR10 as public dataset and CIFAR100 as private dataset. Each client has been assigned a subset of CIFAR100. In the i.i.d. case, the task of a client is to classify each image as one of the 100 possible subclasses, whereas in the non-i.i.d. situation, a client must classify each image as belonging to one of the 20 possible superclasses (a superclass has a larger granularity then a subclass, i.e. multiple subclasses may fall under the same superclass).

## 3. Methodology

In FedMD setting, we no longer have a global model that is being shared among all clients. Instead, the coordinating server picks a certain number of clients per round and a certain dataset which the chosen clients must use for that round for collaborative training.

We have two types of datasets: a public dataset and a private dataset. The public dataset is used for transfer learning, i.e. each client first trains its model on this common dataset, and then each client trains the model on its private dataset.

Once training on both public and private datasets is finished, the clients can start getting involved in collaboration rounds.

A round consists of the following phases:

- public subset and clients selection: the server chooses a subset of the public dataset (or the whole dataset) and the clients that must participate in the round and sends to each of them the chosen subset

- communicate: each client computes the scores on each image of the subset and sends them to the server

- aggregate: the server averages the scores received from the clients to compute a consensus

- distribute: the consensus is sent to all the participating clients

- digest: the clients perform training on the received consensus, i.e. for each image in the public subset the loss is computed between the output of the model and the consensus value corresponding to that image (i.e. in the loss function, instead of giving as input the label of the image, you put the average of the outputs for that image from all the participating clients)

- revisit: each client trains again on its own private dataset for few epochs

 During the execution of our implementation of FedMD, we assumed that all clients are available for collaboration. Also, during the selection of the clients for participation in a round, we made sure that each type of architecture is selected at least once, so that we can see better the effects of FedMD.

 When computing the consensus, we used a weighted sum approach. The scores computed by models with higher accuracy have higher weights.

## 4. Experiment

 In the following we describe our problem setting:

- number of clients: 100
- number of clients participating in a round: 10
- public dataset: CIFAR10
- private dataset: CIFAR100 split in 100 subsets using Dirichlet's distribution to obtain i.i.d. or non-i.i.d. case. We used the splits provided by our teaching assistant
- test dataset: CIFAR100 on its entirety
- network architectures:
  - ResNet20-BN (batch normalization)
  - ResNet20-GN (group normalization)
  - DenseNet20
  - DenseNet10
  - ShuffleNetBig
  - ShuffleNetSmall
- digest phase optimizer: Adam with LR (learing rate) set to 0.001
- revisit phase optimizer: SGD with LR=0.001 for i.i.d. case and LR=0.0001 for non-i.i.d.
- revisit epochs: 5 for i.i.d, 1 for non-i.i.d.
- rounds performed:
  - non-i.i.d.: 77
  - i.i.d.: 62
- platform: Google Colab

 In both i.i.d. and non-i.i.d conditions, the task of a client is to classify each image as one of the possible 100 subclasses. The client models were trained on their private dataset until convergence.

### 4.1 I.i.d. results

![iid best result](/plots/iid_best_results.png "iid best result")*Figure 1.1*

![iid best result with upperbound](/plots/iid_best_results_with_upperbound.png "iid best result")*Figure 1.2*

![iid averaged result](/plots/iid_averaged_results.png "iid best result")*Figure 1.3*

![iid averaged result with upperbound](/plots/iid_averaged_results_with_upperbound.png "iid best result")*Figure 1.4*

### 4.2 Non-i.i.d. results

![non-iid best result](/plots/non_iid_best_results.png "non-iid best result")*Figure 2.1*

![non-iid best result with upperbound](/plots/non_iid_best_results_with_upperbound.png "non-iid best result")*Figure 2.2*

![non-iid averaged result](/plots/non_iid_averaged_results.png "non-iid best result")*Figure 2.3*

![non-iid averaged result with upperbound](/plots/non_iid_averaged_results_with_upperbound.png "non-iid best result")*Figure 2.4*

The "best results" plots were obtained by selecting for each architecture the client model with the highest gain in accuracy, i.e. the highest increase from independent training (i.e., before any collaboration) to the peak of the graph.

The "averaged results" plots were obtained by averaging the results of all clients using the same architecture.

The x-axis corresponds to the number of rounds performed. At x=0 we have the accuracy of a model before participating in the collaborative learning, i.e right after transfer learning on the public dataset and training on the private dataset (we call this independent training).

The constant dashed functions on the left depict the accuracies of the models after independent training. The constant dash-dotted lines on the right correspond to the accuracies of the models if the private datasets were collected, put together and made available to all clients to be used for training their models.

As it can be seen, FedMD helps in boosting test accuracy for all model architectures. However, there is still a quite high gap between the accuracy obtained after collaborative learning and the upper bound, especially in the non-i.i.d. case. We suspect this is caused by the very small private training dataset that each client has at its disposal, since CIFAR100 was split in 100 subsets.

## 5. Conclusion

FedMD is a technique for performing collaborative learning among clients having models of different architectures. Unlike the initial federated learning techniques, which build a global model by averaging local models from all clients, FedMD uses the scores computed by all clients to compute a consensus on which the clients train. Because of this, FedMD solves both statistical heterogeneity and system heterogeneity.

## 6. Bibliography

[ [1](https://arxiv.org/abs/1602.05629) ] Communication-Efficient Learning of Deep Networks
from Decentralized Data

[ [2](https://arxiv.org/abs/1910.03581) ] FedMD: Heterogenous Federated Learning
via Model Distillation

---

---

## Repository description and how to run our experiments

### Repository description

- `./model_implementations`: model architecture implementations and some utilities

- `./data`: CIFAR10 and CIFAR100 splits provided by the teaching assistant

- `./client`: client class implementation, private_dataloader and a document telling the network architecture used by each client

- `./server`: server class implementation

- `./baselines_public_cifar10`: here model checkpoints for each model architecture are stored after training on CIFAR10

- `./upper_baseline_public_cifar100`: here model checkpoints for each model architecture are stored after training on the whole CIFAR100

- `./independent_train_0.0`: here are stored the checkpoints for all 100 clients after transfer learning and training on private dataset under non-i.i.d. condition

- `./independent_train_1000.0`: here are stored the checkpoints for all 100 clients after transfer learning and training on private dataset under i.i.d. condition

- `./fedmd_results_0.0`: FedMD results obtained under non-i.i.d. condition

- `./fedmd_results_1000.0`: FedMD results obtained under i.i.d. condition

- `./notebooks`: notebooks to be run on Google Colab to recreate our experiments

- `./plots`: FedMD results plotted

### Run our experiments

Under `./notebooks` you may find our Jupyter notebooks we wrote to make our experiments.

- `Train_baselines.ipynb`: train the 6 architectures on CIFAR10. It can also be used for training on CIFAR100

- `Independent_client_train.ipynb`: perform transfer learning on all 100 clients and train them on their private datasets

- `FedMD_execution.ipynb`: execute FedMD
