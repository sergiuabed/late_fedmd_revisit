import numpy as np
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fedmd.models_implementations.train_on_cifar import _training, _validation
from fedmd.models_implementations.utils import load_model, save_model
import os

LOCAL_EPOCH = 1 #5
LR_ADAM = 0.001
LR_SGD = 0.0001 #0.001
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9

FILE_PATH = os.getcwd() + '/logs'   # path for storing model checkpoints and logs on intermediary states of the model

class Client:

    def __init__(self, client_id, public_train_dataloader,private_train_dataloader,
                 private_validation_dataloader, current_consensus, model, model_architecture, digest_epochs=10, device=None):
        self._model = model
        self.model_architecture = model_architecture
        self.client_id = client_id
        self.device = device
        self.digest_epochs = digest_epochs

        self.public_train_dataloader = public_train_dataloader  #during collaboration training, this attribute stores the subset of the public
                                                                #dataset on which the scores (logits) will be computed during a round
        self.private_train_dataloader = private_train_dataloader
        self.private_validation_dataloader = private_validation_dataloader

        self.current_local_scores = None
        self.current_consensus = current_consensus

        self.consensus_loss_func = nn.CrossEntropyLoss() #nn.NLLLoss() #nn.L1Loss()
        self.consensus_optimizer = optim.Adam(self._model.parameters(), LR_ADAM)  # optimizer suggested in FedMD paper with starting lr=0.001
        #self.consensus_optimizer = optim.SGD(  
        #    self._model.parameters(), lr=LR_ADAM, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        #)

        self.accuracies = []
        self.losses = []

    def upload(self):
        print(f"Client {self.client_id} starts computing scores.\n")
        self._model.to(self.device)
        self._model.eval()
        nr_batches = len(self.public_train_dataloader)
        size_batch = self.public_train_dataloader.batch_size
        nr_classes = 100
        
        self.current_local_scores = torch.zeros((nr_batches, size_batch, nr_classes))

        i = 0
        for data in self.public_train_dataloader:
            x = data[0]
            x = x.to(self.device)

            #self.current_local_scores.append(self._model(x))
            
            self.current_local_scores[i] = self._model(x).detach() #softmax applied on server side when computing consensus .softmax(dim=1, dtype=float).detach()
            i += 1

        return self.current_local_scores

    def download(self, current_consensus): #calling this also triggers digest and revisit(i.e. private_train)
        self.current_consensus = current_consensus
        print(f"Client {self.client_id} starts digest phase\n")
        self.digest()
        print(f"Client {self.client_id} revisits its private data for {LOCAL_EPOCH} epochs\n")
        self.private_train()

    def private_train(self):
        #_training(
        #    self._model, self.private_train_dataloader, self.private_validation_dataloader, LOCAL_EPOCH , LR, MOMENTUM, WEIGHT_DECAY, FILE_PATH
        #)


        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        optimizer = optim.SGD(  
            self._model.parameters(), lr=LR_SGD, weight_decay=WEIGHT_DECAY
        )
        LR = LR_SGD
        #optimizer = self.consensus_optimizer #parameters to optimize already passed during the init of the client

        # Send to device
        net = self._model
        net = net.to(self.device)
        # Optimize


        acc = _validation(net, self.private_validation_dataloader)
        print(f"Current Val Accuracy right after digest and before revisit = {acc}")
        print()

        # Train
        max_accuracy = 0
        for epoch in range(LOCAL_EPOCH):
            print(
                "Starting epoch {}/{}, LR = {}".format(
                    epoch + 1, LOCAL_EPOCH, LR
                )
            )
            sum_losses = torch.zeros(1).to(self.device)

            # Iterate over the training dataset in batches
            for images, labels in self.private_train_dataloader:
                # Bring data over the device of choice
                images = images.to(self.device)
                labels = labels.to(self.device)

                net.train()  # Sets module in training mode

                optimizer.zero_grad()  # Zero-ing the gradients

                # Forward pass to the network
                outputs = net(images)

                # Compute loss based on output and ground truth
                loss = criterion(outputs, labels)
                sum_losses += loss

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

            # Compute and log the average loss over all batches
            avg_loss = sum_losses.item() / len(self.private_train_dataloader)
            print(f"Current Avg Loss = {avg_loss}")

            # Compute validation accuracy
            acc = _validation(net, self.private_validation_dataloader)
            print(f"Current Val Accuracy = {acc}")

            # Save the best model
            #if acc > max_accuracy:
            #    save_model(net, FILE_PATH + "/best_model.pth", epoch, acc, LR)
            #    max_accuracy = acc

        print("Max Validation Accuracy: {}".format(max_accuracy))

    def digest(self):   # i.e. approach consensus
        running_loss = 0

        self._model.to(self.device)

        for epoch in range(self.digest_epochs):
            i = 0
            for data in self.public_train_dataloader:
                x = data[0].to(self.device)
                y_consensus = self.current_consensus[i].to(self.device)
                self._model.train()
                self.consensus_optimizer.zero_grad()
                y_pred = self._model(x)#.softmax(dim=1, dtype=float)
                loss = self.consensus_loss_func(y_pred, y_consensus)
                loss.backward()
                self.consensus_optimizer.step()
                running_loss += loss.item()

                i += 1
        return running_loss

    def validation_acc(self):
        acc = _validation(self._model, self.private_validation_dataloader)
        return acc