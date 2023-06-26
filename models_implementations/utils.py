"""This file contains useful functions"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

def plot_stats(x: list, data:list[list], save_path:str, title:str=None, x_label:str=None, y_label:str=None, data_labels:list[str]=None, x_lim:tuple=None, y_lim:tuple=None ):
    """
    Plot multiple data series that share the same abscissa.
        - x : a list of values for the abscissa
        - data: a list of lists of values to be plotted
        - save_path: the path of the file in which to save the plot
        - title: title of the plot
        - x_label: label of the x axis
        - y_label: label of the y axis
        - data_label: list of labels for each data series
        - x_lim: limits of the x axis
        - y_lim: limits of the y axis
    """
    plt.figure()
    
    if title is not None:
        plt.title(title) 
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    
    if x_lim is not None:
        if len(x_lim) > 1:
            plt.xlim(x_lim[0], x_lim[1])
        else:
            plt.xlim(x_lim[0])  
    if y_lim is not None:
        if len(y_lim) > 1:
            plt.ylim(y_lim[0], y_lim[1])
        else:
            plt.ylim(y_lim[0])

    for i, y in enumerate(data):
        if len(data_labels) >= i+1:
            label = data_labels[i]
        else:
            label = None
        plt.plot(x, y, label=label)

    plt.legend()    

    plt.savefig(save_path)

    plt.close()


def save_model(model:nn.Module, path:str, epoch:int=None, accuracy:float=None, lr:float=None):
    """
    Saves the state of a model as a dictionary.
        - model: the model whose state needs to be saved
        - path: path in which to save the state 
        - epoch: epoch number, useful for checkpointing
        - accuracy: accuracy of current model
        - lr: learning rate, useful for checkpointing
    """
    state = {'weights': model.state_dict(),
                'accuracy': accuracy,
                'epoch': epoch,
                'lr': lr,
            }
    torch.save(state, path)

def load_model(path:str) -> dict:
    """
    Load a model's state; it's the counterpart of the function "save_model".
        - path: path of the state

    Return a dictionary with the following keys:
        - weights: model's parameters, to be loaded with the function "nn.Module.load_state_dict"
        - accuracy: model's accuracy
        - epoch: epoch number, useful for checkpointing
        - lr: learning rate, useful for checkpointing
    """
    return torch.load(path)

def model_size(net:nn.Module) -> int:
    """
    Return a model's number of parameters.
    """
    tot_size = 0
    for param in net.parameters():
        tot_size += param.size()[0]
    return tot_size
