import os
import csv

import matplotlib.pyplot as plt

class ClientStats:
    def __init__(self, client_id, independent_acc, upper_bound_acc, collab_accs, architecture):
        self.client_id = client_id
        self.independent_acc = independent_acc
        self.upper_bound_acc = upper_bound_acc
        self.collab_accs = collab_accs
        self.architecture = architecture
        
        self.max_collab_acc = max(self.collab_accs.values())
        self.acc_improvement = self.max_collab_acc - self.independent_acc

    def __str__(self):
        return f"client_id: {self.client_id}, independent_acc: {self.independent_acc}, collab_accs: {self.collab_accs}, arch: {self.architecture}, improv: {self.acc_improvement}"

def get_upper_bounds():
    archs = ["/densenet/densenet20", "/densenet/densenet10", "/resnet20/resnet20_bn", "/resnet20/resnet20_gn", "/shufflenetv2/shufflenetbig", "/shufflenetv2/shufflenetsmall"]
    archs_upperAcc = {}
    for a in archs:
        upper_path = f"upper_baseline_public_cifar100{a}/stats.csv"
        with open(upper_path, 'r') as data:
            archs_upperAcc[a] = 0
            for line in csv.reader(data):
                if line[0] != "epoch":
                    if archs_upperAcc[a] < float(line[2]):
                        archs_upperAcc[a] = float(line[2])

    return archs_upperAcc

def init_clientStats(archs_upperAcc, alpha):
    client_ids = [i for i in range(100)]

    clients = []
    for c in client_ids:
        root_dir = f"independent_train_{alpha}"

        stats_path = f"{root_dir}/client{c}/stats_{alpha}.csv"
        
        independent_acc = 0
        with open(stats_path, 'r') as data:
            for line in csv.reader(data):
                if line[0] != "epoch":
                    independent_acc = float(line[2] if alpha==1000.0 else line[1]) #the results for alpha=0.0 are structured differently because the
                                                                                    #accuracy was computed wrongly (however, the checkpoints are correct)
                                                                                    #so, the accuracy was recomputed


        root_dir = f"fedmd_results_{alpha}/"
        stats_path = f"{root_dir}/client{c}/stats_{alpha}.csv"

        collab_accs = {}
        with open(stats_path, 'r') as data:
            for line in csv.reader(data):
                if line[0] != "round":
                    collab_accs[int(line[0])]=float(line[1])

        architecture = None
        with open("client/client_architectures.csv") as data:
            for line in csv.reader(data):
                if line[0] == str(c):
                    architecture = line[1]

        client = ClientStats(c, independent_acc, archs_upperAcc[architecture], collab_accs, architecture)
        clients.append(client)

    return clients

def best_results(clients):
    archs = ["/densenet/densenet20", "/densenet/densenet10", "/resnet20/resnet20_bn", "/resnet20/resnet20_gn", "/shufflenetv2/shufflenetbig", "/shufflenetv2/shufflenetsmall"]

    arch_best = {}
    for arch in archs:
        filtered_clients = [c for c in clients if c.architecture == arch]
        best_client = sorted(filtered_clients, key=lambda c: c.acc_improvement, reverse=True)[0]
        arch_best[arch]=best_client

    return arch_best

def averaged_results(clients, upperBounds):
    """returns a dictionary of ClientStats objects
       Each object corresponds to a fictitious client representing the averaged results for the corresponding architecture"""

    archs = ["/densenet/densenet20", "/densenet/densenet10", "/resnet20/resnet20_bn", "/resnet20/resnet20_gn", "/shufflenetv2/shufflenetbig", "/shufflenetv2/shufflenetsmall"]

    arch_avg = {}
    for arch in archs:
        filtered_clients = [c for c in clients if c.architecture == arch]
        min_nr_rounds = min([len(c.collab_accs) for c in filtered_clients])
        
        avgs = [0 for i in range(min_nr_rounds)]
        for c in filtered_clients:
            for i in range(min_nr_rounds):
                avgs[i] += list(c.collab_accs.values())[i]

        for i in range(min_nr_rounds):
            avgs[i] = avgs[i]/len(filtered_clients)

        avg_acc_before_collab = sum([c.independent_acc for c in filtered_clients]) / len(filtered_clients)

        fake_client = ClientStats(101, avg_acc_before_collab, upperBounds[arch], dict(zip(range(len(avgs)), avgs)), arch)

        arch_avg[arch]=fake_client

    return arch_avg

#    for c in range(100):
#        print(f"key:{c}, value: {clients_accuracy_progress[c]}\n")
#    
#    print(len(clients_accuracy_progress))

def plot_results(arch_clients, alpha, withUpperBound=False, describtion=""):
    plt.figure()
    plt.xlabel("nr. of rounds performed")
    plt.ylabel("test accuracy")

    HIGH_X = 20

    HIGH_Y = 0.25 if withUpperBound==False else max([c.upper_bound_acc for c in arch_clients.values()]) + 0.01#0.1

    color = {"/densenet/densenet20": "green",
             "/densenet/densenet10":"red",
             "/resnet20/resnet20_bn": "blue",
             "/resnet20/resnet20_gn": "orange",
             "/shufflenetv2/shufflenetbig": "purple",
             "/shufflenetv2/shufflenetsmall": "gray"}

    plt.ylim(0, HIGH_Y)
    plt.xlim(0, HIGH_X)
    for arch in arch_clients.keys():

        x = [i for i in range(len(arch_clients[arch].collab_accs.keys()) + 1)] # +1 because x=0 will correspond to the accuracy of the client model before collaborative learning
        y = [i for i in arch_clients[arch].collab_accs.values()]
        y.insert(0, arch_clients[arch].independent_acc)

        plt.plot(x, y,'-s', linewidth=1, markersize=2, label=arch.split('/')[2], color=color[arch])

        x_const = [i for i in range(int(HIGH_X/2))]
        plt.plot(x_const, [arch_clients[arch].independent_acc for _ in x_const], '--', color=color[arch])

        x_const = [i for i in range(int(HIGH_X/2), HIGH_X+1)]
        plt.plot(x_const, [arch_clients[arch].upper_bound_acc for _ in x_const], '-.', color=color[arch])

    plt.legend()
    plt.title((f"I.I.D. {describtion}results " if alpha == 1000.0 else f"Non I.I.D {describtion}results ") + ("with upperbound " if withUpperBound else ""))
    plt.show()

if __name__ == "__main__":
    upperBounds = get_upper_bounds()

    alpha = 1000.0

    clients = init_clientStats(upperBounds, alpha)
    d = best_results(clients)
    av = averaged_results(clients, upperBounds)

    plot_results(d, alpha, False, describtion="best ")
    plot_results(d, alpha, True, describtion="best ")
    plot_results(av, alpha, False, describtion="averaged ")
    plot_results(av, alpha, True, describtion="averaged ")

    alpha = 0.0

    clients = init_clientStats(upperBounds, alpha)
    d = best_results(clients)
    av = averaged_results(clients, upperBounds)

    plot_results(d, alpha, False, describtion="best ")
    plot_results(d, alpha, True, describtion="best ")
    plot_results(av, alpha, False, describtion="averaged ")
    plot_results(av, alpha, True, describtion="averaged ")