import json
import os
from collections import defaultdict

BATCH_SIZE = 128
NUM_WORKERS = 4

#cdata["user_data"] -> is a dictionary having as key the client id and as value another dictionary.

#	this other dictionary has always 2 entries:	-1st one is of the form "x": [img1, img2, ....]
#							- 2nd one is of form "y": [label1, label2, ...]

def read_dir(data_dir, alpha=None):
    clients = []
    groups = []
    data = defaultdict(lambda : None)

    files = os.listdir(data_dir)
    if alpha is None:
        files = [f for f in files if f.endswith('.json')]
    else:
        alpha = 'alpha_{:.2f}'.format(alpha)
        files = [f for f in files if f.endswith(alpha + '.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data_splits(train_data_dir, test_data_dir, alpha=None):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir, alpha)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    # the test dataset in the .json file is assigned to client 100
    # here this test dataset is passed to all clients to be used for accuracy

    return train_clients, train_data, test_data["100"]

#def create_clients(users, train_data, test_data, models):
#    clients = []
#    for u in users:
#        c_traindata = ClientPrivateDataset(train_data[u], train=True)
#        c_testdata = ClientPrivateDataset(test_data, train=False)   # the same test dataset for all clients
#
#        train_dataloader = DataLoader(c_traindata, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
#        test_dataloader = DataLoader(c_testdata, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
#    
#    clients.append(Client())
#    return clients
#
#def setup_clients(args, model, Client, ClientDataset, run=None, device=None,):
#    """Instantiates clients based on given train and test data directories.
#
#    Return:
#        all_clients: list of Client objects.
#    """
#    train_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'train')
#    test_data_dir = os.path.join('..', 'data', args.dataset, 'data', 'test')
#
#    train_users, train_groups, test_users, test_groups, train_data, test_data = read_data(train_data_dir, test_data_dir, args.alpha)
#
#    train_clients = create_clients(train_users, train_data, test_data, model, args, ClientDataset, Client, run, device)
#    test_clients = create_clients(test_users, train_data, test_data, model, args, ClientDataset, Client, run, device)
#
#    return train_clients, test_clients

