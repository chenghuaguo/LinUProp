"""
Data preprocessing for multiclass node classification.

This module loads various graph datasets and splits them into train/validation/test sets
for active learning experiments.
"""

import os.path
import numpy as np
import random
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid, PolBlogs


class Data(object):
    """Data loader and processor for graph datasets."""
    
    def __init__(self, dataset, train_num, val_num, test_num, class_num, data_root='./planetoid-master/data/'):
        """
        Initialize data loader.
        
        Args:
            dataset: Dataset name
            train_num: Number of training samples per class
            val_num: Total number of validation samples
            test_num: Total number of test samples
            class_num: Number of classes
            data_root: Root directory for data files
        """
        self.data_root = data_root
        self.dataset = dataset
        self.train_num = train_num
        self.val_num = val_num
        self.test_num = test_num
        self.class_num = class_num
    
    def data_process(self):
        """
        Process dataset and create train/val/test splits.
        
        Returns:
            Tuple of (train_index, test_index, val_index, label_dict, graph, unlabeled_pool)
        """
        if self.dataset in ['Cora', 'Citeseer', 'Pubmed']:
            dataset = Planetoid(root='./planetoid-master/data/', name=self.dataset, split='random', 
                              num_train_per_class=self.train_num, num_val=self.val_num, num_test=self.test_num)
        
        elif self.dataset == 'PolBlogs':
            dataset = PolBlogs(root='./planetoid-master/data/PolBlogs')

            num_nodes = dataset.data.num_nodes
            torch_indices = torch.randperm(num_nodes)

            train_indices = torch_indices[:self.train_num].tolist()
            val_indices = torch_indices[self.train_num:self.train_num+self.val_num].tolist()
            test_indices = torch_indices[self.train_num+self.val_num:self.train_num+self.val_num+self.test_num].tolist()

            dataset.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            dataset.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            dataset.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            dataset.data.train_mask[train_indices] = True
            dataset.data.val_mask[val_indices] = True
            dataset.data.test_mask[test_indices] = True
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Create label dictionary and graph
        graph = torch_geometric.utils.to_scipy_sparse_matrix(dataset.edge_index)
        label_dict = {}

        for index, value in enumerate(dataset[0].y):
            label_dict[str(index)] = int(value)

        re_train_index = dataset.train_mask.nonzero().view(-1).tolist()
        re_val_index = dataset.val_mask.nonzero().view(-1).tolist()
        re_test_index = dataset.test_mask.nonzero().view(-1).tolist()

        label_dict_node_ls = list(range(dataset.data.num_nodes))
        unlabeled_pool = [i for i in label_dict_node_ls if i not in (re_train_index + re_val_index + re_test_index)]

        return re_train_index, re_test_index, re_val_index, label_dict, graph, unlabeled_pool


if __name__ == '__main__':
    """Main script to preprocess datasets and save splits."""
    
    for index in range(10):
        np.random.seed(index)
        random.seed(index)
        torch.manual_seed(index)

        train_num = 2
        class_num = {'Cora': 7, 'Citeseer': 6, 'Pubmed': 3, 'PolBlogs': 2}
        val_num = {'Cora': 500, 'Citeseer': 500, 'Pubmed': 500, 'PolBlogs': 250}
        test_num = {'Cora': 1000, 'Citeseer': 1000, 'Pubmed': 1000, 'PolBlogs': 500}

        # Process datasets
        for dataset in ['Cora', 'Citeseer', 'Pubmed', 'PolBlogs']:
            data = Data(dataset, train_num, val_num[dataset], test_num[dataset], class_num[dataset])
            train_index, test_index, val_index, label_dict, graph, unlabeled_pool = data.data_process()

            data_dict = {
                'train_index': train_index, 
                'test_index': test_index, 
                'val_index': val_index, 
                'unlabeled_pool': unlabeled_pool
            }

            if not os.path.exists('./dataset'):
                os.mkdir('./dataset')

            np.save('./dataset/dataset={}_datadict_train={}_val={}_test={}_index={}_multiclass.npy'.format(
                dataset, train_num, val_num[dataset], test_num[dataset], index), data_dict)
            
            if index == 0:
                np.save('./dataset/dataset={}_graph_multiclass.npy'.format(dataset), graph)
                np.save('./dataset/dataset={}_labeldict_multiclass.npy'.format(dataset), label_dict)

