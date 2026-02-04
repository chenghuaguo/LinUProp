"""
Active Learning with NETCONF for Multiclass Classification

This module implements active learning strategies using Network Confidence (NETCONF)
propagation on graphs for multiclass node classification tasks.
"""

import os.path
import numpy as np
import scipy.sparse as sp
import networkx as nx
import random
import pandas as pd
import copy
import multiprocessing
from tqdm import tqdm
from scipy.stats import dirichlet, entropy


def load_data(dataset, train_num, val_num, test_num, batch_size, index):
    """Load dataset from files."""
    data = np.load('./dataset/dataset={}_datadict_train={}_val={}_test={}_index={}_multiclass.npy'.format(
        dataset, train_num, val_num, test_num, index), allow_pickle=True).item()
    
    graph = np.load('./dataset/dataset={}_graph_multiclass.npy'.format(dataset), allow_pickle=True).item()
    labeldict = np.load('./dataset/dataset={}_labeldict_multiclass.npy'.format(dataset), allow_pickle=True).item()

    train_index = data['train_index']
    test_index = data['test_index']
    val_index = data['val_index']
    unlabeled_pool = data['unlabeled_pool']
    label_dict = labeldict

    return train_index, test_index, val_index, label_dict, graph, unlabeled_pool


def NETCONF(G, A, D, k_class, CM, max_iter=10000):
    """
    Run NETCONF (Network Confidence) propagation on the graph.
    
    Args:
        G: NetworkX graph with node attributes
        A: Adjacency matrix
        D: Degree matrix  
        k_class: Number of classes
        CM: Compatibility matrix
        max_iter: Maximum iterations
    """
    E = []
    M = sp.csr_matrix((k_class / (k_class - 1)) * np.maximum((CM - 1.0 / k_class), 0))
    M2 = M.dot(M)
    for node in G.nodes():
        E.append(G.nodes[node]['Dir_para'])
    E = sp.csr_matrix(E)
    B = E
    old_B = copy.deepcopy(B)
    for i in range(max_iter):
        B = E + (A.dot(B).dot(M) - D.dot(B).dot(M2)).dot(sp.linalg.inv(sp.eye(k_class) - M2))
        if i == max_iter - 1:
            print("NETCONF is not converged!")
            break
        diff = abs(B - old_B).sum()
        old_B = copy.deepcopy(B)
        if diff < 1e-4:
            print("NETCONF converged!")
            break
    B = B.toarray()
    for node in G.nodes():
        G.nodes[node]['NETCONF'] = B[node]


def eval_acc(G, test_index, label_dict):
    """Evaluate accuracy on test set."""
    total_num, true_num = 0, 0
    for node in G.nodes():
        if node in test_index:
            total_num += 1
            if np.argmax(G.nodes[node]['NETCONF']) == label_dict[str(node)]:
                true_num += 1
    return true_num / total_num


def eval_val_acc(G, val_index, label_dict):
    """Evaluate accuracy on validation set."""
    total_num, true_num = 0, 0
    for node in G.nodes():
        if node in val_index:
            total_num += 1
            if np.argmax(G.nodes[node]['NETCONF']) == label_dict[str(node)]:
                true_num += 1
    return true_num / total_num


def set_node_labeled(G, index_list, label_dict, prob_correct, k_class):
    """
    Set nodes as labeled with simulated label noise.
    
    Args:
        G: NetworkX graph
        index_list: List of node indices to label
        label_dict: Dictionary of true labels
        prob_correct: Probability of correct label
        k_class: Number of classes
    """
    for index in index_list:
        is_true_label = np.random.uniform() < prob_correct
        if is_true_label:
            G.nodes[index]['Dir_para'][label_dict[str(index)]] += 1
        else:
            choice_list = list(range(k_class))
            choice_list.remove(label_dict[str(index)])
            G.nodes[index]['Dir_para'][random.choice(choice_list)] += 1

        G.nodes[index]['is_labeled'] = True

        Dir_para_ls = G.nodes[index]['Dir_para']
        G.nodes[index]['prior'] = Dir_para_ls / np.sum(Dir_para_ls)
        G.nodes[index]['prior_distribution'] = dirichlet(Dir_para_ls)

        prior_mean = G.nodes[index]['prior_distribution'].mean()
        prior_std = np.sqrt(G.nodes[index]['prior_distribution'].var())

        G.nodes[index]['prior_lower'] = prior_mean - prior_std
        G.nodes[index]['prior_upper'] = prior_mean + prior_std


def LinBoundPropagation(G, H1, H2, k_class, max_iter=10000):
    """
    Linear Bound Propagation for uncertainty quantification.
    
    Args:
        G: NetworkX graph
        H1, H2: Propagation matrices
        k_class: Number of classes
        max_iter: Maximum iterations
    """
    num_node = G.number_of_nodes()

    prior_lower_matrix, prior_upper_matrix = [], []
    for node in G.nodes():
        prior_lower_matrix.append(G.nodes[node]['prior_lower'])
        prior_upper_matrix.append(G.nodes[node]['prior_upper'])

    prior_lower_matrix, prior_upper_matrix = np.array(prior_lower_matrix), np.array(prior_upper_matrix)
    prior_bound = (sp.csr_matrix(prior_upper_matrix - prior_lower_matrix)).reshape(num_node * k_class, 1, order='C')

    belief_bound = prior_bound
    old_belief_bound = copy.deepcopy(belief_bound)
    T_matrix = H1 + H2
    
    for i in range(max_iter):
        belief_bound = prior_bound + T_matrix.dot(belief_bound)
        if i == max_iter - 1:
            print("LinBoundPropagation is not converged!")
            break
        diff = abs(belief_bound - old_belief_bound).sum()
        old_belief_bound = copy.deepcopy(belief_bound)
        if diff < 1e-4:
            print("LinBoundPropagation converged!")
            break
    belief_bound_matrix = belief_bound.reshape(num_node, k_class, order='C').toarray()
    for node in G.nodes():
        G.nodes[node]['Bound'] = belief_bound_matrix[node]


def Contribution(G, H1, H2, k_class, max_iter=10000):
    """
    Calculate contribution of each node to overall uncertainty.
    
    Args:
        G: NetworkX graph
        H1, H2: Propagation matrices
        k_class: Number of classes
        max_iter: Maximum iterations
    """
    num_node = G.number_of_nodes()

    prior_lower_matrix, prior_upper_matrix = [], []
    for node in G.nodes():
        prior_lower_matrix.append(G.nodes[node]['prior_lower'])
        prior_upper_matrix.append(G.nodes[node]['prior_upper'])

    prior_lower_matrix, prior_upper_matrix = np.array(prior_lower_matrix), np.array(prior_upper_matrix)
    prior_bound = (sp.csr_matrix(prior_upper_matrix - prior_lower_matrix)).reshape(num_node * k_class, 1, order='C')
    prior_bound_diag = sp.diags(prior_bound.toarray().reshape(-1))
    prior_bound_diag_inv = sp.diags(1.0 / (prior_bound.toarray().reshape(-1)))
    
    contribution_vec = prior_bound
    old_contribution_vec = copy.deepcopy(contribution_vec)
    T_Matrix = H1 + H2
    for i in range(max_iter):
        contribution_vec = prior_bound + prior_bound_diag.dot(T_Matrix.dot(prior_bound_diag_inv.dot(contribution_vec)))
        if i == max_iter - 1:
            print("Contribution is not converged!")
            break
        diff = abs(contribution_vec - old_contribution_vec).sum()
        old_contribution_vec = copy.deepcopy(contribution_vec)
        if diff < 1e-4:
            print("Contribution converged!")
            break
    contribution_vec = contribution_vec.reshape(num_node, k_class, order='C').todense()
    for node in G.nodes():
        G.nodes[node]['Contribution'] = contribution_vec[node]


def query(G, query_method, unlabeled_pool, label_dict, size, prob_correct, comb_lambda, H1, H2, k_class, A, D, CM):
    """
    Select nodes to query based on the specified active learning strategy.
    
    Args:
        G: NetworkX graph
        query_method: Active learning method
        unlabeled_pool: List of unlabeled nodes
        label_dict: Dictionary of true labels
        size: Number of nodes to query
        prob_correct: Probability of correct label
        comb_lambda: Combination weight for hybrid methods
        H1, H2: Propagation matrices
        k_class: Number of classes
        A: Adjacency matrix
        D: Degree matrix
        CM: Compatibility matrix
    
    Returns:
        List of selected node indices
    """
    if len(unlabeled_pool) > size:
        if query_method == 'random':
            index = list(np.random.choice(unlabeled_pool, size, replace=False))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
            
        elif query_method == 'LC':
            confidence_dict = {}
            node_ls = list(G.nodes())
            random.shuffle(node_ls)
            index = []
            for node in node_ls:
                if node in unlabeled_pool:
                    confidence_dict[str(node)] = np.max(G.nodes[node]['NETCONF'])
            sorted_conf = sorted(confidence_dict.items(), key=lambda x: x[1])
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
            
        elif query_method == 'entropy':
            entropy_dict = {}
            node_ls = list(G.nodes())
            random.shuffle(node_ls)
            index = []
            for node in node_ls:
                if node in unlabeled_pool:
                    entropy_dict[str(node)] = entropy(G.nodes[node]['NETCONF'])
            sorted_conf = sorted(entropy_dict.items(), key=lambda x: x[1], reverse=True)
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
            
        elif query_method == 'belief_bound':
            bound_dict = {}
            index = []
            LinBoundPropagation(G, H1, H2, k_class)
            for node in G.nodes():
                if node in unlabeled_pool:
                    bound = G.nodes[node]['Bound']
                    bound_dict[str(node)] = np.sum(bound)
            sorted_conf = sorted(bound_dict.items(), key=lambda x: x[1], reverse=True)
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
            
        elif query_method == 'belief_contribution':
            contribution_dict = {}
            index = []
            Contribution(G, H1, H2, k_class)
            for node in G.nodes():
                if node in unlabeled_pool:
                    contribution = G.nodes[node]['Contribution']
                    contribution_dict[str(node)] = np.sum(contribution)
            sorted_conf = sorted(contribution_dict.items(), key=lambda x: x[1], reverse=True)
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
            
        elif query_method == 'certainty_scores':
            certainty_score_dict = {}
            index = []
            NETCONF(G, A, D, k_class, CM)
            for node in G.nodes():
                if node in unlabeled_pool:
                    netconf = G.nodes[node]['NETCONF']
                    certainty = np.max(netconf) - np.mean(np.delete(netconf, np.argmax(netconf)))
                    certainty_score_dict[str(node)] = certainty
            sorted_conf = sorted(certainty_score_dict.items(), key=lambda x: x[1])
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index

        elif query_method == 'LC+belief_bound':
            LC_dict = {}
            bound_dict = {}
            final_dict = {}
            node_ls = list(G.nodes())
            random.shuffle(node_ls)
            index = []
            LinBoundPropagation(G, H1, H2, k_class)
            for node in node_ls:
                if node in unlabeled_pool:
                    LC_dict[str(node)] = np.max(G.nodes[node]['NETCONF'])
                    bound_dict[str(node)] = np.sum(G.nodes[node]['Bound'])
            ls_LC_dict, ls_bound_dict = list(LC_dict.values()), list(bound_dict.values())
            min_LC, max_LC = np.min(ls_LC_dict), np.max(ls_LC_dict)
            min_bound, max_bound = np.min(ls_bound_dict), np.max(ls_bound_dict)
            for k, v in LC_dict.items():
                LC_dict[k] = (LC_dict[k] - min_LC) / (max_LC - min_LC)
                bound_dict[k] = (bound_dict[k] - min_bound) / (max_bound - min_bound)
                final_dict[k] =  LC_dict[k] - comb_lambda * bound_dict[k]
            sorted_conf = sorted(final_dict.items(), key=lambda x: x[1])
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
    
        elif query_method == 'LC+belief_contribution':
            LC_dict = {}
            contribution_dict = {}
            final_dict = {}
            node_ls = list(G.nodes())
            random.shuffle(node_ls)
            index = []
            Contribution(G, H1, H2, k_class)
            for node in node_ls:
                if node in unlabeled_pool:
                    LC_dict[str(node)] = np.max(G.nodes[node]['NETCONF'])
                    contribution_dict[str(node)] = np.sum(G.nodes[node]['Contribution'])
            ls_LC_dict, ls_contribution_dict = list(LC_dict.values()), list(contribution_dict.values())
            min_LC, max_LC = np.min(ls_LC_dict), np.max(ls_LC_dict)
            min_contribution, max_contribution = np.min(ls_contribution_dict), np.max(ls_contribution_dict)
            for k, v in LC_dict.items():
                LC_dict[k] = (LC_dict[k] - min_LC) / (max_LC - min_LC)
                contribution_dict[k] = (contribution_dict[k] - min_contribution) / (max_contribution - min_contribution)
                final_dict[k] =  LC_dict[k] - comb_lambda * contribution_dict[k]
            sorted_conf = sorted(final_dict.items(), key=lambda x: x[1])
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index
            
        elif query_method == 'LC+certainty_scores':
            LC_dict = {}
            certainty_score_dict = {}
            final_dict = {}
            node_ls = list(G.nodes())
            random.shuffle(node_ls)
            index = []
            NETCONF(G, A, D, k_class, CM)
            for node in node_ls:
                if node in unlabeled_pool:
                    LC_dict[str(node)] = np.max(G.nodes[node]['NETCONF'])
                    netconf = G.nodes[node]['NETCONF']
                    certainty = np.max(netconf) - np.mean(np.delete(netconf, np.argmax(netconf)))
                    certainty_score_dict[str(node)] = certainty
            ls_LC_dict, ls_certainty_dict = list(LC_dict.values()), list(certainty_score_dict.values())
            min_LC, max_LC = np.min(ls_LC_dict), np.max(ls_LC_dict)
            min_certainty, max_certainty = np.min(ls_certainty_dict), np.max(ls_certainty_dict)
            for k, v in LC_dict.items():
                LC_dict[k] = (LC_dict[k] - min_LC) / (max_LC - min_LC)
                certainty_score_dict[k] = (certainty_score_dict[k] - min_certainty) / (max_certainty - min_certainty)
                final_dict[k] =  LC_dict[k] + comb_lambda * certainty_score_dict[k]
            sorted_conf = sorted(final_dict.items(), key=lambda x: x[1])
            for node, conf in sorted_conf[:size]:
                index.append(int(node))
            set_node_labeled(G, index, label_dict, prob_correct, k_class)
            return index

    else:
        index = copy.deepcopy(unlabeled_pool)
        set_node_labeled(G, index, label_dict, prob_correct, k_class)
        return index


def main_function(index, dataset, prob_correct_ls):
    """
    Main function to run active learning experiments.
    
    Args:
        index: Random seed / experiment index
        dataset: Dataset name
        prob_correct_ls: List of label correctness probabilities
    """
    for prob_correct in prob_correct_ls:
        np.random.seed(index)
        random.seed(index)

        train_num = 2
        class_num = {'Cora': 7, 'Citeseer': 6, 'Pubmed': 3, 'PolBlogs': 2}
        val_num = {'Cora': 500, 'Citeseer': 500, 'Pubmed': 500, 'PolBlogs':250}
        test_num = {'Cora': 1000, 'Citeseer': 1000, 'Pubmed': 1000, 'PolBlogs':500}
        k_class = class_num[dataset]

        batch_size = k_class
        comb_lambda = 1
        epsilon = 1e-3
        query_num = 20

        train_index, test_index, val_index, label_dict, graph, unlabeled_pool = load_data(
            dataset, train_num, val_num[dataset], test_num[dataset], batch_size, index)

        labeled_pool = train_index
        random.shuffle(unlabeled_pool)
        G = nx.Graph(graph)

        # Initialize graph
        for node in G.nodes():
            Dir_para_ls = [1.0] * k_class
            if node in labeled_pool:
                Dir_para_ls = [1.0] * k_class
                Dir_para_ls[label_dict[str(node)]] = 10.0
                G.nodes[node]['is_labeled'] = True
            else:
                G.nodes[node]['is_labeled'] = False

            G.nodes[node]['prior'] = Dir_para_ls / np.sum(Dir_para_ls)
            G.nodes[node]['prior_distribution'] = dirichlet(Dir_para_ls)
            G.nodes[node]['Dir_para'] = Dir_para_ls

            prior_mean = G.nodes[node]['prior_distribution'].mean()
            prior_std = np.sqrt(G.nodes[node]['prior_distribution'].var())

            G.nodes[node]['prior_lower'] = prior_mean - prior_std
            G.nodes[node]['prior_upper'] = prior_mean + prior_std

        CM = np.diag([(1.0 / k_class) + epsilon * (k_class - 1)] * k_class)
        for i in range(k_class):
            for j in range(k_class):
                if i != j:
                    CM[i][j] = (1.0 / k_class) - epsilon
                    
        A = sp.csr_matrix(nx.to_numpy_array(G))
        H = abs(CM - 1.0 / k_class)
        H_2 = np.dot(H, H)
        degree_values = list(dict(G.degree()).values())
        degree_values_diag = np.diag([float(val) for val in degree_values])
        D = sp.csr_matrix(degree_values_diag)

        H1 = sp.kron(A, sp.csr_matrix(H), format='csr')
        H2 = sp.kron(D, sp.csr_matrix(H_2), format='csr')

        NETCONF(G, A, D, k_class, CM)
        LinBoundPropagation(G, H1, H2, k_class)
        acc = eval_acc(G, test_index, label_dict)
        val_acc = eval_val_acc(G, val_index, label_dict)

        auc_data_dict = {'random': [], 'LC': [], 'entropy': [], 'belief_bound': [], 'LC+belief_bound':[], 
                         'certainty_scores':[], 'LC+certainty_scores':[]}
        val_auc_data_dict = {'random': [], 'LC': [], 'entropy': [], 'belief_bound': [], 'LC+belief_bound':[], 
                             'certainty_scores':[], 'LC+certainty_scores':[]}

        for k, v in auc_data_dict.items():
            auc_data_dict[k].append(acc)
        for k, v in val_auc_data_dict.items():
            val_auc_data_dict[k].append(val_acc)

        for query_method in ['random', 'LC', 'entropy', 'belief_bound', 'LC+belief_bound', 
                             'certainty_scores', 'LC+certainty_scores']:
            print("Method:", query_method)
            G_copy = copy.deepcopy(G)
            unlabeled_pool_copy = copy.deepcopy(unlabeled_pool)
            labeled_pool_copy = copy.deepcopy(labeled_pool)

            for i in tqdm(range(query_num)):
                query_index = query(G_copy, query_method, unlabeled_pool_copy, label_dict, batch_size, prob_correct,
                                    comb_lambda, H1, H2, k_class, A, D, CM)
                NETCONF(G_copy, A, D, k_class, CM)
                acc = eval_acc(G_copy, test_index, label_dict)
                val_acc = eval_val_acc(G_copy, val_index, label_dict)
                print(acc, val_acc)
                auc_data_dict[query_method].append(acc)
                val_auc_data_dict[query_method].append(val_acc)

        df_auc = pd.DataFrame(auc_data_dict)
        df_val_auc = pd.DataFrame(val_auc_data_dict)

        output_path_name = '{}_correct_{}_querynum_{}_lambda{}'.format(dataset, prob_correct, query_num, comb_lambda)
        if not os.path.exists('./' + output_path_name):
            os.mkdir('./' + output_path_name)

        df_auc.to_csv('./{}/auc_{}.csv'.format(output_path_name, index))
        df_val_auc.to_csv('./{}/val_auc_{}.csv'.format(output_path_name, index))


if __name__ == '__main__':
    num_partition = 10
    dataset = 'PolBlogs'
    process_ls = []
    
    for index in range(num_partition):
        process = multiprocessing.Process(target=main_function, args=(index, dataset, [1]))
        process.start()
        process_ls.append(process)

    for process in process_ls:
        process.join()
