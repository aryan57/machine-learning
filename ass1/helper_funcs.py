# This file contains all the helper/utility functions needed for the implementation
import random
import math
from graphviz import Digraph
import decision_tree_model


def read_data(filename):
    dataset = []
    with open(filename, "r") as index:
        for line in index:
            line = line.rstrip('\n')
            att = line.split(',')
            dataset.append({
                'buying': att[0],
                'maint': att[1],
                'doors': att[2],
                'persons': att[3],
                'lug_boot': att[4],
                'safety': att[5],
                'target': att[6]
            })
    return dataset


def split_data(data):
    random.shuffle(data)
    X_train, X_test = data[0: int(0.8*len(data))], data[ int(0.8*len(data)):]
    return X_train, X_test

def get_majority_target(X_train):
  target_dict = {'unacc': [],
                'acc': [],
                'good': [],
                'vgood': []
                }
  for item in X_train:
      target_dict[item['target']].append(item)
  majority_target = 'unacc'
  for i in target_dict:
      if len(target_dict[i]) > len(target_dict[majority_target]):
          majority_target = i
  return majority_target

def entropy(data):
    n = len(data)
    if n == 0:
        return 0
    freq = {}
    for te in data:
        # print(te['target'])
        if te['target'] in freq:
            freq[te['target']] += 1
        else:
            freq[te['target']] = 1
    val = 0
    for i in freq:
        # print(i)
        frac = freq[i]/n
        val += (-frac*math.log2(frac))
    return val


def gini_index(data):
    n = len(data)
    if n == 0:
        return 0
    freq = {}
    for te in data:
        if te['target'] in freq:
            freq[te['target']] += 1
        else:
            freq[te['target']] = 1
    val = 0
    for i in freq:
        frac = freq[i]/n
        val += frac*(1-frac)
    return val

def entropy_best_attr(data, attr_list):
    gain = -1
    initial_val = entropy(data)
    final_split = {}
    final_attr = None
    n = len(data)
    for i in attr_list:
        temp_split = {}
        for j in data:
            if j[i] in temp_split:
                temp_split[j[i]].append(j)
            else:
                temp_split[j[i]] = [j]
        temp_val = 0
        for j in temp_split:
            temp_val += (len(temp_split[j])/n)*entropy(temp_split[j])
        if initial_val-temp_val > gain:
            gain = initial_val-temp_val
            final_attr = i
            final_split = temp_split
    return final_attr, final_split


def gini_best_attr(data, attr_list):
    gain = -1
    initial_val = gini_index(data)
    final_split = {}
    final_attr = None
    n = len(data)
    for i in attr_list:
        temp_split = {}
        for j in data:
            if j[i] in temp_split:
                temp_split[j[i]].append(j)
            else:
                temp_split[j[i]] = [j]
        temp_val = 0
        for j in temp_split:
            temp_val += (len(temp_split[j])/n)*gini_index(temp_split[j])
        if initial_val-temp_val > gain:
            gain = initial_val-temp_val
            final_attr = i
            final_split = temp_split
    return final_attr, final_split

def get(cur_node):
    if cur_node.is_leaf:
        return "id = {}\n target val = {}".format(cur_node.idx, cur_node.target)
    return "id = {}\n  Attribute = {}".format(cur_node.idx, cur_node.attr)

def print_tree(head):
    f = Digraph('Decision Tree', filename='decision_tree.gv')
    f.attr(rankdir='LR', size='1000,500')

    # border of the nodes is set to rectangle shape
    f.attr('node', shape='rectangle')

    # Do a breadth first search and add all the edges
    # in the output graph
    q = [head]  # queue for the bradth first search
    while len(q) > 0:
        cur_node = q.pop(0)
        # if node.left != None:
        #     f.edge(get(node), get(node.left), label='True')
        # if node.right != None:
        #     f.edge(get(node), get(node.right), label='False')
        #
        # if node.left != None:
        #     q.append(node.left)
        # if node.right != None:
        #     q.append(node.right)
        for i in cur_node.children:
            f.edge(get(cur_node), get(cur_node.children[i]), label=i)
            q.append(cur_node.children[i])

    # save file name :  decision_tree.gv.pdf
    f.render('dectree', view=True)

  


# dataset = read_data("car.data")
# X_train, X_test = split_data(dataset)
# print(len(X_train))
# print(len(X_test))
# print(entropy(X_train[0:2]))
# print(entropy(X_test))
# print(X_train[0:2])
# print(gini_index(X_train[0:2]))
# print(entropy(X_train[0:2]))

