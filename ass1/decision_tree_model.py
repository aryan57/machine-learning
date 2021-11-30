# This file contains the basic structure of the model
import helper_funcs
idx = 0

class node:

    attr_val = {'buying': ['vhigh', 'high', 'med', 'low'],
                     'maint': ['vhigh', 'high', 'med', 'low'],
                     'doors': ['2', '3', '4', '5more'],
                     'persons': ['2', '4', 'more'],
                     'lug_boot': ['small', 'med', 'big'],
                     'safety': ['low', 'med', 'high']
                     }
    def __init__(self, attr, data, target, is_leaf):
        global idx
        self.idx = idx
        idx += 1
        self.children = {}
        self.attr = attr
        self.data = data
        self.target = target
        self.is_leaf = is_leaf

    def prune_node(self):
      self.is_leaf = True

    def restore(self):
      self.is_leaf = False


def build(data, attr_list, cur_h, max_h, glob_max, attribute_selector):
    n = len(data)
    # print("n ", n)
    if n == 0:
        return node(None, None, glob_max, True)

    # find the target value with most frequency
    target_dict = {'unacc':  [],
                   'acc':    [],
                   'good':   [],
                   'vgood':  []
                   }
    for item in data:
        target_dict[item['target']].append(item)
    majority_target = 'unacc'
    for i in target_dict:
        # print(str(i)+" "+str(len(target_dict[i])))
        # print(str(majority_target) + " " + str(len(target_dict[majority_target])))
        if len(target_dict[i]) > len(target_dict[majority_target]):
            majority_target = i
    # print("")
    if len(attr_list) == 0:
        # print("Noo")
        # print("")
        # print('empty ', majority_target)
        return node(None, None, majority_target, True)
    # return if max height reached or all examples have same target value
    if len(target_dict[majority_target]) == n or cur_h >= max_h:
        return node(None, None, majority_target, True)

    best_attr, best_split = attribute_selector(data, attr_list)
    # print("best attr", best_attr)
    # for i in best_split:
    #     print(i)
    #     print(len(best_split[i]))
    # print("")
    head = node(best_attr, data, majority_target, False)
    for val in head.attr_val[best_attr]:
        new_attr_list = []
        for i in attr_list:
            if i != best_attr:
                new_attr_list.append(i)
        if val in best_split:
          head.children[val] = build(best_split[val], new_attr_list, cur_h+1, max_h, majority_target, attribute_selector)
        else:
          head.children[val] = build([], new_attr_list, cur_h+1, max_h, majority_target, attribute_selector)

    return head

def predict(cur_node, item):
  if cur_node.is_leaf:
    return cur_node.target
  # print(cur_node.attr)
  # print(item[cur_node.attr])
  # print(cur_node.children)
  return predict(cur_node.children[item[cur_node.attr]], item)

def get_accuracy(tree_root, X_test):
  correct = 0
  n = len(X_test)
  for item in X_test:
    pred = predict(tree_root, item)
    # print(str(pred) + " " + item['target'])
    correct += (pred==item['target'])
  
  accuracy = (correct/n)*100
  return accuracy

def count_nodes(cur_node):
  if cur_node.is_leaf:
    return 1
  val = 1
  for i in cur_node.children:
    val += count_nodes(cur_node.children[i])
  return val

def clean_tree(cur_node):
  if cur_node.is_leaf:
    cur_node.children = {}
  else:
      for i in cur_node.children:
        clean_tree(cur_node.children[i])

def prune_tree(root, cur_node, X_test):
  if cur_node.is_leaf:
    return cur_node.idx, get_accuracy(root, X_test)
  
  cur_node.prune_node()
  best_acc = get_accuracy(root, X_test)
  best_id = cur_node.idx
  cur_node.restore()
  # print("cur id ", best_id)
  for i in cur_node.children:
    id, accuracy = prune_tree(root, cur_node.children[i], X_test)
    if accuracy > best_acc:
      best_acc = accuracy
      best_id = id
  
  return id, best_acc

def delete_children(cur_node, id):
  if cur_node.idx == id:
    cur_node.prune_node()
    cur_node.children = {}
    return
  if cur_node.is_leaf:
    return 
  for i in cur_node.children:
    delete_children(cur_node.children[i], id)

def prune(root, X_test):
  prev_acc = get_accuracy(root, X_test)
  while True:
    print("pruning////////////")
    id, accuracy = prune_tree(root, root, X_test)
    if accuracy > prev_acc:
      print("Node id pruned = ", id)
      print("Accuracy improved to = ", accuracy)
      delete_children(root, id)
      prev_acc = accuracy
    else:
      break
  print("No More improvement possible")
    



