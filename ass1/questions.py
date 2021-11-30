import decision_tree_model
import helper_funcs
import matplotlib.pyplot as plt

decision_tree_model.idx = 0

print("Reading data")
dataset = helper_funcs.read_data('car.data')
X_train, X_test = helper_funcs.split_data(dataset)
attr_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

'''
Question 1 : Analyzing the differences in accuarcy when using different impurity measures
'''

print('')
print('Question 1 :')
print('')

majority_target = helper_funcs.get_majority_target(X_train)
# full tree using information gain
root1 = decision_tree_model.build(X_train, attr_list, 1, 8, majority_target, helper_funcs.entropy_best_attr)
# helper_funcs.print_tree(root)
accuracy = decision_tree_model.get_accuracy(root1, X_test)
print("The accuracy on the test set when using entropy as impurity measure is = ", accuracy)

# full tree using gini index
root2 = decision_tree_model.build(X_train, attr_list, 1, 8, majority_target, helper_funcs.gini_best_attr)
# helper_funcs.print_tree(root)
accuracy = decision_tree_model.get_accuracy(root2, X_test)
print("The accuracy on the test set when using gini index as impurity measure is = ", accuracy)

'''
Question 2 : Average accuracy over 10 random splits
'''
print('')
print('Question 2 :')
print('')

best_tree = None
best_accuracy = 0
acc_sum = 0
test_set = None
train_test = None

# making 10 random splits and finding the accuracy over them by using both impurity measure
for i in range(10):
  X_train, X_test = helper_funcs.split_data(dataset)
  majority_target = helper_funcs.get_majority_target(X_train)
  decision_tree_model.idx=0
  root = decision_tree_model.build(X_train, attr_list, 1, 8, majority_target, helper_funcs.entropy_best_attr)
  accuracy = decision_tree_model.get_accuracy(root, X_test)
  acc_sum += accuracy
  # print(str(best_accuracy) + " " + str(accuracy))
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = root
    test_set = X_test
    train_test = X_train
  decision_tree_model.idx=0
  root = decision_tree_model.build(X_train, attr_list, 1, 8, majority_target, helper_funcs.gini_best_attr)
  accuracy = decision_tree_model.get_accuracy(root, X_test)
  acc_sum += accuracy
  # print(str(best_accuracy) + " " + str(accuracy))
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_tree = root
    test_set = X_test
    train_test = X_train

print("The best accuracy over 10 splits is  = ", best_accuracy)
print(decision_tree_model.count_nodes(best_tree))
print("The average accuracy over 10 splits is  = ", acc_sum/20)

'''
Question 3 : Finding best depth of the tree
'''
print('')
print('Question 3 :')
print('')

x_h = [] 
x_cnt = []
acc_h = []
best_h = 0
best_accuracy = 0

tree_to_print = None

for h in range(1, 8):
  x_h.append(h)
  decision_tree_model.idx = 0
  root = decision_tree_model.build(X_train, attr_list, 1, h, majority_target, helper_funcs.entropy_best_attr)
  accuracy = decision_tree_model.get_accuracy(root, X_test)
  if accuracy > best_accuracy:
    best_accuracy = accuracy
    best_h = h
    tree_to_print = root

  x_cnt.append(decision_tree_model.count_nodes(root))
  acc_h.append(accuracy)

print('The depth limit for best accuracy is = ', best_h)

plt.title("Accuracy vs count of nodes in tree")
plt.xlabel("no of nodes")
plt.ylabel("accuracy")
plt.plot(x_cnt, acc_h, color ="red")
plt.savefig('Accuracy_vs_count.png')

plt.figure().clear()

plt.title("Accuracy vs depth of tree")
plt.xlabel("depth")
plt.ylabel("accuracy")
plt.plot(x_h, acc_h, color ="red")
plt.savefig('Accuracy_vs_depth.png')

'''
Question 4 : Pruning the tree for increasing accuracy
'''
print('')
print('Question 4 :')
print('')

print(decision_tree_model.count_nodes(best_tree))
print("previous accuracy = ", decision_tree_model.get_accuracy(best_tree, test_set))
decision_tree_model.prune(best_tree, test_set)

'''
Question 5 : printing the best depth tree
'''
print('')
print('Question 5 :')
print('')

helper_funcs.print_tree(tree_to_print)
print('Best detpth tree saved in dectree.pdf')







