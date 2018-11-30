__author__ = 'jasonanema'
import datetime
import random
import sys

time1=datetime.datetime.now()

# how to take in files and convert them to an matrix of tuples (immutable)
#open the text file, map to matrix M, separating with tabs spaces (this is why .split() is without argument inside
with open(sys.argv[1],'r') as file:
#with open("balance.scale.train") as file:
#with open("nursery.train") as file:
#with open("led.train") as file:
#with open("synthetic.social.train") as file:
    raw_train_dataset = []
    for line in file:
        raw_train_dataset.append(list(map(str, line.strip().split(' '))))
#close the text file
file.close()

with open(sys.argv[2],'r') as file:
#with open("balance.scale.test") as file:
#with open("nursery.test") as file:
#with open("led.test") as file:
#with open("synthetic.social.test") as file:
    raw_test_dataset = []
    for line in file:
        raw_test_dataset.append(list(map(str, line.strip().split(' '))))
#close the text file
file.close()

#Model parameters (Random Forest)
balance_max_depth = 4
balance_min_node_size = 2
balance_number_random_attributes = 1
balance_number_trees = 200

nursery_max_depth = 8
nursery_min_node_size = 0
nursery_number_random_attributes = 4
nursery_number_trees = 50

led_max_depth = 7
led_min_node_size = 2
led_number_random_attributes = 1
led_number_trees = 100

synthetic_max_depth = 6
synthetic_min_node_size = 10
synthetic_number_random_attributes = 8
synthetic_number_trees = 100



# parameter choices for various datasets
if sys.argv[1] == 'balance.scale.train':
    max_depth = balance_max_depth
    min_node_size = balance_min_node_size
    number_random_attributes = balance_number_random_attributes
    number_trees = balance_number_trees
if sys.argv[1] == 'nursery.train':
    max_depth = nursery_max_depth
    min_node_size = nursery_min_node_size
    number_random_attributes = nursery_number_random_attributes
    number_trees = nursery_number_trees

if sys.argv[1] == 'led.train':
    max_depth = led_max_depth
    min_node_size = led_min_node_size
    number_random_attributes = led_number_random_attributes
    number_trees = led_number_trees

if sys.argv[1] == 'synthetic.social.train':
    max_depth = synthetic_max_depth
    min_node_size = synthetic_min_node_size
    number_random_attributes = synthetic_number_random_attributes
    number_trees = synthetic_number_trees


#cleans (re-formats) raw_dataset to be in the format I use for the algorithms
#returns dataset ,  labels,  indices (indices of the attritubes [1, 2, ..., k]
def clean_data(raw_dataset):
    #create set of indices and labels, split index:value to [index, value]
    temp_indices = []
    temp_labels = []
    for r in xrange(0, len(raw_dataset)):
        if raw_dataset[r][0] not in temp_labels:
            temp_labels.append(raw_dataset[r][0])
        for c in xrange(1,len(raw_dataset[r])):
            raw_dataset[r][c] = raw_dataset[r][c].split(':')
            if raw_dataset[r][c][0] not in temp_indices:
                temp_indices.append(raw_dataset[r][c][0])


    #convert indices and labels to integers and sort labels to be increasing
    #labels always start at one and increase by one in each step
    for k in xrange(0,len(temp_indices)):
        temp_indices[k]=int(temp_indices[k])
    for k in xrange(0, len(temp_labels)):
        temp_labels[k]=int(temp_labels[k])
    temp_labels.sort()


    # Make indices start at 1, and shift all indices of attributes in raw_dataset accordingly
    smallest_index = min(temp_indices)
    for line in raw_dataset:
        line[0]=int(line[0])
        for i in xrange(1,len(line)):
            #line[i]=line[i].split(':')
            line[i][0]=int(line[i][0]) - smallest_index + 1
            line[i][1]=int(line[i][1])


    # create usable dataset with entry 0 label, next are attribute 1's value, attribute 2's value...
    temp_dataset = []
    for line in raw_dataset:
        temp = [0]*(1+len(temp_indices))
        temp[0] = line[0]
        for k in xrange(1, 1 + len(temp_indices)):
            temp[line[k][0]]=line[k][1]
        temp_dataset.append(temp)
    # update temp_indices to start at 1
    for i in xrange(0,len(temp_indices)):
        temp_indices[i] = temp_indices[i]-smallest_index + 1

    return temp_dataset, temp_labels, temp_indices

# clean raw_train_dataset, set this clean data to dataset
# declare global variables dataset (the training one) and its labels, and indices (indices of attributes)
dataset, labels, indices = clean_data(raw_train_dataset)

# clean raw_train_dataset, set this clean data to dataset
# declare global variables dataset (the training one) and its labels, and indices (indices of attributes)
test_dataset, test_labels, test_indices = clean_data(raw_test_dataset)


#takes in a list of pointers to rows in the dataset, and returns the Gini Index for that
# subset of the dataset
def Gini_Index(pointers):
    number_of_labels = len(labels)
    counts_of_labels = [0.0]*number_of_labels
    number_of_datapoints = len(pointers)
    if number_of_datapoints == 0:
        return 1.0
    else:
        for i in xrange(0,number_of_datapoints):
            for k in xrange(0,number_of_labels):
                if dataset[pointers[i]][0] == labels[k]:
                    counts_of_labels[k] += 1
        Gini_I = 1.0
        for k in xrange(0, number_of_labels):
            Gini_I -= (counts_of_labels[k] / number_of_datapoints)**2
        return Gini_I


#Gini Index of an Attribute,
# using pointers, take in pointers to rows of dataset, and attribute to use
# returns list [gini_A(D), attribute [attribute values], [ [ indices of data with attribute values ] ]
def Gini_Index_A(pointers, attribute):
    gini_A = 0.0
    number_data_points = len(pointers)
    #if list of pointers is empty, return [gini_A = 1, [] , [] ]
    if number_data_points == 0:
        return [1.0, [],[]]
    else:
        # place the attribute value in the first row of data in attribute_values
        # initializes partition with first row [0] has first attribute value
        attribute_values = [dataset[pointers[0]][attribute]]
        partition = [[pointers[0]]]
        for i in xrange(1, number_data_points):
            marker = 0
            for j in xrange(0, len(attribute_values)):
                if dataset[pointers[i]][attribute] == attribute_values[j]:
                    partition[j].append(pointers[i])
                    marker = 1
            if marker == 0:
                attribute_values.append(dataset[pointers[i]][attribute])
                partition.append([pointers[i]])
        #given the partition on attribute, calculated gini_A
        for k in xrange(0,len(partition)):
            gini_A +=len(partition[k])*Gini_Index(partition[k])
        gini_A = gini_A/number_data_points
        return [gini_A, attribute, attribute_values, partition]

#returns the indices of maximum of a list of non-negative values (used for majority voting function)
def index_of_maximums(values):
    #sets max just under threshold of 0
    max_value = -0.1
    index_of_maxes = []
    for i in xrange(0,len(values)):
        if values[i] > max_value:
            index_of_maxes = [i]
            max_value = values[i]
        elif values[i] == max_value:
            index_of_maxes.append(i)
    return index_of_maxes
#returns the indices of minimum of a list of values <= 1.0
# (used for choosing minimum of Gini_Index_A, to select splitting attribute)
def index_of_minimums(values):
    #sets min just above threshold of 1.0
    min_value = 1.1
    index_of_mins = []
    for i in xrange(0, len(values)):
        if values[i] < min_value:
            index_of_mins = [i]
            min_value = values[i]
        elif values[i] == min_value:
            index_of_mins.append(i)
    return index_of_mins



# selects attribute to split a node
# input: pointers to subset of dataset, attribute_list
# outputs, output of Gini_Index_A for minimum gini_index_A
# that is [gini_A(D), attribute, [attribute values], [ [ indices of data with attribute values ] ]
def Attribute_selection_method(pointers, attribute_list):
    Gini_Index_Attributes_full = []
    Gini_values = []
    number_of_attributes = len(attribute_list)
    for i in xrange(0, number_of_attributes):
        Gini_Index_Attributes_full.append(Gini_Index_A(pointers, attribute_list[i]))
        Gini_values.append(Gini_Index_Attributes_full[i][0])
    indices_of_max_ginis = index_of_minimums(Gini_values)
    if len(indices_of_max_ginis) == 1:
        return Gini_Index_Attributes_full[indices_of_max_ginis[0]]
    else:
        random_splitting = random.randrange(0,len(indices_of_max_ginis))
        return Gini_Index_Attributes_full[indices_of_max_ginis[random_splitting]]

# input pointers to subset of dataset
# output majority wins label, if there are ties, broken uniform randomly
def majority_wins_label(pointers):
    count_of_labels = [0]*len(labels)
    for i in xrange(0, len(pointers)):
        for j in xrange(0,len(labels)):
            if dataset[pointers[i]][0] == labels[j]:
                count_of_labels[j] += 1
    temp = index_of_maximums(count_of_labels)
    if len(temp) == 1:
        return labels[temp[0]]
    elif len(temp) == 0:
        random_voter = random.randrange(0,len(labels))
        return labels[random_voter]
    else:
        random_voter = random.randrange(0,len(temp))
        return labels[temp[random_voter]]


#input pointers to dataset
#output True, if all in the same class
def all_same_label(pointers):

    for i in xrange(0,len(pointers)-1):
        if dataset[pointers[i]][0] != dataset[pointers[i+1]][0]:
            return False
    return True

#input remaining attributes at node
#output random subset of attribute indices of size number_random_attributes (if there are more attributes than this)
def random_attributes_for_split(possible_attributes):
    if len(possible_attributes) <= number_random_attributes:
        return possible_attributes
    else:
        return random.sample(possible_attributes,number_random_attributes)

#generates decision tree
#inputs: D = pointers to dataset, attribute_list
def Generate_decision_tree(D, attribute_list, depth):
    #create node N
    N = []
    #if maz_depth is exceeded, return N as a leaf node labeled with majority wins label of D
    if depth > max_depth:
        N= [0, majority_wins_label(D)]
        return N

    #if all tuples in D are of same  C, return N as a leaf node labelled w\ label C
    if all_same_label(D):
        N = [0, dataset[D[0]][0]]
        return N

    # if attribute_list is empty, return leaf node labeled with majority wins label of D
    if attribute_list == []:
        N = [0, majority_wins_label(D)]
        return N
    splitting_att_full = Attribute_selection_method(D,random_attributes_for_split(attribute_list))

    #label N with best splitting criterion
    N = [splitting_att_full[1], splitting_att_full[2]]

    #remove splitting attribute from attribute_list
    attribute_list = [x for x in attribute_list if x != splitting_att_full[1]]

    #for each outcome j of splitting criterion, if D_j (partition with split att value = j)
    for i in xrange(0, len(splitting_att_full[2])):
        #if number of data points in D_j is <=min_node_size, use majority wins
        if len(splitting_att_full[3][i]) <= min_node_size:
            N.append([0,majority_wins_label(D)])

        else:

            N.append(Generate_decision_tree(splitting_att_full[3][i],attribute_list, depth +1))

    return N
### returns mode(s) of an array, copied from my homework 1 problem 1, CS 412 fall 2017 (this class)
def compute_mode(numbers):
    counts = {}
    maxcount = 0
    modes = []
    for number in numbers:
        if number not in counts:
            counts[number] = 0
        counts[number] += 1
        if counts[number] > maxcount:
            maxcount = counts[number]
    for number, count in counts.items():
        if count == maxcount:
            modes.append(number)
            #print(number, count)
    return modes

#################

#   STOPPED HERE ON WEDNESDAY, DEC 6, at 1:20am
#     double check that this predict function works

## prediction function
#input: node (corresponding to where in D_T one is),
#  and row of data [2,3,4,5,3,2] (entry of dataset, e.g.)
#output: predicted label

def predict(node, row):
    if node[0] == 0:
        return node[1]
    else:
        #if attribute node[0]'s value has was not seen in test data (is not in node[1])
        #randomly use one of node[0]'s attribute values to continue with prediction
        value_seen = False
        for i in xrange(0,len(node[1])):
            if row[node[0]]==node[1][i]:
                value_seen = True
                new_node = node[i+2]
                return predict(new_node, row)
        if value_seen == False:
            #assign a random value in node[0]'s attribute values to make prediction with
            k = random.randrange(0, len(node[1]))
            new_node = node[k+2]
            return predict(new_node, row)

#create random_databases, each row is a random bagging of pointers to dataset
# number of rows is number_trees
random_databases = []
len_dataset = len(dataset)
for k in xrange(0, number_trees):
    temp_dataset = []
    for j in xrange(0,len_dataset):
        temp_dataset.append(random.randint(0,len_dataset-1))
    random_databases.append(temp_dataset)

### Make a random_forest[i] being trained on random_databases[i]
#
###
random_forest = []
for i in xrange(0, number_trees):
    random_forest.append(Generate_decision_tree(random_databases[i],indices,1))

### Forest prediction function
# uses global random_forest
#input a row to do prediction on
def forest_predict(row):
    predictions = []
    for i in xrange(0, number_trees):
        predictions.append(predict(random_forest[i], row))
    most_occuring_predictions = compute_mode(predictions)
    return random.choice(most_occuring_predictions)



## make decision tree
#D_T = Generate_decision_tree(list(range(len(dataset))),indices,1)
#print "D_T is:", D_T

def build_confusion_matrix():
    conf_mat = []
    for i in xrange(0,len(labels)):
        conf_mat.append([0]*len(labels))
    #for each data row in test_dataset
    for row in xrange(0,len(test_dataset)):
        actual_label = test_dataset[row][0]
        predicted_label = forest_predict(test_dataset[row])
        conf_mat[actual_label-1][predicted_label-1] += 1
    return conf_mat

def build_confusion_matrix_train():
    conf_mat = []
    for i in xrange(0,len(labels)):
        conf_mat.append([0]*len(labels))
    #for each data row in dataset
    for row in xrange(0,len(dataset)):
        actual_label = dataset[row][0]
        predicted_label = forest_predict(dataset[row])
        conf_mat[actual_label-1][predicted_label-1] += 1
    return conf_mat


# computing model evaluation metrics.
# MOVE THIS OUT TO A SEPARATE PROGRAM LATER?
# Note 1: To evaluate the performance on each class,
#   we regard the target class as positive and all others as negative.

#accuracy of a model, input the confusion matrix (its the percentage of tuples labelled correctly)
def total_accuracy(matrix):
    diagonal_sum = 0.0
    sum_of_entries = 0.0
    for row in xrange(0, len(matrix)):
        diagonal_sum += matrix[row][row]
        for col in xrange(0, len(matrix[row])):
            sum_of_entries += matrix[row][col]
    accuracy = diagonal_sum/sum_of_entries
    return accuracy

#build confusion matrix for testing data
confusion_mat = build_confusion_matrix()
#build confusion matrix for training data
confusion_mat_train = build_confusion_matrix_train()
#print "confusion matrix testing:"
for r in range(0,len(confusion_mat)):
    print ' '.join(str(x) for x in confusion_mat[r])
#print "total accuracy on testing: ", total_accuracy(confusion_mat)
#for i in xrange(1,len(indices)+1):
#    print i, Gini_Index_A(list(range(len(dataset))), i)[0]
#print "D_T:", D_T

#inputs complete confusion matrix, and index i of interest (row of label i +1)
# outputs two by two, treating row i as positive and all others as negative
def conf_mat_two_by_two(matrix, i):
    size = len(matrix[0])
    small_c_m = [[0,0], [0,0]]
    TP = matrix[i][i]
    FN = 0
    for j in xrange(0,size):
        FN += matrix[i][j]
    FN -= TP
    FP = 0
    for j in xrange(0,size):
        FP += matrix[j][i]
    FP -= TP
    TN = 0
    for j in xrange(0,size):
        for k in xrange(0,size):
            TN += matrix[j][k]
    TN = TN - TP - FN - FP
    small_c_m[0][0] = TP
    small_c_m[0][1] = FN
    small_c_m[1][0] = FP
    small_c_m[1][1] = TN
    return small_c_m
#print conf_mat_two_by_two(confusion_mat, 2)[0]
#print conf_mat_two_by_two(confusion_mat, 2)[1]

#takes in the 2 by 2 confusion matrix, outputs class accuracy
def class_accuracy(matrix):
    num = matrix[0][0] + matrix[1][1]
    den = num + matrix[1][0] + matrix[0][1]
    if den == 0:
        return "div by zero"
    else:
        return round(num*1.0/den,3)
#takes in the 2 by 2 confusion matrix, outputs class specificity
def class_specificity(matrix):
    num = matrix[1][1]
    den = matrix[1][0] + matrix[1][1]
    if den == 0:
        return "div by zero"
    else:
        return round(num*1.0/den,3)
#takes in the 2 by 2 confusion matrix, outputs class precision
def class_precision(matrix):
    num = matrix[0][0]
    den = matrix[0][0] + matrix[1][0]
    if den == 0:
        return "div by zero"
    else:
        return round(num*1.0/den,3)
#takes in the 2 by 2 confusion matrix, outputs class recall
def class_recall(matrix):
    num = matrix[0][0]
    den = matrix[0][0] + matrix[0][1]
    if den == 0:
        return "div by zero"
    else:
        return round(num*1.0/den,3)
#takes in the 2 by 2 confusion matrix, outputs class F one score
def class_F_one(matrix):
    precision = class_precision(matrix)
    recall = class_recall(matrix)
    if precision == "div by zero" or recall == "div by zero":
        return "div by zero"
    elif precision + recall == 0:
        return "div by zero"
    else:
        return round(1.0*2*precision*recall/(precision + recall),3)

#takes in the 2 by 2 confusion matrix, outputs class F half score
def class_F_half(matrix):
    precision = class_precision(matrix)
    recall = class_recall(matrix)
    if precision == "div by zero" or recall == "div by zero":
        return "div by zero"
    elif .25*precision + recall == 0:
        return "div by zero"
    else:
        return round(1.0*(1+.25)*precision*recall/((.25)*precision + recall),3)
#takes in the 2 by 2 confusion matrix, outputs class F two score
def class_F_two(matrix):
    precision = class_precision(matrix)
    recall = class_recall(matrix)
    if precision == "div by zero" or recall == "div by zero":
        return "div by zero"
    elif 4.0*precision + recall == 0:
        return "div by zero"
    else:
        return round(1.0*(1+4.0)*precision*recall/((4.0)*precision + recall),3)

# printing by class the 7 measures for testing dataset
#print "by class measures for test"
#for i in xrange(0, len(confusion_mat[0])):
#    small_cm = conf_mat_two_by_two(confusion_mat, i)
#    print "measures for class ", i+1, "are:"
#    #this line if for latexing
#    print "&", class_accuracy(small_cm), "&", class_specificity(small_cm), "&", class_precision(small_cm), "&", class_recall(small_cm), "&", class_F_one(small_cm), "&",class_F_half(small_cm), "&", class_F_two(small_cm)
#    print "class accuracy:", class_accuracy(small_cm)
#    print "class specificity:", class_specificity(small_cm)
#    print "class precision:", class_precision(small_cm)
#    print "class recall:", class_recall(small_cm)
#    print "class F one:", class_F_one(small_cm)
#    print "class F half:", class_F_half(small_cm)
#    print "class F two:", class_F_two(small_cm)

# printing by class the 7 measures for training dataset
#print '\n', "by class measures for training"
#print "total accuracy on training set: ", total_accuracy(confusion_mat_train)
#for i in xrange(0, len(confusion_mat_train[0])):
#    small_cm = conf_mat_two_by_two(confusion_mat_train, i)
#    print "measures for class ", i+1, "are:"
#    #this line is for latexing
#    print "&", class_accuracy(small_cm), "&", class_specificity(small_cm), "&", class_precision(small_cm), "&", class_recall(small_cm), "&", class_F_one(small_cm), "&",class_F_half(small_cm), "&", class_F_two(small_cm)
#    print "class accuracy:", class_accuracy(small_cm)
#    print "class specificity:", class_specificity(small_cm)
#    print "class precision:", class_precision(small_cm)
#    print "class recall:", class_recall(small_cm)
#    print "class F one:", class_F_one(small_cm)
#    print "class F half:", class_F_half(small_cm)
#   print "class F two:", class_F_two(small_cm)

time2=datetime.datetime.now()
#print runtime
#print '\n' , "Runtime: ", time2-time1

