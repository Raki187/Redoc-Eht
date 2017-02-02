import operator
import sys
import math
import random
import itertools
prunNodes = []
count = 0
class Node(object):
    val = ""
    subnode0 = None
    subnode1 = None
    attributes = []
    
    def __init__(self, training_dataset):
        self.training_dataset = training_dataset
        self.classvalues = [item.get(target_attribute) for item in  training_dataset]
        if(len(self.classvalues) > 1):
            self.getClassValues()
            
    def getValues(self, attribute_name, value):
        values = []
        for each in self.training_dataset:
            if each.get(attribute_name) == value:
                values.append(each)
        return values
    
    def getClassValues(self):
        class_values_1 = self.classvalues.count(1)
        zero_class_values =  self.classvalues.count(0)
        no_of_class_values = float(len(self.classvalues))
        no_of_1 = class_values_1 / no_of_class_values
        no_of_0 =  zero_class_values / no_of_class_values
        self.entropy = calculate_entropy(no_of_1, no_of_0)
    
    def decisionTree(self, count):
        if self.subnode0 != None:
            s = "| "*(count + 1)
            print s,
            if(self.subnode0.subnode0!=None):
                print (self.val), "=", "0", ":"
            else:
                print (self.val), "=", "0", ":",
            self.subnode0.decisionTree(count + 1)
        else: 
            print self.val
        if self.subnode1 != None:
            s = "| "*(count + 1)
            print s,
            if(self.subnode1.subnode1!=None):
                print (self.val), "=", "1", ":"
            else:
                print (self.val), "=", "1", ":",
            self.subnode1.decisionTree(count + 1)     

    def best_attribute(self): 
        information_gain = {}
        for each in self.attributes:
            if not each == target_attribute:
                attri_value_1 = self.getValues(each,1)
                attri_value_0 = self.getValues(each,0)
                classEntGivenAttr = attributeEntropy(each,attri_value_1,attri_value_0)
                if(self.entropy - classEntGivenAttr) > 0:
                    information_gain[each] = self.entropy - attributeEntropy(each, attri_value_1 , attri_value_0)
        if len(information_gain)>=1:
            sorted_best_attribute_set = sorted(information_gain.items(), key=operator.itemgetter(1), reverse=True)
            return sorted_best_attribute_set[0][0]
        return ''
    
    def pruning(self,pruning_factor,train):
        trainset = from_file(train)
        prunedTree = copyTree(self) 
        for i in range(pruning_factor):
            global prunNodes
            prunNodes = []
            getLeaves(prunedTree)
            n = len(prunNodes)
            if(n>1):
                    p = random.randint(1, n-1)
                    chosenNode = prunNodes[p]
                    del prunNodes[p]
                    class0 = chosenNode.classvalues.count(0)
                    class1 = chosenNode.classvalues.count(1)
                    chosenNode.subnode0 = None
                    chosenNode.subnode1 = None
                    chosenNode.val= "1" if class0 > class1 else "0"
        pruningAccuracy = findAccuracy(prunedTree, trainset)
        return prunedTree  

def calculate_entropy(no_of_1, no_of_0):
    if no_of_1 != 0 and no_of_0 != 0:
         return -no_of_1 * ((math.log( no_of_1 )) / math.log( 2 )) + (-no_of_0 * (math.log( no_of_0 )) / math.log( 2 ))
    elif no_of_1 == 0:
            return -no_of_0 * (math.log( no_of_0 )) / math.log( 2 )
    else:
        return -no_of_1 * ((math.log( no_of_1 )) / math.log( 2 ))

def attributeEntropy(headerName, attrValuesPos, attrValuesNeg):
        sizePos = float(len(attrValuesPos))
        sizeNeg = float(len(attrValuesNeg))
        posInstance1 = 0
        posInstance0 = 0
        negInstance1 = 0
        negInstance0 = 0
        for item in attrValuesPos:
            if item.get(headerName) == 1 and item.get(target_attribute) == 1:
                posInstance1 = posInstance1 + 1
            else:
                negInstance1 = negInstance1 + 1
        for item in attrValuesNeg:
            if item.get(headerName) == 0 and item.get(target_attribute) == 1:
                posInstance0 = posInstance0 + 1
            else:
                negInstance0 = negInstance0 + 1

        totalDataSize = sizePos + sizeNeg         
        posEntropy = sizePos / (sizePos + sizeNeg) * calculate_entropy(posInstance1 / sizePos, negInstance1 / sizePos)
        negEntropy = sizeNeg / (sizePos + sizeNeg) * calculate_entropy(posInstance0 / sizeNeg, negInstance0 / sizeNeg)
        return (posEntropy + negEntropy)
    
def copyTree(rootNode):
    if(rootNode != None):
        node = Node(rootNode.training_dataset)
        node.val= rootNode.val
        node.subnode0 = copyTree(rootNode.subnode0)
        node.subnode1 = copyTree(rootNode.subnode1)
        return node
    else: 
        return None

def getNoNodes(rootNode):
    noNodes=1
    if rootNode.subnode0 != None:
        noNodes+=getNoNodes(rootNode.subnode0)
    if rootNode.subnode1 != None:
        noNodes+=getNoNodes(rootNode.subnode1)
    return noNodes

def getNoleaves(rootNode):
    if rootNode == None:
        return 0
    if rootNode.subnode0 == None and rootNode.subnode1 == None:
        return 1
    else:
        return getNoleaves(rootNode.subnode0)+getNoleaves(rootNode.subnode1)

def getLeaves(rootNode):
    global prunNodes
    if(rootNode.subnode0 is not None and rootNode.subnode0.subnode0== None):
        prunNodes.append(rootNode)
    elif(rootNode.subnode0!=None and rootNode.subnode1!=None):
        getLeaves(rootNode.subnode0)
        getLeaves(rootNode.subnode1)
    return  
        
def findAccuracy(rootNode, vectorSet):
    global count 
    count = 0
    for d in vectorSet:
        calAccuracy(rootNode, d)
    return count*1.0/len(vectorSet)

def calAccuracy(rootNode, tupledict):
    global count
    if(rootNode.subnode0 == None and rootNode.subnode1 == None):
        if(tupledict.get(target_attribute)==0 and rootNode.val== "0"):
            count = count + 1
        elif(tupledict.get(target_attribute)==1 and rootNode.val== "1"):
            count = count + 1
        return rootNode.val
    if(tupledict[rootNode.val]==0):
        calAccuracy(rootNode.subnode0, tupledict)
    else:
        calAccuracy(rootNode.subnode1, tupledict)

def decisionTreeAlgorithm(attri,best_attribute,edited_dataset): 
    node = Node(edited_dataset)
    node.attributes = list(attri)
    if(len(set(node.classvalues)) == 1):
        n = Node([])
        n.val= "1" if set(node.classvalues).pop()==1 else "0"
        return n
    best_attribute_set = node.best_attribute()
    if(len(attri) == 0 or len(best_attribute_set) < 1 or len(edited_dataset) < 1):
        class0 = [item.get(target_attribute) for item in edited_dataset].count(0)
        class1 = [item.get(target_attribute) for item in edited_dataset].count(1)
        n = Node([])
        n.val= "1" if class1 > class0 else "0"
        return n
    if(len(best_attribute_set) > 1):
        node.val= best_attribute_set
        node.attributes.remove(best_attribute_set)
        node.subnode0 = decisionTreeAlgorithm(node.attributes, best_attribute_set, node.getValues(best_attribute_set, 0))
        node.subnode1 = decisionTreeAlgorithm(node.attributes, best_attribute_set, node.getValues(best_attribute_set, 1))
    return node

def from_file(file_name):
    f = open(file_name)
    global attributes
    global target_attribute
    attributes = f.readline().strip().split("\t")
    target_attribute=attributes[-1]
    lines = f.readlines()
    f.close()
    training_dataset = []
    for line in lines:
        values = map(int, line.strip().split("\t"))
        train_data = dict(zip(attributes,tuple(values)))
        training_dataset.append(train_data)
    return training_dataset

def getNoofInstances(path):
    f = open(path)
    return len(f.readlines())-1

def getNoofAttributes(path):
    f = open(path)
    return len(f.readline().strip().split('\t'))-1
        
def main(training_dataset_path,test_dataset_path,pruning_factor):
    training_dataset = from_file(training_dataset_path)
    root = decisionTreeAlgorithm(list(attributes),None,training_dataset)
    noNodes=getNoNodes(root)
    pruning_factor=pruning_factor*noNodes
    pruning_factor=int(pruning_factor)
    test_dataset = from_file(test_dataset_path)
    accuracyTrain = findAccuracy(root,training_dataset)
    accuracyTest = findAccuracy(root, test_dataset)
    getNoofInstances(training_dataset_path)
    print "Pre-Pruned Accuracy"
    print "-----------------------------"
    print "Number of training instances  = ",getNoofInstances(training_dataset_path)
    print "Number of training attributes = ",getNoofAttributes(training_dataset_path)
    print "Total number of nodes is tree = ",noNodes
    print "Number of leaf nodes in the tree = ",getNoleaves(root)
    print "Accuracy of the model on the training dataset = ",accuracyTrain*100,"%"
    print "\n"
    print "Number of testing instances = ",getNoofInstances(test_dataset_path)
    print "Number of testing attributes = ",getNoofAttributes(test_dataset_path)
    print "Accuracy of the model in the testing dataset = ",accuracyTest*100,"%"
    filename = raw_input("\n\n Print Tree(Y/N)")
    if filename == 'y':
        root.decisionTree(0)
    prunedTree = root.pruning(pruning_factor,training_dataset_path)
    noNodes=getNoNodes(prunedTree)
    accuracyTrain = findAccuracy(prunedTree,training_dataset)
    accuracyTest = findAccuracy(prunedTree,test_dataset)
    print "\n\n\nPost-Pruned Accuracy"
    print "-----------------------------"
    print "Number of training instances  = ",getNoofInstances(training_dataset_path)
    print "Number of training attributes = ",getNoofAttributes(training_dataset_path)
    print "Total number of nodes is tree = ",noNodes
    print "Number of leaf nodes in the tree = ",getNoleaves(prunedTree)
    print "Accuracy of the model on the training dataset = ",accuracyTrain*100,"%"
    print "\n\n\n"
    print "Number of testing instances = ",getNoofInstances(test_dataset_path)
    print "Number of testing attributes = ",getNoofAttributes(test_dataset_path)
    print "Accuracy of the model in the testing dataset = ",accuracyTest*100,"%"
    filename = raw_input("\n\n Print Pruned Tree(Y/N)")
    if filename == 'y':
        prunedTree.decisionTree(0)
    
training_dataset_path = sys.argv[1]
test_dataset_path = sys.argv[2]
pruning_factor = float(sys.argv[3]) 
main(training_dataset_path,test_dataset_path,pruning_factor)
