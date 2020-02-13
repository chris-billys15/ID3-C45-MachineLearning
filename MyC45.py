import pandas as pd
import operator as op
import numpy as np
import math

threshold_dict = dict()

class Node:
    def __init__(self,
                _parent = None, # A Node
                _children = [], # list of NODES
                _infoGain = None, # double?
                _attribute = None, # string?
                _valuesTaken = {}): # rules chosen (decision made in each nodes of the parent)
        self.parent = _parent
        self.children = _children
        self.infoGain = _infoGain
        self.attribute = _attribute
        self.valuesTaken = _valuesTaken

    def addChild(self, _node = None):
        if(_node == None):
            _node = self.__init__()
        self.children.append(_node)
        return _node

    def isLeaf(self):
        return not self.children

    def _printTree(self, space):
        if(self.isLeaf()):
            print(str(space*' ') + '<' +str(self.valuesTaken[self.parent.attribute]) + '>')
            print(str( (4+space)*' ') + '(' +str(self.valuesTaken[self.attribute]) + ')')
            return
        else:
            print(str(space*' ') + '<' +str(self.valuesTaken[self.parent.attribute]) + '>')
            print(str((4+space)*' ') + str(self.attribute))
            for child in self.children:
                #print(self.children)
                child._printTree(space+8)

class MyTree:
    """A custom decision tree.

    Parameters
    ----------
    root : lambda function (returns bool), optional (default=x: True)
        the rule to enter this node (example : rule = x: x.outlook == 'sunny').
    targetAttribute : string, optional (default=None)
        label of the node, applies only to leaf nodes.
    node : array, optional (default=[])
        children nodes of tree, applies only to non-leaf nodes.

    Attributes
    ----------
    root =
    targetAttribute = attribute hasil prediksi
    """
    def __init__(self,
                _root = Node(),
                _targetAttribute = None):
        self.root = _root
        self.targetAttribute = _targetAttribute

    def entropyData(self, data):
        """
        data = dataframes yang sudah difilter sesuai kebutuhan
        """
        valueSet = self.getValuesInAttribute(data, self.targetAttribute)
        valueMap = dict.fromkeys(valueSet, 0)
        instances = len(data)

        for value in data.loc[:,self.targetAttribute]:
            valueMap[value]+=1

        entropy = 0
        for value in valueSet:
            entropy += -valueMap[value]/instances * math.log(valueMap[value]/instances,2)
        # print(data)
        # print("entropy : ", entropy)
        return entropy

    def filterDataFrame(self, data, attr, value):
        """ Filter data based on attr and value parameters

        Parameters
        ----------
        data = dataframe yang ingin difilter
        attr = Atribut filter
        value = value dari attribute filter yang ingin
                diaplikasikan ke dataframe

        Returns
        ----------
        filteredData = data yang sudah difilter
        """

        # Inisialisasi dataframe temporer 'filteredData' dengan dataframe 'data'
        filteredData = data

        # Filter data dengan melakukan drop row
        # Hanya row dengan nilai 'value' pada atribut 'attr' saja yang tetap ada di dataframe
        filteredData = filteredData[filteredData[attr] == value]

        return filteredData

    def informationGain(self, data, attr):
        """
        data = data yang sudah terfilter
        attr = atribut yang ingin dicari information gainnya
        """
        gain = self.entropyData(data)
        # print(gain)
        instances = len(data)
        for value in self.getValuesInAttribute(data, attr):
            # print(value)
            # print(self.filterDataFrame(dataset, attr, value))
            gain = gain -(self.getValueInstance(data,attr,value)/instances) * (self.entropyData(self.filterDataFrame(data, attr, value)))

        #print("gain" , gain)
        return gain

    def getValueInstance(self, data, attr, targetValue):
        count = 0
        for value in data.loc[:,attr]:
            if(targetValue == value):
                count +=1
        return count

    def getManyInstances(self,data):
        return len(data)

    def getValuesInAttribute(self, data, attr):
        return list(set(data.loc[:, attr]))

    def getAttributesInData(self, data):
        atrs = list(data.columns)
        del atrs[-1]
        return atrs

    def splitInfo(self, data, attr):
        """
        value dari attribute sudah di diskritkan
        data = data set yang dipakai
        attr = attribute yang ingin dicari splitInfonya
        """
        valueSet = self.getValuesInAttribute(data, attr)
        valueMap = dict.fromkeys(valueSet, 0)
        instances = self.getManyInstances(data)

        for value in data.loc[:,attr]:
            valueMap[value] += 1

        splitInfoAttr = 0
        for value in self.getValuesInAttribute(data, attr):
            splitInfoAttr -= valueMap[value]/instances * math.log(valueMap[value]/instances,2)

        return splitInfoAttr

    def gainRatio(self, data, attr):
        """
        gainRatio untuk per attribute
        data = dataset yang digunakan
        attr = attribute yang ingin dicari gain rationya
        """
        return self.informationGain(data, attr) / self.splitInfo(data,attr)

    def isNan(self, value):
        return math.isnan(value)

    def mostValue(self, data, attr):
        valueSet = self.getValuesInAttribute(data, attr)
        valueMap = dict.fromkeys(valueSet, 0)
        instances = len(data)

        for value in data.loc[:,attr]:
            valueMap[value] += 1

        return max(valueMap.items(), key=op.itemgetter(1))[0]

    def handleMissingValues(self, data):
        attributes = self.getAttributesInData(data)
        for attribute in attributes:
            mostValueInAttribute = self.mostValue(data,attribute)
            data.loc[data[attribute] == float('NaN'),attribute] = mostValueInAttribute 
            #tolong dicobain bs ato nggak, kalo gabisa coba ganti jadi
            #data.loc[data[attribute] == np.nan,attribute] = mostValueInAttribute
            
    def buildTreeInit(self, trainingSet = None):
        curr_node = self.root
        attr_set = self.getAttributesInData(trainingSet)
        self.buildTree(curr_node, trainingSet, attr_set)

    def buildTree(  self,
                    curr_node = None, # current root
                    trainingSet = None, # dataset training
                    attr_set = None
                    ):


        # initial dataframe pruning from VALUES/decision taken by parents
        dataset = trainingSet
        # print(curr_node.valuesTaken.items())

        #print("EntropyBigData: ", self.entropyData(dataset))
        if self.entropyData(dataset) == 0.0 :
            # leaf node!
            # print("LEAF!!!")
            curr_node.attribute = self.targetAttribute
            try:
                curr_node.valuesTaken[curr_node.attribute] = self.getValuesInAttribute(dataset, curr_node.attribute)[0]
            except:
                print("List index out of range! BECAUSE NO SUCH KIND OF ")
                print("Dataset", dataset)
                print("values taken", curr_node.valuesTaken.items())
                print("curr_node", curr_node.attribute)
                print(self.getValuesInAttribute(dataset, curr_node.attribute))
                return
            curr_node.children = []
            #print("Attribute1: " + curr_node.attribute)
            # print("EntropyData: ", self.entropyData(dataset))
            return
        elif not attr_set:
            # if attr_set is already empty but entropy is not 0: OUTLIER
            print("Entropy not 0 but already ran out attributes")
            curr_node.attribute = self.targetAttribute
            curr_node.valuesTaken[curr_node.attribute] = self.mostValue(data, curr_node.attribute)
            curr_node.children = []
            return

        best_node = (None, -999) # best_node -> (attribute name, information gain value)
        # count Information Gain for every attributes:
        for attr in attr_set:
            candidateIG = self.informationGain(dataset, attr)
            if (candidateIG > best_node[1]):
                best_node = (attr, candidateIG)

        # best attribute = best_node
        curr_node.attribute = best_node[0]
        #print("Attribute2: ", curr_node.attribute)
        vals_set = self.getValuesInAttribute(data, best_node[0])
        for value in vals_set:
            temp = dict(curr_node.valuesTaken)
            temp[best_node[0]] = value

            dataset = self.filterDataFrame(trainingSet, curr_node.attribute, value)
            if(dataset.empty):
                # if such combination not available in dataset
                continue
            temp_attr_set = attr_set.copy()
            temp_attr_set.remove(curr_node.attribute)

            next_node = curr_node.addChild(_node = Node(_parent = curr_node,
                                    _children = [],
                                    _valuesTaken = temp
                                    ))
            # print(curr_node.attribute, " : ")
            # for x in curr_node.children:
            #     print("--", x.attribute)
            self.buildTree(next_node, dataset, set(temp_attr_set))
        #print(curr_node.attribute, next_node.attribute)


    def predict(self, values): # (values = {} dict of facts provided)
        # traverse tree from root
        curr_node = self.root

        while(not curr_node.isLeaf()):
            # what is current node attribute about?
            curr_attr = curr_node.attribute
            # what is target value from values (fact)?
            target_val = values[curr_attr]
            # choose child with appropriate values taken:
            for child in curr_node.children:
                #print(child.valuesTaken[curr_attr], target_val)
                # dataframe data must be converted to string first or else it won't match in comparison!!!
                if str(child.valuesTaken[curr_attr]) == target_val:
                    #print("FOUND!")
                    # found a child that matches with facts, move current node to child:
                    curr_node = child
                    break

        # now curr_node is at leaf, return the prediction of target attribute:
        return curr_node.valuesTaken[self.targetAttribute]

    def printTree(self):
        print(">" + str(self.root.attribute))
        for child in self.root.children:
            child._printTree(space = 4)

def handleContinuousAttribute(data):
    target = data.columns[-1]
    t = MyTree(_targetAttribute=target)
    for attr in t.getAttributesInData(data):
        if (isinstance(data.iloc[0][attr], float)):
            threshold_dict[attr] = max(data[attr])
            best_threshold = threshold_dict[attr]
            best_information_gain = -1
            data = data.sort_values(by=[attr])
            target_value = data.iloc[0][target]
            data_temp = data.copy()
            for index, row in data.iterrows():
                if (row[target] == target_value):
                    pass
                else:
                    threshold_temp = (data.iloc[index-1][attr] + data.iloc[index][attr]) / 2
                    data_temp.loc[data_temp[attr] < threshold_temp, attr] = 0
                    data_temp.loc[data_temp[attr] >= threshold_temp, attr] = 1
                    information_gain_temp = t.informationGain(data_temp, attr)
                    if (information_gain_temp > best_information_gain):
                        best_information_gain = information_gain_temp
                        best_threshold = threshold_temp

            threshold_dict[attr] = best_threshold
            print("best_threshold: " + str(best_threshold))
            data.loc[data[attr] < best_threshold , attr] = 0
            data.loc[data[attr] >= best_threshold , attr] = 1

    return data

data = pd.read_csv("tennis.csv")
# print(data)
#print(entropyData(data))
# print(getValuesInAttribute(data, "play"))
t = MyTree(_targetAttribute = "play")
# print("Humidity Infogain : " + str(t.informationGain(data, "humidity")))
# print("Outlook Infogain : " + str(t.informationGain(data, "outlook")))

t.buildTreeInit(trainingSet = data)
t.printTree()
print(t.predict({"outlook" : "rainy", "windy" : "False"}))
