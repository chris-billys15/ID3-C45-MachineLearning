import pandas as pd
import math
import operator as op

def getValuesInAttribute(data, attr):
    return list(set(data.loc[:, attr]))

def mostValue(data, attr):
        valueSet = getValuesInAttribute(data, attr)
        valueMap = dict.fromkeys(valueSet, 0)
        instances = len(data)

        for value in data.loc[:,attr]:
            valueMap[value] += 1

        return max(valueMap.items(), key=op.itemgetter(1))[0]

data = pd.DataFrame({'Pclass':['Low','Medium','High','Low','Medium'],
                      'Cabin':['Ferdy','Ferdy','Bel','Oksi','Oksi']})
data.loc[data['Cabin'] == 'Ferdy','Cabin'] = 'oksi'
print(data)
# print(data['Cabin'])