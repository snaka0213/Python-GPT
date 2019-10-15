#!/user/bin/env python3
'''
* make `data_set`, dict object {index: data}
    `data` is dict object, whose keys are:
    *  'label' : `label_vector`, list object subset in [0,...,L-1]
    * 'feature': `feature_vector`, dict object {coordinate index: value}
'''

class FileReader(object):
    def __init__(self):
        self.L = None # label vector space dimension
        self.M = None # feature vector space dimension
        self.N = None # the size of `data_set`

    def read(self, file_name: str) -> dict:
        with open(file_name, "r") as f:
            header = f.readline()
            N, M, L = (int(x) for x in header.split())
            self.L, self.M, self.N = L, M, N

            data_index = 0
            data_set = {i:{} for i in range(N)}

            line = f.readline()
            while line:
                data = {}
                label_term = line.split()[0].split(',')
                if ':' in label_term[0]:
                    data['label'] = []
                    feature_term = label_term
                else:
                    data['label'] = [int(l) for l in label_term]
                    feature_term = line.split()[1:]

                data['feature'] = {int(m.split(':')[0]): float(m.split(':')[1]) for m in feature_term}

                data_set[data_index] = data
                data_index += 1
                line = f.readline()

        return data_set
