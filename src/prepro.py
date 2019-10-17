#!/usr/bin/env python3
import os

'''
* For preprocessing to handle with small data set
more detail: http://manikvarma.org/downloads/XC/XMLRepository.html
'''

if __name__ == '__main__':
    path = input("Data set file name: ")
    whole_file = "data/" + path + "/data.txt"
    train_file = "data/" + path + "/trSplit.txt"
    test_file = "data/" + path + "/tstSplit.txt"

    original_lines = []
    with open(whole_file, "r") as f:
        header = f.readline()
        N, M, L = (int(x) for x in header.split())

        line = f.readline()
        while line:
            original_lines.append(line)
            line = f.readline()

    train_index = [[] for i in range(10)]
    test_index = [[] for i in range(10)]

    with open(train_file, "r") as f:
        line = f.readline()
        while line:
            for i in range(10):
                train_index[i].append(int(line.split()[i]))

            line = f.readline()

    with open(test_file, "r") as f:
        line = f.readline()
        while line:
            for i in range(10):
                test_index[i].append(int(line.split()[i]))

            line = f.readline()

    for i in range(10):
        os.mkdir("{}{}".format(path, i))
        new_train_file = "{}{}/train.txt".format(path, i)
        with open (new_train_file, "w") as f:
            f.write("{} {} {}\n".format(len(train_index[i]), M, L))
            for j in train_index[i]:
                f.write(original_lines[j-1])

        new_test_file = "{}{}/test.txt".format(path, i)
        with open (new_test_file, "w") as f:
            f.write("{} {} {}\n".format(len(test_index[i]), M, L))
            for j in test_index[i]:
                f.write(original_lines[j-1])
