import random

class ExampleGenerator(object):
    def __init__(self, M, N):
        self.L = M # label space dimension
        self.M = M # feature vector space dimension
        self.N = N # size of data set
        self.data_set = {} # data_set

        for p in range(N):
            random_size = random.randint(1,10)
            random_index = random.sample(range(self.L), random_size)
            random_vector = {q: 0 for q in random_index}

            for q in random_index:
                if random.choice((True, False)):
                    random_vector[q] = random.uniform(1.0, 2.0)
                else:
                    random_vector[q] = random.uniform(-2.0, -1.0)

            self.data_set[p] = {"features": random_vector}

        for p in range(N):
            vector_dict = self.data_set[p]["features"]
            self.data_set[p]["labels"] = [q for q in vector_dict.keys() if vector_dict[q] > 0]

    def file_write(self, file_name):
        L, M, N = self.L, self.M, self.N
        with open(file_name, 'w') as f:
            f.write("{} {} {}\n".format(N, M, L))
            for p in range(N):
                labels = ','.join(map(str, self.data_set[p]["labels"]))
                vector_dict = self.data_set[p]["features"]
                features = ' '.join(["{}:{}".format(q, vector_dict[q]) for q in vector_dict.keys()])
                f.write("{} {}\n".format(labels, features))

if __name__ == '__main__':
    e = ExampleGenerator(M=10, N=100)
    e.file_write("train.txt")

    e = ExampleGenerator(M=10, N=10)
    e.file_write("test.txt")
