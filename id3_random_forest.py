import tensorflow as tf
from tqdm import tqdm
import numpy as np
import math
from collections import Counter
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.datasets import make_classification

skip = 50
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=200, skip_top=skip)

word_index = tf.keras.datasets.imdb.get_word_index()
index2word = dict((i + 3, word) for (word, i) in word_index.items())
index2word[0] = '[pad]'
index2word[1] = '[bos]'
index2word[2] = '[oov]'

x_train = np.array([' '.join([index2word[idx] for idx in text]) for text in x_train])
x_test = np.array([' '.join([index2word[idx] for idx in text]) for text in x_test])

# create vocabulary
vocabulary = list()
for text in x_train:
    tokens = text.split()
    vocabulary.extend(tokens)

vocabulary = set(vocabulary)

# create binary vectors
x_train_binary = list()
x_test_binary = list()

for text in tqdm(x_train):
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    x_train_binary.append(binary_vector)

x_train_binary = np.array(x_train_binary)

for text in tqdm(x_test):
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    x_test_binary.append(binary_vector)

x_test_binary = np.array(x_test_binary)


def get_entropy(data):
    """
    # data: lista dyadikou dianysmatos
    """
    c = Counter(data)  # lexiko me to plhthos twn 0 kai 1
    if c[0] == 0 or c[1] == 0:
        return 0  # ama ola ta stoixeia einai se mia kathgoria eimai apolyta bebaios
    return -(c[0] / len(data)) * math.log2(c[0] / len(data)) - (c[1] / len(data)) * math.log2(c[1] / len(data))


def attr_entropy(x_data, y_data, label):
    """
    # x_data: lista me listes dyadikwn dianysmatwn (keimeno kritikhs)
    # y_data: lista dyadikou dianysmatos (thetikh/arnhtikh kritikh)
    # label: string lexi apo tis kritikes
    """
    if label not in word_index:
        return
    col = word_index[label]-skip  # se poia thesi brisketai h sygkekrimenh lexi
    has_counter_vector = []  # dyadiko dianysma opoy gia kathe kritikh opou yparxei h lexi se aythn kai me 1 einai thetikh kai 0 arnhtikh
    not_has_counter_vector = []  # dyadiko dianysma opoy gia kathe kritikh opou  DEN yparxei h lexi se aythn kai me 1 einai thetikh kai 0 arnhtikh
    for i in range(len(x_data)):
        if x_data[i][col]:
            if y_data[i]:
                has_counter_vector.append(1)  # thetikh prosthetei 1
            else:
                has_counter_vector.append(0)  # arnhtikh prosthetei 0
        else:
            if y_data[i]:
                not_has_counter_vector.append(1)  # thetikh prosthetei 1
            else:
                not_has_counter_vector.append(0)  # arnhtikh prosthetei 0
    has_at_en = 0
    if len(has_counter_vector) > 0:
        has_at_en = get_entropy(has_counter_vector)

    not_has_at_en = 0
    if len(not_has_counter_vector) > 0:
        not_has_at_en = get_entropy(not_has_counter_vector)

    return (len(has_counter_vector) / len(x_data)) * has_at_en + (
                len(not_has_counter_vector) / len(x_data)) * not_has_at_en


def info_gain(x_data, y_data, label):
    """
    # x_data: lista me listes dyadikwn dianysmatwn (keimeno kritikhs)
    # y_data: lista dyadikou dianysmatos (thetikh/arnhtikh kritikh)
    # label: string lexi apo tis kritikes
    """
    return get_entropy(y_data) - attr_entropy(x_data, y_data, label)


def max_info_gain(x_data, y_data):
    """
    # x_data: lista me listes dyadikwn dianysmatwn (keimeno kritikhs)
    # y_data: lista dyadikou dianysmatos (thetikh/arnhtikh kritikh)
    """
    maxig = ("abcdesfg", -1)
    for i in vocabulary:
        if i in ['[pad]', '[bos]', '[oov]']:
            continue
        igi = (i, info_gain(x_data, y_data, i))
        if igi is not None and igi[1] > maxig[1]:
            maxig = igi

    return maxig


def top_attributes(x_data, y_data, top_words):
    top_ig = []
    for i in vocabulary:
        if i in ['[pad]', '[bos]', '[oov]']:
            continue
        q = info_gain(x_data, y_data, i)
        top_ig.append((i, q))
    top_ig.sort(key=lambda x: x[1])
    return top_ig[:top_words]


class Node:
    def __init__(self, x, y):
        """
        # Creates a new Node instance.
        # Args:
        # -----
        # x: The data to be contained in this node
        # y: pos/neg
        #split_con: The attribute that this data was split on:type split_con: object
        """
        self.x_data = x
        self.y_data = y
        self.split_condition = ""
        self.one = None  # has the split_condition
        self.zero = None  # does not have the split_condition


    def set_split_condition(self, label):
        self.split_condition = label

    def split_Node(self):
        col = word_index[self.split_condition]-skip
        has_word = []
        y_has_word = []
        not_has_word = []
        y_not_has_word = []
        for i in range(len(self.x_data)):
            if self.x_data[i][col]:
                has_word.append(self.x_data[i])
                if self.y_data[i]:
                    y_has_word.append(1)  # thetikh prosthetei 1
                else:
                    y_has_word.append(0)  # arnhtikh prosthetei 0
            else:
                not_has_word.append(self.x_data[i])
                if self.y_data[i]:
                    y_not_has_word.append(1)  # thetikh prosthetei 1
                else:
                    y_not_has_word.append(0)  # arnhtikh prosthetei 0

        self.one = Node(has_word, y_has_word)
        self.zero = Node(not_has_word, y_not_has_word)

    def tree(self):
        c = Counter(self.y_data)
        ig = max_info_gain(self.x_data, self.y_data)
        if len(self.y_data) <= 1 or c[0] == 0 or c[1] == 0 or ig[1] <= 0:  # elegxos an ola anhkoun se mia kathgoria an den exei dedomena kai an exei thetiko information gain
            return self

        self.set_split_condition(ig[0])
        self.split_Node()
        self.one.tree()
        self.zero.tree()


class ID3:
    def fit(self, x_data, y_data): #friaxnei to dentro 
        root = Node(x_data, y_data)
        root.tree()
        self.tree = root
        return root

    def predict(self, x_data): #xrhsimopoiei to dentro g na katataksei ta test
        pred = []
        for i in range(len(x_data)):
            root = self.tree  #epanafora ths rizas wste na mhn prospathei na synexisei apo fyllo
            while root.one is not None and root.zero is not None:
                if x_data[i][word_index[root.split_condition]-skip]:
                    root = root.one
                else:
                    root = root.zero
            c = Counter(root.y_data)
            if c[0] < c[1]:
                pred.append(1)
            else:
                pred.append(0)
        return pred


class Random_Forest():
    def subsample(self, x_data, y_data): #dhmioyrgoyntai 5 dentra
        x_sample = list()
        y_sample = list()
        num_selected = {}
        for i in range(int(len(x_data) / 5)):
            num = random.randint(0, len(x_data) - 1)
            while num in num_selected.keys():
                num = random.randint(0, len(x_data) - 1)
            num_selected[num] = None
            x_sample.append(x_data[num])
            y_sample.append(y_data[num])
        return x_sample, y_sample

    
    def predict(self, x_train, y_train, x_test):#kaleitai h predict g ka8e dentro (xrhsimopoieitai o id3 g to kaue ena // kai pairnoume twn meso oro tous
        trees = list()
        for i in range(5):
            x_sample, y_sample = self.subsample(x_train, y_train)
            tree = ID3()
            tree.fit(x_sample, y_sample)
            trees.append(tree)
        
        predictions = list()
        for t in trees:
            predictions.append(t.predict(x_test))
        
        final_pred = list()

        for i in range(len(x_test)):
            pred = 0
            for j in range(len(trees)):
                pred += predictions[j][i]
            pred = pred / len(trees)
            final_pred.append(1 if pred > 0.5 else 0) # panw apo ta misa dentra symfonoun oti einai thetikh
        return final_pred

##etoima 
def learning_curves():
    train_sizes = [1, 300, 450, 600, 1000, 2500, 5000]

    train_sizes, train_scores, validation_scores = learning_curve(estimator = LinearRegression(),
                                                                X = x_train_binary,
                                                                y = y_train, 
                                                                train_sizes = train_sizes, 
                                                                cv = 5,
                                                                scoring = 'neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')
    plt.xlabel('Training set size')
    plt.title('Learning curves')
    plt.legend()
    plt.ylim(0,1)

    plt.show()


def prec_rec_curve():
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
    trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2)

    model = LogisticRegression(solver='lbfgs')
    model.fit(trainX, trainy)

    lr_probs = model.predict_proba(testX)
    lr_probs = lr_probs[:, 1]

    yhat = model.predict(testX)
    lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)

    no_skill = len(testy[testy==1]) / len(testy)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.show()

####################################################
# ID3

start = time.time()
print("Starting ID3...")

voc = top_attributes(x_train_binary, y_train, 100)
vocabulary = set()
for i in voc:
    vocabulary.add(i[0])

the_root = ID3()
the_root.fit(x_train_binary, y_train)

print(classification_report(y_test, the_root.predict(x_test_binary)))

end = time.time()
print("Runtime of the ID3 is: ", (end - start)/60)

####################################################
# Random Forest
print("\n","#"*20,"\n")

start = time.time()
print("\nStarting Random Forest...")

rf = Random_Forest()
print(classification_report(y_test, rf.predict(x_train_binary, y_train, x_test_binary)))

end = time.time()
print("Runtime of the Random Forest is: ", (end - start)/60)


