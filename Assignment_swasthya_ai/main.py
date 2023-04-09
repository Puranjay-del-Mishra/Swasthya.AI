import torch

train_sentences = []
test_sentences = []
with open('training.data') as f:
    contents = f.read()
    line = str()
    for i in range(len(contents)):
        if contents[i]!='\n':
            line = line + contents[i]
        if contents[i] == '?':
            train_sentences.append(line)
            line = str()

train_test_split = 0.8


from torch import nn
from torchtext.vocab import GloVe
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Module
from torch.utils.data import Dataset

tokenizer = get_tokenizer("basic_english")
global_vectors = GloVe(name='6B', dim=200)
#
train_data = []

max_len = 20

for i in train_sentences:
    embeddings = global_vectors.get_vecs_by_tokens(tokenizer(i), lower_case_backup=True)
    train_data.append(embeddings)

train_data = pad_sequence(train_data,batch_first=True)
print(train_data.shape)
train_data = torch.reshape(train_data,[15141,57*200])
train_data = train_data.detach().numpy()
train_inp = train_data[:15041]
test_inp = train_data[15041:]  ##100 test samples
test_sentences = train_sentences[15041:]

import sklearn
from sklearn.cluster import MiniBatchKMeans

print('Training a Mini batch K means on the training data....')
Kmeans = MiniBatchKMeans(n_clusters=2, random_state=0, batch_size=4096, max_iter= 150, n_init="auto")

out = Kmeans.fit(train_inp)


prediction = Kmeans.predict(test_inp)

for line,label in zip(test_sentences,prediction):
    print(line,label)

zero_cnt = 0
one_cnt = 0

for val in prediction:
    if val==0:
        zero_cnt +=1
    else:
        one_cnt +=1

print(zero_cnt,one_cnt)