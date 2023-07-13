import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
from torch.nn.functional import softmax


def pad_ndarrays(ndarray_list, user_dim):
    padded_list = []
    for arr in ndarray_list:
        current_dim = arr.shape[0]
        pad_width = ((0, max(user_dim - current_dim, 0)), (0, 0))
        padded_arr = np.pad(arr, pad_width, mode='constant')
        padded_list.append(padded_arr)

    return padded_list

def find_median_length(arrays):
    lengths = [len(arr) for arr in arrays]
    median_length = np.median(lengths)
    return median_length


def find_max_length(arrays):
    max_length = max(len(arr) for arr in arrays)
    return max_length

embedding_dim = 100
hidden_size = 128
num_epochs = 25
batch_size = 32
learning_rate = 0.0007

def preprocess_data(raw_data):
    lab1_list = []
    lab2_list = []
    sentence_list = []
    for line in raw_data:
        line = line.strip()
        if ':' in line:
            parts = line.split(':',1)
            lab1 = parts[0]
            parts = parts[1].split(maxsplit=1)
            lab2 = parts[0]
            sent = parts[1]
            lab1_list.append(lab1)
            lab2_list.append(lab2)
            sentence_list.append(sent)
    return lab1_list, lab2_list, sentence_list

class SentenceDataset(Dataset):
    def __init__(self, sentences, label_set1, label_set2):
        self.sentences = sentences
        self.label_set1 = label_set1
        self.label_set2 = label_set2

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.label_set1[idx], self.label_set2[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes_set1, num_classes_set2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, num_classes_set1)
        self.fc_2 = nn.Linear(hidden_size, num_classes_set2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        output, _ = self.lstm(x)
        output_set1 = self.fc_1(output[:, -1, :])
        output_set2 = self.fc_2(output[:, -1, :])
        return output_set1, output_set2


def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokens = sentence.split()
        tokenized_sentences.append(tokens)
    return tokenized_sentences

def get_sentence_embeddings(sentence, glove):
    embeddings = []
    for word in sentence:
        embedding = glove[word]
        embeddings.append(embedding.numpy()) if embedding is not None else embeddings.append(np.zeros(embedding_dim))
    return np.array(embeddings)


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
        raw_data = file.readlines()

    labels_set1, labels_set2, sentences = preprocess_data(raw_data)

    return sentences, labels_set1, labels_set2


def train_and_evaluate_model(sentences, labels_set1, labels_set2):
    glove = GloVe(name='6B', dim=embedding_dim)

    # Convert sentences to GloVe embeddings
    tokenized_sentences = tokenize_sentences(sentences)
    median_len = find_median_length(tokenized_sentences)
    max_lem = find_max_length(tokenized_sentences)
    sentence_embeddings = [get_sentence_embeddings(sentence, glove) for sentence in tokenized_sentences]
    sentence_embeddings = pad_ndarrays(sentence_embeddings,37)

    # Convert labels to numerical indices
    label_set1_to_index = {label: i for i, label in enumerate(set(labels_set1))}
    label_set2_to_index = {label: i for i, label in enumerate(set(labels_set2))}
    indices_set1 = [label_set1_to_index[label] for label in labels_set1]
    indices_set2 = [label_set2_to_index[label] for label in labels_set2]

    num_classes_set1 = len(label_set1_to_index)
    num_classes_set2 = len(label_set2_to_index)


    # Define the loss function and optimizer
    criterion_set1 = nn.CrossEntropyLoss()
    criterion_set2 = nn.CrossEntropyLoss()

    # Performing 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True)
    for fold, (train_index, val_index) in enumerate(kf.split(sentence_embeddings)):
        print(f"Fold: {fold + 1}")

        model = LSTMClassifier(embedding_dim, hidden_size, num_classes_set1, num_classes_set2)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_sentences = [sentence_embeddings[i] for i in train_index]
        train_labels_set1 = [indices_set1[i] for i in train_index]
        train_labels_set2 = [indices_set2[i] for i in train_index]
        val_sentences = [sentence_embeddings[i] for i in val_index]
        val_labels_set1 = [indices_set1[i] for i in val_index]
        val_labels_set2 = [indices_set2[i] for i in val_index]

        # Create data loaders
        train_dataset = SentenceDataset(train_sentences, train_labels_set1, train_labels_set2)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = SentenceDataset(val_sentences, val_labels_set1, val_labels_set2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model.train()
        cnt = 0
        for epoch in range(num_epochs):
            total_loss_set1 = 0
            total_loss_set2 = 0
            for batch_sentences, batch_labels_set1, batch_labels_set2 in train_loader:
                cnt += 1
                optimizer.zero_grad()
                batch_sentences = torch.FloatTensor(batch_sentences)
                batch_labels_set1 = torch.LongTensor(batch_labels_set1)
                batch_labels_set2 = torch.LongTensor(batch_labels_set2)
                predictions_set1, predictions_set2 = model(batch_sentences)
                loss_set1 = criterion_set1(predictions_set1, batch_labels_set1)
                loss_set2 = criterion_set2(predictions_set2, batch_labels_set2)
                loss = loss_set1 + loss_set2
                loss.backward()
                optimizer.step()
                total_loss_set1 += loss_set1.item()
                total_loss_set2 += loss_set2.item()
                if cnt%50==0:
                    print(
                    f"Epoch {epoch + 1}/{num_epochs}, Train Loss (Set1): {loss_set1:.4f}, Train Loss (Set2): {loss_set2:.4f}")

        model.eval()
        with torch.no_grad():
            total1 = 0
            correct1 = 0
            total2 = 0
            correct2 = 0
            cntt = 0
            f1_1_tot = 0
            f1_2_tot = 0
            cm1_tot = None
            cm2_tot = None
            for batch_sentences, batch_labels_set1, batch_labels_set2 in val_loader:
                cntt +=1
                batch_sentences = torch.FloatTensor(batch_sentences)
                output1, output2 = model(batch_sentences)
                _, predicted1 = torch.max(output1.data, 1)
                _, predicted2 = torch.max(output2.data, 1)
                total1 += batch_labels_set1.size(0)
                total2 += batch_labels_set2.size(0)
                correct1 += (predicted1 == batch_labels_set1).sum().item()
                correct2 += (predicted2 == batch_labels_set2).sum().item()
                output1 = torch.argmax(output1, dim = -1)
                output2 = torch.argmax(output2, dim = -1)
                f1_1 = f1_score(batch_labels_set1, output1, average='weighted')
                f1_2 = f1_score(batch_labels_set2, output2, average='weighted')
                f1_1_tot +=f1_1
                f1_2_tot += f1_2

            print('The confusion matrices for set1 and set2 are ', cm1_tot, cm2_tot)
            print('The average f1 scores for set1 and set2 are ', f1_1_tot/cntt, f1_2_tot/cntt)
            accuracy1 = correct1 / total1
            accuracy2 = correct2 / total2
            print(f"Validation Accuracy for set 2 is : {accuracy1:.4f}")
            print(f"Validation Accuracy for set 1 is : {accuracy2:.4f}")


# Main code
file_path = 'C:/Users/puran/PycharmProjects/BlackCoffer/training_data.txt'  # Replace with your file path
sentences, labels_set1, labels_set2 = load_data(file_path)
train_and_evaluate_model(sentences, labels_set1, labels_set2)

