import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_fscore_support, f1_score
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

class AFM(nn.Module):
    def __init__(self, features_info, embedding_dim):
        super().__init__()
        # parse feature_info
        self.dense_features, self.sparse_features, sparse_features_nunique, dense_features_nunique = features_info

        # calculate the number of dense and sparse type features
        self.__dense_features_num = len(self.dense_features)
        self.__sparse_features_num = len(self.sparse_features)

        # building the Embedding layer
        self.embedding_layers = nn.ModuleDict({
            "embed_" + key: nn.Embedding(num, embedding_dim)
            for key, num in sparse_features_nunique.items()
        })

        # building an attention network
        self.attention = nn.Linear(embedding_dim, 12, bias=True)
        self.project = nn.Linear(12, 1)

        # build linear part
        self.linear_part = nn.Linear(self.__dense_features_num + self.__sparse_features_num, 1, bias=True)

        # cross-featured output section
        self.predict_layer = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # separate out sparse_input and dense_input from input x
        dense_inputs, sparse_inputs = x[:, :self.__dense_features_num], x[:, self.__dense_features_num:]
        sparse_inputs = sparse_inputs.long()

        # first-order linear partial calculation
        part_1 = self.linear_part(x)

        # embedding
        embedding_feas = [self.embedding_layers["embed_" + key](sparse_inputs[:, idx]) for idx, key in
                          enumerate(self.sparse_features)]

        embedding_feas = torch.stack(embedding_feas).permute(
            (1, 0, 2))  # [features_nums, batch, embedding_dim] -> [batch, features_nums, embedding_dim]

        # embedding cross-calculation
        row, col = [], []
        for i in range(self.__sparse_features_num):
            for j in range(i + 1, self.__sparse_features_num):
                row.append(i)
                col.append(j)
        cross_feas = embedding_feas[:, row, :] * embedding_feas[:, col, :]

        # compute softmax in the cross feature dimension, which is used to measure the weight of cross features
        attention_scores = F.softmax(self.project(F.relu(self.attention(cross_feas))) / 0.1, dim=1)
        attention_output = torch.sum(cross_feas * attention_scores, dim=1)
        part_2 = self.predict_layer(attention_output)
        # results of first-order features and second-order cross features (bias is included in first-order features)
        output = torch.sigmoid(part_1 + part_2)
        return output


def getCriteo(data_path='./data/train.csv'):
    df_data = pd.read_csv(data_path, sep=',')
    df_data.drop(['Id'], axis=1, inplace=True)

    dense_features = ['I' + str(i + 1) for i in range(13)]
    sparse_features = ['C' + str(i + 1) for i in range(26)]

    # fill missing values
    df_data[sparse_features] = df_data[sparse_features].fillna('-1')
    df_data[dense_features] = df_data[dense_features].fillna(0)

    # labelEncoder encoding of category-type features
    for feature in sparse_features:
        df_data[feature] = LabelEncoder().fit_transform(df_data[feature])

    # numerical features for feature normalization
    df_data[dense_features] = MinMaxScaler().fit_transform(df_data[dense_features])

    label = df_data.pop('Label')
    sparse_features_nunique = {}
    dense_features_nunique = {}
    for fea in sparse_features:
        sparse_features_nunique[fea] = df_data[fea].nunique()
    for fea in dense_features:
        dense_features_nunique[fea] = df_data[fea].nunique()

    features_info = [dense_features, sparse_features, sparse_features_nunique, dense_features_nunique]

    return df_data, label, features_info

class TrainTask:
    def __init__(self, model, lr=0.001, use_cuda=False):
        self.__device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.__model = model.to(self.__device)
        self.__loss_fn = nn.BCELoss().to(self.__device)
        self.__optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.train_loss = []
        self.eval_accuracy = []
        self.eval_precision = []
        self.eval_recall = []
        self.eval_f1 = []

    def __train_one_batch(self, feas, labels):
        """
        train one batch
        """
        self.__optimizer.zero_grad()
        # 1. forward
        outputs = self.__model(feas)
        # 2. loss
        loss = self.__loss_fn(outputs.squeeze(), labels)
        # 3. gradient backward
        loss.backward()
        self.__optimizer.step()
        return loss.item(), outputs

    def __train_one_epoch(self, train_dataloader, epoch_id):
        """
        train one epoch
        """
        self.__model.train()

        loss_sum = 0
        batch_id = 0
        for batch_id, (feas, labels) in enumerate(train_dataloader):
            feas, labels = Variable(feas).to(self.__device), Variable(labels).to(self.__device)
            loss, outputs = self.__train_one_batch(feas, labels)
            loss_sum += loss
        self.train_loss.append(loss_sum / (batch_id + 1))
        print("Training Epoch: %d, mean loss: %.5f" % (epoch_id, loss_sum / (batch_id + 1)))

    def train(self, train_dataset, eval_dataset, epochs, batch_size):
        # 构造DataLoader
        train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            print('-' * 20 + ' Epoch {} starts '.format(epoch) + '-' * 20)
            self.__train_one_epoch(train_data_loader, epoch_id=epoch)
            self.__eval(eval_data_loader, epoch_id=epoch)
    def __eval(self, eval_dataloader, epoch_id):
        """
        reasoning over the validation set
        """
        batch_id = 0
        accuracy_sum = 0
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        self.__model.eval()

        for batch_id, (feas, labels) in enumerate(eval_dataloader):
            with torch.no_grad():
                feas, labels = Variable(feas).to(self.__device), Variable(labels).to(self.__device)
                y_true = []
                y_pred = []
                outputs = self.__model(feas)
                outputs = outputs.view(-1)
                y_true.extend(labels.tolist())
                for output in outputs:
                    if output >= 0.5:
                        y_pred.append(1.0)
                    if output < 0.5:
                        y_pred.append(0.0)
                accuracy = accuracy_score(y_true,y_pred)
                precision = precision_score(y_true,y_pred)
                recall = recall_score(y_true,y_pred)
                f1 = f1_score(y_true,y_pred)
                accuracy_sum += accuracy
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
        self.eval_accuracy.append(accuracy_sum / (batch_id + 1))
        self.eval_precision.append(precision_sum / (batch_id + 1))
        self.eval_recall.append(recall_sum / (batch_id + 1))
        self.eval_f1.append(f1_sum / (batch_id + 1))
        print("Evaluate Epoch: %d, mean accuracy: %.5f, mean precision: %.5f, mean recall: %.5f, mean f1: %.5f" % (epoch_id, accuracy_sum / (batch_id + 1), precision_sum / (batch_id + 1), recall_sum / (batch_id + 1), f1_sum / (batch_id + 1)))

    def __plot_metric_train(self, train_metrics, metric_name):
        """
        metrics visualization
        """
        epochs = range(1, len(train_metrics) + 1)
        plt.plot(epochs, train_metrics, 'bo--')
        plt.title('Training ' + metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend(["train_" + metric_name])
        plt.show()

    def __plot_metric_val_acc(self, val_metrics, metric_name):
        """
        metrics visualization
        """
        epochs = range(1, len(val_metrics) + 1)
        plt.plot(epochs, val_metrics, 'ro--')
        plt.title('Validation ' + metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend(['val_' + metric_name])
        plt.show()

    def __plot_metric_val_prec(self, val_metrics, metric_name):
        """
        metrics visualization
        """
        epochs = range(1, len(val_metrics) + 1)
        plt.plot(epochs, val_metrics, 'ro--')
        plt.title('Validation ' + metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend(['val_' + metric_name])
        plt.show()

    def __plot_metric_val_recall(self, val_metrics, metric_name):
        """
        metrics visualization
        """
        epochs = range(1, len(val_metrics) + 1)
        plt.plot(epochs, val_metrics, 'ro--')
        plt.title('Validation ' + metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend(['val_' + metric_name])
        plt.show()

    def __plot_metric_val_F1(self, val_metrics, metric_name):
        """
        metrics visualization
        """
        epochs = range(1, len(val_metrics) + 1)
        plt.plot(epochs, val_metrics, 'ro--')
        plt.title('Validation ' + metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)
        plt.legend(['val_' + metric_name])
        plt.show()

    def plot_loss_curve(self):
        self.__plot_metric_train(self.train_loss, "Loss")
        self.__plot_metric_val_acc(self.eval_accuracy, "Accuracy")
        self.__plot_metric_val_prec(self.eval_precision, "Precision")
        self.__plot_metric_val_recall(self.eval_recall, "Recall")
        self.__plot_metric_val_F1(self.eval_f1, "F1")

if __name__ == "__main__":
    df_data, label, features_info = getCriteo()

    # partition, build data sets, data channels
    x_train, x_val, y_train, y_val = train_test_split(df_data, label, test_size=0.2, random_state=2022)
    train_dataset = TensorDataset(torch.tensor(x_train.values).float(), torch.tensor(y_train.values).float())
    val_dataset = TensorDataset(torch.tensor(x_val.values).float(), torch.tensor(y_val.values).float())

    # build model
    model = AFM(features_info, embedding_dim=8)

    task = TrainTask(model, lr=0.0001, use_cuda=False)

    task.train(train_dataset, val_dataset, epochs=10, batch_size=16)

    task.plot_loss_curve()
