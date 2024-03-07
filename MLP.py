# Reference: https://python-bloggers.com/2022/05/building-a-pytorch-binary-classification-multi-layer-perceptron-from-the-ground-up/

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='4';
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd 
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss
from torch.optim import SGD, lr_scheduler
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import time, math, argparse
torch.set_num_threads(64)

scaler = preprocessing.MinMaxScaler()

class CSVDataset(Dataset):
    #Constructor for initially loading
    def __init__(self, df, idx, num_features, train_mode):
        # Store the inputs and outputs
        self.X = df.values[idx, 4: 4+num_features]
        self.file_names = df.values[idx,0]
        df['AS_10'] = np.where(df["AS"] > 10, 1, 0)
        if train_mode == True:
            self.X = scaler.fit_transform(self.X)
        else:
            self.X = scaler.transform(self.X)
        self.y = df.values[idx, -1] # df['AS_10']
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))
    # Get the number of rows in the dataset
    def __len__(self):
        return len(self.X)
    # Get a row at an index
    def __getitem__(self,idx):
        return [self.X[idx], self.y[idx]]
    def split_data(self, split_ratio=0.2):
        valid_size = round(split_ratio * len(self.X))
        train_size = len(self.X) - valid_size
        return random_split(self, [train_size, valid_size])
    def n_input(self):
        return len(self.X[0])
    def slide_names(self):
        return self.file_names

# Create model
class MLP(Module):
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # First hidden layer
        self.hidden1 = Linear(n_inputs, 512)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # Second hidden layer
        self.hidden2 = Linear(512, 100)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # Third hidden layer
        self.hidden3 = Linear(100, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self, X):
        # Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        return X

def train_model(train_dl, valid_dl, model, epochs=1000, lr=0.00005, momentum=0.9, save_path='save_best_model.pth'):
    start = time.time()
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    early_stopping_epochs = 200
    best_loss = float('inf')
    early_stop_counter = 0

    loss = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        model.train()
        train_loss = 0.0
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data,1) #Get the class labels
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        print(train_loss)
        #torch.save(model, save_path)

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(valid_dl):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * inputs.size(0)
            print(valid_loss)

        if valid_loss > best_loss:
            early_stop_counter += 1
        else:
            best_loss = valid_loss
            torch.save(model, save_path)
            early_stop_counter =0

        if early_stop_counter >= early_stopping_epochs:
            print("Early Stopping!")
            break

    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))
    
    return model

def evaluate_model(test_dl, model, beta=1.0):
    scores = []
    preds = []
    actuals = []
    for (i, (inputs, targets)) in enumerate(test_dl):
        #Evaluate the model on the test set
        yhat = model(inputs)
        #Retrieve a numpy weights array
        yhat = yhat.detach().numpy()
        # Extract the weights using detach to get the numerical values in an ndarray, instead of tensor
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # Round to get the class value 
        yhat_round = yhat.round()
        # Store the predictions in the empty lists initialised at the start of the class
        scores.append(yhat)
        preds.append(yhat_round)
        actuals.append(actual)
    
    # Stack the predictions and actual arrays vertically
    scores, preds, actuals = np.vstack(scores), np.vstack(preds), np.vstack(actuals)
    #Calculate metrics
    cm = confusion_matrix(actuals, preds)
    # Get descriptions of tp, tn, fp, fn
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())
    
    metrics = {
        'accuracy': accuracy_score(actuals, preds),
        'AU_ROC': roc_auc_score(actuals, scores),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
        'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
        'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        'precision': precision_score(actuals, preds),
        'recall': recall_score(actuals, preds),
        'true_positive_rate_TPR':recall_score(actuals, preds),
        'false_positive_rate_FPR':fp / (fp + tn) ,
        'false_discovery_rate': fp / (fp +tp),
        'false_negative_rate': fn / (fn + tp) ,
        'negative_predictive_value': tn / (tn+fn),
        'misclassification_error_rate': (fp+fn)/total ,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        #'confusion_matrix': confusion_matrix(actuals, preds), 
        'TP': tp,
        'FP': fp, 
        'FN': fn, 
        'TN': tn
    }
    return metrics, preds, actuals, scores

def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    # Get numpy array
    yhat = yhat.detach().numpy()
    return yhat 

def prepare_dataset(df, num_features, n_iter, batch_size):
    train_idx = np.load("./MLP/5_fold_split/AS_train_idx_fold_{}.npy".format(n_iter))
    test_idx = np.load("./MLP/5_fold_split/AS_test_idx_fold_{}.npy".format(n_iter))
    train_dataset = CSVDataset(df, train_idx, num_features, train_mode = True)
    train_split, valid_split = train_dataset.split_data(split_ratio=0.2)
    test_dataset = CSVDataset(df, test_idx, num_features, train_mode = False)
    test_slide_names = test_dataset.slide_names()
    # Prepare data loaders
    train_dl = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_split, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dl, valid_dl, test_dl, test_slide_names

def main():
    ## python MLP.py --feature_extractor='resnet18-ssl' --n_iter=2
    parser = argparse.ArgumentParser(description='Compute deep features from tiles/nucleus')
    parser.add_argument('--feature_extractor', default='resnet18-ssl', type=str, help='feature extractor (resnet18, resnet18-ssl, vgg11, densenet121, morphology)')
    parser.add_argument('--image_source', default='nucleus', type=str, help='tile or nucleus')
    parser.add_argument('--n_iter', default=1, type=int, help='cross validation (1~5)')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size of dataloader')
    args = parser.parse_args()

    model_weight_dir = './MLP/model_weights'
    result_dir = './MLP/metrics'
    if not os.path.exists(model_weight_dir):
        os.makedirs(model_weight_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    feature_df = pd.read_csv("./slide_level_features/CRC_{}_{}.csv".format(args.image_source, args.feature_extractor))
    num_features = len(feature_df.columns)-1
    label_df = pd.read_csv('./slide_level_features/CRC_AS_labels.csv')
    concat_df = pd.concat([label_df, feature_df], axis=1)
    concat_df = concat_df.dropna(axis=0, subset=['AS'])
    concat_df.reset_index(drop=True, inplace=True)

    train_dl, valid_dl, test_dl, test_slide_names = prepare_dataset(concat_df, num_features, args.n_iter, args.batch_size) 
    model = MLP(num_features) 
    save_path = os.path.join(model_weight_dir, '{}_{}_AS_fold_{}.pth'.format(args.image_source, args.feature_extractor, args.n_iter))
    train_model(train_dl, valid_dl, model, save_path=save_path, epochs=2000, lr=0.00005) 

    #### test
    model = torch.load(save_path)
    results = evaluate_model(test_dl, model, beta=1)
    model_metrics = results[0]
    model_preds = results[1]
    actuals = results[2]
    scores = results[3]

    metrics_df = pd.DataFrame.from_dict(model_metrics, orient='index', columns=['metric'])
    metrics_df.index.name = 'metric_type'
    metrics_df.reset_index(inplace=True)
    metrics_df.to_csv(os.path.join(result_dir, '{}_{}_AS_test_metric_fold_{}.csv'.format(args.image_source, args.feature_extractor, args.n_iter)), index=False) 
    predict_df = pd.DataFrame({'slide_name': test_slide_names, 'actuals':actuals.ravel(), 'preds':model_preds.ravel(), 'scores':scores.ravel()})
    predict_df.to_csv(os.path.join(result_dir, '{}_{}_AS_test_predict_fold_{}.csv'.format(args.image_source, args.feature_extractor, args.n_iter)), index=False) 

if __name__ == '__main__':
    main()
