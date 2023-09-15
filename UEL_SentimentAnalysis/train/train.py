import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import LSTMClassifier

def model_fn(model_dir):
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])
    
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    dict_size = model_info['vocab_size']
    print("**************************",dict_size, type(dict_size))
    
    dict_file = "word_dict.pkl"
    dict10k_file = "word_dict_10k.pkl"
    dict_name = str([dict_file if dict_size==5000 else dict10k_file][0])
    
    print("model_fn : Using Word Dict size of : {},from file : {}.".format(dict_size,dict_name))    
    word_dict_path = os.path.join(model_dir, dict_name)
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")
    
    train_file = "train.csv"
    train10k_file = "train_10k.csv"
    
    train_data_file = str([train_file if args.vocab_size==5000 else train10k_file][0])
    print("****** Reading : ",train_data_file)
    train_data = pd.read_csv(os.path.join(training_dir, train_data_file), header=None, names=None)
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
 
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model.forward(batch_X)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--embedding_dim', type=int, default=32, metavar='N',
                        help='size of the word embeddings (default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=100, metavar='N',
                        help='size of the hidden dimension (default: 100)')
    parser.add_argument('--vocab_size', type=int, default=5000, metavar='N',
                        help='size of the vocabulary (default: 5000)')

    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    # Build the model.
    model = LSTMClassifier(args.embedding_dim, args.hidden_dim, args.vocab_size).to(device)

    dict_file = "word_dict.pkl"
    dict10k_file = "word_dict_10k.pkl"
    dict_name = str([dict_file if args.vocab_size==5000 else dict10k_file][0])
    
    print("Using Word Dict size of : {},from file : {}.".format(args.vocab_size,dict_name))
  
    with open(os.path.join(args.data_dir, dict_name), "rb") as f:
        model.word_dict = pickle.load(f)

    print("Model loaded with embedding_dim {}, hidden_dim {}, vocab_size {}.".format(
        args.embedding_dim, args.hidden_dim, args.vocab_size
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters(),betas=(0.7,0.995),lr=0.006)
    loss_fn = torch.nn.BCELoss()

    train(model, train_loader, args.epochs, optimizer, loss_fn, device)


    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'vocab_size': args.vocab_size,            
        }
        torch.save(model_info, f)

    word_dict_path = os.path.join(args.model_dir, dict_name)
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
