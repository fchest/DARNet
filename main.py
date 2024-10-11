import argparse
import logging
import os

from tqdm import tqdm


from model import DARNet
from utils import Setup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dotmap import DotMap
from myutils import *


import torch

import torch.nn as nn
from torch.optim import Adam


result_logger = logging.getLogger('result')
result_logger.setLevel(logging.INFO)

config = dict()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initiate(args, train_loader, valid_loader, test_loader, subject):
    model = DARNet(config)

    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(params=model.parameters(), lr=0.0005, weight_decay=3e-4)


    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion}

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)


def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']

    def train(model, optimizer, criterion):
        model.train()
        train_acc_sum = 0
        train_loss_sum = 0
        batch_size = train_loader.batch_size

        # for x,y in train_loader.dataset:
        #     print(x.shape,y)
        for i_batch, batch_data in enumerate(train_loader):
            train_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            train_data, train_label = train_data.cuda(), train_label.cuda()
            preds = model(train_data)
            # Forward pass

            loss = criterion(preds, train_label.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            with torch.no_grad():
                train_loss_sum += loss.item() * batch_size
                predicted = preds.data.max(1)[1]
                train_acc_sum += predicted.eq(train_label).cpu().sum()

        return train_loss_sum / len(train_loader.dataset), train_acc_sum / len(train_loader.dataset)

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = len(test_loader)
        else:
            loader = valid_loader
            num_batches = len(valid_loader)
        total_loss = 0.0
        test_acc_sum = 0
        proc_size = 0
        batch_size = loader.batch_size
        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                test_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                test_data, test_label = test_data.cuda(), test_label.cuda()
                preds = model(test_data)
                proc_size += batch_size

                # Backward and optimize
                optimizer.zero_grad()

                total_loss += criterion(preds, test_label.long()).item() * batch_size
                preds = preds.detach()
                predicted = preds.data.max(1)[1]  # 32
                # label = test_label.max(1)[1]
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        avg_loss = total_loss / (num_batches * batch_size)

        avg_acc = test_acc_sum / (num_batches * batch_size)

        return avg_loss, avg_acc

    epochs_without_improvement = 0
    best_epoch = 1
    best_valid = float('inf')
    # for epoch in range(1, args.max_epoch + 1):
    for epoch in tqdm(range(1, args.max_epoch + 1), desc='Training Epoch', leave=False):
        train_loss, train_acc = train(model, optimizer, criterion)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print()
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Train Acc {:5.4f} | Valid Loss {:5.4f} | Valid Acc '
            '{:5.4f}'.format(
                epoch,
                args.name,
                train_loss,
                train_acc,
                val_loss,
                val_acc))

        if val_loss < best_valid:
            best_valid = val_loss
            epochs_without_improvement = 0

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > 10:
                break

    model = load_model(args, name=args.name)

    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    return test_loss, test_acc


def main(name="S1", time_len=1, dataset="DTU"):
    setup_seed(42)
    print(name)
    args = DotMap()
    args.name = name
    args.max_epoch = 100
    args.random_seed = 1234
    args.both_feature = False
    train_loader, valid_loader, test_loader = getData(name, time_len, dataset)
    config['Data_shape'] = train_loader.dataset.data.shape
    print('Data shape:', config['Data_shape'])

    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    print(loss, acc.item())

    info_msg = f'{dataset}_{name}_{str(time_len)}s loss:{str(loss)} acc:{str(acc.item())}'
    result_logger.info(info_msg)
    return loss, acc


if __name__ == "__main__":
    file_handler = logging.FileHandler('log/result.log')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    result_logger.addHandler(file_handler)

    dataset = 'DTU'
    config['time_len'] = 1
    for i in range(1, 19):
        name = 'S' + str(i)
        time_len = config['time_len']
        main(name=name, time_len=time_len, dataset=dataset)
