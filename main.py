import os
import torch
from cqt_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *

# multi_size train
def multi_train(**kwargs):
    parallel = True
    opt.model = 'CQTNet'
    opt.notes='CQTNet'
    opt.batch_size=32
    #opt.load_latest=True
    #opt.load_model_path = ''
    opt._parse(kwargs)
    # step1: configure model

    model = getattr(models, opt.model)()
    if parallel is True:
        model = torch.nn.DataParallel(model)
    if parallel is True:
        if opt.load_latest is True:
            model.module.load_latest(opt.notes)
        elif opt.load_model_path:
            model.module.load(opt.load_model_path)
    else:
        if opt.load_latest is True:
            model.load_latest(opt.notes)
        elif opt.load_model_path:
            model.load(opt.load_model_path)
    model.to(opt.device)
    print(model)
    # step2: data

    train_data0 = CQT('train', out_length=200)
    train_data1 = CQT('train', out_length=300)
    train_data2 = CQT('train', out_length=400)
    val_data350 = CQT('songs350', out_length=None)
    val_data80 = CQT('songs80', out_length=None)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    val_datatMazurkas = CQT('Mazurkas', out_length=None)
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    val_dataloader350 = DataLoader(val_data350, 1, shuffle=False, num_workers=1)
    val_dataloaderMazurkas = DataLoader(val_datatMazurkas,1, shuffle=False,num_workers=1)
    #step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    lr = opt.lr
    if parallel is True:
        optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=opt.lr_decay,patience=2, verbose=True,min_lr=5e-6)
    #train
    best_MAP=0
    val_slow(model, val_dataloader350, -1)
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data0, label0),(data1, label1),(data2, label2) in tqdm(zip(train_dataloader0, train_dataloader1, train_dataloader2)):
            for flag in range(3):
                if flag==0:
                    data=data0
                    label=label0
                elif flag==1:
                    data=data1
                    label=label1
                else:
                    data=data2
                    label=label2
                # train model
                input = data.requires_grad_()
                input = input.to(opt.device)
                target = label.to(opt.device)

                optimizer.zero_grad()
                score, _ = model(input)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num += target.shape[0]
        running_loss /= num
        print(running_loss)
        if parallel is True:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        # update learning rate
        scheduler.step(running_loss)
        # validate
        MAP=0
        MAP += val_slow(model, val_dataloader350, epoch)
        MAP += val_slow(model, val_dataloader80, epoch)
        val_slow(model, val_dataloaderMazurkas, epoch)
        if MAP>best_MAP:
            best_MAP=MAP
            print('*****************BEST*****************')
        print('')
        model.train()


@torch.no_grad()
def multi_val_slow(model, dataloader1,dataloader2, epoch):
    model.eval()
    labels, features,features2 = None, None, None
    for ii, (data, label) in enumerate(dataloader1):
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels,label))
        else:
            features = feature
            labels = label
    for ii, (data, label) in enumerate(dataloader2):
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        if features2 is not None:
            features2 = np.concatenate((features2, feature), axis=0)
        else:
            features2 = feature

    features = norm(features+features2)
    dis2d = get_dis2d4(features)

    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(epoch, MAP, top10, rank1 )
    model.train()
    return MAP

@torch.no_grad()
def val_slow2(model, dataloader, epoch, name):
    model.eval()
    total, correct = 0, 0
    labels, features, levels = None, None, None

    for ii, (data, label, version, level) in enumerate(dataloader):
        # print(version, level)
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label))
            levels = np.concatenate((levels, level))
        else:
            features = feature
            labels = label
            levels = level
    features = norm(features)

    features_l0 = features[levels == 0]
    features_l1 = features[levels == 1]
    features_l2 = features[((levels == 1) | (levels == 2))]
    features_l3 = features[levels > 0]

    labels_l0 = labels[levels == 0]
    labels_l1 = labels[levels == 1]
    labels_l2 = labels[((levels == 1) | (levels == 2))]
    labels_l3 = labels[levels > 0]

    print(calc_rank(features_l1, features_l0, labels_l1, labels_l0))
    print(calc_rank(features_l2, features_l0, labels_l2, labels_l0))
    print(calc_rank(features_l3, features_l0, labels_l3, labels_l0))


def calc_rank(features, features_l0, labels, labels_l0):
    dis2d = -np.matmul(features, features_l0.T)
    ranks = []
    for i, row in enumerate(dis2d):
        ans = labels[i]
        l = labels_l0[np.argsort(row)]
        rank = list(l).index(ans)
        ranks.append(rank)
    return sum(np.array(ranks) < 10) / len(ranks)


@torch.no_grad()
def val_slow(model, dataloader, epoch, name):
    model.eval()
    total, correct = 0, 0
    labels, features = None, None

    for ii, (data, label, version, level) in enumerate(dataloader):
        input = data.to(opt.device)
        #print(input.shape)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label))
        else:
            features = feature
            labels = label
    features = norm(features)
    #dis2d = get_dis2d4(features)
    dis2d = -np.matmul(features, features.T) # [-1,1] Because normalized, so mutmul is equal to ED
    np.save('dis_%s.npy' % name, dis2d)
    np.save('label_%s.npy' % name, labels)
    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels,[100, 350])
    #elif len(labels) == 160:    MAP, top10, rank1 = calc_MAP(dis2d, labels,[80, 160])
    else :
        MAP, top10, rank1 = calc_MAP(dis2d, labels)

    print(epoch, MAP, top10, rank1 )
    model.train()
    return MAP




def test(**kwargs):
    opt.batch_size=1
    opt.num_workers=1
    opt.model = 'CQTNet'
    opt.load_latest = False
    opt.load_model_path = 'check_points/CQTNet.pth'
    opt._parse(kwargs)

    model = getattr(models, opt.model)()
    #print(model)
    if opt.load_latest is True:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # val_data350 = CQT('songs350', out_length=None)
    # val_data80 = CQT('songs80', out_length=None)
    val_audio = CQT('audio', out_length=None)
    # val_audio_l1 = CQT('audio_l1', out_length=None)
    # val_audio_l2 = CQT('audio_l2', out_length=None)
    # val_data = CQT('val', out_length=None)
    # test_data = CQT('test', out_length=None)
    # val_datatMazurkas = CQT('Mazurkas', out_length=None)
    # val_dataloader = DataLoader(val_data, 1, shuffle=False,num_workers=1)
    # test_dataloader = DataLoader(test_data, 1, shuffle=False,num_workers=1)
    # val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    val_dataloaderaudio = DataLoader(val_audio, 1, shuffle=False, num_workers=1)
    # val_dataloaderaudio_l1 = DataLoader(val_audio_l1, 1, shuffle=False, num_workers=1)
    # val_dataloaderaudio_l2 = DataLoader(val_audio_l2, 1, shuffle=False, num_workers=1)
    # val_dataloader350 = DataLoader(val_data350, 1, shuffle=False, num_workers=1)
    # val_dataloaderMazurkas = DataLoader(val_datatMazurkas,1, shuffle=False,num_workers=1)

    # val_slow(model, val_dataloader350, 0)
    # val_slow(model, val_dataloader80, 0, '80')
    val_slow2(model, val_dataloaderaudio, 0, 'audio')
    # val_slow(model, val_dataloaderaudio_l1, 0, 'audio_l1')
    # val_slow(model, val_dataloaderaudio_l2, 0, 'audio_l2')
    # val_slow(model, val_dataloaderMazurkas, 0)



if __name__=='__main__':
    import fire
    fire.Fire()
