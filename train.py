import os
import logging
import sys
import itertools
import time 

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model import MatchPrior, SSD
from multibox_loss import MultiboxLoss
from config import *
from data_transform import TrainAugmentation, TestTransform
from dataset import VOCDataset


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = 'cpu'
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    classes = ['motorcycle', 'car', 'bus', 'truck']
    num_classes = len(classes)+1
    dataset_path = '../data_0612'
    validation_dataset_path = '../data_0612'
    batch_size = 2
    num_workers = 1
    num_epochs = 100
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    validation_epochs = 5
    pretrained_ssd_path = './models/vgg16-ssd-Epoch-170-Loss-1.8997838258743287.pth'
    base_net_path =''
    checkpoint_folder = 'models/'
    image_size = 300
    image_mean = np.array([123, 117, 104])  # RGB layout
    image_std = 1.0
    specs = [
        SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
        SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
        SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
        SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
        SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
        SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
    ]
    priors = generate_ssd_priors(specs, image_size)
    last_epoch = -1
    train_transform = TrainAugmentation(image_size, image_mean, image_std)
    target_transform = MatchPrior(priors, center_variance,
                                  size_variance, 0.5)

    test_transform = TestTransform(image_size, image_mean, image_std)

    logging.info("Prepare training datasets.")
    train_dataset = VOCDataset(dataset_path, transform=train_transform,
                                target_transform=target_transform)

        
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(validation_dataset_path, transform=test_transform,
                                target_transform=target_transform, is_test=True)
    
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = SSD(num_classes)
    min_loss = -10000.0

    params = [
        {'params': net.base_net.parameters(), 'lr': lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]
    time_start = time.time()
    if base_net_path!='':
        logging.info(f"Init from base net {base_net_path}")
        net.init_from_base_net(base_net_path)
    elif pretrained_ssd_path != '':
        logging.info(f"Init from pretrained ssd {pretrained_ssd_path}")
        net.init_from_pretrained_ssd(pretrained_ssd_path)
    
    logging.info(f'Took {(time.time() - time_start):.2f} seconds to load the model.')

    net.to(DEVICE)

    criterion = MultiboxLoss(priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                weight_decay=weight_decay)
    logging.info(f"Learning rate: {lr}")

    logging.info("Uses MultiStepLR scheduler.")
    scheduler = MultiStepLR(optimizer, milestones=[120,160])
    

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, epoch=epoch)
        
        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(checkpoint_folder, f"{net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
