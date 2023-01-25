DATA_DIR =  './pizzaGANdata'

import pandas as pd
import numpy as np
import os
import time
from PIL import Image

from tqdm import tqdm

import seaborn as sns
sns.set_style('darkgrid')

#import shutil

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# scikitlearn
from sklearn.metrics import multilabel_confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Pytorch
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim
# from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn as nn


import sys, select

#!pip install torchmetrics
#from torchmetrics.classification import MultilabelF1Score, MultilabelAccuracy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("cuda:0" if torch.cuda.is_available() else "cpu")


# ===================================== READ DATA
print("Reading data")


y_all = np.loadtxt(os.path.join(DATA_DIR, 'imageLabels.txt'))[:2000]
x_all = np.arange(y_all.shape[0])

x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=123)

print(y_train.shape)
print(y_test.shape)

print(x_train.shape)
print(x_test.shape)

# ===================================== LOAD TRAIN IMAGES

train_data_path=os.path.join(DATA_DIR, '')


img_all = []

for idx in tqdm(x_all):
	img_name = "{:05d}.jpg".format(idx+1)
	x = Image.open(os.path.join(train_data_path, 'images', img_name))
	img_all.append(np.array(x.resize((500, 500))))


class myDataset(Dataset):
    """Pizza dataset"""
    
    def __init__(self, x_idx, y, img_path, img_data = None, transform=None):
      """
      Args:
      """
      self.x_idx = x_idx
      self.y = y
      self.img_path = img_path
      self.transform = transform
      self.img_data = img_data     
    
    def __getitem__(self, idx):
      if isinstance(self.img_data,np.ndarray):
        x = Image.fromarray(self.img_data[idx,:,:,:])
      else:
        img_name = "{:05d}.jpg".format(idx+1)
        x = Image.open(os.path.join(self.img_path, img_name))
      y = self.y[idx,:]
      if self.transform:
          x = self.transform(x)
      y = np.int64(y)
      return x, y
            
    def __len__(self):
        return int(len(self.x_idx))


# ===================================== CREATE DATASETs

print("creating datasets")

batch_size = 5

input_size = 500

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.RandomVerticalFlip(),
                                         transforms.ColorJitter(brightness=32. / 255., saturation=0.5),
                                         transforms.Resize((input_size, input_size)),
                                         transforms.ToTensor(),
                                     normalize])

val_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                   normalize])
        
kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}


img_data=np.array(img_all)

# TRAIN DATA
train_data_path=img_path = os.path.join(DATA_DIR, 'images')
img_data_train = img_data[x_train,:,:,:]

train_set_raw = myDataset(x_train, y_train, img_path=train_data_path, img_data=img_data_train, transform = train_transform)
train_dataloader = DataLoader(train_set_raw, batch_size=batch_size, shuffle=True, **kwargs)

# TEST DATA
test_data_path=img_path = os.path.join(DATA_DIR, 'images')
img_data_test = img_data[x_test,:,:,:]

test_set_raw = myDataset(x_test, y_test, img_path=test_data_path, img_data=img_data_train, transform = val_transform)
test_dataloader = DataLoader(test_set_raw, batch_size=batch_size, shuffle=False, **kwargs)


# ========================= NETWORK

nlabel = y_all.shape[1]

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

network = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
num_ftrs = network.classifier[1].in_features
network.classifier[1]= nn.Linear(num_ftrs, nlabel)

# Name of the network
tag = "efficientNet"

# Read the last learned network (if stored)
if (os.path.exists(os.path.join("networks", 'network_real_{:s}.pth'.format(tag)))):
    print('Loading from file')
    network = torch.load(os.path.join("networks", 'network_real_{:s}.pth'.format(tag)))

# ======================================= TRAIN LOOP

#from sklearn.metrics import confusion_matrix
from IPython.display import clear_output
import copy
import random


def train_model_multilabel(model, nlabel, trainloader, criterion, optimizer, scheduler, num_epochs=5):
  # list for saving accuracies
  for epoch in range(num_epochs): # on va iterer sur toutes les données num_epochs fois
      print("Epoch {}".format(epoch))
      model.train()

      for inputs, targets in trainloader: # on itère sur les données par batch de batch_size (= 25)
          inputs, targets = inputs.cuda(),targets.cuda() # 25x3x224x224 et 25x10

          predictions = model(inputs)    ## on les fait rentrer dans le réseau
          targets = targets.to(torch.float) # FloatTensor needed

          loss = criterion(predictions,targets)    ## on compare la sortie courante à la sortie voulue
          optimizer.zero_grad() ## supprime les gradients courants
          loss.backward() ## le gradient -- la magie par rapport à comment c'était long en cours :-)
          optimizer.step() ## on actualise les poids pour que la sortie courante soit plus proche que la sortie voulue

          if random.randint(0,90)==0:
              print("\tloss=",loss) ## on affiche pour valider que ça diverge pas

      # Learning step

  return model


# ======================================= evaluation

def model_evaluation(network, nb_labels, dataloader, labels=None, display=0):

  # set the model to evaluation mode
  network.eval()

  # create the vectors necessary for KPI
  perf_label_test = np.zeros((1,nb_labels))
  all_eval_pred = np.zeros(shape=(0,nb_labels))
  all_eval_targets = np.zeros((0,nb_labels))


  # tell not to reserve memory space for gradients (much faster)
  with torch.no_grad():
      for inputs, targets in tqdm(dataloader, ncols=80):

          inputs = inputs.to(device)
          targets = targets.to(device)

          # compute outputs
          outputs = network(inputs)
          outputs_np = outputs.cpu().detach().numpy()
          targets_np = targets.cpu().detach().numpy()

          # compute the predictions
          pred = (outputs_np > 0)

          # Rq : multilabel confusion matrix does not seem to be useful as it uses a one-vs-rest representation for each class

          # concatenate pred and targets to calculate the classification report
          all_eval_pred = np.concatenate((all_eval_pred, pred))
          all_eval_targets = np.concatenate((all_eval_targets, targets_np))
          # update the performance
          perf_label_test = perf_label_test + (targets_np == pred).sum(axis=0)

          # https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
          # https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report


  # Calculate KPI
  perf_label_test = perf_label_test / len(dataloader.dataset)
  metrics_report = classification_report(all_eval_targets, all_eval_pred, target_names=labels)

  # Prints the classification report and the performance (per label)
  if display==1:
    print("\nPerformance per label :",perf_label_test)
    print("Mean of performance :",sum(perf_label_test[0])/len(perf_label_test[0]))
    print(metrics_report)

  return perf_label_test, metrics_report


# ====================================================== TRAIN ROUTINE
topping_labels = ["Pepperoni", "Bacon", "Mushrooms", "Onions",
              "Peppers", "Black olives", "Tomatoes", "Spinach", "Fresh basil",
               "Arugula", "Broccoli", "Corn", "Pineapple"]

# Transfer network to GPU
network.to(device)

# Define learning components (to be used in the learning function)

optimizer = optim.Adam(network.parameters(), lr=0.00001)
criterion = nn.CrossEntropyLoss()
scheduler = None

i = 0

while True:
	max_epoch = 50
	
	# Learning
	network = train_model_multilabel(network, nlabel, train_dataloader, criterion, optimizer, scheduler, num_epochs=max_epoch)
	
	# print infos
	i += 1
	print(f"Total Epoch {i*max_epoch}")
	model_evaluation(network, len(topping_labels), test_dataloader, topping_labels, display=1)
	torch.save(network, os.path.join("networks", 'network_real_{:s}.pth'.format(tag)))
	
	# Leave training
	print("You have 1 second to answer!")
	should_stop, _, _ = select.select([sys.stdin], [], [], 1)
	if should_stop:
	  print("You said something. Stoping!")
	  break
	else:
	  print("You said nothing! Continuing!")

	

