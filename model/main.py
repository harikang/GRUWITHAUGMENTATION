from train_test_split import train_data_loader, test_data_loader, device
from model import change_hiddim
from preprocessing import dataloader
def main(X,y):
  train_data_loader, test_data_loader, device = dataloader(X,y)
  epochs = 500
  learning_rate = 1e-3
  num_classes  = 6
  
  model, best_acc, best_epoch, training_time = change_hiddim(300, 'gru_modified', num_classes, epochs, learning_rate)
  print(f"Best Accuracy: {best_acc:.2f} %, Epoch: {best_epoch}, Time: {training_time:.2f} seconds")
