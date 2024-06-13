# main.py

from train_test_split import change_hiddim

epochs = 500
learning_rate = 1e-3

model, best_acc, best_epoch, training_time = change_hiddim(300, 'gru_modified', num_classes, epochs, learning_rate)
print(f"Best Accuracy: {best_acc:.2f} %, Epoch: {best_epoch}, Time: {training_time:.2f} seconds")
