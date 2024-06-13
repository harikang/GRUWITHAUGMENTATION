# train_test_split.py

from sklearn.model_selection import train_test_split
import torch

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Hyperparameters
sequence_length = 192
input_size = 52
num_classes = 6

# Custom Dataset Class
from torch.utils.data import DataLoader, Dataset

class TensorData(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.FloatTensor(y_data)
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# Data Loaders
batch_size = 512

train_dataset = TensorData(X_train, y_train)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorData(X_test, y_test)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Function
import time
import torch.nn as nn

def change_hiddim(hid_dim, model_check, output_dim, epochs, learning_rate):
    if model_check == 'lstm':
        model = LSTM(input_size, hid_dim, output_dim).to(device)
    elif model_check == 'lstm_modified':
        model = LSTM_Modified(input_size, hid_dim, output_dim).to(device)
    elif model_check == 'gru':
        model = GRU(input_size, hid_dim, output_dim).to(device)
    elif model_check == 'gru_modified':
        model = modified_GRU(input_size, hid_dim, output_dim).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    model.train()
    start_time = time.time()
    best_acc = 0 

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        for i, (currx, labels) in enumerate(train_data_loader):
            currx = currx.to(device)
            labels = labels.to(device)

            model.zero_grad()
            optimizer.zero_grad()

            if i <= int(len(train_data_loader) * 0.3):
                currx, targets_a, targets_b, lam = cutmix(currx, labels, alpha=0.5)
                output = model(currx)                
                loss = lam * criterion(output, targets_a) + (1. - lam) * criterion(output, targets_b)
            else:
                output = model(currx)
                loss = criterion(output, labels)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, labels_index = torch.max(labels, 1)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += predicted.eq(labels_index).sum().item()

            if i + 1 == len(train_data_loader):
                print(f"Epoch {epoch + 1}, Loss: {train_loss / (i + 1):.3f}")
                print(f'Train Accuracy: {100 * correct / total:.3f} %')
        
        scheduler.step()
        with torch.no_grad():
            test_correct = 0
            test_total = 0

            for currx, labels in test_data_loader:
                currx = currx.to(device)
                labels = labels.to(device)
                _, labels_index = torch.max(labels, 1)
                outputs = model(currx)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels_index).sum().item()

            test_acc = 100 * test_correct / test_total
            print(f'Test Accuracy: {test_acc:.3f} %')
            if best_acc < test_acc:
                best_acc = test_acc
                curr_epoch = epoch
                if best_acc >= 80 and curr_epoch >= 99:
                    torch.save(model.state_dict(), f'best_model_{model_check}_{curr_epoch}.pth')
                print(f"Best accuracy updated: {best_acc:.2f} %")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    return model, best_acc, curr_epoch, total_time
