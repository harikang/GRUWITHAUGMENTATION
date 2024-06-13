import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Custom Dataset Class
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
