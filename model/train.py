X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# torch의 Dataset 을 상속.
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
    

batch_size = 512

train_dataset = TensorData(X_train, y_train)
train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorData(X_test, y_test)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=len(X_test), shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def change_hiddim(hid_dim,model_check, output_dim, epochs,learingrate): 
    if model_check == 'lstm' : 
        model = LSTM(input_size, hid_dim, output_dim).to(device)
    elif model_check =='lstm_modified':
        model =  LSTM_Modified(input_size, hid_dim, output_dim).to(device)        
    elif model_check =='gru':
        model =  GRU(input_size, hid_dim, output_dim).to(device) 
    elif model_check =='gru_modified':
        model = modified_GRU(input_size, hid_dim, output_dim).to(device) 
    
    #define optimizer and loss function 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learingrate) #,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    # optimizer = torch.optim.SGD(mo/del.parameters(),lr = learingrate,wei
    model.train()
    start_time, end_time, total_time = 0.0,0.0,0.0
    start_time = time.time()
    n = len(train_data_loader) 
    best_acc = 0 
    for epoch in range(epochs):
        train_loss = 0.0 # 한 에폭이 돌 때 그안에서 배치마다 loss가 나온다. 즉 한번 학습할 때 그렇게 쪼개지면서 loss가 다 나오니 MSE를 구하기 위해서 사용한다.
        correct =0
        total=0
        for i, (currx, labels) in enumerate(train_data_loader):
            currx = currx.to(device)
            # print(currx.shape)
            labels = labels.to(device)
            # Clear gradients w.r.t. parameters
            model.zero_grad()
            optimizer.zero_grad()
            if i<= int(len(train_data_loader)*0.3):
                currx, targets_a, targets_b, lam = cutmix(currx, labels, alpha=0.5)
                #forward pass
                output = model(currx)                
                loss = lam * criterion(output, targets_a) + (1. - lam) * criterion(output, targets_b)
                # loss = criterion(output, labels) 
            else: 
                #forward pass
                output = model(currx)
                loss = criterion(output, labels) 
            # # #forward pass
            # output = model(currx)
            # loss = criterion(output, labels) 
            # 역전파 & 최적화        
            loss.backward()
            optimizer.step()        
            train_loss += loss.item() # epoch 마다 평균 loss를 계산하기 위해 배치 loss를 더한다.
            _, labels_index=torch.max(labels,1)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += predicted.eq(labels_index).sum().item()
            if i + 1 == n :
                print("epoch",epoch+1,"loss: ","{:.3f}".format(train_loss/(i+1)))
                print('Train Accuracy : {:.3f} %'.format(100 * correct / total))
        scheduler.step()
        with torch.no_grad():    

            test_correct = 0
            test_total = 0
            for currx, labels in test_data_loader:           
                currx = currx.to(device) 
                labels = labels.to(device)
                _, labels_index=torch.max(labels,1)
                outputs= model(currx)
                _, predicted = torch.max(outputs, 1) # logit(확률)이 가장 큰 class index 반환        
                test_total += labels.size(0)
                test_correct += (predicted == labels_index).sum().item()             
            print('Test Accuracy : {:.3f} %'.format(100 * test_correct / test_total)) 
        if best_acc < 100 * test_correct / test_total:
            best_acc = 100 * test_correct / test_total
            curr_epoch = epoch
            if best_acc>=80 and curr_epoch>=99:
                torch.save(model.state_dict(), f'best_model_{model_check}_{curr_epoch}.pth')
            print("best_acc is updated as","{:.2f}".format(best_acc),"%")
    end_time = time.time()
    total_time = end_time - start_time
    print(total_time)
    return model ,best_acc,curr_epoch,total_time

epoch,l_r =500, 1e-3
model2 ,best,best_epoch,times= change_hiddim(300,'gru_modified',6,epoch,l_r) 
print(best,best_epoch,times) 
