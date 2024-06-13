def addgaussian(train_data,train_label): # #Gaussian noise add
  noise_mu, noise_sigma = 0, 0.0001  
                
  #X
  noisetrain1 = np.random.normal(noise_mu, noise_sigma, train_data.reshape(-1,52).shape)
  noisetrain2 = np.random.normal(noise_mu, noise_sigma, train_data.reshape(-1,52).shape)
  noisetrain3 = np.random.normal(noise_mu, noise_sigma, train_data.reshape(-1,52).shape)
  noisy_data1 = train_data.reshape(-1,52) + train_data.reshape(-1,52)*noisetrain1
  noisy_data2 = train_data.reshape(-1,52) + train_data.reshape(-1,52)*noisetrain2
  noisy_data3 = train_data.reshape(-1,52) + train_data.reshape(-1,52)*noisetrain3
  noisy_data_tensor1 = torch.tensor(noisy_data1.reshape(-1,192,52), dtype=torch.float32)
  noisy_data_tensor2 = torch.tensor(noisy_data2.reshape(-1,192,52), dtype=torch.float32)
  noisy_data_tensor3 = torch.tensor(noisy_data3.reshape(-1,192,52), dtype=torch.float32)
  train_data_tensor = torch.tensor(train_data)
  X_train_gn = torch.cat([train_data_tensor,noisy_data_tensor1,noisy_data_tensor2,noisy_data_tensor3],0)
  
  #y  
  y_train_en = torch.Tensor(y_train_en)
  y_train_gn = torch.cat([y_train_en,y_train_en,y_train_en,y_train_en],0)
  return X_train_gn, y_train_gn
