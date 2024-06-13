def shifting(data, shift_steps, axis=1):
    # 3차원 데이터를 shift_steps만큼 이동시킵니다.
    shifted_data = np.roll(data, shift=shift_steps, axis=axis)
    return shifted_data
  
def augment_labels(labels, num_shifts):    
    return np.tile(labels, (num_shifts, 1))
  
augmented_data_fill = []
shifts = range(-10,10)

for shift in shifts:
    shifted_data_fill = shifting(X_train, shift)
    augmented_data_fill.append(shifted_data_fill)

# Combining all augmented data
augmented_data_fill = np.concatenate(augmented_data_fill, axis=0)

X_train_sh = augmented_data_fill

# Number of shifts is 20
num_shifts = 20

# Augmenting the labels
augmented_labels = augment_labels(y_train_en, num_shifts)

y_train_sh =augmented_labels
