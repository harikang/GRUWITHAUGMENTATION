#cutmix를 추가 
def cutmix(data, target, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target = target[indices]

    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    cut_ratio = 1. - lam
    cut_size = int(data.size(1) * cut_ratio)
    start = torch.randint(0, data.size(1) - cut_size + 1, (1,)).item()

    data[:, start:start+cut_size, :] = shuffled_data[:, start:start+cut_size, :]

    lam = 1 - cut_size / data.size(1)

    return data, target, shuffled_target, lam
