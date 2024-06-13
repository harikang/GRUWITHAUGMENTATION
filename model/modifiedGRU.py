class modified_GRUCell(nn.Module) :
    def __init__(self, input_size, hidden_size, bias=True) :
        super(modified_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.xx2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def reset_parameters(self) :
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters() : 
            w.data.uniform_(-std, std)
            
    def forward(self, x, pastx, hidden) : 
        
        x = x.view(-1, x.size(1))
        pastx = x
        gate_xx = self.xx2h(pastx)
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        ii_r, ii_i, ii_n = gate_xx.chunk(3, 1)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        resetgate = self.sigmoid(i_r + h_r + ii_r)
        inputgate = self.sigmoid(i_i + h_i + ii_i)
        newgate = self.tanh(i_n + ii_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        return hy, pastx

class modified_GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gru_cell = modified_GRUCell(input_dim, hidden_dim, 1) # input dim : 52 , hid dim : 300 

        # linear
        self.fc1 = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # init x : [data, length, input_dim]
        x0 = Variable(torch.randn(x.size(0),x.size(1),x.size(2))).cuda()
        # init hidden [1, datasize, 52]
        h0 = Variable(torch.randn(1, x.size(0), self.hidden_dim)).cuda() 
        
        hn = h0[0,:,:]
        pastx = x0
        for seq in range(x.size(1)) :
           hn, pastx= self.gru_cell(x[:, seq, :],pastx, hn)           
        out = self.fc1(hn)
        return out
