import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as ts

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DanQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(4,8) # in original DanQ the input is one-hot encoded, aka not learnable
        self.conv = nn.Conv1d(8,320,26,padding=0)
        self.pool = nn.MaxPool1d(13,13)
        self.drop1= nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(320,320,bidirectional=True)
        self.drop2= nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(640*75,928)
        self.fc2 = nn.Linear(928,919)
        
    def forward(self,x):                  #(bs, 1000)
        x = self.emb(x)                   #(bs, 1000, 8)
        #Conv1d
        x = x.permute(0,2,1).contiguous() #(bs, 8, 1000)
        x = F.relu(self.conv(x))          #(bs, 320, 975)
        x = self.drop1(self.pool(x))      #(bs, 320, 75)
        #LSTM
        x = x.permute(0,2,1).contiguous() #(bs, 75, 320)
        x,h = self.lstm(x)                #(bs, 75, 640)
        x = self.drop2(x)
        #FC
        x = x.view(x.shape[0],-1)         #(bs,75*640)
        x = F.relu(self.fc1(x))           #(bs,928)
        x = self.fc2(x)                   #(bs,919)
        return x # return logits instead of sigmoid


class ConvBN(nn.Module):
    # https://arxiv.org/pdf/1603.05027.pdf 
    def __init__(self,inp_dim,out_dim,ks=1,pad=0,stride=1):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp_dim)
        self.cv = nn.Conv1d(inp_dim,out_dim, kernel_size=ks, padding=pad,stride=stride,bias=False)
    def forward(self, x): return self.cv(F.relu(self.bn(x)))
    
    
class ResBlock1D(nn.Module):
    def __init__(self,out_dim,first_stride=2,k=1):
        super().__init__()
        width = int(k * out_dim//4) # https://arxiv.org/abs/1605.07146
        self.cvbn1 = ConvBN(out_dim//2, width,stride=first_stride) #(bs,inp_dim,seq_len)
        self.cvbn2 = ConvBN(width,      width,3,1)
        self.cvbn3 = ConvBN(width,      out_dim)
        self.id    = ConvBN(out_dim//2, out_dim,stride=first_stride)
        
    def forward(self,inp):
        x = self.cvbn1(inp)
        x = self.cvbn2(x)
        x = self.cvbn3(x)
        return x + self.id(inp)
         
    
class Resnet1D(nn.Module):
    def __init__(self,n_blocks,emb_dim,d_model,k=1):
        super().__init__()
        self.out_seq_len = int(1000/2**n_blocks)
        assert d_model%2**n_blocks==0                           # n_blocks = 3
                                                                # (bs,8,1000) input
        layers =[nn.Conv1d(emb_dim,d_model//2**n_blocks,7,1,3,bias=False)] # (bs,64,1000)

        for i in range(n_blocks,0,-1):
            out_dim = d_model//2**(i-1)
            layers.append(ResBlock1D(out_dim,k=k))                  # (bs, d_model//2**(i-1), seq_len/i)
        self.layers = nn.Sequential(*layers)
        self.RBN = nn.Sequential(nn.BatchNorm1d(d_model),nn.ReLU(inplace=True))
    def forward(self, x): return self.RBN(self.layers(x))
    
    
class ResTransXLHead(nn.Module):
    def __init__(self,inp_dim,hid_dim=512,p=0.3,WVN=False):
        super().__init__()
        layers = []
        if hid_dim!=0:
            layers = [ 
                nn.Linear(inp_dim,hid_dim,bias=False),
                nn.BatchNorm1d(hid_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p)
            ]
        else: hid_dim = inp_dim

        self.fc_end = nn.Linear(hid_dim,919)
        self.init_umbalanced()
        layers.append(self.fc_end)
        self.layers = nn.Sequential(*layers)
        
        if WVN:
            #https://arxiv.org/pdf/1912.01857.pdf
            self.fc_end.register_backward_hook(self.__WVN__)

    def __WVN__(self, module, grad_input, grad_output):
        W = module.weight.data
        W_norm = W / torch.norm(W, p=2, dim=1, keepdim=True)
        module.weight.data.copy_(W_norm)
        
    def init_umbalanced(self):
        # https://arxiv.org/pdf/1901.05555.pdf
        pi = torch.tensor(2/100) #prob of positive class
        b = -torch.log((1-pi)/pi)
        nn.init.constant_(self.fc_end.bias,b)
        
    def forward(self, x): return self.layers(x) 
    
    
class ResTransXL(nn.Module):
    def __init__(self,vocab_size,d_emb, tsfm_cfg, n_res_blocks=3,res_k=1, res_p=0.2, 
                 skip_cnt=False, fc_h_dim=512,lin_p=0.3, WVN=False,LSTM=False,LSTM_p=0.5):
        super(ResTransXL, self).__init__()
        self.LSTM = LSTM
        self.emb = nn.Embedding(vocab_size,d_emb)
        self.res = Resnet1D(n_res_blocks,d_emb,tsfm_cfg.d_model,k=res_k)
        self.res_drop = nn.Dropout(res_p)
        self.cfg = tsfm_cfg
        if LSTM: 
            self.core = nn.LSTM(tsfm_cfg.d_model,tsfm_cfg.d_model,bidirectional=True,batch_first=True)
            self.core_drop = nn.Dropout(LSTM_p)
        else: self.core = ts.TransfoXLModel(tsfm_cfg)
        self.skip_cnt = skip_cnt
        self.lin_inp_dim = self.res.out_seq_len * tsfm_cfg.d_model *(1 + skip_cnt + LSTM)
        self.lm_head = ResTransXLHead(self.lin_inp_dim,fc_h_dim,p=lin_p,WVN=False)
        
    def summary(self):
        print(f'Model parameters:\t\t\t\t')
        print(f'Resnet part:\t\t{count_parameters(self.res)//1000}k')
        print(f'Transformer-XL part:\t{count_parameters(self.core)//1000}k')
        print(f'Linear part:\t\t{count_parameters(self.lm_head)//1000}k')
        print(f'Total:\t\t\t{count_parameters(self)//1000}k')
        
    def forward(self,x,mems=None):
        x = self.emb(x)                   #(bs, 1000, d_emb)
        x = x.permute(0,2,1).contiguous() #(bs, d_emb, 1000)
        
        x = self.res(x)                   #(bs, d_model,125)
        x = self.res_drop(x)
        x = x.permute(0,2,1).contiguous() #(bs, 125, d_model)
        
        if self.LSTM: 
            last_h, _ = self.core(x)            # (bs, seq_len, num_directions * hidden_size)
            last_h = self.core_drop(last_h)
        else: last_h,  mems = self.core(inputs_embeds=x,mems=mems) #(bs, 125, d_model)
            
        if self.skip_cnt:
            lin_inp = torch.cat([x,last_h],dim=-1)
            lin_inp = lin_inp.view(-1, self.lin_inp_dim)
        else:    
            lin_inp = last_h.view(-1, self.lin_inp_dim)
        out = self.lm_head(lin_inp)
        return out, mems
