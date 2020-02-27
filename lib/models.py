# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/3. Model design.ipynb (unless otherwise specified).

__all__ = ['count_parameters', 'DanQ', 'ConvBN', 'SEblock', 'ResBlock1D', 'Resnet1D', 'NoRNN', 'BiLSTM', 'BERT',
           'TransXL', 'ClsfHead', 'ResSeqLin']

# Cell
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as ts

# Cell
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Cell
class DanQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_amp = True
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

# Cell
class ConvBN(nn.Module):
    # https://arxiv.org/pdf/1603.05027.pdf
    def __init__(self,inp_dim,out_dim,ks=1,pad=0,stride=1,p=0,C=1):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp_dim)
        self.cv = nn.Conv1d(inp_dim,out_dim, kernel_size=ks, padding=pad, stride=stride,bias=False,groups=C)
        self.r  = nn.ReLU(inplace=True)
        self.p = p
        if p > 0: self.drop = nn.Dropout2d(p)

    def forward(self, x):
        x = self.r(self.bn(x))
        if self.p > 0: x = self.drop(x)
        return self.cv(x)

class SEblock(nn.Module):
    # https://arxiv.org/pdf/1709.01507.pdf
    def __init__(self,C,r=8):
        super().__init__()
        self.fc1 = nn.Linear(C,C//r)
        self.fc2 = nn.Linear(C//r,C)
        self.r   = nn.ReLU(inplace=True)

    def forward(self,inp):
        x = inp.mean(dim=-1)
        x = self.r(self.fc1(x))
        x = self.r(self.fc2(x))
        x = torch.sigmoid(x).unsqueeze(-1)
        return x * inp

class ResBlock1D(nn.Module):
    def __init__(self,inp_dim,out_dim, bn_ks=3,first_stride=2,k=0,p=0,C=1,d=0,lvl=0,SE=True):
        super().__init__()
        if k>0:
            #Wide Resnet https://arxiv.org/abs/1605.07146
            assert d==0
            width = int(k * out_dim//4)
        elif d>0:
            # ResNext https://arxiv.org/pdf/1611.05431.pdf
            assert k==0
            width = C*d*2**lvl
        else:
            width = out_dim//4

        self.cvbn1 = ConvBN(inp_dim, width,stride=first_stride) #(bs,inp_dim,seq_len)
        self.cvbn2 = ConvBN(width,   width, bn_ks,pad=bn_ks//2,C=C)
        self.cvbn3 = ConvBN(width,   out_dim,p=p)
        self.SE    = SEblock(out_dim) if SE else None
        self.id    = ConvBN(inp_dim, out_dim,stride=first_stride)

    def forward(self,inp):
        x = self.cvbn1(inp)
        x = self.cvbn2(x)
        x = self.cvbn3(x)
        if self.SE: x = self.SE(x)
        return x + self.id(inp)


class Resnet1D(nn.Module):
    def __init__(self,n_blocks,emb_dim,d_model,k=1,p=0,block_stride=None, C=1, d=0, bn_ks=3):
        super().__init__()
        if not block_stride: block_stride=[2]*n_blocks
        n_dsmpl = (torch.tensor(block_stride)==2).sum().item()
        assert d_model%2**n_dsmpl==0
        self.out_seq_len = int(1000/2**n_dsmpl)
                                                              # n_blocks = 6
        out_dim = d_model//2**n_dsmpl                         # (bs,emb_dim,1000) input
        inp_dim = max(emb_dim,out_dim)
        layers =[nn.Conv1d(emb_dim,inp_dim,7,1,3,bias=False)] # (bs,44,1000)

        lvl=0
        for i in range(n_blocks):
            if  block_stride[i]==2: lvl+=1
            out_dim *= block_stride[i]
            # (bs, d_model//2**(i-1), seq_len/i)
            layers.append(ResBlock1D(inp_dim,out_dim, bn_ks,block_stride[i], k, p, C, d,lvl))
            inp_dim = out_dim

        self.layers = nn.Sequential(*layers)
        self.RBN = nn.Sequential(nn.BatchNorm1d(d_model),nn.ReLU(inplace=True))
        self.res_drop = nn.Dropout(p)

    def forward(self, x): return self.res_drop(self.RBN(self.layers(x)))

class NoRNN(nn.Module):
    #dummy model to test the effect of recurrnce part of net
    def __init__(self,d_model):
        super().__init__()
        self.use_amp = True
        self.d_model = d_model

    def forward(self,x):  return x

class BiLSTM(nn.Module):
    def __init__(self,d_model,p=0.5, bidir=True):
        super().__init__()
        self.use_amp = True
        self.d_model = d_model
        self.core = nn.LSTM(d_model, d_model, bidirectional=bidir,batch_first=True)
        self.ln   = nn.LayerNorm([125, d_model])
        self.drop = nn.Dropout(p)

    def forward(self,x):
        last_h, _ = self.core(x)            # (bs, seq_len, num_directions * d_model)
        last_h = last_h[:, :, :self.d_model] + last_h[:, :, self.d_model:]
        last_h = self.drop(self.ln(last_h))
        return last_h


class BERT(nn.Module):
    def __init__(self,d_model, n_layer, n_head):
        super().__init__()
        self.use_amp = True
        cfg = ts.BertConfig(vocab_size=4, hidden_size=d_model, num_hidden_layers=n_layer,
                            num_attention_heads=n_head, intermediate_size=256)

        self.tsfm = ts.BertModel(cfg)
        self.d_model = d_model

        self.tsfm.embeddings.word_embeddings.weight.requires_grad = False
        self.tsfm.pooler.dense.weight.requires_grad = False

    def forward(self,x):
        last_h,  _ = self.tsfm(inputs_embeds=x)
        return last_h


class TransXL(nn.Module):
    def __init__(self,d_model, n_layer, n_head, d_head, d_inner):
        super().__init__()
        self.use_amp = False
        cfg = ts.TransfoXLConfig(
            vocab_size=4, d_embed=8, d_model=d_model, n_head=n_head, d_head=d_head,
            d_inner=d_inner,n_layer=n_layer, tgt_len=0, ext_len=0, mem_len=256, cutoffs=[1])

        self.tsfm = ts.TransfoXLModel(cfg)
        self.d_model = d_model

        self.tsfm.word_emb.emb_layers[0].weight.requires_grad = False
        self.tsfm.word_emb.emb_layers[1].weight.requires_grad = False
        self.tsfm.word_emb.emb_projs[0].requires_grad = False
        self.tsfm.word_emb.emb_projs[1].requires_grad = False

    def forward(self,x, mems=None):
        last_h,  mems = self.tsfm(inputs_embeds=x,mems=mems)
        return last_h

class ClsfHead(nn.Module):
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

        self.fc = nn.Sequential(*layers)
        self.fc_end = nn.Linear(hid_dim,919)
        self.init_umbalanced()

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

    def forward(self, x): return self.fc_end(self.fc(x))

class ResSeqLin(nn.Module):
    def __init__(self,vocab_size,d_emb, seq_model, hot=False,
                 n_res_blocks=3, res_k=0,res_p=0.3, block_stride=None, C=1, d=0, bn_ks=3,
                 skip_cnt=False, fc_h_dim=512,lin_p=0.3, WVN=False):
        super(ResSeqLin, self).__init__()
        # Embedding
        if not hot:
            self.emb = nn.Embedding(vocab_size,d_emb)
            self.emb_ln = nn.LayerNorm([1000, d_emb])
        self.hot = hot
        #Resnet
        self.res = Resnet1D(n_res_blocks,d_emb,seq_model.d_model,res_k,res_p,
                            block_stride, C, d, bn_ks)
        # Sequence
        self.seq_model = seq_model
        # Linear
        self.skip_cnt = skip_cnt
        self.lin_inp_dim = self.res.out_seq_len * seq_model.d_model *(1 + skip_cnt)
        self.head = ClsfHead(self.lin_inp_dim,fc_h_dim,p=lin_p,WVN=False)

    def summary(self):
        print(f'Model parameters:\t\t\t\t')
        print(f'Resnet part:\t{count_parameters(self.res)//1000}k')
        print(f'Sequence part:\t{count_parameters(self.seq_model)//1000}k')
        print(f'Linear part:\t{count_parameters(self.head)//1000}k')
        print(f'Total:\t\t{count_parameters(self)//1000}k')
        return f'{count_parameters(self)/1e6:.2f}'+'m'

    def forward(self,x,mems=None):
        if not self.hot:
            x = self.emb_ln(self.emb(x))      #(bs, 1000, d_emb)
            x = x.permute(0,2,1).contiguous() #(bs, d_emb, 1000)

        x = self.res(x)                       #(bs, d_model,125)
        x = x.permute(0,2,1).contiguous()     #(bs, 125, d_model)

        last_h = self.seq_model(x)            #(bs, 125, d_model)

        if self.skip_cnt:
            lin_inp = torch.cat([x,last_h],dim=-1)
            lin_inp = lin_inp.view(-1, self.lin_inp_dim)    #(bs, 125*2*d_model)
        else:
            lin_inp = last_h.reshape(-1, self.lin_inp_dim)  #(bs, 125*d_model)

        out = self.head(lin_inp)                         #(bs, 919)
        return out, mems