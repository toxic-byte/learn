import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        assert d_model % num_heads ==0,"d_model must be divisible by num_heads"
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.linear_q=nn.Linear(d_model,d_model)
        self.linear_k=nn.Linear(d_model,d_model)
        self.linear_v=nn.Linear(d_model,d_model)
        self.linear_out=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(dropout)

        self.scale=math.sqrt(self.d_k)

    #query,key,value: [batch_size, seq_len, d_model]
    def forward(self,query,key,value,mask=None):
        batch_size=query.size(0)

        # 线性变换并分头[batch_size, num_heads, seq_len, d_k]
        Q=self.linear_q(query).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        K=self.linear_k(key).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        V=self.linear_v(value).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)

        #scores: [batch_size, num_heads, seq_len, seq_len]
        scores=torch.matmul(Q,K.transpose(-2,-1))/self.scale

        #后续做softmax时这些位置的权重接近于0
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)

        attention_weights=self.dropout(F.softmax(scores,dim=-1))

        #output: [batch_size, num_heads, seq_len, d_k]
        output=torch.matmul(attention_weights,V)

        #拼接多头[batch_size, seq_len, d_model]如果你调用了 transpose / permute，后面打算用 view，通常要先 contiguous()。​
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)

        return self.linear_out(output),attention_weights