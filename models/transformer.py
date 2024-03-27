import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) # shape of the head: [30, 8, 200, 200]
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask # transformation done for pytorch to allow broadcasting --> new shape: [8, 30, 200, 200]
        scaled = scaled.permute(1, 0, 2, 3) # return to original shape: [30, 8, 200, 200]
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length # 200
        self.d_model = d_model # 512

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float() # generates a list of even numbers up to d_model index --> that's the 2i from the formula in the paper (here instead of multiplying i by to up to len of d_model/1 -1, it just create a list of even numbers, which has the same efect)
        denominator = torch.pow(10000, even_i/self.d_model) # elevates a numebr to a tensor (in this case: 2i/512)
        position = (torch.arange(self.max_sequence_length) # pos variable in paper formula ranges from 0 to 199 --> since the max_sequence_len = 200
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator) # 200 x 256
        odd_PE = torch.cos(position / denominator) # 200 x 256
        stacked = torch.stack([even_PE, odd_PE], dim=2) # mergers them in parallel # 200 x 256 x 2
        PE = torch.flatten(stacked, start_dim=1, end_dim=2) # flatten alternating between them # 200 x 512 
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length # 200
        self.embedding = nn.Embedding(self.vocab_size, d_model) # nn layer with learnable parametrs --> works by applying a weight matrix to the vocab and generating vectors of length d_model
        self.language_to_index = language_to_index # dictionary --> maps an index to a token in a language
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length) # applyies positional encoder across d_model vectors --> for more information see PositionalEncoding class
        self.dropout = nn.Dropout(p=0.1) # dropout for more generalization
        self.START_TOKEN = START_TOKEN           #
        self.END_TOKEN = END_TOKEN               # vocab tokens
        self.PADDING_TOKEN = PADDING_TOKEN       #
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)] # transform the sentence into vacab with token indices --> each letter is a token
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN]) # inset start token if its set to true
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN]) # inset end token if set to true
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN]) # for each blank space between the end of the sentence and the max_sequence_len (200), insert padding_tokens
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)): # batch will be a list of sentences, and how many sentences we have will be our batch_size
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) ) # make a list of tokenized sentences
        tokenized = torch.stack(tokenized) # stack them in a tensor creating a new batch_size dimension
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        print(x)
        x = self.embedding(x) # embed
        pos = self.position_encoder().to(get_device()) # apply positional encoder
        x = self.dropout(x + pos) # sum positional encoder to embbedding
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 3 * 64
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 3 * 64
        q, k, v = qkv.chunk(3, dim=-1) # 30 x 8 x 200 x 64
        values, attention = scaled_dot_product(q, k, v, mask) # v: 30 x 8 x 200 x 64 // att: 30 x 8 x 200 x 200
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim) # 30 x 200 x 8 x 64 --> 30 x 200 x 512
        out = self.linear_layer(values) # linear layer feed forward
        return out # 30 x 200 x 512 (original shape but now with attention applied to embeddings)


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape # 512
        self.eps=eps # avoid division by zero error
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        # dims = [-(i + 1) for i in range(len(self.parameters_shape))] # (-1) --> dims >>> original code line
        dims = (-1)
        mean = inputs.mean(dim=dims, keepdim=True) # computs the mean across the last dimension --> 30 x 200 x ((512)) --> every vector of len 512 becomes now just one value that represents the mean of all the values in the vector
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True) # computes the variance of each element in the vector of 512
        std = (var + self.eps).sqrt() # converts var --> std
        y = (inputs - mean) / std # normalized value
        out = self.gamma * y + self.beta # gamma and beta multiplicative and additive parameters for more generalization
        return out # 30 x 200 x 512 (original shape but now normalized)

  
class PositionwiseFeedForward(nn.Module): # simple feed forward layer present in almost every neural net
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        # is extracts features from d_model vector for a hidden vector space and then compresses again in a d_model vector space
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module): # encoder --> all parts
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone() # save residual x to prevent gradint vanishing
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x) # add residual x
        residual_x = x.clone() # save it again
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x) # add it again
        return x # 30 x 200 x 512
    
class SequentialEncoder(nn.Sequential): # create an nn.Sequential like class for stacking encoders
    def forward(self, *inputs): # this is created because nn.sequential does not suport multiple inputs, thats the case for the layers of the encoder thjat require multiple parameters
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN) # transforms from token to d_model vectors
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob) # EncoderLayer returns a list with all the encoders in range of num_layers --> the "*" in the code separates that list in single elements (since thats the format nn.Sequential suports)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module): # attention using query (Q) from the decoder and key, value (KV) from encoder
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.head_dim = d_model // num_heads # 64
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # expands encoder output to kv dimensiona space (2 * 512 = 1024)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention
        kv = self.kv_layer(x) # k v tensors from multihead-masked-attention
        q = self.q_layer(y) # q tensor from encoder
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim) # separate for multihead attention
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim) # same here
        kv = kv.permute(0, 2, 1, 3) # permute dimensions
        q = q.permute(0, 2, 1, 3) # permute dimensions
        k, v = kv.chunk(2, dim=-1) # break tensors in 2 across last dimension
        values, attention = scaled_dot_product(q, k, v, mask) # perform dot product attention using q from encoder and k, v from masked attention (previous decoder step)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model) # transform into its original size
        out = self.linear_layer(values) # linear activation
        return out # returns default shape to be normalized: 30 x 200 x 512


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone() # residual techinque from resnet paper
        y = self.self_attention(y, mask=self_attention_mask) # masked attention
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y) # sum residual here

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, # 512 --> length of embed dim 
                ffn_hidden, # feed forward layer hidden layers
                num_heads, # number of times multihead attention is performed in parallel // it also represents the number of times we are gonna break d_mode dim and dividing the two, we get d_k --> 512/8 = 64
                drop_prob, # probability of turning neurons of --> allows the model to learn different paths across the training set
                num_layers, # number of encoders and decoders (refered in the paper as Nx)
                max_sequence_length, # maximum number of tokens to be passed into the model, we'll set this to 200 tokens for this project
                kn_vocab_size, # number of the combinations of tokens we're expecting our model to learn (example: if our model only learns the letters of the alphabet, this value will be 26)
                english_to_index, # dictionary mapping an index to an existng token
                kannada_to_index, # dictionary mapping an index to an existng token
                START_TOKEN, # string that represents the start of a sentence
                END_TOKEN, # string that represents the end of a sentence
                PADDING_TOKEN # string that fill empty spaces --> sentences can be supported up to 200 tokens, but not every sentence will have that, some will have less, so we add padding tokens (because our model has to receive a input of the same size every time)
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, kannada_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, kn_vocab_size) # maps the embed dim to a vector where each number represent a logit to the next token
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out