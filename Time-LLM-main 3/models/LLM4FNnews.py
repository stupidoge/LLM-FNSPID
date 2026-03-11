# summaris are 128, 1024, 768
# we reshape it into -1, 768
# and then projection into text prototype size..

# but we could try another way, directly use 128, 1024, 768 without loss any information
# and then


from math import sqrt

import torch
import torch.nn as nn

import pandas as pd

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.losses import ContrastiveLoss
transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8, df_news=False):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.df_news = df_news

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000 # set text prototype linear tokens
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        # self.mapping_layernews = nn.Linear(2048, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayerSample(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)
        self.avg_projection = nn.Linear(configs.llm_dim*2, configs.llm_dim)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums # ???

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_seq, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, contrast_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_seq)
            return dec_out[:, -self.pred_len:, :], contrast_loss
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_seq ):
        # self.description = self.description + 'Related news:'
        x_enc = self.normalize_layers(x_enc, 'norm')
        B, T, N = x_enc.size()

        # # x_seq is the time window
        summaries = []
        for b in range(B):
            seq_timestamps = pd.to_datetime(x_seq[b].cpu(), unit='s')
            # get start_time and end_time
            start_time = seq_timestamps[0]
            end_time = seq_timestamps[-1]

            # select news based on mask
            mask = (self.df_news.date >= start_time) & (self.df_news.date <= end_time)
            news_summary = ' '.join(self.df_news.loc[mask, 'summary'].tolist())
            if not news_summary:
                news_summary = "There is no news in this period of time."
            summaries.append(news_summary)

        # append news into description中
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0] # [0] is the min, [1] is the index
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc) # calculate lags by fourier
        trends = x_enc.diff(dim=1).sum(dim=1) # to calculate the trends using n-(n-1)


        # create prompts by using statistical methods
        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                # f"Related news: {summaries[b]}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        # embedding the prompt
        # prompt_embeddings: embedding for prompt
        # source_embeddings: embedding for llm corpus
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        summaries = self.tokenizer(summaries, return_tensors="pt", padding=True, truncation=True).input_ids

        news_embeddings = self.llm_model.get_input_embeddings()(summaries.to(x_enc.device))
        # print(news_embeddings.shape)

        # if summaries.shape[1] ==0:
        #     news_embeddings = torch.zeros(summaries.shape[0], 1024, 768, device=x_enc.device)
        #     print(news_embeddings.shape)
        # else:
        #     news_embeddings = self.llm_model.get_input_embeddings()(summaries.to(x_enc.device))
        #     print(news_embeddings.shape)

        # print('before projection:', news_embeddings.shape)
        news_embeddings = news_embeddings.view(-1,768)
        news_projection = nn.Linear(news_embeddings.shape[0], self.num_tokens).to(x_enc.device)

        source_embeddings = news_projection(news_embeddings.permute(1,0)).permute(1,0)
        # print('after projection:',source_embeddings.shape)
        # source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))

        enc_out, positive_sample, negative_sample = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # contrastive learning loss
        contrast_loss_fn = ContrastiveLoss()
        contrast_loss = contrast_loss_fn(enc_out, positive_sample, negative_sample)

        avg_out = torch.cat([enc_out, positive_sample], dim=-1)
        avg_out = self.avg_projection(avg_out)
        avg_out = self.dropout(avg_out)


        llama_enc_out = torch.cat([prompt_embeddings, avg_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out, contrast_loss

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm) # ?????
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    # Attention(Q,K,V): Attention(patch time-series, text prototype, text prototype)
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E) # scaling attention

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding) # calculate the dot product of Q,K

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

class ReprogrammingLayerSample(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None):
        super(ReprogrammingLayerSample, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.d_keys = d_keys
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm) # ?????
        self.sample_projection = nn.Linear(d_keys * n_heads, d_llm)

        self.W_score1 = nn.Parameter(torch.randn(d_keys, d_keys))
        self.W_score2 = nn.Parameter(torch.randn(d_keys, d_keys))
        # self.W_value = nn.Parameter(torch.randn(d_keys, n_heads))
        # self.W_target = nn.Parameter(torch.randn(d_keys, n_heads))
        # self.W_combine = nn.Parameter(torch.randn(2*d_keys, d_keys))
        self.combine_weight1 = nn.Parameter(torch.randn(d_keys, n_heads))
        self.combine_weight2 = nn.Parameter(torch.randn(d_keys, n_heads))

        self.n_heads = n_heads
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.activation = nn.LeakyReLU()
    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out, max_indices, min_indices = self.reprogramming(target_embedding, source_embedding, value_embedding)
        # select prototype
        positive_samples = []
        negative_samples = []

        # 遍历每个批次和每个时间步
        for b in range(B):
            for l in range(L):
                # 从source_embedding中提取对应于max_indices的正样本
                positive_index = max_indices[b, l].item()  # 获取具体的索引值
                positive_sample = source_embedding[positive_index].unsqueeze(0)  # 提取并增加一个维度
                positive_samples.append(positive_sample)

                # 类似地，从source_embedding中提取对应于min_indices的负样本
                negative_index = min_indices[b, l].item()
                negative_sample = source_embedding[negative_index].unsqueeze(0)
                negative_samples.append(negative_sample)

        # 将列表转换为张量
        positive_samples = torch.cat(positive_samples, dim=0).view(B, L, -1)
        negative_samples = torch.cat(negative_samples, dim=0).view(B, L, -1)

        # output
        out = out.reshape(B, L, -1)


        return self.out_projection(out), self.sample_projection(positive_samples), self.sample_projection(negative_samples)

    # Attention(Q,K,V): Attention(patch time-series, text prototype, text prototype)
    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E) # scaling attention

        # scores = self.activation(self.dropout2(torch.einsum("blhe,ee,she->bhls",
        #                                                     target_embedding,
        #                                                     self.W_score,
        #                                                     source_embedding))) # calculate the dot product of Q,K

        scores = self.activation(self.dropout2(torch.einsum("blhe,ee,she,ee->bhls",
                                                            target_embedding,
                                                            self.W_score1,
                                                            source_embedding,
                                                            self.W_score2))) # calculate the dot product of Q,K
        A = self.dropout1(torch.softmax(scale * scores, dim=-1))

        # 假设A的维度为(B, H, L, S)，其中B是batch size, H是头部数量, L是时间步长, S是source prototypes的数量
        A_mean = A.mean(dim=1)  # 在头部维度上取平均，得到的维度为(B, L, S)
        max_indices = torch.argmax(A_mean, dim=-1)  # 在prototype维度上找到最大值的索引，维度为(B, L)
        min_indices = torch.argmin(A_mean, dim=-1)

        reprogramming_embedding_value = torch.einsum("bhls,she->blhe", A, value_embedding)
        reprogramming_embedding_target = torch.einsum("bhls,blhe->blhe", A, target_embedding)

        # reprogramming_embedding_value = torch.einsum("bhls,she,eh->blhe", A, value_embedding, self.W_value)
        # reprogramming_embedding_target = torch.einsum("bhls,blhe,eh->blhe", A, target_embedding, self.W_target)

        reprogramming_embedding = self.activation(self.dropout2(torch.einsum("blhe,eh, blhe,eh->blhe",
                                                            reprogramming_embedding_value,
                                                            self.combine_weight1,
                                                            reprogramming_embedding_target,
                                                            self.combine_weight2)))

        # # merge
        # reprogramming_merged = torch.cat((reprogramming_embedding_target, reprogramming_embedding_value), dim=-1)
        # reprogramming_embedding = self.activation(self.dropout2(torch.einsum("blhf,fe ->blhe",
        #                                                     reprogramming_merged,
        #                                                     self.W_combine,
        #                                                     )))

        return reprogramming_embedding, max_indices, min_indices

