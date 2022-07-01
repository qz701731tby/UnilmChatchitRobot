"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/8/4 11:25
"""
import torch
import torch.nn.functional as F
from tokenization_unilm import UnilmTokenizer
from modeling_unilm import UnilmForSeq2SeqDecodeSampleCached, UnilmConfig
import copy
import os
import argparse
import re
from dirty_recognize import dirty_reg
import time
import random
from tqdm import tqdm
import numpy as np

def remove_dirty_sentence(dirty_obj, sentence):
    if len(dirty_obj.match(sentence)) == 0:
        return False
    else:
        return True


def remove_multi_symbol(text):
    r = re.compile(r'([.,，/\\#!！？?。$%^&*;；:：{}=_`´︵~（）()-])[.,，/\\#!！？?。$%^&*;；:：{}=_`´︵~（）()-]+')
    text = r.sub(r'\1', text)
    return text


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def read_data(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            data.append(line)
    
    return data

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='生成设备')
    parser.add_argument('--topk', default=3, type=int, help='取前k个词')
    parser.add_argument('--topp', default=0.95, type=float, help='取超过p的词')
    parser.add_argument('--dirty_path', default='data/dirty_words.txt', type=str, help='敏感词库')
    parser.add_argument('--model_name_or_path', default='kuakua_robot_model/', type=str, help='模型路径')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help="重复词的惩罚项")
    parser.add_argument('--max_len', type=int, default=32, help='生成的对话的最大长度')
    parser.add_argument('--no_cuda', type=bool, default=False, help='是否使用GPU进行预测')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print("cuda:", args.cuda)
    # device = 'cuda' if args.cuda else 'cpu'
    device = 'cuda'
    print('using device:{}'.format(device))
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = UnilmConfig.from_pretrained(args.model_name_or_path, max_position_embeddings=512)
    tokenizer = UnilmTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=False)
    model = UnilmForSeq2SeqDecodeSampleCached.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    model.eval()
    print('Chitchat Robot Starting')
    dirty_obj = dirty_reg(args.dirty_path)
    _tril_matrix = torch.tril(torch.ones((args.max_len, args.max_len), dtype=torch.long))
    set_seed(42, 0)
    test_data_path = "./data/lccc_test.txt"
    test_data = read_data(test_data_path)
    total_time = 0
    results = []
    for text in tqdm(test_data):
        start_time = time.time()
        # if remove_dirty_sentence(dirty_obj, text):
        #     print("chatbot:" + "换个话题聊聊吧。")
        #     continue
        input_ids = tokenizer.encode(text)
        input_len = len(input_ids)
        token_type_ids = [4] * len(input_ids)
        generated = []
        prev_embedding, prev_encoded_layers, position_ids = None, None, None
        ori_embedding, ori_encoded_layers = None, None
        for idx in range(args.max_len):
            if idx == 0:
                curr_input_ids = copy.deepcopy(input_ids)
                curr_input_ids.append(tokenizer.mask_token_id)
                curr_token_type_ids = copy.deepcopy(token_type_ids)
                curr_token_type_ids.extend([5])  
            else:
                curr_input_ids = copy.deepcopy(input_ids)
                curr_input_ids.append(tokenizer.mask_token_id)
                curr_token_type_ids = copy.deepcopy(token_type_ids)
                curr_token_type_ids.extend([5])
                position_ids = [input_len+idx-1, input_len+idx]
                position_ids = curr_input_tensor = torch.tensor(position_ids).long().to(device).view([1, -1])
            curr_input_tensor = torch.tensor(curr_input_ids).long().to(device).view([1, -1])
            curr_token_type_ids = torch.tensor(curr_token_type_ids).long().to(device).view([1, -1])
            # attention mask
            input_mask = torch.zeros(input_len+1+idx, input_len+1+idx, dtype=torch.long)
            input_mask[:, :input_len].fill_(1)
            second_st, second_end = input_len, input_len+1+idx
            input_mask[second_st:second_end, second_st:second_end].copy_(_tril_matrix[:second_end-second_st, :second_end-second_st])
            if idx >= 1:
                input_mask = input_mask[-2:, :]
            attn_mask = input_mask.unsqueeze(0).to(device)
            prev_embedding, prev_encoded_layers, outputs = model(input_ids=curr_input_tensor, token_type_ids=curr_token_type_ids, 
                                            attention_mask=attn_mask, position_ids=position_ids, output_all_encoded_layers=True, prev_embedding=ori_embedding, 
                                            prev_encoded_layers=ori_encoded_layers)
            # prev_embedding, prev_encoded_layers删除最后一个timestamp[MASK]的数据,拼接到之前的embedding和encoded_layers上
            if idx == 0:
                # (1, n, 768) -> (1, n-1, 768)
                ori_embedding = prev_embedding[:,:-1,:]
                # (12, 1, n, 768) -> (12, 1, n-1, 768)
                ori_encoded_layers = [layer[:, :-1, :] for layer in prev_encoded_layers]
            else:
                # (1, n, 768) -> (1, n-1, 768) -> (1, n, 768)
                ori_embedding = torch.cat((ori_embedding, prev_embedding[:,:-1,:]), dim=1)
                # (12, 1, n, 768) -> (12, 1, n-1, 768) -> (12, 1, n, 768) 
                ori_encoded_layers = [torch.cat((ori_encoded_layers[i], prev_encoded_layers[i][:,:-1,:]), dim=1) for i in range(len(ori_encoded_layers))]
            next_token_logits = outputs[-1, -1, :]
            for id in set(generated):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if next_token == tokenizer.sep_token_id:  # 遇到[SEP]则表明生成结束
                break
            generated.append(next_token.item())
            input_ids = [next_token.item()]
            token_type_ids = [5]
        text = tokenizer.convert_ids_to_tokens(generated)
        text = remove_multi_symbol("".join(text))
        end_time = time.time()
        total_time += (end_time-start_time)
        results.append(text)
        # print("前向推理时间：{} s".format(end_time-start_time))
    
    print("average_time:", total_time/len(test_data))

    with open("./data/cached_output.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(res + "\n")


if __name__ == "__main__":
    main()




