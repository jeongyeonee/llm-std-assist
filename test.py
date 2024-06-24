import warnings
warnings.filterwarnings("ignore")
                                                             
import os
import time
import json
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    StoppingCriteria
)

from peft import LoraConfig, PeftModel
from trl import SFTTrainer

root = "model"
    
def load_model(model_name, peft_model_name=None, quantization=False):
    if quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        bnb_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map='auto',
        use_cache=False,
    )
    
    if peft_model_name is None:
        return model
    
    return PeftModel.from_pretrained(model, peft_model_name)

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    return tokenizer
    
def make_q_turn(question):
    return f"<start_of_turn>user\n{question}<end_of_turn>"

def make_a_turn(answer):
    return f"<start_of_turn>model\n{answer}<end_of_turn>"

def clean_text(text):
    text = text.replace('\n', '')\
                .replace('질문 :', '질문:')\
                .replace('질문: ', '질문:')\
                .replace('답변 :', '답변:')\
                .replace('답변: ', '답변:')
    return text

def make_gemma_prompt(query):
    query = clean_text(query)
    
    prompt = []
    if '질문' not in query: # zero-shot
        prompt.append(make_q_turn(query))
        prompt.append("<start_of_turn>model\n")
    else: # few-shot
        for shot in query.split("질문:")[1:]:
            if '답변:' in shot: # few shot
                shot_q, shot_a = shot.split('답변:')
                prompt.append(make_q_turn(shot_q))
                prompt.append(make_a_turn(shot_a))
            else: # real question
                prompt.append(make_q_turn(shot))
                prompt.append("<start_of_turn>model\n")
    prompt = "\n".join(prompt)
    return prompt

def ask(model, tokenizer, query, max_seq_length):
    query = make_gemma_prompt(query)
    input_ids = tokenizer(query, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **input_ids,
        max_length=max_seq_length,
        temperature=0.,
        stopping_criteria=[EosListStoppingCriteria()]
    )
    return tokenizer.decode(outputs[0][len(input_ids['input_ids'][0]):])

def test(model, tokenizer, query_list, max_seq_length, task):
    answers = []
    for query in tqdm(query_list, desc=task):
        answer = ask(model, tokenizer, query, max_seq_length)
        #import pdb; pdb.set_trace()
        #answers.append(answer.split(query)[1])
        answers.append(answer)
    return answers

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [235298, 559, 235298, 15508, 235313]):
        self.eos_sequence = eos_sequence
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

def postprocess_outputs(output, task, eos_token_list=['<end_of_turn>', '<eos>']):
    pass
    

def aggregate_results(save_folder):
    result_paths = os.listdir(save_folder)
    
    # 단어 분리
    splits = []
    for path in filter(lambda x: '단어분리' in x, result_paths):
        task = path.replace('.csv', '')
        splits.append(
            pd.read_csv(
                os.path.join(save_folder, path),
                usecols=[
                    '분야',
                    '용어명/단어명',
                    '단어 분리/분류어유무',
                    task,
                    f'{task}_답변'
                ]
            )
        )
    split_df = pd.DataFrame()
    if len(splits):
        keys = ['분야', '용어명/단어명', '단어 분리/분류어유무']
        split_df = splits[0]
        
        if len(splits) > 1:
            for s in splits[1:]:
                split_df = pd.join(split_df, s,how='left', on=keys)
    
    # 단어 설명
    word_desc = []
    for path in filter(lambda x: '단어설명' in x, result_paths):
        task = path.replace('.csv', '')
        word_desc.append(
            pd.read_csv(
                os.path.join(save_folder, path),
                usecols=[
                    '분야',
                    '용어명/단어명',
                    '정의',
                    task,
                    f'{task}_답변'
                ]
            )
        )
        
    word_desc_df = pd.DataFrame()
    if len(word_desc):
        keys = ['분야', '용어명/단어명', '정의']
        word_desc_df = word_desc[0]
        
        if len(word_desc) > 1:
            for w in word_desc[1:]:
                word_desc_df = pd.join(word_desc_df, w, how='left', on=keys)
    
    # 용어 설명
    term_desc = []
    for path in filter(lambda x: '용어설명' in x, result_paths):
        task = path.replace('.csv', '')
        term_desc.append(
            pd.read_csv(
                os.path.join(save_folder, path),
                usecols=[
                    '분야',
                    '용어명/단어명',
                    '정의',
                    task,
                    f'{task}_답변'
                ]
            )
        )
        
    term_desc_df = pd.DataFrame()
    if len(term_desc):
        keys = ['분야', '용어명/단어명', '정의']
        term_desc_df = term_desc[0]
        
        if len(term_desc) > 1:
            for t in term_desc[1:]:
                term_desc_df = pd.join(term_desc_df, t, how='left', on=keys)
                
    with pd.ExcelWriter(os.path.join(save_folder, "평가셋.xlsx")) as writer:
        split_df.to_excel(writer, sheet_name = '단어분리', engine='xlsxwriter')
        word_desc_df.to_excel(writer, sheet_name = '단어설명', engine='xlsxwriter')
        term_desc_df.to_excel(writer, sheet_name = '용어설명', engine='xlsxwriter')

def work_func(args, task):
    print("task:", task, "PID", os.getpid())
    source = pd.read_csv(args['word_path']) if '단어설명' in task else pd.read_csv(args['term_path'])
    
    # 
    if '용어설명' in task:
        source = source[source['테스크']=='용어']
    
    # source = source.iloc[:1]
    
    query_list = source[task].tolist()
    
    model = load_model(
        model_name=args['model_name'],
        peft_model_name=args['peft_model_name'],
        quantization=args['quantization']
    )
    
    tokenizer = load_tokenizer(model_name=args['model_name'])
    
    answers = test(model, tokenizer, query_list, args['max_seq_length'], task)
    source[f'{task}_답변'] = answers
    
    # Save Folder
    model_name = args.get('peft_model_name', None)
    if model_name is None:
        model_name = args['model_name']
        
    save_folder = os.path.join('test', model_name)\
                  .replace('results/', '')
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    source.to_csv(os.path.join(save_folder, task+'.csv'), index=False)
    print(task, "완료")
    
def work_func_star(args):
    work_func(*args)
        
def main(args):
    start = time.time()
    
    # # Single Threading
    # for task in args['task']:
    #     work_func(args, task)
    
    # Multi Threading
    num_cores = os.cpu_count()
    pool = Pool(num_cores)
    pool.map(
        work_func_star,
        list(zip([args for _ in range(len(args['task']))], args['task']))
    )
    
    # 결과 취합
    # Save Folder
    model_name = args.get('peft_model_name', None)
    if model_name is None:
        model_name = args['model_name']
    save_folder = os.path.join('test', model_name)\
                  .replace('results/', '')
    aggregate_results(save_folder)
    
    end = time.time()
    duration = end-start
    print(f'time: {duration//60}min {duration%60}sec')

if __name__=='__main__':
    ##########################################################
    args = {
        'peft_model_name' : None,
        'quantization' : False,
        'max_seq_length' : 512,
    }
    args['word_path'] = "../../../data/Team1/240325_테스트셋-단어.csv"
    args['term_path'] = "../../../data/Team1/240325_테스트셋-용어.csv"
    
    args['dist'] = True
    args['task'] = [
        # '단어분리_0',
        # '단어분리_1',
        '단어분리_2',
        '단어설명_0',
        # '단어설명_1',
        # '단어설명_2',
        '용어설명_0',
        # '용어설명_1',
        # '용어설명_2',
    ]

    ##########################################################
    
    for model_name in [
        "20240329_1_0차_green증강_fullFT/checkpoint-4000",
        "20240330_8_1차_green_fullFT/checkpoint-45000",
    ]:
        args['model_name'] = model_name
        model_path = os.path.join(root, args['model_name'])
        
        if 'adapter_config.json' in os.listdir(model_path):
            args['peft_model_name'] = model_path
            args['quantization'] = True
            args['model_name'] = '../parksh/gemma-2b-data-std'
        else:
            args['model_name'] = model_path

        main(args)
    
    for m in range(2,11):
        model_name = f"20240329_{m}_0차_green증강_QLoRA_r_alpha/checkpoint-4000"
        args['model_name'] = model_name
        model_path = os.path.join(root, args['model_name'])
        
        if 'adapter_config.json' in os.listdir(model_path):
            args['peft_model_name'] = model_path
            args['quantization'] = True
            args['model_name'] = '../parksh/gemma-2b-data-std'
        else:
            args['model_name'] = model_path

        main(args)
        
    for m in range(1,8):
        model_name = f"20240330_{m}_0차_green증강_QLoRA_r_alpha/checkpoint-4000"
        args['model_name'] = model_name
        model_path = os.path.join(root, args['model_name'])
        
        if 'adapter_config.json' in os.listdir(model_path):
            args['peft_model_name'] = model_path
            args['quantization'] = True
            args['model_name'] = '../parksh/gemma-2b-data-std'
        else:
            args['model_name'] = model_path

        main(args)