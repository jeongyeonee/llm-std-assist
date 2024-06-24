import os
import json
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig

import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

import argparse
import pdb


# def argument():
#     parser = argparse.ArgumentParser(
#                         prog='Supervised Fine Tuning Gemma 2B',
#                         description='Supervised Fine Tuning Gemma 2B',
#                         epilog='Text at the bottom of help')
#     # dataset
#     parser.add_argument('--data', type=str) # 데이터셋 path
#     parser.add_argument('--test_size', type=int, default=30)
#     parser.add_arguemnt('--seed', type=int, default=2024)

#     # bnb config
#     parser.add_argument('--use_quantization', type=bool, default=True)

#     # lora config
#     parser.add_argument('--use_lora', type=bool, default=True)
#     parser.add_argument('--r', type=int, default=8)
#     parser.add_argument('--lora_alpha', type=int, default=8)
#     parser.add_argument('--lora_dropout', type=float, default=0.05)

#     # model config
#     parser.add_argument('--use_cache', type=boolean, default=True)
#     # parser.add_argument('--device_map', type=str, default='0')

#     # train config
#     parser.add_argument('--max_seq_length', type=int, default=512)
#     parser.add_argument('--num_train_epochs', type=int, default=5)
#     parser.add_argument('--max_steps', type=int, default=-1)
#     parser.add_argument('--per_device_train_batch_size', type=int, default=8)
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=2)

#     # eval
#     parser.add_argument('--evaluation_strategy', type=str, default='steps')
#     parser.add_argument('--eval_steps_per_epoch', type=int, default=10)

#     # optimizer
#     parser.add_argument('--optim', type=str, default='paged_adamw_32bit')
#     parser.add_argument('--learning_rate', type=float, default=1e-5)
#     parser.add_argument('--lr_scheduler_type', type=str, default='linear')
#     parser.add_argument('--warmup_steps', type=int, default=2)
#     parser.add_argument('--weight_decay', type=float, default=0.)

#     # Logging
#     parser.add_argument('--logging_steps', type=int, default=1)
#     parser.add_argument('--logging_dir', type=str, default='results_logs')
#     parser.add_argument('--logging_strategy', type=str, default='steps')

#     # Save
#     parser.add_argument('--save_strategy', type=str, default='steps')
#     parser.add_argument('--save_steps_per_epoch', type=int, default=10)

#     parser.add_argument('--max-seq-length', type=int, default=512)
#     parser.add_argument('--max-seq-length', type=int, default=512)
#     parser.add_argument('--max-seq-length', type=int, default=512)
#     parser.add_argument('--max-seq-length', type=int, default=512)

#     args = parser.parse_args()
    
def new_exp_name(output_root, prefix, sep='_', tag=None):
    ''' prefix_expnum
    ex) yyyymmdd_1'''
    dirs = list(os.walk(output_root))[0][1]
    dirs = list(filter(lambda x: x.split(sep)[0]==prefix, dirs))
    if len(dirs):
        # exp_num
        exp_nums = [int(d.split(sep)[1]) for d in dirs]
        max_num = max(exp_nums)
        folder_name = f'{prefix}{sep}{max_num + 1}'
        if tag is not None:
            folder_name += f'{sep}{tag}'
        return folder_name
    else:
        folder_name = f'{prefix}{sep}{1}'
        if tag is not None:
            folder_name += f'{sep}{tag}'
        return folder_name

def arguments():
    args = {
        # exp
        'tag': None,
        
        # dataset
        'data_path': "../../../data/Team1/생성_green_sampling_240322.csv",
        'test_size': 0.1,

        # model
        'model_path': 'jylee420/gemma-2b-data-std',
        'use_cache': True,
        'device_map': {'': 0},

        # bnb quantization
        'use_quantization': True,
        'load_in_4bit': True,
        'bnb_4bit_quant_type': 'nf4',
        'bnb_4bit_compute_dtype': 'torch.bfloat16',

        # lora
        'use_lora': True,
        'r': 8,
        'lora_alpha': 8,
        'lora_dropout': 0.05,

        # Train
        'max_seq_length': 512,
        "per_epoch_eval_steps": 10,
        "per_epoch_save_steps": 2,
        'ta': {
            "num_train_epochs": 3,
            "max_steps": -1,
            "evaluation_strategy": "steps",
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "bf16": True,
            "data_seed": 2024,

            ## optimizer
            "optim": "paged_adamw_32bit",
            "learning_rate": 1e-5,
            "lr_scheduler_type": "linear",
            # "warmup_steps": 2,

            ## log
            "logging_steps": 1,
            "logging_dir": f"results_logs",
            "logging_strategy": "steps",

            ## save
            "save_strategy": "steps",
            
            # "save_total_limit": 100,
            "output_dir": f"model",
            "save_only_model": False,
        }
    }
    return args

def make_dirs(args):
    prefix = datetime.today().strftime("%Y%m%d")
    new_exp_folder_name = new_exp_name(args['ta']['output_dir'], prefix, tag=args['tag'])
    args['ta']['output_dir'] = os.path.join(args['ta']['output_dir'], new_exp_folder_name)
    args['ta']['logging_dir'] = os.path.join(args['ta']['logging_dir'], new_exp_folder_name)
    return args
    
def validate(args):
    assert 'results/' in args['output_dir']
    assert 'results_logs' in args['logging_dir']
    
    
def train(args):
    # quantization
    if args['bnb_4bit_compute_dtype']=='torch.bfloat16':
        bnb_4bit_compute_dtype = torch.bfloat16
    else:
        raise
        
    if args['use_quantization']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args['load_in_4bit'],
            bnb_4bit_quant_type=args['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
        )
    else:
        bnb_config = None
    
    # lora
    if args['use_lora']:
        lora_config = LoraConfig(
            r=args['r'],
            lora_alpha = args['lora_alpha'],
            lora_dropout = args['lora_dropout'],
            target_modules=[
                "q_proj", 
                "o_proj", 
                "k_proj", 
                "v_proj", 
                "gate_proj", 
                "up_proj", 
                "down_proj"],
            task_type="CAUSAL_LM",
        )
    else:
        lora_config = None
        
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args['model_path'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args['model_path'],
        vocab_size=len(tokenizer),
        # n_ctx=max_seq_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=args['use_cache'],
        attn_implementation="flash_attention_2",
        device_map=args['device_map'],
        quantization_config=bnb_config
    )
    
    def count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"model size: {model_size/1000**2:.1f}M parameters")
    print(f"trainable params: {count_trainable_params(model)/1000**2:.2f}M\t{count_trainable_params(model)/model_size*100:.2f}%")
    
    # Data
    data = load_dataset("csv", data_files=args['data_path'])
    data = data.shuffle(seed=args['ta']['data_seed'])['train']\
               .train_test_split(test_size=args['test_size'], seed=args['ta']['data_seed'])
    
    # Eval and Logging Step
    args['ta']['eval_steps'] = round(len(data['train'])/args['ta']['per_device_train_batch_size']/args['per_epoch_eval_steps'], -2)
    args['ta']['save_steps'] = round(len(data['train'])/args['ta']['per_device_train_batch_size']/args['per_epoch_save_steps'], -2)
    print(f"eval_steps: {args['ta']['eval_steps']}")
    print(f"save_steps: {args['ta']['save_steps']}")
                                     
                                     
    # Trainer
    def formatting_func(example):
        output_texts = []
        questions = example['question']
        answers = example['answer']

        for qs, an in zip(questions, answers):
            text = (f"<bos><start_of_turn>user\n{qs}<end_of_turn>\n"
                    f"<start_of_turn>model\n{an}<end_of_turn><eos>")
            output_texts.append(text)
        return output_texts
    
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"],
        max_seq_length=args['max_seq_length'],
        args=transformers.TrainingArguments(**args['ta']),
        formatting_func=formatting_func,
        peft_config=lora_config,
        data_collator = collator,
    )
    
    trainer.train()
    model.save_pretrained(os.path.join(args['ta']['output_dir'], "final"))
    tokenizer.save_pretrained(os.path.join(args['ta']['output_dir'], "final"))
    

def do_experiment(tag, experiment_set=None, fix: dict=None):
    if experiment_set is not None:
        for param_k, values in experiment_set.items():
            for v in values:
                args = arguments()
                if isinstance(fix, dict):
                    for fk, fv in fix.items():
                        if fk in args:
                            args[fk] = fv
                        else:
                            args['ta'][fk] = fv
                args['tag'] = tag
                args = make_dirs(args)

                # train
                # args['ta']['num_train_epochs'] = -1
                # args['ta']['max_steps'] = 1
                # args['ta']['save_steps'] = 1
                args[param_k] = v

                # save args
                if not os.path.exists(args['ta']['output_dir']):
                    os.makedirs(args['ta']['output_dir'])
                with open(os.path.join(args['ta']['output_dir'], 'args.json'), 'w') as f:
                    json.dump(args, f, ensure_ascii=False, indent=4)

                train(args)
                print("-"*50)
                
                return
            
    if fix is not None:
        args = arguments()
        for fk, fv in fix.items():
            if fk in args:
                args[fk] = fv
            else:
                args['ta'][fk] = fv
        args['tag'] = tag
        args = make_dirs(args)
            
        if not os.path.exists(args['ta']['output_dir']):
            os.makedirs(args['ta']['output_dir'])
        with open(os.path.join(args['ta']['output_dir'], 'args.json'), 'w') as f:
            json.dump(args, f, ensure_ascii=False, indent=4)
            
        train(args)
        print("-"*50)
        return
            
    
if __name__=='__main__':
    # Gemma Full FT    
    do_experiment(
        'save_dir_name',
        fix = {
            'use_lora': False,
            'use_quantization': False,
        }
    )