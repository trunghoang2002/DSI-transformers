from data import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, TrainerCallback
from trainer import IndexingTrainer
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

class QueryEvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, restrict_decode_vocab, args: TrainingArguments, tokenizer: T5Tokenizer):
        self.tokenizer = tokenizer
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.restrict_decode_vocab = restrict_decode_vocab
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=QueryEvalCollator(
                self.tokenizer,
                padding='longest'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    # use to evalute the 'real' eval set or test_dataset in this code
    def on_epoch_end(self, args, state, control, **kwargs):
        '''Event called at the end of an epoch.'''
        hit_at_1 = 0
        hit_at_10 = 0
        model = kwargs['model'].eval()
        for batch in tqdm(self.dataloader, desc='Evaluating dev queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=20,
                    num_beams=10,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=10,
                    early_stopping=True, ).reshape(inputs['input_ids'].shape[0], 10, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    hits = np.where(np.array(rank_list)[:10] == label)[0]
                    # print("label:", label) # ex: '4001'
                    # print("hits:", np.array(rank_list)[:10]) # ex: ['1939' '9898' '12798' '9895' '9395' '9798' '9393' '12795' '9795' '9495']
                    if len(hits) != 0:
                        hit_at_10 += 1
                        if hits[0] == 0:
                            hit_at_1 += 1
        self.logger.log({"Hits@1": hit_at_1 / len(self.test_dataset), "Hits@10": hit_at_10 / len(self.test_dataset)})

# use to evaluate the 'fake' eval set
def compute_metrics(eval_preds):
    '''The function that will be used to compute metrics at evaluation'''
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}


def main(args):
    model_name = args.model_name
    L = args.max_length  # only use the first 32 tokens of documents (including title)
    max_steps = args.max_steps

    # We use wandb to log Hits scores after each epoch. Note, this script does not save model checkpoints.
    wandb.login()
    wandb.init(project="DSI", name=args.wandb_name)

    tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=args.cache_dir)

    train_dataset = IndexingTrainDataset(path_to_data=args.train_data, max_length=L, cache_dir=args.cache_dir, tokenizer=tokenizer)
    eval_dataset = IndexingTrainDataset(path_to_data=args.eval_data, max_length=L, cache_dir=args.cache_dir, tokenizer=tokenizer)
    test_dataset = IndexingTrainDataset(path_to_data=args.test_data, max_length=L, cache_dir=args.cache_dir, tokenizer=tokenizer)
    
    
    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    # eval_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_multi_task_train.jsonl',

    
    # This is the actual eval set.
    # test_dataset = IndexingTrainDataset(path_to_data='data/NQ/NQ_10k_valid.jsonl',

    ################################################################
    # docid generation constrain, we only generate integer docids.
    SPIECE_UNDERLINE = "▁"
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    ################################################################

    # https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        max_steps=args.max_steps,
        dataloader_drop_last=args.dataloader_drop_last,
        report_to=args.report_to,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    trainer = IndexingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer,
            padding='longest',
        ),
        compute_metrics=compute_metrics,
        callbacks=[QueryEvalCallback(test_dataset, wandb, restrict_decode_vocab, training_args, tokenizer)],
        restrict_decode_vocab=restrict_decode_vocab
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for document indexing model")

    # Các tham số của TrainingArguments
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for model checkpoints and logs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate for training")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay factor")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size per device")
    parser.add_argument("--evaluation_strategy", type=str, default='steps', choices=['no', 'steps', 'epoch'], help="Evaluation strategy to use")
    parser.add_argument("--eval_steps", type=int, default=50, help="Number of steps for evaluation")
    parser.add_argument("--max_steps", type=int, default=100, help="Total number of training steps")
    parser.add_argument("--dataloader_drop_last", type=bool, default=False, help="Whether to drop last incomplete batch")
    parser.add_argument("--report_to", type=str, default='wandb', help="Reporting tool for logging (e.g., wandb, tensorboard)")
    parser.add_argument("--logging_steps", type=int, default=50, help="Number of steps for logging")
    parser.add_argument("--save_strategy", type=str, default='steps', choices=['no', 'steps', 'epoch'], help="Save strategy to use")
    parser.add_argument("--save_steps", type=int, default=50, help="Number of steps before saving a checkpoint")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Limit of total saved checkpoints")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True, help="Whether to load the best model at the end")
    parser.add_argument("--fp16", type=bool, default=False, help="Whether to use 16-bit (mixed) precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=10, help="Number of workers for data loading")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    # Các tham số bổ sung
    parser.add_argument("--model_name", type=str, default="t5-large", help="Name of the model to use")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum token length for documents")
    parser.add_argument("--train_data", type=str, default="data/NQ/test_train.jsonl", help="Path to training data")
    parser.add_argument("--eval_data", type=str, default="data/NQ/test_train.jsonl", help="Path to evaluation data")
    parser.add_argument("--test_data", type=str, default="data/NQ/test_val.jsonl", help="Path to test data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to cache model and tokenizer")
    parser.add_argument("--wandb_name", type=str, default="NQ-10k-t5-large", help="Name for Weights & Biases logging")

    args = parser.parse_args()
    main(args)