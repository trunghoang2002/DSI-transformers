import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments
from data import IndexingTrainDataset, QueryEvalCollator
from torch.utils.data import DataLoader
import argparse

def calculate_mrr(true_list, pred_list, k=10):
    mrr_scores = []
    for i in range(len(true_list)):
        relevant_docs = true_list[i]
        retrieved_docs = pred_list[i][:k]

        # Find the rank of the first relevant document in the top k retrieved documents
        rank = None
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                rank = i + 1  # Rank starts from 1
                break
        
        # If no relevant document, use 0
        if rank is not None:
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)
    
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0

def calculate_recall(true_list, pred_list, k=10):
    recall_scores = []
    for i in range(len(true_list)):
        relevant_docs = true_list[i]
        retrieved_docs = pred_list[i][:k]

        # Count relevant documents in the top k retrieved documents
        count_retrieval = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        score = count_retrieval / len(relevant_docs) if relevant_docs else 0
        recall_scores.append(score)

    return sum(recall_scores) / len(recall_scores) if recall_scores else 0

def calculate_myrecall(true_list, pred_list, k=10):
    recall_scores = []
    for i in range(len(true_list)):
        relevant_docs = true_list[i]
        retrieved_docs = pred_list[i][:k]

        # Count relevant documents in the top k retrieved documents
        count_retrieval = sum(1 for doc in retrieved_docs if doc in relevant_docs)
        score = count_retrieval / min(k, len(relevant_docs)) if relevant_docs else 0
        recall_scores.append(score)

    return sum(recall_scores) / len(recall_scores) if recall_scores else 0

def calculate_map(true_list, pred_list, k=10):
    ap_scores = []
    for i in range(len(true_list)):
        relevant_docs = true_list[i]
        retrieved_docs = pred_list[i][:k]

        num_relevant_retrieved = 0
        precision_sum = 0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                num_relevant_retrieved += 1
                precision_at_i = num_relevant_retrieved / (i + 1)
                precision_sum += precision_at_i
        
        ap = precision_sum / len(relevant_docs) if relevant_docs else 0
        ap_scores.append(ap)


    return sum(ap_scores) / len(ap_scores) if ap_scores else 0

class Evaluator():
    def __init__(self, test_dataset, args: TrainingArguments, tokenizer: T5Tokenizer, model: T5ForConditionalGeneration):
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.test_dataset = test_dataset
        ################################################################
        # docid generation constrain, we only generate integer docids.
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in self.tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(self.tokenizer.eos_token_id)

        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ################################################################
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
    def evaluate(self, num_beams=200):
        '''Event called at the end of an epoch.'''
        model = self.model.eval()
        true_labels = []
        pred_labels = []
        for batch in tqdm(self.dataloader, desc='Evaluating test queries'):
            inputs, labels = batch
            with torch.no_grad():
                batch_beams = model.generate(
                    inputs['input_ids'].to(model.device),
                    max_length=20,
                    num_beams=num_beams,
                    prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                    num_return_sequences=num_beams,
                    early_stopping=True, ).reshape(inputs['input_ids'].shape[0], num_beams, -1)
                for beams, label in zip(batch_beams, labels):
                    rank_list = self.tokenizer.batch_decode(beams,
                                                            skip_special_tokens=True)  # beam search should not return repeated docids but somehow due to T5 tokenizer there some repeats.
                    true_labels.append(label)
                    pred_labels.append(rank_list)
                    print("\ntrue_labels:", label)
                    print("pred_labels:", rank_list)
        mrr = calculate_mrr(true_labels, pred_labels)
        map_score = calculate_map(true_labels, pred_labels)

        recall1 = calculate_recall(true_labels, pred_labels, 1)
        recall3 = calculate_recall(true_labels, pred_labels, 3)
        recall5 = calculate_recall(true_labels, pred_labels, 5)
        recall10 = calculate_recall(true_labels, pred_labels, 10)
        recall20 = calculate_recall(true_labels, pred_labels, 20)
        recall50 = calculate_recall(true_labels, pred_labels, 50)
        recall100 = calculate_recall(true_labels, pred_labels, 100)
        recall200 = calculate_recall(true_labels, pred_labels, 200)

        myrecall1 = calculate_myrecall(true_labels, pred_labels, 1)
        myrecall3 = calculate_myrecall(true_labels, pred_labels, 3)
        myrecall5 = calculate_myrecall(true_labels, pred_labels, 5)
        myrecall10 = calculate_myrecall(true_labels, pred_labels, 10)
        myrecall20 = calculate_myrecall(true_labels, pred_labels, 20)
        myrecall50 = calculate_myrecall(true_labels, pred_labels, 50)
        myrecall100 = calculate_myrecall(true_labels, pred_labels, 100)
        myrecall200 = calculate_myrecall(true_labels, pred_labels, 200)

        print(f"MRR@10: {mrr:.4f}")
        print(f"MAP@10: {map_score:.4f}")

        print(f"Recall@1: {recall1:.4f}\t\tMy_recall@1: {myrecall1:.4f}")
        print(f"Recall@3: {recall3:.4f}\t\tMy_recall@3: {myrecall3:.4f}")
        print(f"Recall@5: {recall5:.4f}\t\tMy_recall@5: {myrecall5:.4f}")
        print(f"Recall@10: {recall10:.4f}\t\tMy_recall@10: {myrecall10:.4f}")
        print(f"Recall@20: {recall20:.4f}\t\tMy_recall@20: {myrecall20:.4f}")
        print(f"Recall@50: {recall50:.4f}\t\tMy_recall@50: {myrecall50:.4f}")
        print(f"Recall@100: {recall100:.4f}\t\tMy_recall@100: {myrecall100:.4f}")
        print(f"Recall@200: {recall200:.4f}\t\tMy_recall@200: {myrecall200:.4f}")

def main(args):
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
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    test_dataset = IndexingTrainDataset(path_to_data=args.test_data, max_length=args.max_length, cache_dir=args.cache_dir, tokenizer=tokenizer)
    evaluator = Evaluator(test_dataset, training_args, tokenizer, model)
    evaluator.evaluate(args.num_beams)

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
    parser.add_argument("--model_name_or_path", type=str, default="t5-large", help="Name or path of the model to use")
    parser.add_argument("--max_length", type=int, default=32, help="Maximum token length for documents")
    parser.add_argument("--train_data", type=str, default="data/NQ/test_train.jsonl", help="Path to training data")
    parser.add_argument("--eval_data", type=str, default="data/NQ/test_train.jsonl", help="Path to evaluation data")
    parser.add_argument("--test_data", type=str, default="data/NQ/test_val.jsonl", help="Path to test data")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Directory to cache model and tokenizer")
    parser.add_argument("--wandb_name", type=str, default="NQ-10k-t5-large", help="Name for Weights & Biases logging")
    parser.add_argument("--num_beams", type=int, default=200, help="Numer of beams use for generating")
    args = parser.parse_args()
    main(args)