from transformers import AutoTokenizer
from transformers import DefaultDataCollator, AutoConfig, Trainer, TrainingArguments, AutoModelForCausalLM, HfArgumentParser
from torch.utils.data import Dataset
from torch import nn
from tqdm import tqdm
from transformers.utils import check_min_version
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import pandas as pd

import logging , os , wandb , pickle , torch , gc , sys , transformers , deepspeed

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)


class ConditionalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], labels=inputs['labels'])
        loss = outputs[0]
        return (loss, outputs) if return_outputs else loss


class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=512, conditional_mask=True):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples,self.mask = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            self.examples = []
            self.mask = []
            df = pd.read_csv(file_path)
            eot_token = tokenizer.eos_token_id
            tokenized_s = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)) if len(s) > 0 else [] for s in tqdm(df['s'].fillna(''))  ]
            tokenized_t = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t+'<|endoftext|>')) if len(t) > 0 else [] for t in tqdm(df['t'].fillna(''))  ]
            tokenized_text = [tokenized_s[cc] + tokenized_t[cc] for cc in range(len(tokenized_s))]
            tokenized_len = [len(tokenized_text[cc]) for cc in range(len(tokenized_text))]
            tokenized_padded = [tokenized_text[cc] + [eot_token]*(block_size-(block_size % tokenized_len[cc])) for cc in range(len(tokenized_text))]
            tokenized_text = [item for sublist in tokenized_padded for item in sublist]
            if conditional_mask:
                mask = [[-100]*len(tokenized_s[cc]) + tokenized_t[cc] for cc in range(len(tokenized_s))]
                mask_len = [len(mask[cc]) for cc in range(len(mask))]
                mask_padded = [mask[cc] + [-100]*(block_size-(block_size % mask_len[cc])) for cc in range(len(mask))]
                mask = [item for sublist in mask_padded for item in sublist]
            else:
                mask = tokenized_text
            

            for i in range(0, len(tokenized_text)-block_size+1, block_size): 
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
                self.mask.append(mask[i:i+block_size])

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump((self.examples,self.mask), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return {'input_ids':torch.tensor(self.examples[item]) , 'labels': torch.tensor(self.mask[item])}



def main():
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)
    logger.setLevel(logging.INFO) 
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    gc.collect()
    wandb.login()

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    # load dataset
    roam_dataset = TextDataset(tokenizer , 
                               'context_training_pairs_22_03_27.csv',
                               block_size=1024)
    data_collator = DefaultDataCollator()
    # load model
    # set up trainer

    config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
    config.gradient_checkpointing = True
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B",config=config)


    trainer = ConditionalTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=roam_dataset,
        )
    gc.collect()
    trainer.train()
    trainer.save_model()
    wandb.finish()

main()

if __name__ == "__main__":
    main()
