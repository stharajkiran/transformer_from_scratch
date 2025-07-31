import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from config import get_config, get_weights_file_path
from tqdm import tqdm
import warnings

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
 
def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
    else:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    #build tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # keep 90% for train and 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    #  create train and val datasets
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # Get max length of the sentences in the dataset
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print(f"Max length of source sentences: {max_len_src}")
    print(f"Max length of target sentences: {max_len_tgt}")
    
    # create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(config['model_folder']).mkdir(parents= True, exist_ok= True)
    
    # Get the data loaders
    train_dataloader, val_dataloader,  tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # create the model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # Tenserboard
    writer = SummaryWriter(config['experiment_name'])
    
    # get the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9 )
    
    initial_epoch = 0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        if Path(model_filename).exists():
            state = torch.load(model_filename)
            initial_epoch = state['epoch']
            optimizer.load_state_dict(state['optimizer'])
            global_step = state['global_step']
        else:
            print(f"Model {model_filename} not found. Starting from scratch.")
    
    # loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index= tokenizer_src.token_to_id('[PAD]'), label_smoothing = 0.1)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc = f"processing epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # source sentence (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # target sentence (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # source sentence mask (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # target sentence mask (B, 1, seq_len, seq_len)

             # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)
            label = batch['label'].to(device) # target sentence (B, seq_len)
            # calculate the loss
            # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # log the loss to tensorboard
            writer.add_scalar('loss', loss.item(), global_step)
            writer.flush()

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
        
        # save the modeel every epoch
        model_filename = get_weights_file_path(config, f"{epoch:0.2d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # Load the configuration
    config = get_config()
    train_model(config)
