import torch
from model import create_mask
from tqdm.notebook import tqdm
from itertools import repeat
from torch import Tensor


@torch.no_grad()
def generate(model, tokenizer, batch_size, pad_idx, prefix=None, max_len=384):
    model.eval()
    if prefix is None:
        prefix = torch.full((batch_size, 1), fill_value=tokenizer.bos_id()).to(next(model.parameters()).device)
    count = max_len - prefix.shape[-1]
    for _ in range(count):
        prefix = prefix.clone().detach()
        tgt_mask, tgt_padding_mask = create_mask(prefix, pad_idx, device='cuda')
        output_logits = torch.nn.functional.softmax(model.forward(prefix, tgt_mask, tgt_padding_mask)[:, -1, :], dim=-1)
        prefix = torch.cat((prefix, torch.multinomial(output_logits, 1)), dim=-1)
    return prefix


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def evaluate(model, dataloader, loss_fn, pad_idx, device):
    model.eval()
    losses = 0
    for tgt, _ in tqdm(dataloader):
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(tgt_input, tgt_mask, tgt_padding_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(dataloader)


def train(n_epochs, model, pad_idx, optimizer, train_loader, val_loader, device, dataset, scheduler, wandb_instance, len_epoch=10000, log_step=500):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    train_loader_inf = inf_loop(train_loader)
    best_loss = 1e6
    for epoch in range(n_epochs):
        losses = 0
        for i, (tgt, _) in enumerate(tqdm(train_loader_inf, desc="train", total=len_epoch)):
            model.train()
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(tgt_input, tgt_mask, tgt_padding_mask)
                tgt_out = tgt[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            losses += loss.item()
            cur_step = epoch * len_epoch + i

            if i % len_epoch == 0:
                val_loss = evaluate(model, val_loader, loss_fn, pad_idx, device)
                if val_loss < best_loss:
                    print(f'checkpoint at {cur_step}')
                    torch.save(model.state_dict(), 'checkpoint.pth')
                    best_loss = val_loss
                prefix = generate(model, dataset.sp_model, 3, pad_idx)
                texts = dataset.ids2text(prefix)
                for t_num, text in enumerate(texts):
                    wandb_instance.log({
                        f'stepN{cur_step}_textN{t_num}': wandb_instance.Html(text)
                    }, step=cur_step)
                wandb_instance.log({
                    'train_loss': losses / (i % log_step + 1),
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0]
                }, step=cur_step)
                print(f"epoch: {epoch}, train loss: {(losses / (i % log_step + 1)):.3f}, val loss: {val_loss:.3f}")
                break

            if i % log_step == 0:
                wandb_instance.log({
                    'train_loss': losses / log_step,
                    'lr': scheduler.get_last_lr()[0]
                }, step=cur_step)
                print(f"epoch: {epoch}, train loss: {(losses / log_step):.3f}")
                losses = 0
            scheduler.step()
