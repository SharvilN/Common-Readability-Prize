import torch
import config
import torch.nn as nn

from metrics import AverageMeter


class Trainer:

    def __init__(self, model, log_interval, eval_internal, epochs, 
                optimizer, lr_scheduler):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.log_interval = log_interval
        self.eval_interval = eval_internal
        self.evaluator = Evaluator(self.model)


    def train(self, train_loader, valid_loader, result_dict, fold):
        for epoch in range(self.epochs):
            result_dict["epoch"] = epoch
            result_dict = self._train_loop_for_one_epoch(
                epoch=epoch,
                model=self.model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                result_dict=result_dict
            )
        
        return result_dict

    def _train_loop_for_one_epoch(self, epoch, train_loader, valid_loader, result_dict):
        self.model.train()
        losses = AverageMeter()
        for batch_idx, batch in enumerate(train_loader):
            input_ids, = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            token_type_ids = batch["token_type_ids"].to(config.DEVICE)
            label = batch["label"].to(config.DEVICE)
            model = model.to(config.DEVICE)

            loss = self._train_loop_for_one_step(
                input_ids,
                attention_mask,
                token_type_ids,
                label
            )
            losses.update(loss.item(), input_ids.size(0))

            if batch_idx % self.log_interval == 0:
                print(f"Epoch={epoch}, Avg Loss={losses.avg}, Batch Idx={batch_idx}")

            if batch_idx % self.eval_interval == 0:
                result_dict = self.evaluator.evaluate(
                    valid_loader=valid_loader,
                    result_dict=result_dict
                )
    
    def _train_loop_for_one_step(self, input_ids, attention_mask, token_type_ids, label):
        self.optimizer.zero_grad()
        outputs = self.model(input_ids, attention_mask, token_type_ids, label)
        loss, logits = outputs[:2]
        loss.backward()
        self.optimizer.step()
        return loss



class Evaluator:

    def __init__(self, model):
        self.model = model

    def evaluate(self, valid_loader, result_dict):
        self.model.eval()
        losses = AverageMeter()
        for batch_idx, batch in enumerate(valid_loader):
            input_ids, = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            token_type_ids = batch["token_type_ids"].to(config.DEVICE)
            model = model.to(config.DEVICE)

            loss = self._eval_loop_for_one_step(
                input_ids,
                attention_mask,
                token_type_ids
            )
            losses.update(loss.item(), input_ids.size(0))
        return result_dict        

    def _eval_loop_for_one_step(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask, token_type_ids)
            loss, logits = outputs[:2]
            return loss