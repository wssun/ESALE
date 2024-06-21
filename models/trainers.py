import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
import tqdm


class Trainer:

    def __init__(self, model,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 with_mlm=True, with_ulm=True, with_awp=True,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = False, cuda_devices=None, log_freq: int = 5, eps=1e-8):

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.with_mlm = with_mlm
        self.with_ulm = with_ulm
        self.with_awp = with_awp
        # Initialize the BERT Language Model. This BERT model will be saved every epoch
        self.model = model.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1 and len(cuda_devices) > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.ids = torch.tensor(range(128)).to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        self.optim = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
        self.criterion = nn.CrossEntropyLoss(ignore_index=1).to(self.device)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement()
              for p in self.model.parameters()]))

    def train(self, epoch, is_test=False):
        if not is_test:
            self.model.train()
            data_iter = tqdm.tqdm(enumerate(self.train_data),
                                  desc="EP%d_train" % (epoch),
                                  total=len(self.train_data),
                                  bar_format="{l_bar}{r_bar}")
        else:
            self.model.eval()
            data_iter = tqdm.tqdm(enumerate(self.test_data),
                                  desc="EP%d_test" % (epoch),
                                  total=len(self.test_data),
                                  bar_format="{l_bar}{r_bar}")
        avg_loss = 0
        iter_num = 0
        total_mlm_correct = 0
        total_ulm_correct = 0
        total_awp_correct = 0
        total_mlm_element = 0
        total_ulm_element = 0
        total_awp_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            batch_size = data["code_ids"].shape[0]
            if not is_test:
                awp_output, ulm_scores, mlm_scores = self.model(
                    data["code_ids"], data["comments_ids"], data["masked_comments_ids"])
            else:
                with torch.no_grad():
                    awp_output, ulm_scores, mlm_scores = self.model(
                        data["code_ids"], data["comments_ids"], data["masked_comments_ids"])
            active_mask = data["comments_ids"].ne(1)

            if self.with_mlm:
                masked_labels = torch.masked_select(
                    data["comments_ids"], data["masked_ids"].bool())
                masked_logits = torch.index_select(
                    mlm_scores[0], 0, self.ids[data["masked_ids"][0].ne(0)])  # get the first data from batch
                for i in range(batch_size-1):
                    masked_logits = torch.cat((masked_logits, torch.index_select(
                        mlm_scores[i+1], 0, self.ids[data["masked_ids"][i+1].ne(0)])), dim=0)  # get the masked data

                mlm_loss = self.criterion(masked_logits, masked_labels)
                masked_preds = masked_logits.argmax(dim=-1)
                mlm_correct = masked_preds.eq(
                    masked_labels).sum().item()
            else:
                mlm_correct = 0
                mlm_loss = 0

            if self.with_ulm:
                ulm_correct = ulm_scores.argmax(
                    dim=-1)[active_mask].eq(data["comments_ids"][active_mask]).sum().item()

                ulm_loss = self.criterion(
                    ulm_scores.permute(0, 2, 1), data["comments_ids"])
            else:
                ulm_correct = 0
                ulm_loss = 0

            if self.with_awp:
                awp_correct = awp_output.argmax(
                    dim=-1).eq(data["awp_label"]).sum().item()
                awp_loss = self.criterion(awp_output, data["awp_label"])
            else:
                awp_correct = 0
                awp_loss = 0

            loss = mlm_loss + ulm_loss + awp_loss

            if not is_test:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            
            total_ulm_element += data["comments_ids"][active_mask].nelement()
            total_ulm_correct += ulm_correct
            total_mlm_correct += mlm_correct
            if self.with_mlm:
                total_mlm_element += masked_labels.shape[0]
            else:
                total_mlm_element  = total_ulm_element # anything but 0 
            total_awp_correct += awp_correct
            total_awp_element += data["awp_label"].nelement()

            avg_loss += loss.item()
            iter_num += 1

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_mlm_acc": total_mlm_correct / total_mlm_element * 100,
                "avg_ulm_acc": total_ulm_correct / total_ulm_element * 100,
                "avg_awp_acc": total_awp_correct / total_awp_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        if not is_test:
            print("EP%d_train" % (epoch),
                  " total_mlm_acc= ", total_mlm_correct * 100.0 / total_mlm_element,
                  " total_awp_acc= ", total_awp_correct * 100.0 / total_awp_element,
                  " total_ulm_acc= ", total_ulm_correct * 100.0 / total_ulm_element)
        else:
            avg_loss = avg_loss/iter_num
            print("EP%d_test" % (epoch),
                  " total_mlm_acc= ", total_mlm_correct * 100.0 / total_mlm_element,
                  " total_awp_acc= ", total_awp_correct * 100.0 / total_awp_element,
                  " total_ulm_acc= ", total_ulm_correct * 100.0 / total_ulm_element,
                  " avg_loss= ", avg_loss)
            return avg_loss

    def save(self, epoch, output_dir):
        output_dir = os.path.join(output_dir,"unified_encoder_model")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir,"model.pth")
        torch.save(self.model.to(self.device), output_path)
        self.model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
