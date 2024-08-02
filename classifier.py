
import torch
from torchmetrics import Metric
import pytorch_lightning as pl
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics.classification import F1Score, Precision, Recall
import torch.optim as optim




class SaveBestModel(pl.Callback):
    def __init__(self, filepath = './best_model/', monitor='val_loss', save_best_only=True):
        
        
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf')

    def on_validation_end(self, trainer, pl_module):
        current = trainer.callback_metrics[self.monitor].item()
        
        if self.save_best_only:
            if current < self.best:
                self.best = current
                torch.save(pl_module.state_dict(), self.filepath)
                print(f"Validation {self.monitor} improved to {current:.4f}, saving model to {self.filepath}")


class pAUC(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_preds = []
        self.labels = []

    
    def update(self, preds, target):
        # Convert predictions to class labels
        preds = torch.argmax(preds, dim=1)
        self.total_preds.extend(preds)
        self.labels.extend(target)
        
    def compute(self, min_tpr = 0.8):
        preds = torch.tensor(self.total_preds)
        lbls = torch.tensor(self.labels)
        
        v_gt = abs(np.array(lbls.cpu())-1)
        v_pred = -1.0*np.array(preds.cpu())
    
        max_fpr = abs(1-min_tpr)
    
        # using sklearn.metric functions: (1) roc_curve and (2) auc
        fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
        if max_fpr is None or max_fpr == 1:
            return auc(fpr, tpr)
        if max_fpr <= 0 or max_fpr > 1:
            raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
            
        # Add a single point at max_fpr by linear interpolation
        stop = np.searchsorted(fpr, max_fpr, "right")
        x_interp = [fpr[stop - 1], fpr[stop]]
        y_interp = [tpr[stop - 1], tpr[stop]]
        tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
        fpr = np.append(fpr[:stop], max_fpr)
        partial_auc = auc(fpr, tpr)
    
        #     # Equivalent code that uses sklearn's roc_auc_score
        #     v_gt = abs(np.asarray(solution.values)-1)
        #     v_pred = np.array([1.0 - x for x in submission.values])
        #     max_fpr = abs(1-min_tpr)
        #     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
        #     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
        #     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
        #     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
        
        return torch.tensor(partial_auc)




class EfficientNetBinaryClassifier(nn.Module):
    def __init__(self, efficientnet, num_features):
        super(EfficientNetBinaryClassifier, self).__init__()

        self.efficientnet = efficientnet
        self.dropout = nn.Dropout()
        self.num_features = num_features
        
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.dropout(x)
        x = self.classifier(x.view(-1, self.num_features))
        return x

class EfficientNetBinaryClassifierLightning(pl.LightningModule):
    def __init__(self, efficientnet, val_loader, num_features, optim, lr=1e-3):
        super(EfficientNetBinaryClassifierLightning, self).__init__()
        self.model = EfficientNetBinaryClassifier(efficientnet, num_features)
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = lr
        self.train_accuracy = Accuracy('binary')
        self.val_accuracy = Accuracy('binary')
        num_classes = 2
        self.f1 = F1Score(num_classes=num_classes, average='weighted', task='binary')
        self.precision = Precision(num_classes=num_classes, average='weighted', task='binary')
        self.recall = Recall(num_classes=num_classes, average='weighted', task='binary')
        self.p_auc = pAUC()
        self.validation_dataloader = val_loader
        self.optim_choice = optim
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1).float()
        #pdb.set_trace()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = self.train_accuracy(preds, y.int())
        labels = y.int()
        f1 = self.f1(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_epoch = True, on_step = True)
        
        # self.eval()  # Switch to evaluation mode
        # # with torch.no_grad():  # Disable gradient computation
        # #     for val_batch_idx, val_batch in enumerate(self.validation_dataloader):
        # #         self.validation_step(val_batch, val_batch_idx)
        # # self.train()  # Switch back to training mode
        # x,y = self.validation_step()
        return loss

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1, 1).float()
        
        logits = self(x)
        #pdb.set_trace()
        loss = self.criterion(logits, y)
        preds = torch.sigmoid(logits) > 0.5
        acc = self.val_accuracy(preds, y.int())
        labels = y.int()
        self.f1(preds, labels)
        self.precision(preds, labels)
        self.recall(preds, labels)
        self.p_auc(preds, labels)
        self.log('val_loss', loss, prog_bar=True, on_epoch = True)
        self.log('val_acc', acc, prog_bar=True, on_epoch = True)
        self.log('val_f1', self.f1, prog_bar=True, on_epoch = True)
        self.log('val_precision', self.precision, prog_bar=True, on_epoch = True)
        self.log('val_recall', self.recall, prog_bar=True, on_epoch = True)
        self.log('val_pAUC', self.p_auc, prog_bar=True, on_epoch=True)
   
        

    def configure_optimizers(self):
        if self.optim_choice == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
