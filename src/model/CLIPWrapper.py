import open_clip
import torch
import torch.nn as nn


class CLIPWrapper(torch.nn.Module):
    def __init__(self, model_config, classifier_head, clip, preprocess_fn, device):
        super().__init__()
        self.model_config = model_config
        self.device = device
        self.clip = clip
        self.preprocess_fn = preprocess_fn

        if clip is None:
            clip, _, preprocess_fn = open_clip.create_model_and_transforms(
                model_config.pretrained_model,
                pretrained=model_config.pretraining_mixture,
            )
            self.clip = clip
            self.preprocess_fn = preprocess_fn
        if model_config.freeze_backbone:
            for name, param in self.clip.named_parameters():
                param.requires_grad = False
        else:
            for name, param in self.clip.named_parameters():
                param.requires_grad = True

        if classifier_head is not None:
            self.classifier_head = classifier_head
            # Freeze the classifier head since we are using open vocabulary to
            # classify
            self.classifier_head.weight.requires_grad = False

        self.loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch):
        # (batch_size, d_emb)
        img_emb = self.clip.encode_image(batch["image"])
        img_logits = self.classifier_head(img_emb)

        perImage_loss = self.loss(img_logits, batch["lbl"])

        loss = torch.mean(perImage_loss)

        return loss, {"loss": loss.cpu().item()}

    def predict(self, batch):
        # (batch_size, num_lbl)
        clip_emb = self.clip.encode_image(batch["image"])
        img_logits = self.classifier_head(clip_emb)
        lbl_prob = torch.softmax(img_logits, dim=-1)
        pred_prob, preb_lbl = torch.max(lbl_prob, dim=-1)
        return (
            preb_lbl.cpu().numpy().tolist(),
            pred_prob.cpu().numpy().tolist(),
        )
