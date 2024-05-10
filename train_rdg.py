import argparse
import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

import wandb
from data.data_collator import rdg_collate_fn
from data.dataset import RDGDataset
from s2igan.loss import KLDivergenceLoss, RSLoss
from s2igan.rdg import (
    DenselyStackedGenerator,
    DiscriminatorFor64By64,
    DiscriminatorFor128By128,
    DiscriminatorFor256By256,
    RelationClassifier,
)
from s2igan.rdg.utils import rdg_train_epoch
from s2igan.sen import ImageEncoder, SpeechEncoder
from s2igan.utils import set_non_grad

config_path = "conf"
config_name = "rdg_config"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    bs = cfg.data.general.batch_size
    attn_heads = cfg.model.speech_encoder.attn_heads
    attn_dropout = cfg.model.speech_encoder.attn_dropout
    rnn_dropout = cfg.model.speech_encoder.rnn_dropout
    lr = cfg.optimizer.lr
    train_version_name = cfg.kaggle.name
    train_project_name = cfg.kaggle.project
    if cfg.experiment.log_wandb:
        wandb.init(project=f"{train_project_name}", name=f"{train_version_name}_RDG_bs{bs}_lr{lr}_attn{attn_heads}_ad{attn_dropout}_rd{rnn_dropout}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    multi_gpu = torch.cuda.device_count() > 1
    device_ids = list(range(torch.cuda.device_count()))

    train_set = RDGDataset(**cfg.data.train)
    test_set = RDGDataset(**cfg.data.test)

    bs = cfg.data.general.batch_size
    nwkers = cfg.data.general.num_workers
    train_dataloader = DataLoader(
        train_set, bs, shuffle=True, num_workers=nwkers, collate_fn=rdg_collate_fn
    )
    test_dataloder = DataLoader(
        test_set, bs, shuffle=False, num_workers=nwkers, collate_fn=rdg_collate_fn
    )

    generator = DenselyStackedGenerator(**cfg.model.generator)
    discrminator_64 = DiscriminatorFor64By64(**cfg.model.discriminator)
    discrminator_128 = DiscriminatorFor128By128(**cfg.model.discriminator)
    discrminator_256 = DiscriminatorFor256By256(**cfg.model.discriminator)
    relation_classifier = RelationClassifier(**cfg.model.relation_classifier)
    image_encoder = ImageEncoder(**cfg.model.image_encoder)
    speech_encoder = SpeechEncoder(**cfg.model.speech_encoder)

    if cfg.ckpt.image_encoder:
        print("Loading Image Encoder state dict...")
        print(image_encoder.load_state_dict(torch.load(cfg.ckpt.image_encoder)))
        set_non_grad(image_encoder)
    if cfg.ckpt.speech_encoder:
        print("Loading Speech Encoder state dict...")
        print(speech_encoder.load_state_dict(torch.load(cfg.ckpt.speech_encoder)))
        set_non_grad(speech_encoder)
    if cfg.ckpt.generator:
        print("Loading Generator state dict...")
        print(generator.load_state_dict(torch.load(cfg.ckpt.generator)))
        # set_non_grad(generator)
    if cfg.ckpt.discrminator_64:
        print("Loading Discriminator 64 state dict...")
        print(discrminator_64.load_state_dict(torch.load(cfg.ckpt.discrminator_64)))
        # set_non_grad(discrminator_64)
    if cfg.ckpt.discrminator_128:
        print("Loading Discriminator 128 state dict...")
        print(discrminator_128.load_state_dict(torch.load(cfg.ckpt.discrminator_128)))
        # set_non_grad(discrminator_128)
    if cfg.ckpt.discrminator_256:
        print("Loading Discriminator 256 state dict...")
        print(discrminator_256.load_state_dict(torch.load(cfg.ckpt.discrminator_256)))
        # set_non_grad(discrminator_256)
    if cfg.ckpt.relation_classifier:
        print("Loading Relation Classifier state dict...")
        print(relation_classifier.load_state_dict(torch.load(cfg.ckpt.relation_classifier)))
        # set_non_grad(relation_classifier)

    if multi_gpu:
        generator = nn.DataParallel(generator, device_ids=device_ids)
        discrminator_64 = nn.DataParallel(discrminator_64, device_ids=device_ids)
        #discrminator_128 = nn.DataParallel(discrminator_128, device_ids=device_ids)
        #discrminator_256 = nn.DataParallel(discrminator_256, device_ids=device_ids)
        relation_classifier = nn.DataParallel(
            relation_classifier, device_ids=device_ids
        )
        image_encoder = nn.DataParallel(image_encoder, device_ids=device_ids)
        speech_encoder = nn.DataParallel(speech_encoder, device_ids=device_ids)

    generator = generator.to(device)
    discrminator_64 = discrminator_64.to(device)
    # discrminator_128 = discrminator_128.to(device)
    # discrminator_256 = discrminator_256.to(device)
    relation_classifier = relation_classifier.to(device)
    image_encoder = image_encoder.to(device)
    speech_encoder = speech_encoder.to(device)


    discriminators = {
        64: discrminator_64,
        # 128: discrminator_128,
        # 256: discrminator_256,
    }

    models = {
        "gen": generator,
        "disc": discriminators,
        "rs": relation_classifier,
        "ied": image_encoder,
        "sed": speech_encoder,
    }

    optimizer_generator = torch.optim.AdamW(generator.parameters(), **cfg.optimizer)
    optimizer_discrminator = {
        key: torch.optim.AdamW(discriminators[key].parameters(), **cfg.optimizer)
        for key in discriminators.keys()
    }
    optimizer_rs = torch.optim.AdamW(relation_classifier.parameters(), **cfg.optimizer)
    optimizers = {
        "gen": optimizer_generator,
        "disc": optimizer_discrminator,
        "rs": optimizer_rs,
    }

    if cfg.ckpt.optimizer.optimizer_generator:
        optimizer_rs.load_state_dict(torch.load(cfg.ckpt.optimizer.optimizer_rs))
        optimizer_generator.load_state_dict(torch.load(cfg.ckpt.optimizer.optimizer_generator))
        optimizer_discrminator[64].load_state_dict(torch.load(cfg.ckpt.optimizer.optimizer_discrminator_64))
        # optimizer_discrminator[128].load_state_dict(torch.load(cfg.ckpt.optimizer.optimizer_discrminator_128))
        # optimizer_discrminator[256].load_state_dict(torch.load(cfg.ckpt.optimizer.optimizer_discrminator_256))
        
    steps_per_epoch = len(train_dataloader)
    sched_dict = dict(
        epochs=cfg.experiment.max_epoch,
        steps_per_epoch=steps_per_epoch,
        max_lr=cfg.optimizer.lr,
        pct_start=cfg.scheduler.pct_start,
    )
    schedulers = {
        "gen": torch.optim.lr_scheduler.OneCycleLR(optimizer_generator, **sched_dict),
        "disc": {
            key: torch.optim.lr_scheduler.OneCycleLR(
                optimizer_discrminator[key], **sched_dict
            )
            for key in discriminators.keys()
        },
        "rs": torch.optim.lr_scheduler.OneCycleLR(optimizer_rs, **sched_dict),
    }

    criterions = {
        "kl": KLDivergenceLoss().to(device),
        "rs": RSLoss().to(device),
        "bce": nn.BCELoss().to(device),
        "ce": nn.CrossEntropyLoss().to(device),
    }

    log_wandb = cfg.experiment.log_wandb
    specific_params = cfg.experiment.specific_params
    if cfg.experiment.train:
        for epoch in range(cfg.experiment.max_epoch):
            train_result = rdg_train_epoch(
                models,
                train_dataloader,
                optimizers,
                schedulers,
                criterions,
                specific_params,
                device,
                epoch,
                log_wandb,
            )

            save_dir = "/kaggle/working/save_ckpt"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(speech_encoder.state_dict(), os.path.join(save_dir, "speech_encoder.pt"))
            torch.save(image_encoder.state_dict(), os.path.join(save_dir, "image_encoder.pt"))
            torch.save(generator.state_dict(), os.path.join(save_dir, "generator.pt"))
            torch.save(discrminator_64.state_dict(), os.path.join(save_dir, "discrminator_64.pt"))
            #torch.save(discrminator_128.state_dict(), os.path.join(save_dir, "discrminator_128.pt"))
            #torch.save(discrminator_256.state_dict(), os.path.join(save_dir, "discrminator_256.pt"))
            torch.save(relation_classifier.state_dict(), os.path.join(save_dir, "relation_classifier.pt"))
            torch.save(optimizer_generator.state_dict(), os.path.join(save_dir, "optimizer_generator.pt"))
            # torch.save(optimizer_discrminator.state_dict(), os.path.join(save_dir, "optimizer_discrminator.pt"))
            torch.save(optimizer_rs.state_dict(), os.path.join(save_dir, "optimizer_rs.pt"))
            for key in optimizer_discrminator.keys():
                torch.save(optimizer_discrminator[key].state_dict(), os.path.join(save_dir, f"optimizer_discrminator_{key}.pt"))
        
    print("Train result:", train_result)


if __name__ == "__main__":
    load_dotenv()
    main()