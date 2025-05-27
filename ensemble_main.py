"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from tqdm import tqdm
from typing import Dict, List, Union
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool, get_embeddings_wav2vec
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=FutureWarning)


class MoEGate(nn.Module):
    def __init__(self, input_dim, num_experts=2, hidden_dim=128, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        self.temperature = 1.0

        if self.use_conv:
            self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
            self.fc1 = nn.Linear(4 * input_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # x: [B, D]
        if self.use_conv:
            x = x.unsqueeze(1)  # [B, 1, D]
            x = F.relu(self.conv(x))  # [B, 4, D]
            x = x.view(x.size(0), -1)  # [B, 4*D]
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        weights = F.softmax(logits / self.temperature, dim=1)
        return weights


class CombinedModel(nn.Module):
    def __init__(self, model1, model2, dim1, dim2, num_classes, use_conv_gate=True):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.feature_dim = dim1 + dim2
        self.num_classes = num_classes

        self.gate = MoEGate(input_dim=self.feature_dim,
                            num_experts=2,
                            hidden_dim=128,
                            use_conv=use_conv_gate)

        # Optional: additional processing layer after gating (like in the paper)
        self.post_gate_conv = nn.Sequential(
            nn.Linear(256 + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.proj1 = nn.Linear(dim1, 256)
        self.proj2 = nn.Linear(dim2, 256)

    def forward(self, x, Freq_aug=None):
        # Extract features from each expert
        feat1 = self.model1.extract_features(x)
        feat2 = self.model2.extract_features(x)
        proj_feat1 = self.proj1(feat1)
        proj_feat2 = self.proj2(feat2)
        combined_feat = torch.cat([feat1, feat2], dim=1)  # [B, D1 + D2]

        # Get per-sample weights for experts
        weights = self.gate(combined_feat)  # [B, 2]

        # Get expert predictions
        _, out1 = self.model1(x, Freq_aug=Freq_aug)
        _, out2 = self.model2(x, Freq_aug=Freq_aug)

        # Combine predictions via expert weights
        out = weights[:, 0:1] * out1 + weights[:, 1:2] * out2  # [B, C]

        # Optional convolutional layer after gating (from MoE paper)
        weighted_feat = weights[:, 0:1] * proj_feat1 + weights[:, 1:2] * proj_feat2
        post_input = torch.cat([weighted_feat, out], dim=1)
        out = self.post_gate_conv(post_input)

        return combined_feat, out


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config1, "r") as f_json1:
        config1 = json.loads(f_json1.read())
    model_config1 = config1["model_config"]

    # Load second model config
    with open(args.config2, "r") as f_json2:
        config2 = json.loads(f_json2.read())
    model_config2 = config2["model_config"]
    config = config1

    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["LA", "PA", "DF"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2019.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config1))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config1, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    if args.wav2vec:
        get_embeddings_wav2vec(config['database_path'], 'train_wav2vec_embs.pkl', trn_loader)
        get_embeddings_wav2vec(config['database_path'], 'dev_wav2vec_embs.pkl', dev_loader)
        get_embeddings_wav2vec(config['database_path'], 'eval_wav2vec_embs.pkl', eval_loader)
        return

    # define model architecture
    model1 = get_model(model_config1, device)
    model1.load_state_dict(torch.load(config1["model_path"], map_location=device))
    model1.eval()
    model2 = get_model(model_config2, device)
    model2.load_state_dict(torch.load(config2["model_path"], map_location=device))
    model2.eval()

    for p in model1.parameters():
        p.requires_grad = False
    for p in model2.parameters():
        p.requires_grad = False

    sample_batch = next(iter(trn_loader))
    sample_input = sample_batch[0].to(device)

    with torch.no_grad():
        feat1 = model1.extract_features(sample_input)
        feat2 = model2.extract_features(sample_input)
        feature_dim1 = feat1.shape[1]
        feature_dim2 = feat2.shape[1]
    num_classes = model1.out_layer.out_features
    model = CombinedModel(model1, model2, feature_dim1, feature_dim2, num_classes).to(device)

    # evaluates pretrained model and exit script
    if args.eval:
        # Load configs
        with open(args.config1, "r") as f1, open(args.config2, "r") as f2:
            config1 = json.load(f1)
            config2 = json.load(f2)

        model_config1 = config1["model_config"]
        model_config2 = config2["model_config"]
        config = config1  # ensure consistent use

        # Load base models
        model1 = get_model(model_config1, device)
        model2 = get_model(model_config2, device)
        model1.load_state_dict(torch.load(config1["model_path"], map_location=device))
        model2.load_state_dict(torch.load(config2["model_path"], map_location=device))
        model1.eval()
        model2.eval()
        for p in model1.parameters():
            p.requires_grad = False
        for p in model2.parameters():
            p.requires_grad = False

        # Get feature dims from eval set
        sample_input = next(iter(eval_loader))[0].to(device)
        with torch.no_grad():
            feature_dim1 = model1.extract_features(sample_input).shape[1]
            feature_dim2 = model2.extract_features(sample_input).shape[1]
        num_classes = model1.out_layer.out_features

        # Build combined model
        model = CombinedModel(model1, model2, feature_dim1, feature_dim2, num_classes).to(device)

        # Load trained ensemble weights
        model.load_state_dict(torch.load(config1["ens_path"], map_location=device))
        print("Model loaded : {}".format(config1["ens_path"]))

        model.eval()

        # Reconstruct optimizer for SWA logic
        optim_config = config["optim_config"]
        optim_config["epochs"] = config["num_epochs"]
        optim_config["steps_per_epoch"] = len(trn_loader)
        optimizer, _ = create_optimizer(model.parameters(), optim_config)
        optimizer_swa = SWA(optimizer)

        # Optional: if you used SWA, update BN stats
        optimizer_swa.bn_update(trn_loader, model, device=device)

        # Run evaluation
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device, eval_score_path, eval_trial_path)
        eval_eer, eval_tdcf = calculate_tDCF_EER(
            cm_scores_file=eval_score_path,
            asv_score_file=database_path / config["asv_score_path"],
            output_file=model_tag / "loaded_model_t-DCF_EER.txt"
        )
        print("DONE.")
        print(f"EER: {eval_eer:.3f}, t-DCF: {eval_tdcf:.5f}")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_eval_eer = 100.
    best_dev_tdcf = 1.
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)
    loss_history = []
    dev_loss_history = []

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        loss_history.append(running_loss)
        print("Training loss: " + str(running_loss))

        if epoch == 0 or (epoch + 1) % 3 == 0:
            dev_loss = produce_evaluation_file(dev_loader, model, device,
                                    metric_path/"dev_score.txt", dev_trial_path)
            print("Dev Loss: " + str(dev_loss))
            dev_loss_history.append(dev_loss)
            dev_eer, dev_tdcf = calculate_tDCF_EER(
                cm_scores_file=metric_path/"dev_score.txt",
                asv_score_file=database_path/config["asv_score_path"],
                output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
                printout=False)
            print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_tdcf:{:.5f}".format(
                running_loss, dev_eer, dev_tdcf))
            writer.add_scalar("loss", running_loss, epoch)
            writer.add_scalar("dev_eer", dev_eer, epoch)
            writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

            best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
            if best_dev_eer >= dev_eer:
                print("best model find at epoch", epoch)
                best_dev_eer = dev_eer
                torch.save(model.state_dict(),
                       model_save_path / "best.pth")

                print("Saving epoch {} for swa".format(epoch))
                optimizer_swa.update_swa()
                n_swa_update += 1
            writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
            writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    with open("losses.txt", 'w') as file:
        file.write(str(loss_history))
        file.write('\n')
        file.write(str(dev_loss_history))

    plt.plot(range(config["num_epochs"]), loss_history, marker='o', label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()

    print("Start final evaluation")
    best_model_path = model_save_path / "best.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
        torch.save(model.state_dict(), model_save_path / "swa_best.pth")
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}".format(eval_eer, eval_tdcf))
    f_log.close()


def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    prefix_2019 = "ASVspoof2019.{}".format(track)

    trn_database_path = database_path / "ASVspoof2019_{}_train/".format(track)
    dev_database_path = database_path / "ASVspoof2019_{}_dev/".format(track)
    eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)

    trn_list_path = (database_path /
                     "ASVspoof2019_{}_cm_protocols/{}.cm.train.trn.txt".format(
                         track, prefix_2019))
    dev_trial_path = (database_path /
                      "ASVspoof2019_{}_cm_protocols/{}.cm.dev.trl.txt".format(
                          track, prefix_2019))
    eval_trial_path = (
        database_path /
        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))

    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,is_eval=False)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                            is_train=False,
                                            is_eval=False)
    dev_set = Dataset_ASVspoof2019_train(list_IDs=file_dev,
                                            base_dir=dev_database_path,
                                            labels=d_label_dev, is_eval=False)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    d_label_eval, file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=False)
    eval_set = Dataset_ASVspoof2019_train(list_IDs=file_eval,
                                             base_dir=eval_database_path,
                                             labels=d_label_eval, is_eval=True)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return trn_loader, dev_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    total_loss = 0.0
    num_batches = 0
    for batch_x, batch_y, utt_id in tqdm(data_loader, desc="Processing batches"):
        batch_x = batch_x.to(device)
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        batch_loss = criterion(batch_out, batch_y)  # Calculate batch loss
        total_loss += batch_loss.item()
        num_batches += 1

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

    average_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return average_loss


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y, _ in tqdm(trn_loader, desc="Processing"):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument('--config1', type=str, required=True, help="Path to first model config")
    parser.add_argument('--config2', type=str, required=True, help="Path to second model config")
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--wav2vec",
                        action="store_true",
                        help="when this flag is given, compute the wav2vec2 embeddings")
    main(parser.parse_args())
