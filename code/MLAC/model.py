# -*- coding: utf-8 -*-
import logging
import random

from code.MLAC.network import MultiLevelAttentionCNN, novel_distance_loss
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class REModel(object):
    def __init__(self, opt, embedding=None, state_dict=None):
        self.opt = opt
        self.updates = state_dict["updates"] if state_dict else 0

        # build network
        self.network = MultiLevelAttentionCNN(opt, embedding=embedding)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs")
            self.network = torch.nn.DataParallel(self.network)

        if state_dict:
            old_state = list(state_dict["network"].keys())
            if torch.cuda.device_count() > 1:
                new_state = set(self.network.module.state_dict().keys())
            else:
                new_state = set(self.network.state_dict().keys())

            for k in old_state:
                if k not in new_state:
                    del state_dict["network"][k]

            if torch.cuda.device_count() > 1:
                self.network.module.load_state_dict(state_dict["network"])
            else:
                self.network.load_state_dict(state_dict["network"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        # build optimizer
        self.opt_state_dict = state_dict["optimizer"] if state_dict else None
        self.build_optimizer()

    def build_optimizer(self):
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        print("parameters required grad: {}".format(len(parameters)))

        if self.opt["optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(parameters, lr=self.opt["learning_rate"],
                                             momentum=self.opt["momentum"],
                                             weight_decay=self.opt["weight_decay"])
        elif self.opt["optimizer"] == "adamax":
            self.optimizer = torch.optim.Adamax(parameters, weight_decay=self.opt["weight_decay"])
        else:
            raise RuntimeError("Unsupported optimizer: {}".format(self.opt["optimizer"]))

    def update(self, ex):
        # train mode
        self.network.train()

        # transfer to gpu
        inputs = [e.to(self.device) for e in ex[:-1]]
        target_rel = ex[-1].to(self.device)

        # run forward
        wo, rel_weight = self.network(*inputs)

        # compute loss and accuracies
        loss = novel_distance_loss(wo, rel_weight, target_rel, self.opt["num_relations"])

        # clear gradients and run backward
        self.optimizer.zero_grad()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.opt["grad_clipping"])

        # update parameters
        self.optimizer.step()
        self.updates += 1

    def predict(self, ex):
        # eval mode
        self.network.eval()

        # transfer to gpu
        inputs = [e.to(self.device) for e in ex[:-1]]

        # run forward
        with torch.no_grad():
            wo, rel_weight = self.network(*inputs)
            wo_norm = F.normalize(wo)  # [b, dc] Wo/||Wo||
            b, dc = wo_norm.size()
            wo_norm_tile = wo_norm.unsqueeze(1).repeat(1, self.opt["num_relations"], 1)  # [b, nr, dc]
            batched_rel_w = F.normalize(rel_weight).unsqueeze(0).repeat(b, 1, 1)  # [b, nr, dc]
            all_distance = torch.norm(wo_norm_tile - batched_rel_w, p=2, dim=2)  # [b, nr]

            predict_prob, predict = torch.min(all_distance, dim=1)

        return predict

    def save(self, epoch, scores, filename):
        precision, recall, f1, best_f1 = scores
        state_dict = self.network.module.state_dict() if torch.cuda.device_count() > 1 else self.network.state_dict()
        params = {
            "state_dict": {
                "network": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
            },
            "config": self.opt,
            "epoch": epoch,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "best_eval": best_f1,
            "random_state": random.getstate(),
            "torch_state": torch.random.get_rng_state(),
            "torch_cuda_state": torch.cuda.get_rng_state()

        }

        try:
            torch.save(params, filename)
            logger.info('model saved to {}'.format(filename))
        except BaseException:
            logger.warning('[ WARN: Saving failed... continuing anyway. ]')



