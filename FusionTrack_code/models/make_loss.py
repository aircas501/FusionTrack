
import torch.nn.functional as F
from .loss.softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .loss.triplet_loss import TripletLoss
from .loss.center_loss import CenterLoss


def make_loss(cfg):    # modified by gu
    sampler = cfg["SAMPLER"]
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=cfg["NUM_REID"], feat_dim=feat_dim, use_gpu=True)  # center loss
    if 'triplet' in cfg["METRIC_LOSS_TYPE"]:
        if cfg["NO_MARGIN"]:
            triplet = TripletLoss()
            print("using soft triplet loss for training")
        else:
            triplet = TripletLoss(cfg["SOLVER"])  # triplet loss
            print("using triplet loss with margin:{}".format(cfg["MARGIN"]))
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg["METRIC_LOSS_TYPE"]))

    if cfg["IF_LABELSMOOTH"] == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=cfg["NUM_REID"])
        print("label smooth on, numclasses:", cfg["NUM_REID"])

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif cfg["SAMPLER"] == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg["METRIC_LOSS_TYPE"] == 'triplet':
                if cfg["IF_LABELSMOOTH"] == 'on':
                    if isinstance(score, list):
                        ID_LOSS = [xent(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * xent(score[0], target)
                    else:
                        ID_LOSS = xent(score, target)

                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg["ID_LOSS_WEIGHT"] * ID_LOSS + \
                               cfg["TRIPLET_LOSS_WEIGHT"] * TRI_LOSS
                else:
                    if isinstance(score, list):
                        ID_LOSS = [F.cross_entropy(scor, target) for scor in score[1:]]
                        ID_LOSS = sum(ID_LOSS) / len(ID_LOSS)
                        ID_LOSS = 0.5 * ID_LOSS + 0.5 * F.cross_entropy(score[0], target)
                    else:
                        ID_LOSS = F.cross_entropy(score, target)
                    if target.shape[0] < 2:
                        return ID_LOSS
                    if isinstance(feat, list):
                            TRI_LOSS = [triplet(feats, target)[0] for feats in feat[1:]]
                            TRI_LOSS = sum(TRI_LOSS) / len(TRI_LOSS)
                            TRI_LOSS = 0.5 * TRI_LOSS + 0.5 * triplet(feat[0], target)[0]
                    else:
                            TRI_LOSS = triplet(feat, target)[0]

                    return cfg["ID_LOSS_WEIGHT"] * ID_LOSS + \
                               cfg["TRIPLET_LOSS_WEIGHT"] * TRI_LOSS
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg["METRIC_LOSS_TYPE"]))

    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg["SAMPLER"]))
    return loss_func, center_criterion


