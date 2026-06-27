import os
import argparse
import torch.distributed
import torch.backends.cuda
import torch.backends.cudnn

from utils.utils import distributed_rank
from utils.utils import yaml_to_dict
from configs.utils import update_config


def parse_option():
    parser = argparse.ArgumentParser("Network training and evaluation script.", add_help=True)

    parser.add_argument("--git-version", type=str)

    # About system, e.g. GPUs
    parser.add_argument("--available-gpus", type=str, help="Available GPUs, like '0,1,2,3'.")
    parser.add_argument("--use-distributed", action="store_true", help="Use distributed training.")
    parser.add_argument("--use-checkpoint", action="store_true", help="Use gradient checkpoint to save GPU memory.")
    parser.add_argument("--checkpoint-level", type=int)
    parser.add_argument("--local_rank",type=int,default=0,help="Local rank for distributed training (used by torch.distributed.launch/torchrun).")


    # Running mode: training, evaluation, etc.
    parser.add_argument("--mode", type=str, help="Running mode.")

    # Only for result submission (inference)
    parser.add_argument("--submit-dir", type=str)
    parser.add_argument("--submit-model1", type=str, help="Main model checkpoint (tracking model).")
    parser.add_argument("--submit-model2", type=str, help="ReID model checkpoint.")
    parser.add_argument("--submit-data-split", type=str)

    # Only for model evaluation
    parser.add_argument("--eval-dir", type=str)
    parser.add_argument("--eval-mode", type=str)
    parser.add_argument("--eval-model", type=str)
    parser.add_argument("--eval-threads", type=int)
    parser.add_argument("--eval-port", type=int)
    parser.add_argument("--eval-data-split", type=str)

    # Pretrained model load
    parser.add_argument("--pretrained-model", type=str, help="Pretrained model path.")  # separate from resume1/resume2 for initial pretraining
    # Resume: load two checkpoints — MeMOTR (tracking) and ReID
    parser.add_argument("--resume1", type=str, help="Resume1 checkpoint path.")
    parser.add_argument("--resume2", type=str, help="Resume2 checkpoint path.")
    parser.add_argument("--resume-scheduler", type=str, help="Whether resume the training scheduler.")

    # About Paths:
    # Config file
    parser.add_argument("--config-path", type=str, help="Config file path.",
                        default="./configs/train_dancetrack.yaml")
    # Data Path:
    parser.add_argument("--data-root", type=str, help="Dataset root dir.")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data-path", type=str)
    # Log outputs:
    parser.add_argument("--outputs-dir", type=str, help="Outputs dir path.")

    # Data:
    parser.add_argument("--accumulation-steps", type=int, help="Gradient accumulation steps.")
    parser.add_argument("--batch-size", type=int, help="Batch size for training.")
    parser.add_argument("--coco-size", type=str)
    parser.add_argument("--overflow-bbox", type=str)
    parser.add_argument("--reverse-clip", type=float)
    parser.add_argument("--use-motsynth", type=str)
    parser.add_argument("--use-crowdhuman", type=str)
    parser.add_argument("--motsynth-rate", type=float)
    parser.add_argument("--sample-steps", type=int, nargs="*")
    parser.add_argument("--sample-lengths", type=int, nargs="*")

    # Training settings
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-points", type=float)
    parser.add_argument("--lr-backbone", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-drop-milestones", type=int, nargs="*")

    # Submit settings
    parser.add_argument("--miss-tolerance", type=float)

    # Model settings: number of detection queries
    parser.add_argument("--num-det-queries", type=int)
    parser.add_argument("--merge-det-track-layer", type=int)

    # Training augmentation:
    parser.add_argument("--tp-drop-rate", type=float)
    parser.add_argument("--fp-insert-rate", type=float)
    args, unknown = parser.parse_known_args()


    return parser.parse_args()


def main(config: dict):
    # Do not override CUDA_VISIBLE_DEVICES if already set
    # Note: torchrun/launch scripts should set CUDA_VISIBLE_DEVICES externally
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["AVAILABLE_GPUS"]
    
    # TF32 toggle: False is more accurate but slower
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import time
    if config["USE_DISTRIBUTED"]:  # enable distributed training
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(distributed_rank())
        print(f"[Rank {distributed_rank()}] Using logical GPU {torch.cuda.current_device()}, "
              f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    from train_engine_fixed import train
    from submit_engine import submit
    from eval_engine import evaluate
    
    if config["MODE"] == "train":
        train(config=config)
    elif config["MODE"] == "submit":
        submit(config=config)
    elif config["MODE"] == "eval":
        evaluate(config=config)
    else:
        raise ValueError(f"Unsupported mode '{config['MODE']}'")
    return


if __name__ == '__main__':
    opt = parse_option()                  # runtime options
    cfg = yaml_to_dict(opt.config_path)   # configs
    # Merge parser option and .yaml config, then run main function.
    merged_config = update_config(config=cfg, option=opt)  # merge CLI options into config
    merged_config["CONFIG_PATH"] = opt.config_path
    main(config=merged_config)
