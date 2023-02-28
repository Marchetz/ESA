import argparse
from trainer import trainer_mantra_tran
import torch

PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=int, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=100)

    parser.add_argument("--past_len", type=int, default=20)
    parser.add_argument("--future_len", type=int, default=40)
    parser.add_argument("--preds", type=int, default=10)
    parser.add_argument("--dim_embedding_key", type=int, default=48)
    parser.add_argument("--model_classic_flag", type=bool, default=False)

    # MODEL CONTROLLER
    parser.add_argument("--model", default='pretrained_models/MANTRA/model_MANTRA')
    #parser.add_argument("--model", default='training/training_tran/2022-04-29 00:12:08_multihead_5_linear2/model_IRM_epoch_599_2022-04-29 00:12:08')

    #new setting
    # parser.add_argument("--model_ae", default='memnet_kitti_PAMI/training/training_ae/2022-06-25 13:14:22_new_senza_dropout/model_ae_epoch_1339_2022-06-25 13:14:22')
    # parser.add_argument("--model_controller", default='pretrained_models/MANTRA/model_MANTRA')
    # parser.add_argument("--model_IRM", default='pretrained_models/MANTRA/model_MANTRA')

    parser.add_argument("--saved_memory", default=False)
    parser.add_argument("--saveImages", default=True, help="plot qualitative examples in tensorboard")
    parser.add_argument("--dataset_file", default="kitti_dataset.json", help="dataset file")
    parser.add_argument("--info", type=str, default='', help='Name of training. '
                                                             'It will be used in tensorboard log and test folder')
    return parser.parse_args()


def main(config):
    t = trainer_mantra_tran.Trainer(config)
    print('start training mantra_transformer')
    t.fit()


if __name__ == "__main__":
    config = parse_config()
    torch.backends.cudnn.enabled = False
    print(torch.__version__)
    main(config)
