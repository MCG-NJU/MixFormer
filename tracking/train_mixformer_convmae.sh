# There are the detailed training settings for MixFormer-Convmae-B and MixFormer-Convmae-L.
# 1. download pretrained ConvMAE models (convmae_base.pth.pth/convmae_large.pth) at https://github.com/Alpha-VL/ConvMAE
# 2. set the proper pretrained convmae models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer_convmae/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training MixFormer-Convmae-B
# Stage1: train mixformer without SPM
python tracking/train.py --script mixformer_convmae --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER --mode multiple --nproc_per_node 8
## Stage2: train mixformer_online, i.e., SPM (score prediction module)
# python tracking/train.py --script mixformer_convmae_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-Convmae-L
#python tracking/train.py --script mixformer_convmae --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_convmae_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-Convmae-B_GOT
#python tracking/train.py --script mixformer_convmae --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_convmae_online --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT_ONLINE --mode multiple --nproc_per_node 8 \
#    --stage1_model /STAGE1/MODEL
