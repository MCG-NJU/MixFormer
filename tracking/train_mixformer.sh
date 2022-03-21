# There are the detailed training settings for MixFormer, MixFormer-L and MixFormer-1k.
# 1. download pretrained CvT models (CvT-21-384x384-IN-22k.pth/CvT-w24-384x384-IN-22k.pth/CvT-21-384x384-IN-1k.pth) at https://onedrive.live.com/?authkey=%21AMXesxbtKwsdryE&id=56B9F9C97F261712%2115004&cid=56B9F9C97F261712
# 2. set the proper pretrained CvT models path 'MODEL:BACKBONE:PRETRAINED_PATH' at experiment/mixformer/CONFIG_NAME.yaml.
# 3. uncomment the following code to train corresponding trackers.

### Training MixFormer-22k
# Stage1: train mixformer without SPM
python tracking/train.py --script mixformer --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER --mode multiple --nproc_per_node 8
## Stage2: train mixformer_online, i.e., SPM (score prediction module)
python tracking/train.py --script mixformer_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-L-22k
#python tracking/train.py --script mixformer --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline_large --save_dir /YOUR/PATH/TO/SAVE/MIXFORMERL_ONLINE --mode multiple --nproc_per_node 8 --stage1_model /STAGE1/MODEL


### Training MixFormer-1k
#python tracking/train.py --script mixformer --config baseline_1k --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_1K --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_1K_ONLINE --mode multiple --nproc_per_node 8 \
#     --stage1_model /STAGE1/MODEL


### Training MixFormer-22k_GOT
#python tracking/train.py --script mixformer --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT --mode multiple --nproc_per_node 8
#python tracking/train.py --script mixformer_online --config baseline_got --save_dir /YOUR/PATH/TO/SAVE/MIXFORMER_GOT_ONLINE --mode multiple --nproc_per_node 8 \
#    --stage1_model /STAGE1/MODEL