# Different test settings for MixFormer-22k, MixFormerL-22k & MixFormer-1k on LaSOT/TrackingNet/GOT10k/UAV123/OTB100
# First, put your trained MixFormer-online models on SAVE_DIR/models directory. ('vim lib/test/evaluation/local.py' to set your SAVE_DIR)
# Then, uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH. (see 'lib/test/evaluation/local.py')

##########-------------- MixFormer-22k-----------------##########
### LaSOT test and evaluation
python tracking/test.py mixformer_online baseline --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.55
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

### TrackingNet test and pack
#python tracking/test.py mixformer_online baseline --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.5
#python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_online --cfg_name baseline

### MixFormer-22k-got-only GOT10k test and pack
#python tracking/test.py mixformer_online baseline --dataset got10k_test --threads 32 --num_gpus 8 --params__search_area_scale 4.55 \
#  --params__model mixformer_online_22k_got.pth.tar --params__max_score_decay 0.98
#python lib/test/utils/transform_got10k.py --tracker_name mixformer_online --cfg_name baseline

### UAV123
#python tracking/test.py mixformer_online baseline --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name uav --tracker_param baseline

### OTB100
#python tracking/test.py mixformer_online baseline --dataset otb --threads 28 --num_gpus 7 --params__model mixformer_online_22k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline


##########-------------- MixFormer-L-22k-----------------##########
### LaSOT
#python tracking/test.py mixformer_online baseline_large --dataset lasot --threads 32 --num_gpus 8 --params__model mixformerL_online_22k.pth.tar --params__search_area_scale 4.55
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large

### TrackingNet
#python tracking/test.py mixformer_online baseline_large --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformerL_online_22k.pth.tar --params__search_area_scale 4.5 --params__max_score_decay 0.98
#python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_online --cfg_name baseline_large

### UAV123
#python tracking/test.py mixformer_online baseline_large --dataset uav --threads 32 --num_gpus 8 --params__model mixformerL_online_22k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_large

### OTB100
#python tracking/test.py mixformer_online baseline_large --dataset otb --threads 32 --num_gpus 8 --params__model mixformerL_online_22k.pth.tar --params__search_area_scale 4.55
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_large


##########-------------- MixFormer-1k-----------------##########
### LaSOT test and evaluation
#python tracking/test.py mixformer_online baseline_1k --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_online_1k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_1k

### Trackingnet test and pack
#python tracking/test.py mixformer baseline_1k --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_online_1k.pth.tar --params__search_area_scale 4.5
#python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_online --cfg_name baseline_1k

### MixFormer-1k-got-only GOT10k test and pack
#python tracking/test.py mixformer_online baseline_1k --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_online_got_1k.pth.tar --params__search_area_scale 4.55
#python lib/test/utils/transform_got10k.py --tracker_name mixformer_online --cfg_name baseline_1k

#python tracking/test.py mixformer_online baseline_1k --dataset got10k_test --threads 32 --num_gpus 32 --params__model mixformer_online_1k.pth.tar --params__search_area_scale 4.55
#python lib/test/utils/transform_got10k.py --tracker_name mixformer_online --cfg_name baseline_1k

### OTB100 test and evaluation
#python tracking/test.py mixformer_online baseline_1k --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_online_1k.pth.tar --params__search_area_scale 4.5
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_1k

### UAV123 test and evaluation
#python tracking/test.py mixformer_online baseline_1k --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_online_1k.pth.tar --params__search_area_scale 4.55
#python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_1k
