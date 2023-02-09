# Different test settings for MixFormer-Convmae-b, MixFormer-Convmae-l on LaSOT/TrackingNet/GOT10K/UAV123/OTB100
# First, put your trained MixFomrer-online models on SAVE_DIR/models directory. 
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- MixConvmae-B -----------------##########
### LaSOT test and evaluation
python tracking/test.py mixformer_convmae_online baseline --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_convmae_base_online.pth.tar --params__search_area_scale 4.8
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline

### TrackingNet test and pack
# python tracking/test.py mixformer_convmae_online baseline --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_convmae_base_online.pth.tar
# python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_convmae_online --cfg_name baseline

### GOT10k test and pack
# python tracking/test.py mixformer_convmae_online baseline --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_convmae_base_online_got.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name mixformer_convmae_online --cfg_name baseline

### UAV123
# python tracking/test.py mixformer_convmae_online baseline --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_convmae_base_online.pth.tar --params__search_area_scale 4.5
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline

### OTB100
#python tracking/test.py mixformer_cvt_online baseline --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_convmae_base_online_22k.pth.tar
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline


##########-------------- MixConvmae-L -----------------##########
### LaSOT test and evaluation
# python tracking/test.py mixformer_convmae_online baseline_large --dataset lasot --threads 32 --num_gpus 8 --params__model mixformer_convmae_large_online.pth.tar --params__search_area_scale 4.5
# python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large

### TrackingNet test and pack
# python tracking/test.py mixformer_convmae_online baseline_large --dataset trackingnet --threads 32 --num_gpus 8 --params__model mixformer_convmae_large_online.pth.tar --params__search_area_scale 4.6
# python lib/test/utils/transform_trackingnet.py --tracker_name mixformer_convmae_online --cfg_name baseline_large

### GOT10k test and pack
# python tracking/test.py mixformer_convmae_online baseline_large --dataset got10k_test --threads 32 --num_gpus 8 --params__model mixformer_convmae_large_online_got.pth.tar
# python lib/test/utils/transform_got10k.py --tracker_name mixformer_convmae_online --cfg_name baseline_large

### UAV123
# python tracking/test.py mixformer_convmae_online baseline_large --dataset uav --threads 32 --num_gpus 8 --params__model mixformer_convmae_large_online.pth.tar --params__search_area_scale 4.55
# python tracking/analysis_results.py --dataset_name uav --tracker_param baseline_large

### OTB100
#python tracking/test.py mixformer_cvt_online baseline_large --dataset otb --threads 32 --num_gpus 8 --params__model mixformer_convmae_large_online_22k.pth.tar --params__search_area_scale 4.45
#python tracking/analysis_results.py --dataset_name otb --tracker_param baseline_large


