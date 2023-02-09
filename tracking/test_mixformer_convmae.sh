# Different test settings for MixFormer-Convmae-b, MixFormer-Convmae-l on LaSOT/TrackingNet/GOT10K/UAV123/OTB100
# First, put your trained MixFomrer-online models on SAVE_DIR/models directory. 
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- MixConvmae-L -----------------##########
### LaSOT test and evaluation
python tracking/test.py mixformer_convmae_online baseline_large --dataset lasot --threads 0 --num_gpus 2 --params__model MixFormer-convmae-L-uphead-384-online.pth.tar --params__search_area_scale 4.5
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline_large
