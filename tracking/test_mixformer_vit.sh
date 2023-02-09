# Different test settings for MixFormer-ViT-b, MixFormer-ViT-l on LaSOT/TrackingNet/GOT10K/UAV123/OTB100
# First, put your trained MixFomrer-online models on SAVE_DIR/models directory. 
# Then,uncomment the code of corresponding test settings.
# Finally, you can find the tracking results on RESULTS_PATH and the tracking plots on RESULTS_PLOT_PATH.

##########-------------- MixViT-b -----------------##########
### LaSOT test and evaluation
python tracking/test.py mixformer_vit_online baseline --dataset lasot --threads 8 --num_gpus 2 --params__model MixFormer-ViT-B-uphead-288-online.pth.tar --params__search_area_scale 5.05
python tracking/analysis_results.py --dataset_name lasot --tracker_param baseline
