# We only support manually setting the bounding box of first frame and save the results in debug directory.
# We plan to release a colab for running your own video demo in the future.

##########-------------- MixFormer-22k-----------------##########
python tracking/video_demo.py mixformer_online baseline /YOURS/VIDEO/PATH  \
   --optional_box [YOURS_X] [YOURS_Y] [YOURS_W] [YOURS_H] --params__model mixformer_online_22k.pth.tar --debug 1 \
  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3

##########-------------- MixFormerL-22k-----------------##########
#python tracking/video_demo.py mixformer_online baseline /home/cyt/project/MixFormer/test.mp4  \
#   --optional_box 408 240 94 254 --params__model mixformerL_online_22k.pth.tar --debug 1 \
#  --params__search_area_scale 4.5 --params__update_interval 25 --params__online_sizes 3

