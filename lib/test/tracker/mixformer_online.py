from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.models.mixformer import build_mixformer_online_score
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box
from lib.test.tracker.tracker_utils import vis_attn_maps


class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name):
        super(MixFormerOnline, self).__init__(params)
        network = build_mixformer_online_score(params.cfg,  train=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # self.z_dict1 = {}

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
            self.online_size = self.online_sizes[0]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        print("Online size is: ", self.online_size)
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        print("Update interval is: ", self.update_interval)
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("max score decay = {}".format(self.max_score_decay))


    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        if self.params.vis_attn==1:
            self.z_patch = z_patch_arr
            self.oz_patch = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.online_state = info['init_bbox']
        
        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0
        
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        with torch.no_grad():
            if self.online_size==1:
                # for visualize attention maps
                if self.params.vis_attn==1 and self.frame_id % 200 == 0:
                    attn_weights = []
                    hooks = []
                    for i in range(len(self.network.backbone.stage2.blocks)):
                        hooks.append(self.network.backbone.stage2.blocks[i].attn.attn_drop.register_forward_hook(
                            lambda self, input, output: attn_weights.append(output)))
                out_dict, _ = self.network(self.template, self.online_template, search, run_score_head=True)
                if self.params.vis_attn==1 and self.frame_id % 200 == 0:
                    for hook in hooks:
                        hook.remove()
                    # attn0(t_ot) / 1(t_ot) / 2(t_ot_s)
                    # shape: torch.Size([1, 6, 64, 32]), torch.Size([1, 6, 64, 32]), torch.Size([1, 6, 400, 132])
                    # vis attn weights: online_template-to-template
                    vis_attn_maps(attn_weights[::3], q_w=8, k_w=4, skip_len=16, x1=self.oz_patch, x2=self.z_patch,
                                  x1_title='Online Template', x2_title='Template',
                                  save_path= 'vis_attn_weights/t2ot_vis/%04d' % self.frame_id)
                    # vis attn weights: template-to-online_template
                    vis_attn_maps(attn_weights[1::3], q_w=8, k_w=4, skip_len=0, x1=self.z_patch, x2=self.oz_patch,
                                  x1_title='Template', x2_title='Online Template',
                                  save_path='vis_attn_weights/ot2t_vis/%04d' % self.frame_id)
                    # vis attn weights: template-to-search
                    vis_attn_maps(attn_weights[2::3], q_w=20, k_w=4, skip_len=0, x1=self.z_patch, x2=x_patch_arr,
                                  x1_title='Template', x2_title='Search',
                                  save_path='vis_attn_weights/s2t_vis/%04d' % self.frame_id)
                    # vis attn weights: online_template-to-search
                    vis_attn_maps(attn_weights[2::3], q_w=20, k_w=4, skip_len=16, x1=self.oz_patch, x2=x_patch_arr,
                                  x1_title='Online Template', x2_title='Search',
                                  save_path='vis_attn_weights/s2ot_vis/%04d' % self.frame_id)
                    # vis attn weights: search-to-search
                    vis_attn_maps(attn_weights[2::3], q_w=20, k_w=10, skip_len=32, x1=x_patch_arr, x2=x_patch_arr,
                                  x1_title='Search1', x2_title='Search2', idxs=[(160, 160)],
                                  save_path='vis_attn_weights/s2s_vis/%04d' % self.frame_id)
                    print("save vis_attn of frame-{} done.".format(self.frame_id))
            else:
                out_dict, _ = self.network.forward_test(search, run_score_head=True)

        pred_boxes = out_dict['pred_boxes'].view(-1, 4)
        pred_score = out_dict['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
                                                        self.params.template_factor,
                                                        output_sz=self.params.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            if self.params.vis_attn == 1:
                self.oz_patch_max = z_patch_arr
            self.max_pred_score = pred_score
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
                if self.params.vis_attn == 1:
                    self.oz_patch = self.oz_patch_max
            elif self.online_template.shape[0] < self.online_size:
                self.online_template = torch.cat([self.online_template, self.online_max_template])
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            if self.online_size > 1:
                with torch.no_grad():
                    self.network.set_online(self.template, self.online_template)

            self.max_pred_score = -1
            self.online_max_template = self.template

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return MixFormerOnline
