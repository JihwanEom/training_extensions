import torch

from mmseg.models import SEGMENTORS, build_segmentor
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
import functools
from collections import OrderedDict
from otx.mpa.utils.logger import get_logger

import numpy as np

logger = get_logger()

class BoxMaskGenerator (object):
    def __init__(self, prop_range=(0.25, 0.4), n_boxes=1, random_aspect_ratio=True,
                 prop_by_area=True, within_bounds=True, invert=True):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks


@SEGMENTORS.register_module()
class CutmixSegNaive(BaseSegmentor):
    def __init__(self, orig_type=None, unsup_weight=0.1, warmup_start_iter=30, **kwargs):
        print('CutmixSegNaive init!')
        super(CutmixSegNaive, self).__init__()
        self.test_cfg = kwargs['test_cfg']
        self.warmup_start_iter = warmup_start_iter
        self.count_iter = 0

        cfg = kwargs.copy()
        if orig_type == 'SemiSLSegmentor':
            cfg['type'] = 'SemiSLSegmentor'
            self.align_corners = cfg['decode_head'][-1].align_corners
        else:
            cfg['type'] = 'EncoderDecoder'
            self.align_corners = cfg['decode_head'].align_corners
        self.model_s = build_segmentor(cfg)
        self.model_t = build_segmentor(cfg)

        self.unsup_weight = unsup_weight
        self.mask_generator = BoxMaskGenerator()

        # Hooks for super_type transparent weight load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(
            functools.partial(self.load_state_dict_pre_hook, self)
        )

    def extract_feat(self, imgs):
        return self.model_s.extract_feat(imgs)

    def simple_test(self, img, img_metas, **kwargs):
        return self.model_s.simple_test(img, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.model_s.aug_test(imgs, img_metas, **kwargs)

    def forward_dummy(self, img, **kwargs):
        return self.model_s.forward_dummy(img, **kwargs)
    
    def save_img(self, img_tensor, gt_tensor, filename):
        from torchvision.utils import save_image

        image = img_tensor[0].clone().detach().cpu()
        save_image(image, f'/home/soobee/training_extensions/cutmix_images/{filename}_data.png')

        if gt_tensor is not None:
            gt = gt_tensor[0].clone().detach().cpu()
            save_image(gt, f'/home/soobee/training_extensions/cutmix_images/{filename}_gt.png')

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):

        ul_data = kwargs['extra_0']

        breakpoint()

        self.count_iter += 1
        if self.warmup_start_iter > self.count_iter:
            x = self.model_s.extract_feat(img)
            loss_decode, _ = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
            return loss_decode

        ul_s_img0 = ul_data['ul_w_img'].clone().detach()
        ul_w_img0 = ul_data['ul_w_img']

        ##
        self.save_img(ul_s_img0, ul_w_img0, 'is_aug')
        
        ul_s_img1 = ul_data['cutmix.ul_w_img'].clone().detach()
        ul_w_img1 = ul_data['cutmix.ul_w_img']

        ul_img0_metas = ul_data['img_metas']
        ul_img1_metas = ul_data['cutmix.img_metas']

        mask_size = ul_s_img0.shape[2:]
        n_masks = ul_s_img0.shape[0]
        masks = torch.Tensor(self.mask_generator.generate_params(n_masks, mask_size))
        if ul_s_img0.is_cuda:
            masks = masks.cuda()
        
        try:
            ul_img_cutmix = (1-masks) * ul_s_img0 + masks * ul_s_img1
        except Exception as e:
            breakpoint()

        with torch.no_grad():
            ul1_feat = self.model_t.extract_feat(ul_w_img0)
            ul1_logit = self.model_t._decode_head_forward_test(ul1_feat, ul_img0_metas)
            ul1_logit = resize(input=ul1_logit,
                               size=ul_w_img0.shape[2:],
                               mode='bilinear',
                               align_corners=self.align_corners)
            ul1_conf, ul1_pl = torch.max(torch.softmax(ul1_logit, axis=1), axis=1, keepdim=True)

            ul2_feat = self.model_t.extract_feat(ul_w_img1)
            ul2_logit = self.model_t._decode_head_forward_test(ul2_feat, ul_img1_metas)
            ul2_logit = resize(input=ul2_logit,
                               size=ul_w_img1.shape[2:],
                               mode='bilinear',
                               align_corners=self.align_corners)
            ul2_conf, ul2_pl = torch.max(torch.softmax(ul2_logit, axis=1), axis=1, keepdim=True)

            ##
            self.save_img(ul1_pl.float(), ul2_pl.float(), 'pseudo_labels')
            self.save_img(ul_w_img0, ul_w_img1, 'imgs')

            pl_cutmixed = (1-masks)*ul1_pl + masks*ul2_pl
            
            ##
            self.save_img(ul_img_cutmix, pl_cutmixed, 'cutmixed')
            # breakpoint()
            
            pl_cutmixed = pl_cutmixed.long()

        losses = dict()

        x = self.model_s.extract_feat(img)
        x_u_cutmixed = self.model_s.extract_feat(ul_img_cutmix)
        loss_decode, _ = self.model_s._decode_head_forward_train(x, img_metas, gt_semantic_seg=gt_semantic_seg)
        loss_decode_u, _ = self.model_s._decode_head_forward_train(x_u_cutmixed, ul_img0_metas, gt_semantic_seg=pl_cutmixed)

        for key in loss_decode_u.keys():
            losses[key] = (loss_decode[key] + loss_decode_u[key]*self.unsup_weight)

        return losses

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect student model as output state_dict (teacher as auxilliary)
        """
        logger.info('----------------- MeanTeacher.state_dict_hook() called')
        output = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('model_s.'):
                k = k.replace('model_s.', '')
                output[k] = v
        return output

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, *args, **kwargs):
        """Redirect input state_dict to teacher model
        """
        logger.info('----------------- MeanTeacher.load_state_dict_pre_hook() called')
        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            state_dict['model_s.' + k] = v
            state_dict['model_t.' + k] = v

