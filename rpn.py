import torch
from torchvision import transforms
from torch.nn import functional as F
from torch import nn, Tensor
from dataset import *
from utils import *
import torchvision
import math
import bisect
import numpy as np

class RPNHead(torch.nn.Module):
    # The input of the initialization of the RPN is:
    # Input:
    #       computed_anchors: the anchors computed in the dataset
    #       num_anchors: the number of anchors that are assigned to each grid cell
    #       in_channels: number of channels of the feature maps that are outputed from the backbone
    #       device: the device that we will run the model
    def __init__(self, num_anchors=3, in_channels=256, device='cuda',
                 anchors_param=dict(ratio=[[1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2], [1, 0.5, 2]],
                                    scale=[32, 64, 128, 256, 512],
                                    grid_size=[(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)],
                                    stride=[4, 8, 16, 32, 64])
                 ):
        ######################################
        # TODO initialize RPN
        #######################################
        super(RPNHead, self).__init__()
        
        # device
        self.device = device
        self.grid_size = anchors_param['grid_size']
        self.stride = anchors_param['stride']
        self.scale = anchors_param['scale']

        # intermediate Layer
        self.intermediate = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
            )

        # classifier Head
        self.classifier_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1*3, kernel_size=(1, 1), padding='same'),
            nn.Sigmoid()
            )

        # regressor Head
        self.regressor_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4*3, kernel_size=(1, 1), padding='same'),
            nn.Sigmoid()
            )
        
        #  find anchors
        self.anchors_param = anchors_param
        self.anchors = self.create_anchors(
            self.anchors_param['ratio'], self.anchors_param['scale'], self.anchors_param['grid_size'], self.anchors_param['stride'])

        #print(self.anchors)
        self.ground_dict = {}
    
    # Forward each level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       X: list:len(FPN){(bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logits: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       bbox_regs: list:len(FPN){(bz,4*num_anchors, grid_size[0],grid_size[1])}
    def forward(self, X):
        logits = []
        bbox_regs = []
        logits, bbox_regs = MultiApply(self.forward_single, X)

        return logits, bbox_regs

    # Forward a single level of the FPN output through the intermediate layer and the RPN heads
    # Input:
    #       feature: (bz,256,grid_size[0],grid_size[1])}
    # Ouput:
    #       logit: (bz,1*num_anchors,grid_size[0],grid_size[1])
    #       bbox_regs: (bz,4*num_anchors, grid_size[0],grid_size[1])
    def forward_single(self, feature):

        logit, bbox_reg = self.classifier_head(self.intermediate(feature)), \
                            self.regressor_head(self.intermediate(feature))
        return logit, bbox_reg


    # This function creates the anchor boxes for all FPN level
    # Input:
    #       aspect_ratio: list:len(FPN){list:len(number_of_aspect_ratios)}
    #       scale:        list:len(FPN)
    #       grid_size:    list:len(FPN){tuple:len(2)}
    #       stride:       list:len(FPN)
    # Output:
    #       anchors_list: list:len(FPN){(grid_size[0]*grid_size[1]*num_anchors,4)}
    def create_anchors(self, aspect_ratio, scale, grid_size, stride):
        anchors_list = []

        for i in range(5):
            anchors_one = self.create_anchors_single(aspect_ratio[i], scale[i], grid_size[i], stride[i])
            anchors_list.append(anchors_one)
            del anchors_one

        assert len(anchors_list) == 5
        return anchors_list

    # This function creates the anchor boxes for one FPN level
    # Input:
    #      aspect_ratio: list:len(number_of_aspect_ratios)
    #      scale: scalar
    #      grid_size: tuple:len(2)
    #      stride: scalar
    # Output:
    #       anchors: (grid_size[0]*grid_size[1]*num_anchors,4)
    def create_anchors_single(self, aspect_ratio, scale, grid_size, stride):
        anchors  = torch.zeros((grid_size[0]*grid_size[1]*3, 4))

        yv, xv = torch.meshgrid(torch.arange(grid_size[0]), torch.arange(grid_size[1]))

        y_ctr, x_ctr = (yv + 0.5)*stride, (xv + 0.5)*stride
        y_ctr, x_ctr = y_ctr.flatten(), x_ctr.flatten()
        w_anchor  = [math.sqrt(scale**2 * ar) for ar in aspect_ratio]
        h_anchor = [w / ar for w, ar in zip(w_anchor, aspect_ratio)]
        
        index = grid_size[0]*grid_size[1]
            
        # anchor 1
        anchors[:index, 0] = x_ctr
        anchors[:index, 1] = y_ctr
        anchors[:index, 2] = torch.full_like(x_ctr,w_anchor[0])
        anchors[:index, 3] = torch.full_like(x_ctr,h_anchor[0])
        
        # anchor 2
        anchors[index:2*index, 0] = x_ctr
        anchors[index:2*index, 1] = y_ctr
        anchors[index:2*index, 2] = torch.full_like(x_ctr,w_anchor[1])
        anchors[index:2*index, 3] = torch.full_like(x_ctr,h_anchor[1])
        
        # anchor 3
        anchors[2*index:, 0] = x_ctr
        anchors[2*index:, 1] = y_ctr
        anchors[2*index:, 2] = torch.full_like(x_ctr,w_anchor[2])
        anchors[2*index:, 3] = torch.full_like(x_ctr,h_anchor[2])

        assert anchors.shape == (grid_size[0]*grid_size[1]*3, 4)

        del x_ctr, y_ctr, yv, xv, w_anchor, h_anchor
        
        return anchors

    def get_anchors(self):
        return self.anchors

    # This function creates the ground truth for a batch of images
    # Input:
    #      bboxes_list: list:len(bz){(number_of_boxes,4)}
    #      indexes: list:len(bz)
    #      image_shape: list:len(bz){tuple:len(2)}
    # Ouput:
    #      ground: list:len(FPN){(bz,num_anchors,grid_size[0],grid_size[1])}
    #      ground_coord: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    def create_batch_truth(self, bboxes_list, indexes, image_shape):

        ground_list, ground_coord_list = MultiApply(self.create_ground_truth, bboxes_list, indexes, grid_sizes=self.grid_size, anchors=self.anchors, image_size=image_shape)
    
        # iterate through each batch list
        ground, ground_coord = [], []
        
        # create tensor for correct output
        bz = len(bboxes_list)
        
        # change order for return
        fpn0_list, fpn0_c_list = [], []
        fpn1_list, fpn1_c_list = [], []
        fpn2_list, fpn2_c_list = [], []
        fpn3_list, fpn3_c_list = [], []
        fpn4_list, fpn4_c_list = [], []
        
        for b in range(bz):
            ground_list_batch, ground_coord_list_batch = ground_list[b],ground_coord_list[b]
            fpn0_list.append(ground_list_batch[0]), fpn0_c_list.append(ground_coord_list_batch[0])
            fpn1_list.append(ground_list_batch[1]), fpn1_c_list.append(ground_coord_list_batch[1])
            fpn2_list.append(ground_list_batch[2]), fpn2_c_list.append(ground_coord_list_batch[2])
            fpn3_list.append(ground_list_batch[3]), fpn3_c_list.append(ground_coord_list_batch[3])
            fpn4_list.append(ground_list_batch[4]), fpn4_c_list.append(ground_coord_list_batch[4])
            del ground_list_batch, ground_coord_list_batch
            
        fpn0, fpn0_c = torch.stack(fpn0_list, dim=0), torch.stack(fpn0_c_list, dim=0)
        fpn1, fpn1_c = torch.stack(fpn1_list, dim=0), torch.stack(fpn1_c_list, dim=0)
        fpn2, fpn2_c = torch.stack(fpn2_list, dim=0), torch.stack(fpn2_c_list, dim=0)
        fpn3, fpn3_c = torch.stack(fpn3_list, dim=0), torch.stack(fpn3_c_list, dim=0)
        fpn4, fpn4_c = torch.stack(fpn4_list, dim=0), torch.stack(fpn4_c_list, dim=0)

        ground.append(fpn0), ground.append(fpn1), ground.append(fpn2), ground.append(fpn3), ground.append(fpn4)
        ground_coord.append(fpn0_c), ground_coord.append(fpn1_c), ground_coord.append(fpn2_c), ground_coord.append(fpn3_c), ground_coord.append(fpn4_c)
        del fpn0, fpn0_c, fpn1, fpn1_c, fpn2, fpn2_c, fpn3, fpn3_c, fpn4, fpn4_c
        del fpn0_list, fpn0_c_list, fpn1_list, fpn1_c_list, fpn2_list, fpn2_c_list, fpn3_list, fpn3_c_list, fpn4_list, fpn4_c_list
        del ground_list, ground_coord_list
        
        return ground, ground_coord

    # This function create the ground truth for one image for all the FPN levels
    # It also caches the ground truth for the image using its index
    # Input:
    #       bboxes:      (n_boxes,4)
    #       index:       scalar (the index of the image in the total dataset)
    #       grid_size:   list:len(FPN){tuple:len(2)}
    #       anchor_list: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}   [(A,4),(B,4),,,] -> (A+B+C..,4)
    # Output:
    #       ground_clas: list:len(FPN){(num_anchors,grid_size[0],grid_size[1])}
    #       ground_coord: list:len(FPN){(4*num_anchors,grid_size[0],grid_size[1])}
    
    def create_ground_truth(self, bboxes, index, grid_sizes, anchors, image_size):
        key = str(index)
        if key in self.ground_dict:
            groundt, ground_coord = self.ground_dict[key]
            return groundt, ground_coord
        #####################################################
        ground_class = []
        ground_coord = []
        n_fpn = 5
        num_anchors = 3
        img_w, img_h = image_size[1], image_size[0]
        
        # use utils function to process ground truths and anchors
        gt_boxes, anchors_stacked_all = process_for_iou(bboxes, torch.cat(anchors, dim=0), image_size)

        # IOU
        ious = IOU(anchors_stacked_all, gt_boxes)
        
        # remove cross boundary 
        cross_boundary = ((anchors_stacked_all[:, 0] < 0) | (anchors_stacked_all[:, 1] < 0) | (anchors_stacked_all[:, 2] > 1088) | (anchors_stacked_all[:, 3] > 800)).nonzero()[:,0]
        ious[cross_boundary, :] = -1
        
        # find maximum iou
        max_ious = ious.max(dim=0).values
        max_indices = (ious >=  0.999*max_ious).nonzero()
        box_max_indices = max_indices[:,1]
        max_indices = max_indices[:,0] # indices: [a, b, c, ...]
        
        # find the iou > 0.7
        iou_idx_high = (ious > 0.7).nonzero()[:, 0]
        box_iou_idx_high = (ious > 0.7).nonzero()[:, 1]
        
        pos_indices = torch.cat((max_indices, iou_idx_high),0)
        corresponding_boxes = torch.cat((box_max_indices, box_iou_idx_high))

        # find iou < 0.3 and non-positive
        neg_indices = torch.logical_and(ious >= 0, ious < max_ious.clamp(max=0.3)).all(dim=1).nonzero()[:, 0]

        # find what fpn level
        grid_total = [0, 163200, 204000, 214200, 216750, 217413]

        for fpn_idx in range(n_fpn):
            # extract grid size at the given fpn level
            grid_size_fpn = self.grid_size[fpn_idx]
            
            # define tensors for list
            ground_class_one = torch.full(size=(num_anchors, grid_size_fpn[0], grid_size_fpn[1]), fill_value=-1, dtype=torch.float) 
            ground_coord_one = torch.full(size=(4*num_anchors, grid_size_fpn[0], grid_size_fpn[1]), fill_value=-1, dtype=torch.float)
            
            # correct positive index
            pos_indices_fpn_all = pos_indices[torch.logical_and(pos_indices < grid_total[fpn_idx+1],pos_indices >= grid_total[fpn_idx])]
            pos_indices_fpn = pos_indices_fpn_all - grid_total[fpn_idx]
           
            # correct negative index
            neg_indices_fpn_all = neg_indices[torch.logical_and(neg_indices < grid_total[fpn_idx+1],neg_indices >= grid_total[fpn_idx])]
            neg_indices_fpn = neg_indices_fpn_all - grid_total[fpn_idx]
            
            # set class information
            idx = np.unravel_index(pos_indices_fpn, shape=(num_anchors, grid_size_fpn[0], grid_size_fpn[1]))
            ground_class_one[idx[0], idx[1], idx[2]] = 1
            
            idx_neg = np.unravel_index(neg_indices_fpn, shape=(num_anchors, grid_size_fpn[0], grid_size_fpn[1]))
            ground_class_one[idx_neg[0], idx_neg[1], idx_neg[2]] = 0
            
            # set bounding boxes
            corresponding_boxes_fpn = corresponding_boxes[torch.logical_and(pos_indices < grid_total[fpn_idx+1],pos_indices >= grid_total[fpn_idx])]
            
            corresponding_boxes_fpn = gt_boxes[corresponding_boxes_fpn]
            corresponding_anchors_fpn = anchors_stacked_all[pos_indices_fpn_all]
            
            transformed_boxes = normalize_box(corresponding_anchors_fpn, corresponding_boxes_fpn).float()

            if len(pos_indices_fpn) > 0:
                ground_coord_one[4*idx[0], idx[1], idx[2]] = transformed_boxes[:,0]
                ground_coord_one[4*idx[0] + 1, idx[1], idx[2]] = transformed_boxes[:,1]
                ground_coord_one[4*idx[0] + 2, idx[1], idx[2]] = transformed_boxes[:,2]
                ground_coord_one[4*idx[0] + 3, idx[1], idx[2]] = transformed_boxes[:,3]

            ground_class.append(ground_class_one)
            ground_coord.append(ground_coord_one)
            
            del ground_class_one, ground_coord_one
            del pos_indices_fpn_all, pos_indices_fpn, neg_indices_fpn_all, neg_indices_fpn
            del idx_neg, idx, corresponding_boxes_fpn, corresponding_anchors_fpn, transformed_boxes
            
        del anchors_stacked_all, ious, pos_indices, neg_indices, cross_boundary, gt_boxes, corresponding_boxes, max_indices
        #####################################################
        self.ground_dict[key] = (ground_class, ground_coord)
        
        return ground_class, ground_coord


    # Compute the loss of the classifier
    # Input:
    #      p_out:     (positives_on_mini_batch)  (output of the classifier for sampled anchors with positive gt labels)
    #      n_out:     (negatives_on_mini_batch) (output of the classifier for sampled anchors with negative gt labels
    def loss_class(self, p_out, n_out):

        # torch.nn.BCELoss()
        # TODO compute classifier's loss
        ones = torch.ones_like(p_out, dtype=torch.float32)
        zeros = torch.zeros_like(n_out, dtype=torch.float32)

        criterion = torch.nn.BCELoss(reduction='sum')
        loss_ones = criterion(p_out, ones)
        loss_zeros = criterion(n_out, zeros)

        loss = (loss_ones + loss_zeros)/(len(p_out) + len(n_out)) # averaged over the mini-batch (M)
        
        del ones, zeros

        return loss

    # Compute the loss of the regressor
    # Input:
    #       pos_target_coord: (positive_on_mini_batch,4) (ground truth of the regressor for sampled anchors with positive gt labels)
    #       pos_out_r: (positive_on_mini_batch,4)        (output of the regressor for sampled anchors with positive gt labels)
    def loss_reg(self, pos_target_coord, pos_out_r):
        # torch.nn.SmoothL1Loss()
        # TODO compute regressor's loss
        criterion = torch.nn.SmoothL1Loss(reduction = 'sum')
        loss = criterion(pos_target_coord, pos_out_r)
        loss = loss / len(pos_target_coord)

        return loss

    # Compute the total loss for the FPN heads
    # Input:
    #       clas_out_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       regr_out_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       targ_clas_list: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       targ_regr_list: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       l: weighting lambda between the two losses
    # Output:
    #       loss: scalar
    #       loss_c: scalar
    #       loss_r: scalar
    def compute_loss(self, clas_out_list, regr_out_list, targ_clas_list, targ_regr_list, l=2, effective_batch=32):
        n_fpn = 5
        
        targ_reg = []
        targ_clas = []
        pred_reg = []
        pred_clas = []
    
        # place in correct format
        # iterate through the fpn levels
        for f in range(5):
            # target regression
            targ_r_fpn = targ_regr_list[f].permute(0, 2, 3, 1).contiguous().view(-1, 12)
            targ_r_fpn_correct = torch.cat((targ_r_fpn[:, 0:4], targ_r_fpn[:, 4:8], targ_r_fpn[:, 8:12]), dim=0)
            # target class
            targ_c_fpn = targ_clas_list[f].permute(0, 2, 3, 1).contiguous().view(-1, 3)
            targ_c_fpn_correct = torch.cat((targ_c_fpn[:, 0], targ_c_fpn[:, 1], targ_c_fpn[:, 2]), dim=0)
            # pred regression
            pred_r_fpn = regr_out_list[f].permute(0, 2, 3, 1).contiguous().view(-1, 12)
            pred_r_fpn_correct = torch.cat((pred_r_fpn[:, 0:4], pred_r_fpn[:, 4:8], pred_r_fpn[:, 8:12]), dim=0)
            # pred class
            pred_c_fpn = clas_out_list[f].permute(0, 2, 3, 1).contiguous().view(-1, 3)
            pred_c_fpn_correct = torch.cat((pred_c_fpn[:, 0], pred_c_fpn[:, 1], pred_c_fpn[:, 2]), dim=0)

            # append values to lists
            targ_reg.append(targ_r_fpn_correct)
            targ_clas.append(targ_c_fpn_correct)
            pred_reg.append(pred_r_fpn_correct)
            pred_clas.append(pred_c_fpn_correct)
            
            del targ_r_fpn_correct, targ_c_fpn_correct, pred_r_fpn_correct, pred_c_fpn_correct
            del targ_r_fpn, targ_c_fpn, pred_r_fpn, pred_c_fpn
 
        
        targ_reg_t = torch.cat(targ_reg).to('cuda:0')
        targ_clas_t = torch.cat(targ_clas).to('cuda:0')
        pred_reg_t = torch.cat(pred_reg)
        pred_clas_t = torch.cat(pred_clas)
        
        # find number of positive and negative in mini-batch
        pos_inds = (targ_clas_t == 1).nonzero()[:,0]
        neg_inds = (targ_clas_t == 0).nonzero()[:,0]
        
        num_pos = min(effective_batch//2, len(pos_inds))
        num_neg = effective_batch - num_pos
        
        indices_pos = np.random.choice(pos_inds.cpu(), int(num_pos))
        indices_neg = np.random.choice(neg_inds.cpu(), int(num_neg))

        # find individual losses
        loss_c = self.loss_class(pred_clas_t[indices_pos], pred_clas_t[indices_neg])

        # Regressor Loss
        loss_r = self.loss_reg(targ_reg_t[indices_pos], pred_reg_t[indices_pos])

        # Total Loss
        loss = (loss_c + l * loss_r)
        
        del targ_reg, targ_clas, pred_reg, pred_clas
        del targ_reg_t, targ_clas_t, pred_reg_t, pred_clas_t
        del pos_inds, neg_inds, indices_pos, indices_neg
       
        return loss, loss_c, loss_r


    # Post process for the outputs for a batch of images
    # Input:
    #       out_c: list:len(FPN){(bz,1*num_anchors,grid_size[0],grid_size[1])}
    #       out_r: list:len(FPN){(bz,4*num_anchors,grid_size[0],grid_size[1])}
    #       IOU_thresh: scalar that is the IOU threshold for the NMS
    #       keep_num_preNMS: number of masks we will keep from each image before the NMS
    #       keep_num_postNMS: number of masks we will keep from each image after the NMS
    # Output:
    #       nms_clas_list: list:len(bz){(Post_NMS_boxes)} (the score of the boxes that the NMS kept)
    #       nms_prebox_list: list:len(bz){(Post_NMS_boxes,4)} (the coordinate of the boxes that the NMS kept)
    def postprocess(self, out_c, out_r, IOU_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=10):
        nms_clas_list, nms_prebox_list = [], []
        # batch size number
        bsz = out_c[0].shape[0]
        
        # call postprocess per img
        for b in range(bsz):
            out_c_img = []
            out_r_img = []
            for f in range(5):
                out_c_img.append(out_c[f][b].unsqueeze(dim=0))
                out_r_img.append(out_r[f][b].unsqueeze(dim=0))
                
            nms_clas_one, nms_prebox_one = self.postprocessImg(out_c_img, out_r_img, IOU_thresh=IOU_thresh, keep_num_preNMS=keep_num_preNMS, keep_num_postNMS=keep_num_postNMS)
            
            nms_clas_list.append(nms_clas_one)
            nms_prebox_list.append(nms_prebox_one)
            
            del out_c_img, out_r_img, nms_clas_one, nms_prebox_one
          
        return nms_clas_list, nms_prebox_list

    # Post process the output for one image
    # Input:
    #      mat_clas: list:len(FPN){(1,1*num_anchors,grid_size[0],grid_size[1])}  (score of the output boxes)
    #      mat_coord: list:len(FPN){(1,4*num_anchors,grid_size[0],grid_size[1])} (encoded coordinates of the output boxess)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def postprocessImg(self, mat_clas, mat_coord, IOU_thresh, keep_num_preNMS, keep_num_postNMS):
        
        flatten_regr, flatten_clas, flatten_anchors = output_flattening(mat_coord, mat_clas, self.anchors)
        flatten_anchors = flatten_anchors.to('cuda:0')
        prebox = output_decoding(flatten_regr, flatten_anchors)

        # clip the outside values
        prebox[:,0] = torch.clip(prebox[:,0], min = 4, max = 1084)
        prebox[:,1] = torch.clip(prebox[:,1], min = 4, max = 796)
        prebox[:,2] = torch.clip(prebox[:,2], min = 4, max= 1084)
        prebox[:,3] = torch.clip(prebox[:,3], min = 4, max = 796)
    
        # filter based on sorted indices
        sorted_idx = torch.argsort(flatten_clas,descending=True)[:keep_num_preNMS]
        filtered_clas = flatten_clas[sorted_idx]
        filtered_decoded_boxes = prebox[sorted_idx]
        
        # apply nms
        nms_clas, nms_prebox = self.NMS(filtered_clas, filtered_decoded_boxes, IOU_thresh)
        nms_clas = nms_clas[:keep_num_postNMS]
        nms_prebox = nms_prebox[:keep_num_postNMS]
        
        return nms_clas, nms_prebox

    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self, clas, prebox, thresh):
        '''
        n = len(clas)
        ious = IOU(prebox, prebox).triu(diagonal=1)
        ious_cmax = ious.max(0).values.expand(n, n).T
        
        gauss_sigma = 0.5
        decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        decay = decay.min(dim=0).values.view(-1, 1).to('cuda:0')
        decayed_scores = clas * decay
        decayed_score_ind = torch.argsort(decayed_scores, dim=0, descending=True)
        nms_clas = clas[decayed_score_ind[:, 0]]
        nms_prebox = prebox[decayed_score_ind[:, 0]]
        '''
        N = len(clas)
        ious = IOU(prebox, prebox).triu(diagonal=1)
        ious_cmax = ious.max(0).values.expand(N, N).T
        
        gauss_sigma = 0.5
        decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        decay = decay.min(dim=0).values.view(-1, 1).to('cuda:0')
        decayed_scores = clas * decay
       
        decayed_score_ind = torch.argsort(decayed_scores, dim=0, descending=True)
        nms_clas = clas[decayed_score_ind[:, 0]]
        nms_prebox = prebox[decayed_score_ind[:, 0]]

        return nms_clas,nms_prebox


if __name__ == "__main__":
    print('RPN complete')