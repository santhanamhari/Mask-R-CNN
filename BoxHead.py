import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from utils import *

        
class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        self.C=Classes
        self.P=P
        # TODO initialize BoxHead
        super(BoxHead, self).__init__()
        self.intermediate_layer = nn.Sequential(
            nn.Linear(in_features=256*P*P , out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024 , out_features=1024),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024 , out_features=self.C+1)
            )
        
        self.regressor = nn.Linear(in_features=1024, out_features=4*self.C)

     #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth_batch(self,proposals, gt_labels, bbox):
        img_w = 1088
        img_h = 800
        
        labels_one, regressor_target_one = MultiApply(self.create_ground_truth, proposals, gt_labels, bbox)
        
        labels = torch.cat(labels_one).view(-1,1)
        regressor_target = torch.cat(regressor_target_one).view(-1,4)
        
        return labels,regressor_target

    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: {(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: {(n_obj)}
    #       bbox: {(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals_per_img,gt_labels_per_img,bbox_per_img):
        img_w = 1088
        img_h = 800
        
        # return labels, regressor_target
        labels = torch.zeros((len(proposals_per_img)), dtype=torch.long, device='cuda')
        regressor_target = torch.zeros((len(proposals_per_img),4), device='cuda')
        
        gt_box_per_img = torch.clone(bbox_per_img)
        gt_box_per_img[:, 0] = bbox_per_img[:,0] * img_w
        gt_box_per_img[:, 1] = bbox_per_img[:,1] * img_h
        gt_box_per_img[:, 2] = bbox_per_img[:,2] * img_w
        gt_box_per_img[:, 3] = bbox_per_img[:,3] * img_h
        
        iou = IOU(proposals_per_img, gt_box_per_img)
        
        #CLASSES
        # find the ious that are above 0.5
        indices = (iou > 0.4).nonzero()
        indices_sorted = torch.argsort(iou[iou > 0.4])
        indices_sorted = indices[indices_sorted]
        labels[indices_sorted[:,0]] = gt_labels_per_img[indices_sorted[:,1]]

        #BOXES
        no_background_idx = (labels != 0)
        regressor_target[indices_sorted[:,0]] = gt_box_per_img[indices_sorted[:,1]]
        prop_no_bck = proposals_per_img[no_background_idx]
        reg_no_bck = regressor_target[no_background_idx]
        
        regressor_target[no_background_idx] = normalize_box(prop_no_bck, reg_no_bck)
        
        return labels,regressor_target


    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        # Loop through proposals to find correct feature level
        bz = len(proposals)
        img_w = 1088
        img_h = 800
        fpn_index_to_pool = []

        scaled_proposals = [p.detach().clone() for p in proposals]
        for num, b in enumerate(scaled_proposals):
            w = b[:,2] - b[:,0]
            h = b[:, 3] - b[:, 1]
            k = torch.floor(4 + torch.log2(torch.sqrt(w*h)/224)).clamp(2,5)
            fpn_index_to_pool.append(torch.tensor([ki.item() for ki in k]))
            
            batch_idx_list = (num * torch.ones((b.shape[0]), device='cuda')).reshape(-1, 1)
            scaled_proposals[num] = torch.cat((batch_idx_list, b), dim=1)

        assert len(fpn_index_to_pool) == bz
        
        # Scale every proposal by fpn_level
        scaled_proposals_flatten = torch.cat(scaled_proposals).reshape((-1, 5))
        fpn_index_to_pool_flatten = torch.cat(fpn_index_to_pool).reshape((-1))
        feature_vectors = torch.tensor([] ,device='cuda')
        ordered_indices = torch.tensor([], dtype=torch.long, device='cuda')
        
        for f in range(2, 6): # iterate through the fpn levels
            fpn = fpn_feat_list[f - 2] # bz x 256 x h x w
            fpn_h_feat = fpn.shape[2] # h 
            fpn_w_feat = fpn.shape[3] # w
            
            # find proposals that correspond to fpn level
            indexes = (fpn_index_to_pool_flatten == f)

            ordered_idx = torch.tensor(indexes.nonzero(), device='cuda')
            ordered_indices = torch.cat((ordered_indices, ordered_idx), dim=0)

            sp_fpn = scaled_proposals_flatten[indexes]
            
            # Scale the proposals based on fpn level sizes
            sp_fpn[:, 1] = sp_fpn[:, 1] * fpn_w_feat / img_w
            sp_fpn[:, 2] = sp_fpn[:, 2] * fpn_h_feat / img_h
            sp_fpn[:, 3] = sp_fpn[:, 3] * fpn_w_feat / img_w
            sp_fpn[:, 4] = sp_fpn[:, 4] * fpn_h_feat / img_h 

            f_feature_vectors = torchvision.ops.roi_align(input=fpn, boxes=sp_fpn, output_size=P).view(-1, 256*P*P)
       
            feature_vectors = torch.cat((feature_vectors, f_feature_vectors), dim=0)

           
        ordered_indices = ordered_indices.squeeze(1)
        feature_vectors[ordered_indices] = feature_vectors.clone()

 
        return feature_vectors
    
    
    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50): # ORIGINAL VALUES conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50

        split_size = int(class_logits.shape[0] / len(proposals))
        
        class_logits_split = []
        box_regression_split = []
        start = 0
        end = proposals[0].shape[0]
        for i in range(len(proposals)):
            class_logits_split.append(class_logits[start: end])
            box_regression_split.append(box_regression[start:end])
            
            if i == (len(proposals) - 1):
                break
            
            start = end
            end = end + proposals[i+1].shape[0]
        
        boxes_img, scores_img, labels_img = MultiApply(self.postprocess_detections_img, class_logits_split, box_regression_split, proposals, conf_thresh=conf_thresh, keep_num_preNMS=keep_num_preNMS, keep_num_postNMS=keep_num_postNMS)
        
        boxes = torch.cat((boxes_img), dim=0).view(-1,4)
        scores = torch.cat((scores_img), dim=0).view(-1,1)
        labels = torch.cat((labels_img), dim=0).view(-1,1)
        
        return boxes, scores, labels
        
      
    def postprocess_detections_img(self, class_logits, box_regression, proposals, conf_thresh, keep_num_preNMS, keep_num_postNMS):
 
        boxes = torch.tensor([], device='cuda')
        scores = torch.tensor([], device='cuda')
        labels = torch.tensor([], device='cuda')
        
        # Sort by confidence
        sorted_indices = (class_logits[:, 1:] > conf_thresh).any(dim=1)
        
        class_logits = class_logits[sorted_indices]
        box_regression = box_regression[sorted_indices]
        proposals = proposals[sorted_indices]
        
        # Crop bounding boxes 
        class_indices = class_logits.argmax(dim=1)

        for c in range(1,4):
            c_inds = (class_indices == c)

            if c_inds.sum() == 0:
                continue
            
            c_logits = class_logits[c_inds, c]
            c_proposals = proposals[c_inds]
            start = (4*(c-1))
            end = start + 4
            
            c_boxes = output_decodingd(box_regression[c_inds,start:end], c_proposals)
            
            c_boxes[:, 0] = torch.clip(c_boxes[:, 0], min = 4, max = 1084)
            c_boxes[:, 1] = torch.clip(c_boxes[:, 1], min = 4, max = 796)
            c_boxes[:, 2] = torch.clip(c_boxes[:, 2], min = 4, max= 1084)
            c_boxes[:, 3] = torch.clip(c_boxes[:, 3], min = 4, max = 796)
            
            # Sort logits to feed to matrix nms
            sort_inds = torch.argsort(c_logits, descending=True)[:keep_num_preNMS]
            s_logits = c_logits[sort_inds]
            s_boxes = c_boxes[sort_inds]
            decay_scores = self.NMS(s_logits, s_boxes, conf_thresh)
            
            nms_sort_inds =  torch.argsort(decay_scores,descending=True)
            boxes = torch.cat((boxes, s_boxes[nms_sort_inds]), dim=0) 
            scores = torch.cat((scores, decay_scores[nms_sort_inds]), dim=0)
            labels = torch.cat((labels, torch.full_like(decay_scores, c)), dim=0)
            
            
        idx = torch.argsort(scores,descending=True)  
        boxes = boxes[idx][:keep_num_postNMS]
        scores = scores[idx][:keep_num_postNMS]
        labels = labels[idx][:keep_num_postNMS]
        
        return boxes, scores, labels


   # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox, thresh):
        ##################################
        # TODO perform NSM
        method = 'gauss'
        gauss_sigma=0.5
        n = len(clas)
        sorted_boxs = prebox.reshape(n, -1)
        intersection = torch.mm(sorted_boxs, sorted_boxs.T)
        areas = sorted_boxs.sum(dim=1).expand(n, n)
        union = areas + areas.T - intersection
        ious = (intersection / union).triu(diagonal=1)
    
        ious_cmax = ious.max(0)[0].expand(n, n).T
        
        if method == 'gauss':
            decay = torch.exp(-(ious ** 2 - ious_cmax ** 2) / gauss_sigma)
        else:
            decay = (1 - ious) / (1 - ious_cmax)
        
        # move decay to device
        decay = decay.min(dim=0)[0].to('cuda')
        
        # just keep certain ones based on decayed scores
        decay_scores = clas * decay

        return decay_scores
    
    

    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self, class_logits, box_preds, labels, regression_targets, l=20, effective_batch=48):
        
        fgd_labels_idx = (labels > 0).nonzero()[:, 0]
        bgd_labels_idx = (labels == 0).nonzero()[:, 0]
        
        # Sampling is 3:1 of effective batch, 3:Foreground Classes & 1:Background Classes
        n_fgd_labels = min(int(0.75*effective_batch),  len(fgd_labels_idx))
        n_bgd_labels = effective_batch - n_fgd_labels
  
        fgd_labels_idx = fgd_labels_idx[:n_fgd_labels]
        bgd_labels_idx = bgd_labels_idx[:n_bgd_labels]

        #print('Sampling ratio of no-background to background labels:', len(fgd_labels_idx)/len(bgd_labels_idx))
        fgd_class_logits = class_logits[fgd_labels_idx]
        bgd_class_logits = class_logits[bgd_labels_idx]
        
        fgd_labels = labels[fgd_labels_idx] 
        bgd_labels = labels[bgd_labels_idx]
        
        # Classifier Loss
        criterion1 = nn.CrossEntropyLoss(reduction='sum')
        # labels need to be 1D not 2D for cross entropy - hence the squeeze
        loss_class = (criterion1(fgd_class_logits, fgd_labels.squeeze()) + criterion1(bgd_class_logits, bgd_labels.squeeze()))/effective_batch

        # Regressor Loss
        fgd_regression_targets = regression_targets[fgd_labels_idx]
        fgd_box_preds = box_preds[fgd_labels_idx]

        fgd_box_preds_class = torch.zeros_like(fgd_regression_targets)

        for i in range(len(fgd_box_preds)):
            label_slice_start = 4 * (fgd_labels[i] - 1)
            label_slice_end = label_slice_start + 4
            fgd_box_preds_class[i] = fgd_box_preds[i][label_slice_start.item():label_slice_end.item()]
        
        criterion2 = nn.SmoothL1Loss(reduction = 'sum')
        loss_regr = criterion2(fgd_box_preds_class, fgd_regression_targets) / effective_batch
        
        # Loss
        loss = loss_class + l * loss_regr

        return loss, loss_class, loss_regr
    
    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):
        output = self.intermediate_layer(feature_vectors)
        
        class_logits = self.classifier(output)
        box_preds = self.regressor(output)
        
        return class_logits, box_preds

if __name__ == '__main__':
    print('Main')