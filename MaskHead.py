import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import matplotlib.pyplot as plt

class MaskHead(torch.nn.Module):
    def __init__(self,Classes=3,P=14):
        self.C=Classes
        self.P=P
        # TODO initialize MaskHead
        super(MaskHead, self).__init__()
        self.MaskHead = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding='same'),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=self.C, kernel_size=(1,1)),
            nn.Sigmoid()
            )

    # This function does the pre-prossesing of the proposals created by the Box Head (during the training of the Mask Head)
    # and create the ground truth for the Mask Head
    #
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)  ([t_x,t_y,t_w,t_h])
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       masks: list:len(bz){(n_obj,800,1088)}
    #       IOU_thresh: scalar (threshold to filter regressed with low IOU with a bounding box)
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box)
    #       gt_masks: list:len(bz){(post_NMS_boxes_per_image,2*P,2*P)}
    def preprocess_ground_truth_creation(self, class_logits, box_regression, proposals, gt_labels, bbox, masks, IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):
        
    
        split_sz = int(class_logits.shape[0]// len(proposals))
        
        boxes, scores, labels, gt_masks = MultiApply(self.preprocess_ground_truth_creation_image, 
            torch.split(class_logits, split_sz), torch.split(box_regression, split_sz), proposals, gt_labels, bbox, masks, IOU_thresh=IOU_thresh, keep_num_preNMS=keep_num_preNMS, keep_num_postNMS=keep_num_postNMS)

        #boxes = torch.cat((boxes), dim=0).view(-1,4)
        #scores = torch.cat((scores), dim=0).view(-1,1)
        #labels = torch.cat((labels), dim=0).view(-1,1)
        

        return boxes, scores, labels, gt_masks


    def preprocess_ground_truth_creation_image(self, class_logits, box_regression, proposals, gt_labels, bbox, masks, IOU_thresh=0.5, keep_num_preNMS=1000, keep_num_postNMS=100):
        
        img_w = 1088
        img_h = 800
        
        boxes = torch.tensor([], device='cuda')
        scores = torch.tensor([], device='cuda')
        labels = torch.tensor([], device='cuda')
        gt_masks = torch.tensor([], device='cuda')
        
        # process gt boxes
        gt_boxes = torch.zeros_like(bbox)
        gt_boxes[:, 0] = bbox[:,0] * img_w
        gt_boxes[:, 1] = bbox[:,1] * img_h
        gt_boxes[:, 2] = bbox[:,2] * img_w
        gt_boxes[:, 3] = bbox[:,3] * img_h
        
        # find initial boxes, scores, clabels
        class_scores, class_labels = torch.max(class_logits, dim=1)
        class_boxes = box_regression[class_labels != 0] # (27,12)
        class_proposals = proposals[class_labels != 0]
        class_scores, class_labels = class_scores[class_labels != 0], class_labels[class_labels != 0]
        
        indices = torch.stack([4*(class_labels-1),  4*(class_labels-1)+1, 4*(class_labels-1)+2, 4*(class_labels-1)+3])
        indices = torch.transpose(indices, 1, 0) 
        
        new_boxes = []
        for i in range(len(indices)):
            b = class_boxes[i, indices[i]]
            new_boxes.append(b)
        
        class_boxes = torch.stack(new_boxes)
        del new_boxes
        
        # decode the boxes
        class_decode_boxes = output_decodingd(class_boxes, class_proposals)
        
        # remove cross boundary boxes
        cond1 = torch.logical_and(class_boxes[:,0] >= 0, class_boxes[:,1] >= 0)
        cond2 = torch.logical_and(class_boxes[:,2] < 1088, class_boxes[:,3] < 800)
        cross_boundary = torch.logical_and(cond1, cond2)
        
        class_scores = class_scores[cross_boundary]
        class_labels = class_labels[cross_boundary]
        class_decode_boxes = class_decode_boxes[cross_boundary]
        
        # remove based on low iou
        iou = IOU(class_decode_boxes, gt_boxes)
        iou_index = (iou > IOU_thresh).nonzero()
        prop_index = iou_index[:, 0]
        mask_index = iou_index[:, 1]
        
        class_scores = class_scores[prop_index] 
        class_labels = class_labels[prop_index]
        class_decode_boxes = class_decode_boxes[prop_index]
        
        if len(class_scores) == 0:
            return boxes, scores, labels, gt_masks
        
        # process masks
        temp_masks = masks[mask_index]
        class_masks = []
        
        for i in range(len(temp_masks)): #6
            process_mask = torch.zeros((800, 1088), device='cuda:0')
            single_box = class_decode_boxes[i] #x1, y1, x2, y2
            single_gt_mask = temp_masks[i]
            process_mask[int(single_box[1]):int(single_box[3]), int(single_box[0]):int(single_box[2])] = 1
            intersection = torch.logical_and(single_gt_mask, process_mask).type(torch.float) 
            intersection = torch.nn.functional.interpolate(intersection.unsqueeze(dim=0).unsqueeze(dim=0), size=(28,28), mode='bilinear')
            class_masks.append(intersection.squeeze().squeeze())
        class_masks = torch.stack(class_masks)
        
        # keep pre-NMS
        sort_inds = torch.argsort(class_scores, descending=True)[:keep_num_preNMS]
        class_scores = class_scores[sort_inds] 
        class_labels = class_labels[sort_inds]
        class_decode_boxes = class_decode_boxes[sort_inds]
        class_masks = class_masks[sort_inds]

        # apply NMS
        for c in range(1,4):
            
            c_inds = (class_labels == c)
            if c_inds.sum() == 0:
                continue
            '''
            c_logits = class_logits[c_inds, c]
            c_proposals = proposals[c_inds]
            start = (4*(c-1))
            end = start + 4
            
            c_boxes = output_decodingd(box_regression[c_inds,start:end], c_proposals)

            # remove cross boundary
            cond1 = torch.logical_and(c_boxes[:,0] >= 0, c_boxes[:,1] >= 0)
            cond2 = torch.logical_and(c_boxes[:,2] < 1088, c_boxes[:,3] < 800)
            cross_boundary = torch.logical_and(cond1, cond2)
            c_logits = c_logits[cross_boundary] 
            c_proposals = c_proposals[cross_boundary]
            c_boxes = c_boxes[cross_boundary]
            
            # check iou
            iou = IOU(c_boxes, gt_boxes)
            iou_index = (iou > IOU_thresh)[:,0]
            c_logits = c_logits[iou_index] 
            c_proposals = c_proposals[iou_index]
            c_boxes = c_boxes[iou_index]
            
            if len(c_logits) == 0:
                continue
            
             # Sort logits to feed to matrix nms
            sort_inds = torch.argsort(c_logits, descending=True)[:keep_num_preNMS]
            s_logits = c_logits[sort_inds]
            s_boxes = c_boxes[sort_inds]
            '''
            c_scores = class_scores[c_inds]
            c_boxes = class_decode_boxes[c_inds]
            c_labels = class_labels[c_inds]
            c_masks = class_masks[c_inds]
            decay_scores = self.NMS(c_scores, c_boxes)


            boxes = torch.cat((boxes, c_boxes), dim=0) 
            scores = torch.cat((scores, decay_scores), dim=0)
            labels = torch.cat((labels, c_labels), dim=0)
            gt_masks = torch.cat((gt_masks, c_masks), dim=0)


        nms_sort_inds =  torch.argsort(scores,descending=True)[:keep_num_postNMS]
        
        return boxes[nms_sort_inds], scores[nms_sort_inds], labels[nms_sort_inds], gt_masks[nms_sort_inds]
        
        
    # Input:
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox, thresh=0.5):
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
    

    # general function that takes the input list of tensors and concatenates them along the first tensor dimension
    # Input:
    #      input_list: list:len(bz){(dim1,?)}
    # Output:
    #      output_tensor: (sum_of_dim1,?)
    def flatten_inputs(self,input_list):
        output_tensor = torch.cat(input_list, dim=0)
        return output_tensor


    # This function does the post processing for the result of the Mask Head for a batch of images. It project the predicted mask
    # back to the original image size
    # Use the regressed boxes to distinguish between the images
    # Input:
    #       masks_outputs: (total_boxes,C,2*P,2*P)
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)} ([x1,y1,x2,y2] format) ; bz = 1
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}  (top category of each regressed box) ; bz = 1
    #       image_size: tuple:len(2)
    # Output:
    #       projected_masks: list:len(bz){(post_NMS_boxes_per_image,image_size[0],image_size[1]
    def postprocess_mask(self, masks_outputs, boxes, labels, image_size=(800,1088)):
        # choose masks that correspond to the classes predicted by the Box Head
        boxes_stacked = self.flatten_inputs(boxes)
        labels_stacked = self.flatten_inputs(labels)

        mask_target = []
        for i in range(len(labels_stacked)):
            one_mask_output = masks_outputs[i]
            temp = one_mask_output[int(labels_stacked[i].item()) - 1, :, :]
            temp = torch.nn.functional.interpolate(temp.unsqueeze(dim=0).unsqueeze(dim=0), size=(800,1088), mode='bilinear')
            mask_target.append(temp.squeeze().squeeze())
            
        projected_masks = torch.stack(mask_target)
 
        # make it binary
        projected_masks[projected_masks > 0.5] = 1
        projected_masks[projected_masks < 0.5] = 0

        return projected_masks

    # Compute the total loss of the Mask Head
    # Input:
    #      mask_output: (total_boxes,C,2*P,2*P)
    #      labels: (total_boxes)
    #      gt_masks: (total_boxes,2*P,2*P)
    # Output:
    #      mask_loss
    def compute_loss(self,mask_output,labels,gt_masks):
        # parse training mask using labels 
        mask_target = []
        for i in range(len(labels)):
            one_mask_output = mask_output[i]
            mask_target.append(one_mask_output[int(labels[i].item()) - 1, :, :])
        mask_target = torch.stack(mask_target)
        
        criterion = nn.BCELoss()

        mask_loss = criterion(mask_target, gt_masks)

        return mask_loss.mean()



    # Forward the pooled feature map Mask Head
    # Input:
    #        features: (total_boxes, 256,P,P)
    # Outputs:
    #        mask_outputs: (total_boxes,C,2*P,2*P)
    def forward(self, features):
        mask_outputs = self.MaskHead(features)

        return mask_outputs

if __name__ == '__main__':
    print('Mask Head')
