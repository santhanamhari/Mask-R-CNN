import numpy as np
import torch
import torchvision

from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))

# This function compute the IOU between two set of boxes 
def IOU(boxA, boxB):
    ##################################
    #TODO compute the IOU between the boxA, boxB boxes
    ##################################
    boxA = boxA.cpu().detach().numpy()
    boxB = boxB.cpu().detach().numpy()

    if boxA.shape == (4,):
        boxA = boxA.reshape(1,-1)
    if boxB.shape == (4,):
        boxB = boxB.reshape(1,-1)

    boxA_x1, boxA_y1, boxA_x2, boxA_y2 = boxA[:,0].reshape(-1, 1), boxA[:,1].reshape(-1, 1), boxA[:,2].reshape(-1, 1), boxA[:,3].reshape(-1, 1)
    
    boxB_x1, boxB_y1, boxB_x2, boxB_y2 = boxB[:,0].reshape(-1, 1), boxB[:,1].reshape(-1, 1), boxB[:,2].reshape(-1, 1), boxB[:,3].reshape(-1, 1)

    area_a = (boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1).reshape(-1, 1)
    area_b = (boxB_x2 - boxB_x1) * (boxB_y2 - boxB_y1).reshape(-1, 1)

    intersect_w = np.minimum(boxA_x2, boxB_x2.T) - np.maximum(boxA_x1, boxB_x1.T)
    intersect_h = np.minimum(boxA_y2, boxB_y2.T) - np.maximum(boxA_y1, boxB_y1.T)
    np.maximum(intersect_w, 0, intersect_w)
    np.maximum(intersect_h, 0, intersect_h)

    inter_area = intersect_w * intersect_h
    union_area = (area_a + area_b.T) - inter_area
    iou = inter_area / union_area
    return torch.tensor(iou)

# This function assists in the ground truth creation of RPN
# Input:
#       bboxes:      (n_boxes,4)
#       anchor:      {(num_anchors*grid_size[0]*grid_size[1],4)}
#       image size
# Output:
def process_for_iou(bboxes, anchor, image_size):
    img_w, img_h = image_size[1], image_size[0]
    
    # transform bboxes into correct format
    gt_boxes = torch.zeros_like(bboxes)
    gt_boxes[:, 0] = bboxes[:,0] * img_w
    gt_boxes[:, 1] = bboxes[:,1] * img_h
    gt_boxes[:, 2] = bboxes[:,2] * img_w
    gt_boxes[:, 3] = bboxes[:,3] * img_h
        
    # transform anchors into correct format (currently - xcenter, ycenter, w, h) -> (x1, y1, x2, y2)
    anchor_boxes = torch.zeros_like(anchor)
    anchor_boxes[:, 0] = anchor[:,0] - (anchor[:,2]/2)
    anchor_boxes[:, 1] = anchor[:,1] - (anchor[:,3]/2)
    anchor_boxes[:, 2] = anchor[:,0] + (anchor[:,2]/2)
    anchor_boxes[:, 3] = anchor[:,1] + (anchor[:,3]/2)
    anchor_boxes = anchor_boxes.view(-1,4)

    return gt_boxes, anchor_boxes

# process this
def process_class_coord(clas, regr):
    out_r_fpn = regr.permute(0, 2, 3, 1).contiguous().view(-1, 12)
    out_r_fpn_correct = torch.cat((out_r_fpn[:, 0:4], out_r_fpn[:, 4:8], out_r_fpn[:, 8:12]), dim=0)

    #out_c_fpn = out_c[f].view(-1,1)
    out_c_fpn = clas.permute(0, 2, 3, 1).contiguous().view(-1, 3)
    out_c_fpn_correct = torch.cat((out_c_fpn[:, 0], out_c_fpn[:, 1], out_c_fpn[:, 2]), dim=0)
    
    del out_r_fpn, out_c_fpn
        
    return out_c_fpn_correct, out_r_fpn_correct
    
    
# This function flattens the output of the network and the corresponding anchors
# in the sense that it concatenate  the outputs and the anchors from all the grid cells from all
# the FPN levels from all the images into 2D matrices
# Each row correspond of the 2D matrices corresponds to a specific grid cell
# Input:
#       out_r: list:len(FPN){(bz,num_anchors*4,grid_size[0],grid_size[1])}  
#       out_c: list:len(FPN){(bz,num_anchors*1,grid_size[0],grid_size[1])}
#       anchors: list:len(FPN){(num_anchors*grid_size[0]*grid_size[1],4)}
# Output:
#       flatten_regr: (total_number_of_anchors*bz,4)
#       flatten_clas: (total_number_of_anchors*bz)
#       flatten_anchors: (total_number_of_anchors*bz,4)
def output_flattening(out_r, out_c, anchors):
    
    bz = out_c[0].shape[0]
    flatten_regr = []
    flatten_clas = []
    flatten_anchors = []
    
    # iterate through the fpn levels
    for f in range(5):
        out_r_fpn = out_r[f].permute(0, 2, 3, 1).contiguous().view(-1, 12)
        out_r_fpn_correct = torch.cat((out_r_fpn[:, 0:4], out_r_fpn[:, 4:8], out_r_fpn[:, 8:12]), dim=0)

        #out_c_fpn = out_c[f].view(-1,1)
        out_c_fpn = out_c[f].permute(0, 2, 3, 1).contiguous().view(-1, 3)
        out_c_fpn_correct = torch.cat((out_c_fpn[:, 0], out_c_fpn[:, 1], out_c_fpn[:, 2]), dim=0)
        
        anchors_fpn = anchors[f].view(-1,4) # need to fix this - problem when batch size is increased

        # append values to lists
        flatten_regr.append(out_r_fpn_correct)
        flatten_clas.append(out_c_fpn_correct)
        flatten_anchors.append(anchors_fpn)
    
    flatten_regr = torch.cat(flatten_regr)
    flatten_clas = torch.cat(flatten_clas)
    flatten_anchors = torch.cat(flatten_anchors)
    #flatten_anchors = flatten_anchors.repeat(bz,1)

    return flatten_regr, flatten_clas, flatten_anchors


# This function decodes the output that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it returns the upper left and lower right corner of the bbox
# Input:
#       flatten_out: (total_number_of_anchors*bz,4)
#       flatten_anchors: (total_number_of_anchors*bz,4)
# Output:
#       box: (total_number_of_anchors*bz,4)
def output_decoding(flatten_out, flatten_anchors, device='cpu'):
    
    x = flatten_out[:,0] * flatten_anchors[:,2] + flatten_anchors[:,0]  # tx * wa + xa
    y = flatten_out[:,1] * flatten_anchors[:,3] + flatten_anchors[:,1] # ty * ha + ya
    w = flatten_anchors[:,2] * torch.exp(flatten_out[:,2])
    h = flatten_anchors[:,3] * torch.exp(flatten_out[:,3])
    
    xmin = x - (w/2)
    xmax = x + (w/2)
    ymin = y - (h/2)
    ymax = y + (h/2)

    box = torch.vstack((xmin, ymin, xmax, ymax))
    box = torch.permute(box, (1,0))

    return box

# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decodingd(regressed_boxes_t,flatten_proposals, device='cpu'):
    
    # convert proposals to xcenter, ycenter, w , h format
    fp_x = (flatten_proposals[:,0] + flatten_proposals[:,2])/2
    fp_y = (flatten_proposals[:,1] + flatten_proposals[:,3])/2
    fp_w = flatten_proposals[:, 2] - flatten_proposals[:, 0]
    fp_h = flatten_proposals[:, 3] - flatten_proposals[:, 1]
     
    # find unormalized values
    x = regressed_boxes_t[:,0] * fp_w + fp_x
    y = regressed_boxes_t[:,1] * fp_h + fp_y
    w = fp_w * torch.exp(regressed_boxes_t[:,2])
    h = fp_h * torch.exp(regressed_boxes_t[:,3])
    
    xmin = x - (w/2)
    xmax = x + (w/2)
    ymin = y - (h/2)
    ymax = y + (h/2)

    box = torch.vstack((xmin, ymin, xmax, ymax))
    box = torch.permute(box, (1,0))
    
    return box

# This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
# a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
# Input:
#      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
#      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
#      P: scalar
# Output:
#      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
def MultiScaleRoiAlign(fpn_feat_list,proposals,P=7):
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
    scaled_proposals_flatten = torch.stack(scaled_proposals).reshape((-1, 5))
    fpn_index_to_pool_flatten = torch.stack(fpn_index_to_pool).reshape((-1))
        
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

        f_feature_vectors = torchvision.ops.roi_align(input=fpn, boxes=sp_fpn, output_size=P)
       
        feature_vectors = torch.cat((feature_vectors, f_feature_vectors), dim=0)

           
    ordered_indices = ordered_indices.squeeze(1)
    feature_vectors[ordered_indices] = feature_vectors.clone()
    
    return feature_vectors
    
'''
# define roi align
def roi_align(fpn, boxes, P):
    
    # iterate and do this for each channel
    for c in range(256):
        x_intervals = (boxes[:,3] - boxes[:, 1])/ P     
        y_intervals = (boxes[:,4] - boxes[:, 2])/ P     
    
        # iterate through each P
        for i in range(P):
            x_low = boxes[:, 1] + i * x_intervals
            #x_high = boxes[:, 1] + (i + 1) * x_intervals
            y_low = boxes[:, 2] + i * y_intervals
            #y_high = boxes[:, 2] + (i + 1) * y_intervals
        
            # coordinates for 4 randomly sampled points
            pt1_x = torch.rand(len(x_intervals), device='cuda:0') * x_intervals + x_low
            pt1_y = torch.rand(len(y_intervals), device='cuda:0') * y_intervals + y_low
            pt2_x = torch.rand(len(x_intervals), device='cuda:0') * x_intervals + x_low
            pt2_y = torch.rand(len(y_intervals), device='cuda:0') * y_intervals + y_low
            pt3_x = torch.rand(len(x_intervals), device='cuda:0') * x_intervals + x_low
            pt3_y = torch.rand(len(y_intervals), device='cuda:0') * y_intervals + y_low
            pt4_x = torch.rand(len(x_intervals), device='cuda:0') * x_intervals + x_low
            pt4_y = torch.rand(len(y_intervals), device='cuda:0') * y_intervals + y_low
        
            # define points
            pt1 = torch.stack((boxes[:, 0], pt1_y, pt1_x), dim=1)
            pt2 = torch.stack((boxes[:, 0], pt2_y, pt2_x), dim=1)
            pt3 = torch.stack((boxes[:, 0], pt3_y, pt3_x), dim=1)
            pt4 = torch.stack((boxes[:, 0], pt4_y, pt4_x), dim=1)
   
            # fpn for channel
            fpn_channel = fpn[:, c, :, :]
'''  

# normalize box
def normalize_box(proposal, box):
     # box : x1, y1, x2, y2
    x_box, y_box, w_box, h_box = (box[:,0] + box[:,2])/2, (box[:,1] + box[:,3])/2, box[:,2] - box[:,0], box[:,3] - box[:,1]
    x_prop, y_prop, w_prop, h_prop = (proposal[:,0] + proposal[:,2])/2, (proposal[:,1] + proposal[:,3])/2, proposal[:,2] - proposal[:,0], proposal[:,3] - proposal[:,1]
    
    #print('x box in xcenter format ', x_box)
    #print('x anchor in anchor ', x_prop)
    #print('w anchor in anchoro ', w_prop)
    tx = (x_box - x_prop)/w_prop
    ty = (y_box - y_prop)/h_prop
    tw = torch.log(w_box/w_prop)
    th = torch.log(h_box/h_prop)
    
    tx = tx.reshape(-1,1)
    ty = ty.reshape(-1,1)
    tw = tw.reshape(-1,1)
    th = th.reshape(-1,1)
    
    norm_box = torch.cat((tx, ty, tw, th), dim=1)
    
    del tx, ty, tw, th

    return norm_box