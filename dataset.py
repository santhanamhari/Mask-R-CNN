import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # TODO Initialize  Dataset
        #############################################
        img_path, mask_path, label_path, bb_path = path
        self.images = h5py.File(img_path, 'r')['data']
        self.masks = h5py.File(mask_path, 'r')['data']
        self.labels = np.load(label_path , allow_pickle=True)
        self.bbs = np.load(bb_path , allow_pickle=True)
        
        self.masks_idx = []
        idx = 0
        for i in range(len(self.labels)):
          self.masks_idx.append(idx)
          idx += len(self.labels[i])
        self.masks_idx = np.array(self.masks_idx)



    # In this function for given index we rescale the image and the corresponding  masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, index):
        ################################
        # TODO return transformed images,labels,masks,boxes,index
        ################################
        image = self.images[index]
        label = self.labels[index]
        bbox = self.bbs[index]
        mask = self.masks[self.masks_idx[index]: self.masks_idx[index]+len(label)]

        # use pre process batch
        label = torch.tensor(label)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(image, mask, bbox)

        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]

        return transed_img, label, transed_mask, transed_bbox, index


    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox):
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################

        img_h = img.shape[1]
        img_w = img.shape[2]

        # IMAGE
        # Normalize pixel values to [0,1]
        transform_image = torch.from_numpy((img / 255.0).astype('float64'))
        # Rescale the images to 800 x 1066
        transform_image = torch.nn.functional.interpolate(transform_image.unsqueeze(dim=0), size=(800, 1066))
        transform_image = transform_image.squeeze(dim=0)
        # Normalize each channel with means  [0.485,0.456,0.406]  and standard deviations  [0.229,0.224,0.225] .
        transform_image = torchvision.transforms.functional.normalize(transform_image, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        #Zero pad the image to  800 x 1088
        transform_image = torch.nn.functional.pad(transform_image, (11,11))
        
        # MASK
        transform_mask = torch.from_numpy((mask.astype('float32')))
        transform_mask = torch.nn.functional.interpolate(transform_mask.unsqueeze(dim=0), size=(800, 1066))
        transform_mask = torch.nn.functional.pad(transform_mask, (11,11))

        # BBOXES
        bbox = torch.tensor(bbox)
        transform_bbox = torch.zeros_like(bbox)
        
        x1, y1, x2, y2 = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
        transform_bbox[:, 0] = bbox[:,0]/img_w
        transform_bbox[:, 1] = bbox[:,1]/img_h
        transform_bbox[:, 2] = bbox[:,2]/img_w
        transform_bbox[:, 3] = bbox[:,3]/img_h

        img = transform_image
        mask = transform_mask
        bbox = transform_bbox
        

        assert img.squeeze(0).shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.squeeze(0).shape[0]

        return img.squeeze(0), mask.squeeze(0), bbox
    

    
    def __len__(self):
        return len(self.images)


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
      images = torch.from_numpy(np.zeros((len(batch), 3, 800, 1088)).astype('float64'))
      out_batch = {"images": images, "labels": [], "masks": [], "bbox": [], "index": []}
        
      for i, b in enumerate(batch):
          #out_batch["images"].append(b[0])
          out_batch["images"][i] = b[0]
          out_batch["labels"].append(b[1])
          out_batch["masks"].append(b[2])
          out_batch["bbox"].append(b[3])
          out_batch["index"].append(b[4])
      
      return out_batch


    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)
        

if __name__ == '__main__':
    # file path and make a list
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

  
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    rpn_net = RPNHead()
    # push the randomized training data into the dataloader

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    batch_size = 1
    train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()
    
    old_flat = 0
    new_flat = 0

    for i,batch in enumerate(train_loader,0):
        images=batch['images'][0]
        indexes=batch['index']
        boxes=batch['bbox']
        gt,ground_coord=rpn_net.create_batch_truth(boxes,indexes,images.shape[-2:])

        # Flatten the ground truth and the anchors
        flatten_coord,flatten_gt,flatten_anchors=output_flattening(ground_coord,gt,rpn_net.get_anchors())

        # Decode the ground truth box to get the upper left and lower right corners of the ground truth boxes
        decoded_coord=output_decoding(flatten_coord,flatten_anchors)
        
        # Plot the image and the anchor boxes with the positive labels and their corresponding ground truth box
        images = transforms.functional.normalize(images,
                                                      [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                                      [1/0.229, 1/0.224, 1/0.225], inplace=False)
        fig,ax=plt.subplots(1,1)
        ax.imshow(images.permute(1,2,0))
        
        find_cor=(flatten_gt==1).nonzero()[0]
        find_neg=(flatten_gt==-1).nonzero()
             
        for elem in find_cor:
            coord=torch.tensor(decoded_coord[elem,:]).view(-1)
            anchor=torch.tensor(flatten_anchors[elem,:]).view(-1)

            col='r'
            rect=patches.Rectangle((coord[0],coord[1]),coord[2]-coord[0],coord[3]-coord[1],fill=False,color=col)
            ax.add_patch(rect)
            rect=patches.Rectangle((anchor[0]-anchor[2]/2,anchor[1]-anchor[3]/2),anchor[2],anchor[3],fill=False,color='b')
            ax.add_patch(rect)

        plt.show()
 
        if(i>20):
            break


