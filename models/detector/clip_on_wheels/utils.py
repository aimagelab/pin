import torch
import torchvision
import numpy as np

@torch.no_grad()
def squared_crop(images, masks, resize_dim=224, padding=10, device="cuda", avoid_resizing=False):
    # images = images.permute(0, 2, 3, 1).to(self.device)
    # masks = masks.to(self.device)
    shortest_side = int(min(images.shape[1:3]) / 2)
    image_size_x = images.shape[2]
    image_size_y = images.shape[1]
    batch_size = images.shape[0]
        
    # Find mask bounding boxes
    
    cumsum_x = masks.sum(dim=1).cumsum(dim=1).float()
    xmaxs = cumsum_x.argmax(dim=1, keepdim=True) + padding
    cumsum_x[cumsum_x == 0] = np.inf
    xmins = cumsum_x.argmin(dim=1, keepdim=True) - padding
    
    cumsum_y = masks.sum(dim=2).cumsum(dim=1).float()
    ymaxs = cumsum_y.argmax(dim=1, keepdim=True) + padding
    cumsum_y[cumsum_y == 0] = np.inf
    ymins = cumsum_y.argmin(dim=1, keepdim=True) - padding
    
    # Compute mask centers        
    mask_center_x = (xmaxs+xmins) / 2
    mask_center_y = (ymaxs+ymins) / 2
    
    # Get squared bounding boxes
    
    left_distance = (mask_center_x - xmins).unsqueeze(-1)
    right_distance = (xmaxs - mask_center_x).unsqueeze(-1)
    top_distance = (mask_center_y - ymins).unsqueeze(-1)
    bottom_distance = (ymaxs - mask_center_y).unsqueeze(-1)

    max_distance = torch.cat((left_distance, right_distance, top_distance, bottom_distance), dim=2).max(dim=2).values.int()
    max_distance[max_distance > shortest_side] = shortest_side
    
    del left_distance, right_distance, top_distance, bottom_distance
    
    xmaxs = mask_center_x + max_distance
    xmins = mask_center_x - max_distance
    ymaxs = mask_center_y + max_distance
    ymins = mask_center_y - max_distance
    
    xmins[xmaxs > image_size_x] = xmins[xmaxs > image_size_x] - (xmaxs[xmaxs > image_size_x] - image_size_x).int()
    xmaxs[xmaxs > image_size_x] = image_size_x
    ymins[ymaxs > image_size_y] = ymins[ymaxs > image_size_y] - (ymaxs[ymaxs > image_size_y] - image_size_y).int()
    ymaxs[ymaxs > image_size_y] = image_size_y
    xmaxs[xmins < 0] = xmaxs[xmins < 0] - xmins[xmins < 0]
    xmins[xmins < 0] = 0
    ymaxs[ymins < 0] = ymaxs[ymins < 0] - ymins[ymins < 0]
    ymins[ymins < 0] = 0

    batch_index = torch.arange(batch_size).unsqueeze(1).to(device)
    boxes = torch.cat((batch_index, xmins, ymins, xmaxs, ymaxs), 1)
    
    if avoid_resizing:
        resize_dim = int(xmaxs.max().item() - xmins.min().item())
    
    del xmins, ymins, xmaxs, ymaxs, mask_center_x, mask_center_y, cumsum_x, cumsum_y
            
    cropped_images = torchvision.ops.roi_align(images.permute(0, 3, 1, 2).float(), boxes.float(), resize_dim, aligned=True)
    cropped_masks = torchvision.ops.roi_align(masks.float().unsqueeze(1), boxes.float(), resize_dim, aligned=True).bool().squeeze(1)
        
    return cropped_images.int(), cropped_masks.bool()
