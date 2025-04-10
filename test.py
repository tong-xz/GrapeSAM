import torch

def group_small_masks_by_instance(large_masks, small_masks, threshold=0.5):
    """
    Groups small masks based on their overlap with large masks.
    
    Args:
        large_masks (torch.Tensor): A tensor of shape (N, H, W), where N is the number of large masks.
        small_masks (torch.Tensor): A tensor of shape (M, H, W), where M is the number of small masks.
        threshold (float): A float between 0 and 1, representing the minimum overlap percentage required for a small mask to be considered valid.

    Returns:
        list of lists: A list of N lists, where each sublist contains small masks that overlap with the corresponding large mask instance.
    """
    
    
    # Initialize a list of N empty lists (one for each large mask instance)
    grouped_masks = [[] for _ in range(large_masks.shape[0])]

    # Iterate over each small mask
    for j in range(small_masks.shape[0]):  # M small masks
        small_mask = small_masks[j]
        
        # Compute the overlap with all large masks (N masks)
        for i in range(large_masks.shape[0]):  # N large masks
            large_mask = large_masks[i]
            
            # Calculate overlap by element-wise multiplication
            overlap = large_mask * small_mask  # (H, W)
            
            # Calculate the percentage of overlap
            overlap_area = overlap.sum().item()  # Sum of non-zero elements in the overlap
            total_area = small_mask.sum()  # Total number of elements (H * W)
            overlap_percentage = overlap_area / total_area

            # If the overlap percentage is above the threshold, group this small mask with the large mask instance
            if overlap_percentage >= threshold:
                grouped_masks[i].append(small_mask)
                break  # We found the large mask instance for this small mask, no need to check other large masks

    return grouped_masks

# Example Usage
N, M, H, W = 5, 10, 128, 128  # Example dimensions
large_masks = torch.randint(0, 2, (N, H, W))  # Random binary large masks
small_masks = torch.randint(0, 2, (M, H, W))  # Random binary small masks

threshold = 0.2  # 20% overlap threshold
grouped_small_masks = group_small_masks_by_instance(large_masks, small_masks, threshold)

# Print the number of valid small masks for each large mask instance
for i, masks in enumerate(grouped_small_masks):
    print(f"Instance {i} has {len(masks)} valid small masks.")
