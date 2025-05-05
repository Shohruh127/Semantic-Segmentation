# datasets/cityscapes.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob # Use glob for easier file searching

class CityScapes(Dataset):
    # (Keep the label mapping dictionaries as provided in the previous example,
    # e.g., self.mapping_20 or self.kitti_eval_mapping)
    # We'll use self.kitti_eval_mapping here, assuming evaluation uses the 19 official classes
    # and we need an ignore index (e.g., 255 or -1). Let's map non-eval classes to 255 (ignore).
    ignore_index = 255
    # Based on https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    id_to_trainid = {
        0: ignore_index, 1: ignore_index, 2: ignore_index, 3: ignore_index, 4: ignore_index, 5: ignore_index, 
        6: ignore_index, 7: 0, 8: 1, 9: ignore_index, 10: ignore_index, 11: 2, 12: 3, 13: 4, 
        14: ignore_index, 15: ignore_index, 16: ignore_index, 17: 5, 18: ignore_index, 19: 6, 20: 7, 
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: ignore_index, 
        30: ignore_index, 31: 16, 32: 17, 33: 18, -1: ignore_index
    } # Maps Cityscapes IDs (0-33, -1) to Train IDs (0-18) + ignore_index

    def __init__(self, root_dir, split='train', transform=None):
        super(CityScapes, self).__init__()
        self.root_dir = root_dir
        self.split = split # 'train', 'val', or 'test'
        self.transform = transform
        self.files = []
        self.labels = [] # Let's store labels separately for clarity

        self._set_files() # Call helper function to populate self.files and self.labels

    def _set_files(self):
        # Define paths based on Cityscapes structure
        # e.g., root_dir/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        # e.g., root_dir/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        img_dir = os.path.join(self.root_dir, 'leftImg8bit', self.split)
        lbl_dir = os.path.join(self.root_dir, 'gtFine', self.split) # Use gtFine for labels

        # Check if directories exist
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(lbl_dir) and self.split != 'test': # Labels might not exist for test split
             print(f"Warning: Label directory not found: {lbl_dir}. Ensure it exists for train/val.")
             # Depending on use case, might want to raise FileNotFoundError here too for train/val

        # Use glob to find all image files recursively within the split directory
        # The pattern '*' accounts for the city names (e.g., 'aachen', 'bochum')
        search_pattern_img = os.path.join(img_dir, '*', '*_leftImg8bit.png')
        found_files = glob(search_pattern_img)
        found_files.sort() # Ensure consistent order

        if not found_files:
             print(f"Warning: No image files found matching pattern: {search_pattern_img}")
             return # Stop if no images found

        for img_path in found_files:
            # Construct corresponding label path from image path
            # Example: aachen_000000_000019_leftImg8bit.png -> aachen_000000_000019_gtFine_labelIds.png
            base_name = os.path.basename(img_path)
            city = os.path.basename(os.path.dirname(img_path)) # Get city name (e.g., 'aachen')
            
            # Build label name and path
            lbl_name_base = base_name.replace('_leftImg8bit.png', '')
            lbl_name = f"{lbl_name_base}_gtFine_labelIds.png" # Use labelIds for training
            lbl_path = os.path.join(lbl_dir, city, lbl_name)

            # Add the pair if the label file exists (important for train/val)
            # For the 'test' split, label files won't exist, but we still add the image path
            if os.path.exists(lbl_path):
                self.files.append(img_path)
                self.labels.append(lbl_path)
            elif self.split == 'test':
                 # If testing, we might only have images, append image path and None for label
                 self.files.append(img_path)
                 self.labels.append(None) # Or handle test case differently if needed
            else:
                 print(f"Warning: Label file not found for image {img_path}. Expected at: {lbl_path}")

        print(f"Found {len(self.files)} images in split '{self.split}'")
        if len(self.files) != len(self.labels):
             print("Warning: Mismatch between number of image files and label files found.")

    # --- __len__ and __getitem__ will go here ---
    # ... (Implementation from previous example, adapted for self.files/self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        lbl_path = self.labels[idx] # This might be None if split is 'test'

        image = Image.open(img_path).convert('RGB')
        
        # Handle case where labels might not exist (e.g., test set)
        if lbl_path is None:
            # If no label path, maybe return image and a dummy target or handle in transform
            label = None # Or create a dummy tensor: torch.zeros(image.size[1], image.size[0], dtype=torch.long)
        else:
            label_pil = Image.open(lbl_path) # Load as PIL Image
            label = np.array(label_pil, dtype=np.uint8) # Convert to numpy array

            # Remap label IDs to train IDs
            label_copy = label.copy()
            for k, v in self.id_to_trainid.items():
                 label_copy[label == k] = v
            label = Image.fromarray(label_copy.astype(np.uint8)) # Convert back to PIL Image for transforms

        # Apply transforms (ensure transforms handle image and potentially None label)
        if self.transform:
             # Modify transform call if label can be None or needs specific handling
             # Assuming transform takes image and label, returns transformed versions
             image, label = self.transform(image, label) 

        # If label is not None and transform didn't convert it, ensure it's LongTensor
        if label is not None and not isinstance(label, torch.Tensor):
             label = torch.from_numpy(np.array(label)).long() # Convert final PIL label to LongTensor
        elif isinstance(label, torch.Tensor):
             label = label.long() # Ensure it's long type
             
        # Handle the case where label is None for test set if necessary
        if label is None:
            # Return image and maybe a dummy tensor or handle outside the dataset
            return image, torch.zeros(1, dtype=torch.long) # Example dummy return

        return image, label # Return image and label tensor
