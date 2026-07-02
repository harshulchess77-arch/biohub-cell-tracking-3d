import numpy as np
from skimage import exposure

def normalize_volume(volume_slice):
    """
    Standardizes intensity variations by clipping extreme values 
    and scaling the intensities between 0 and 1.
    """
    # Convert 16-bit data to float32 for processing mathematical operations safely
    img_float = volume_slice.astype(np.float32)
    
    # Robust contrast stretching (ignores top/bottom 1% pixel intensity outliers)
    p2, p98 = np.percentile(img_float, (2, 98))
    img_rescaled = exposure.rescale_intensity(img_float, in_range=(p2, p98), out_range=(0, 1))
    
    return img_rescaled

def extract_3d_crop(volume, t, z, y, x, crop_size=(16, 64, 64)):
    """
    Extracts a small 3D bounding box context around a cell coordinate.
    Useful for feeding small, local target regions into a neural network.
    """
    z_dim, y_dim, x_dim = volume.shape[1], volume.shape[2], volume.shape[3]
    cz, cy, cx = crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2
    
    # Calculate bounding ranges while preventing edge boundaries from clipping
    z_start = max(0, min(int(z) - cz, z_dim - crop_size[0]))
    y_start = max(0, min(int(y) - cy, y_dim - crop_size[1]))
    x_start = max(0, min(int(x) - cx, x_dim - crop_size[2]))
    
    # Pull out the raw sub-cube from the lazy streaming volume
    crop = volume[int(t), z_start:z_start+crop_size[0], y_start:y_start+crop_size[1], x_start:x_start+crop_size[2]].compute()
    
    return normalize_volume(crop)

if __name__ == "__main__":
    print("--- Biohub Image Processing Engine Test ---")
    from data_loader import get_zarr_stream
    import os
    
    TRAIN_DIR = os.path.join("data", "train")
    zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.zarr')]
    
    if zarr_folders:
        sample_name = zarr_folders[0]
        video_cube = get_zarr_stream(sample_name, base_dir=TRAIN_DIR)
        
        # Test coordinates using our first cell node from the graph output: Time=0, Z=63, Y=222, X=249
        print("\nExtracting normalized 3D patch around cell target point...")
        cell_patch = extract_3d_crop(video_cube, t=0, z=63, y=222, x=249)
        
        print("\n--- IMAGE PROCESSING SUCCESSFUL ---")
        print(f"Sub-Volume Patch Shape: {cell_patch.shape}")
        print(f"Intensity Range: Min={cell_patch.min():.4f}, Max={cell_patch.max():.4f}")