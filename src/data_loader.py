import os
import zarr
import dask.array as da

def get_zarr_stream(dataset_name, base_dir=os.path.join("data", "train")):
    """
    Opens a 4D Biohub cell tracking dataset without loading the whole file into RAM.
    Returns a dask array mapped to dimensions (Time, Z, Y, X).
    
    Parameters:
    -----------
    dataset_name : str
        The name of the specific .zarr folder (e.g., 'sample_001.zarr')
    base_dir : str
        The parent folder directory where the dataset sits. Default is 'data/train'.
    """
    # Build path to the '0' sub-folder where the voxel tensor lives
    target_path = os.path.join(base_dir, dataset_name, "0")
    
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Could not find data matrix at: {target_path}")
        
    print(f"Connecting stream to: {target_path}")
    
    # Open the zarr structure (reads metadata only, 0 bytes of images loaded)
    zarr_store = zarr.open(target_path, mode='r')
    
    # Wrap it in a Dask array for lazy-loaded chunk handling
    lazy_volume = da.from_zarr(zarr_store)
    return lazy_volume

if __name__ == "__main__":
    print("--- Biohub Data Loader Test ---")
    
    # Since your data has 'train' and 'test' subfolders, we target the train folder first
    TRAIN_DIR = os.path.join("data", "train")
    
    try:
        # Check if the folder exists
        if not os.path.exists(TRAIN_DIR):
            raise FileNotFoundError(f"The path '{TRAIN_DIR}' does not exist. Please check your folder structure.")
            
        # Automatically find the first .zarr folder inside data/train/
        zarr_folders = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.zarr')]
        
        if not zarr_folders:
            print(f"No .zarr folders found inside '{TRAIN_DIR}'!")
            print("Please double check that your unzipped folders are inside data/train/")
        else:
            # Pick the first training sample found dynamically
            sample_name = zarr_folders[0]
            print(f"Found training sample: {sample_name}")
            
            # Pass the full subfolder path to our streamer function
            video_cube = get_zarr_stream(sample_name, base_dir=TRAIN_DIR)
            
            print("\n--- CONNECTION SUCCESSFUL ---")
            print(f"Dimensions (T, Z, Y, X): {video_cube.shape}")
            print(f"Data Type: {video_cube.dtype}")
            
            print("Testing dynamic frame slice retrieval...")
            sample_slice = video_cube[0, 0, :, :].compute()
            print(f"Successfully streamed matrix slice! Shape: {sample_slice.shape}")
            
    except Exception as e:
        print(f"Error during test run: {e}")