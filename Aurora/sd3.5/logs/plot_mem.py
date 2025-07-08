import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_gpu_memory_utilization(file_path, device_id=0, tile_id=0, start_row=None, end_row=None):
    """
    Plot memory utilization curve for specified GPU device and tile
    
    Args:
        file_path: Data file path
        device_id: GPU device ID
        tile_id: Tile ID
        start_row: Starting row index (0-based, inclusive). If None, starts from beginning
        end_row: Ending row index (0-based, exclusive). If None, goes to end
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        print(f"Data file loaded successfully, total {len(df)} rows")
        print(f"Column names: {list(df.columns)}")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    # Filter data for specified GPU and tile
    filtered_data = df[(df['DeviceId'] == device_id) & (df['TileId'] == tile_id)].copy()
    print(f"Filtered data for GPU {device_id}, Tile {tile_id}: {len(filtered_data)} rows")
    
    if len(filtered_data) == 0:
        print("No matching data found")
        return
    
    # Process GPU Memory Utilization data
    memory_util_col = 'GPU Memory Utilization (%)'
    
    # Replace N/A values with NaN, then convert to numeric
    filtered_data[memory_util_col] = filtered_data[memory_util_col].replace(['N/A', '  N/A'], np.nan)
    filtered_data[memory_util_col] = pd.to_numeric(filtered_data[memory_util_col], errors='coerce')
    
    # Remove NaN values
    valid_data = filtered_data.dropna(subset=[memory_util_col])
    print(f"Valid memory utilization data: {len(valid_data)} rows")
    
    if len(valid_data) == 0:
        print("No valid memory utilization data")
        return
    
    # Apply row range selection if specified
    if start_row is not None or end_row is not None:
        original_length = len(valid_data)
        start_idx = start_row if start_row is not None else 0
        end_idx = end_row if end_row is not None else len(valid_data)
        
        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(valid_data), end_idx)
        
        if start_idx >= end_idx:
            print(f"Invalid row range: start_row ({start_row}) >= end_row ({end_row})")
            return
            
        valid_data = valid_data.iloc[start_idx:end_idx]
        print(f"Selected row range [{start_idx}:{end_idx}] from {original_length} total rows, resulting in {len(valid_data)} rows")
    
    # Prepare plot data
    memory_utilization = valid_data[memory_util_col].values
    sample_points = np.arange(len(memory_utilization))
    
    # Create figure
    plt.figure(figsize=(12, 6))
    plt.plot(sample_points, memory_utilization, 'b-', linewidth=1.5, alpha=0.8)
    
    # Set axes
    plt.xlabel('Sample Points', fontsize=12)
    plt.ylabel('Memory Utilization (%)', fontsize=12)
    
    # Create title with row range info if applicable
    title = f'GPU {device_id} Tile {tile_id} Memory Utilization'
    if start_row is not None or end_row is not None:
        range_str = f"[Rows {start_row if start_row is not None else 0}:{end_row if end_row is not None else 'end'}]"
        title += f" {range_str}"
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis ticks, show every 20 samples
    max_samples = len(memory_utilization)
    x_ticks = np.arange(0, max_samples, 20)
    if max_samples - 1 not in x_ticks:  # Ensure the last point is also shown
        x_ticks = np.append(x_ticks, max_samples - 1)
    plt.xticks(x_ticks)
    
    # Set y-axis range
    plt.ylim(65, 85)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_util = np.mean(memory_utilization)
    max_util = np.max(memory_utilization)
    min_util = np.min(memory_utilization)
    
    plt.text(0.02, 0.98, f'Mean: {mean_util:.2f}%\nMax: {max_util:.2f}%\nMin: {min_util:.2f}%', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"Average memory utilization: {mean_util:.2f}%")
    print(f"Maximum memory utilization: {max_util:.2f}%")
    print(f"Minimum memory utilization: {min_util:.2f}%")
    print(f"Standard deviation: {np.std(memory_utilization):.2f}%")

def plot_gpu_memory_utilization_multiple_tiles(file_path, device_id=0, tile_ids=[0, 1], start_row=None, end_row=None):
    """
    Plot memory utilization curves for multiple tiles of specified GPU device on the same figure
    
    Args:
        file_path: Data file path
        device_id: GPU device ID
        tile_ids: List of Tile IDs
        start_row: Starting row index (0-based, inclusive). If None, starts from beginning
        end_row: Ending row index (0-based, exclusive). If None, goes to end
    """
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        print(f"Data file loaded successfully, total {len(df)} rows")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Color and line style lists
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    linestyles = ['-', '--', '-.', ':']
    
    all_data = {}
    max_samples = 0
    
    # Process data for each tile
    for i, tile_id in enumerate(tile_ids):
        # Filter data for specified GPU and tile
        filtered_data = df[(df['DeviceId'] == device_id) & (df['TileId'] == tile_id)].copy()
        print(f"Filtered data for GPU {device_id}, Tile {tile_id}: {len(filtered_data)} rows")
        
        if len(filtered_data) == 0:
            print(f"No data found for GPU {device_id}, Tile {tile_id}")
            continue
        
        # Process GPU Memory Utilization data
        memory_util_col = 'GPU Memory Utilization (%)'
        
        # Replace N/A values with NaN, then convert to numeric
        filtered_data[memory_util_col] = filtered_data[memory_util_col].replace(['N/A', '  N/A'], np.nan)
        filtered_data[memory_util_col] = pd.to_numeric(filtered_data[memory_util_col], errors='coerce')
        
        # Remove NaN values
        valid_data = filtered_data.dropna(subset=[memory_util_col])
        print(f"GPU {device_id}, Tile {tile_id} valid memory utilization data: {len(valid_data)} rows")
        
        if len(valid_data) == 0:
            print(f"GPU {device_id}, Tile {tile_id} has no valid memory utilization data")
            continue
        
        # Apply row range selection if specified
        if start_row is not None or end_row is not None:
            original_length = len(valid_data)
            start_idx = start_row if start_row is not None else 0
            end_idx = end_row if end_row is not None else len(valid_data)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(valid_data), end_idx)
            
            if start_idx >= end_idx:
                print(f"Invalid row range for GPU {device_id}, Tile {tile_id}: start_row ({start_row}) >= end_row ({end_row})")
                continue
                
            valid_data = valid_data.iloc[start_idx:end_idx]
            print(f"GPU {device_id}, Tile {tile_id}: Selected row range [{start_idx}:{end_idx}] from {original_length} total rows, resulting in {len(valid_data)} rows")
        
        # Prepare plot data
        memory_utilization = valid_data[memory_util_col].values
        sample_points = np.arange(len(memory_utilization))
        
        # Store data for later analysis
        all_data[f'GPU{device_id}_Tile{tile_id}'] = memory_utilization
        max_samples = max(max_samples, len(memory_utilization))
        
        # Plot curve
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        plt.plot(sample_points, memory_utilization, 
                color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8,
                label=f'GPU {device_id} Tile {tile_id}')
    
    if not all_data:
        print("No valid data to plot")
        return
    
    # Set axes
    plt.xlabel('Sample Points', fontsize=12)
    plt.ylabel('Memory Utilization (%)', fontsize=12)
    
    # Create title with row range info if applicable
    title = f'GPU {device_id} Memory Utilization Comparison (Tiles {tile_ids})'
    if start_row is not None or end_row is not None:
        range_str = f" [Rows {start_row if start_row is not None else 0}:{end_row if end_row is not None else 'end'}]"
        title += range_str
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis ticks, show every 20 samples
    x_ticks = np.arange(0, max_samples, 20)
    if max_samples - 1 not in x_ticks:  # Ensure the last point is also shown
        x_ticks = np.append(x_ticks, max_samples - 1)
    plt.xticks(x_ticks)
    
    # Set y-axis range
    plt.ylim(65, 85)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Add statistics
    stats_text = []
    for key, data in all_data.items():
        mean_util = np.mean(data)
        max_util = np.max(data)
        min_util = np.min(data)
        stats_text.append(f'{key}:\n  Mean: {mean_util:.1f}%\n  Max: {max_util:.1f}%\n  Min: {min_util:.1f}%')
    
    plt.text(0.02, 0.98, '\n\n'.join(stats_text), 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistics:")
    for key, data in all_data.items():
        print(f"\n{key}:")
        print(f"  Average memory utilization: {np.mean(data):.2f}%")
        print(f"  Maximum memory utilization: {np.max(data):.2f}%")
        print(f"  Minimum memory utilization: {np.min(data):.2f}%")
        print(f"  Standard deviation: {np.std(data):.2f}%")
        print(f"  Data points: {len(data)}")

def plot_multiple_gpu_tiles(file_path, gpu_tile_combinations, start_row=None, end_row=None, save_png=True, output_dir="./"):
    """
    Plot memory utilization curves for multiple GPU-Tile combinations on the same figure
    
    Args:
        file_path: Data file path
        gpu_tile_combinations: List of tuples [(gpu_id, tile_id), (gpu_id, tile_id), ...]
        start_row: Starting row index (0-based, inclusive). If None, starts from beginning
        end_row: Ending row index (0-based, exclusive). If None, goes to end
        save_png: Whether to save the plot as PNG file
        output_dir: Directory to save the PNG file
    
    Example:
        gpu_tile_combinations = [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    
    import os
    
    # Read CSV file
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        print(f"Data file loaded successfully, total {len(df)} rows")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return
    
    if not gpu_tile_combinations:
        print("No GPU-Tile combinations specified")
        return
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Color and marker combinations for better distinction
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    all_data = {}
    max_samples = 0
    valid_combinations = []
    
    # Process data for each GPU-Tile combination
    for i, (gpu_id, tile_id) in enumerate(gpu_tile_combinations):
        print(f"\nProcessing GPU {gpu_id}, Tile {tile_id}...")
        
        # Filter data for specified GPU and tile
        filtered_data = df[(df['DeviceId'] == gpu_id) & (df['TileId'] == tile_id)].copy()
        print(f"Filtered data for GPU {gpu_id}, Tile {tile_id}: {len(filtered_data)} rows")
        
        if len(filtered_data) == 0:
            print(f"No data found for GPU {gpu_id}, Tile {tile_id}")
            continue
        
        # Process GPU Memory Utilization data
        memory_util_col = 'GPU Memory Utilization (%)'
        
        # Replace N/A values with NaN, then convert to numeric
        filtered_data[memory_util_col] = filtered_data[memory_util_col].replace(['N/A', '  N/A'], np.nan)
        filtered_data[memory_util_col] = pd.to_numeric(filtered_data[memory_util_col], errors='coerce')
        
        # Remove NaN values
        valid_data = filtered_data.dropna(subset=[memory_util_col])
        print(f"GPU {gpu_id}, Tile {tile_id} valid memory utilization data: {len(valid_data)} rows")
        
        if len(valid_data) == 0:
            print(f"GPU {gpu_id}, Tile {tile_id} has no valid memory utilization data")
            continue
        
        # Apply row range selection if specified
        if start_row is not None or end_row is not None:
            original_length = len(valid_data)
            start_idx = start_row if start_row is not None else 0
            end_idx = end_row if end_row is not None else len(valid_data)
            
            # Ensure indices are within bounds
            start_idx = max(0, start_idx)
            end_idx = min(len(valid_data), end_idx)
            
            if start_idx >= end_idx:
                print(f"Invalid row range for GPU {gpu_id}, Tile {tile_id}: start_row ({start_row}) >= end_row ({end_row})")
                continue
                
            valid_data = valid_data.iloc[start_idx:end_idx]
            print(f"GPU {gpu_id}, Tile {tile_id}: Selected row range [{start_idx}:{end_idx}] from {original_length} total rows, resulting in {len(valid_data)} rows")
        
        # Prepare plot data
        memory_utilization = valid_data[memory_util_col].values
        sample_points = np.arange(len(memory_utilization))
        
        # Store data for later analysis
        data_key = f'GPU{gpu_id}_Tile{tile_id}'
        all_data[data_key] = memory_utilization
        max_samples = max(max_samples, len(memory_utilization))
        valid_combinations.append((gpu_id, tile_id))
        
        # Plot curve with distinct style
        color = colors[i % len(colors)]
        linestyle = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        
        plt.plot(sample_points, memory_utilization, 
                color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8,
                marker=marker, markersize=3, markevery=max(1, len(sample_points)//20),
                label=f'GPU {gpu_id} Tile {tile_id}')
    
    if not all_data:
        print("No valid data to plot")
        plt.close()
        return
    
    # Set axes
    plt.xlabel('Sample Points', fontsize=12)
    plt.ylabel('Memory Utilization (%)', fontsize=12)
    
    # Create title with combination info
    gpu_list = sorted(list(set([gpu for gpu, tile in valid_combinations])))
    tile_list = sorted(list(set([tile for gpu, tile in valid_combinations])))
    title = f'Multi-GPU Memory Utilization Comparison\nGPUs: {gpu_list}, Tiles: {tile_list}'
    
    if start_row is not None or end_row is not None:
        range_str = f" [Rows {start_row if start_row is not None else 0}:{end_row if end_row is not None else 'end'}]"
        title += range_str
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis ticks, show every 20 samples
    x_ticks = np.arange(0, max_samples, 20)
    if max_samples - 1 not in x_ticks:  # Ensure the last point is also shown
        x_ticks = np.append(x_ticks, max_samples - 1)
    plt.xticks(x_ticks)
    
    # Set y-axis range
    plt.ylim(65, 85)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add statistics in a more compact format
    stats_text = []
    for key, data in all_data.items():
        mean_util = np.mean(data)
        stats_text.append(f'{key}: {mean_util:.1f}%')
    
    # Display stats in multiple columns if too many
    if len(stats_text) > 6:
        mid = len(stats_text) // 2
        left_stats = '\n'.join(stats_text[:mid])
        right_stats = '\n'.join(stats_text[mid:])
        stats_display = f'{left_stats}\n\n{right_stats}'
    else:
        stats_display = '\n'.join(stats_text)
    
    plt.text(0.02, 0.98, f'Mean Utilization:\n{stats_display}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    # Save PNG file if requested
    if save_png:
        # Create filename
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        gpu_str = '_'.join([f'GPU{gpu}' for gpu in gpu_list])
        tile_str = '_'.join([f'Tile{tile}' for tile in tile_list])
        
        range_str = ""
        if start_row is not None or end_row is not None:
            range_str = f"_rows{start_row if start_row is not None else 0}to{end_row if end_row is not None else 'end'}"
        
        filename = f"{base_filename}_{gpu_str}_{tile_str}{range_str}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nPlot saved as: {filepath}")
    
    plt.show()
    
    # Print detailed statistics
    print(f"\nDetailed Statistics:")
    for key, data in all_data.items():
        print(f"\n{key}:")
        print(f"  Average memory utilization: {np.mean(data):.2f}%")
        print(f"  Maximum memory utilization: {np.max(data):.2f}%")
        print(f"  Minimum memory utilization: {np.min(data):.2f}%")
        print(f"  Standard deviation: {np.std(data):.2f}%")
        print(f"  Data points: {len(data)}")

def create_gpu_tile_combinations(gpu_ids, tile_ids):
    """
    Helper function to create GPU-Tile combinations
    
    Args:
        gpu_ids: List of GPU IDs [0, 1, 2, ...]
        tile_ids: List of Tile IDs [0, 1, ...]
    
    Returns:
        List of tuples [(gpu_id, tile_id), ...]
    """
    combinations = []
    for gpu_id in gpu_ids:
        for tile_id in tile_ids:
            combinations.append((gpu_id, tile_id))
    return combinations
    """
    Explore data to see available GPU devices and tiles
    """
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        print("Data Exploration:")
        print(f"Available GPU device IDs: {sorted(df['DeviceId'].unique())}")
        print(f"Available Tile IDs: {sorted(df['TileId'].unique())}")
        
        # Check data volume for each GPU and tile combination
        print("\nData count for each GPU-Tile combination:")
        combination_counts = df.groupby(['DeviceId', 'TileId']).size()
        for (device, tile), count in combination_counts.items():
            print(f"GPU {device}, Tile {tile}: {count} rows")
            
        # Check availability of memory utilization data
        memory_util_col = 'GPU Memory Utilization (%)'
        valid_memory_data = df[df[memory_util_col] != 'N/A'][df[memory_util_col] != '  N/A']
        print(f"\nTotal valid memory utilization data: {len(valid_memory_data)} rows")
        
        # Show row range information
        print(f"\nRow Range Information:")
        print(f"Total rows in file: {len(df)}")
        print(f"Row indices range from 0 to {len(df)-1}")
        print(f"Example row selections:")
        print(f"  - First 100 rows: start_row=0, end_row=100")
        print(f"  - Middle 100 rows: start_row=100, end_row=200")
        print(f"  - Last 100 rows: start_row={max(0, len(df)-100)}, end_row={len(df)}")
        print(f"  - From row 50 to end: start_row=50, end_row=None")
        
    except Exception as e:
        print(f"Failed to explore data: {e}")

def get_valid_data_range(file_path, device_id=0, tile_id=0):
    """
    Get the range of valid data rows for a specific GPU device and tile
    
    Args:
        file_path: Data file path
        device_id: GPU device ID
        tile_id: Tile ID
    
    Returns:
        tuple: (total_rows, valid_rows_count, first_valid_index, last_valid_index)
    """
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        
        # Filter data for specified GPU and tile
        filtered_data = df[(df['DeviceId'] == device_id) & (df['TileId'] == tile_id)].copy()
        
        if len(filtered_data) == 0:
            print(f"No data found for GPU {device_id}, Tile {tile_id}")
            return None
        
        # Process GPU Memory Utilization data
        memory_util_col = 'GPU Memory Utilization (%)'
        filtered_data[memory_util_col] = filtered_data[memory_util_col].replace(['N/A', '  N/A'], np.nan)
        filtered_data[memory_util_col] = pd.to_numeric(filtered_data[memory_util_col], errors='coerce')
        
        # Find valid data
        valid_mask = filtered_data[memory_util_col].notna()
        valid_indices = filtered_data[valid_mask].index.tolist()
        
        if len(valid_indices) == 0:
            print(f"No valid memory utilization data for GPU {device_id}, Tile {tile_id}")
            return None
        
        # Map back to original filtered data indices
        filtered_valid_indices = [i for i, valid in enumerate(valid_mask) if valid]
        
        result = {
            'total_filtered_rows': len(filtered_data),
            'valid_rows_count': len(filtered_valid_indices),
            'first_valid_index': min(filtered_valid_indices),
            'last_valid_index': max(filtered_valid_indices)
        }
        
        print(f"\nValid Data Range for GPU {device_id}, Tile {tile_id}:")
        print(f"  Total filtered rows: {result['total_filtered_rows']}")
        print(f"  Valid data rows: {result['valid_rows_count']}")
        print(f"  First valid row index: {result['first_valid_index']}")
        print(f"  Last valid row index: {result['last_valid_index']}")
        print(f"  Suggested ranges:")
        print(f"    - All valid data: start_row={result['first_valid_index']}, end_row={result['last_valid_index']+1}")
        print(f"    - First half: start_row={result['first_valid_index']}, end_row={result['first_valid_index'] + result['valid_rows_count']//2}")
        print(f"    - Second half: start_row={result['first_valid_index'] + result['valid_rows_count']//2}, end_row={result['last_valid_index']+1}")
        
        return result
        
    except Exception as e:
        print(f"Failed to get valid data range: {e}")
        return None
def explore_data(file_path):
    """
    Explore data to see available GPU devices and tiles
    """
    try:
        df = pd.read_csv(file_path, skipinitialspace=True)
        print("Data Exploration:")
        print(f"Available GPU device IDs: {sorted(df['DeviceId'].unique())}")
        print(f"Available Tile IDs: {sorted(df['TileId'].unique())}")
        
        # Check data volume for each GPU and tile combination
        print("\nData count for each GPU-Tile combination:")
        combination_counts = df.groupby(['DeviceId', 'TileId']).size()
        for (device, tile), count in combination_counts.items():
            print(f"GPU {device}, Tile {tile}: {count} rows")
            
        # Check availability of memory utilization data
        memory_util_col = 'GPU Memory Utilization (%)'
        valid_memory_data = df[df[memory_util_col] != 'N/A'][df[memory_util_col] != '  N/A']
        print(f"\nTotal valid memory utilization data: {len(valid_memory_data)} rows")
        
        # Show row range information
        print(f"\nRow Range Information:")
        print(f"Total rows in file: {len(df)}")
        print(f"Row indices range from 0 to {len(df)-1}")
        print(f"Example row selections:")
        print(f"  - First 100 rows: start_row=0, end_row=100")
        print(f"  - Middle 100 rows: start_row=100, end_row=200")
        print(f"  - Last 100 rows: start_row={max(0, len(df)-100)}, end_row={len(df)}")
        print(f"  - From row 50 to end: start_row=50, end_row=None")
        
    except Exception as e:
        print(f"Failed to explore data: {e}")




# Usage example
if __name__ == "__main__":
    file_path = "ddp_smi_ipex_50.txt"  # Your file name
    
    # First explore the data
    explore_data(file_path)
    
    # Get valid data range for specific GPU and tile
    print("\n" + "="*50)
    get_valid_data_range(file_path, device_id=0, tile_id=0)
    
    print("\n" + "="*70 + "\n")
    
    # Example 1: Plot multiple GPU-Tile combinations manually specified
    # print("Example 1: Multiple GPU-Tile combinations (manual selection)")
    # gpu_tile_combos = [(0, 0), (0, 1), (1, 0), (1, 1)]  # GPU0-Tile0, GPU0-Tile1, GPU1-Tile0, GPU1-Tile1
    # plot_multiple_gpu_tiles(file_path, gpu_tile_combos, save_png=True, output_dir="./plots/")
    
    # print("\n" + "="*70 + "\n")
    
    # # Example 2: Use helper function to create combinations
    # print("Example 2: All combinations of specified GPUs and Tiles")
    # gpu_ids = [0, 1]      # GPUs to include
    # tile_ids = [0, 1]     # Tiles to include
    # all_combos = create_gpu_tile_combinations(gpu_ids, tile_ids)
    # print(f"Generated combinations: {all_combos}")
    # plot_multiple_gpu_tiles(file_path, all_combos, save_png=True, output_dir="./plots/")
    
    # print("\n" + "="*70 + "\n")
    
    # Example 3: Specific row range with multiple GPU-Tile combinations
    print("Example 3: Multiple GPU-Tile combinations with row range (rows 50-200)")
    selected_combos = [(0, 0), (0, 1), (1,0),(1,1),(2, 0),(2,1),(3,0),(3,1),(4,0),(4,1),(5,0),(5,1)]  # Mix of different GPUs and tiles
    plot_multiple_gpu_tiles(file_path, selected_combos, 
                           start_row=1000, end_row=1300, 
                           save_png=True, output_dir="./plots/")
    
    # print("\n" + "="*50 + "\n")
    
    # # Example 4: Single GPU, multiple tiles (original functionality)
    # print("Example 4: GPU 0 Tile 0 and Tile 1 comparison (original function):")
    # plot_gpu_memory_utilization_multiple_tiles(file_path, device_id=0, tile_ids=[0, 1], 
    #                                            start_row=50, end_row=150)
    
    # print("\n" + "="*30 + "\n")
    
    # More examples for different scenarios:
    
    # All tiles for GPU 0
    # gpu0_all_tiles = create_gpu_tile_combinations([0], [0, 1, 2, 3])
    # plot_multiple_gpu_tiles(file_path, gpu0_all_tiles, save_png=True)
    
    # Compare same tile across different GPUs
    # tile0_all_gpus = create_gpu_tile_combinations([0, 1, 2, 3], [0])
    # plot_multiple_gpu_tiles(file_path, tile0_all_gpus, save_png=True)
    
    # Custom selection
    # custom_combos = [(0, 0), (1, 1), (2, 0), (3, 1)]
    # plot_multiple_gpu_tiles(file_path, custom_combos, save_png=True)
    
    # print("\nCustomization Examples:")
    # print("# All tiles for GPU 0:")
    # print("gpu0_all_tiles = create_gpu_tile_combinations([0], [0, 1, 2, 3])")
    # print("plot_multiple_gpu_tiles(file_path, gpu0_all_tiles, save_png=True)")
    # print("\n# Compare same tile across different GPUs:")
    # print("tile0_all_gpus = create_gpu_tile_combinations([0, 1, 2, 3], [0])")
    # print("plot_multiple_gpu_tiles(file_path, tile0_all_gpus, save_png=True)")
    # print("\n# Custom selection:")
    # print("custom_combos = [(0, 0), (1, 1), (2, 0), (3, 1)]")
    # print("plot_multiple_gpu_tiles(file_path, custom_combos, save_png=True)")