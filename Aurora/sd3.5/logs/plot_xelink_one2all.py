import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def save_plot(filename, plots_folder='plots'):
    """
    Save the current plot as PNG file in the specified folder
    
    Args:
        filename: Name of the file (without extension)
        plots_folder: Folder to save plots in
    """
    # Create plots folder if it doesn't exist
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        print(f"Created directory: {plots_folder}")
    
    # Add timestamp to filename for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp}.png"
    filepath = os.path.join(plots_folder, full_filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved: {filepath}")

def read_and_plot_gpu_throughput(filename='paste.txt', use_time_axis=True, save_plot_flag=True):
    """
    Read GPU Xe link throughput data and plot curves for two tiles
    
    Args:
        filename: Path to the data file
        use_time_axis: If True, use time (seconds) as x-axis based on row positions (50ms per row);
                      if False, use row indices
        save_plot_flag: If True, save the plot as PNG file in plots folder
                      
    Note: Each row in the file represents a 50ms time interval, regardless of whether 
          the data in that row is valid (some rows may contain N/A values)
    """
    # Read CSV file
    df = pd.read_csv(filename)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Check data structure
    print("Data shape:", df.shape)
    print("Column names:", df.columns.tolist())
    print("Unique TileId values:", df['TileId'].unique())
    
    # Get all XL throughput columns (columns ending with kB/s)
    xl_columns = [col for col in df.columns if 'kB/s' in col]
    print(f"Found {len(xl_columns)} XL throughput columns")
    
    # Handle "N/A" values, replace with 0
    for col in xl_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Extract data for each tile separately
    tile0_data = df[df['TileId'] == 0].copy()
    tile1_data = df[df['TileId'] == 1].copy()
    
    # Calculate total throughput for each tile (sum of all XL links)
    tile0_data['total_throughput'] = tile0_data[xl_columns].sum(axis=1)
    tile1_data['total_throughput'] = tile1_data[xl_columns].sum(axis=1)
    
    # Create x-axis data based on row positions (every row represents 50ms interval)
    if use_time_axis:
        # Create time axis based on row index (50ms interval between any two rows)
        sample_interval_ms = 50  # 50ms between any two rows
        tile0_x_data = [tile0_data.index[i] * sample_interval_ms / 1000.0 for i in range(len(tile0_data))]
        tile1_x_data = [tile1_data.index[i] * sample_interval_ms / 1000.0 for i in range(len(tile1_data))]
        x_label = 'Time (seconds)'
        title_suffix = ' (50ms per row interval)'
    else:
        # Use original row indices from the file
        tile0_x_data = tile0_data.index.tolist()
        tile1_x_data = tile1_data.index.tolist()
        x_label = 'Row Index'
        title_suffix = ''
    
    # Convert throughput units from kB/s to GB/s
    tile0_throughput_gb = tile0_data['total_throughput'] / 1024 / 1024 /2 # kB/s -> GB/s
    tile1_throughput_gb = tile1_data['total_throughput'] / 1024 / 1024  /2# kB/s -> GB/s
    
    # Plot charts
    plt.figure(figsize=(12, 8))
    
    # Draw two curves with appropriate x-axis
    plt.plot(tile0_x_data, tile0_throughput_gb, 'b-', linewidth=2, label='Tile 0', alpha=0.8)
    plt.plot(tile1_x_data, tile1_throughput_gb, 'r-', linewidth=2, label='Tile 1', alpha=0.8)
    
    # Set chart properties
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Throughput (GB/s)', fontsize=12)
    plt.title(f'GPU Xe Link Throughput Comparison - Tile 0 vs Tile 1{title_suffix}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set axes
    max_x = max(max(tile0_x_data) if tile0_x_data else 0, max(tile1_x_data) if tile1_x_data else 0)
    plt.xlim(0, max_x)
    plt.ylim(0, max(tile0_throughput_gb.max(), tile1_throughput_gb.max()) * 1.1)
    
    # Add statistical information
    tile0_avg = tile0_throughput_gb.mean()/2
    tile1_avg = tile1_throughput_gb.mean()/2
    tile0_max = tile0_throughput_gb.max()/2
    tile1_max = tile1_throughput_gb.max()/2
    
    # Display statistics on chart
    if use_time_axis:
        stats_text = f'Tile 0: Avg={tile0_avg:.2f} GB/s, Max={tile0_max:.2f} GB/s\n'
        stats_text += f'Tile 1: Avg={tile1_avg:.2f} GB/s, Max={tile1_max:.2f} GB/s\n'
        total_duration = max(tile0_x_data[-1] if tile0_x_data else 0, tile1_x_data[-1] if tile1_x_data else 0)
        stats_text += f'Test Duration: {total_duration:.2f}s'
    else:
        stats_text = f'Tile 0: Avg={tile0_avg:.2f} GB/s, Max={tile0_max:.2f} GB/s\n'
        stats_text += f'Tile 1: Avg={tile1_avg:.2f} GB/s, Max={tile1_max:.2f} GB/s'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot_flag:
        plot_name = "gpu_xe_link_throughput_comparison_2"
        if use_time_axis:
            plot_name += "_time_axis"
        else:
            plot_name += "_row_index"
        save_plot(plot_name)
    
    plt.show()
    
    # Print detailed statistics
    print("\n=== Statistics ===")
    if use_time_axis:
        print(f"File structure: 50ms interval between any two rows")
        print(f"Tile 0:")
        print(f"  Valid data points: {len(tile0_data)}")
        print(f"  Time range: {tile0_x_data[0]:.2f}s to {tile0_x_data[-1]:.2f}s")
        print(f"  Total duration: {tile0_x_data[-1] - tile0_x_data[0]:.2f} seconds")
        print(f"  Average throughput: {tile0_avg:.2f} GB/s")
        print(f"  Maximum throughput: {tile0_max:.2f} GB/s")
        print(f"  Minimum throughput: {tile0_throughput_gb.min():.2f} GB/s")
        
        print(f"\nTile 1:")
        print(f"  Valid data points: {len(tile1_data)}")
        print(f"  Time range: {tile1_x_data[0]:.2f}s to {tile1_x_data[-1]:.2f}s")
        print(f"  Total duration: {tile1_x_data[-1] - tile1_x_data[0]:.2f} seconds")
        print(f"  Average throughput: {tile1_avg:.2f} GB/s")
        print(f"  Maximum throughput: {tile1_max:.2f} GB/s")
        print(f"  Minimum throughput: {tile1_throughput_gb.min():.2f} GB/s")
    else:
        print(f"Tile 0:")
        print(f"  Valid data points: {len(tile0_data)}")
        print(f"  Row range: {tile0_x_data[0]} to {tile0_x_data[-1]}")
        print(f"  Average throughput: {tile0_avg:.2f} GB/s")
        print(f"  Maximum throughput: {tile0_max:.2f} GB/s")
        print(f"  Minimum throughput: {tile0_throughput_gb.min():.2f} GB/s")
        
        print(f"\nTile 1:")
        print(f"  Valid data points: {len(tile1_data)}")
        print(f"  Row range: {tile1_x_data[0]} to {tile1_x_data[-1]}")
        print(f"  Average throughput: {tile1_avg:.2f} GB/s")
        print(f"  Maximum throughput: {tile1_max:.2f} GB/s")
        print(f"  Minimum throughput: {tile1_throughput_gb.min():.2f} GB/s")
    
    return tile0_data, tile1_data

def plot_individual_links(filename='paste.txt', tile_id=0, use_time_axis=True, save_plot_flag=True):
    """
    Plot detailed throughput for individual XL links of a single tile
    
    Args:
        filename: Path to the data file
        tile_id: Tile ID to plot (0 or 1)
        use_time_axis: If True, use time (seconds) as x-axis based on row positions (50ms per row);
                      if False, use row indices
        save_plot_flag: If True, save the plot as PNG file in plots folder
                      
    Note: Each row in the file represents a 50ms time interval, regardless of whether 
          the data in that row is valid (some rows may contain N/A values)
    """
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    
    # Get data for specified tile
    tile_data = df[df['TileId'] == tile_id].copy()
    
    # Get XL link columns
    xl_columns = [col for col in df.columns if 'kB/s' in col]
    
    # Handle N/A values
    for col in xl_columns:
        tile_data[col] = pd.to_numeric(tile_data[col], errors='coerce').fillna(0)
    
    # Create x-axis data based on row positions (every row represents 50ms interval)
    if use_time_axis:
        sample_interval_ms = 50
        x_data = [tile_data.index[i] * sample_interval_ms / 1000.0 for i in range(len(tile_data))]
        x_label = 'Time (seconds)'
        title_suffix = ' (50ms per row interval)'
    else:
        x_data = tile_data.index.tolist()
        x_label = 'Row Index'
        title_suffix = ''
    
    # Plot throughput for each link
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(xl_columns[:8]):  # Show only first 8 links to avoid overly complex chart
        throughput_gb = tile_data[col] / 1024 / 1024 / 2  # Convert to GB/s
        plt.plot(x_data, throughput_gb, label=col.replace(' (kB/s)', ''), alpha=0.7)
    
    plt.xlabel(x_label)
    plt.ylabel('Throughput (GB/s)')
    plt.title(f'Individual XL Links Throughput - Tile {tile_id}{title_suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot_flag:
        plot_name = f"gpu_xe_individual_links_tile_{tile_id}"
        if use_time_axis:
            plot_name += "_time_axis"
        else:
            plot_name += "_row_index"
        save_plot(plot_name)
    
    plt.show()

# Main function call
if __name__ == "__main__":
    # Plot total throughput comparison for two tiles (using time axis based on row positions)
    # Note: Each row in the file represents 50ms interval, regardless of data validity
    # Plots will be automatically saved to 'plots' folder as PNG files
    tile0_data, tile1_data = read_and_plot_gpu_throughput('fsdp_smi_ipex_50_xe.txt', use_time_axis=True, save_plot_flag=True)
    
    # Alternative options:
    # Plot with row indices instead of time
    # tile0_data, tile1_data = read_and_plot_gpu_throughput('paste.txt', use_time_axis=False, save_plot_flag=True)
    
    # Plot without saving
    # tile0_data, tile1_data = read_and_plot_gpu_throughput('paste.txt', use_time_axis=True, save_plot_flag=False)
    
    # Optional: Plot detailed link information for individual tiles
    # plot_individual_links('paste.txt', tile_id=0, use_time_axis=True, save_plot_flag=True)
    # plot_individual_links('paste.txt', tile_id=1, use_time_axis=True, save_plot_flag=True)