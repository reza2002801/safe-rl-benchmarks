import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set up the plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

def plot_results():
    # Map your folder names to the display names for the legend
    algorithm_folders = {
        'ppo': 'PPO (Unconstrained)',
        'ppo_lagr': 'PPO-Lagrangian',
        'cpo': 'CPO'
    }

    data_frames = []

    print("Loading CSV files...")
    # 2. Loop through your specific folder structure
    for folder, algo_name in algorithm_folders.items():
        for seed in range(3):
            file_path = os.path.join(folder, f'progress{seed}.csv')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Algorithm'] = algo_name  # Tag the data with the algorithm name
                df['Seed'] = seed            # Tag the data with the seed
                data_frames.append(df)
            else:
                print(f"Warning: Could not find {file_path}")

    if not data_frames:
        print("Error: No data loaded. Check your folder paths.")
        return

    # Combine all data into one large table for Seaborn
    combined_data = pd.concat(data_frames, ignore_index=True)

    print("Generating plots with Standard Deviation shading...")
    # 3. Create a side-by-side figure layout
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Reward vs Epoch ---
    # errorbar='sd' tells Seaborn to plot the Mean as the line and Standard Deviation as the shadow
    sns.lineplot(
        data=combined_data, 
        x='Train/Epoch', 
        y='Metrics/EpRet', 
        hue='Algorithm', 
        errorbar='sd', 
        linewidth=2.5,
        ax=axes[0]
    )
    axes[0].set_title('Task Performance (Reward)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Episodic Return')

    # --- Plot 2: Cost vs Epoch ---
    sns.lineplot(
        data=combined_data, 
        x='Train/Epoch', 
        y='Metrics/EpCost', 
        hue='Algorithm', 
        errorbar='sd', 
        linewidth=2.5,
        ax=axes[1]
    )
    # Add the strict safety limit line (d=25) for visual reference
    axes[1].axhline(y=25, color='red', linestyle='--', linewidth=2, label='Safety Limit (d=25)')
    axes[1].set_title('Safety Performance (Cost)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average Episodic Cost')
    
    # Clean up the legends
    axes[0].legend(loc='lower right')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    
    # 4. Save and show
    output_pdf = 'algorithm_comparison_results.png'
    plt.savefig(output_pdf, format='png', bbox_inches='tight')
    print(f"Success! Plot saved locally as: {output_pdf}")
    
    plt.show()

if __name__ == '__main__':
    plot_results()