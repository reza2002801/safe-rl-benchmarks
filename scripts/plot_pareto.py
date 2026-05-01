import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the academic plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

def plot_pareto_front():
    # Map your folder names to the display names
    algorithm_folders = {
        'ppo': 'PPO (Unconstrained)',
        'ppo_lagr': 'PPO-Lagrangian',
        'cpo': 'CPO'
    }

    final_results = []

    print("Extracting final performance data...")
    # Loop through folders and seeds
    for folder, algo_name in algorithm_folders.items():
        for seed in range(3):
            file_path = os.path.join(folder, f'progress{seed}.csv')
            
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # To get a stable final performance, we average the last 10 epochs
                # rather than just taking the very last single epoch which might be noisy.
                last_epochs = df.tail(10)
                mean_ret = last_epochs['Metrics/EpRet'].mean()
                mean_cost = last_epochs['Metrics/EpCost'].mean()
                
                final_results.append({
                    'Algorithm': algo_name,
                    'Seed': seed,
                    'Final Return': mean_ret,
                    'Final Cost': mean_cost
                })
            else:
                print(f"Warning: Could not find {file_path}")

    if not final_results:
        print("Error: No data loaded.")
        return

    # Convert to DataFrame
    results_df = pd.DataFrame(final_results)

    print("Generating Pareto Trade-off Plot...")
    plt.figure(figsize=(10, 8))
    
    # Create the scatter plot. Each dot is a single seed's final performance.
    sns.scatterplot(
        data=results_df, 
        x='Final Cost', 
        y='Final Return', 
        hue='Algorithm', 
        style='Algorithm',
        s=250,      # Make the markers large and visible
        alpha=0.8,  # Slight transparency
        edgecolor='black'
    )

    # Draw the strict safety limit line from your paper (d=25)
    plt.axvline(x=25, color='red', linestyle='--', linewidth=2.5, label='Safety Limit (d=25)')
    
    # Shade the "Safe Zone" (Cost from 0 to 25) in light green
    plt.axvspan(0, 25, color='green', alpha=0.1, label='Safe Zone')

    # Formatting the axes and title
    plt.title('Algorithm Trade-offs (Reward vs. Cost)')
    plt.xlabel('Average Episodic Cost (Further Left is Safer)')
    plt.ylabel('Average Episodic Return (Higher is Better)')
    
    # Move the legend outside the plot so it doesn't cover data points
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    
    # Save as high-res PDF for LaTeX
    output_pdf = 'pareto_tradeoff_plot.pdf'
    plt.savefig(output_pdf, format='png', bbox_inches='tight')
    print(f"Success! Pareto plot saved locally as: {output_pdf}")
    
    plt.show()

if __name__ == '__main__':
    plot_pareto_front()