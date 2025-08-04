import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import argparse

warnings.filterwarnings('ignore')

def device_count_vs_power_contribution(power_df, save_path=None):
    
    """
    Device count vs total power contribution
    """
    power_df['node_type'] = power_df['node'].str.extract(r'([a-zA-Z]+)')[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    type_stats = power_df.groupby('node_type').agg({
        'node': 'nunique',
        'power_watts': 'sum'
    }).round(2)
    
    type_stats.columns = ['device_count', 'total_power']
    type_stats = type_stats.reset_index()
    
    # Create a combined bar chart
    x = np.arange(len(type_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, type_stats['device_count'], width, 
                   label='Device Count', alpha=0.7, color='skyblue')
    
    # Create secondary y-axis for power
    ax_twin = ax.twinx()
    bars2 = ax_twin.bar(x + width/2, type_stats['total_power'], width,
                        label='Total Power (W)', alpha=0.7, color='lightcoral')
    
    ax.set_xlabel('Node Type')
    ax.set_ylabel('Device Count', color='blue')
    ax_twin.set_ylabel('Total Power (W)', color='red')
    ax.set_title('Device Count vs Total Power Contribution', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_stats['node_type'], rotation=45)
    
    # Add value labels
    for bar, val in zip(bars1, type_stats['device_count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(int(val)), ha='center', va='bottom', fontsize=10)
    
    for bar, val in zip(bars2, type_stats['total_power']):
        ax_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f'{val:.0f}W', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def detailed_energy_analysis_focused(power_df, energy_df, save_path=None):
    """
    Create focused energy analysis with only requested visualizations
    """
    # Prepare data
    df = energy_df.copy()
    df['node_type'] = df['node'].str.extract(r'([a-zA-Z]+)')[0]
    power_df['node_type'] = power_df['node'].str.extract(r'([a-zA-Z]+)')[0]
    
    # Create figure with specific layout
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Energy consumption heatmap by top 20 nodes over time
    plt.subplot(3, 3, 1)
    sample_df = df[::5].copy()  # Sample every 5th for readability
    pivot_data = sample_df.pivot_table(values='energy_wh', index='node', columns='timestamp', fill_value=0)
    high_energy_nodes = df.groupby('node')['energy_wh'].sum().sort_values(ascending=False).head(20).index
    valid_nodes = [node for node in high_energy_nodes if node in pivot_data.index]
    if len(valid_nodes) == 0:
        print("⚠️ No valid high energy nodes found in pivot_data index for heatmap.")
        # Optionally, skip plotting or plot an empty heatmap
        pivot_subset = pd.DataFrame()
    else:
        pivot_subset = pivot_data.loc[valid_nodes]
    
    sns.heatmap(pivot_subset, cmap='YlOrRd', cbar_kws={'label': 'Energy (Wh)'})
    plt.title('Energy Consumption Heatmap (Top 20 Nodes)', fontsize=12, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Node')
    plt.xticks([])
    
    # 2. Energy consumption distribution by node type
    plt.subplot(3, 3, 2)
    node_types_for_plot = df[df['energy_wh'] > 0]['node_type'].unique()
    data_for_box = [df[(df['node_type'] == nt) & (df['energy_wh'] > 0)]['energy_wh'] for nt in node_types_for_plot]
    
    bp = plt.boxplot(data_for_box, labels=node_types_for_plot, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(node_types_for_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.title('Energy Consumption Distribution by Node Type', fontsize=12, fontweight='bold')
    plt.xlabel('Node Type')
    plt.ylabel('Energy (Wh)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 3. Cumulative energy consumption over time
    plt.subplot(3, 3, 3)
    df_sorted = df.sort_values('timestamp')
    
    for node_type in df['node_type'].unique():
        if node_type and node_type != 'registry':
            subset = df_sorted[df_sorted['node_type'] == node_type]
            if len(subset) > 0:
                subset_grouped = subset.groupby('timestamp')['energy_wh'].sum().reset_index()
                subset_grouped['cumulative_energy'] = subset_grouped['energy_wh'].cumsum()
                plt.plot(subset_grouped['timestamp'], subset_grouped['cumulative_energy'], 
                        label=node_type, linewidth=2, marker='o', markersize=3)
    
    plt.title('Cumulative Energy Consumption Over Time', fontsize=12, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Energy (Wh)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 4. Average power consumption by node type
    plt.subplot(3, 3, 4)
    avg_power_by_type = power_df.groupby('node_type')['power_watts'].mean().sort_values(ascending=False)
    
    bars = plt.bar(avg_power_by_type.index, avg_power_by_type.values, 
                  color=plt.cm.Set3(np.linspace(0, 1, len(avg_power_by_type))))
    plt.title('Average Power Consumption by Node Type', fontsize=12, fontweight='bold')
    plt.xlabel('Node Type')
    plt.ylabel('Average Power (Watts)')
    plt.xticks(rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}W', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 5. Average energy per node by type
    plt.subplot(3, 3, 5)
    energy_per_node = df.groupby('node_type').agg({
        'energy_wh': 'sum',
        'node': 'nunique'
    }).reset_index()
    energy_per_node['energy_per_node'] = energy_per_node['energy_wh'] / energy_per_node['node']
    energy_per_node = energy_per_node.sort_values('energy_per_node', ascending=False)
    
    bars = plt.bar(energy_per_node['node_type'], energy_per_node['energy_per_node'],
                  color=plt.cm.viridis(np.linspace(0, 1, len(energy_per_node))))
    plt.title('Average Energy per Node by Type', fontsize=12, fontweight='bold')
    plt.xlabel('Node Type')
    plt.ylabel('Energy per Node (Wh)')
    plt.xticks(rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # 6. Energy consumption variability
    plt.subplot(3, 3, 6)
    energy_variance = df.groupby('node_type')['energy_wh'].agg(['mean', 'std']).reset_index()
    energy_variance['cv'] = energy_variance['std'] / energy_variance['mean']
    energy_variance = energy_variance.sort_values('cv', ascending=True)
    
    bars = plt.bar(energy_variance['node_type'], energy_variance['cv'],
                  color=plt.cm.RdYlBu_r(np.linspace(0, 1, len(energy_variance))))
    plt.title('Energy Consumption Variability\n(Lower = More Stable)', fontsize=12, fontweight='bold')
    plt.xlabel('Node Type')
    plt.ylabel('Coefficient of Variation')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, energy_variance['cv']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



def energy_consumption_dashboard_focused(power_df, energy_df, save_path=None):
    """
    Focused energy consumption dashboard with only requested components
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Add node type column
    power_df['node_type'] = power_df['node'].str.extract(r'([a-zA-Z]+)')[0]
    energy_df['node_type'] = energy_df['node'].str.extract(r'([a-zA-Z]+)')[0]
    
    # 1. Total Energy Consumption by Node Type
    ax1 = fig.add_subplot(gs[0, 0:2])
    energy_by_type = energy_df.groupby('node_type')['energy_joules'].sum().sort_values(ascending=True)
    bars = ax1.barh(energy_by_type.index, energy_by_type.values, 
                   color=plt.cm.Set3(np.linspace(0, 1, len(energy_by_type))))
    ax1.set_title('Total Energy Consumption by Node Type', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Energy Consumption (Joules)')
    
    for bar, val in zip(bars, energy_by_type.values):
        ax1.text(val + max(energy_by_type.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}J', ha='left', va='center', fontsize=10)
    
    # 3. Energy Consumption Trends
    ax3 = fig.add_subplot(gs[1, 0:2])
    if 'timestamp' in energy_df.columns:
        time_energy = energy_df.groupby(['timestamp', 'node_type'])['energy_joules'].sum().reset_index()
        
        for node_type in time_energy['node_type'].unique():
            subset = time_energy[time_energy['node_type'] == node_type]
            ax3.plot(subset['timestamp'], subset['energy_joules'], marker='o', markersize=4, 
                    label=node_type, linewidth=2, alpha=0.8)
        
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Energy Consumption (J)')
        ax3.set_title('Energy Consumption Trends')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Power Consumption Distribution by Node Type
    ax4 = fig.add_subplot(gs[1, 2:])
    node_types = power_df['node_type'].unique()
    power_data = [power_df[power_df['node_type'] == nt]['power_watts'].values for nt in node_types]
    
    violin_parts = ax4.violinplot(power_data, positions=range(len(node_types)), showmeans=True)
    ax4.set_xticks(range(len(node_types)))
    ax4.set_xticklabels(node_types, rotation=45)
    ax4.set_ylabel('Power Consumption (W)')
    ax4.set_title('Power Consumption Distribution by Node Type')
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy Efficiency Heatmap (Wh by node type and time)
    ax5 = fig.add_subplot(gs[2, 0:2])
    if 'timestamp' in energy_df.columns:
        efficiency_matrix = energy_df.pivot_table(
            values='energy_wh', 
            index='node_type', 
            columns='timestamp', 
            aggfunc='mean',
            fill_value=0
        )
        
        # Limit columns for readability
        if efficiency_matrix.shape[1] > 10:
            step = efficiency_matrix.shape[1] // 10
            efficiency_matrix = efficiency_matrix.iloc[:, ::step]
        
        sns.heatmap(efficiency_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5, cbar=True)
        ax5.set_title('Energy Efficiency Heatmap\n(Wh by Node Type & Time)')
        ax5.set_xlabel('Timestamp')
        ax5.set_ylabel('Node Type')
    
    # 6. Top 10 Energy Consumers by Type
    ax6 = fig.add_subplot(gs[2, 2:])
    top_consumers_by_type = energy_df.groupby('node_type')['energy_joules'].sum().nlargest(10)
    
    if len(top_consumers_by_type) > 0:
        y_pos = np.arange(len(top_consumers_by_type))
        bars = ax6.barh(y_pos, top_consumers_by_type.values, color='lightcoral')
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(top_consumers_by_type.index, fontsize=10)
        ax6.set_xlabel('Energy (J)')
        ax6.set_title('Top 10 Energy Consumers by Type')
        
        for bar, val in zip(bars, top_consumers_by_type.values):
            ax6.text(val + max(top_consumers_by_type.values) * 0.01, 
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', ha='left', va='center', fontsize=9)
    
    plt.suptitle('Energy Consumption Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


def generate_detailed_text_summary(power_df, energy_df, output_path=None):
    """
    Generate detailed text summary similar to the FAAS simulation format
    """
    # Prepare data
    power_df['node_type'] = power_df['node'].str.extract(r'([a-zA-Z]+)')[0]
    energy_df['node_type'] = energy_df['node'].str.extract(r'([a-zA-Z]+)')[0]
    
    # Calculate comprehensive statistics
    total_energy_j = energy_df['energy_joules'].sum()
    total_energy_wh = energy_df['energy_wh'].sum()
    total_nodes = power_df['node'].nunique()
    total_node_types = power_df['node_type'].nunique()
    
    avg_power = power_df['power_watts'].mean()
    peak_power = power_df['power_watts'].max()
    min_power = power_df['power_watts'].min()
    
    # Node distribution
    node_energy_dist = energy_df.groupby('node_type')['energy_joules'].sum().sort_values(ascending=False)
    total_energy_for_pct = node_energy_dist.sum()
    
    # Power statistics by type
    power_stats = power_df.groupby('node_type').agg({
        'power_watts': ['mean', 'min', 'max', 'std'],
        'node': 'nunique'
    }).round(3)
    
    # Energy statistics by type
    energy_stats = energy_df.groupby('node_type').agg({
        'energy_joules': ['sum', 'mean', 'min', 'max', 'std'],
        'energy_wh': ['sum', 'mean'],
        'node': 'nunique'
    }).round(3)
    
    # Time-based statistics
    if 'timestamp' in power_df.columns:
        duration = len(power_df['timestamp'].unique())
        time_range = f"{power_df['timestamp'].min()} to {power_df['timestamp'].max()}"
    else:
        duration = len(power_df)
        time_range = f"Index 0 to {len(power_df)-1}"
    
    # Generate report
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""FAAS ENERGY CONSUMPTION ANALYSIS REPORT
========================================

Report generated: {report_time}

System Overview:
  Total nodes: {total_nodes}
  Node types: {total_node_types}
  Monitoring duration: {duration} time periods
  Time range: {time_range}

Energy Consumption Summary:
  Total energy consumed: {total_energy_j:,.2f} Joules ({total_energy_wh:.2f} Wh)
  Average energy per time period: {total_energy_j/duration:.2f} J
  Total energy in kWh: {total_energy_wh/1000:.4f} kWh

Power Consumption Summary:
  Average system power: {avg_power:.2f} W
  Peak system power: {peak_power:.2f} W
  Minimum system power: {min_power:.2f} W
  Power range: {peak_power - min_power:.2f} W

Node Type Energy Distribution:
"""

    for node_type, energy in node_energy_dist.items():
        percentage = (energy / total_energy_for_pct) * 100
        summary += f"  {node_type}: {energy:,.2f} J ({percentage:.1f}%)\n"

    summary += f"""
Node Type Power Statistics:
"""
    
    for node_type in power_stats.index:
        node_count = power_stats.loc[node_type, ('node', 'nunique')]
        avg_power_type = power_stats.loc[node_type, ('power_watts', 'mean')]
        max_power_type = power_stats.loc[node_type, ('power_watts', 'max')]
        min_power_type = power_stats.loc[node_type, ('power_watts', 'min')]
        std_power_type = power_stats.loc[node_type, ('power_watts', 'std')]
        
        summary += f"  {node_type} ({node_count} nodes):\n"
        summary += f"    Average power: {avg_power_type:.2f} W\n"
        summary += f"    Power range: {min_power_type:.2f} - {max_power_type:.2f} W\n"
        summary += f"    Power variability (std): {std_power_type:.3f} W\n"

    summary += f"""
Node Type Energy Statistics:
"""
    
    for node_type in energy_stats.index:
        total_energy_type = energy_stats.loc[node_type, ('energy_joules', 'sum')]
        avg_energy_type = energy_stats.loc[node_type, ('energy_joules', 'mean')]
        total_energy_wh_type = energy_stats.loc[node_type, ('energy_wh', 'sum')]
        node_count = energy_stats.loc[node_type, ('node', 'nunique')]
        energy_per_node = total_energy_type / node_count
        
        summary += f"  {node_type}:\n"
        summary += f"    Total energy: {total_energy_type:,.2f} J ({total_energy_wh_type:.2f} Wh)\n"
        summary += f"    Average energy per measurement: {avg_energy_type:.3f} J\n"
        summary += f"    Energy per node: {energy_per_node:,.2f} J\n"

    # Efficiency analysis
    most_efficient = node_energy_dist.index[-1]  # Lowest energy consumer
    highest_consumer = node_energy_dist.index[0]  # Highest energy consumer
    
    # Power model assessment
    power_variance = power_df['power_watts'].var()
    if power_variance < 1.0:
        power_model = "STATIC (Constant power consumption)"
    else:
        power_model = "DYNAMIC (Variable power consumption)"

    summary += f"""
Efficiency Analysis:
  Most energy efficient node type: {most_efficient}
  Highest energy consuming node type: {highest_consumer}
  Energy efficiency ratio: {node_energy_dist[highest_consumer]/node_energy_dist[most_efficient]:.2f}x

Power Model Assessment:
  Power model type: {power_model}
  System power variance: {power_variance:.3f} W²
  CPU-Power correlation: {power_df['cpu_util'].corr(power_df['power_watts']):.3f}

Environmental Impact (Estimates):
  CO2 emissions (avg grid): {total_energy_wh * 0.000233:.4f} kg CO2
  Cost estimate (@$0.12/kWh): ${total_energy_wh/1000 * 0.12:.4f}

Recommendations:
"""

    if power_variance < 1.0:
        summary += """  - Implement dynamic power scaling based on CPU utilization
  - Add DVFS (Dynamic Voltage and Frequency Scaling) modeling
  - Consider workload-dependent power consumption patterns
"""
    else:
        summary += """  - Current power model shows good dynamic behavior
  - Consider adding non-linear power curves for more accuracy
  - Monitor for power anomalies and optimization opportunities
"""

    # Top consumers detail
    summary += f"""
Top Energy Consuming Nodes:
"""
    top_nodes = energy_df.groupby('node')['energy_joules'].sum().nlargest(10)
    for i, (node, energy) in enumerate(top_nodes.items(), 1):
        node_type = energy_df[energy_df['node'] == node]['node_type'].iloc[0]
        summary += f"  {i:2d}. {node} ({node_type}): {energy:,.2f} J\n"

    summary += f"""
Analysis Summary:
  Report covers {duration} measurement periods across {total_nodes} nodes
  Total system energy consumption: {total_energy_j:,.0f} J ({total_energy_wh:.2f} Wh)
  System shows {power_model.lower()} power consumption patterns
  Energy distribution varies {node_energy_dist.max()/node_energy_dist.min():.1f}x between node types

End of Report
=============
"""

    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"Detailed summary saved to {output_path}")
    
    # Also print to console
    print(summary)
    
    return summary

def generate_focused_visualizations(power_df, energy_df, output_dir='./visualizations/'):
    """
    Generate only the requested visualizations
    """
    import os
    output_dir = os.path.abspath(output_dir)  # Convert to absolute path
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Device Count vs Total Power Contribution...")
    device_count_vs_power_contribution(power_df, f"{output_dir}/device_count_vs_power.png")
    
    print("Generating Detailed Energy Analysis (Focused)...")
    detailed_energy_analysis_focused(power_df, energy_df, f"{output_dir}/detailed_energy_analysis_focused.png")
    
    print("Generating Energy Dashboard (Focused)...")
    energy_consumption_dashboard_focused(power_df, energy_df, f"{output_dir}/energy_dashboard_focused.png")
    
    print("Generating Detailed Text Summary...")
    summary = generate_detailed_text_summary(power_df, energy_df, f"{output_dir}/energy_analysis_summary.txt")
    
    print(f"All focused visualizations and summary saved to {output_dir}")
    
    return summary

def main(power_df=None, energy_df=None, output_dir="./visualizations/", data_path=None):
    output_dir = os.path.abspath(output_dir)
    if power_df is None or energy_df is None:
        if data_path is None:
            raise ValueError("data_path must be provided if dataframes are not passed directly.")
        power_df_path = os.path.join(data_path, "power_df.csv")
        energy_df_path = os.path.join(data_path, "energy_df.csv")
        print(f"Loading power data from {power_df_path}")
        power_df = pd.read_csv(power_df_path)
        print(f"Loading energy data from {energy_df_path}")
        energy_df = pd.read_csv(energy_df_path)
    summary = generate_focused_visualizations(power_df, energy_df, output_dir)
    print("Focused visualization generation complete!")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FaaS energy analysis report and visualizations.")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to directory containing power_df.csv and energy_df.csv")
    parser.add_argument("--output-dir", type=str, default="./visualizations/",
                        help="Directory to save output visualizations and report")
    args = parser.parse_args()
    main(output_dir=args.output_dir, data_path=args.data_path)