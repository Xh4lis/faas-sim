import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def create_strategy_comparison_table():
    """Create comprehensive comparison table from experimental results"""
    
    # Data extracted from your experiment files
    strategies_data = {
        'Strategy': ['Kubernetes', 'HPST', 'LPLT'],
        
        # Power Metrics
        'Infrastructure Power (W)': [519.4, 435.1, 544.8],
        'Workload Power (W)': [6577.5, 8885.9, 2649.9],
        'Power per Replica (W)': [3.22, 3.64, 2.99],
        'Power per Node (W)': [4.36, 3.66, 4.58],
        
        # Performance Metrics
        'Median Response Time (s)': [1121.010, 1052.111, 1180.832],
        '95th Percentile (s)': [4485.887, 4909.099, 4909.099],
        'Warm-up Avg (s)': [10310.931, 4262.024, 11809.096],
        
        # Scaling Behavior
        'Total Replicas': [2040, 2443, 885],
        'Unique Nodes Used': [39, 119, 46],
        'Avg Replicas/Node': [52.3, 20.5, 19.2],
        'Max Replicas/Node': [228, 66, 193],
        
        # Scaling Actions
        'Scale Up Actions': [401, 558, 175],
        'Scale Down Actions': [1064, 311, 482],
        'Total Scaling Actions': [1465, 869, 657],
        'Scheduling Events': [1254, 2445, 576],
        
        # Efficiency Ratios
        'Consolidation Factor': [2.72, 1.07, 1.0],  # vs baseline
        'Scheduling Success Rate': [1.0, 0.706, 1.0],
    }
    
    # Create DataFrame
    df = pd.DataFrame(strategies_data)
    
    # Calculate relative performance (Kubernetes as baseline)
    baseline_idx = 0  # Kubernetes
    
    # Add percentage comparisons
    comparison_data = {
        'Strategy': ['K8S vs HPST', 'K8S vs LPLT', 'HPST vs LPLT'],
        
        # Power Comparisons (negative = savings)
        'Infrastructure Power Δ (%)': [
            ((435.1 - 519.4) / 519.4) * 100,  # K8S vs HPST: -16.2%
            ((544.8 - 519.4) / 519.4) * 100,  # K8S vs LPLT: +4.9%
            ((435.1 - 544.8) / 544.8) * 100   # HPST vs LPLT: -20.1%
        ],
        
        'Workload Efficiency Δ (%)': [
            ((3.64 - 3.22) / 3.22) * 100,     # K8S vs HPST: +13.0%
            ((2.99 - 3.22) / 3.22) * 100,     # K8S vs LPLT: -7.1%
            ((3.64 - 2.99) / 2.99) * 100      # HPST vs LPLT: +21.7%
        ],
        
        # Performance Comparisons (negative = improvement)
        'Response Time Δ (%)': [
            ((1052.111 - 1121.010) / 1121.010) * 100,  # K8S vs HPST: -6.1%
            ((1180.832 - 1121.010) / 1121.010) * 100,  # K8S vs LPLT: +5.3%
            ((1052.111 - 1180.832) / 1180.832) * 100   # HPST vs LPLT: -10.9%
        ],
        
        'Cold Start Δ (%)': [
            ((4262.024 - 10310.931) / 10310.931) * 100,  # K8S vs HPST: -58.7%
            ((11809.096 - 10310.931) / 10310.931) * 100, # K8S vs LPLT: +14.5%
            ((4262.024 - 11809.096) / 11809.096) * 100   # HPST vs LPLT: -63.9%
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return df, comparison_df

def export_to_csv(main_df, comp_df, output_dir="output"):
    """Export tables to CSV files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Export main metrics
    main_df.to_csv(f"{output_dir}/strategy_metrics.csv", index=False)
    
    # Export comparisons
    comp_df.to_csv(f"{output_dir}/strategy_comparisons.csv", index=False)
    
    # Create rankings DataFrame
    rankings_data = {
        'Metric': [
            'Energy Efficiency (W/replica)',
            'Infrastructure Power',
            'Response Time',
            'Cold Start Performance',
            'Resource Utilization',
            'Scaling Stability'
        ],
        'Best': ['LPLT (2.99)', 'HPST (435.1W)', 'HPST (1052s)', 'HPST (4262s)', 'K8S (52.3 rep/node)', 'LPLT (657 actions)'],
        'Second': ['K8S (3.22)', 'K8S (519.4W)', 'K8S (1121s)', 'K8S (10311s)', 'LPLT (19.2)', 'HPST (869)'],
        'Third': ['HPST (3.64)', 'LPLT (544.8W)', 'LPLT (1181s)', 'LPLT (11809s)', 'HPST (20.5)', 'K8S (1465)']
    }
    rankings_df = pd.DataFrame(rankings_data)
    rankings_df.to_csv(f"{output_dir}/strategy_rankings.csv", index=False)
    
    print(f"✅ CSV files exported to '{output_dir}/' directory")

def export_to_pdf(main_df, comp_df, output_dir="output"):
    """Export tables to PDF file"""
    os.makedirs(output_dir, exist_ok=True)
    
    with PdfPages(f"{output_dir}/strategy_comparison_report.pdf") as pdf:
        # Page 1: Main Metrics
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')
        
        table1 = ax.table(cellText=main_df.values,
                         colLabels=main_df.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.1] + [0.15] * (len(main_df.columns) - 1))
        
        table1.auto_set_font_size(False)
        table1.set_fontsize(8)
        table1.scale(1, 2)
        
        # Style the header
        for i in range(len(main_df.columns)):
            table1[(0, i)].set_facecolor('#4472C4')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Autoscaling Strategy Comparison - Absolute Metrics', 
                 fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 2: Comparisons
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table2 = ax.table(cellText=comp_df.round(1).values,
                         colLabels=comp_df.columns,
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.2] + [0.2] * (len(comp_df.columns) - 1))
        
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1, 2)
        
        # Style the header
        for i in range(len(comp_df.columns)):
            table2[(0, i)].set_facecolor('#70AD47')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Strategy Performance Comparisons (%)', 
                 fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Page 3: Rankings Summary
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        rankings_text = """
🏆 STRATEGY RANKINGS SUMMARY

Energy Efficiency (W/replica):
  1. LPLT (2.99)    2. K8S (3.22)    3. HPST (3.64)

Infrastructure Power:
  1. HPST (435.1W)    2. K8S (519.4W)    3. LPLT (544.8W)

Response Time:
  1. HPST (1052s)    2. K8S (1121s)    3. LPLT (1181s)

Cold Start Performance:
  1. HPST (4262s)    2. K8S (10311s)    3. LPLT (11809s)

Resource Utilization:
  1. K8S (52.3 rep/node)    2. LPLT (19.2)    3. HPST (20.5)

Scaling Stability:
  1. LPLT (657 actions)    2. HPST (869)    3. K8S (1465)
        """
        
        ax.text(0.1, 0.9, rankings_text, fontsize=12, fontfamily='monospace',
                verticalalignment='top', transform=ax.transAxes)
        
        plt.title('Performance Rankings by Category', 
                 fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"✅ PDF report exported to '{output_dir}/strategy_comparison_report.pdf'")

def print_formatted_tables():
    """Print nicely formatted comparison tables"""
    
    main_df, comp_df = create_strategy_comparison_table()
    
    print("=" * 80)
    print("🔋 AUTOSCALING STRATEGY COMPARISON TABLE")
    print("=" * 80)
    
    # Main metrics table
    print("\n📊 ABSOLUTE METRICS:")
    print(main_df.to_string(index=False, float_format='%.2f'))
    
    print("\n📈 RELATIVE COMPARISONS:")
    print(comp_df.to_string(index=False, float_format='%.1f'))
    
    # Summary rankings
    print("\n🏆 STRATEGY RANKINGS:")
    
    rankings = {
        'Energy Efficiency (W/replica)': ['LPLT (2.99)', 'K8S (3.22)', 'HPST (3.64)'],
        'Infrastructure Power': ['HPST (435.1W)', 'K8S (519.4W)', 'LPLT (544.8W)'],
        'Response Time': ['HPST (1052s)', 'K8S (1121s)', 'LPLT (1181s)'],
        'Cold Start Performance': ['HPST (4262s)', 'K8S (10311s)', 'LPLT (11809s)'],
        'Resource Utilization': ['K8S (52.3 rep/node)', 'LPLT (19.2)', 'HPST (20.5)'],
        'Scaling Stability': ['LPLT (657 actions)', 'HPST (869)', 'K8S (1465)']
    }
    
    for metric, ranking in rankings.items():
        print(f"\n{metric}:")
        for i, strategy in enumerate(ranking, 1):
            print(f"  {i}. {strategy}")

def export_all_formats():
    """Export to both CSV and PDF formats"""
    main_df, comp_df = create_strategy_comparison_table()
    
    # Export to CSV
    export_to_csv(main_df, comp_df)
    
    # Export to PDF
    export_to_pdf(main_df, comp_df)
    
    print("\n🎉 All exports completed successfully!")

if __name__ == "__main__":
    print_formatted_tables()
    print("\n" + "="*50)
    export_all_formats()