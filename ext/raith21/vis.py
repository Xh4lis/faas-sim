import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

class MetricsVisualizer:
    def __init__(self, data_dir='analysis_data', output_dir='visualization_resultsv2'):
        self.data_dir = data_dir
        self.output_dir = self._ensure_dir(output_dir)
        self.dfs = {}
        self.nodes = []
        self.functions = []
        self.images = []
    
    def _ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def load_data(self):
        """Load all CSV files and extract metadata"""
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                name = file.replace('.csv', '')
                self.dfs[name] = pd.read_csv(os.path.join(self.data_dir, file))
                
        # Extract metadata
        if 'invocations_df' in self.dfs:
            inv_df = self.dfs['invocations_df']
            self.nodes = inv_df['node'].unique()
            self.functions = inv_df['function_name'].unique()
            self.images = inv_df['function_image'].unique()
            
        return len(self.dfs) > 0

    def generate_reports(self):
        """Generate all visualization reports"""
        if not self.dfs:
            print("No data loaded. Please load data first.")
            return
        
        # Generate visualizations
        self.generate_deployment_timeline()
        self.generate_function_comparison_report()
        self.generate_function_invocation_report()
        self.generate_function_utilization_report()
        self.generate_image_comparison()
        
        # Generate summaries
        self.generate_invocation_summary()
        self.generate_overall_summary()
        self.generate_utilization_summary()

    def generate_deployment_timeline(self):
        """Generate deployment timeline visualization"""
        if 'function_deployment_lifecycle_df' not in self.dfs:
            return
                
        lifecycle_df = self.dfs['function_deployment_lifecycle_df']
        deploy_df = self.dfs['replica_deployment_df']
        
        # Get all unique function names from both dataframes
        all_functions = set()
        if 'name' in lifecycle_df.columns:
            all_functions.update(lifecycle_df['name'].unique())
        if 'function_name' in deploy_df.columns:
            all_functions.update(deploy_df['function_name'].unique())
        
        plt.figure(figsize=(15, 8))
        
        # Plot function lifecycle events
        y_positions = {}
        for i, func in enumerate(sorted(all_functions)):
            y_positions[func] = i * 2
            
            # Plot lifecycle events
            if 'name' in lifecycle_df.columns:
                func_events = lifecycle_df[lifecycle_df['name'] == func]
                plt.plot([0], [y_positions[func]], 'o', label=func)
                
                for _, event in func_events.iterrows():
                    plt.annotate(event['value'], 
                               xy=(event['function_id'], y_positions[func]),
                               xytext=(0, 10), textcoords='offset points',
                               rotation=45)
        
        # Plot replica deployments
        if 'replica_deployment_df' in self.dfs:
            for _, deploy in deploy_df.iterrows():
                func_name = deploy['function_name']
                if func_name in y_positions:
                    y = y_positions[func_name]
                    plt.scatter(deploy['replica_id'], y, marker='s', 
                              c='green' if deploy['value'] == 'deployed' else 'red')
                else:
                    print(f"Warning: Function {func_name} found in deployments but not in lifecycle events")
        
        plt.yticks(list(y_positions.values()), list(y_positions.keys()))
        plt.xlabel('Timeline')
        plt.title('Function Deployment Timeline')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'deployment-timeline.png'))
        plt.close()

    def generate_function_comparison_report(self):
        """Generate function comparison visualization"""
        if 'invocations_df' not in self.dfs:
            return
            
        inv_df = self.dfs['invocations_df']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time comparison
        sns.boxplot(data=inv_df, x='function_name', y='t_exec', ax=axes[0,0])
        axes[0,0].set_title('Execution Time by Function')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        
        # Wait time comparison
        sns.boxplot(data=inv_df, x='function_name', y='t_wait', ax=axes[0,1])
        axes[0,1].set_title('Wait Time by Function')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        
        # Invocation count
        inv_counts = inv_df['function_name'].value_counts()
        axes[1,0].bar(inv_counts.index, inv_counts.values)
        axes[1,0].set_title('Number of Invocations per Function')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # Memory usage
        sns.boxplot(data=inv_df, x='function_name', y='memory', ax=axes[1,1])
        axes[1,1].set_title('Memory Usage by Function')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'function_comparison_report.png'))
        plt.close()

    def generate_function_invocation_report(self):
        """Generate function invocation visualization"""
        if 'invocations_df' not in self.dfs or 'fets_df' not in self.dfs:
            return
            
        inv_df = self.dfs['invocations_df']
        fets_df = self.dfs['fets_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Execution time distribution
        sns.histplot(data=inv_df, x='t_exec', hue='function_name', 
                    multiple="stack", ax=axes[0,0])
        axes[0,0].set_title('Execution Time Distribution')
        
        # FET timeline 
        for func in self.functions:
            func_df = fets_df[fets_df['function_name'] == func]
            duration = func_df['t_fet_end'] - func_df['t_fet_start']
            axes[0,1].scatter(func_df['t_fet_start'], duration,
                            label=func, alpha=0.5)
        axes[0,1].set_title('Function Execution Timeline')
        axes[0,1].set_xlabel('Start Time (s)')
        axes[0,1].set_ylabel('Duration (s)')
        if len(self.functions) <= 10:  # Only show legend if not too many functions
            axes[0,1].legend()
        
        # Wait time analysis
        wait_times = fets_df['t_wait_end'] - fets_df['t_wait_start']
        sns.boxplot(data=fets_df, x='function_name', y=wait_times, ax=axes[1,0])
        axes[1,0].set_title('Wait Time Distribution by Function')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # Cumulative invocations over time
        for func in self.functions:
            func_df = fets_df[fets_df['function_name'] == func]
            sorted_times = np.sort(func_df['t_fet_start'])
            axes[1,1].step(sorted_times, np.arange(len(sorted_times)), 
                          label=func, where='post')
        axes[1,1].set_title('Cumulative Invocations')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Number of Invocations')
        if len(self.functions) <= 10:
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'function_invocation_report.png'))
        plt.close()

    def generate_function_utilization_report(self):
        """Generate function utilization visualization"""
        if 'fets_df' not in self.dfs:
            return
            
        fets_df = self.dfs['fets_df']
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Calculate utilization
        for node in self.nodes:
            node_df = fets_df[fets_df['node'] == node]
            times = np.sort(np.concatenate([node_df['t_fet_start'], node_df['t_fet_end']]))
            util = np.zeros_like(times)
            
            for _, row in node_df.iterrows():
                start_idx = np.searchsorted(times, row['t_fet_start'])
                end_idx = np.searchsorted(times, row['t_fet_end'])
                util[start_idx:end_idx] += 1
                
            axes[0].plot(times, util, label=node)
            
        axes[0].set_title('Node Utilization Over Time')
        axes[0].legend()
        
        # Function utilization distribution
        for func in self.functions:
            func_df = fets_df[fets_df['function_name'] == func]
            durations = func_df['t_fet_end'] - func_df['t_fet_start']
            sns.kdeplot(data=durations, ax=axes[1], label=func)
            
        axes[1].set_title('Function Duration Distribution')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'function_utilization_report.png'))
        plt.close()

    def generate_image_comparison(self):
        """Generate image comparison visualization"""
        if 'invocations_df' not in self.dfs:
            return
            
        inv_df = self.dfs['invocations_df']
        
        plt.figure(figsize=(12, 8))
        
        # Compare metrics across images
        metrics = {
            't_exec': 'Execution Time (ms)',
            't_wait': 'Wait Time (ms)',
            'memory': 'Memory Usage'
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        plot_positions = [(0,0), (0,1), (1,0)]
        
        for (metric, label), pos in zip(metrics.items(), plot_positions):
            sns.boxplot(data=inv_df, x='function_image', y=metric, 
                       ax=axes[pos[0], pos[1]])
            axes[pos[0], pos[1]].set_title(f'{label} by Image')
            axes[pos[0], pos[1]].set_xticklabels(
                axes[pos[0], pos[1]].get_xticklabels(), rotation=45)
        
        # Add invocation counts
        inv_counts = inv_df['function_image'].value_counts()
        axes[1,1].bar(inv_counts.index, inv_counts.values)
        axes[1,1].set_title('Invocations per Image')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'image_comparison.png'))
        plt.close()

    # Additional methods for generating text summaries
    def generate_invocation_summary(self):
        """Generate detailed invocation summary"""
        if 'invocations_df' not in self.dfs:
            return
            
        inv_df = self.dfs['invocations_df']
        summary = []
        
        # Overall statistics
        summary.extend([
            "FUNCTION INVOCATION SUMMARY",
            "========================",
            f"Total invocations: {len(inv_df)}",
            f"Total functions: {len(self.functions)}",
            f"Total nodes: {len(self.nodes)}",
            f"Simulation duration: {inv_df['t_start'].max() - inv_df['t_start'].min():.2f} seconds",
            "\nGLOBAL STATISTICS",
            "----------------",
            f"Mean execution time: {inv_df['t_exec'].mean():.2f} ms",
            f"Mean wait time: {inv_df['t_wait'].mean():.2f} ms",
            f"Mean memory usage: {inv_df['memory'].mean():.2f} units",
            "\nPER-FUNCTION STATISTICS",
            "---------------------"
        ])
        
        for func in self.functions:
            func_df = inv_df[inv_df['function_name'] == func]
            summary.extend([
                f"\n{func}:",
                f"  Invocations: {len(func_df)}",
                f"  Execution time (mean/min/max): {func_df['t_exec'].mean():.2f}/{func_df['t_exec'].min():.2f}/{func_df['t_exec'].max():.2f} ms",
                f"  Wait time (mean/min/max): {func_df['t_wait'].mean():.2f}/{func_df['t_wait'].min():.2f}/{func_df['t_wait'].max():.2f} ms",
                f"  Memory usage (mean): {func_df['memory'].mean():.2f} units"
            ])
            
        with open(os.path.join(self.output_dir, 'invocation_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))

    def generate_overall_summary(self):
        """Generate comprehensive overall summary"""
        summary = [
            "FAAS SIMULATION OVERALL SUMMARY",
            "============================",
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\nSYSTEM CONFIGURATION",
            "-------------------",
            f"Number of nodes: {len(self.nodes)}",
            f"Number of functions: {len(self.functions)}",
            f"Number of images: {len(self.images)}"
        ]
        
        # Add deployment information
        if 'function_deployment_df' in self.dfs:
            deploy_df = self.dfs['function_deployment_df']
            summary.extend([
                "\nDEPLOYMENT STATISTICS",
                "--------------------",
                f"Total deployments: {len(deploy_df)}",
                f"Deployment distribution:"
            ])
            for node in self.nodes:
                node_deployments = deploy_df[deploy_df['node'] == node]
                summary.append(f"  {node}: {len(node_deployments)} deployments")
        
        # Add network statistics
        if 'flow_df' in self.dfs:
            flow_df = self.dfs['flow_df']
            summary.extend([
                "\nNETWORK STATISTICS",
                "-----------------",
                f"Total transfers: {len(flow_df)}",
                f"Total data transferred: {flow_df['bytes'].sum()/1024/1024:.2f} MB",
                f"Average transfer duration: {flow_df['duration'].mean():.2f} seconds"
            ])
        
        # Save both text and image versions
        with open(os.path.join(self.output_dir, 'overall_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))
            
        # Create visual summary
        plt.figure(figsize=(12, 8))
        plt.text(0.1, 0.1, '\n'.join(summary), fontsize=10, 
                family='monospace', verticalalignment='top')
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, 'overall_summary.png'))
        plt.close()

    def generate_utilization_summary(self):
        """Generate utilization summary from FETs data"""
        if 'fets_df' not in self.dfs:
            return
            
        fets_df = self.dfs['fets_df']
        summary = [
            "RESOURCE UTILIZATION SUMMARY",
            "=========================",
            "\nOVERALL STATISTICS",
            "-----------------"
        ]
        
        # Calculate global utilization metrics
        total_duration = fets_df['t_fet_end'].max() - fets_df['t_fet_start'].min()
        total_compute_time = (fets_df['t_fet_end'] - fets_df['t_fet_start']).sum()
        
        summary.extend([
            f"Total simulation time: {total_duration:.2f} seconds",
            f"Total compute time: {total_compute_time:.2f} seconds",
            f"Average utilization: {(total_compute_time/total_duration/len(self.nodes))*100:.2f}%",
            "\nPER-NODE STATISTICS",
            "-----------------"
        ])
        
        # Calculate per-node statistics
        for node in self.nodes:
            node_df = fets_df[fets_df['node'] == node]
            node_duration = node_df['t_fet_end'].max() - node_df['t_fet_start'].min()
            node_compute = (node_df['t_fet_end'] - node_df['t_fet_start']).sum()
            
            summary.extend([
                f"\n{node}:",
                f"  Active time: {node_compute:.2f} seconds",
                f"  Utilization: {(node_compute/node_duration)*100:.2f}%",
                f"  Number of executions: {len(node_df)}"
            ])
            
        with open(os.path.join(self.output_dir, 'utilization_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))
        
    def generate_invocation_report(self):
        if 'invocations_df' not in self.dfs:
            return
            
        inv_df = self.dfs['invocations_df']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Execution time distribution
        sns.histplot(data=inv_df, x='t_exec', ax=axes[0,0])
        axes[0,0].set_title('Function Execution Time Distribution')
        
        # Timeline plot
        sns.scatterplot(data=inv_df, x='t_start', y='t_exec', ax=axes[0,1])
        axes[0,1].set_title('Execution Timeline')
        
        # Response time by function
        if 'function_name' in inv_df.columns:
            sns.boxplot(data=inv_df, x='function_name', y='t_exec', ax=axes[1,0])
            axes[1,0].set_title('Response Time by Function')
            plt.xticks(rotation=45)
        
        # Wait time distribution
        sns.histplot(data=inv_df, x='t_wait', ax=axes[1,1])
        axes[1,1].set_title('Wait Time Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'invocation_analysis.png'))
        plt.close()
    
    def generate_resource_report(self):
        if 'utilization_df' not in self.dfs:
            return
            
        util_df = self.dfs['utilization_df']
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # CPU utilization timeline
        sns.lineplot(data=util_df, x=util_df.index, y='cpu_util', ax=axes[0])
        axes[0].set_title('CPU Utilization Timeline')
        
        # Resource allocation
        if 'cpu' in util_df.columns and 'memory' in util_df.columns:
            util_df[['cpu', 'memory']].plot(ax=axes[1])
            axes[1].set_title('Resource Allocation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'resource_analysis.png'))
        plt.close()
    
    def generate_network_report(self):
        if 'network_df' not in self.dfs:
            return
            
        net_df = self.dfs['network_df']
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Network traffic over time
        if 'bytes' in net_df.columns:
            sns.lineplot(data=net_df, x=net_df.index, y='bytes', ax=axes[0])
            axes[0].set_title('Network Traffic Over Time')
        
        # Traffic by node if available
        if 'source' in net_df.columns and 'destination' in net_df.columns:
            traffic_by_node = net_df.groupby(['source', 'destination'])['bytes'].sum()
            traffic_by_node.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Traffic by Node Pair')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'network_analysis.png'))
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary of all metrics"""
        summary = ["FaaS Simulation Analysis Summary", "=" * 30, ""]
        
        for name, df in self.dfs.items():
            if df is not None and not df.empty:
                summary.append(f"\n{name}:")
                summary.append("-" * len(name))
                summary.append(f"Records: {len(df)}")
                summary.append(f"Columns: {', '.join(df.columns)}")
                
                # Add metric-specific summaries
                if name == 'invocations_df':
                    summary.append(f"Mean execution time: {df['t_exec'].mean():.2f} ms")
                    summary.append(f"Mean wait time: {df['t_wait'].mean():.2f} ms")
                elif name == 'utilization_df' and 'cpu_util' in df.columns:
                    summary.append(f"Mean CPU utilization: {df['cpu_util'].mean():.2f}%")
                
                summary.append("")
        
        # Save summary
        with open(os.path.join(self.output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))

def main():
    visualizer = MetricsVisualizer()
    if visualizer.load_data():
        visualizer.generate_reports()
        print(f"Reports generated in {visualizer.output_dir}")
    else:
        print("No data found to analyze")

if __name__ == '__main__':
    main()