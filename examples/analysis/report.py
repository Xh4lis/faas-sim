import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
from datetime import datetime
from collections import defaultdict
from . import nrg  

import warnings
warnings.filterwarnings('ignore')


# Set matplotlib style for better plots
mpl.rcParams["figure.figsize"] = (10, 6)
mpl.rcParams["font.size"] = 12
mpl.style.use("ggplot")


class ReportGenerator:
    """Modular report generator for faas-sim analysis data"""

    def __init__(self, data_dir, output_dir):
        """Initialize the report generator with input and output directories"""
        self.data_dir = data_dir  # Store for energy analysis fallback
        self.output_dir = output_dir
        self.data = {}
        self.nodes = []
        self.functions = []
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all available data
        self.load_data()

    def ensure_output_dir(self, path):
        """Create directory if it doesn't exist"""
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def load_data(self):
        """Load all CSV files from the data directory"""
        if not os.path.exists(self.data_dir):
            print(f"Warning: Data directory {self.data_dir} does not exist!")
            return False

        for file in os.listdir(self.data_dir):
            if file.endswith(".csv"):
                name = file.replace(".csv", "")
                file_path = os.path.join(self.data_dir, file)
                try:
                    self.data[name] = pd.read_csv(file_path)
                    print(f"Loaded {name} with {len(self.data[name])} rows")
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        # Extract unique values for nodes, functions, and images if invocations data exists
        if "invocations_df" in self.data and not self.data["invocations_df"].empty:
            df = self.data["invocations_df"]
            self.nodes = df["node"].unique() if "node" in df.columns else []
            self.functions = (
                df["function_name"].unique() if "function_name" in df.columns else []
            )
            self.images = (
                df["function_image"].unique() if "function_image" in df.columns else []
            )

            print(
                f"Found {len(self.nodes)} nodes, {len(self.functions)} functions, and {len(self.images)} images"
            )

        return len(self.data) > 0

    def generate_all_reports(self):
        """Generate all available reports including energy analysis"""
        if not self.data:
            print("No data available. Please load data first.")
            return False

        print("Generating visualizations...")

        # Generate all existing reports
        self.generate_function_invocation_report()
        self.generate_resource_utilization_report()
        self.generate_deployment_timeline()
        self.generate_node_comparison_report()
        self.generate_function_comparison_report()
        self.generate_overall_summary()

        # Generate energy analysis - this will create files in the same output directory
        energy_summary = self.generate_energy_analysis()

        print(f"All visualizations saved to {os.path.abspath(self.output_dir)}")
        self.print_report_summary(energy_summary is not None)

        return True
    def generate_energy_analysis(self):
        """Generate energy analysis using the nrg module"""
        # Check if power and energy data are available
        if "power_df" in self.data and "energy_df" in self.data:
            print("Generating energy consumption analysis...")
            try:
                # Call nrg.main with the loaded dataframes and same output directory
                energy_summary = nrg.main(
                    power_df=self.data["power_df"],
                    energy_df=self.data["energy_df"],
                    output_dir=self.output_dir  # Same directory as other reports
                )
                print("✅ Energy analysis completed successfully!")
                return energy_summary
            except Exception as e:
                print(f"❌ Error generating energy analysis: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # Check if CSV files exist in data directory for fallback
            if hasattr(self, 'data_dir'):
                power_csv = os.path.join(self.data_dir, "power_df.csv")
                energy_csv = os.path.join(self.data_dir, "energy_df.csv")
                
                if os.path.exists(power_csv) and os.path.exists(energy_csv):
                    print("Power/Energy dataframes not in memory, but CSV files found. Loading from files...")
                    try:
                        energy_summary = nrg.main(
                            data_path=self.data_dir,
                            output_dir=self.output_dir
                        )
                        print("✅ Energy analysis completed from CSV files!")
                        return energy_summary
                    except Exception as e:
                        print(f"❌ Error generating energy analysis from CSV: {e}")
                        return None
            
            print("⚠️  Power and energy data not found, skipping energy analysis")
            return None

    # Update the print_report_summary method to include energy reports
    def print_report_summary(self, energy_analysis_generated=False):
        """Print summary of generated reports"""
        print("\n" + "="*50)
        print("📊 REPORT GENERATION SUMMARY")
        print("="*50)
        
        print("The following reports were generated:")
        print("  1. function_invocation_report.png - Overall analysis of function invocations")
        print("  2. function_utilization_report.png - CPU utilization analysis")
        print("  3. deployment_timeline.png - Timeline of deployment events")

        if len(self.nodes) > 1:
            print("  4. node_comparison_report.png - Performance comparison across nodes")
            print("  5. node_distribution.png - Distribution of workload across nodes")

        if len(self.functions) > 1:
            print("  6. function_comparison_report.png - Performance comparison across functions")
            print("  7. function_distribution.png - Distribution of functions across nodes")

        print("  8. overall_summary.png - Complete simulation summary")
        print("  9. overall_summary.txt - Text summary of simulation")
        
        # Add energy analysis reports if they were generated
        if energy_analysis_generated:
            print("\n🔋 ENERGY ANALYSIS REPORTS:")
            energy_files = [
                ("device_count_vs_power.png", "Device count vs total power contribution"),
                ("detailed_energy_analysis_focused.png", "Comprehensive energy analysis (6 visualizations)"),
                ("energy_dashboard_focused.png", "Energy consumption dashboard (6 visualizations)"), 
                ("energy_analysis_summary.txt", "Detailed energy analysis text report")
            ]
            
            for i, (filename, description) in enumerate(energy_files, 10):
                if os.path.exists(os.path.join(self.output_dir, filename)):
                    print(f"  {i}. {filename} - {description}")
        
        print(f"\n📁 All files saved to: {os.path.abspath(self.output_dir)}")
        print("="*50)

    def generate_function_invocation_report(self):
        """Generate function invocation analysis report with multi-node support"""
        if "invocations_df" not in self.data or self.data["invocations_df"].empty:
            print("No invocation data available")
            return
        if "fets_df" not in self.data or self.data["fets_df"].empty:
            print("No FETS data available")
            return
        inv_df = self.data["invocations_df"]
        fet_df = self.data["fets_df"].copy()
        fet_df["wait_time"] = fet_df["t_wait_end"] - fet_df["t_wait_start"]

        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Execution time distribution (top-left)
        axes[0, 0].hist(inv_df["t_exec"], bins=20, color="skyblue", edgecolor="black")
        axes[0, 0].set_title("Function Execution Time Distribution")
        axes[0, 0].set_xlabel("Execution Time (ms)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)

        # Plot 2: Execution timeline by node (top-right)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.nodes)))
        for i, node in enumerate(self.nodes):
            node_df = inv_df[inv_df["node"] == node]
            axes[0, 1].scatter(
                node_df["t_start"],
                node_df["t_exec"],
                alpha=0.6,
                color=colors[i % len(colors)],
                label=node,
            )
        axes[0, 1].set_title("Function Execution Timeline by Node")
        axes[0, 1].set_xlabel("Start Time (s)")
        axes[0, 1].set_ylabel("Execution Time (ms)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.7)
        if len(self.nodes) <= 10:  # Only show legend if not too many nodes
            axes[0, 1].legend(loc="upper right")

        # Plot 3: Cumulative invocations by function (bottom-left)
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.functions)))
        for i, function in enumerate(self.functions):
            func_df = inv_df[inv_df["function_name"] == function]
            start_times = sorted(func_df["t_start"])
            cum_invocations = list(range(1, len(start_times) + 1))
            axes[1, 0].step(
                start_times,
                cum_invocations,
                where="post",
                color=colors[i % len(colors)],
                label=function,
            )
        axes[1, 0].set_title("Cumulative Invocations by Function")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Total Invocations")
        axes[1, 0].grid(True, linestyle="--", alpha=0.7)
        if len(self.functions) <= 10:  # Only show legend if not too many functions
            axes[1, 0].legend(loc="upper left")

        # Plot 4: Wait time vs Execution time by node (bottom-right)
        for i, node in enumerate(self.nodes):
            node_fet_df = fet_df[fet_df["node"] == node]
            node_inv_df = inv_df[inv_df["node"] == node]
            axes[1, 1].scatter(
                node_fet_df["wait_time"],
                node_inv_df["t_exec"],
                alpha=0.6,
                color=colors[i % len(colors)],
                label=node,
            )
        axes[1, 1].set_title("Wait Time (FETS) vs Execution Time by Node")
        axes[1, 1].set_xlabel("Wait Time (ms)")
        axes[1, 1].set_ylabel("Execution Time (ms)")
        axes[1, 1].grid(True, linestyle="--", alpha=0.7)
        if len(self.nodes) <= 10:  # Only show legend if not too many nodes
            axes[1, 1].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "function_invocation_report.png"), dpi=300
        )
        plt.close()

        # Generate the summary text file
        self._generate_invocation_summary_text(inv_df, fet_df)

    def _generate_invocation_summary_text(self, inv_df, fet_df):
        """Generate detailed invocation summary text file"""
        with open(os.path.join(self.output_dir, "invocation_summary.txt"), "w") as f:
            f.write("FUNCTION INVOCATION SUMMARY\n")
            f.write("==========================\n\n")
            f.write(f"Total invocations: {len(inv_df)}\n")
            f.write(
                f"Overall average execution time: {inv_df['t_exec'].mean():.2f} ms\n"
            )
            f.write(f"Overall average wait time: {fet_df['wait_time'].mean():.2f} ms\n")
            f.write(f"Simulation start time: {inv_df['t_start'].min():.2f} s\n")
            f.write(
                f"Simulation end time: {inv_df['t_start'].max() + inv_df['t_exec'].max()/1000:.2f} s\n"
            )
            f.write(
                f"Total duration: {inv_df['t_start'].max() - inv_df['t_start'].min():.2f} s\n\n"
            )

            # Add node summary section
            if len(self.nodes) > 0:
                f.write("NODE SUMMARY\n")
                f.write("============\n\n")
                for node in self.nodes:
                    node_inv_df = inv_df[inv_df["node"] == node]
                    node_fet_df = fet_df[fet_df["node"] == node]
                    f.write(f"Node: {node}\n")
                    f.write(f"  Invocations: {len(node_inv_df)}\n")
                    f.write(
                        f"  Average execution time: {node_inv_df['t_exec'].mean():.2f} ms\n"
                    )
                    f.write(
                        f"  Min/Max execution time: {node_inv_df['t_exec'].min():.2f}/{node_inv_df['t_exec'].max():.2f} ms\n"
                    )
                    f.write(
                        f"  Average wait time: {node_fet_df['wait_time'].mean():.2f} ms\n"
                    )
                    f.write("\n")

            # Add function summary section
            if len(self.functions) > 0:
                f.write("FUNCTION SUMMARY\n")
                f.write("================\n\n")
                for function in self.functions:
                    func_inv_df = inv_df[inv_df["function_name"] == function]
                    func_fet_df = fet_df[fet_df["function_name"] == function]
                    f.write(f"Function: {function}\n")
                    f.write(f"  Invocations: {len(func_inv_df)}\n")
                    f.write(
                        f"  Average execution time: {func_inv_df['t_exec'].mean():.2f} ms\n"
                    )
                    f.write(
                        f"  Min/Max execution time: {func_inv_df['t_exec'].min():.2f}/{func_inv_df['t_exec'].max():.2f} ms\n"
                    )
                    f.write(
                        f"  Average wait time: {func_fet_df['wait_time'].mean():.2f} ms\n"
                    )
                    f.write("\n")

            # Add detailed breakdown by node, function, and image
            f.write("DETAILED BREAKDOWN BY NODE AND FUNCTION\n")
            f.write("=====================================\n\n")
            grouped = inv_df.groupby(["node", "function_name", "function_image"])
            for (node, fn_name, fn_image), group in grouped:
                # Get corresponding wait times from fet_df
                fet_group = fet_df[
                    (fet_df["node"] == node)
                    & (fet_df["function_name"] == fn_name)
                    & (fet_df["function_image"] == fn_image)
                ]
                avg_wait = (
                    fet_group["wait_time"].mean()
                    if not fet_group.empty
                    else float("nan")
                )
                f.write(f"Node: {node}\n")
                f.write(f"Function: {fn_name}\n")
                f.write(f"Image: {fn_image}\n")
                f.write(f"Invocations: {len(group)}\n")
                f.write(f"Average execution time: {group['t_exec'].mean():.2f} ms\n")
                f.write(
                    f"Min/Max execution time: {group['t_exec'].min():.2f}/{group['t_exec'].max():.2f} ms\n"
                )
                f.write(f"Average wait time: {avg_wait:.2f} ms\n")
                f.write(f"First invocation: {group['t_start'].min():.2f} s\n")
                f.write(f"Last invocation: {group['t_start'].max():.2f} s\n")
                f.write("\n")

    def generate_resource_utilization_report(self):
        """Generate resource utilization report with multi-node support"""
        if (
            "function_utilization_df" not in self.data
            or self.data["function_utilization_df"].empty
        ):
            print("No function utilization data available")
            return

        util_df = self.data["function_utilization_df"]

        # Get unique nodes in utilization data
        nodes = util_df["node"].unique() if "node" in util_df.columns else []

        # Create a figure with subplots for utilization data
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot CPU utilization over time for each node
        if len(nodes) > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, len(nodes)))
            for i, node in enumerate(nodes):
                node_df = util_df[util_df["node"] == node]
                if not node_df.empty:
                    # Sort by time or sample number if available
                    if "time" in node_df.columns:
                        node_df = node_df.sort_values("time")
                        x_values = node_df["time"]
                        x_label = "Time (s)"
                    else:
                        # Use sample index as x-axis
                        x_values = range(len(node_df))
                        x_label = "Sample"

                    axes[0].plot(
                        x_values,
                        node_df["cpu_util"],
                        marker=".",
                        linestyle="-",
                        color=colors[i % len(colors)],
                        label=f"{node}",
                    )

            axes[0].set_title("CPU Utilization by Node")
            axes[0].set_ylabel("CPU Utilization (%)")
            axes[0].set_xlabel(x_label)
            axes[0].grid(True, linestyle="--", alpha=0.7)

            if len(nodes) <= 10:  # Only show legend if not too many nodes
                axes[0].legend(loc="upper right")

            # Add average line for each node
            for i, node in enumerate(nodes):
                node_df = util_df[util_df["node"] == node]
                avg_cpu = node_df["cpu_util"].mean()
                axes[0].axhline(
                    y=avg_cpu, color=colors[i % len(colors)], linestyle="--", alpha=0.5
                )
        else:
            # No node information, just plot overall CPU utilization
            x_values = range(len(util_df))
            axes[0].plot(
                x_values, util_df["cpu_util"], marker=".", linestyle="-", color="blue"
            )
            axes[0].set_title("CPU Utilization")
            axes[0].set_ylabel("CPU Utilization (%)")
            axes[0].set_xlabel("Sample")
            axes[0].grid(True, linestyle="--", alpha=0.7)

            # Add average line
            avg_cpu = util_df["cpu_util"].mean()
            axes[0].axhline(
                y=avg_cpu, color="red", linestyle="--", label=f"Avg: {avg_cpu:.2f}%"
            )
            axes[0].legend()

        # Plot average CPU utilization by node as bar chart
        if len(nodes) > 0:
            node_avg_cpu = util_df.groupby("node")["cpu_util"].mean()
            axes[1].bar(
                node_avg_cpu.index,
                node_avg_cpu.values,
                color=colors[: len(node_avg_cpu)],
            )
            axes[1].set_title("Average CPU Utilization by Node")
            axes[1].set_xlabel("Node")
            axes[1].set_ylabel("Average CPU Utilization (%)")
            axes[1].grid(True, axis="y", linestyle="--", alpha=0.7)
            axes[1].set_xticklabels(node_avg_cpu.index, rotation=45)
        else:
            # No node information, show CPU vs allocation
            axes[1].plot(
                x_values,
                util_df["cpu"] / 1000,
                marker="s",
                linestyle="-",
                color="green",
                label="CPU Allocation",
            )
            axes[1].set_title("CPU Allocation")
            axes[1].set_xlabel("Sample")
            axes[1].set_ylabel("CPU Allocation (units)")
            axes[1].grid(True, linestyle="--", alpha=0.7)

            # Add average line
            avg_cpu_alloc = util_df["cpu"].mean() / 1000
            axes[1].axhline(
                y=avg_cpu_alloc,
                color="red",
                linestyle="--",
                label=f"Avg: {avg_cpu_alloc:.2f} units",
            )
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "function_utilization_report.png"), dpi=300
        )
        plt.close()

        # Generate utilization summary text file
        self._generate_utilization_summary_text(util_df, nodes)

    def _generate_utilization_summary_text(self, util_df, nodes):
        """Generate detailed utilization summary text file"""
        with open(os.path.join(self.output_dir, "utilization_summary.txt"), "w") as f:
            f.write("RESOURCE UTILIZATION SUMMARY\n")
            f.write("===========================\n\n")
            f.write(
                f"Overall average CPU utilization: {util_df['cpu_util'].mean():.2f}%\n"
            )
            f.write(f"Overall peak CPU utilization: {util_df['cpu_util'].max():.2f}%\n")
            f.write(f"Number of samples: {len(util_df)}\n\n")

            # Add per-node breakdown if available
            if len(nodes) > 0:
                f.write("CPU UTILIZATION BY NODE\n")
                f.write("======================\n\n")

                for node in nodes:
                    node_df = util_df[util_df["node"] == node]
                    f.write(f"Node: {node}\n")
                    f.write(
                        f"  Average CPU utilization: {node_df['cpu_util'].mean():.2f}%\n"
                    )
                    f.write(
                        f"  Peak CPU utilization: {node_df['cpu_util'].max():.2f}%\n"
                    )

                    if "cpu" in node_df.columns:
                        f.write(
                            f"  CPU allocation: {node_df['cpu'].mean()/1000:.2f} CPU units\n"
                        )

                    f.write(f"  Number of samples: {len(node_df)}\n")

                    # Add function-specific info if available
                    if "function_name" in node_df.columns:
                        function_groups = node_df.groupby("function_name")
                        for function, func_df in function_groups:
                            f.write(f"  Function {function}:\n")
                            f.write(
                                f"    Average CPU: {func_df['cpu_util'].mean():.2f}%\n"
                            )
                            f.write(f"    Peak CPU: {func_df['cpu_util'].max():.2f}%\n")

                    f.write("\n")

            # Add details for the default node if no node information
            else:
                if (
                    "invocations_df" in self.data
                    and not self.data["invocations_df"].empty
                ):
                    inv_df = self.data["invocations_df"]
                    if "function_name" in inv_df.columns:
                        f.write(f"Function: {inv_df['function_name'].iloc[0]}\n")
                    if "node" in inv_df.columns:
                        f.write(f"Node: {inv_df['node'].iloc[0]}\n")

    def _generate_deployment_timeline_text(self, lifecycle_events, groups):
        """Generate detailed deployment timeline text file"""
        with open(os.path.join(self.output_dir, "deployment_timeline.txt"), "w") as f:
            f.write("FUNCTION DEPLOYMENT TIMELINE\n")
            f.write("============================\n\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write(f"Total deployment events: {len(lifecycle_events)}\n")
            f.write(f"Number of node-function combinations: {len(groups)}\n\n")
            
            # Event type breakdown
            event_counts = {"deployment": 0, "replica": 0, "network": 0}
            for event in lifecycle_events:
                event_counts[event["type"]] += 1
            
            f.write("Event Type Summary:\n")
            f.write(f"  Deployment events: {event_counts['deployment']}\n")
            f.write(f"  Replica events: {event_counts['replica']}\n")
            f.write(f"  Network events: {event_counts['network']}\n\n")
            
            # Timeline sorted by time
            f.write("CHRONOLOGICAL TIMELINE\n")
            f.write("======================\n\n")
            sorted_events = sorted(lifecycle_events, key=lambda x: x["time"])
            
            for event in sorted_events:
                f.write(f"t={event['time']:6.1f}s | {event['type']:10} | {event['node']:20} | {event['function']:25} | {event['event']}\n")
            
            f.write("\n")
            
            # Grouped by node-function combination
            f.write("GROUPED BY NODE-FUNCTION\n")
            f.write("========================\n\n")
            
            sorted_groups = sorted(groups.keys())
            for group_name in sorted_groups:
                events = groups[group_name]
                f.write(f"{group_name}:\n")
                f.write("-" * (len(group_name) + 1) + "\n")
                
                # Sort events in this group by time
                events_sorted = sorted(events, key=lambda x: x["time"])
                
                for event in events_sorted:
                    f.write(f"  t={event['time']:6.1f}s - {event['event']} [{event['type']}]\n")
                
                f.write(f"  Total events: {len(events)}\n\n")
            
            # Node summary
            node_stats = {}
            for event in lifecycle_events:
                node = event["node"]
                if node not in node_stats:
                    node_stats[node] = {"deployment": 0, "replica": 0, "network": 0, "functions": set()}
                node_stats[node][event["type"]] += 1
                node_stats[node]["functions"].add(event["function"])
            
            f.write("NODE SUMMARY\n")
            f.write("============\n\n")
            
            for node, stats in sorted(node_stats.items()):
                f.write(f"Node: {node}\n")
                f.write(f"  Deployment events: {stats['deployment']}\n")
                f.write(f"  Replica events: {stats['replica']}\n")
                f.write(f"  Network events: {stats['network']}\n")
                f.write(f"  Functions involved: {len(stats['functions'])}\n")
                f.write(f"  Function list: {', '.join(sorted(stats['functions']))}\n\n")
            
            # Function summary
            function_stats = {}
            for event in lifecycle_events:
                function = event["function"]
                if function not in function_stats:
                    function_stats[function] = {"deployment": 0, "replica": 0, "network": 0, "nodes": set()}
                function_stats[function][event["type"]] += 1
                function_stats[function]["nodes"].add(event["node"])
            
            f.write("FUNCTION SUMMARY\n")
            f.write("================\n\n")
            
            for function, stats in sorted(function_stats.items()):
                f.write(f"Function: {function}\n")
                f.write(f"  Deployment events: {stats['deployment']}\n")
                f.write(f"  Replica events: {stats['replica']}\n")
                f.write(f"  Network events: {stats['network']}\n")
                f.write(f"  Nodes involved: {len(stats['nodes'])}\n")
                f.write(f"  Node list: {', '.join(sorted(stats['nodes']))}\n\n")
            
            # Scaling analysis (specifically for your debugging)
            f.write("SCALING ANALYSIS\n")
            f.write("================\n\n")
            
            scaling_functions = set()
            for event in lifecycle_events:
                if event["type"] == "replica" and "replica of" in event["event"]:
                    scaling_functions.add(event["function"])
            
            if scaling_functions:
                f.write(f"Functions with scaling events: {len(scaling_functions)}\n")
                for func in sorted(scaling_functions):
                    replica_events = [e for e in lifecycle_events if e["function"] == func and e["type"] == "replica"]
                    f.write(f"  {func}: {len(replica_events)} replica events\n")
                    for event in sorted(replica_events, key=lambda x: x["time"]):
                        f.write(f"    t={event['time']:6.1f}s - {event['event']} on {event['node']}\n")
                    f.write("\n")
            else:
                f.write("⚠️  NO SCALING EVENTS DETECTED!\n")
                f.write("This indicates that auto-scaling is not working properly.\n")
                f.write("All functions remained at their initial replica count.\n\n")
                
                # Show what we expected vs what we got
                f.write("Expected scaling events would look like:\n")
                f.write("  - 'deploy replica of function-name' events\n")
                f.write("  - Multiple replica events per function\n")
                f.write("  - Replica events distributed across different nodes\n\n")


    def generate_deployment_timeline(self):
        """Generate deployment timeline as a waterfall visualization"""
        lifecycle_events = []

        # Collect lifecycle events from different dataframes
        if (
            "function_deployment_lifecycle_df" in self.data
            and not self.data["function_deployment_lifecycle_df"].empty
        ):
            for _, row in self.data["function_deployment_lifecycle_df"].iterrows():
                lifecycle_events.append(
                    {
                        "time": row.get("timestamp", 0),
                        "event": f"{row['value']} {row['name']}",
                        "type": "deployment",
                        "node": "global",
                        "function": row["name"],
                    }
                )

        if (
            "replica_deployment_df" in self.data
            and not self.data["replica_deployment_df"].empty
        ):
            for i, row in enumerate(self.data["replica_deployment_df"].itertuples()):
                timestamp = (
                    getattr(row, "timestamp", i) if hasattr(row, "timestamp") else i
                )
                lifecycle_events.append(
                    {
                        "time": timestamp,
                        "event": f"{row.value} replica of {row.function_name}",
                        "type": "replica",
                        "node": row.node_name,
                        "function": row.function_name,
                    }
                )

        if "flow_df" in self.data and not self.data["flow_df"].empty:
            for j, row in enumerate(self.data["flow_df"].itertuples()):
                timestamp = (
                    getattr(row, "timestamp", 2 + j / 10)
                    if hasattr(row, "timestamp")
                    else 2 + j / 10
                )
                lifecycle_events.append(
                    {
                        "time": timestamp,
                        "event": f"{row.action_type} ({row.bytes/1024/1024:.1f} MB)",
                        "type": "network",
                        "node": f"{row.source} → {row.sink}",
                        "function": (
                            getattr(row, "function_name", "unknown")
                            if hasattr(row, "function_name")
                            else "unknown"
                        ),
                    }
                )

        if not lifecycle_events:
            print("No deployment events to visualize")
            return

        # Create a waterfall chart - a more traditional waterfall visualization
        plt.figure(figsize=(14, 8))

        # Sort events by time
        lifecycle_events.sort(key=lambda x: x["time"])

        # Group events by both node and function for better organization
        groups = defaultdict(list)
        for event in lifecycle_events:
            key = f"{event['node']} - {event['function']}"
            groups[key].append(event)

        # Define event colors - using more distinct colors
        event_colors = {
            "deployment": "royalblue",
            "replica": "forestgreen",
            "network": "darkorange",
        }

        # Calculate number of groups and set up y positions with good spacing
        num_groups = len(groups)
        y_positions = {}

        # Create sorted groups for consistent y positions
        sorted_groups = sorted(groups.keys())
        for i, group in enumerate(sorted_groups):
            y_positions[group] = (
                num_groups - i - 1
            ) * 2  # Reverse order and double spacing

        # Plot events as horizontal bars in waterfall style
        for group_name, events in groups.items():
            y_pos = y_positions[group_name]

            # Plot each event in the group
            for event in events:
                # Plot the bar
                bar_height = 0.6
                bar = plt.barh(
                    y_pos,
                    width=0.8,
                    left=event["time"],
                    height=bar_height,
                    color=event_colors[event["type"]],
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=1,
                )

                # Add event label with slight offset
                plt.text(
                    event["time"] + 0.1,  # Small right offset
                    y_pos,  # Centered vertically
                    event["event"],
                    verticalalignment="center",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                )

        # Set y-ticks to show clean group labels
        plt.yticks(list(y_positions.values()), list(y_positions.keys()))
        plt.title("Function Deployment Timeline", fontsize=14, fontweight="bold")
        plt.xlabel("Time/Sequence", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Expand axes for better visibility
        plt.xlim(
            -0.5, max([e["time"] for e in lifecycle_events]) + 5
        )  # Add padding on right for labels

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=event_colors["deployment"],
                lw=4,
                label="Deployment Events",
            ),
            Line2D(
                [0], [0], color=event_colors["replica"], lw=4, label="Replica Events"
            ),
            Line2D(
                [0], [0], color=event_colors["network"], lw=4, label="Network Events"
            ),
        ]
        plt.legend(handles=legend_elements, loc="upper right", frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "deployment_timeline.png"), dpi=300)
        plt.close()
        self._generate_deployment_timeline_text(lifecycle_events, groups)

    
    def generate_node_comparison_report(self):
        """Generate node comparison report if multiple nodes are present"""
        if "invocations_df" not in self.data or self.data["invocations_df"].empty:
            print("No invocation data available")
            return
        if "fets_df" not in self.data or self.data["fets_df"].empty:
            print("No FETS data available")
            return
        inv_df = self.data["invocations_df"]
        fet_df = self.data["fets_df"].copy()
        fet_df["wait_time"] = fet_df["t_wait_end"] - fet_df["t_wait_start"]

        # Skip if only one node
        if len(self.nodes) <= 1:
            print("Only one node found, skipping node comparison")
            return

        # Generate node comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Boxplot of execution times by node
        node_data = [inv_df[inv_df["node"] == node]["t_exec"] for node in self.nodes]
        axes[0, 0].boxplot(node_data, labels=self.nodes)
        axes[0, 0].set_title("Function Execution Time by Node")
        axes[0, 0].set_ylabel("Execution Time (ms)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)
        axes[0, 0].set_xticklabels(self.nodes, rotation=45)

        # Plot 2: Bar chart of average execution times
        node_means = [
            inv_df[inv_df["node"] == node]["t_exec"].mean() for node in self.nodes
        ]
        axes[0, 1].bar(self.nodes, node_means, color="skyblue")
        axes[0, 1].set_title("Average Execution Time by Node")
        axes[0, 1].set_ylabel("Average Execution Time (ms)")
        axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[0, 1].set_xticklabels(self.nodes, rotation=45)

        # Plot 3: Invocation count per node
        node_counts = inv_df["node"].value_counts()
        axes[1, 0].bar(node_counts.index, node_counts.values, color="lightgreen")
        axes[1, 0].set_title("Number of Invocations per Node")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_xlabel("Node")
        axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[1, 0].set_xticklabels(node_counts.index, rotation=45)

        # Plot 4: Wait time by node (from FETS)
        node_wait_means = [
            fet_df[fet_df["node"] == node]["wait_time"].mean() for node in self.nodes
        ]
        axes[1, 1].bar(self.nodes, node_wait_means, color="coral")
        axes[1, 1].set_title("Average Wait Time by Node (FETS)")
        axes[1, 1].set_ylabel("Average Wait Time (ms)")
        axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[1, 1].set_xticklabels(self.nodes, rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "node_comparison_report.png"), dpi=300
        )
        plt.close()

        # Create function distribution across nodes
        if len(self.functions) > 1:
            # Create a matrix of function counts per node
            function_node_matrix = pd.crosstab(inv_df["function_name"], inv_df["node"])

            plt.figure(figsize=(12, 6))
            function_node_matrix.plot(kind="bar", stacked=True)
            plt.title("Function Distribution Across Nodes")
            plt.ylabel("Number of Invocations")
            plt.xlabel("Function")
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.legend(title="Node")
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "function_distribution.png"), dpi=300
            )
            plt.close()

            # Also create node distribution chart
            node_function_matrix = pd.crosstab(inv_df["node"], inv_df["function_name"])

            plt.figure(figsize=(12, 6))
            node_function_matrix.plot(kind="bar", stacked=True)
            plt.title("Node Workload Distribution")
            plt.ylabel("Number of Invocations")
            plt.xlabel("Node")
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.legend(title="Function")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "node_distribution.png"), dpi=300)
            plt.close()

    def generate_function_comparison_report(self):
        """Generate function comparison report if multiple functions are present"""
        if "invocations_df" not in self.data or self.data["invocations_df"].empty:
            print("No invocation data available")
            return
        if "fets_df" not in self.data or self.data["fets_df"].empty:
            print("No FETS data available")
            return
        inv_df = self.data["invocations_df"]
        fet_df = self.data["fets_df"].copy()
        fet_df["wait_time"] = fet_df["t_wait_end"] - fet_df["t_wait_start"]

        # Skip if only one function
        if len(self.functions) <= 1:
            print("Only one function found, skipping function comparison")
            return

        # Generate function comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Boxplot of execution times by function
        function_data = [
            inv_df[inv_df["function_name"] == fn]["t_exec"] for fn in self.functions
        ]
        axes[0, 0].boxplot(function_data, labels=self.functions)
        axes[0, 0].set_title("Execution Time by Function")
        axes[0, 0].set_ylabel("Execution Time (ms)")
        axes[0, 0].grid(True, linestyle="--", alpha=0.7)
        axes[0, 0].set_xticklabels(self.functions, rotation=45)

        # Plot 2: Bar chart of average execution times
        function_means = [
            inv_df[inv_df["function_name"] == fn]["t_exec"].mean()
            for fn in self.functions
        ]
        axes[0, 1].bar(self.functions, function_means, color="skyblue")
        axes[0, 1].set_title("Average Execution Time by Function")
        axes[0, 1].set_ylabel("Average Execution Time (ms)")
        axes[0, 1].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[0, 1].set_xticklabels(self.functions, rotation=45)

        # Plot 3: Invocation count per function
        function_counts = inv_df["function_name"].value_counts()
        axes[1, 0].bar(
            function_counts.index, function_counts.values, color="lightgreen"
        )
        axes[1, 0].set_title("Number of Invocations per Function")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_xlabel("Function")
        axes[1, 0].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[1, 0].set_xticklabels(function_counts.index, rotation=45)

        # Plot 4: Wait time by function (from FETS)
        function_wait_means = [
            fet_df[fet_df["function_name"] == fn]["wait_time"].mean()
            for fn in self.functions
        ]
        axes[1, 1].bar(self.functions, function_wait_means, color="coral")
        axes[1, 1].set_title("Average Wait Time by Function (FETS)")
        axes[1, 1].set_ylabel("Average Wait Time (ms)")
        axes[1, 1].grid(True, axis="y", linestyle="--", alpha=0.7)
        axes[1, 1].set_xticklabels(self.functions, rotation=45)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "function_comparison_report.png"), dpi=300
        )
        plt.close()

        # Create image comparison if multiple images
        if len(self.images) > 1:
            # Compare performance by image type
            plt.figure(figsize=(12, 6))

            image_means = []
            image_labels = []

            for image in self.images:
                image_df = inv_df[inv_df["function_image"] == image]
                if not image_df.empty:
                    image_means.append(image_df["t_exec"].mean())
                    image_labels.append(image)

            plt.bar(image_labels, image_means, color="purple")
            plt.title("Average Execution Time by Image")
            plt.ylabel("Average Execution Time (ms)")
            plt.xlabel("Function Image")
            plt.grid(True, axis="y", linestyle="--", alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "image_comparison.png"), dpi=300)
            plt.close()

    def generate_overall_summary(self):
        """Generate overall summary visualization with all metrics"""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Prepare FETS data if available
        fet_df = None
        if "fets_df" in self.data and not self.data["fets_df"].empty:
            fet_df = self.data["fets_df"].copy()
            fet_df["wait_time"] = fet_df["t_wait_end"] - fet_df["t_wait_start"]

        # Set up summary text with much more detail
        summary_text = [
            "FAAS SIMULATION SUMMARY",
            "======================",
            "",
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Add topology information
        if len(self.nodes) > 0 or len(self.functions) > 0 or len(self.images) > 0:
            summary_text.extend(
                [
                    "Topology Overview:",
                    f"  Number of nodes: {len(self.nodes)}",
                    f"  Number of functions: {len(self.functions)}",
                    f"  Number of images: {len(self.images)}",
                    "",
                ]
            )

            # Add node distribution information if multiple nodes
            if len(self.nodes) > 1 and "invocations_df" in self.data:
                inv_df = self.data["invocations_df"]
                summary_text.append("Node Distribution:")
                for node in self.nodes:
                    node_count = len(inv_df[inv_df["node"] == node])
                    summary_text.append(
                        f"  {node}: {node_count} invocations ({node_count/len(inv_df)*100:.1f}%)"
                    )
                summary_text.append("")

            # Add function distribution information if multiple functions
            if len(self.functions) > 1 and "invocations_df" in self.data:
                inv_df = self.data["invocations_df"]
                summary_text.append("Function Distribution:")
                for function in self.functions:
                    func_count = len(inv_df[inv_df["function_name"] == function])
                    summary_text.append(
                        f"  {function}: {func_count} invocations ({func_count/len(inv_df)*100:.1f}%)"
                    )
                "Overall Invocation Statistics:",
                f"  Total invocations: {len(inv_df)}",
                f"  Avg execution time: {inv_df['t_exec'].mean():.2f} ms",
                f"  Max execution time: {inv_df['t_exec'].max():.2f} ms",
                f"  Avg execution time: {inv_df['t_exec'].mean():.2f} ms",
                f"  Max execution time: {inv_df['t_exec'].max():.2f} ms",
                f"  Avg wait time: {fet_df['wait_time'].mean():.2f} ms",
                f"  Max wait time: {fet_df['wait_time'].max():.2f} ms",
                f"  Simulation duration: {inv_df['t_start'].max() - inv_df['t_start'].min():.2f} s",
                "",

        # Add resource utilization
        if (
            "function_utilization_df" in self.data
            and not self.data["function_utilization_df"].empty
        ):
            util_df = self.data["function_utilization_df"]
            summary_text.extend(
                [
                    "Resource Utilization:",
                    f"  Overall avg CPU utilization: {util_df['cpu_util'].mean():.2f}%",
                    f"  Overall peak CPU utilization: {util_df['cpu_util'].max():.2f}%",
                    "",
                ]
            )

            # Add per-node resource utilization
            if len(self.nodes) > 1:
                for node in self.nodes:
                    node_df = util_df[util_df["node"] == node]
                    if not node_df.empty:
                        summary_text.append(f"  {node}:")
                        summary_text.append(
                            f"    Avg CPU: {node_df['cpu_util'].mean():.2f}%"
                        )
                        summary_text.append(
                            f"    Peak CPU: {node_df['cpu_util'].max():.2f}%"
                        )
                summary_text.append("")

        # Add network statistics
        if "flow_df" in self.data and not self.data["flow_df"].empty:
            flow_df = self.data["flow_df"]
            summary_text.extend(
                [
                    "Network Activity:",
                    f"  Total transfers: {len(flow_df)}",
                    f"  Total bytes transferred: {flow_df['bytes'].sum()/1024/1024:.2f} MB",
                    f"  Transfer duration: {flow_df['duration'].sum():.2f} s",
                    "",
                ]
            )

            # Add per-node network statistics if available
            if "source" in flow_df.columns and "sink" in flow_df.columns:
                # Group by source-sink pairs
                flow_pairs = flow_df.groupby(["source", "sink"]).agg(
                    {"bytes": "sum", "duration": "sum"}
                )

                if not flow_pairs.empty:
                    summary_text.append("Network Flows:")
                    for (source, sink), row in flow_pairs.iterrows():
                        summary_text.append(
                            f"  {source} → {sink}: {row['bytes']/1024/1024:.2f} MB in {row['duration']:.2f} s"
                        )
                    summary_text.append("")

        # Hide axes
        ax.axis("off")

        # Add text to the figure
        plt.text(
            0.05,
            0.95,
            "\n".join(summary_text),
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            family="monospace",
        )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "overall_summary.png"), dpi=300)
        plt.close()

        # Also save as text file
        with open(os.path.join(self.output_dir, "overall_summary.txt"), "w") as f:
            f.write("\n".join(summary_text))


def main():
    """Main function to generate all reports with command line argument support"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate FaaS simulation analysis reports"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="raith_data",
        help="Directory containing analysis data (default: raith_data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vis-no-autoscale",
        help="Directory for visualization output (default: vis-no-autoscale)",
    )
    args = parser.parse_args()

    # Create report generator with provided or default parameters
    report_gen = ReportGenerator(data_dir=args.data_dir, output_dir=args.output_dir)

    # Load data
    print(f"Loading data from {report_gen.data_dir}")
    if not report_gen.load_data():
        print("No data found or error loading data.")
        return

    # Generate all reports
    report_gen.generate_all_reports()


if __name__ == "__main__":
    main()
