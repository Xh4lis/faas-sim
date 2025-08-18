import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Experiment data directories
K8S_EXPERIMENT_DIR = "./data/sine_k8s_strategy_d120_r35"
LPLT_EXPERIMENT_DIR = "./data/sine_power_strategy_d120_r35"

def compare_experiments():
    """Simple comparison of two experiment results using robust metrics"""

    # Load the CSV files from both experiments - CORRECTED PATHS
    k8s_power = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/power_df.csv")
    lplt_power = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/power_df.csv")

    # Use autoscaler metrics for response times
    k8s_metrics = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/autoscaler_detailed_metrics_df.csv")
    lplt_metrics = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/autoscaler_detailed_metrics_df.csv")


    # Simple comparisons
    print("=== POWER CONSUMPTION COMPARISON ===")
    k8s_avg_power = k8s_power['power_watts'].mean()
    lplt_avg_power = lplt_power['power_watts'].mean()
    power_savings = ((k8s_avg_power - lplt_avg_power) / k8s_avg_power) * 100

    print(f"Kubernetes average power: {k8s_avg_power:.1f}W")
    print(f"LPLT average power: {lplt_avg_power:.1f}W")
    print(f"Power savings: {power_savings:.1f}%")

    print("\n=== PERFORMANCE COMPARISON (ROBUST METRICS) ===")
    
    # Use MEDIAN instead of MEAN for response times
    k8s_response_median = k8s_metrics['avg_response_time'].median()
    lplt_response_median = lplt_metrics['avg_response_time'].median()
    median_penalty = ((lplt_response_median - k8s_response_median) / k8s_response_median) * 100

    print(f"Kubernetes MEDIAN response time: {k8s_response_median:.3f}s")
    print(f"LPLT MEDIAN response time: {lplt_response_median:.3f}s")
    print(f"MEDIAN performance penalty: {median_penalty:.1f}%")

    # 95th percentile comparison (excludes extreme outliers)
    k8s_p95 = k8s_metrics['avg_response_time'].quantile(0.95)
    lplt_p95 = lplt_metrics['avg_response_time'].quantile(0.95)
    p95_penalty = ((lplt_p95 - k8s_p95) / k8s_p95) * 100

    print(f"Kubernetes 95th percentile: {k8s_p95:.3f}s")
    print(f"LPLT 95th percentile: {lplt_p95:.3f}s")
    print(f"95th percentile penalty: {p95_penalty:.1f}%")

    # Exclude cold start period (first 10% of measurements)
    k8s_warm = k8s_metrics.iloc[int(len(k8s_metrics) * 0.1):]
    lplt_warm = lplt_metrics.iloc[int(len(lplt_metrics) * 0.1):]
    
    k8s_warm_mean = k8s_warm['avg_response_time'].mean()
    lplt_warm_mean = lplt_warm['avg_response_time'].mean()
    warm_penalty = ((lplt_warm_mean - k8s_warm_mean) / k8s_warm_mean) * 100

    print(f"Kubernetes warm-up avg: {k8s_warm_mean:.3f}s")
    print(f"LPLT warm-up avg: {lplt_warm_mean:.3f}s")
    print(f"Warm-up performance penalty: {warm_penalty:.1f}%")

    print("\n=== WAIT TIME ANALYSIS ===")
    k8s_wait_median = k8s_metrics['avg_wait_time'].median()
    lplt_wait_median = lplt_metrics['avg_wait_time'].median()
    
    # Handle division by zero
    if k8s_wait_median > 0:
        wait_improvement = ((k8s_wait_median - lplt_wait_median) / k8s_wait_median) * 100
    else:
        wait_improvement = 0
    
    print(f"K8s median wait time: {k8s_wait_median:.3f}s")
    print(f"LPLT median wait time: {lplt_wait_median:.3f}s")
    print(f"Wait time improvement: {wait_improvement:.1f}%")

    print("\n=== REVISED HYPOTHESIS RESULT ===")
    # Use median metrics for hypothesis testing
    if power_savings > 0 and median_penalty > 0:
        print(f"✅ HYPOTHESIS CONFIRMED (MEDIAN): LPLT saves {power_savings:.1f}% energy")
        print(f"   at the cost of {median_penalty:.1f}% slower median response times")
    elif power_savings > 0 and median_penalty < 0:
        print(f"🎉 UNEXPECTED RESULT: LPLT saves {power_savings:.1f}% energy")
        print(f"   AND improves median response times by {abs(median_penalty):.1f}%!")
        print("   This suggests LPLT strategy is superior in both dimensions")
    else:
        print("❌ HYPOTHESIS NOT CONFIRMED")

    # Return values for plotting (use median for more stable plot)
    return {
        'k8s_avg_power': k8s_avg_power,
        'lplt_avg_power': lplt_avg_power,
        'k8s_response_time': k8s_response_median,  # Use median
        'lplt_response_time': lplt_response_median,  # Use median
        'power_savings_percent': power_savings,
        'performance_penalty_percent': median_penalty
    }


def create_simple_trade_off_plot(k8s_avg_power, lplt_avg_power, k8s_response_time, lplt_response_time):
    """One chart that shows your research contribution"""

    strategies = ['Kubernetes', 'LPLT']
    power_consumption = [k8s_avg_power, lplt_avg_power]
    response_times = [k8s_response_time, lplt_response_time]

    plt.figure(figsize=(10, 6))
    plt.scatter(response_times, power_consumption, s=200, alpha=0.7, 
                c=['blue', 'green'], label=strategies)

    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (response_times[i], power_consumption[i]),
                    xytext=(10, 10), textcoords='offset points')

    plt.xlabel('Median Response Time (seconds)')
    plt.ylabel('Average Power Consumption (W)')
    plt.title('Power-Performance Trade-off: LPLT vs Kubernetes (Robust Metrics)')
    plt.grid(True, alpha=0.3)

    # Show the trade-off direction
    if len(response_times) >= 2:
        plt.arrow(response_times[0], power_consumption[0],
                  response_times[1] - response_times[0],
                  power_consumption[1] - power_consumption[0],
                  head_width=1, head_length=0.05, fc='red', alpha=0.5, length_includes_head=True)

    plt.tight_layout()
    plt.savefig("tradeoff_plot.png", dpi=300, bbox_inches='tight')
    # Remove plt.show() since we're running headless
    print("Plot saved as tradeoff_plot.png")


def analyze_scaling_behavior():
    """Understand the unexpected performance differences"""
    
    # Load scaling action logs
    k8s_scaling = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/scaling_actions.csv")
    lplt_scaling = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/scaling_actions.csv")
    
    print("=== SCALING BEHAVIOR ANALYSIS ===")
    print(f"K8s total scaling actions: {len(k8s_scaling)}")
    print(f"LPLT total scaling actions: {len(lplt_scaling)}")
    
    # Node distribution analysis
    k8s_deployments = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/deployments_df.csv")
    lplt_deployments = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/deployments_df.csv")
    
    print(f"K8s average replicas: {k8s_deployments['replicas'].mean():.1f}")
    print(f"LPLT average replicas: {lplt_deployments['replicas'].mean():.1f}")
    
    # Check if LPLT is actually more conservative
    return {
        'k8s_scaling_frequency': len(k8s_scaling),
        'lplt_scaling_frequency': len(lplt_scaling),
        'scaling_difference': len(k8s_scaling) - len(lplt_scaling)
    }


def analyze_node_distribution():
    """Check if K8s is spreading too much vs LPLT consolidation"""
    
    # Load replica deployment data
    k8s_replicas = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/replica_deployment_df.csv")
    lplt_replicas = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/replica_deployment_df.csv")
    
    print("=== NODE DISTRIBUTION ANALYSIS ===")
    
    # Count unique nodes used
    k8s_unique_nodes = k8s_replicas['node_name'].nunique()
    lplt_unique_nodes = lplt_replicas['node_name'].nunique()
    
    print(f"K8s uses {k8s_unique_nodes} unique nodes")
    print(f"LPLT uses {lplt_unique_nodes} unique nodes")
    print(f"Node spreading difference: {k8s_unique_nodes - lplt_unique_nodes}")
    
    # Replicas per node distribution
    k8s_node_load = k8s_replicas.groupby('node_name').size()
    lplt_node_load = lplt_replicas.groupby('node_name').size()
    
    print(f"\nK8s node load distribution:")
    print(f"  Average replicas per node: {k8s_node_load.mean():.1f}")
    print(f"  Max replicas on one node: {k8s_node_load.max()}")
    print(f"  Min replicas on one node: {k8s_node_load.min()}")
    
    print(f"\nLPLT node load distribution:")
    print(f"  Average replicas per node: {lplt_node_load.mean():.1f}")
    print(f"  Max replicas on one node: {lplt_node_load.max()}")
    print(f"  Min replicas on one node: {lplt_node_load.min()}")
    
    # Check if LPLT is consolidating (higher load per node)
    consolidation_factor = lplt_node_load.mean() / k8s_node_load.mean()
    print(f"\nConsolidation factor: {consolidation_factor:.2f}")
    if consolidation_factor > 1.2:
        print("✅ LPLT is consolidating workload (higher density per node)")
    elif consolidation_factor < 0.8:
        print("❌ LPLT is spreading more than K8s")
    else:
        print("➡️ Similar distribution patterns")
    
    return {
        'k8s_nodes': k8s_unique_nodes,
        'lplt_nodes': lplt_unique_nodes,
        'consolidation_factor': consolidation_factor
    }


def analyze_cold_start_behavior():
    """Check if K8s has more cold start overhead"""
    
    # Load invocation data with timing details
    k8s_invocations = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/invocations_df.csv")
    lplt_invocations = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/invocations_df.csv")
    
    # Load scheduling data to see deployment patterns
    k8s_schedule = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/schedule_df.csv")
    lplt_schedule = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/schedule_df.csv")
    
    print("=== COLD START ANALYSIS ===")
    
    # Count successful vs failed schedules
    k8s_success_rate = k8s_schedule['successful'].mean() if 'successful' in k8s_schedule.columns else "N/A"
    lplt_success_rate = lplt_schedule['successful'].mean() if 'successful' in lplt_schedule.columns else "N/A"
    
    print(f"K8s scheduling success rate: {k8s_success_rate}")
    print(f"LPLT scheduling success rate: {lplt_success_rate}")
    
    # Analyze execution times (t_exec) for cold starts
    # First 10% of invocations are likely cold starts
    k8s_early = k8s_invocations.head(int(len(k8s_invocations) * 0.1))
    lplt_early = lplt_invocations.head(int(len(lplt_invocations) * 0.1))
    
    k8s_cold_exec = k8s_early['t_exec'].median()
    lplt_cold_exec = lplt_early['t_exec'].median()
    
    print(f"\nCold start execution times:")
    print(f"K8s early median t_exec: {k8s_cold_exec:.3f}s")
    print(f"LPLT early median t_exec: {lplt_cold_exec:.3f}s")
    
    if lplt_cold_exec < k8s_cold_exec:
        improvement = ((k8s_cold_exec - lplt_cold_exec) / k8s_cold_exec) * 100
        print(f"✅ LPLT has {improvement:.1f}% faster cold starts")
    
    # Count total scheduling events (more events = more overhead)
    print(f"\nTotal scheduling events:")
    print(f"K8s: {len(k8s_schedule)} scheduling events") 
    print(f"LPLT: {len(lplt_schedule)} scheduling events")
    
    return {
        'k8s_cold_exec': k8s_cold_exec,
        'lplt_cold_exec': lplt_cold_exec,
        'k8s_schedule_events': len(k8s_schedule),
        'lplt_schedule_events': len(lplt_schedule)
    }

def analyze_workload_differences():
    """Check which functions show the biggest differences"""
    
    k8s_invocations = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/invocations_df.csv")
    lplt_invocations = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/invocations_df.csv")
    
    print("=== WORKLOAD-SPECIFIC PERFORMANCE ===")
    
    # Calculate response time = t_wait + t_exec
    k8s_invocations['response_time'] = k8s_invocations['t_wait'] + k8s_invocations['t_exec']
    lplt_invocations['response_time'] = lplt_invocations['t_wait'] + lplt_invocations['t_exec']
    
    workload_results = {}
    
    for workload in ['resnet50-inference', 'fio', 'speech-inference', 'python-pi', 'resnet50-training']:
        # Filter by function name (partial match)
        k8s_workload = k8s_invocations[k8s_invocations['function_name'].str.contains(workload, na=False)]
        lplt_workload = lplt_invocations[lplt_invocations['function_name'].str.contains(workload, na=False)]
        
        if len(k8s_workload) > 10 and len(lplt_workload) > 10:  # Enough samples
            k8s_median = k8s_workload['response_time'].median()
            lplt_median = lplt_workload['response_time'].median()
            
            k8s_p95 = k8s_workload['response_time'].quantile(0.95)
            lplt_p95 = lplt_workload['response_time'].quantile(0.95)
            
            median_diff = ((lplt_median - k8s_median) / k8s_median) * 100
            p95_diff = ((lplt_p95 - k8s_p95) / k8s_p95) * 100
            
            print(f"\n{workload} ({len(k8s_workload)} vs {len(lplt_workload)} samples):")
            print(f"  Median: K8s={k8s_median:.3f}s, LPLT={lplt_median:.3f}s ({median_diff:+.1f}%)")
            print(f"  P95: K8s={k8s_p95:.3f}s, LPLT={lplt_p95:.3f}s ({p95_diff:+.1f}%)")
            
            workload_results[workload] = {
                'median_diff_pct': median_diff,
                'p95_diff_pct': p95_diff,
                'k8s_samples': len(k8s_workload),
                'lplt_samples': len(lplt_workload)
            }
    
    return workload_results


def analyze_scaling_decisions():
    """Deep dive into scaling decision patterns"""
    
    k8s_decisions = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/scaling_decisions_df.csv")
    lplt_decisions = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/scaling_decisions_df.csv")
    
    print("=== SCALING DECISION ANALYSIS ===")
    
    # Count scaling actions by type
    k8s_scale_up = len(k8s_decisions[k8s_decisions['action'] == 'scale_up'])
    k8s_scale_down = len(k8s_decisions[k8s_decisions['action'] == 'scale_down'])
    k8s_no_action = len(k8s_decisions[k8s_decisions['action'] == 'no_action'])
    
    lplt_scale_up = len(lplt_decisions[lplt_decisions['action'] == 'scale_up'])
    lplt_scale_down = len(lplt_decisions[lplt_decisions['action'] == 'scale_down'])
    lplt_no_action = len(lplt_decisions[lplt_decisions['action'] == 'no_action'])
    
    print(f"K8s scaling actions:")
    print(f"  Scale up: {k8s_scale_up}")
    print(f"  Scale down: {k8s_scale_down}")
    print(f"  No action: {k8s_no_action}")
    print(f"  Total actions: {k8s_scale_up + k8s_scale_down}")
    
    print(f"\nLPLT scaling actions:")
    print(f"  Scale up: {lplt_scale_up}")
    print(f"  Scale down: {lplt_scale_down}")
    print(f"  No action: {lplt_no_action}")
    print(f"  Total actions: {lplt_scale_up + lplt_scale_down}")
    
    # Check response time triggers
    k8s_high_rt = k8s_decisions[k8s_decisions['avg_response_time_ms'] > 1000]
    lplt_high_rt = lplt_decisions[lplt_decisions['avg_response_time_ms'] > 1000]
    
    print(f"\nHigh response time events (>1s):")
    print(f"K8s: {len(k8s_high_rt)} events")
    print(f"LPLT: {len(lplt_high_rt)} events")
    
    # Node type preferences
    if 'selected_node_type' in k8s_decisions.columns:
        k8s_node_types = k8s_decisions['selected_node_type'].value_counts()
        lplt_node_types = lplt_decisions['selected_node_type'].value_counts()
        
        print(f"\nNode type selection frequency:")
        print("K8s preferences:")
        for node_type, count in k8s_node_types.head().items():
            print(f"  {node_type}: {count}")
        
        print("LPLT preferences:")
        for node_type, count in lplt_node_types.head().items():
            print(f"  {node_type}: {count}")
    
    return {
        'k8s_total_actions': k8s_scale_up + k8s_scale_down,
        'lplt_total_actions': lplt_scale_up + lplt_scale_down,
        'k8s_high_rt_events': len(k8s_high_rt),
        'lplt_high_rt_events': len(lplt_high_rt)
    }


def analyze_resource_contention():
    """Check if K8s creates resource contention"""
    
    k8s_power = pd.read_csv(f"{K8S_EXPERIMENT_DIR}/power_df.csv")
    lplt_power = pd.read_csv(f"{LPLT_EXPERIMENT_DIR}/power_df.csv")
    
    print("=== RESOURCE CONTENTION ANALYSIS ===")
    
    # Average utilization by node type
    k8s_util_by_type = k8s_power.groupby('node_type')[['cpu_util', 'memory_util']].mean()
    lplt_util_by_type = lplt_power.groupby('node_type')[['cpu_util', 'memory_util']].mean()
    
    print("K8s average utilization by node type:")
    print(k8s_util_by_type)
    
    print("\nLPLT average utilization by node type:")
    print(lplt_util_by_type)
    
    # Check for nodes with very high utilization (>90%)
    k8s_high_util = k8s_power[k8s_power['cpu_util'] > 0.9]
    lplt_high_util = lplt_power[lplt_power['cpu_util'] > 0.9]
    
    print(f"\nHigh CPU utilization events (>90%):")
    print(f"K8s: {len(k8s_high_util)} events")
    print(f"LPLT: {len(lplt_high_util)} events")
    
    # Average power efficiency (requests per watt approximation)
    k8s_avg_power_per_node = k8s_power.groupby('node')['power_watts'].mean().mean()
    lplt_avg_power_per_node = lplt_power.groupby('node')['power_watts'].mean().mean()
    
    print(f"\nAverage power per active node:")
    print(f"K8s: {k8s_avg_power_per_node:.2f}W")
    print(f"LPLT: {lplt_avg_power_per_node:.2f}W")
    
    return {
        'k8s_high_util_events': len(k8s_high_util),
        'lplt_high_util_events': len(lplt_high_util),
        'k8s_avg_power_per_node': k8s_avg_power_per_node,
        'lplt_avg_power_per_node': lplt_avg_power_per_node
    }

def run_deep_analysis():
    """Run all deep analysis functions to understand the anomalies"""
    
    print("🔍 DEEP ANALYSIS: Understanding Why LPLT Outperforms K8s\n")
    
    # 1. Node distribution
    node_results = analyze_node_distribution()
    
    # 2. Cold start behavior  
    cold_start_results = analyze_cold_start_behavior()
    
    # 3. Workload-specific differences
    workload_results = analyze_workload_differences()
    
    # 4. Scaling decision patterns
    scaling_results = analyze_scaling_decisions()
    
    # 5. Resource contention
    contention_results = analyze_resource_contention()
    
    print("\n" + "="*50)
    print("HYPOTHESIS VALIDATION SUMMARY")
    print("="*50)
    
    # Validate each hypothesis
    if node_results['consolidation_factor'] > 1.2:
        print("✅ H1: LPLT consolidates workload better than K8s")
    
    if cold_start_results['lplt_cold_exec'] < cold_start_results['k8s_cold_exec']:
        print("✅ H2: LPLT has faster cold start performance")
    
    if scaling_results['lplt_total_actions'] < scaling_results['k8s_total_actions']:
        print("✅ H3: LPLT scales more conservatively (less churn)")
    
    if contention_results['lplt_high_util_events'] < contention_results['k8s_high_util_events']:
        print("✅ H4: LPLT avoids resource contention better")
    
    return {
        'node_analysis': node_results,
        'cold_start_analysis': cold_start_results,
        'workload_analysis': workload_results,
        'scaling_analysis': scaling_results,
        'contention_analysis': contention_results
    }

# Add to main execution
if __name__ == "__main__":
    print("=== RUNNING HYPOTHESIS TEST ===")
    print("Hypothesis: LPLT reduces power consumption vs K8s baseline, but increases response times\n")
    
    results = compare_experiments()
    
    # NEW: Run deep analysis
    deep_results = run_deep_analysis()
    
    print(f"\n=== SUMMARY ===")
    print(f"Trade-off ratio: {results['power_savings_percent']:.1f}% energy savings for {results['performance_penalty_percent']:.1f}% performance cost")
    
    create_simple_trade_off_plot(
        results['k8s_avg_power'],
        results['lplt_avg_power'],
        results['k8s_response_time'],
        results['lplt_response_time']
    )