import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Experiment configuration
STRATEGY_1_NAME = "Kubernetes"
STRATEGY_1_DIR = "./data/sine_k8s_strategy_d500_r145"

STRATEGY_2_NAME = "LPLT"
STRATEGY_2_DIR = "./data/sine_power_strategy_d500_r145"

STRATEGY_3_NAME = "HPST"
STRATEGY_3_DIR = "./data/sine_perf_strategy_d500_r145"


def compare_experiments():
    """Three-way comparison of experiment results using robust metrics"""

    # Load the CSV files from all three experiments
    strategy1_power = pd.read_csv(f"{STRATEGY_1_DIR}/power_df.csv")
    strategy2_power = pd.read_csv(f"{STRATEGY_2_DIR}/power_df.csv")
    strategy3_power = pd.read_csv(f"{STRATEGY_3_DIR}/power_df.csv")

    # Use autoscaler metrics for response times
    strategy1_metrics = pd.read_csv(f"{STRATEGY_1_DIR}/autoscaler_detailed_metrics_df.csv")
    strategy2_metrics = pd.read_csv(f"{STRATEGY_2_DIR}/autoscaler_detailed_metrics_df.csv")
    strategy3_metrics = pd.read_csv(f"{STRATEGY_3_DIR}/autoscaler_detailed_metrics_df.csv")

    # Simple comparisons
    print("=== POWER CONSUMPTION COMPARISON ===")
        
    # 1. TOTAL SYSTEM POWER (Infrastructure Perspective)
    strategy1_total = strategy1_power.groupby('timestamp')['power_watts'].sum().mean()
    strategy2_total = strategy2_power.groupby('timestamp')['power_watts'].sum().mean()
    strategy3_total = strategy3_power.groupby('timestamp')['power_watts'].sum().mean()
    
    print(f" TOTAL INFRASTRUCTURE POWER:")
    print(f"{STRATEGY_1_NAME} average total system power: {strategy1_total:.1f}W")
    print(f"{STRATEGY_2_NAME} average total system power: {strategy2_total:.1f}W")
    print(f"{STRATEGY_3_NAME} average total system power: {strategy3_total:.1f}W")
    
    # Power savings calculations (vs HPST baseline)
    power_savings_s2 = ((strategy1_total - strategy2_total) / strategy1_total) * 100
    power_savings_s3 = ((strategy1_total - strategy3_total) / strategy1_total) * 100
    
    print(f"\nPower efficiency vs {STRATEGY_1_NAME} baseline:")
    print(f"{STRATEGY_2_NAME} power savings: {power_savings_s2:.1f}% ({strategy2_total:.1f}W / {strategy1_total:.1f}W)")
    print(f"{STRATEGY_3_NAME} power savings: {power_savings_s3:.1f}% ({strategy3_total:.1f}W / {strategy1_total:.1f}W)")

    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Use MEDIAN instead of MEAN for response times
    strategy1_response_median = strategy1_metrics['avg_response_time'].median()
    strategy2_response_median = strategy2_metrics['avg_response_time'].median()
    strategy3_response_median = strategy3_metrics['avg_response_time'].median()
    
    print(f"MEDIAN response times:")
    print(f"{STRATEGY_1_NAME}: {strategy1_response_median:.3f}s")
    print(f"{STRATEGY_2_NAME}: {strategy2_response_median:.3f}s")
    print(f"{STRATEGY_3_NAME}: {strategy3_response_median:.3f}s")
    
    # Performance penalties (vs HPST baseline)
    median_penalty_s2 = ((strategy2_response_median - strategy1_response_median) / strategy1_response_median) * 100
    median_penalty_s3 = ((strategy3_response_median - strategy1_response_median) / strategy1_response_median) * 100
    
    print(f"\nPerformance penalty vs {STRATEGY_1_NAME} baseline:")
    print(f"{STRATEGY_2_NAME} penalty: {median_penalty_s2:.1f}% ({strategy2_response_median:.3f}s / {strategy1_response_median:.3f}s)")
    print(f"{STRATEGY_3_NAME} penalty: {median_penalty_s3:.1f}% ({strategy3_response_median:.3f}s / {strategy1_response_median:.3f}s)")

    # 95th percentile comparison
    strategy1_p95 = strategy1_metrics['avg_response_time'].quantile(0.95)
    strategy2_p95 = strategy2_metrics['avg_response_time'].quantile(0.95)
    strategy3_p95 = strategy3_metrics['avg_response_time'].quantile(0.95)
    
    p95_penalty_s2 = ((strategy2_p95 - strategy1_p95) / strategy1_p95) * 100
    p95_penalty_s3 = ((strategy3_p95 - strategy1_p95) / strategy1_p95) * 100

    print(f"\n95th percentile response times:")
    print(f"{STRATEGY_1_NAME}: {strategy1_p95:.3f}s")
    print(f"{STRATEGY_2_NAME}: {strategy2_p95:.3f}s ({p95_penalty_s2:+.1f}%)")
    print(f"{STRATEGY_3_NAME}: {strategy3_p95:.3f}s ({p95_penalty_s3:+.1f}%)")

    print("\n=== WAIT TIME ANALYSIS ===")
    strategy1_wait_median = strategy1_metrics['avg_wait_time'].median()
    strategy2_wait_median = strategy2_metrics['avg_wait_time'].median()
    strategy3_wait_median = strategy3_metrics['avg_wait_time'].median()
    
    print(f"MEDIAN wait times:")
    print(f"{STRATEGY_1_NAME}: {strategy1_wait_median:.3f}s")
    print(f"{STRATEGY_2_NAME}: {strategy2_wait_median:.3f}s")
    print(f"{STRATEGY_3_NAME}: {strategy3_wait_median:.3f}s")

    print("\n=== STRATEGY RANKING ===")
    # Rank strategies by power consumption (lower is better)
    power_rankings = sorted([
        (STRATEGY_1_NAME, strategy1_total),
        (STRATEGY_2_NAME, strategy2_total),
        (STRATEGY_3_NAME, strategy3_total)
    ], key=lambda x: x[1])
    
    print("Power consumption ranking (best to worst):")
    for i, (name, power) in enumerate(power_rankings, 1):
        print(f"  {i}. {name}: {power:.1f}W")
    
    # Rank strategies by response time (lower is better)
    response_rankings = sorted([
        (STRATEGY_1_NAME, strategy1_response_median),
        (STRATEGY_2_NAME, strategy2_response_median),
        (STRATEGY_3_NAME, strategy3_response_median)
    ], key=lambda x: x[1])
    
    print("\nResponse time ranking (best to worst):")
    for i, (name, response_time) in enumerate(response_rankings, 1):
        print(f"  {i}. {name}: {response_time:.3f}s")

    # Return values for plotting
    return {
        'strategy1_total': strategy1_total,
        'strategy2_total': strategy2_total,
        'strategy3_total': strategy3_total,
        'strategy1_response_time': strategy1_response_median,
        'strategy2_response_time': strategy2_response_median,
        'strategy3_response_time': strategy3_response_median,
        'power_savings_s2': power_savings_s2,
        'power_savings_s3': power_savings_s3,
        'performance_penalty_s2': median_penalty_s2,
        'performance_penalty_s3': median_penalty_s3
    }


def create_simple_trade_off_plot(strategy1_total, strategy2_total, strategy3_total, 
                                 strategy1_response_time, strategy2_response_time, strategy3_response_time):
    """Three-way chart that shows your research contribution"""

    strategies = [STRATEGY_1_NAME, STRATEGY_2_NAME, STRATEGY_3_NAME]
    power_consumption = [strategy1_total, strategy2_total, strategy3_total]
    response_times = [strategy1_response_time, strategy2_response_time, strategy3_response_time]

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red']
    plt.scatter(response_times, power_consumption, s=300, alpha=0.7, 
                c=colors, label=strategies)

    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (response_times[i], power_consumption[i]),
                    xytext=(15, 15), textcoords='offset points', fontsize=12, fontweight='bold')

    plt.xlabel('Median Response Time (seconds)', fontsize=12)
    plt.ylabel('Average Power Consumption (W)', fontsize=12)
    plt.title(f'Power-Performance Trade-off: Three-Strategy Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add arrows to show relationships
    # HPST to LPLT
    plt.arrow(response_times[0], power_consumption[0],
              response_times[1] - response_times[0],
              power_consumption[1] - power_consumption[0],
              head_width=0.5, head_length=1, fc='orange', alpha=0.6, length_includes_head=True,
              label=f'{STRATEGY_1_NAME}→{STRATEGY_2_NAME}')
    
    # HPST to Kubernetes
    plt.arrow(response_times[0], power_consumption[0],
              response_times[2] - response_times[0],
              power_consumption[2] - power_consumption[0],
              head_width=0.5, head_length=1, fc='purple', alpha=0.6, length_includes_head=True,
              label=f'{STRATEGY_1_NAME}→{STRATEGY_3_NAME}')

    plt.tight_layout()
    plt.savefig("tradeoff_plot_3way.png", dpi=300, bbox_inches='tight')
    print("Plot saved as tradeoff_plot_3way.png")


def analyze_scaling_behavior():
    """Three-way scaling behavior analysis"""
    
    # Load scaling decision logs for all strategies (using correct dataframe)
    strategy1_scaling = pd.read_csv(f"{STRATEGY_1_DIR}/scaling_decisions_df.csv")
    strategy2_scaling = pd.read_csv(f"{STRATEGY_2_DIR}/scaling_decisions_df.csv")
    strategy3_scaling = pd.read_csv(f"{STRATEGY_3_DIR}/scaling_decisions_df.csv")
    
    print("=== SCALING BEHAVIOR ANALYSIS ===")
    
    # Count actual scaling actions (exclude 'no_action')
    strategy1_actions = strategy1_scaling[strategy1_scaling['action'] != 'no_action']
    strategy2_actions = strategy2_scaling[strategy2_scaling['action'] != 'no_action']
    strategy3_actions = strategy3_scaling[strategy3_scaling['action'] != 'no_action']
    
    print(f"Total scaling actions:")
    print(f"{STRATEGY_1_NAME}: {len(strategy1_actions)} actions")
    print(f"{STRATEGY_2_NAME}: {len(strategy2_actions)} actions")
    print(f"{STRATEGY_3_NAME}: {len(strategy3_actions)} actions")
    
    # Node distribution analysis (use correct deployment dataframe)
    strategy1_deployments = pd.read_csv(f"{STRATEGY_1_DIR}/replica_deployment_df.csv")
    strategy2_deployments = pd.read_csv(f"{STRATEGY_2_DIR}/replica_deployment_df.csv")
    strategy3_deployments = pd.read_csv(f"{STRATEGY_3_DIR}/replica_deployment_df.csv")
    
    print(f"\nTotal replica deployments:")
    print(f"{STRATEGY_1_NAME}: {len(strategy1_deployments)} deployments")
    print(f"{STRATEGY_2_NAME}: {len(strategy2_deployments)} deployments")
    print(f"{STRATEGY_3_NAME}: {len(strategy3_deployments)} deployments")
    
    # Scaling frequency analysis (vs HPST baseline)
    scaling_ratio_s2 = len(strategy2_actions) / len(strategy1_actions) if len(strategy1_actions) > 0 else 0
    scaling_ratio_s3 = len(strategy3_actions) / len(strategy1_actions) if len(strategy1_actions) > 0 else 0
    
    print(f"\nScaling frequency vs {STRATEGY_1_NAME}:")
    print(f"{STRATEGY_2_NAME} scaling ratio: {scaling_ratio_s2:.2f} ({len(strategy2_actions)} / {len(strategy1_actions)})")
    print(f"{STRATEGY_3_NAME} scaling ratio: {scaling_ratio_s3:.2f} ({len(strategy3_actions)} / {len(strategy1_actions)})")
    
    # Scale up vs scale down analysis
    strategy1_scale_up = len(strategy1_actions[strategy1_actions['action'] == 'scale_up'])
    strategy1_scale_down = len(strategy1_actions[strategy1_actions['action'] == 'scale_down'])
    strategy2_scale_up = len(strategy2_actions[strategy2_actions['action'] == 'scale_up'])
    strategy2_scale_down = len(strategy2_actions[strategy2_actions['action'] == 'scale_down'])
    strategy3_scale_up = len(strategy3_actions[strategy3_actions['action'] == 'scale_up'])
    strategy3_scale_down = len(strategy3_actions[strategy3_actions['action'] == 'scale_down'])
    
    print(f"\nScaling direction breakdown:")
    print(f"{STRATEGY_1_NAME}: {strategy1_scale_up} up, {strategy1_scale_down} down")
    print(f"{STRATEGY_2_NAME}: {strategy2_scale_up} up, {strategy2_scale_down} down")
    print(f"{STRATEGY_3_NAME}: {strategy3_scale_up} up, {strategy3_scale_down} down")
    
    # Interpretation
    print(f"\nScaling behavior patterns:")
    if scaling_ratio_s2 < 0.8:
        print(f"✅ {STRATEGY_2_NAME} scales more conservatively (20%+ fewer actions)")
    elif scaling_ratio_s2 > 1.2:
        print(f"❌ {STRATEGY_2_NAME} scales more aggressively (20%+ more actions)")
    else:
        print(f"➡️ {STRATEGY_2_NAME} has similar scaling frequency to {STRATEGY_1_NAME}")
        
    if scaling_ratio_s3 < 0.8:
        print(f"✅ {STRATEGY_3_NAME} scales more conservatively (20%+ fewer actions)")
    elif scaling_ratio_s3 > 1.2:
        print(f"❌ {STRATEGY_3_NAME} scales more aggressively (20%+ more actions)")
    else:
        print(f"➡️ {STRATEGY_3_NAME} has similar scaling frequency to {STRATEGY_1_NAME}")
    
    return {
        'strategy1_scaling_frequency': len(strategy1_actions),
        'strategy2_scaling_frequency': len(strategy2_actions),
        'strategy3_scaling_frequency': len(strategy3_actions),
        'scaling_ratio_s2': scaling_ratio_s2,
        'scaling_ratio_s3': scaling_ratio_s3
    }


def analyze_node_distribution():
    """Three-way node distribution analysis"""
    
    # Load replica deployment data for all strategies
    strategy1_replicas = pd.read_csv(f"{STRATEGY_1_DIR}/replica_deployment_df.csv")
    strategy2_replicas = pd.read_csv(f"{STRATEGY_2_DIR}/replica_deployment_df.csv")
    strategy3_replicas = pd.read_csv(f"{STRATEGY_3_DIR}/replica_deployment_df.csv")
    
    print("=== NODE DISTRIBUTION ANALYSIS ===")
    
    # Count unique nodes used
    strategy1_unique_nodes = strategy1_replicas['node_name'].nunique()
    strategy2_unique_nodes = strategy2_replicas['node_name'].nunique()
    strategy3_unique_nodes = strategy3_replicas['node_name'].nunique()
    
    print(f"Unique nodes used:")
    print(f"{STRATEGY_1_NAME}: {strategy1_unique_nodes} nodes")
    print(f"{STRATEGY_2_NAME}: {strategy2_unique_nodes} nodes")
    print(f"{STRATEGY_3_NAME}: {strategy3_unique_nodes} nodes")
    
    # Replicas per node distribution
    strategy1_node_load = strategy1_replicas.groupby('node_name').size()
    strategy2_node_load = strategy2_replicas.groupby('node_name').size()
    strategy3_node_load = strategy3_replicas.groupby('node_name').size()
    
    print(f"\nNode load distribution:")
    print(f"{STRATEGY_1_NAME} - Avg: {strategy1_node_load.mean():.1f}, Max: {strategy1_node_load.max()}, Min: {strategy1_node_load.min()}")
    print(f"{STRATEGY_2_NAME} - Avg: {strategy2_node_load.mean():.1f}, Max: {strategy2_node_load.max()}, Min: {strategy2_node_load.min()}")
    print(f"{STRATEGY_3_NAME} - Avg: {strategy3_node_load.mean():.1f}, Max: {strategy3_node_load.max()}, Min: {strategy3_node_load.min()}")
    
    # Consolidation analysis (vs HPST baseline)
    consolidation_factor_s2 = strategy2_node_load.mean() / strategy1_node_load.mean()
    consolidation_factor_s3 = strategy3_node_load.mean() / strategy1_node_load.mean()
    
    print(f"\nConsolidation analysis (replicas per node vs {STRATEGY_1_NAME}):")
    print(f"{STRATEGY_2_NAME} consolidation factor: {consolidation_factor_s2:.2f} ({strategy2_node_load.mean():.1f} / {strategy1_node_load.mean():.1f})")
    print(f"{STRATEGY_3_NAME} consolidation factor: {consolidation_factor_s3:.2f} ({strategy3_node_load.mean():.1f} / {strategy1_node_load.mean():.1f})")
    
    # Interpretation
    print(f"\nConsolidation patterns:")
    if consolidation_factor_s2 > 1.2:
        print(f"✅ {STRATEGY_2_NAME} consolidates workload significantly (20%+ higher density)")
    elif consolidation_factor_s2 < 0.8:
        print(f"❌ {STRATEGY_2_NAME} spreads workload more than {STRATEGY_1_NAME} (20%+ lower density)")
    else:
        print(f"➡️ {STRATEGY_2_NAME} has similar consolidation to {STRATEGY_1_NAME}")
        
    if consolidation_factor_s3 > 1.2:
        print(f"✅ {STRATEGY_3_NAME} consolidates workload significantly (20%+ higher density)")
    elif consolidation_factor_s3 < 0.8:
        print(f"❌ {STRATEGY_3_NAME} spreads workload more than {STRATEGY_1_NAME} (20%+ lower density)")
    else:
        print(f"➡️ {STRATEGY_3_NAME} has similar consolidation to {STRATEGY_1_NAME}")
    
    return {
        'strategy1_nodes': strategy1_unique_nodes,
        'strategy2_nodes': strategy2_unique_nodes,
        'strategy3_nodes': strategy3_unique_nodes,
        'consolidation_factor_s2': consolidation_factor_s2,
        'consolidation_factor_s3': consolidation_factor_s3
    }


def analyze_cold_start_behavior():
    """Check if Strategy 1 has more cold start overhead"""
    
    # Load invocation data with timing details
    strategy1_invocations = pd.read_csv(f"{STRATEGY_1_DIR}/invocations_df.csv")
    strategy2_invocations = pd.read_csv(f"{STRATEGY_2_DIR}/invocations_df.csv")
    
    # Load scheduling data to see deployment patterns
    strategy1_schedule = pd.read_csv(f"{STRATEGY_1_DIR}/schedule_df.csv")
    strategy2_schedule = pd.read_csv(f"{STRATEGY_2_DIR}/schedule_df.csv")
    
    print("=== COLD START ANALYSIS ===")
    
    # Count successful vs failed schedules
    strategy1_success_rate = strategy1_schedule['successful'].mean() if 'successful' in strategy1_schedule.columns else "N/A"
    strategy2_success_rate = strategy2_schedule['successful'].mean() if 'successful' in strategy2_schedule.columns else "N/A"
    
    print(f"{STRATEGY_1_NAME} scheduling success rate: {strategy1_success_rate}")
    print(f"{STRATEGY_2_NAME} scheduling success rate: {strategy2_success_rate}")
    
    # Analyze execution times (t_exec) for cold starts
    # First 10% of invocations are likely cold starts
    strategy1_early = strategy1_invocations.head(int(len(strategy1_invocations) * 0.1))
    strategy2_early = strategy2_invocations.head(int(len(strategy2_invocations) * 0.1))
    
    strategy1_cold_exec = strategy1_early['t_exec'].median()
    strategy2_cold_exec = strategy2_early['t_exec'].median()
    
    print(f"\nCold start execution times:")
    print(f"{STRATEGY_1_NAME} early median t_exec: {strategy1_cold_exec:.3f}s")
    print(f"{STRATEGY_2_NAME} early median t_exec: {strategy2_cold_exec:.3f}s")
    
    if strategy2_cold_exec < strategy1_cold_exec:
        improvement = ((strategy1_cold_exec - strategy2_cold_exec) / strategy1_cold_exec) * 100
        print(f"✅ {STRATEGY_2_NAME} has {improvement:.1f}% faster cold starts")
    
    # Count total scheduling events (more events = more overhead)
    print(f"\nTotal scheduling events:")
    print(f"{STRATEGY_1_NAME}: {len(strategy1_schedule)} scheduling events") 
    print(f"{STRATEGY_2_NAME}: {len(strategy2_schedule)} scheduling events")
    
    return {
        'strategy1_cold_exec': strategy1_cold_exec,
        'strategy2_cold_exec': strategy2_cold_exec,
        'strategy1_schedule_events': len(strategy1_schedule),
        'strategy2_schedule_events': len(strategy2_schedule)
    }

def analyze_workload_differences():
    """Check which functions show the biggest differences"""
    
    strategy1_invocations = pd.read_csv(f"{STRATEGY_1_DIR}/invocations_df.csv")
    strategy2_invocations = pd.read_csv(f"{STRATEGY_2_DIR}/invocations_df.csv")
    
    print("=== WORKLOAD-SPECIFIC PERFORMANCE ===")
    
    # Calculate response time = t_wait + t_exec
    strategy1_invocations['response_time'] = strategy1_invocations['t_wait'] + strategy1_invocations['t_exec']
    strategy2_invocations['response_time'] = strategy2_invocations['t_wait'] + strategy2_invocations['t_exec']
    
    workload_results = {}
    
    for workload in ['resnet50-inference', 'fio', 'speech-inference', 'python-pi', 'resnet50-training']:
        # Filter by function name (partial match)
        strategy1_workload = strategy1_invocations[strategy1_invocations['function_name'].str.contains(workload, na=False)]
        strategy2_workload = strategy2_invocations[strategy2_invocations['function_name'].str.contains(workload, na=False)]
        
        if len(strategy1_workload) > 10 and len(strategy2_workload) > 10:  # Enough samples
            strategy1_median = strategy1_workload['response_time'].median()
            strategy2_median = strategy2_workload['response_time'].median()
            
            strategy1_p95 = strategy1_workload['response_time'].quantile(0.95)
            strategy2_p95 = strategy2_workload['response_time'].quantile(0.95)
            
            median_diff = ((strategy2_median - strategy1_median) / strategy1_median) * 100
            p95_diff = ((strategy2_p95 - strategy1_p95) / strategy1_p95) * 100
            
            print(f"\n{workload} ({len(strategy1_workload)} vs {len(strategy2_workload)} samples):")
            print(f"  Median: {STRATEGY_1_NAME}={strategy1_median:.3f}s, {STRATEGY_2_NAME}={strategy2_median:.3f}s ({median_diff:+.1f}%)")
            print(f"  P95: {STRATEGY_1_NAME}={strategy1_p95:.3f}s, {STRATEGY_2_NAME}={strategy2_p95:.3f}s ({p95_diff:+.1f}%)")
            
            workload_results[workload] = {
                'median_diff_pct': median_diff,
                'p95_diff_pct': p95_diff,
                'strategy1_samples': len(strategy1_workload),
                'strategy2_samples': len(strategy2_workload)
            }
    
    return workload_results


def analyze_scaling_decisions():
    """Deep dive into scaling decision patterns"""
    
    strategy1_decisions = pd.read_csv(f"{STRATEGY_1_DIR}/scaling_decisions_df.csv")
    strategy2_decisions = pd.read_csv(f"{STRATEGY_2_DIR}/scaling_decisions_df.csv")
    
    print("=== SCALING DECISION ANALYSIS ===")
    
    # Count scaling actions by type
    strategy1_scale_up = len(strategy1_decisions[strategy1_decisions['action'] == 'scale_up'])
    strategy1_scale_down = len(strategy1_decisions[strategy1_decisions['action'] == 'scale_down'])
    strategy1_no_action = len(strategy1_decisions[strategy1_decisions['action'] == 'no_action'])
    
    strategy2_scale_up = len(strategy2_decisions[strategy2_decisions['action'] == 'scale_up'])
    strategy2_scale_down = len(strategy2_decisions[strategy2_decisions['action'] == 'scale_down'])
    strategy2_no_action = len(strategy2_decisions[strategy2_decisions['action'] == 'no_action'])
    
    print(f"{STRATEGY_1_NAME} scaling actions:")
    print(f"  Scale up: {strategy1_scale_up}")
    print(f"  Scale down: {strategy1_scale_down}")
    print(f"  No action: {strategy1_no_action}")
    print(f"  Total actions: {strategy1_scale_up + strategy1_scale_down}")
    
    print(f"\n{STRATEGY_2_NAME} scaling actions:")
    print(f"  Scale up: {strategy2_scale_up}")
    print(f"  Scale down: {strategy2_scale_down}")
    print(f"  No action: {strategy2_no_action}")
    print(f"  Total actions: {strategy2_scale_up + strategy2_scale_down}")
    
    # Check response time triggers
    strategy1_high_rt = strategy1_decisions[strategy1_decisions['avg_response_time_ms'] > 1000]
    strategy2_high_rt = strategy2_decisions[strategy2_decisions['avg_response_time_ms'] > 1000]
    
    print(f"\nHigh response time events (>1s):")
    print(f"{STRATEGY_1_NAME}: {len(strategy1_high_rt)} events")
    print(f"{STRATEGY_2_NAME}: {len(strategy2_high_rt)} events")
    
    # Node type preferences
    if 'selected_node_type' in strategy1_decisions.columns:
        strategy1_node_types = strategy1_decisions['selected_node_type'].value_counts()
        strategy2_node_types = strategy2_decisions['selected_node_type'].value_counts()
        
        print(f"\nNode type selection frequency:")
        print(f"{STRATEGY_1_NAME} preferences:")
        for node_type, count in strategy1_node_types.head().items():
            print(f"  {node_type}: {count}")
        
        print(f"{STRATEGY_2_NAME} preferences:")
        for node_type, count in strategy2_node_types.head().items():
            print(f"  {node_type}: {count}")
    
    return {
        'strategy1_total_actions': strategy1_scale_up + strategy1_scale_down,
        'strategy2_total_actions': strategy2_scale_up + strategy2_scale_down,
        'strategy1_high_rt_events': len(strategy1_high_rt),
        'strategy2_high_rt_events': len(strategy2_high_rt)
    }


def analyze_resource_contention():
    """Check if Strategy 1 creates resource contention"""
    
    strategy1_power = pd.read_csv(f"{STRATEGY_1_DIR}/power_df.csv")
    strategy2_power = pd.read_csv(f"{STRATEGY_2_DIR}/power_df.csv")
    
    print("=== RESOURCE CONTENTION ANALYSIS ===")
    
    # Average utilization by node type
    strategy1_util_by_type = strategy1_power.groupby('node_type')[['cpu_util', 'memory_util']].mean()
    strategy2_util_by_type = strategy2_power.groupby('node_type')[['cpu_util', 'memory_util']].mean()
    
    print(f"{STRATEGY_1_NAME} average utilization by node type:")
    print(strategy1_util_by_type)
    
    print(f"\n{STRATEGY_2_NAME} average utilization by node type:")
    print(strategy2_util_by_type)
    
    # Check for nodes with very high utilization (>90%)
    strategy1_high_util = strategy1_power[strategy1_power['cpu_util'] > 0.9]
    strategy2_high_util = strategy2_power[strategy2_power['cpu_util'] > 0.9]
    
    print(f"\nHigh CPU utilization events (>90%):")
    print(f"{STRATEGY_1_NAME}: {len(strategy1_high_util)} events")
    print(f"{STRATEGY_2_NAME}: {len(strategy2_high_util)} events")
    
    # Average power efficiency (requests per watt approximation)
    strategy1_total_per_node = strategy1_power.groupby('node')['power_watts'].mean().mean()
    strategy2_total_per_node = strategy2_power.groupby('node')['power_watts'].mean().mean()
    
    print(f"\nAverage power per active node:")
    print(f"{STRATEGY_1_NAME}: {strategy1_total_per_node:.2f}W")
    print(f"{STRATEGY_2_NAME}: {strategy2_total_per_node:.2f}W")
    
    return {
        'strategy1_high_util_events': len(strategy1_high_util),
        'strategy2_high_util_events': len(strategy2_high_util),
        'strategy1_total_per_node': strategy1_total_per_node,
        'strategy2_total_per_node': strategy2_total_per_node
    }

def run_deep_analysis():
    """Run all deep analysis functions for three-way comparison"""
    
    print(f"🔍 DEEP ANALYSIS: Three-Strategy Comparison\n")
    
    # 1. Node distribution
    node_results = analyze_node_distribution()
    
    # 2. Scaling behavior  
    scaling_results = analyze_scaling_behavior()
    
    print("\n" + "="*50)
    print("THREE-WAY STRATEGY ANALYSIS SUMMARY")
    print("="*50)
    
    # Best strategy analysis
    print("📊 STRATEGY RANKINGS:")
    
    # Node efficiency ranking
    node_efficiency_scores = [
        (STRATEGY_1_NAME, node_results['strategy1_nodes'], "baseline"),
        (STRATEGY_2_NAME, node_results['strategy2_nodes'], f"{node_results['consolidation_factor_s2']:.2f}x"),
        (STRATEGY_3_NAME, node_results['strategy3_nodes'], f"{node_results['consolidation_factor_s3']:.2f}x")
    ]
    node_efficiency_scores.sort(key=lambda x: x[1])  # Sort by node count (fewer is better)
    
    print("\nNode efficiency (fewer nodes = better):")
    for i, (name, nodes, factor) in enumerate(node_efficiency_scores, 1):
        print(f"  {i}. {name}: {nodes} nodes (consolidation: {factor})")
    
    # Scaling stability ranking
    scaling_stability_scores = [
        (STRATEGY_1_NAME, scaling_results['strategy1_scaling_frequency'], "baseline"),
        (STRATEGY_2_NAME, scaling_results['strategy2_scaling_frequency'], f"{scaling_results['scaling_ratio_s2']:.2f}x"),
        (STRATEGY_3_NAME, scaling_results['strategy3_scaling_frequency'], f"{scaling_results['scaling_ratio_s3']:.2f}x")
    ]
    scaling_stability_scores.sort(key=lambda x: x[1])  # Sort by scaling frequency (fewer is better)
    
    print("\nScaling stability (fewer actions = better):")
    for i, (name, actions, ratio) in enumerate(scaling_stability_scores, 1):
        print(f"  {i}. {name}: {actions} actions (frequency: {ratio})")
    
    return {
        'node_analysis': node_results,
        'scaling_analysis': scaling_results
    }


def analyze_real_power_consumption():
    """Three-way power consumption analysis using actual measured data"""
    
    # Load real power measurements for all strategies
    strategy1_power = pd.read_csv(f"{STRATEGY_1_DIR}/power_df.csv")
    strategy2_power = pd.read_csv(f"{STRATEGY_2_DIR}/power_df.csv")
    strategy3_power = pd.read_csv(f"{STRATEGY_3_DIR}/power_df.csv")
    
    print("🔋 REAL INFRASTRUCTURE POWER ANALYSIS (THREE-WAY)")
    print("="*50)
    
    # DEBUG: Check data structure for all strategies
    print(f"Power data measurements:")
    print(f"{STRATEGY_1_NAME}: {len(strategy1_power)} measurements, {strategy1_power['node'].nunique()} unique nodes")
    print(f"{STRATEGY_2_NAME}: {len(strategy2_power)} measurements, {strategy2_power['node'].nunique()} unique nodes")
    print(f"{STRATEGY_3_NAME}: {len(strategy3_power)} measurements, {strategy3_power['node'].nunique()} unique nodes")
    
    # 1. CORRECT TOTAL SYSTEM POWER (aggregate by timestamp first)
    strategy1_total_power = strategy1_power.groupby('timestamp')['power_watts'].sum()
    strategy2_total_power = strategy2_power.groupby('timestamp')['power_watts'].sum()
    strategy3_total_power = strategy3_power.groupby('timestamp')['power_watts'].sum()
    
    strategy1_avg_total = strategy1_total_power.mean()
    strategy2_avg_total = strategy2_total_power.mean()
    strategy3_avg_total = strategy3_total_power.mean()
    
    print(f"\n📊 CORRECTED SYSTEM POWER:")
    print(f"{STRATEGY_1_NAME}: {strategy1_avg_total:.1f}W")
    print(f"{STRATEGY_2_NAME}: {strategy2_avg_total:.1f}W")
    print(f"{STRATEGY_3_NAME}: {strategy3_avg_total:.1f}W")
    
    # 2. INFRASTRUCTURE EFFICIENCY (vs HPST baseline)
    power_savings_s2 = ((strategy1_avg_total - strategy2_avg_total) / strategy1_avg_total) * 100
    power_savings_s3 = ((strategy1_avg_total - strategy3_avg_total) / strategy1_avg_total) * 100
    
    print(f"\nPower efficiency vs {STRATEGY_1_NAME}:")
    print(f"{STRATEGY_2_NAME} power savings: {power_savings_s2:.1f}% ({strategy2_avg_total:.1f}W / {strategy1_avg_total:.1f}W)")
    print(f"{STRATEGY_3_NAME} power savings: {power_savings_s3:.1f}% ({strategy3_avg_total:.1f}W / {strategy1_avg_total:.1f}W)")
    
    # 3. POWER DISTRIBUTION ANALYSIS
    print(f"\n📊 POWER DISTRIBUTION BY NODE TYPE:")
    
    # Helper function to analyze power by node type
    def analyze_power_by_type(power_df, strategy_name):
        by_type = power_df.groupby(['node_type', 'node'])['power_watts'].mean().groupby('node_type').agg(['mean', 'count'])
        print(f"\n{strategy_name} power distribution:")
        total_nodes = 0
        total_power = 0
        for node_type in by_type.index:
            avg_power = by_type.loc[node_type, 'mean']
            node_count = by_type.loc[node_type, 'count']
            type_total_power = avg_power * node_count
            total_nodes += node_count
            total_power += type_total_power
            print(f"  {node_type}: {node_count} nodes × {avg_power:.1f}W = {type_total_power:.1f}W")
        print(f"  Total: {total_nodes} nodes = {total_power:.1f}W")
        return by_type
    
    strategy1_by_type = analyze_power_by_type(strategy1_power, STRATEGY_1_NAME)
    strategy2_by_type = analyze_power_by_type(strategy2_power, STRATEGY_2_NAME)
    strategy3_by_type = analyze_power_by_type(strategy3_power, STRATEGY_3_NAME)
    
    # 4. ENERGY CONSUMPTION
    strategy1_duration_hours = (strategy1_power['timestamp'].max() - strategy1_power['timestamp'].min()) / 3600
    strategy2_duration_hours = (strategy2_power['timestamp'].max() - strategy2_power['timestamp'].min()) / 3600
    strategy3_duration_hours = (strategy3_power['timestamp'].max() - strategy3_power['timestamp'].min()) / 3600
    
    strategy1_energy = strategy1_avg_total * strategy1_duration_hours
    strategy2_energy = strategy2_avg_total * strategy2_duration_hours
    strategy3_energy = strategy3_avg_total * strategy3_duration_hours
    
    print(f"\n⚡ TOTAL ENERGY CONSUMPTION:")
    print(f"Simulation duration: ~{strategy1_duration_hours:.2f} hours")
    print(f"{STRATEGY_1_NAME}: {strategy1_energy:.1f} Wh")
    print(f"{STRATEGY_2_NAME}: {strategy2_energy:.1f} Wh")
    print(f"{STRATEGY_3_NAME}: {strategy3_energy:.1f} Wh")
    
    energy_savings_s2 = ((strategy1_energy - strategy2_energy) / strategy1_energy) * 100
    energy_savings_s3 = ((strategy1_energy - strategy3_energy) / strategy1_energy) * 100
    
    print(f"\nEnergy efficiency vs {STRATEGY_1_NAME}:")
    print(f"{STRATEGY_2_NAME} energy savings: {energy_savings_s2:.1f}%")
    print(f"{STRATEGY_3_NAME} energy savings: {energy_savings_s3:.1f}%")
    
    # 5. POWER RANKING
    power_rankings = sorted([
        (STRATEGY_1_NAME, strategy1_avg_total),
        (STRATEGY_2_NAME, strategy2_avg_total),
        (STRATEGY_3_NAME, strategy3_avg_total)
    ], key=lambda x: x[1])
    
    print(f"\n🏆 POWER CONSUMPTION RANKING (best to worst):")
    for i, (name, power) in enumerate(power_rankings, 1):
        print(f"  {i}. {name}: {power:.1f}W")
    
    return {
        'strategy1_avg_power': strategy1_avg_total,
        'strategy2_avg_power': strategy2_avg_total,
        'strategy3_avg_power': strategy3_avg_total,
        'power_savings_s2': power_savings_s2,
        'power_savings_s3': power_savings_s3,
        'strategy1_total_energy': strategy1_energy,
        'strategy2_total_energy': strategy2_energy,
        'strategy3_total_energy': strategy3_energy,
        'energy_savings_s2': energy_savings_s2,
        'energy_savings_s3': energy_savings_s3
    }


# Add to main execution
if __name__ == "__main__":
    print("=== RUNNING THREE-WAY STRATEGY COMPARISON ===")
    print(f"Comparing: {STRATEGY_1_NAME} vs {STRATEGY_2_NAME} vs {STRATEGY_3_NAME}\n")
    
    results = compare_experiments()
    
    # Run real power consumption analysis
    print("\n" + "="*60)
    real_power_results = analyze_real_power_consumption()
    
    # Run deep analysis
    print("\n" + "="*60)
    deep_results = run_deep_analysis()
    
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Three-strategy comparison complete:")
    print(f"• Power analysis: {STRATEGY_1_NAME} baseline vs {STRATEGY_2_NAME} vs {STRATEGY_3_NAME}")
    print(f"• Performance analysis: Response time and scaling behavior")
    print(f"• Infrastructure analysis: Node distribution and resource usage")
    
    # Create the three-way plot
    create_simple_trade_off_plot(
        results['strategy1_total'],
        results['strategy2_total'],
        results['strategy3_total'],
        results['strategy1_response_time'],
        results['strategy2_response_time'],
        results['strategy3_response_time']
    )


