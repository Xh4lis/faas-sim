import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Experiment configuration
STRATEGY_1_NAME = "HPST"
STRATEGY_1_DIR = "./data/sine_perf_strategy_d120_r35"

STRATEGY_2_NAME = "LPLT"
STRATEGY_2_DIR = "./data/sine_power_strategy_d120_r35"


def compare_experiments():
    """Simple comparison of two experiment results using robust metrics"""

    # Load the CSV files from both experiments - CONFIGURABLE PATHS
    strategy1_power = pd.read_csv(f"{STRATEGY_1_DIR}/power_df.csv")
    strategy2_power = pd.read_csv(f"{STRATEGY_2_DIR}/power_df.csv")

    # Use autoscaler metrics for response times
    strategy1_metrics = pd.read_csv(f"{STRATEGY_1_DIR}/autoscaler_detailed_metrics_df.csv")
    strategy2_metrics = pd.read_csv(f"{STRATEGY_2_DIR}/autoscaler_detailed_metrics_df.csv")


    # Simple comparisons
    print("=== POWER CONSUMPTION COMPARISON ===")
        
    # 1. TOTAL SYSTEM POWER (Infrastructure Perspective)
    strategy1_total = strategy1_power.groupby('timestamp')['power_watts'].sum().mean()
    strategy2_total = strategy2_power.groupby('timestamp')['power_watts'].sum().mean()
    

    strategy1_replicas = pd.read_csv(f"{STRATEGY_1_DIR}/replica_deployment_df.csv")
    strategy2_replicas = pd.read_csv(f"{STRATEGY_2_DIR}/replica_deployment_df.csv")
    
    # 2. WORKLOAD EFFICIENCY (Energy per Unit of Work)  
    def calculate_workload_efficiency(power_df, replicas_df):
        workload_distribution = replicas_df.groupby('node_name').size()
        
        total_workload_power = 0
        total_workload_units = 0
        
        for node_name, replica_count in workload_distribution.items():
            # Try both 'node' and 'node_name' columns
            node_power_data = power_df[
                (power_df['node'] == node_name) if 'node' in power_df.columns 
                else (power_df['node_name'] == node_name)
            ]
            
            if len(node_power_data) > 0:
                avg_node_power = node_power_data['power_watts'].mean()
                workload_power = avg_node_power * replica_count
                total_workload_power += workload_power
                total_workload_units += replica_count
                
                print(f"  {node_name}: {replica_count} replicas × {avg_node_power:.2f}W = {workload_power:.1f}W")
        
        if total_workload_units == 0:
            print("  ⚠️ Warning: No workload units found!")
            return 0
            
        efficiency = total_workload_power / total_workload_units
        print(f"  Total: {total_workload_power:.1f}W across {total_workload_units} replicas = {efficiency:.2f}W/replica")
        
        return efficiency
    
    print(f"\n🔋 {STRATEGY_1_NAME} Workload Analysis:")
    s1_efficiency = calculate_workload_efficiency(strategy1_power, strategy1_replicas)
    
    print(f"\n🔋 {STRATEGY_2_NAME} Workload Analysis:")
    s2_efficiency = calculate_workload_efficiency(strategy2_power, strategy2_replicas)
    
    print(f"📊 TOTAL INFRASTRUCTURE POWER:")
    print(f"{STRATEGY_1_NAME} average total system power: {strategy1_total:.1f}W")
    print(f"{STRATEGY_2_NAME} average total system power: {strategy2_total:.1f}W")
    power_savings = ((strategy1_total - strategy2_total) / strategy1_total) * 100
    print(f"Infrastructure power savings: {power_savings:.1f}%")

    print(f"\n⚡ WORKLOAD ENERGY EFFICIENCY:")
    print(f"{STRATEGY_1_NAME}: {s1_efficiency:.2f}W per replica")
    print(f"{STRATEGY_2_NAME}: {s2_efficiency:.2f}W per replica") 
    
    # FIXED: Correct efficiency improvement calculation
    # Lower W/replica = better efficiency, so positive improvement when s2 < s1
    efficiency_improvement = ((s1_efficiency - s2_efficiency) / s1_efficiency) * 100
    print(f"Workload efficiency improvement: {efficiency_improvement:.1f}%")
    
    if efficiency_improvement > 0:
        print(f"✅ {STRATEGY_2_NAME} is more energy-efficient per replica")
    else:
        print(f"❌ {STRATEGY_1_NAME} is more energy-efficient per replica")
    
    print("\n=== PERFORMANCE COMPARISON ===")
    
    # Use MEDIAN instead of MEAN for response times
    strategy1_response_median = strategy1_metrics['avg_response_time'].median()
    strategy2_response_median = strategy2_metrics['avg_response_time'].median()
    median_penalty = ((strategy2_response_median - strategy1_response_median) / strategy1_response_median) * 100

    print(f"{STRATEGY_1_NAME} MEDIAN response time: {strategy1_response_median:.3f}s")
    print(f"{STRATEGY_2_NAME} MEDIAN response time: {strategy2_response_median:.3f}s")
    print(f"MEDIAN performance penalty: {median_penalty:.1f}%")

    # 95th percentile comparison (excludes extreme outliers)
    strategy1_p95 = strategy1_metrics['avg_response_time'].quantile(0.95)
    strategy2_p95 = strategy2_metrics['avg_response_time'].quantile(0.95)
    p95_penalty = ((strategy2_p95 - strategy1_p95) / strategy1_p95) * 100

    print(f"{STRATEGY_1_NAME} 95th percentile: {strategy1_p95:.3f}s")
    print(f"{STRATEGY_2_NAME} 95th percentile: {strategy2_p95:.3f}s")
    print(f"95th percentile penalty: {p95_penalty:.1f}%")

    # Exclude cold start period (first 10% of measurements)
    strategy1_warm = strategy1_metrics.iloc[int(len(strategy1_metrics) * 0.1):]
    strategy2_warm = strategy2_metrics.iloc[int(len(strategy2_metrics) * 0.1):]
    
    strategy1_warm_mean = strategy1_warm['avg_response_time'].mean()
    strategy2_warm_mean = strategy2_warm['avg_response_time'].mean()
    warm_penalty = ((strategy2_warm_mean - strategy1_warm_mean) / strategy1_warm_mean) * 100

    print(f"{STRATEGY_1_NAME} warm-up avg: {strategy1_warm_mean:.3f}s")
    print(f"{STRATEGY_2_NAME} warm-up avg: {strategy2_warm_mean:.3f}s")
    print(f"Warm-up performance penalty: {warm_penalty:.1f}%")

    print("\n=== WAIT TIME ANALYSIS ===")
    strategy1_wait_median = strategy1_metrics['avg_wait_time'].median()
    strategy2_wait_median = strategy2_metrics['avg_wait_time'].median()
    
    # Handle division by zero
    if strategy1_wait_median > 0:
        wait_improvement = ((strategy1_wait_median - strategy2_wait_median) / strategy1_wait_median) * 100
    else:
        wait_improvement = 0
    
    print(f"{STRATEGY_1_NAME} median wait time: {strategy1_wait_median:.3f}s")
    print(f"{STRATEGY_2_NAME} median wait time: {strategy2_wait_median:.3f}s")
    print(f"Wait time improvement: {wait_improvement:.1f}%")

    print("\n=== REVISED HYPOTHESIS RESULT ===")
    # Use median metrics for hypothesis testing
    if power_savings > 0 and median_penalty > 0:
        print(f"✅ HYPOTHESIS CONFIRMED (MEDIAN): {STRATEGY_2_NAME} saves {power_savings:.1f}% infrastructure power")
        print(f"   and {efficiency_improvement:.1f}% workload efficiency")
        print(f"   at the cost of {median_penalty:.1f}% slower median response times")
    elif power_savings > 0 and median_penalty < 0:
        print(f"🎉 UNEXPECTED RESULT: {STRATEGY_2_NAME} saves {power_savings:.1f}% infrastructure power")
        print(f"   and {efficiency_improvement:.1f}% workload efficiency")
        print(f"   AND improves median response times by {abs(median_penalty):.1f}%!")
        print(f"   This suggests {STRATEGY_2_NAME} strategy is superior in both dimensions")
    else:
        print("❌ HYPOTHESIS NOT CONFIRMED")
        print(f"   Power savings: {power_savings:.1f}%, Efficiency improvement: {efficiency_improvement:.1f}%")

    # Return values for plotting (use median for more stable plot)
    return {
        'strategy1_total': strategy1_total,
        'strategy2_total': strategy2_total,
        'strategy1_efficiency': s1_efficiency,
        'strategy2_efficiency': s2_efficiency,
        'strategy1_response_time': strategy1_response_median,  # Use median
        'strategy2_response_time': strategy2_response_median,  # Use median
        'power_savings_percent': power_savings,
        'efficiency_improvement_percent': efficiency_improvement,
        'performance_penalty_percent': median_penalty
    }


def create_simple_trade_off_plot(strategy1_total, strategy2_total, strategy1_response_time, strategy2_response_time):
    """One chart that shows your research contribution"""

    strategies = [STRATEGY_1_NAME, STRATEGY_2_NAME]
    power_consumption = [strategy1_total, strategy2_total]
    response_times = [strategy1_response_time, strategy2_response_time]

    plt.figure(figsize=(10, 6))
    plt.scatter(response_times, power_consumption, s=200, alpha=0.7, 
                c=['blue', 'green'], label=strategies)

    for i, strategy in enumerate(strategies):
        plt.annotate(strategy, (response_times[i], power_consumption[i]),
                    xytext=(10, 10), textcoords='offset points')

    plt.xlabel('Median Response Time (seconds)')
    plt.ylabel('Average Power Consumption (W)')
    plt.title(f'Power-Performance Trade-off: {STRATEGY_2_NAME} vs {STRATEGY_1_NAME} (Robust Metrics)')
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
    strategy1_scaling = pd.read_csv(f"{STRATEGY_1_DIR}/scaling_actions.csv")
    strategy2_scaling = pd.read_csv(f"{STRATEGY_2_DIR}/scaling_actions.csv")
    
    print("=== SCALING BEHAVIOR ANALYSIS ===")
    print(f"{STRATEGY_1_NAME} total scaling actions: {len(strategy1_scaling)}")
    print(f"{STRATEGY_2_NAME} total scaling actions: {len(strategy2_scaling)}")
    
    # Node distribution analysis
    strategy1_deployments = pd.read_csv(f"{STRATEGY_1_DIR}/deployments_df.csv")
    strategy2_deployments = pd.read_csv(f"{STRATEGY_2_DIR}/deployments_df.csv")
    
    print(f"{STRATEGY_1_NAME} average replicas: {strategy1_deployments['replicas'].mean():.1f}")
    print(f"{STRATEGY_2_NAME} average replicas: {strategy2_deployments['replicas'].mean():.1f}")
    
    # Check if Strategy 2 is actually more conservative
    return {
        'strategy1_scaling_frequency': len(strategy1_scaling),
        'strategy2_scaling_frequency': len(strategy2_scaling),
        'scaling_difference': len(strategy1_scaling) - len(strategy2_scaling)
    }


def analyze_node_distribution():
    """Check if Strategy 1 is spreading too much vs Strategy 2 consolidation"""
    
    # Load replica deployment data
    strategy1_replicas = pd.read_csv(f"{STRATEGY_1_DIR}/replica_deployment_df.csv")
    strategy2_replicas = pd.read_csv(f"{STRATEGY_2_DIR}/replica_deployment_df.csv")
    
    print("=== NODE DISTRIBUTION ANALYSIS ===")
    
    # Count unique nodes used
    strategy1_unique_nodes = strategy1_replicas['node_name'].nunique()
    strategy2_unique_nodes = strategy2_replicas['node_name'].nunique()
    
    print(f"{STRATEGY_1_NAME} uses {strategy1_unique_nodes} unique nodes")
    print(f"{STRATEGY_2_NAME} uses {strategy2_unique_nodes} unique nodes")
    print(f"Node spreading difference: {strategy1_unique_nodes - strategy2_unique_nodes}")
    
    # Replicas per node distribution
    strategy1_node_load = strategy1_replicas.groupby('node_name').size()
    strategy2_node_load = strategy2_replicas.groupby('node_name').size()
    
    print(f"\n{STRATEGY_1_NAME} node load distribution:")
    print(f"  Average replicas per node: {strategy1_node_load.mean():.1f}")
    print(f"  Max replicas on one node: {strategy1_node_load.max()}")
    print(f"  Min replicas on one node: {strategy1_node_load.min()}")
    
    print(f"\n{STRATEGY_2_NAME} node load distribution:")
    print(f"  Average replicas per node: {strategy2_node_load.mean():.1f}")
    print(f"  Max replicas on one node: {strategy2_node_load.max()}")
    print(f"  Min replicas on one node: {strategy2_node_load.min()}")
    
    # Check if Strategy 2 is consolidating (higher load per node)
    consolidation_factor = strategy2_node_load.mean() / strategy1_node_load.mean()
    print(f"\nConsolidation factor: {consolidation_factor:.2f}")
    if consolidation_factor > 1.2:
        print(f"✅ {STRATEGY_2_NAME} is consolidating workload (higher density per node)")
    elif consolidation_factor < 0.8:
        print(f"❌ {STRATEGY_2_NAME} is spreading more than {STRATEGY_1_NAME}")
    else:
        print("➡️ Similar distribution patterns")
    
    return {
        'strategy1_nodes': strategy1_unique_nodes,
        'strategy2_nodes': strategy2_unique_nodes,
        'consolidation_factor': consolidation_factor
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
    """Run all deep analysis functions to understand the anomalies"""
    
    print(f"🔍 DEEP ANALYSIS: Understanding Why {STRATEGY_2_NAME} Compares to {STRATEGY_1_NAME}\n")
    
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
        print(f"✅ H1: {STRATEGY_2_NAME} consolidates workload better than {STRATEGY_1_NAME}")
    
    if cold_start_results['strategy2_cold_exec'] < cold_start_results['strategy1_cold_exec']:
        print(f"✅ H2: {STRATEGY_2_NAME} has faster cold start performance")
    
    if scaling_results['strategy2_total_actions'] < scaling_results['strategy1_total_actions']:
        print(f"✅ H3: {STRATEGY_2_NAME} scales more conservatively (less churn)")
    
    if contention_results['strategy2_high_util_events'] < contention_results['strategy1_high_util_events']:
        print(f"✅ H4: {STRATEGY_2_NAME} avoids resource contention better")
    
    return {
        'node_analysis': node_results,
        'cold_start_analysis': cold_start_results,
        'workload_analysis': workload_results,
        'scaling_analysis': scaling_results,
        'contention_analysis': contention_results
    }


def validate_workload_analysis():
    """Validate the workload-weighted power analysis with detailed breakdown"""
    
    strategy1_power = pd.read_csv(f"{STRATEGY_1_DIR}/power_df.csv")
    strategy2_power = pd.read_csv(f"{STRATEGY_2_DIR}/power_df.csv")
    strategy1_replicas = pd.read_csv(f"{STRATEGY_1_DIR}/replica_deployment_df.csv")
    strategy2_replicas = pd.read_csv(f"{STRATEGY_2_DIR}/replica_deployment_df.csv")
    
    print("🔍 DETAILED WORKLOAD ANALYSIS VALIDATION")
    print("=" * 50)
    
    def extract_node_type(node_name):
        """Extract node type from node name"""
        node_name = node_name.lower()
        for node_type in ['xeongpu', 'xeoncpu', 'nuc', 'nx', 'tx2', 'rockpi', 'nano', 'coral', 'rpi4', 'rpi3']:
            if node_type in node_name:
                return node_type
        return 'unknown'
    
    def detailed_workload_analysis(power_df, replicas_df, strategy_name):
        print(f"\n📊 {strategy_name} Detailed Breakdown:")
        
        # Get workload distribution by node
        workload_by_node = replicas_df.groupby('node_name').size()
        
        # Get workload distribution by node type
        replicas_with_type = replicas_df.copy()
        replicas_with_type['node_type'] = replicas_with_type['node_name'].apply(extract_node_type)
        workload_by_type = replicas_with_type.groupby('node_type').size()
        
        print(f"Workload distribution by node type:")
        for node_type, count in workload_by_type.sort_values(ascending=False).items():
            avg_power = power_df[power_df['node_type'] == node_type]['power_watts'].mean()
            total_power = avg_power * count
            print(f"  {node_type}: {count} replicas × {avg_power:.2f}W = {total_power:.1f}W")
        
        # Calculate node-level analysis
        total_workload_power = 0
        total_replicas = 0
        
        node_column = 'node' if 'node' in power_df.columns else 'node_name'
        
        for node_name, replica_count in workload_by_node.items():
            node_power_data = power_df[power_df[node_column] == node_name]
            
            if len(node_power_data) > 0:
                avg_power = node_power_data['power_watts'].mean()
                node_total_power = avg_power * replica_count
                total_workload_power += node_total_power
                total_replicas += replica_count
        
        efficiency = total_workload_power / total_replicas if total_replicas > 0 else 0
        
        print(f"\n{strategy_name} Summary:")
        print(f"  Total replicas: {total_replicas}")
        print(f"  Total workload power: {total_workload_power:.1f}W")
        print(f"  Average efficiency: {efficiency:.3f}W per replica")
        print(f"  Unique nodes used: {workload_by_node.nunique()}")
        
        return efficiency, total_workload_power, total_replicas, workload_by_type
    
    # Analyze both strategies
    s1_eff, s1_power, s1_replicas, s1_types = detailed_workload_analysis(
        strategy1_power, strategy1_replicas, STRATEGY_1_NAME)
    s2_eff, s2_power, s2_replicas, s2_types = detailed_workload_analysis(
        strategy2_power, strategy2_replicas, STRATEGY_2_NAME)
    
    # Calculate improvements
    efficiency_improvement = ((s1_eff - s2_eff) / s1_eff) * 100
    power_reduction = ((s1_power - s2_power) / s1_power) * 100
    
    print(f"\n🎯 COMPARATIVE ANALYSIS:")
    print(f"Workload efficiency improvement: {efficiency_improvement:.1f}%")
    print(f"Total workload power reduction: {power_reduction:.1f}%")
    print(f"Replica count difference: {s1_replicas} vs {s2_replicas}")
    
    # Validate results
    print(f"\n✅ VALIDATION CHECKS:")
    if s2_eff < s1_eff:
        print(f"✓ {STRATEGY_2_NAME} has better energy efficiency per replica")
    else:
        print(f"✗ {STRATEGY_1_NAME} has better energy efficiency per replica")
    
    if s2_power < s1_power:
        print(f"✓ {STRATEGY_2_NAME} uses less total workload power")
    else:
        print(f"✗ {STRATEGY_1_NAME} uses less total workload power")
    
    return {
        'strategy1_efficiency': s1_eff,
        'strategy2_efficiency': s2_eff,
        'efficiency_improvement': efficiency_improvement,
        'power_reduction': power_reduction
    }
#     """Calculate power based on actual node usage patterns"""
    
#     hpst_power = pd.read_csv("./data/sine_perf_strategy_d120_r35/power_df.csv")
#     lplt_power = pd.read_csv("./data/sine_power_strategy_d120_r35/power_df.csv")
    
#     hpst_replicas = pd.read_csv("./data/sine_perf_strategy_d120_r35/replica_deployment_df.csv")
#     lplt_replicas = pd.read_csv("./data/sine_power_strategy_d120_r35/replica_deployment_df.csv")
    
#     print("=== CORRECTED POWER ANALYSIS ===")
    
#     # Calculate workload-weighted power consumption
#     def calculate_strategy_power(power_df, replicas_df, strategy_name):
#         # Get nodes with actual workload
#         active_nodes = replicas_df['node_name'].value_counts()
        
#         # Calculate power consumption weighted by workload
#         total_weighted_power = 0
#         total_workload = 0
        
#         for node_name, replica_count in active_nodes.items():
#             # Extract node type
#             node_type = 'unknown'
#             for nt in ['xeongpu', 'xeoncpu', 'nuc', 'nx', 'tx2', 'rockpi', 'nano', 'coral', 'rpi4', 'rpi3']:
#                 if nt in node_name.lower():
#                     node_type = nt
#                     break
            
#             # Get power consumption for this node type
#             node_power_data = power_df[power_df['node_type'] == node_type]
#             if len(node_power_data) > 0:
#                 avg_node_power = node_power_data['power_watts'].mean()
#                 weighted_power = avg_node_power * replica_count
#                 total_weighted_power += weighted_power
#                 total_workload += replica_count
                
#                 print(f"  {node_name} ({node_type}): {replica_count} replicas × {avg_node_power:.2f}W = {weighted_power:.1f}W")
        
#         avg_weighted_power = total_weighted_power / total_workload if total_workload > 0 else 0
        
#         print(f"\n{strategy_name} workload-weighted power: {avg_weighted_power:.2f}W per replica")
#         print(f"{strategy_name} total workload power: {total_weighted_power:.1f}W")
#         print(f"{strategy_name} total replicas: {total_workload}")
        
#         return avg_weighted_power, total_weighted_power, total_workload
    
#     # Calculate for both strategies
#     hpst_avg, hpst_total, hpst_replicas = calculate_strategy_power(hpst_power, hpst_replicas, "HPST")
#     lplt_avg, lplt_total, lplt_replicas = calculate_strategy_power(lplt_power, lplt_replicas, "LPLT")
    
#     # Calculate actual energy efficiency
#     power_per_replica_savings = ((hpst_avg - lplt_avg) / hpst_avg) * 100
#     total_power_savings = ((hpst_total - lplt_total) / hpst_total) * 100
    
#     print(f"\n=== CORRECTED ENERGY EFFICIENCY ===")
#     print(f"Power per replica savings: {power_per_replica_savings:.1f}%")
#     print(f"Total system power savings: {total_power_savings:.1f}%")
#     print(f"Replica count difference: {hpst_replicas} vs {lplt_replicas}")
    
#     return {
#         'hpst_power_per_replica': hpst_avg,
#         'lplt_power_per_replica': lplt_avg,
#         'power_per_replica_savings': power_per_replica_savings,
#         'total_power_savings': total_power_savings
#     }

# # Run this analysis
# corrected_results = calculate_correct_power_consumption()


# def analyze_node_selection_impact():
#     """Analyze the power impact of different node type selections"""
    
#     strategy1_decisions = pd.read_csv(f"{STRATEGY_1_DIR}/scaling_decisions_df.csv")
#     strategy2_decisions = pd.read_csv(f"{STRATEGY_2_DIR}/scaling_decisions_df.csv")

#     print("=== NODE SELECTION POWER IMPACT ===")
    
#     # Power costs by node type
#     node_power_costs = {
#         'rpi3': 1.4, 'rpi4': 2.9, 'rockpi': 3.2, 'coral': 2.4, 'nano': 1.9,
#         'tx2': 5.0, 'nx': 7.3, 'nuc': 6.0, 'xeoncpu': 45.0, 'xeongpu': 65.0
#     }
    
#     def calculate_selection_cost(decisions_df, strategy_name):
#         total_power_cost = 0
#         total_selections = 0
        
#         node_selections = decisions_df['selected_node_type'].value_counts()
        
#         print(f"\n{strategy_name} selections and power cost:")
#         for node_type, count in node_selections.items():
#             power_cost = node_power_costs.get(node_type, 5.0)
#             total_cost = power_cost * count
#             total_power_cost += total_cost
#             total_selections += count
            
#             print(f"  {node_type}: {count} selections × {power_cost}W = {total_cost:.1f}W")
        
#         avg_power_per_selection = total_power_cost / total_selections if total_selections > 0 else 0
#         print(f"{strategy_name} average power per selection: {avg_power_per_selection:.2f}W")
        
#         return avg_power_per_selection, total_power_cost, total_selections

#     s1_avg, s1_total, s1_count = calculate_selection_cost(strategy1_decisions, STRATEGY_1_NAME)
#     s2_avg, s2_total, s2_count = calculate_selection_cost(strategy2_decisions, STRATEGY_2_NAME)

#     selection_efficiency = ((s1_avg - s2_avg) / s1_avg) * 100
#     print(f"\nNode selection efficiency: {selection_efficiency:.1f}% power savings with {STRATEGY_1_NAME} vs {STRATEGY_2_NAME}")
    
#     return selection_efficiency

# # Run this too
# selection_savings = analyze_node_selection_impact()


# Add to main execution
if __name__ == "__main__":
    print("=== RUNNING HYPOTHESIS TEST ===")
    print(f"Hypothesis: {STRATEGY_2_NAME} reduces power consumption vs {STRATEGY_1_NAME} baseline, but increases response times\n")
    
    results = compare_experiments()
    
    # NEW: Run deep analysis
    deep_results = run_deep_analysis()
    
    # NEW: Validate workload analysis
    print("\n" + "="*60)
    validation_results = validate_workload_analysis()
    
    print(f"\n=== SUMMARY ===")
    print(f"Infrastructure power savings: {results['power_savings_percent']:.1f}%")
    print(f"Workload efficiency improvement: {results['efficiency_improvement_percent']:.1f}%")
    print(f"Performance trade-off: {results['performance_penalty_percent']:.1f}% response time penalty")
    
    create_simple_trade_off_plot(
        results['strategy1_total'],
        results['strategy2_total'],
        results['strategy1_response_time'],
        results['strategy2_response_time']
    )