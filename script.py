import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_experiments():
    """Simple comparison of two experiment results using robust metrics"""

    # Load the CSV files from both experiments - FIXED PATHS
    k8s_power = pd.read_csv("./data/experiment_1_kubernetes_baseline/power_df.csv")
    lplt_power = pd.read_csv("./data/experiment_2_power_strategy/power_df.csv")

    # Use autoscaler metrics for response times
    k8s_metrics = pd.read_csv("./data/experiment_1_kubernetes_baseline/autoscaler_detailed_metrics_df.csv")
    lplt_metrics = pd.read_csv("./data/experiment_2_power_strategy/autoscaler_detailed_metrics_df.csv")

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
    wait_improvement = ((k8s_wait_median - lplt_wait_median) / k8s_wait_median) * 100

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
    plt.show()
    print("Plot saved as tradeoff_plot.png")


if __name__ == "__main__":
    print("=== RUNNING HYPOTHESIS TEST (ROBUST METRICS) ===")
    print("Hypothesis: LPLT reduces power consumption vs K8s baseline, but increases response times\n")
    
    results = compare_experiments()
    
    print(f"\n=== SUMMARY ===")
    print(f"Trade-off ratio: {results['power_savings_percent']:.1f}% energy savings for {results['performance_penalty_percent']:.1f}% performance cost")
    
    create_simple_trade_off_plot(
        results['k8s_avg_power'],
        results['lplt_avg_power'],
        results['k8s_response_time'],
        results['lplt_response_time']
    )