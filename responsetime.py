import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('detailed_metrics.csv')
# Plotting the distribution of wait percentage for each deployment
plt.figure(figsize=(14, 6))
sns.boxplot(x='deployment_name', y='wait_percentage', data=df)
plt.title('Distribution of Wait Percentage by Deployment')
plt.xlabel('Deployment Name')
plt.ylabel('Wait Percentage')
plt.savefig('wait_percentage_distribution.png')

# Plotting the high wait count for each deployment
plt.figure(figsize=(14, 6))
sns.barplot(x='deployment_name', y='high_wait_count', data=df, ci=None)
plt.title('High Wait Count per Deployment')
plt.xlabel('Deployment Name')
plt.ylabel('High Wait Count')
plt.savefig('high_wait_count_per_deployment.png')

# Plotting response time over time for all deployments
plt.figure(figsize=(14, 8))
for deployment in df['deployment_name'].unique():
    deployment_data = df[df['deployment_name'] == deployment]
    plt.plot(deployment_data['timestamp'], deployment_data['avg_response_time'], label=deployment)

plt.title('Response Time Over Time for All Deployments')
plt.xlabel('Timestamp')
plt.ylabel('Average Response Time')
plt.legend()
plt.savefig('response_time_over_time.png')