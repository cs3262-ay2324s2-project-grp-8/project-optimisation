import re
import pandas as pd
import matplotlib.pyplot as plt

data  = []

pattern = r"Playoff Iteration (\d+), Profit (-?\d+), Final Timestamp (\d+), Done\? (\w+)"

with open('train.log.txt', 'r') as file:
    for line in file:
        match = re.match(pattern, line)
        if match:
            iter, profit, timestamp, done = match.groups()
            data.append(
                {'Iteration': int(iter), 'Profit': int(profit), 'Timestamp': int(timestamp)}
            )

# print(data)

df = pd.DataFrame(data)
df.sort_values(by='Iteration', inplace=True)

GROUP_SIZE = 10

df['IterationGroup'] = df['Iteration'] // GROUP_SIZE
avg_profit_per_group = df.groupby('IterationGroup')['Profit'].mean().reset_index()
avg_profit_per_group['Iteration'] = avg_profit_per_group['IterationGroup'] * GROUP_SIZE

plt.plot(avg_profit_per_group['Iteration'], avg_profit_per_group['Profit'], linestyle='-')
plt.show()