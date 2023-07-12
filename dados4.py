import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset with UTF-16 encoding and tab delimiter
train_data = pd.read_csv('cars_train.csv', encoding='UTF-16', delimiter='\t')

# Calculate the value counts for the categorical variables
cat_vars = ["marca", "modelo", "versao", "cambio", "tipo", "blindado", "cor", "tipo_vendedor", "cidade_vendedor", "estado_vendedor", "anunciante"]
cat_counts = {var: train_data[var].value_counts().idxmax() for var in cat_vars}

# Calculate the descriptive statistics for the numerical variables
num_vars = ["num_fotos", "ano_de_fabricacao", "ano_modelo", "hodometro", "num_portas", "preco"]
num_stats = train_data[num_vars].describe()

# Format and print the results
print("Categorical Variables:")
for var, value in cat_counts.items():
    print(f"{var}: {value}")
print()

print("Numerical Variables:")
for var in num_vars:
    print(f"{var}:")
    print(f"  Mean: {num_stats.loc['mean', var]}")
    print(f"  Minimum: {num_stats.loc['min', var]}")
    print(f"  Maximum: {num_stats.loc['max', var]}")
print()

# Plot the histograms and boxplots
fig, axes = plt.subplots(nrows=len(num_vars), ncols=2, figsize=(10, 20))

for i, var in enumerate(num_vars):
    sns.histplot(data=train_data, x=var, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {var}')

    sns.boxplot(data=train_data, x=var, ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {var}')

plt.tight_layout()
plt.show()
