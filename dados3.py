import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

# Load the training data
train_df = pd.read_csv('cars_train.csv', encoding='utf-16', sep='\t')
# Basic statistics
train_df.describe(include='all')
# Drop unnecessary columns
columns_to_drop = ['id', 'elegivel_revisao', 'veiculo_alienado']
train_df = train_df.drop(columns=columns_to_drop)

# Check the data after dropping
train_df.head()
# Plot histogram for price
plt.figure(figsize=(10, 6))
plt.hist(train_df['preco'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot histogram for year of manufacture
plt.figure(figsize=(10, 6))
plt.hist(train_df['ano_de_fabricacao'], bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Year of Manufacture')
plt.xlabel('Year of Manufacture')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# Plot bar plot for top 10 states
top_states = train_df['estado_vendedor'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
top_states.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 10 States')
plt.xlabel('State')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
# Plot bar plot for transmission types
transmission_counts = train_df['cambio'].value_counts()

plt.figure(figsize=(10, 6))
transmission_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Transmission Types')
plt.xlabel('Transmission Type')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
# Plot bar plot for top 10 brands
top_brands = train_df['marca'].value_counts().nlargest(10)

plt.figure(figsize=(12, 6))
top_brands.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Top 10 Car Brands')
plt.xlabel('Brand')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
# Calculate the mean price by brand
mean_price_by_brand = train_df.groupby('marca')['preco'].mean()

# Plot the mean price by brand
plt.figure(figsize=(10, 5))
sns.barplot(x=mean_price_by_brand.index, y=mean_price_by_brand.values)
plt.xticks(rotation=90)
plt.xlabel('Brand')
plt.ylabel('Mean Price')
plt.title('Mean Price by Brand')
plt.show()


# Hypothesis 1: Cars from popular brands are cheaper than those from other brands.

# Define popular brands
popular_brands = ['VOLKSWAGEN', 'CHEVROLET', 'FORD']

# Calculate average price for popular brands
avg_price_popular_brands = train_df[train_df['marca'].isin(popular_brands)]['preco'].mean()

# Calculate average price for other brands
avg_price_other_brands = train_df[~train_df['marca'].isin(popular_brands)]['preco'].mean()

avg_price_popular_brands, avg_price_other_brands


# Hypothesis 2: Cars with automatic transmission are more expensive than cars with other types of transmission.

# Calculate average price for cars with automatic transmission
avg_price_auto = train_df[train_df['cambio'] == 'Automática']['preco'].mean()

# Calculate average price for cars with other types of transmission
avg_price_other_trans = train_df[train_df['cambio'] != 'Automática']['preco'].mean()

avg_price_auto, avg_price_other_trans



# Hypothesis 3: Cars that are still under factory warranty are more expensive than those that are not.

# Calculate average price for cars under factory warranty
avg_price_warranty = train_df[train_df['garantia_de_fábrica'] == 'Garantia de fábrica']['preco'].mean()

# Calculate average price for cars not under factory warranty
avg_price_no_warranty = train_df[train_df['garantia_de_fábrica'] != 'Garantia de fábrica']['preco'].mean()

avg_price_warranty, avg_price_no_warranty

# Business Question 1: Which is the best state registered in the database to sell a popular brand car and why?

# Filter dataframe for cars from popular brands
popular_brands_df = train_df[train_df['marca'].isin(popular_brands)]

# Group by state and calculate average price
state_avg_price_popular_brands = popular_brands_df.groupby('estado_vendedor')['preco'].mean()

# Find the state with the highest average price
best_state_to_sell_popular_brand = state_avg_price_popular_brands.idxmax()

best_state_to_sell_popular_brand


# Business Question 2: Which is the best state to buy a pickup with automatic transmission and why?

# Filter dataframe for pickups with automatic transmission
auto_pickups_df = train_df[(train_df['cambio'] == 'Automática') & (train_df['tipo'] == 'Picape')]

# Group by state and calculate average price
state_avg_price_auto_pickups = auto_pickups_df.groupby('estado_vendedor')['preco'].mean()

# Find the state with the lowest average price
best_state_to_buy_auto_pickup = state_avg_price_auto_pickups.idxmin()

best_state_to_buy_auto_pickup

# Business Question 3: Which is the best state to buy cars that are still under factory warranty and why?

# Filter dataframe for cars under factory warranty
warranty_df = train_df[train_df['garantia_de_fábrica'] == 'Garantia de fábrica']

# Group by state and calculate average price
state_avg_price_warranty = warranty_df.groupby('estado_vendedor')['preco'].mean()

# Find the state with the lowest average price
best_state_to_buy_warranty = state_avg_price_warranty.idxmin()

best_state_to_buy_warranty


