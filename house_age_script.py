import numpy as np
import Orange

# Get the index of the 'house_age' attribute
year_built_idx = in_data.domain.index('house_age')

# Extract the year built column as a NumPy array
year_built = in_data.X[:, year_built_idx]

# Compute the house age
house_age = 2025 - year_built

# Reshape to 2D column vector for concatenation
house_age = house_age.reshape(-1, 1)

# Create new variable
new_var = Orange.data.ContinuousVariable("house_years_old")

# Extend domain
new_domain = Orange.data.Domain(
    in_data.domain.attributes + (new_var,),
    in_data.domain.class_vars,
    in_data.domain.metas
)

# Concatenate new column to feature matrix
new_X = np.hstack((in_data.X, house_age))

# Build new table
out_data = Orange.data.Table(new_domain, new_X, in_data.Y, in_data.metas)
