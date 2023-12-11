import pandas as pd

def generate_car_matrix(dataset_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Create a pivot table with id_1 as index, id_2 as columns, and car as values
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Set diagonal values to 0
    car_matrix.values[[range(car_matrix.shape[0])]*2] = 0

    return car_matrix


def get_type_count(data_frame):
    # Define the conditions and corresponding categories
    conditions = [
        (data_frame['car'] <= 15),
        ((data_frame['car'] > 15) & (data_frame['car'] <= 25)),
        (data_frame['car'] > 25)
    ]

    categories = ['low', 'medium', 'high']

    # Create a new column 'car_type' based on conditions and categories
    data_frame['car_type'] = pd.cut(data_frame['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=categories)

    # Count occurrences of each car_type category
    type_counts = data_frame['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


def get_bus_indexes(data_frame):
    # Calculate the mean value of the 'bus' column
    mean_bus_value = data_frame['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = data_frame[data_frame['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    sorted_bus_indexes = sorted(bus_indexes)

    return sorted_bus_indexes


def filter_routes(data_frame):
    # Calculate the mean value of the 'truck' column for each route
    route_means = data_frame.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_means[route_means > 7].index.tolist()

    # Sort the list of routes in ascending order
    sorted_routes = sorted(selected_routes)

    return sorted_routes


def multiply_matrix(matrix):
    # Create a copy of the input matrix to avoid modifying the original matrix
    modified_matrix = matrix.copy()

    # Apply the specified logic to each value in the matrix
    for i in range(modified_matrix.shape[0]):
        for j in range(modified_matrix.shape[1]):
            if modified_matrix.iloc[i, j] > 20:
                modified_matrix.iloc[i, j] *= 0.75
            else:
                modified_matrix.iloc[i, j] *= 1.25

    # Round values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

