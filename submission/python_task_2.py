import pandas as pd
import numpy as np

def calculate_distance_matrix(dataset_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(dataset_path)

    # Create a DataFrame with unique toll IDs as both index and columns
    unique_tolls = sorted(set(df['toll_booth_A'].tolist() + df['toll_booth_B'].tolist()))
    distance_matrix = pd.DataFrame(index=unique_tolls, columns=unique_tolls)

    # Initialize the distance matrix with zeros
    distance_matrix = distance_matrix.fillna(0)

    # Iterate through the rows of the DataFrame and update distances
    for _, row in df.iterrows():
        toll_A, toll_B, distance = row['toll_booth_A'], row['toll_booth_B'], row['distance']

        # Update distance in both directions (A to B and B to A)
        distance_matrix.loc[toll_A, toll_B] += distance
        distance_matrix.loc[toll_B, toll_A] += distance

    return distance_matrix

def unroll_distance_matrix(distance_matrix):
    # Get the lower triangular part of the distance matrix (excluding diagonal)
    lower_triangle = distance_matrix.where(np.tril(np.ones(distance_matrix.shape), k=-1).astype(bool))

    # Extract indices and values from the lower triangle
    indices = np.column_stack(np.where(~np.isnan(lower_triangle)))
    values = lower_triangle.dropna().values.flatten()

    # Create a DataFrame with columns 'id_start', 'id_end', and 'distance'
    unrolled_df = pd.DataFrame(indices, columns=['id_start', 'id_end'])
    unrolled_df['distance'] = values

    return unrolled_df


def find_ids_within_ten_percentage_threshold(unrolled_df, reference_value):
    # Filter the DataFrame for the specified reference value
    reference_df = unrolled_df[unrolled_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_df['distance'].mean()

    # Calculate the lower and upper bounds within 10% of the average
    lower_bound = average_distance - (average_distance * 0.1)
    upper_bound = average_distance + (average_distance * 0.1)

    # Filter the DataFrame for values within the 10% threshold
    within_threshold_df = unrolled_df[
        (unrolled_df['distance'] >= lower_bound) & (unrolled_df['distance'] <= upper_bound)]

    # Get the unique sorted list of 'id_start' values
    sorted_ids_within_threshold = sorted(within_threshold_df['id_start'].unique())

    return sorted_ids_within_threshold


def calculate_toll_rate(distance_matrix):
    # Define rate coefficients for different vehicle types
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create new columns for each vehicle type and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        distance_matrix[vehicle_type] = distance_matrix['distance'] * rate_coefficient

    return distance_matrix


from datetime import time

def calculate_time_based_toll_rates(df):
    # Define time ranges and discount factors
    time_ranges_weekdays = [(time(0, 0, 0), time(10, 0, 0)),
                            (time(10, 0, 0), time(18, 0, 0)),
                            (time(18, 0, 0), time(23, 59, 59))]

    time_ranges_weekends = [(time(0, 0, 0), time(23, 59, 59))]

    discount_factors_weekdays = [0.8, 1.2, 0.8]
    discount_factor_weekends = 0.7

    # Create new columns for start_day, start_time, end_day, and end_time
    df['start_day'] = df['start_timestamp'].dt.day_name()
    df['start_time'] = df['start_timestamp'].dt.time
    df['end_day'] = df['end_timestamp'].dt.day_name()
    df['end_time'] = df['end_timestamp'].dt.time

    # Calculate toll rates based on time intervals
    for time_range, discount_factor in zip(time_ranges_weekdays, discount_factors_weekdays):
        mask = (df['start_time'] >= time_range[0]) & (df['start_time'] < time_range[1])
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor

    for time_range in time_ranges_weekends:
        mask = (df['start_time'] >= time_range[0]) & (df['start_time'] < time_range[1])
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor_weekends

    return df

