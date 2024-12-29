# main.py
from config import Config
from Bbu_pool import BBUPool
from Rrh import RRH
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from math import radians, cos, sin, sqrt, atan2
from sklearn.cluster import KMeans
import numpy as np
import random
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Plotting functions

def plot_multiplexing_gain_across_scenarios(bbu_pools_by_scenario, scenario_labels, config):
    """
    Plot the multiplexing gain for each BBU pool across multiple scenarios in the same hour.
    :param bbu_pools_by_scenario: List of BBU pools for each scenario.
    :param scenario_labels: List of labels for the scenarios.
    :param config: Configuration of the scenario hour
    """
    traffic_df = pd.read_csv(config.traffic_csv_uri)
    simulation_hour = get_simulation_hour(traffic_df, config)
    residential_max = get_max_hour(traffic_df, "residential")
    office_max = get_max_hour(traffic_df, "office")
    mixed_max = get_max_hour(traffic_df, "mixed")
    print(f"Hour selected: {simulation_hour}")
    selected_row = traffic_df.loc[traffic_df['hour'] == simulation_hour]
    average_traffic_per_rrh = (selected_row.iloc[0]['residential'] +
                               selected_row.iloc[0]['office'] + selected_row.iloc[0]['mixed']) / 3
    bbu_pool_multiplexing_gains = {}  # Store multiplexing gains by BBU pool ID
    bbu_pool_ids = []  # Store unique BBU pool IDs (for x-axis)

    # Loop over each scenario
    for scenario_idx, (bbu_pools, label) in enumerate(zip(bbu_pools_by_scenario, scenario_labels)):
        for pool in bbu_pools:
            # Compute total traffic for the current BBU pool in Mbps
            total_traffic_mbps = sum(rrh['traffic'] for rrh in pool.connected_rrh)
            maximum_possible_traffic_mbps = 0
            for rrh in pool.connected_rrh:
                if rrh['type'] == "residential":
                    maximum_possible_traffic_mbps += residential_max
                elif rrh['type'] == "office":
                    maximum_possible_traffic_mbps += office_max
                elif rrh['type'] == "mixed":
                    maximum_possible_traffic_mbps += mixed_max
            # Compute the number of RRHs connected to the pool
            num_rrhs = len(pool.connected_rrh)

            if num_rrhs > 0:
                # Calculate the actual traffic capacity per RRH (in Mbps)
                actual_capacity_mbps = total_traffic_mbps

                # Calculate the theoretical traffic capacity for the pool (ideal capacity)
                expected_capacity_mbps = maximum_possible_traffic_mbps

                # Calculate the multiplexing gain
                multiplexing_gain = expected_capacity_mbps / actual_capacity_mbps if expected_capacity_mbps > 0 else 0

                # Store the multiplexing gain for each BBU pool (by ID) and scenario
                if pool.identifier not in bbu_pool_multiplexing_gains:
                    bbu_pool_multiplexing_gains[pool.identifier] = []
                    bbu_pool_ids.append(pool.identifier)

                bbu_pool_multiplexing_gains[pool.identifier].append(multiplexing_gain)

    # Plotting the BBU pool multiplexing gains for each scenario
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the bar width and the spacing between bars
    bar_width = 0.2
    x_positions = np.arange(len(bbu_pool_ids))  # Positions for the x-axis (BBU pool IDs)

    # Plot the bars for each scenario
    for i, label in enumerate(scenario_labels):
        # Extract the multiplexing gains for the current scenario
        scenario_multiplexing_gains = [
            bbu_pool_multiplexing_gains[bbu_id][i] if i < len(bbu_pool_multiplexing_gains[bbu_id]) else 0
            for bbu_id in bbu_pool_ids]

        # Plot the bars for this scenario
        ax.bar(x_positions + i * bar_width, scenario_multiplexing_gains, width=bar_width, label=label)

    # Add labels and title
    ax.set_xlabel('BBU Pool ID')
    ax.set_ylabel('Multiplexing Gain')
    ax.set_title('Multiplexing Gain Across BBU Pools and Scenarios')
    ax.set_xticks(x_positions + bar_width * (len(scenario_labels) - 1) / 2)  # Center the x-ticks
    ax.set_xticklabels(bbu_pool_ids)

    # Add a legend
    ax.legend(title="Scenarios")

    plt.tight_layout()
    plt.show()


def plot_average_bbu_pool_capacity_across_scenarios(bbu_pools_by_scenario, scenario_labels):
    """
    Plot the average traffic capacity (in GB/h) per RRH for each BBU pool across multiple scenarios.
    :param bbu_pools_by_scenario: List of BBU pools for each scenario.
    :param scenario_labels: List of labels for the scenarios.
    """
    bbu_pool_capacities = {}  # Store BBU pool capacities by BBU pool ID
    bbu_pool_ids = []  # Store unique BBU pool IDs (for x-axis)

    # Loop over each scenario
    for scenario_idx, (bbu_pools, label) in enumerate(zip(bbu_pools_by_scenario, scenario_labels)):
        for pool in bbu_pools:
            # Compute total traffic for the current BBU pool in Mbps
            total_traffic_mbps = sum(rrh['traffic'] for rrh in pool.connected_rrh)

            # Compute the number of RRHs connected to the pool
            num_rrhs = len(pool.connected_rrh)

            if num_rrhs > 0:
                # Calculate the average traffic capacity per RRH (in Mbps)
                average_capacity_mbps = total_traffic_mbps
                # print(average_capacity_mbps)

                # Convert from Mbps to GB/h: Traffic in GB/h = (Average Capacity in Mbps * 3600) / (8 * 1000)
                average_capacity_gb_per_hour = (average_capacity_mbps * 3600) / (8 * 1000)

                # Store the average capacity for each BBU pool (by ID) and scenario
                if pool.identifier not in bbu_pool_capacities:
                    bbu_pool_capacities[pool.identifier] = []
                    bbu_pool_ids.append(pool.identifier)

                bbu_pool_capacities[pool.identifier].append(average_capacity_gb_per_hour)

    # Plotting the BBU pool average capacities for each scenario
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define the bar width and the spacing between bars
    bar_width = 0.2
    x_positions = np.arange(len(bbu_pool_ids))  # Positions for the x-axis (BBU pool IDs)

    # Plot the bars for each scenario
    for i, label in enumerate(scenario_labels):
        # Extract the average capacities for the current scenario
        scenario_capacities = [bbu_pool_capacities[bbu_id][i] if i < len(bbu_pool_capacities[bbu_id]) else 0
                               for bbu_id in bbu_pool_ids]

        # Plot the bars for this scenario
        ax.bar(x_positions + i * bar_width, scenario_capacities, width=bar_width, label=label)

    # Add labels and title
    ax.set_xlabel('BBU Pool ID')
    ax.set_ylabel('Average BBU Pool Capacity (GB/h)')
    ax.set_title('Average BBU Pool Capacities Across Scenarios')
    ax.set_xticks(x_positions + bar_width * (len(scenario_labels) - 1) / 2)  # Center the x-ticks
    ax.set_xticklabels(bbu_pool_ids)

    # Add a legend
    ax.legend(title="Scenarios")

    plt.tight_layout()
    plt.show()


def track_rrh_reassignments(bbu_pools_by_hour, hours):
    """
    Track how many RRHs change their BBU assignment between different hours, including non-consecutive pairs.
    :param bbu_pools_by_hour: List of BBU pool assignments for each hour (e.g., [pools_8am, pools_3pm, pools_9pm]).
    :param hours: List of hours corresponding to the BBU pool assignments (e.g., [8, 15, 21]).
    :return: DataFrame with the number of RRHs that change assignments between each pair of hours.
    """
    # Extract RRH-to-BBU mappings for each hour
    rrh_assignments = []
    for pools in bbu_pools_by_hour:
        assignment = {}
        for pool in pools:
            for rrh in pool.connected_rrh:
                assignment[rrh['id']] = pool.identifier  # Assign RRH to its BBU
        rrh_assignments.append(assignment)

    # Compare assignments between all relevant hour pairs
    reassignment_counts = []
    hour_pairs = []
    for i in range(len(rrh_assignments) - 1):
        # Compare consecutive hour pairs
        current = rrh_assignments[i]
        next_ = rrh_assignments[i + 1]
        reassignments = sum(1 for rrh_id in current if rrh_id in next_ and current[rrh_id] != next_[rrh_id])
        reassignment_counts.append(reassignments)
        hour_pairs.append(f"{hours[i]}-{hours[i + 1]}")

    # Add comparison for non-consecutive hours (e.g., 8-21)
    first = rrh_assignments[0]
    last = rrh_assignments[-1]
    reassignments_8_21 = sum(1 for rrh_id in first if rrh_id in last and first[rrh_id] != last[rrh_id])
    reassignment_counts.append(reassignments_8_21)
    hour_pairs.append(f"{hours[0]}-{hours[-1]}")

    # Create a table summarizing the changes
    df_reassignments = pd.DataFrame({
        "Hour Pair": hour_pairs,
        "RRHs Changed": reassignment_counts
    })

    # Plot changes as a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(df_reassignments["Hour Pair"], df_reassignments["RRHs Changed"], color='skyblue', edgecolor='black')
    plt.xlabel("Hour Pair")
    plt.ylabel("Number of RRHs Changed")
    plt.title("RRH Reassignments Between Hours")
    plt.tight_layout()
    plt.show()

    return df_reassignments


def plot_rrh_distribution_by_type(bbu_pools):
    """
    Generate a bar chart showing the number of connected RRHs by type for each BBU.
    """
    # Prepare data
    data = {}
    for pool in bbu_pools:
        rrh_types = [rrh['type'] for rrh in pool.connected_rrh]
        type_counts = pd.Series(rrh_types).value_counts()
        data[pool.identifier] = type_counts

    # Create a DataFrame for easier plotting
    df = pd.DataFrame(data).fillna(0).astype(int)

    # Transpose to have BBU IDs on the x-axis
    df = df.T

    # Plot stacked bar chart
    df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis', edgecolor='black')
    plt.xlabel("BBU ID")
    plt.ylabel("Number of Connected RRHs")
    plt.title("Distribution of Connected RRHs by Type for Each BBU")
    plt.xticks(rotation=0)
    plt.legend(title="RRH Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_traffic_per_bbu(bbu_pools):
    """
    Generate a bar chart of total traffic per BBU.
    """
    bbu_ids = []
    total_traffic = []

    for pool in bbu_pools:
        bbu_ids.append(pool.identifier)
        traffic_sum = sum(rrh['traffic'] for rrh in pool.connected_rrh)
        total_traffic.append(traffic_sum)

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(bbu_ids, total_traffic, color='orange', edgecolor='black')
    plt.xlabel("BBU ID")
    plt.ylabel("Total Traffic (Mbps)")
    plt.title("Total Traffic per BBU")
    plt.xticks(bbu_ids)
    plt.tight_layout()
    plt.show()


def plot_total_traffic_across_scenarios(bbu_pools_list, labels):
    """
    Generate a bar chart comparing total network traffic across different scenarios.
    :param bbu_pools_list: List of BBU pools (e.g., [bbu_pools1, bbu_pools2, bbu_pools3]).
    :param labels: List of labels for the scenarios (e.g., ["Scenario 1", "Scenario 2", "Scenario 3"]).
    """
    total_traffic = []

    # Calculate total traffic for each scenario
    for bbu_pools in bbu_pools_list:
        scenario_traffic = sum(
            sum(rrh['traffic'] for rrh in pool.connected_rrh) for pool in bbu_pools
        )
        total_traffic.append(scenario_traffic)

    # Plot the bar chart
    plt.figure(figsize=(8, 5))
    plt.plot(labels, total_traffic, marker='o', color='blue', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel("Scenarios simulation hour ")
    plt.ylabel("Total Network Traffic (Mbps)")
    plt.title("Total Network Traffic Across Scenarios")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_traffic_comparison(bbu_pools_list, labels):
    """
    Generate a grouped bar chart of total traffic per BBU for multiple BBU pools.
    :param bbu_pools_list: List of BBU pools (e.g., [bbu_pools1, bbu_pools2, bbu_pools3]).
    :param labels: List of labels for the scenarios (e.g., ["Scenario 1", "Scenario 2", "Scenario 3"]).
    """
    num_scenarios = len(bbu_pools_list)
    bbu_ids = [pool.identifier for pool in bbu_pools_list[0]]  # Assume all lists have the same BBUs
    traffic_data = []

    # Collect traffic data for each BBU pool
    for bbu_pools in bbu_pools_list:
        traffic_data.append([sum(rrh['traffic'] for rrh in pool.connected_rrh) for pool in bbu_pools])

    # Convert traffic data to a DataFrame for easier plotting
    traffic_df = pd.DataFrame(traffic_data, index=labels, columns=bbu_ids).T

    # Plot grouped bar chart
    x = np.arange(len(bbu_ids))  # Positions for BBU IDs
    width = 0.25  # Width of each bar

    plt.figure(figsize=(12, 6))
    for i, label in enumerate(labels):
        plt.bar(x + i * width, traffic_df[label], width, label=label)

    # Customize the plot
    plt.xlabel("BBU ID")
    plt.ylabel("Total Traffic (Mbps)")
    plt.title("Total Traffic per BBU Across Scenarios")
    plt.xticks(x + width, bbu_ids)
    plt.legend(title="Scenarios")
    plt.tight_layout()
    plt.show()


def plot_rrhs_per_bbu(bbu_pools):
    """Bar chart showing the number of RRHs connected to each BBU."""
    bbu_ids = [pool.identifier for pool in bbu_pools]
    rrh_counts = [len(pool.connected_rrh) for pool in bbu_pools]

    plt.figure(figsize=(10, 6))
    plt.bar(bbu_ids, rrh_counts, color='orange', edgecolor='black')
    plt.xlabel('BBU ID')
    plt.ylabel('Number of Connected RRHs')
    plt.title('Number of RRHs Connected to Each BBU')
    plt.xticks(bbu_ids)
    plt.show()


def plot_rrh_type_distribution(df):
    # Pie chart
    df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['orange', 'blue', 'green'])
    plt.title('RRH Type Distribution')
    plt.ylabel('')
    plt.show()


def plot_distance_histogram(bbu_pools, frequency_ylim=None):
    """Histogram of distances with scaled cumulative percentage line and independent y-axis limits."""
    distances = []
    for pool in bbu_pools:
        for rrh in pool.connected_rrh:
            dist = haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude'])
            distances.append(dist)

    distances = np.array(distances)

    # Create the figure and primary axis for the histogram
    fig, ax1 = plt.subplots(figsize=(10, 6))
    counts, bins, _ = ax1.hist(distances, bins=40, color='orange', edgecolor='black', alpha=0.7, label='Frequency')

    # Adjust y-axis limits for the histogram (primary y-axis)
    if frequency_ylim:
        ax1.set_ylim(0, frequency_ylim)

    # Add labels and legend for the histogram
    ax1.set_xlabel('Distance (meters)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Distances Between BBUs and Connected RRHs')
    ax1.legend(loc='upper left')

    # Create the cumulative percentage line (secondary axis)
    cumulative = np.cumsum(counts)
    ax2 = ax1.twinx()  # Create a twin y-axis
    ax2.plot(0.5 * (bins[1:] + bins[:-1]), cumulative / cumulative[-1] * 100, color='blue', linestyle='-',
             linewidth=2, label='Cumulative (%)')

    # Set the secondary y-axis for percentage
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_ylim(0, 100)  # Percentage always ranges from 0 to 100

    # Add a legend for the cumulative line
    ax2.legend(loc='upper right')

    plt.show()


def plot_distance_histogram_across_scenarios(bbu_pools_by_scenario, scenario_labels, frequency_ylim=None):
    """
    Plot a histogram of distances and cumulative percentage for multiple scenarios.
    :param bbu_pools_by_scenario: List of BBU pools for each scenario.
    :param scenario_labels: List of labels for the scenarios.
    :param frequency_ylim: Optional limit for the histogram y-axis.
    """
    colors = ['orange', 'blue', 'green', 'red']  # Define colors for scenarios
    markers = ['o', 's', 'D', '^']  # Define markers for cumulative lines
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ""

    # Define the bin edges for the histogram
    bin_edges = np.arange(0, 11000, 1000)  # 0-1 km, 1-2 km, ..., 10-11 km
    interval_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Midpoints of each interval

    # Bar width and offset for spacing
    bar_width = 500
    bar_offset = bar_width / len(scenario_labels)

    frequency_handles = []  # Store frequency bar handles for the legend
    cumulative_handles = []  # Store cumulative line handles for the legend

    for i, (bbu_pools, label) in enumerate(zip(bbu_pools_by_scenario, scenario_labels)):
        # Compute distances
        distances = []
        for pool in bbu_pools:
            for rrh in pool.connected_rrh:
                dist = haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude'])
                distances.append(dist)

        distances = np.array(distances)

        if len(distances) == 0:
            print(f"No valid distances for scenario '{label}'")
            continue

        # Histogram and cumulative data
        counts, bins = np.histogram(distances, bins=bin_edges)  # Use predefined bin edges
        cumulative = np.cumsum(counts)

        # Plot histogram (grouped bars with spacing)
        bars = ax1.bar(interval_centers + i * bar_offset - (len(scenario_labels) - 1) * bar_offset / 2,  # Centering
                       counts, width=bar_width / len(scenario_labels), color=colors[i % len(colors)], alpha=0.7,
                       edgecolor='black', label=f'{label} (Frequency)')

        # Save handle for frequency legend
        frequency_handles.append(bars)

        # Adjust histogram y-axis limits
        if frequency_ylim:
            ax1.set_ylim(0, frequency_ylim)

        # Add labels for the histogram
        ax1.set_xlabel('Distance Intervals (km)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Distances Across Scenarios')

        ax2 = ax1.twinx()
        # Plot cumulative line (secondary y-axis)
        line, = ax2.plot(interval_centers, cumulative / cumulative[-1] * 100,
                         linestyle='-', linewidth=2, color=colors[i % len(colors)], marker=markers[i % len(markers)],
                         label=f'{label} (Cumulative %)')

        # Save handle for cumulative legend
        cumulative_handles.append(line)

        # Set cumulative percentage y-axis limits
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('Cumulative Percentage (%)')

    # Format the x-axis with intervals
    interval_labels = [f"{int(bin_edges[i] / 1000)}-{int(bin_edges[i + 1] / 1000)} km" for i in
                       range(len(bin_edges) - 1)]
    ax1.set_xticks(interval_centers)
    ax1.set_xticklabels(interval_labels, rotation=45)

    # Combine legends
    frequency_legend = ax1.legend(handles=frequency_handles, loc='upper left', title='Frequency')
    cumulative_legend = ax2.legend(handles=cumulative_handles, loc='upper right', title='Cumulative')

    # Finalize layout
    plt.tight_layout()
    plt.show()


def plot_scatter_bbu_rrh(bbu_pools):
    """Scatter plot showing the geographic distribution of BBUs and RRHs."""
    bbu_coords = [(pool.latitude, pool.longitude) for pool in bbu_pools]
    rrh_coords = [(rrh['latitude'], rrh['longitude']) for pool in bbu_pools for rrh in pool.connected_rrh]

    bbu_lats, bbu_lons = zip(*bbu_coords)
    rrh_lats, rrh_lons = zip(*rrh_coords)

    plt.figure(figsize=(10, 6))
    plt.scatter(rrh_lons, rrh_lats, c='blue', label='RRHs', alpha=0.6)
    plt.scatter(bbu_lons, bbu_lats, c='red', label='BBUs', marker='x', s=100)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of BBUs and RRHs')
    plt.legend()
    plt.show()


def plot_distance_boxplot(bbu_pools):
    """Boxplot showing distance variability per BBU."""
    distances = {pool.identifier: [haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude'])
                                   for rrh in pool.connected_rrh]
                 for pool in bbu_pools}

    distance_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in distances.items()]))
    distance_df.boxplot(figsize=(12, 6), grid=True)
    plt.xlabel('BBU ID')
    plt.ylabel('Distance (meters)')
    plt.title('Distance Variability Between BBUs and Connected RRHs')
    plt.show()


# Export to csv
def export_bbu_details_to_csv(bbu_pools, output_file="bbu_details.csv"):
    data = []
    # print("hello")
    for pool in bbu_pools:
        for rrh in pool.connected_rrh:
            distance = haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude'])
            data.append({
                "BBU_ID": pool.identifier,
                "BBU_Latitude": pool.latitude,
                "BBU_Longitude": pool.longitude,
                "RRH_id": rrh['id'],
                "RRH_Latitude": rrh['latitude'],
                "RRH_Longitude": rrh['longitude'],
                "RRH_Type": rrh['type'],
                "Distance_meters": distance
            })

    # Save data to a CSV
    pd.DataFrame(data).to_csv(output_file, index=False)


# Haversine formula to calculate distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of the Earth in meters
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)

    # Corrected a formula
    a = sin(delta_phi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2

    # Compute c
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Return distance
    return R * c


def assign_remaining_rrhs_to_pools(remaining_rrhs, bbu_pools, config):
    assigned_rrh = 0
    selected_row = ""
    bbu_traffic_capacity = config.bbu_capacity
    if config.test_mode == 2 or 3:
        print("Test using allocation with traffic initiated '2 or 3' initiated")
        traffic_df = pd.read_csv(config.traffic_csv_uri)
        simulation_hour = get_simulation_hour(traffic_df, config)
        selected_row = traffic_df.loc[traffic_df['hour'] == simulation_hour]
    for _, rrh in remaining_rrhs.iterrows():
        # Find the nearest BBU pool within coverage radius and with available capacity
        rrh_traffic = 1
        if config.test_mode == 2 or 3:
            rrh_traffic = selected_row.iloc[0][rrh['type']]
        nearest_bbu = None
        min_distance = float('inf')
        for pool in bbu_pools:
            distance = haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude'])
            if config.test_mode == 2 or 3:
                if (distance <= config.coverage_radius and sum(rrh['traffic'] for rrh in pool.connected_rrh) +
                        rrh_traffic <= bbu_traffic_capacity):
                    if distance < min_distance:
                        nearest_bbu = pool
                        min_distance = distance
            else:
                if distance <= config.coverage_radius and len(pool.connected_rrh) < config.num_RRHs_per_BBU:
                    if distance < min_distance:
                        nearest_bbu = pool
                        min_distance = distance

        # Assign RRH to the nearest eligible BBU pool
        if nearest_bbu:
            nearest_bbu.connected_rrh.append(rrh)
            assigned_rrh += 1

    return assigned_rrh


def log_unassigned_rrhs(remaining_rrhs, bbu_pools, config):
    unassigned_rrhs = []
    for _, rrh in remaining_rrhs.iterrows():
        # Check if it is out of range or no slots are available
        assignable = any(
            haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude']) <= config.coverage_radius and
            len(pool.connected_rrh) < config.num_RRHs_per_BBU
            for pool in bbu_pools
        )
        if not assignable:
            unassigned_rrhs.append(rrh)
    # Log or return unassigned RRHs for further action
    return unassigned_rrhs


def balance_rrh_distribution(bbu_pools, config):
    # Step 1: Calculate the target number of RRHs per BBU
    total_rrhs = sum(len(pool.connected_rrh) for pool in bbu_pools)
    target_rrhs_per_bbu = min(config.num_RRHs_per_BBU, max(1, int(np.ceil(total_rrhs / len(bbu_pools)))))
    print(f"The target RRHs per BBU are: {target_rrhs_per_bbu}")

    # Step 2: Identify overloaded and underloaded BBUs
    overloaded_pools = [pool for pool in bbu_pools if len(pool.connected_rrh) > target_rrhs_per_bbu]
    underloaded_pools = [pool for pool in bbu_pools if len(pool.connected_rrh) < target_rrhs_per_bbu]
    unassigned_rrhs = []

    # Step 3: Reassign RRHs to balance the load
    for overloaded_pool in overloaded_pools:
        # Sort connected RRHs by their distance from the BBU, in descending order
        overloaded_pool.connected_rrh.sort(
            key=lambda rrh: haversine(
                rrh['latitude'], rrh['longitude'],
                overloaded_pool.latitude, overloaded_pool.longitude
            ),
            reverse=True  # Prioritize further RRHs
        )

        while len(overloaded_pool.connected_rrh) > target_rrhs_per_bbu and underloaded_pools:
            # Remove the furthest RRH from the overloaded pool
            rrh_to_reassign = overloaded_pool.connected_rrh.pop()
            found = False
            underloaded_pools.sort(
                key=lambda under_load_pool: haversine(
                    rrh_to_reassign['latitude'], rrh_to_reassign['longitude'],
                    under_load_pool.latitude, under_load_pool.longitude
                ),
                reverse=False  # Prioritize nearer underloaded pools
            )
            # Find a suitable underloaded BBU pool
            for underloaded_pool in underloaded_pools:
                distance = haversine(
                    rrh_to_reassign['latitude'],
                    rrh_to_reassign['longitude'],
                    underloaded_pool.latitude,
                    underloaded_pool.longitude
                )
                # Check if the RRH can be reassigned to the underloaded pool
                if (
                        distance <= config.coverage_radius and
                        len(underloaded_pool.connected_rrh) < target_rrhs_per_bbu
                ):
                    underloaded_pool.connected_rrh.append(rrh_to_reassign)
                    found = True
                    break

            if not found:
                # Try to reassign within overloaded pools if possible
                for overloaded_pool_inside in overloaded_pools:
                    if overloaded_pool_inside == overloaded_pool:
                        continue
                    distance = haversine(
                        rrh_to_reassign['latitude'],
                        rrh_to_reassign['longitude'],
                        overloaded_pool_inside.latitude,
                        overloaded_pool_inside.longitude
                    )
                    if (
                            distance <= config.coverage_radius and
                            len(overloaded_pool_inside.connected_rrh) < target_rrhs_per_bbu
                    ):
                        overloaded_pool_inside.connected_rrh.append(rrh_to_reassign)
                        found = True
                        break

            if not found:
                # Add to unassigned RRHs if no suitable BBU is found
                unassigned_rrhs.append(rrh_to_reassign)

            # Update underloaded pools list
            underloaded_pools = [
                pool for pool in underloaded_pools
                if len(pool.connected_rrh) < target_rrhs_per_bbu
            ]

    return pd.DataFrame(unassigned_rrhs)


def balance_rrh_distribution_traffic(bbu_pools, config, selected_row):
    # Step 1: Calculate the target number of RRHs per BBU
    total_traffic = sum(sum(rrh['traffic'] for rrh in pool.connected_rrh) for pool in bbu_pools)
    target_traffic_per_bbu = min(config.bbu_capacity, max(1, int(np.ceil(total_traffic / len(bbu_pools)))))
    print(f"The target Traffic per BBU are: {target_traffic_per_bbu} Mbps")

    # Step 2: Identify overloaded and underloaded BBUs
    overloaded_pools = [pool for pool in bbu_pools if sum(rrh['traffic'] for rrh in pool.connected_rrh) >
                        target_traffic_per_bbu]
    underloaded_pools = [pool for pool in bbu_pools if sum(rrh['traffic'] for rrh in pool.connected_rrh) <
                         target_traffic_per_bbu]
    unassigned_rrhs = []

    # Step 3: Reassign RRHs to balance the load
    for overloaded_pool in overloaded_pools:
        # Sort connected RRHs by their distance from the BBU, in descending order
        overloaded_pool.connected_rrh.sort(
            key=lambda rrh: haversine(
                rrh['latitude'], rrh['longitude'],
                overloaded_pool.latitude, overloaded_pool.longitude
            ),
            reverse=True  # Prioritize further RRHs
        )

        while (sum(rrh['traffic'] for rrh in overloaded_pool.connected_rrh) > target_traffic_per_bbu and
               underloaded_pools):
            # Remove the furthest RRH from the overloaded pool
            rrh_to_reassign = overloaded_pool.connected_rrh.pop()
            found = False
            underloaded_pools.sort(
                key=lambda under_load_pool: haversine(
                    rrh_to_reassign['latitude'], rrh_to_reassign['longitude'],
                    under_load_pool.latitude, under_load_pool.longitude
                ),
                reverse=False  # Prioritize nearer underloaded pools
            )
            # Find a suitable underloaded BBU pool
            for underloaded_pool in underloaded_pools:
                distance = haversine(
                    rrh_to_reassign['latitude'],
                    rrh_to_reassign['longitude'],
                    underloaded_pool.latitude,
                    underloaded_pool.longitude
                )
                # Check if the RRH can be reassigned to the underloaded pool
                if (
                        distance <= config.coverage_radius and
                        sum(rrh['traffic'] for rrh in underloaded_pool.connected_rrh) < target_traffic_per_bbu
                ):
                    underloaded_pool.connected_rrh.append(rrh_to_reassign)
                    found = True
                    break

            if not found:
                # Try to reassign within overloaded pools if possible
                for overloaded_pool_inside in overloaded_pools:
                    if overloaded_pool_inside == overloaded_pool:
                        continue
                    distance = haversine(
                        rrh_to_reassign['latitude'],
                        rrh_to_reassign['longitude'],
                        overloaded_pool_inside.latitude,
                        overloaded_pool_inside.longitude
                    )
                    if (
                            distance <= config.coverage_radius and
                            sum(rrh['traffic'] for rrh in overloaded_pool_inside.connected_rrh) < target_traffic_per_bbu
                    ):
                        overloaded_pool_inside.connected_rrh.append(rrh_to_reassign)
                        found = True
                        break

            if not found:
                # Add to unassigned RRHs if no suitable BBU is found
                unassigned_rrhs.append(rrh_to_reassign)

            # Update underloaded pools list
            underloaded_pools = [
                pool for pool in underloaded_pools
                if sum(rrh['traffic'] for rrh in pool.connected_rrh) < target_traffic_per_bbu
            ]

    return pd.DataFrame(unassigned_rrhs)


def optimize_bbu_pools_with_constraints(rrhs_df, config):
    # Step 1: Perform initial clustering with K-Means
    rrh_coords = rrhs_df[['latitude', 'longitude']].to_numpy()
    kmeans = KMeans(n_clusters=config.num_BBU_pools, random_state=0).fit(rrh_coords)

    # Initialize BBU pools
    bbu_pools = []
    used_rrhs = set()
    selected_row = -1

    traffic_df = pd.read_csv(config.traffic_csv_uri)
    simulation_hour = get_simulation_hour(traffic_df, config)
    print(f"Hour selected: {simulation_hour}")
    selected_row = traffic_df.loc[traffic_df['hour'] == simulation_hour]
    # print(selected_row)
    rrhs_df['traffic'] = rrhs_df['type'].apply(lambda t: selected_row.iloc[0][t])

    for idx, center in enumerate(kmeans.cluster_centers_):
        # Select RRHs belonging to this cluster
        cluster_rrhs = rrhs_df[kmeans.labels_ == idx]

        # Sort the RRHs by distance to the current BBU center
        cluster_rrhs = cluster_rrhs.assign(
            distance_to_center=cluster_rrhs.apply(
                lambda rrh: haversine(center[0], center[1], rrh['latitude'], rrh['longitude']),
                axis=1
            )
        ).sort_values(by='distance_to_center')

        # Step 2: Filter RRHs within coverage radius and respect RRH limit
        connected_rrhs = []
        bbu_traffic_capacity = config.bbu_capacity

        for _, rrh in cluster_rrhs.iterrows():
            rrh_traffic = 1  # Default traffic per RRH
            if config.test_mode == 2 or 3:
                rrh_traffic = selected_row.iloc[0][rrh['type']]  # Use traffic from CSV for each type
                # Check distance and traffic constraints
                if (
                        haversine(center[0], center[1], rrh['latitude'], rrh['longitude']) <= config.coverage_radius and
                        sum(rrh['traffic'] for rrh in connected_rrhs) + rrh_traffic <= bbu_traffic_capacity
                ):
                    connected_rrhs.append(rrh)
            else:
                if haversine(center[0], center[1], rrh['latitude'], rrh['longitude']) <= config.coverage_radius:
                    connected_rrhs.append(rrh)
                if len(connected_rrhs) == config.num_RRHs_per_BBU:
                    break

        # Step 3: Place BBU pool and ensure 1km distance constraint
        bbu_location = adjust_bbu_location(center, bbu_pools, connected_rrhs, config)
        if len(bbu_location) == 2:
            # Add new BBU pool
            bbu_pools.append(BBUPool(
                identifier=idx + 1,
                latitude=bbu_location[0],
                longitude=bbu_location[1],
                connected_rrh=connected_rrhs
            ))
            used_rrhs.update([rrh.name for rrh in connected_rrhs])

    # Step 4: Assign unused RRHs (those not in any pool)
    remaining_rrhs = rrhs_df[~rrhs_df.index.isin(used_rrhs)]
    print(f"Remaining RRHs without connected BBUs: {len(remaining_rrhs)}")
    # Step 5: Handle remaining RRHs by assigning to nearby pools if possible
    assigned_rrhs_1 = assign_remaining_rrhs_to_pools(remaining_rrhs, bbu_pools, config)
    print(f"Remaining RRHs without connected BBUs after 1st assignment: {len(remaining_rrhs) - assigned_rrhs_1}")

    # Step 6: Adjust BBU positions after all RRHs are connected
    for pool in bbu_pools:
        if pool.connected_rrh:
            # Recalculate position
            latitudes = [rrh['latitude'] for rrh in pool.connected_rrh]
            longitudes = [rrh['longitude'] for rrh in pool.connected_rrh]
            pool.latitude = np.mean(latitudes)
            pool.longitude = np.mean(longitudes)
    if config.test_mode == 1:
        print("Test using uniform allocation '1' initiated")
        # Balance rrh distribution
        unassigned_rrhs = balance_rrh_distribution(bbu_pools, config)
        print(f"Remaining RRHs without connected BBUs after balance: {len(remaining_rrhs) - assigned_rrhs_1 +
                                                                      len(unassigned_rrhs)}")
        assigned_rrhs_2 = assign_remaining_rrhs_to_pools(unassigned_rrhs, bbu_pools, config)
        print(f"Remaining RRHs without connected BBUs after 2nd assignment: {len(remaining_rrhs) - assigned_rrhs_1 +
                                                                             len(unassigned_rrhs) - assigned_rrhs_2}")
        # Step 6: Adjust BBU positions after all RRHs are connected
        for pool in bbu_pools:
            if pool.connected_rrh:
                # Recalculate position
                latitudes = [rrh['latitude'] for rrh in pool.connected_rrh]
                longitudes = [rrh['longitude'] for rrh in pool.connected_rrh]
                pool.latitude = np.mean(latitudes)
                pool.longitude = np.mean(longitudes)
    if config.test_mode == 3:
        print("Load Balancing initiated")
        # Balance rrh distribution
        unassigned_rrhs = balance_rrh_distribution_traffic(bbu_pools, config, selected_row)
        print(f"Remaining RRHs without connected BBUs after balance: {len(remaining_rrhs) - assigned_rrhs_1 +
                                                                      len(unassigned_rrhs)}")
        assigned_rrhs_2 = assign_remaining_rrhs_to_pools(unassigned_rrhs, bbu_pools, config)
        print(f"Remaining RRHs without connected BBUs after 2nd assignment: {len(remaining_rrhs) - assigned_rrhs_1 +
                                                                             len(unassigned_rrhs) - assigned_rrhs_2}")
        # Step 6: Adjust BBU positions after all RRHs are connected
        for pool in bbu_pools:
            if pool.connected_rrh:
                # Recalculate position
                latitudes = [rrh['latitude'] for rrh in pool.connected_rrh]
                longitudes = [rrh['longitude'] for rrh in pool.connected_rrh]
                pool.latitude = np.mean(latitudes)
                pool.longitude = np.mean(longitudes)
    print("The number of final  unasignable rrhs is: ", len(log_unassigned_rrhs(remaining_rrhs, bbu_pools, config)))
    return bbu_pools


def get_simulation_hour(traffic_df, config):
    if config.simulation_hour == -1:
        bbu_columns = [col for col in traffic_df.columns if col != 'hour']
        if not bbu_columns:
            print("Error: No traffic columns found.")
            return None
        traffic_df['total_traffic'] = traffic_df[bbu_columns].sum(axis=1)
        peak_hour = traffic_df.loc[traffic_df['total_traffic'].idxmax(), 'hour']
        print(f"Peak hour selected automatically: {peak_hour}")
        return peak_hour
    else:
        print(f"Manual hour selected: {config.simulation_hour}")
        return config.simulation_hour


def get_max_hour(traffic_df, mode="mixed"):
    if mode == "residential":
        bbu_columns = [col for col in traffic_df.columns if col != 'hour']
        if not bbu_columns:
            print("Error: No traffic columns found.")
            return None
        max_residential = traffic_df.loc[traffic_df['residential'].idxmax(), 'residential']
        print(f"Maximum bandwidth residential obtained: {max_residential}")
        return max_residential
    elif mode == "office":
        bbu_columns = [col for col in traffic_df.columns if col != 'hour']
        if not bbu_columns:
            print("Error: No traffic columns found.")
            return None
        max_office = traffic_df.loc[traffic_df['office'].idxmax(), 'office']
        print(f"Maximum bandwidth office obtained: {max_office}")
        return max_office
    elif mode == "mixed":
        bbu_columns = [col for col in traffic_df.columns if col != 'hour']
        if not bbu_columns:
            print("Error: No traffic columns found.")
            return None
        max_mixed = traffic_df.loc[traffic_df['mixed'].idxmax(), 'mixed']
        print(f"Maximum bandwidth mixed obtained: {max_mixed}")
        return max_mixed


def adjust_bbu_location(center, existing_pools, rrhs, config):
    """Adjust BBU pool location to ensure it meets the 1.5 km distance constraint."""
    for pool in existing_pools:
        if haversine(center[0], center[1], pool.latitude, pool.longitude) < config.distance_between_BBUs:
            # Shift center slightly (towards weighted centroid of RRHs) to resolve conflict
            latitudes = [rrh['latitude'] for rrh in rrhs]
            longitudes = [rrh['longitude'] for rrh in rrhs]
            return np.mean(latitudes), np.mean(longitudes)
    return center


def randomize_residential_office(config, df):
    def assign_type(_):
        random_value = random.random()
        # For the mixed ones
        if random_value <= config.percentage_RRH_mixed / 100:
            return "mixed"
        # For the ones that are for office
        elif (config.percentage_RRH_mixed / 100) < random_value <= (
                (config.percentage_RRH_mixed / 100) + config.percentage_RRH_office / 100):
            return "office"
        # For the residential ones
        elif ((config.percentage_RRH_mixed / 100 + config.percentage_RRH_office / 100) <
              random_value <= 1):
            return "residential"

    df['type'] = df.apply(assign_type, axis=1)
    return df


def initialize(config):
    # Load the CSV file
    df = pd.read_csv(config.csv_uri)  # Use the correct CSV
    bbu_pools = []
    # Remove rows with missing coordinates
    df = df.dropna(subset=['ns1:coordinates'])
    antennas_map = []
    # Split 'ns1:coordinates' and extract longitude, latitude
    df[['longitude', 'latitude', 'altitude']] = (
        df['ns1:coordinates'].str.split(",", expand=True)[[0, 1, 2]].astype(float))
    df = randomize_residential_office(config, df)
    return df


def test(config, df):
    # Create point geometry using longitude and latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    rrhs_df = df[['latitude', 'longitude', 'type']].copy()
    rrhs_df['id'] = df['ns1:name9']
    # Place the BBU pools
    bbu_pools = optimize_bbu_pools_with_constraints(rrhs_df, config)
    # Calculate center for the map
    center_lat = df['latitude'].mean()
    # print(center_lat)
    center_lon = df['longitude'].mean()
    # print(center_lon)
    initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for idx, row in gdf.iterrows():
        color_marker = 'black'
        if row['type'] == "mixed":
            color_marker = 'orange'
        elif row['type'] == "office":
            color_marker = 'blue'
        elif row['type'] == "residential":
            color_marker = 'green'
        # print([row['latitude'], row['longitude']])
        folium.Marker(
            icon=folium.Icon(icon="tower-cell", prefix="fa",
                             color=color_marker, icon_color="black"),
            location=[row['latitude'], row['longitude']],
            popup=f"Name: {row.get('ns1:name9', '')}, Altitude: {row['altitude']} m, Type: {row['type']}"
        ).add_to(initial_map)
        # Add BBU pools to the map
    # print(bbu_pools)
    # Add BBU pools and their connections to RRHs
    for pool in bbu_pools:
        # Add BBU pool marker
        folium.Marker(
            location=[pool.latitude, pool.longitude],
            icon=folium.Icon(color="red", icon="server",
                             prefix="fa", icon_color="black"),
            popup=f"BBU Pool number {pool.identifier} "
                  f"with {len(pool.connected_rrh)} RRHs connected"
        ).add_to(initial_map)

        # Draw polylines from BBU pool to its connected RRHs
        for rrh in pool.connected_rrh:
            folium.PolyLine(
                locations=[
                    [pool.latitude, pool.longitude],  # BBU pool location
                    [rrh['latitude'], rrh['longitude']]  # Connected RRH location
                ],
                color="blue",  # Line color
                weight=4,  # Line thickness
                opacity=0.7,  # Line transparency
                popup=f"Distance: {haversine(pool.latitude,
                                             pool.longitude, rrh['latitude'], rrh['longitude'])}"
            ).add_to(initial_map)

    # Save the map
    initial_map.save("map_antennas.html")
    # Save BBU details to a CSV
    export_bbu_details_to_csv(bbu_pools, output_file="bbu_details.csv")
    # Plot the results
    plot_rrhs_per_bbu(bbu_pools)
    plot_distance_histogram(bbu_pools, 20)
    # plot_scatter_bbu_rrh(bbu_pools)
    plot_rrh_type_distribution(rrhs_df)
    plot_distance_boxplot(bbu_pools)
    if config.test_mode == 2 or 3:
        plot_rrh_distribution_by_type(bbu_pools)
        plot_traffic_per_bbu(bbu_pools)
    print("BBU details saved to bbu_details.csv")


def test_without_plotting(config, df):
    # Create point geometry using longitude and latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]

    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    rrhs_df = df[['latitude', 'longitude', 'type']].copy()
    rrhs_df['id'] = df['ns1:name9']
    # Place the BBU pools
    bbu_pools = optimize_bbu_pools_with_constraints(rrhs_df, config)
    # Calculate center for the map
    center_lat = df['latitude'].mean()
    # print(center_lat)
    center_lon = df['longitude'].mean()
    # print(center_lon)
    initial_map = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    for idx, row in gdf.iterrows():
        color_marker = 'black'
        if row['type'] == "mixed":
            color_marker = 'orange'
        elif row['type'] == "office":
            color_marker = 'blue'
        elif row['type'] == "residential":
            color_marker = 'green'
        # print([row['latitude'], row['longitude']])
        folium.Marker(
            icon=folium.Icon(icon="tower-cell", prefix="fa",
                             color=color_marker, icon_color="black"),
            location=[row['latitude'], row['longitude']],
            popup=f"Name: {row.get('ns1:name9', '')}, Altitude: {row['altitude']} m, Type: {row['type']}"
        ).add_to(initial_map)
        # Add BBU pools to the map
    # print(bbu_pools)
    # Add BBU pools and their connections to RRHs
    for pool in bbu_pools:
        # Add BBU pool marker
        folium.Marker(
            location=[pool.latitude, pool.longitude],
            icon=folium.Icon(color="red", icon="server",
                             prefix="fa", icon_color="black"),
            popup=f"BBU Pool number {pool.identifier} "
                  f"with {len(pool.connected_rrh)} RRHs connected"
        ).add_to(initial_map)

        # Draw polylines from BBU pool to its connected RRHs
        for rrh in pool.connected_rrh:
            folium.PolyLine(
                locations=[
                    [pool.latitude, pool.longitude],  # BBU pool location
                    [rrh['latitude'], rrh['longitude']]  # Connected RRH location
                ],
                color="blue",  # Line color
                weight=4,  # Line thickness
                opacity=0.7,  # Line transparency
                popup=f"Distance: {haversine(pool.latitude,
                                             pool.longitude, rrh['latitude'], rrh['longitude'])}"
            ).add_to(initial_map)

    # Save the map
    initial_map.save("map_antennas.html")
    # Save BBU details to a CSV
    export_bbu_details_to_csv(bbu_pools, output_file="bbu_details.csv")
    # Plot the results
    # plot_rrhs_per_bbu(bbu_pools)
    # plot_distance_histogram(bbu_pools, 20)
    # # plot_scatter_bbu_rrh(bbu_pools)
    # plot_rrh_type_distribution(rrhs_df)
    # plot_distance_boxplot(bbu_pools)
    # if config.test_mode == 2 or 3:
    #     plot_rrh_distribution_by_type(bbu_pools)
    #     plot_traffic_per_bbu(bbu_pools)
    print("BBU details saved to bbu_details.csv")
    return bbu_pools, rrhs_df


def main():
    config = Config()
    vector_max_traffic = []
    vector_labels = []
    hour = 0
    df = initialize(config)
    config.simulation_hour = 15
    config.test_mode = 2
    bbu_pool_distance, rrh_df_distance = test_without_plotting(config, df)
    config.test_mode = 3
    bbu_pool_load_balancing, rrh_df_load_balancing = test_without_plotting(config, df)
    config.test_mode = 1
    bbu_pool_num_rrh, rrh_df_num_rrh = test_without_plotting(config, df)
    plot_distance_histogram_across_scenarios([bbu_pool_distance, bbu_pool_load_balancing, bbu_pool_num_rrh],
                                              ["Minimum distance", "Load balancing", "Uniform distribution RRHs"])
    plot_average_bbu_pool_capacity_across_scenarios([bbu_pool_distance, bbu_pool_load_balancing, bbu_pool_num_rrh],
                                                    ["Minimum distance", "Load balancing", "Uniform distribution RRHs"])
    plot_multiplexing_gain_across_scenarios([bbu_pool_distance, bbu_pool_load_balancing, bbu_pool_num_rrh],
                                            ["Minimum distance", "Load balancing", "Uniform distribution RRHs"], config)
    # track_rrh_reassignments([bbu_pool_8, bbu_pool_15, bbu_pool_21], [8, 15, 21])
    # config.percentage_RRH_mixed = 50
    # config.percentage_RRH_office = 20
    # config.percentage_RRH_residential = 30
    # df = initialize(config)
    # bbu_pools_real, rrhs_df_normal = test_without_plotting(config, df)
    # config.percentage_RRH_mixed = 0
    # config.percentage_RRH_office = 0
    # config.percentage_RRH_residential = 100
    # df = initialize(config)
    # bbu_pools_residential, rrhs_df_residential = test_without_plotting(config, df)
    # config.percentage_RRH_mixed = 0
    # config.percentage_RRH_office = 100
    # config.percentage_RRH_residential = 0
    # df = initialize(config)
    # bbu_pools_office, rrhs_df_office = test_without_plotting(config, df)
    # plot_traffic_comparison([bbu_pools_real, bbu_pools_residential, bbu_pools_office],
    #                         ["Real Scenario", "100% residential", "100% office"])
    # while hour <= 24:
    #     config.simulation_hour = hour
    #     bbu_pools_office, rrhs_df_office = test_without_plotting(config, df)
    #     vector_max_traffic.append(bbu_pools_office)
    #     vector_labels.append(str(hour))
    #     hour = hour + 1
    # plot_total_traffic_across_scenarios(vector_max_traffic, vector_labels)


if __name__ == "__main__":
    main()
