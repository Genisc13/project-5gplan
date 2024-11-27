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


def export_bbu_details_to_csv(bbu_pools, output_file="bbu_details.csv"):
    data = []
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


# Function to place BBU pools
# def place_bbu_pools(rrhs_df, config):
#     bbu_pools = []
#     remaining_rrhs = rrhs_df.copy()
#     identifier = 1
#     while not remaining_rrhs.empty and len(bbu_pools) < config.num_BBU_pools:
#         # Select the first RRH as the base location for the BBU pool
#         first_rrh = remaining_rrhs.iloc[0]
#         bbu_location = (first_rrh['latitude'], first_rrh['longitude'])
#
#         bbu_pool = BBUPool(identifier, bbu_location[0], bbu_location[1], [])
#
#         # Find and connect nearby RRHs
#         for _, rrh in remaining_rrhs.iterrows():
#             distance = haversine(bbu_location[0], bbu_location[1], rrh['latitude'], rrh['longitude'])
#             if distance <= config.coverage_radius and len(bbu_pool.connected_rrh) < config.num_RRHs_per_BBU:
#                 bbu_pool.connected_rrh.append(rrh)
#         # Add the new BBU pool to the list
#         bbu_pools.append(bbu_pool)
#         identifier += 1
#         # Remove connected RRHs from the remaining list
#         connected_indices = [rrh.name for rrh in bbu_pool.connected_rrh]
#         remaining_rrhs = remaining_rrhs.drop(index=connected_indices)
#
#         # Ensure BBU pools are at least 1.5 km apart
#         for pool in bbu_pools:
#             for _, rrh in remaining_rrhs.iterrows():
#                 if haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude']) < 1500:
#                     remaining_rrhs = remaining_rrhs[remaining_rrhs['latitude'] != rrh['latitude']]
#
#     return bbu_pools
def assign_remaining_rrhs_to_pools(remaining_rrhs, bbu_pools, config):
    for _, rrh in remaining_rrhs.iterrows():
        # Find the nearest BBU pool within coverage radius and with available capacity
        nearest_bbu = None
        min_distance = float('inf')
        for pool in bbu_pools:
            distance = haversine(pool.latitude, pool.longitude, rrh['latitude'], rrh['longitude'])
            if distance <= config.coverage_radius and len(pool.connected_rrh) < config.num_RRHs_per_BBU:
                if distance < min_distance:
                    nearest_bbu = pool
                    min_distance = distance

        # Assign RRH to the nearest eligible BBU pool
        if nearest_bbu:
            nearest_bbu.connected_rrh.append({
                "id": rrh['id'],
                "latitude": rrh['latitude'],
                "longitude": rrh['longitude']
            })


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

    # Step 2: Identify overloaded and underloaded BBUs
    overloaded_pools = [pool for pool in bbu_pools if len(pool.connected_rrh) > target_rrhs_per_bbu]
    underloaded_pools = [pool for pool in bbu_pools if len(pool.connected_rrh) < target_rrhs_per_bbu]

    # Step 3: Reassign RRHs to balance the load
    for overloaded_pool in overloaded_pools:
        while len(overloaded_pool.connected_rrh) > target_rrhs_per_bbu and underloaded_pools:
            # Remove excess RRH from overloaded pool
            rrh_to_reassign = overloaded_pool.connected_rrh.pop()

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
                    break

            # Update underloaded pools list
            underloaded_pools = [
                pool for pool in underloaded_pools
                if len(pool.connected_rrh) < target_rrhs_per_bbu
            ]


def optimize_bbu_pools_with_constraints(rrhs_df, config):
    # Step 1: Perform initial clustering with K-Means
    rrh_coords = rrhs_df[['latitude', 'longitude']].to_numpy()
    kmeans = KMeans(n_clusters=config.num_BBU_pools, random_state=0).fit(rrh_coords)

    # Initialize BBU pools
    bbu_pools = []
    used_rrhs = set()

    for idx, center in enumerate(kmeans.cluster_centers_):
        # Select RRHs belonging to this cluster
        cluster_rrhs = rrhs_df[kmeans.labels_ == idx]

        # Step 2: Filter RRHs within coverage radius and respect RRH limit
        connected_rrhs = []
        for _, rrh in cluster_rrhs.iterrows():
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

    # Remove unused RRHs (those not in any pool)
    remaining_rrhs = rrhs_df[~rrhs_df.index.isin(used_rrhs)]

    # Optionally: Handle remaining RRHs by assigning to nearby pools if possible
    assign_remaining_rrhs_to_pools(remaining_rrhs, bbu_pools, config)
    remaining_rrhs = rrhs_df[~rrhs_df.index.isin(used_rrhs)]
    # Balance rrh distribution
    balance_rrh_distribution(bbu_pools, config)
    print("The number of unassigned rrhs is: ", len(log_unassigned_rrhs(remaining_rrhs, bbu_pools, config)))
    return bbu_pools


def adjust_bbu_location(center, existing_pools, rrhs, config):
    """Adjust BBU pool location to ensure it meets the 1.5 km distance constraint."""
    for pool in existing_pools:
        if haversine(center[0], center[1], pool.latitude, pool.longitude) < config.distance_between_BBUs:
            # Shift center slightly (towards weighted centroid of RRHs) to resolve conflict
            latitudes = [rrh['latitude'] for rrh in rrhs]
            longitudes = [rrh['longitude'] for rrh in rrhs]
            return np.mean(latitudes), np.mean(longitudes)
    return center


def randomize_residential_office(config):
    random_value = random.random()
    color = "orange"
    # For the mixed ones
    if random_value <= config.percentage_RRH_mixed / 100:
        color = "orange"
    # For the ones that are for office
    elif (config.percentage_RRH_mixed / 100) < random_value <= (
            (config.percentage_RRH_mixed / 100) + config.percentage_RRH_office / 100):
        color = "blue"
    # For the residential ones
    elif ((config.percentage_RRH_mixed / 100 + config.percentage_RRH_office / 100) <
          random_value <= 1):
        color = "green"
    return color


def main():
    # Load general config
    config = Config()
    # Load the CSV file
    df = pd.read_csv(config.csv_uri)  # Use the correct CSV
    bbu_pools = []
    # Remove rows with missing coordinates
    df = df.dropna(subset=['ns1:coordinates'])
    antennas_map = []
    # Split 'ns1:coordinates' and extract longitude, latitude
    df[['longitude', 'latitude', 'altitude']] = (
        df['ns1:coordinates'].str.split(",", expand=True)[[0, 1, 2]].astype(float))

    # Create point geometry using longitude and latitude
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    rrhs_df = df[['latitude', 'longitude']].copy()
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
        # print([row['latitude'], row['longitude']])
        folium.Marker(
            icon=folium.Icon(icon="tower-cell", prefix="fa",
                             color=randomize_residential_office(config), icon_color="black"),
            location=[row['latitude'], row['longitude']],
            popup=f"Name: {row.get('ns1:name9', '')}, Altitude: {row['altitude']} m"
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
    print("BBU details saved to bbu_details.csv")


if __name__ == "__main__":
    main()
