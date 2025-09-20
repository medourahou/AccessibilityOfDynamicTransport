
import csv
import math
import os
import time
import warnings
import zipfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from geopy.distance import geodesic
from shapely import wkt
from shapely.geometry import Point, Polygon, MultiPolygon, mapping
from tqdm import tqdm


class TripDataProcessor:
    """
    A comprehensive trip data processor that handles CSV parsing, data filtering,
    time-based feature engineering, and distance calculations for transportation datasets.
    """
    
    def __init__(self, osrm_base_url: str = "http://localhost:5000"):
        """
        Initialize the processor with supported datetime formats and OSRM calculator.
        
        Args:
            osrm_base_url (str): Base URL for OSRM service
        """
        self.datetime_formats = [
            '%Y-%m-%d %H:%M:%S',  # 2025-01-01 09:43:46
            '%Y-%m-%d %H:%M',     # 2025-01-01 09:43
            '%d/%m/%Y %H:%M:%S',  # 01/01/2025 09:43:46
            '%d/%m/%Y %H:%M'      # 01/01/2025 09:43
        ]
        self.osrm_calculator = OSRMTableCalculator(osrm_base_url)
    
    def process_time_columns(self, input_file: str) -> pd.DataFrame:
        """
        Process raw trip data CSV file and convert to standardized DataFrame format.

        Args:
            input_file (str): Path to input CSV file

        Returns:
            pd.DataFrame: Processed dataframe with standardized columns
        """
        print(f"Processing time columns from {input_file}...")

        # Determine file format from extension
        file_ext = os.path.splitext(input_file)[1].lower()

        # Set delimiter based on file extension
        delimiter = ',' if file_ext == '.csv' else ';'

        # Read the input file
        df = pd.read_csv(input_file, delimiter=delimiter)

        processed_rows = []
        error_count = 0

        for idx, row in df.iterrows():
            try:
                # Parse departure time
                departure_dt = self._parse_datetime(row['departure_time'])

                # Parse arrival time
                arrival_dt = self._parse_datetime(row['arrival_time'])

                if not departure_dt or not arrival_dt:
                    raise ValueError(f"Could not parse datetime. Departure: {row['departure_time']}, Arrival: {row['arrival_time']}")

                # Calculate travel time in seconds
                travel_time = round((arrival_dt - departure_dt).total_seconds())

                # Create standardized output row
                new_row = {
                    'origin_lat': row['departure_latitude'],
                    'origin_lon': row['departure_longitude'],
                    'destination_lat': row['arrival_latitude'],
                    'destination_lon': row['arrival_longitude'],
                    'departure_hour': departure_dt.hour,
                    'departure_min': departure_dt.minute,
                    'travel_time': travel_time,
                }

                processed_rows.append(new_row)

            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                error_count += 1
                continue

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_rows)

        print(f"Processing complete: {len(processed_df)} rows processed, {error_count} rows skipped")

        return processed_df

    def _parse_datetime(self, datetime_string: str) -> Optional[datetime]:
        """
        Parse datetime string using multiple formats.
        
        Args:
            datetime_string (str): Datetime string to parse
            
        Returns:
            datetime object or None if parsing fails
        """
        for fmt in self.datetime_formats:
            try:
                return datetime.strptime(datetime_string, fmt)
            except ValueError:
                continue
        return None
    
    def filter_travel_times(self, df: pd.DataFrame, min_time: int = 60, max_time: int = 6000) -> pd.DataFrame:
        """
        Filter trips based on travel time constraints.
        
        Args:
            df (pd.DataFrame): Input dataframe with trip data
            min_time (int): Minimum travel time in seconds (default: 60)
            max_time (int): Maximum travel time in seconds (default: 6000)
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        print(f"Filtering trips with travel time between {min_time} and {max_time} seconds...")
        
        original_count = len(df)
        
        # Apply travel time filters
        filtered_df = df[
            (df["travel_time"] >= min_time) & 
            (df["travel_time"] <= max_time)
        ].copy()
        
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count
        
        print(f"Filtered dataset: {filtered_count:,} trips remaining ({removed_count:,} removed)")
        
        return filtered_df
    
    def calculate_distances_for_dataframe(self, df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
        """
        Calculate distances for all trips in the dataframe using OSRM table service.
        
        Args:
            df (pd.DataFrame): DataFrame with origin/destination coordinates
            batch_size (int): Batch size for OSRM requests
            
        Returns:
            pd.DataFrame: DataFrame with added distance column
        """
        print("Starting distance calculation...")
        
        # Extract unique coordinates
        print("Extracting unique coordinates...")
        unique_sources = df[['origin_lon', 'origin_lat']].drop_duplicates().values
        unique_dests = df[['destination_lon', 'destination_lat']].drop_duplicates().values
        
        source_coords = [(x, y) for x, y in unique_sources]
        dest_coords = [(x, y) for x, y in unique_dests]
        
        print(f"Found {len(source_coords)} unique source points")
        print(f"Found {len(dest_coords)} unique destination points")
        
        # Create coordinate lookups
        source_lookup = {(x, y): idx for idx, (x, y) in enumerate(source_coords)}
        dest_lookup = {(x, y): idx for idx, (x, y) in enumerate(dest_coords)}
        
        # Initialize distance matrix
        distance_matrix = np.full((len(source_coords), len(dest_coords)), np.nan)
        
        # Process in batches
        total_batches = ((len(source_coords) + batch_size - 1) // batch_size) * \
                        ((len(dest_coords) + batch_size - 1) // batch_size)
        
        with tqdm(total=total_batches, desc="Processing distance matrix") as pbar:
            for i in range(0, len(source_coords), batch_size):
                source_batch = source_coords[i:i + batch_size]
                
                for j in range(0, len(dest_coords), batch_size):
                    dest_batch = dest_coords[j:j + batch_size]
                    
                    # Combine coordinates and create indices
                    all_coords = source_batch + dest_batch
                    source_indices = list(range(len(source_batch)))
                    dest_indices = list(range(len(source_batch), len(all_coords)))
                    
                    # Calculate distances
                    distances = self.osrm_calculator.calculate_table_distances(
                        coordinates=all_coords,
                        sources=source_indices,
                        destinations=dest_indices
                    )
                    
                    if distances is not None:
                        # Update distance matrix
                        distance_matrix[i:i + len(source_batch), j:j + len(dest_batch)] = distances
                    
                    pbar.update(1)
        
        # Assign distances to original dataframe
        print("Assigning distances to dataframe...")
        result_df = df.copy()
        result_df['distance'] = df.apply(
            lambda row: distance_matrix[
                source_lookup[(row['origin_lon'], row['origin_lat'])],
                dest_lookup[(row['destination_lon'], row['destination_lat'])]
            ],
            axis=1
        )
        
        print(f"Distance calculation complete. Added distances for {len(result_df)} trips")
        return result_df
    
    def select_final_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Select specific columns for the final dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe with all features
            feature_columns (list): List of column names to keep
            
        Returns:
            pd.DataFrame: DataFrame with selected columns only
        """
        print(f"Selecting {len(feature_columns)} final features...")
        
        # Verify all requested columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
            feature_columns = [col for col in feature_columns if col in df.columns]
        
        return df[feature_columns].copy()
    
    def run_complete_pipeline(self, 
                        input_file: str, 
                        output_file: str,
                        min_travel_time: int = 60,
                        max_travel_time: int = 6000,
                        calculate_distances: bool = True,
                        distance_batch_size: int = 100) -> pd.DataFrame:
        """
        Execute the complete data processing pipeline entirely in memory.

        Args:
            input_file (str): Path to raw input CSV file
            output_file (str): Path for final output CSV
            min_travel_time (int): Minimum travel time filter (seconds)
            max_travel_time (int): Maximum travel time filter (seconds)
            calculate_distances (bool): Whether to calculate distances using OSRM
            distance_batch_size (int): Batch size for distance calculations

        Returns:
            pd.DataFrame: Final processed dataframe
        """
        print("="*60)
        print("STARTING COMPLETE TRIP DATA PROCESSING PIPELINE (IN MEMORY)")
        print("="*60)

        # Step 1: Process raw CSV and standardize format in memory
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at {input_file}")
            return None

        print(f"Processing time columns from {input_file}...")

        # Process data directly to DataFrame instead of intermediate file
        trips = self.process_time_columns(input_file)
        print(f"Loaded {len(trips):,} trips")

        # Step 2: Apply travel time filters
        filtered_trips = self.filter_travel_times(trips, min_travel_time, max_travel_time)

        # Step 3: Calculate distances if requested
        if calculate_distances:
            print("\nCalculating distances using OSRM...")
            filtered_trips = self.calculate_distances_for_dataframe(filtered_trips, distance_batch_size)

        # Step 4: Reconstruct datetime columns for feature engineering
        print("\nReconstructing datetime columns...")
        filtered_trips['departure_time'] = pd.to_datetime(
            filtered_trips['departure_hour'].astype(str) + ':' + 
            filtered_trips['departure_min'].astype(str),
            format='%H:%M'
        )

       
        # In practice, you might want to reconstruct this differently
        filtered_trips['arrival_time'] = filtered_trips['departure_time'] + pd.to_timedelta(filtered_trips['travel_time'], unit='s')

        # Step 5: Engineer time features
        time_columns = ['departure_time']
        trips_with_features = engineer_time_features(
            df=filtered_trips, 
            time_columns=time_columns, 
            is_generated_data=False
        )

        # Step 6: Select final feature set
        final_features = [
            'origin_lat', 'origin_lon', 'destination_lat', 'destination_lon',
            'travel_time', 'departure_time_hour', 'departure_time_minute',
            'departure_time_seconds', 'departure_time_day_of_week', 
            'departure_time_day_of_month', 'departure_time_month', 
            'departure_time_hour_sin', 'departure_time_hour_cos', 
            'departure_time_day_of_week_sin', 'departure_time_day_of_week_cos', 
            'departure_time_month_sin', 'departure_time_month_cos'
        ]

        # Add distance to final features if calculated
        if calculate_distances and 'distance' in trips_with_features.columns:
            final_features.append('distance')

        final_dataset = self.select_final_features(trips_with_features, final_features)

        # Step 7: Save final dataset
        final_dataset.to_csv(output_file, index=False)
        print(f"\nFinal dataset saved to {output_file}")
        print(f"Final dataset shape: {final_dataset.shape}")
        print(f"Final features: {list(final_dataset.columns)}")

        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)

        return final_dataset
    
    def process_dataframe_with_distances(self, 
                                       df: pd.DataFrame, 
                                       batch_size: int = 100) -> pd.DataFrame:
        """
        Process an existing dataframe by adding distance calculations.
        This method works directly with dataframes without file I/O.
        
        Args:
            df (pd.DataFrame): Input dataframe with coordinate columns
            batch_size (int): Batch size for OSRM requests
            
        Returns:
            pd.DataFrame: DataFrame with added distance column
        """
        print("="*60)
        print("PROCESSING DATAFRAME WITH DISTANCE CALCULATIONS")
        print("="*60)
        
        # Check for coordinate columns (handle both naming conventions)
        has_original_names = all(col in df.columns for col in ['departure_longitude', 'departure_latitude', 'arrival_longitude', 'arrival_latitude'])
        has_new_names = all(col in df.columns for col in ['origin_lon', 'origin_lat', 'destination_lon', 'destination_lat'])
        
        if not (has_original_names or has_new_names):
            print("Error: Missing required coordinate columns")
            print("Expected either: ['departure_longitude', 'departure_latitude', 'arrival_longitude', 'arrival_latitude']")
            print("Or: ['origin_lon', 'origin_lat', 'destination_lon', 'destination_lat']")
            return df
        
        # Calculate distances
        result_df = self.calculate_distances_for_dataframe(df, batch_size)
        
        print(f"\nDistance calculation complete!")
        print(f"Processed {len(result_df)} trips")
        print(f"Columns: {list(result_df.columns)}")
        
        return result_df

class DRTVrirualLinesGenerator:
    """Main class for conducting DRT-CPT accessibility analysis."""
    
    def __init__(self, 
                 hexagon_edge_km: float = 1.0,
                 walking_speed_ms: float = 1.31,
                 osrm_url: str = "http://localhost:5000",
                 buffer_seconds: int = 30):
        """Initialize the DRTVrirualLinesGenerator."""
        self.hexagon_edge_km = hexagon_edge_km
        self.walking_speed_ms = walking_speed_ms
        self.osrm_url = osrm_url
        self.buffer_seconds = buffer_seconds
        
        # Initialize components
        self.grid_generator = HexagonalGridGenerator(grid_edge_km=hexagon_edge_km)
        self.router = None
        
        # Storage for intermediate results
        self.hexagons_df = None
        self.walking_times_df = None
        self.drt_trips_df = None
    
    def generate_hexagonal_grid(self, bbox: List[float], region_name: str = "study_area") -> gpd.GeoDataFrame:
        """Generate hexagonal grid for the study area."""
        print(f"Generating hexagonal grid for {region_name}...")
        
        # Generate hexagon data
        hexagon_data = self.grid_generator.generate_grid(bbox, region_name)
        
        # Convert to GeoDataFrame
        self.hexagons_df = gpd.GeoDataFrame(hexagon_data, crs="EPSG:4326")
        
        print(f"Generated {len(self.hexagons_df)} hexagons")
        return self.hexagons_df
    
    def calculate_barycenters(self, drt_trips_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Calculate barycenters of DRT pickup/dropoff points for each hexagon."""
        print("Calculating barycenters of DRT stops...")
        
        if self.hexagons_df is None:
            raise ValueError("Generate hexagonal grid first using generate_hexagonal_grid()")
        
        # Extract DRT stops from trips
        drt_stops_df = self._extract_drt_stops_from_trips(drt_trips_df)
        
        # Assign stops to hexagons
        stops_in_hexagons = self._assign_stops_to_hexagons(self.hexagons_df, drt_stops_df)
        
        # Calculate barycenters
        self.hexagons_df = self._calculate_barycenters_for_hexagons(self.hexagons_df, stops_in_hexagons)
        
        print(f"Calculated barycenters for {self.hexagons_df['has_drt_stops'].sum()} hexagons with DRT stops")
        return self.hexagons_df
    
    def compute_all_walking_times_optimized(self, 
                                          cpt_stops_df: Optional[pd.DataFrame] = None,
                                          batch_size: int = 100) -> pd.DataFrame:
        """
        Compute all walking times using a single batch operation with OSRMTableCalculator.
        
        This optimized version merges all walking time calculations:
        - Centroid to Barycenter (within same hexagon)
        - Centroid to Centroid (between different hexagons) 
        - Barycenter to Barycenter (between different hexagons)
        - Points to CPT stops (if provided)
        
        Parameters:
        -----------
        cpt_stops_df : pd.DataFrame, optional
            CPT stops data
        batch_size : int
            Batch size for OSRM table calculations
            
        Returns:
        --------
        pd.DataFrame
            All walking time connections
        """
        print("Computing all walking times using optimized batch operation...")
        
        if self.hexagons_df is None:
            raise ValueError("Generate hexagonal grid and calculate barycenters first")
        
        # Prepare all coordinate pairs and connection metadata
        coordinate_pairs, connections_metadata = self._prepare_all_coordinate_pairs(cpt_stops_df)
        
        if len(coordinate_pairs) == 0:
            print("No coordinate pairs to process")
            self.walking_times_df = pd.DataFrame()
            return self.walking_times_df
        
        # Use OSRMTableCalculator for batch processing
        print(f"Processing {len(coordinate_pairs)} coordinate pairs in batches...")
        
        
        # Create coordinate list and source/destination indices
        all_coordinates = []
        coord_to_index = {}
        
        # Build unique coordinate list
        for pair in coordinate_pairs:
            for coord in [pair['from_coord'], pair['to_coord']]:
                coord_tuple = (coord[0], coord[1])
                if coord_tuple not in coord_to_index:
                    coord_to_index[coord_tuple] = len(all_coordinates)
                    all_coordinates.append(coord_tuple)
        
        # Process in batches using table service
        calculator = OSRMTableCalculator(base_url=self.osrm_url)
        
        all_walking_times = []
        
        # Process coordinate pairs in chunks
        for i in tqdm(range(0, len(coordinate_pairs), batch_size), desc="Processing walking times"):
            batch_pairs = coordinate_pairs[i:i + batch_size]
            
            # Extract unique coordinates for this batch
            batch_coords = set()
            for pair in batch_pairs:
                batch_coords.add((pair['from_coord'][0], pair['from_coord'][1]))
                batch_coords.add((pair['to_coord'][0], pair['to_coord'][1]))
            
            batch_coords = list(batch_coords)
            batch_coord_to_index = {coord: idx for idx, coord in enumerate(batch_coords)}
            
            # Create source and destination indices for this batch
            sources = []
            destinations = []
            pair_mappings = []
            
            for pair_idx, pair in enumerate(batch_pairs):
                from_coord = (pair['from_coord'][0], pair['from_coord'][1])
                to_coord = (pair['to_coord'][0], pair['to_coord'][1])
                
                from_idx = batch_coord_to_index[from_coord]
                to_idx = batch_coord_to_index[to_coord]
                
                if from_idx not in sources:
                    sources.append(from_idx)
                if to_idx not in destinations:
                    destinations.append(to_idx)
                
                pair_mappings.append({
                    'pair_idx': pair_idx,
                    'from_idx': from_idx,
                    'to_idx': to_idx,
                    'pair': pair
                })
            
            # Calculate distance matrix for this batch
            distance_matrix = calculator.calculate_table_distances(
                coordinates=batch_coords,
                sources=sources,
                destinations=destinations
            )
            
            if distance_matrix is not None:
                # Extract walking times for each pair
                for mapping in pair_mappings:
                    try:
                        source_pos = sources.index(mapping['from_idx'])
                        dest_pos = destinations.index(mapping['to_idx'])
                        distance_meters = distance_matrix[source_pos][dest_pos]
                        
                        if distance_meters is not None and not np.isnan(distance_meters):
                            walking_time = distance_meters / self.walking_speed_ms + self.buffer_seconds
                            
                            walking_time_data = {
                                "from_id": mapping['pair']['from_id'],
                                "to_id": mapping['pair']['to_id'],
                                "from_x": mapping['pair']['from_coord'][0],
                                "from_y": mapping['pair']['from_coord'][1],
                                "to_x": mapping['pair']['to_coord'][0],
                                "to_y": mapping['pair']['to_coord'][1],
                                "walking_time_seconds": walking_time,
                                "road_distance":distance_meters,
                                "connection_type": mapping['pair']['connection_type']
                            }
                            all_walking_times.append(walking_time_data)
                    except (ValueError, IndexError) as e:
                        print(f"Error processing pair {mapping['pair_idx']}: {e}")
                        continue
        
        # Convert to DataFrame and add bidirectional connections
        self.walking_times_df = pd.DataFrame(all_walking_times)
        if len(self.walking_times_df) > 0:
            self.walking_times_df = self._add_bidirectional_connections(self.walking_times_df)
        
        print(f"Computed {len(self.walking_times_df)} walking time connections")
        return self.walking_times_df
    
    def _prepare_all_coordinate_pairs(self, cpt_stops_df: Optional[pd.DataFrame] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Prepare all coordinate pairs for batch processing.
        
        Returns:
        --------
        Tuple[List[Dict], List[Dict]]
            List of coordinate pairs and their metadata
        """
        coordinate_pairs = []
        hexagons_with_stops = self.hexagons_df[self.hexagons_df['has_drt_stops']]
        
        # 1. Centroid to Barycenter (within same hexagon)
        for idx, hexagon in hexagons_with_stops.iterrows():
            centroid_x, centroid_y = hexagon["Centroid_X"], hexagon["Centroid_Y"]
            barycenter_x, barycenter_y = hexagon["Barycenter_X"], hexagon["Barycenter_Y"]
            
            # Skip if coordinates are identical (will be set to 0 walking time)
            if abs(centroid_x - barycenter_x) > 0.00001 or abs(centroid_y - barycenter_y) > 0.00001:
                coordinate_pairs.append({
                    'from_id': f"centroid_{idx}",
                    'to_id': f"barycenter_{idx}",
                    'from_coord': [centroid_x, centroid_y],
                    'to_coord': [barycenter_x, barycenter_y],
                    'connection_type': 'centroid_to_barycenter'
                })
        
        # 2. Centroid to Centroid (between different hexagons)
        hex_indices = list(hexagons_with_stops.index)
        for i, hex_idx1 in enumerate(hex_indices):
            for j, hex_idx2 in enumerate(hex_indices):
                if i >= j:  # Avoid duplicates and self-connections
                    continue
                    
                hex1 = hexagons_with_stops.loc[hex_idx1]
                hex2 = hexagons_with_stops.loc[hex_idx2]
                
                coordinate_pairs.append({
                    'from_id': f"centroid_{hex_idx1}",
                    'to_id': f"centroid_{hex_idx2}",
                    'from_coord': [hex1["Centroid_X"], hex1["Centroid_Y"]],
                    'to_coord': [hex2["Centroid_X"], hex2["Centroid_Y"]],
                    'connection_type': 'centroid_to_centroid'
                })
        
        # 3. Barycenter to Barycenter (between different hexagons)
        for i, hex_idx1 in enumerate(hex_indices):
            for j, hex_idx2 in enumerate(hex_indices):
                if i >= j:  # Avoid duplicates and self-connections
                    continue
                    
                hex1 = hexagons_with_stops.loc[hex_idx1]
                hex2 = hexagons_with_stops.loc[hex_idx2]
                
                coordinate_pairs.append({
                    'from_id': f"barycenter_{hex_idx1}",
                    'to_id': f"barycenter_{hex_idx2}",
                    'from_coord': [hex1["Barycenter_X"], hex1["Barycenter_Y"]],
                    'to_coord': [hex2["Barycenter_X"], hex2["Barycenter_Y"]],
                    'connection_type': 'barycenter_to_barycenter'
                })
        
        # 4. Points to CPT stops (if provided)
        if cpt_stops_df is not None:
            for hex_idx, hexagon in hexagons_with_stops.iterrows():
                for _, cpt_stop in cpt_stops_df.iterrows():
                    stop_id = cpt_stop.get("stop_id", f"cpt_stop_{_}")
                    
                    # Centroid to CPT stop
                    coordinate_pairs.append({
                        'from_id': f"centroid_{hex_idx}",
                        'to_id': f"cpt_stop_{stop_id}",
                        'from_coord': [hexagon["Centroid_X"], hexagon["Centroid_Y"]],
                        'to_coord': [cpt_stop["stop_lon"], cpt_stop["stop_lat"]],
                        'connection_type': 'centroid_to_cpt_stop'
                    })
                    
                    # Barycenter to CPT stop
                    coordinate_pairs.append({
                        'from_id': f"barycenter_{hex_idx}",
                        'to_id': f"cpt_stop_{stop_id}",
                        'from_coord': [hexagon["Barycenter_X"], hexagon["Barycenter_Y"]],
                        'to_coord': [cpt_stop["stop_lon"], cpt_stop["stop_lat"]],
                        'connection_type': 'barycenter_to_cpt_stop'
                    })
        
        return coordinate_pairs, []
    
    
    def compute_all_walking_times(self, cpt_stops_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Backward compatibility wrapper - calls the optimized version."""
        return self.compute_all_walking_times_optimized(cpt_stops_df)
    
    def generate_drt_trips(self, 
                          cpt_stops_df: Optional[pd.DataFrame] = None,
                          cpt_schedule_df: Optional[pd.DataFrame] = None,
                          time_window: Tuple[str, str] = ("07:00:00", "10:00:00"),
                          max_walking_time_min: int = 10,
                          base_departure_hour: int = 7) -> pd.DataFrame:
        """
        Generate synthetic DRT trips for first and last mile connections.
        
        Parameters:
        -----------
        cpt_stops_df : pd.DataFrame, optional
            CPT stops data
        cpt_schedule_df : pd.DataFrame, optional
            CPT schedule data with columns: stop_id, arrival_time
        time_window : Tuple[str, str]
            Time window for analysis (start_time, end_time)
        max_walking_time_min : int
            Maximum walking time to CPT stops in minutes
        base_departure_hour : int
            Base departure hour to add to all trips (e.g., 7 for 7:00 AM)
            
        Returns:
        --------
        pd.DataFrame
            Synthetic DRT trips dataset
        """
        print("Generating synthetic DRT trips...")
        
        if self.walking_times_df is None:
            raise ValueError("Compute walking times first using compute_all_walking_times()")
        
        all_trips = []
        base_departure_seconds = base_departure_hour * 3600  # Convert hour to seconds
        
        # Parse time window
        start_time = datetime.strptime(time_window[0], "%H:%M:%S")
        end_time = datetime.strptime(time_window[1], "%H:%M:%S")
        start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
        end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second
        
        # 1. DRT as First Mile (Barycenter to Barycenter)
        print("  Generating first mile DRT trips...")
        first_mile_trips = self._generate_first_mile_trips(base_departure_seconds, start_seconds, end_seconds)
        all_trips.extend(first_mile_trips)
        
        # 2. DRT as Last Mile (Barycenter to CPT stops)
        if cpt_stops_df is not None and cpt_schedule_df is not None:
            print("  Generating last mile DRT trips...")
            last_mile_trips = self._generate_last_mile_trips(
                cpt_stops_df, cpt_schedule_df, time_window, max_walking_time_min, 
                base_departure_seconds, start_seconds, end_seconds
            )
            all_trips.extend(last_mile_trips)
        
        self.drt_trips_df = pd.DataFrame(all_trips)
        print(f"Generated {len(self.drt_trips_df)} synthetic DRT trips")
        
        return self.drt_trips_df
    
    def _extract_drt_stops_from_trips(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """Extract departure and arrival points from DRT trips."""
        # Standardize column names
        required_cols = ["origin_lat", "origin_lon","destination_lat","destination_lon"]
        
        # Check if columns exist or need renaming
        col_mapping = {
            "departure_stop_lon": "origin_lon",
            "departure_stop_lat": "origin_lat",
            "arrival_stop_lon": "destination_lon", 
            "arrival_stop_lat": "destination_lat"
        }
        
        trips_df = trips_df.rename(columns=col_mapping)
        
        if not all(col in trips_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in trips_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Extract departure points
        departures = trips_df[["origin_lon", "origin_lat"]].copy()
        departures.columns = ["stop_lon", "stop_lat"]
        departures["stop_type"] = "departure"
        
        # Extract arrival points
        arrivals = trips_df[["destination_lon", "destination_lat"]].copy()
        arrivals.columns = ["stop_lon", "stop_lat"]
        arrivals["stop_type"] = "arrival"
        
        # Combine
        all_stops = pd.concat([departures, arrivals], ignore_index=True)
        
        return all_stops
    
    def _assign_stops_to_hexagons(self, hexagons_df: gpd.GeoDataFrame, stops_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Assign DRT stops to hexagons using spatial join."""
        # Create GeoDataFrame for stops
        stops_gdf = gpd.GeoDataFrame(
            stops_df, 
            geometry=gpd.points_from_xy(stops_df.stop_lon, stops_df.stop_lat),
            crs="EPSG:4326"
        )
        
        # Spatial join
        stops_in_hexagons = gpd.sjoin(stops_gdf, hexagons_df, predicate="within")
        
        return stops_in_hexagons
    
    def _calculate_barycenters_for_hexagons(self, hexagons_df: gpd.GeoDataFrame, 
                                          stops_in_hexagons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Calculate barycenter coordinates for each hexagon."""
        hexagons_with_barycenters = hexagons_df.copy()
        
        # Initialize barycenter columns with centroid values
        hexagons_with_barycenters["Barycenter_X"] = hexagons_with_barycenters["Centroid_X"]
        hexagons_with_barycenters["Barycenter_Y"] = hexagons_with_barycenters["Centroid_Y"]
        hexagons_with_barycenters["has_drt_stops"] = False
        
        # Calculate barycenters for hexagons with stops
        for hex_id in stops_in_hexagons["index_right"].unique():
            hex_stops = stops_in_hexagons[stops_in_hexagons["index_right"] == hex_id]
            
            if len(hex_stops) > 0:
                barycenter_x = hex_stops["stop_lon"].mean()
                barycenter_y = hex_stops["stop_lat"].mean()
                
                hexagons_with_barycenters.loc[hex_id, "Barycenter_X"] = barycenter_x
                hexagons_with_barycenters.loc[hex_id, "Barycenter_Y"] = barycenter_y
                hexagons_with_barycenters.loc[hex_id, "has_drt_stops"] = True
        
        return hexagons_with_barycenters
    
    def _add_bidirectional_connections(self, walking_times_df: pd.DataFrame) -> pd.DataFrame:
        """Add reverse connections with same walking times."""
        reverse_connections = []
        
        for _, row in walking_times_df.iterrows():
            # Create reverse connection
            reverse_row = {
                "from_id": row["to_id"],
                "to_id": row["from_id"],
                "from_x": row["to_x"],
                "from_y": row["to_y"],
                "to_x": row["from_x"],
                "to_y": row["from_y"],
                "walking_time_seconds": row["walking_time_seconds"],
                "road_distance": row["road_distance"],
                "connection_type": f"{row['connection_type']}_reverse"
            }
            reverse_connections.append(reverse_row)
        
        # Combine original and reverse connections
        all_connections = pd.concat([walking_times_df, pd.DataFrame(reverse_connections)], ignore_index=True)
        
        return all_connections
    
    def _generate_first_mile_trips(self, base_departure_seconds: int, 
                                  start_seconds: int, end_seconds: int) -> List[Dict]:
        """Generate DRT trips as first mile connections (barycenter to barycenter)."""
        trips = []
        hexagons_with_stops = self.hexagons_df[self.hexagons_df["has_drt_stops"]]
        
        # Get centroid to barycenter walking times for departure time calculation
        centroid_to_barycenter_times = self.walking_times_df[
            self.walking_times_df["connection_type"] == "centroid_to_barycenter"
        ]
        
        for i, hex1 in hexagons_with_stops.iterrows():
            for j, hex2 in hexagons_with_stops.iterrows():
                if i == j:  # Skip same hexagon
                    continue
                
                # Get walking time from centroid to barycenter for departure hexagon
                centroid_barycenter_time = centroid_to_barycenter_times[
                    centroid_to_barycenter_times["from_id"] == f"centroid_{i}"
                ]
                
                if len(centroid_barycenter_time) == 0:
                    departure_buffer = self.buffer_seconds
                else:
                    departure_buffer = centroid_barycenter_time.iloc[0]["walking_time_seconds"]
                
                # Calculate final departure time in seconds from midnight
                departure_time_seconds = base_departure_seconds + departure_buffer
                
                # Check if departure time falls within the specified time window
                if start_seconds <= departure_time_seconds <= end_seconds:
                    # Convert seconds to HH:MM:SS format
                    hours = int(departure_time_seconds // 3600)
                    minutes = int((departure_time_seconds % 3600) // 60)
                    seconds = int(departure_time_seconds % 60)
                    departure_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    # Create trip from barycenter i to barycenter j
                    trips.append({
                        "start_x": hex1["Barycenter_X"],
                        "start_y": hex1["Barycenter_Y"],
                        "end_x": hex2["Barycenter_X"],
                        "end_y": hex2["Barycenter_Y"],
                        "departure_time": departure_time_str,
                        "road_distance": centroid_barycenter_time.iloc[0]["road_distance"],
                        "trip_type": "first_mile_drt",
                        "from_hexagon_id": i,
                        "to_hexagon_id": j
                    })
        
        return trips
    
    def _generate_last_mile_trips(self, 
                             cpt_stops_df: pd.DataFrame,
                             cpt_schedule_df: pd.DataFrame,
                             time_window: Tuple[str, str],
                             max_walking_time_min: int,
                             base_departure_seconds: int,
                             start_seconds: int,
                             end_seconds: int) -> List[Dict]:
        """Generate DRT trips as last mile connections (barycenter to same barycenter via CPT)."""
        

        trips = []
        max_walking_time_sec = max_walking_time_min * 60
        hexagons_with_stops = self.hexagons_df[self.hexagons_df["has_drt_stops"]]


        # Filter CPT schedule by time window
        start_time = datetime.strptime(time_window[0], "%H:%M:%S").time()
        end_time = datetime.strptime(time_window[1], "%H:%M:%S").time()



        # Convert arrival_time to time format for filtering
        cpt_schedule_df = cpt_schedule_df.copy()

        # Handle GTFS times that can exceed 24:00:00 (late night services)
        def normalize_gtfs_time(time_str):
            """Convert GTFS time format (can be >24:00:00) to standard time format."""
            try:
                hours, minutes, seconds = map(int, str(time_str).split(':'))
                if hours >= 24:
                    hours = hours - 24  # Convert to next day equivalent
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            except Exception as e:
                print(f"DEBUG: Error normalizing time '{time_str}': {e}")
                return "00:00:00"

        

        cpt_schedule_df["normalized_time"] = cpt_schedule_df["arrival_time"].apply(normalize_gtfs_time)
        cpt_schedule_df["arrival_time_obj"] = pd.to_datetime(cpt_schedule_df["normalized_time"], format="%H:%M:%S").dt.time

        

        filtered_schedule = cpt_schedule_df[
            (cpt_schedule_df["arrival_time_obj"] >= start_time) & 
            (cpt_schedule_df["arrival_time_obj"] <= end_time)
        ]

       

        if len(filtered_schedule) == 0:
            return trips

        # Debug walking times data
        barycenter_cpt_connections = self.walking_times_df[
            self.walking_times_df["connection_type"] == "barycenter_to_cpt_stop"
        ]

       
        if len(barycenter_cpt_connections) == 0:
           
            return trips

        # Main generation loop
        total_pairs = len(hexagons_with_stops) * (len(hexagons_with_stops) - 1)
       

        pair_count = 0
        trips_created = 0

        for i, hex1 in hexagons_with_stops.iterrows():
            for j, hex2 in hexagons_with_stops.iterrows():
                if i == j:  # Skip same hexagon
                    continue

                pair_count += 1
                
                # Find CPT stops accessible from hex1 barycenter within max walking time
                barycenter_to_cpt_times = self.walking_times_df[
                    (self.walking_times_df["from_id"] == f"barycenter_{i}") &
                    (self.walking_times_df["connection_type"] == "barycenter_to_cpt_stop") &
                    (self.walking_times_df["walking_time_seconds"] <= max_walking_time_sec)
                ]

               

                if len(barycenter_to_cpt_times) == 0:
                    
                    continue

                # Show walking times for debugging
                if len(barycenter_to_cpt_times) > 0:
                   
                    for _, conn in barycenter_to_cpt_times.head(3).iterrows():
                        stop_id = conn["to_id"].replace("cpt_stop_", "")
                        walking_time_min = conn["walking_time_seconds"] / 60
                       

                for idx, cpt_connection in barycenter_to_cpt_times.iterrows():
                    # Extract stop_id from to_id
                    cpt_stop_id = cpt_connection["to_id"].replace("cpt_stop_", "")

                    # Get walking time from CPT stop back to departure barycenter (hex1)
                    walking_time_from_cpt_to_departure = cpt_connection["walking_time_seconds"]
                    road_distance_from_cpt_to_departure = cpt_connection["road_distance"]

                    # Get CPT vehicle arrivals at this stop
                    stop_schedule = filtered_schedule[filtered_schedule["stop_id"] == cpt_stop_id]

                    

                    if len(stop_schedule) == 0:
                       
                        continue

                    

                    trips_for_this_stop = 0

                    for sched_idx, schedule_row in stop_schedule.iterrows():
                        try:
                            # Calculate departure time
                            cpt_arrival_time = datetime.strptime(schedule_row["arrival_time"], "%H:%M:%S")
                            cpt_arrival_seconds = (cpt_arrival_time.hour * 3600 + 
                                                 cpt_arrival_time.minute * 60 + 
                                                 cpt_arrival_time.second)

                            departure_time_seconds = ( 
                                                    cpt_arrival_seconds + 
                                                    self.buffer_seconds + 
                                                    walking_time_from_cpt_to_departure)

                            # Check if departure time falls within the specified time window
                            if start_seconds <= departure_time_seconds <= end_seconds:
                                # Convert seconds to HH:MM:SS format
                                hours = int(departure_time_seconds // 3600)
                                minutes = int((departure_time_seconds % 3600) // 60)
                                seconds = int(departure_time_seconds % 60)
                                departure_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                                trip_data = {
                                    "start_x": hex1["Barycenter_X"],
                                    "start_y": hex1["Barycenter_Y"],
                                    "end_x": hex2["Barycenter_X"],
                                    "end_y": hex2["Barycenter_Y"],
                                    "departure_time": departure_time_str,
        
                                    
                                    "road_distance": road_distance_from_cpt_to_departure,
                                    "trip_type": "last_mile_drt",
                                    "from_hexagon_id": i,
                                    "to_hexagon_id": j,
                                    "via_cpt_stop": cpt_stop_id,
                                    "cpt_arrival_time": schedule_row["arrival_time"]
                                }

                                trips.append(trip_data)
                                trips_for_this_stop += 1
                                trips_created += 1

                               

                            else:
                                if trips_for_this_stop == 0:  # Only show first failed attempt per stop
                                   
                                    dep_time_str = f"{int(departure_time_seconds//3600):02d}:{int((departure_time_seconds%3600)//60):02d}:{int(departure_time_seconds%60):02d}"
                                    

                        except Exception as e:
                            continue

                

        return trips
    
    def save_results(self, output_dir: str):
        """Save all results to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.hexagons_df is not None:
            # Convert geometry to WKT for saving
            hexagons_save = self.hexagons_df.copy()
            if 'geometry' in hexagons_save.columns:
                hexagons_save['geometry'] = hexagons_save['geometry'].apply(lambda x: x.wkt if x is not None else None)
            hexagons_save.to_csv(os.path.join(output_dir, "hexagons_with_barycenters.csv"), index=False)
        
        if self.walking_times_df is not None:
            self.walking_times_df.to_csv(os.path.join(output_dir, "walking_times.csv"), index=False)
        
        if self.drt_trips_df is not None:
            self.drt_trips_df.to_csv(os.path.join(output_dir, "synthetic_drt_trips.csv"), index=False)
        
        print(f"Results saved to {output_dir}")


class HexagonalGridGenerator:
    """Generate hexagonal grids for geographical areas."""
    
    def __init__(self, grid_edge_km: float = 1):
        self.grid_edge = grid_edge_km
        self.cosines = [math.cos(2 * math.pi / 6 * i) for i in range(6)]
        self.sines = [math.sin(2 * math.pi / 6 * i) for i in range(6)]
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate geodesic distance between two points."""
        return geodesic((point1[1], point1[0]), (point2[1], point2[0])).kilometers
    
    def create_hexagon(self, center: List[float], rx: float, ry: float) -> Polygon:
        """Create a hexagon polygon centered at the given point."""
        vertices = []
        for i in range(6):
            x = center[0] + rx * self.cosines[i]
            y = center[1] + ry * self.sines[i]
            vertices.append([x, y])
        vertices.append(vertices[0])  # Close the polygon
        return Polygon(vertices)
    
    def generate_grid(self, bbox: List[float], region: str) -> List[Dict[str, Any]]:
        """Generate hexagonal grid for the given bounding box."""
        # Calculate grid dimensions
        x_fraction = self.grid_edge / self.calculate_distance((bbox[0], bbox[1]), (bbox[2], bbox[1]))
        cell_width = x_fraction * (bbox[2] - bbox[0])
        y_fraction = self.grid_edge / self.calculate_distance((bbox[0], bbox[1]), (bbox[0], bbox[3]))
        cell_height = y_fraction * (bbox[3] - bbox[1])

        radius = cell_width / 2
        hex_width = radius * 2
        hex_height = math.sqrt(3) / 2 * cell_height

        # Calculate grid parameters
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        x_interval = 3 / 4 * hex_width
        y_interval = hex_height

        # Calculate grid size
        x_span = box_width / (hex_width - radius / 2)
        x_count = int(math.ceil(x_span))
        if round(x_span) == x_count:
            x_count += 1

        # Calculate adjustments
        x_adjust = ((x_count * x_interval - radius / 2) - box_width) / 2 - radius / 2
        y_count = int(math.ceil(box_height / hex_height))
        y_adjust = (box_height - y_count * hex_height) / 2

        has_offset_y = y_count * hex_height - box_height > hex_height / 2
        if has_offset_y:
            y_adjust -= hex_height / 4

        # Generate hexagons
        cell_data = []
        count = 0

        for x in range(x_count):
            for y in range(y_count + 1):
                is_odd = x % 2 == 1

                if (y == 0 and is_odd) or (y == 0 and has_offset_y):
                    continue

                center_x = x * x_interval + bbox[0] - x_adjust
                center_y = y * y_interval + bbox[1] + y_adjust

                if is_odd:
                    center_y -= hex_height / 2

                hex_poly = self.create_hexagon([center_x, center_y], cell_width / 2, cell_height / 2)

                cell_data.append({
                    "id": count,
                    "Centroid_X": center_x,
                    "Centroid_Y": center_y,
                    "centroid": Point(center_x, center_y),
                    "geometry": hex_poly,
                    "region": region,
                })

                count += 1

        return cell_data




class OSRMTableCalculator:
    """A class to calculate distances between points using the OSRM Table service."""

    def __init__(self, base_url: str = "http://localhost:5000"):
        """Initialize the OSRM Table Calculator.

        Args
        ----
            base_url: Base URL for the OSRM service
        """
        self.base_url = base_url
        self.session = requests.Session()

    def calculate_table_distances(
        self, coordinates: List[Tuple[float, float]], sources: List[int], destinations: List[int]
    ) -> Optional[np.ndarray]:
        """Calculate distances using OSRM table service.

        Args
        ----
            coordinates: List of coordinate pairs (longitude, latitude)
            sources: Indices of source coordinates
            destinations: Indices of destination coordinates

        Returns
        -------
            Matrix of distances, or None if an error occurred
        """
        url = f"{self.base_url}/table/v1/driving/"

        # Prepare coordinates string
        coords_str = ";".join([f"{lon},{lat}" for lon, lat in coordinates])

        # Full URL with parameters
        full_url = f"{url}{coords_str}"
        params = {
            "sources": ";".join(map(str, sources)),
            "destinations": ";".join(map(str, destinations)),
            "annotations": "duration,distance",
        }

        try:
            response = self.session.get(full_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data["code"] != "Ok":
                raise Exception(f"OSRM error: {data['code']}")

            return np.array(data["distances"])

        except Exception as e:
            print(f"Error in table calculation: {str(e)}")
            return None




def load_cpt_stops_from_gtfs(gtfs_path: str) -> pd.DataFrame:
    """
    Load CPT stops from GTFS zip file.
    
    Parameters:
    -----------
    gtfs_path : str
        Path to GTFS zip file
        
    Returns:
    --------
    pd.DataFrame
        CPT stops with stop_id, stop_lon, stop_lat
    """
    try:
        with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
            # Check if stops.txt exists in the zip file
            if "stops.txt" not in zip_ref.namelist():
                raise ValueError("GTFS zip file does not contain stops.txt")
            
            # Extract stops.txt and read it
            with zip_ref.open("stops.txt") as stops_file:
                stops_df = pd.read_csv(stops_file)
                
                # Ensure required columns exist
                if "stop_lon" not in stops_df.columns or "stop_lat" not in stops_df.columns:
                    raise ValueError("GTFS stops.txt file missing required columns: stop_lon, stop_lat")
                
                # Ensure stop_id exists
                if "stop_id" not in stops_df.columns:
                    raise ValueError("GTFS stops.txt file missing required column: stop_id")
        
        print(f"Loaded {len(stops_df)} CPT stops from GTFS")
        return stops_df
        
    except Exception as e:
        print(f"Error loading CPT stops from GTFS: {e}")
        raise


def load_cpt_schedule_from_gtfs(gtfs_path: str, service_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load CPT schedule from GTFS zip file.
    
    Parameters:
    -----------
    gtfs_path : str
        Path to GTFS zip file
    service_date : str, optional
        Service date in YYYYMMDD format (e.g., "20241215")
        If None, loads all stop times without date filtering
        
    Returns:
    --------
    pd.DataFrame
        Schedule with stop_id, arrival_time
    """
    try:
        with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
            # Load stop_times.txt
            if "stop_times.txt" not in zip_ref.namelist():
                raise ValueError("GTFS zip file does not contain stop_times.txt")
            
            with zip_ref.open("stop_times.txt") as stop_times_file:
                stop_times_df = pd.read_csv(stop_times_file)
            
            # Check required columns
            required_columns = ["stop_id", "arrival_time"]
            missing_columns = [col for col in required_columns if col not in stop_times_df.columns]
            if missing_columns:
                raise ValueError(f"GTFS stop_times.txt missing required columns: {missing_columns}")
            
            # If service_date is specified, filter by active services
            if service_date:
                # Load calendar and trips for service filtering
                schedule_df = _filter_by_service_date(zip_ref, stop_times_df, service_date)
            else:
                # Use all stop times without date filtering
                schedule_df = stop_times_df[["stop_id", "arrival_time"]].copy()
            
            # Remove duplicates and sort
            schedule_df = schedule_df.drop_duplicates().sort_values(["stop_id", "arrival_time"])
            
        print(f"Loaded {len(schedule_df)} schedule entries from GTFS")
        return schedule_df
        
    except Exception as e:
        print(f"Error loading CPT schedule from GTFS: {e}")
        raise

def _filter_by_service_date(zip_ref: zipfile.ZipFile, stop_times_df: pd.DataFrame, service_date: str) -> pd.DataFrame:
    """
    Filter stop times by service date using GTFS calendar and trips data.
    
    Parameters:
    -----------
    zip_ref : zipfile.ZipFile
        Open GTFS zip file
    stop_times_df : pd.DataFrame
        Stop times data
    service_date : str
        Service date in YYYYMMDD format
        
    Returns:
    --------
    pd.DataFrame
        Filtered schedule data
    """
    try:
        # Load trips.txt to get service_id for each trip
        if "trips.txt" not in zip_ref.namelist():
            print("Warning: trips.txt not found, returning all stop times")
            return stop_times_df[["stop_id", "arrival_time"]].copy()
        
        with zip_ref.open("trips.txt") as trips_file:
            trips_df = pd.read_csv(trips_file)
        
        # Load calendar.txt to check which services are active on the given date
        active_services = set()
        
        if "calendar.txt" in zip_ref.namelist():
            with zip_ref.open("calendar.txt") as calendar_file:
                calendar_df = pd.read_csv(calendar_file)
            
            # Convert service_date to datetime for comparison
            target_date = pd.to_datetime(service_date, format="%Y%m%d")
            day_of_week = target_date.strftime("%A").lower()
            
            # Check which services are active on this day
            for _, service in calendar_df.iterrows():
                start_date = pd.to_datetime(str(service['start_date']), format="%Y%m%d")
                end_date = pd.to_datetime(str(service['end_date']), format="%Y%m%d")
                
                # Check if service is active on this date and day of week
                if (start_date <= target_date <= end_date and 
                    service.get(day_of_week, 0) == 1):
                    active_services.add(service['service_id'])
        
        # Load calendar_dates.txt for exceptions
        if "calendar_dates.txt" in zip_ref.namelist():
            with zip_ref.open("calendar_dates.txt") as calendar_dates_file:
                calendar_dates_df = pd.read_csv(calendar_dates_file)
            
            # Filter for the specific date
            date_exceptions = calendar_dates_df[
                calendar_dates_df['date'].astype(str) == service_date
            ]
            
            for _, exception in date_exceptions.iterrows():
                if exception['exception_type'] == 1:  # Service added
                    active_services.add(exception['service_id'])
                elif exception['exception_type'] == 2:  # Service removed
                    active_services.discard(exception['service_id'])
        
        # If no active services found, return all stop times with warning
        if not active_services:
            print(f"Warning: No active services found for date {service_date}, returning all stop times")
            return stop_times_df[["stop_id", "arrival_time"]].copy()
        
        # Filter trips by active services
        active_trips = trips_df[trips_df['service_id'].isin(active_services)]['trip_id']
        
        # Filter stop times by active trips
        filtered_stop_times = stop_times_df[stop_times_df['trip_id'].isin(active_trips)]
        
        return filtered_stop_times[["stop_id", "arrival_time"]].copy()
        
    except Exception as e:
        print(f"Warning: Error filtering by service date: {e}")
        print("Returning all stop times without date filtering")
        return stop_times_df[["stop_id", "arrival_time"]].copy()


# Also add this simplified version if you want to skip date filtering entirely
def load_cpt_schedule_from_gtfs_simple(gtfs_path: str) -> pd.DataFrame:
    """
    Simplified version that loads all stop times without date filtering.
    Use this if you're having issues with service date filtering.
    """
    try:
        with zipfile.ZipFile(gtfs_path, 'r') as zip_ref:
            if "stop_times.txt" not in zip_ref.namelist():
                raise ValueError("GTFS zip file does not contain stop_times.txt")
            
            with zip_ref.open("stop_times.txt") as stop_times_file:
                stop_times_df = pd.read_csv(stop_times_file)
            
            # Check required columns
            required_columns = ["stop_id", "arrival_time"]
            missing_columns = [col for col in required_columns if col not in stop_times_df.columns]
            if missing_columns:
                raise ValueError(f"GTFS stop_times.txt missing required columns: {missing_columns}")
            
            # Use all stop times without date filtering
            schedule_df = stop_times_df[["stop_id", "arrival_time"]].copy()
            
            # Remove duplicates and sort
            schedule_df = schedule_df.drop_duplicates().sort_values(["stop_id", "arrival_time"])
            
        print(f"Loaded {len(schedule_df)} schedule entries from GTFS (no date filtering)")
        return schedule_df
        
    except Exception as e:
        print(f"Error loading CPT schedule from GTFS: {e}")
        raise

def engineer_time_features(df: pd.DataFrame, 
                          time_columns: List[str], 
                          is_generated_data: bool = False,
                          date_str: Optional[str] = None) -> pd.DataFrame:
    """
    Create comprehensive time-based features from datetime columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        time_columns (list): List of datetime column names to process
        is_generated_data (bool): True for synthetic/generated data, False for real data
        date_str (str, optional): Date in DD/MM/YYYY format, required for generated data
        
    Returns:
        pd.DataFrame: DataFrame with additional time-based features
    """
    print("Engineering time-based features...")
    df_processed = df.copy()
    
    def add_time_features(df_proc, col, datetime_col):
        """Helper function to add time features - avoids code duplication"""
        df_proc[f'{col}_hour'] = df_proc[datetime_col].dt.hour
        df_proc[f'{col}_minute'] = df_proc[datetime_col].dt.minute
        df_proc[f'{col}_day_of_week'] = df_proc[datetime_col].dt.dayofweek
        df_proc[f'{col}_day_of_month'] = df_proc[datetime_col].dt.day
        df_proc[f'{col}_month'] = df_proc[datetime_col].dt.month
        
        # Create cyclical features for periodic time variables
        df_proc[f'{col}_hour_sin'] = np.sin(2 * np.pi * df_proc[f'{col}_hour'] / 24)
        df_proc[f'{col}_hour_cos'] = np.cos(2 * np.pi * df_proc[f'{col}_hour'] / 24)
        df_proc[f'{col}_day_of_week_sin'] = np.sin(2 * np.pi * df_proc[f'{col}_day_of_week'] / 7)
        df_proc[f'{col}_day_of_week_cos'] = np.cos(2 * np.pi * df_proc[f'{col}_day_of_week'] / 7)
        df_proc[f'{col}_month_sin'] = np.sin(2 * np.pi * df_proc[f'{col}_month'] / 12)
        df_proc[f'{col}_month_cos'] = np.cos(2 * np.pi * df_proc[f'{col}_month'] / 12)
    
    if is_generated_data:
        # Handle generated/synthetic data
        if not date_str:
            raise ValueError("date_str is required when is_generated_data=True")
        
        # Parse the date
        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            print(f"Error: Invalid date format '{date_str}'. Expected DD/MM/YYYY format.")
            return df_processed
        
        for col in time_columns:
            if col not in df_processed.columns:
                print(f"Warning: Column '{col}' not found in dataframe. Skipping...")
                continue
            
            # Create full datetime by combining date and time
            def create_full_datetime(time_str):
                try:
                    if pd.isna(time_str):
                        return None
                    time_obj = datetime.strptime(str(time_str), "%H:%M:%S").time()
                    return datetime.combine(date_obj.date(), time_obj)
                except:
                    return None
            
            df_processed[f'{col}_datetime'] = df_processed[col].apply(create_full_datetime)
            df_processed = df_processed.dropna(subset=[f'{col}_datetime'])
            
            # Use helper function to add features
            add_time_features(df_processed, col, f'{col}_datetime')
        
        df_processed['analysis_date'] = date_str
        
    else:
        # Handle real data
        for col in time_columns:
            # Ensure column is datetime
            df_processed[col] = pd.to_datetime(df_processed[col])
            
            # Use helper function to add features
            add_time_features(df_processed, col, col)
    
    print(f"Feature engineering complete: {len(df_processed.columns)} total columns")
    return df_processed


def run_complete_analysis_with_time_features(bbox: List[float],
                                           drt_trips_df: pd.DataFrame,
                                           analysis_date: str,  
                                           gtfs_path: Optional[str] = None,
                                           cpt_stops_df: Optional[pd.DataFrame] = None,
                                           cpt_schedule_df: Optional[pd.DataFrame] = None,
                                           hexagon_edge_km: float = 1.0,
                                           max_walking_time_min: int = 10,
                                           time_windows: List[Tuple[str, str]] = [("07:00:00", "10:00:00"), ("10:00:00", "12:00:00")],
                                           base_departure_hours: List[int] = [7, 10],
                                           osrm_url: str = "http://localhost:5000",
                                           output_dir: str = "results",
                                           batch_size: int = 100,
                                           add_time_features: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Run complete DRT accessibility analysis with optimized walking time computation and time features.
    Now supports multiple departure hours and corresponding time windows.
    
    Parameters:
    -----------
    bbox : List[float]
        Study area bounding box [min_lon, min_lat, max_lon, max_lat]
    drt_trips_df : pd.DataFrame
        Observed DRT trips data
    analysis_date : str
        Date of analysis in DD/MM/YYYY format (e.g., "15/12/2024")
    gtfs_path : str, optional
        Path to GTFS zip file for CPT data
    cpt_stops_df : pd.DataFrame, optional
        Pre-loaded CPT stops data
    cpt_schedule_df : pd.DataFrame, optional
        Pre-loaded CPT schedule data
    hexagon_edge_km : float
        Size of hexagon edge in kilometers
    max_walking_time_min : int
        Maximum walking time to CPT stops in minutes
    time_windows : List[Tuple[str, str]]
        List of time windows for analysis, each corresponding to a departure hour
    base_departure_hours : List[int]
        List of base departure hours (e.g., [7, 10] for 7:00 AM and 10:00 AM)
    osrm_url : str
        OSRM service URL
    output_dir : str
        Output directory for results
    batch_size : int
        Batch size for OSRM table calculations
    add_time_features : bool
        Whether to add comprehensive time features to synthetic trips
        
    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary containing all analysis results with time features
    """
    
    print("=" * 60)
    print("DRT ACCESSIBILITY ANALYSIS WITH MULTIPLE TIME WINDOWS")
    print("=" * 60)
    print(f"Analysis Date: {analysis_date}")
    print(f"Departure Hours: {base_departure_hours}")
    print(f"Time Windows: {time_windows}")
    
    # Validate and normalize inputs
   
    print(f"  time_windows type: {type(time_windows)}, content: {time_windows}")
    print(f"  base_departure_hours type: {type(base_departure_hours)}, content: {base_departure_hours}")
    
    # Handle case where parameters might be passed as single nested lists
    if isinstance(base_departure_hours, list) and len(base_departure_hours) == 1 and isinstance(base_departure_hours[0], list):
        base_departure_hours = base_departure_hours[0]
        print(f"  Flattened base_departure_hours: {base_departure_hours}")
    
    if isinstance(time_windows, list) and len(time_windows) == 1 and isinstance(time_windows[0], list):
        time_windows = time_windows[0]
        print(f"  Flattened time_windows: {time_windows}")
    
    # Validate inputs
    if len(time_windows) != len(base_departure_hours):
        raise ValueError(f"Number of time_windows ({len(time_windows)}) must match number of base_departure_hours ({len(base_departure_hours)})")
    
    # Validate date format
    try:
        date_obj = datetime.strptime(analysis_date, "%d/%m/%Y")
        print(f"Parsed date: {date_obj.strftime('%Y-%m-%d (%A)')}")
    except ValueError:
        raise ValueError(f"Invalid date format '{analysis_date}'. Expected DD/MM/YYYY format (e.g., '15/12/2024')")
    
    # Initialize analyzer
    analyzer = DRTVrirualLinesGenerator(
        hexagon_edge_km=hexagon_edge_km,
        osrm_url=osrm_url
    )
    
    # Step 1: Generate hexagonal grid
    print("\n1. GENERATING HEXAGONAL GRID")
    print("-" * 30)
    hexagons_gdf = analyzer.generate_hexagonal_grid(bbox, "study_area")
    
    # Step 2: Calculate barycenters
    print("\n2. CALCULATING BARYCENTERS")
    print("-" * 30)
    hexagons_gdf = analyzer.calculate_barycenters(drt_trips_df)
    
    # Step 3: Load CPT data if needed
    if gtfs_path and not cpt_stops_df:
        print("\n3. LOADING CPT DATA FROM GTFS")
        print("-" * 30)
        cpt_stops_df = load_cpt_stops_from_gtfs(gtfs_path)
        cpt_schedule_df = load_cpt_schedule_from_gtfs(gtfs_path)
    
    # Step 4: Compute all walking times using optimized batch operation
    print("\n4. COMPUTING ALL WALKING TIMES (OPTIMIZED)")
    print("-" * 30)
    walking_times_df = analyzer.compute_all_walking_times_optimized(
        cpt_stops_df=cpt_stops_df.head(200) if cpt_stops_df is not None else None,
        batch_size=batch_size
    )
    
    # Step 5: Generate synthetic DRT trips for each departure hour and time window
    print("\n5. GENERATING SYNTHETIC DRT TRIPS FOR MULTIPLE TIME WINDOWS")
    print("-" * 30)
    
    # Check if we have any walking times computed
    if len(walking_times_df) == 0:
        print("WARNING: No walking times computed. Cannot generate DRT trips.")
        print("This is likely due to OSRM connection issues.")
        print("Please ensure OSRM server is running on", osrm_url)
        
        # Create empty results
        results = {
            "hexagons": analyzer.hexagons_df,
            "walking_times": walking_times_df,
            "synthetic_drt_trips": pd.DataFrame(),
            "analysis_date": analysis_date,
            "departure_hours": base_departure_hours,
            "time_windows": time_windows
        }
        
        if cpt_stops_df is not None:
            results["cpt_stops"] = cpt_stops_df
        if cpt_schedule_df is not None:
            results["cpt_schedule"] = cpt_schedule_df
            
        return results
    
    # Generate trips for each departure hour and time window combination
    all_synthetic_trips = []
    
    for i, (departure_hour, time_window) in enumerate(zip(base_departure_hours, time_windows)):
        print(f"\n  Processing departure hour {departure_hour} with time window {time_window}")
        print(f"    Time window type: {type(time_window)}, Content: {time_window}")
        
        # Ensure time_window is a tuple
        if isinstance(time_window, (list, tuple)) and len(time_window) == 2:
            window_tuple = tuple(time_window)
        else:
            raise ValueError(f"Invalid time_window format: {time_window}. Expected tuple with 2 elements.")
        
        # Generate trips for this specific departure hour and time window
        drt_trips_for_window = analyzer.generate_drt_trips(
            cpt_stops_df=cpt_stops_df,
            cpt_schedule_df=cpt_schedule_df,
            time_window=window_tuple,
            max_walking_time_min=max_walking_time_min,
            base_departure_hour=departure_hour
        )
        
        # Add metadata to identify which window/departure hour these trips belong to
        if len(drt_trips_for_window) > 0:
            drt_trips_for_window['departure_hour_group'] = departure_hour
            drt_trips_for_window['time_window'] = f"{time_window[0]}-{time_window[1]}"
            drt_trips_for_window['window_index'] = i
            all_synthetic_trips.append(drt_trips_for_window)
            
            print(f"    Generated {len(drt_trips_for_window)} trips for departure hour {departure_hour}")
        else:
            print(f"    No trips generated for departure hour {departure_hour}")
    
    # Combine all trips from different time windows
    if all_synthetic_trips:
        combined_drt_trips = pd.concat(all_synthetic_trips, ignore_index=True)
        analyzer.drt_trips_df = combined_drt_trips
        print(f"\n  Total synthetic trips generated: {len(combined_drt_trips)}")
        
        # Show breakdown by departure hour
        for hour in base_departure_hours:
            count = len(combined_drt_trips[combined_drt_trips['departure_hour_group'] == hour])
            print(f"    Departure hour {hour}: {count} trips")
    else:
        print("  No synthetic trips generated for any time window")
        analyzer.drt_trips_df = pd.DataFrame()
    
    # Step 6: Add comprehensive time features to synthetic trips
    if add_time_features and len(analyzer.drt_trips_df) > 0:
        print("\n6. ENGINEERING TIME FEATURES")
        print("-" * 30)
        trips_with_features = engineer_time_features(
            df=analyzer.drt_trips_df, 
            time_columns=['departure_time'], 
            is_generated_data=True,
            date_str=analysis_date
        )
        analyzer.drt_trips_df = trips_with_features
        
        # Show feature engineering results by departure hour
        for hour in base_departure_hours:
            hour_trips = trips_with_features[trips_with_features['departure_hour_group'] == hour]
            if len(hour_trips) > 0:
                print(f"    Added time features for {len(hour_trips)} trips starting at {hour}:00")
    
    # Step 7: Save results
    print("\n7. SAVING RESULTS")
    print("-" * 30)
    
    # Save hexagons and walking times using the existing save_results method
    os.makedirs(output_dir, exist_ok=True)
    
    if analyzer.hexagons_df is not None:
        # Convert geometry to WKT for saving
        hexagons_save = analyzer.hexagons_df.copy()
        if 'geometry' in hexagons_save.columns:
            hexagons_save['geometry'] = hexagons_save['geometry'].apply(lambda x: x.wkt if x is not None else None)
        hexagons_save.to_csv(os.path.join(output_dir, "hexagons_with_barycenters.csv"), index=False)
    
    if analyzer.walking_times_df is not None:
        analyzer.walking_times_df.to_csv(os.path.join(output_dir, "walking_times.csv"), index=False)
    
    # Save all synthetic trips as a single concatenated file
    if len(analyzer.drt_trips_df) > 0:
        all_trips_filename = os.path.join(output_dir, "all_synthetic_trips.csv")
        analyzer.drt_trips_df.to_csv(all_trips_filename, index=False)
        print(f"    Saved all {len(analyzer.drt_trips_df)} synthetic trips to: {all_trips_filename}")
        
        # Show breakdown by departure hour in the log
        for hour in base_departure_hours:
            hour_count = len(analyzer.drt_trips_df[analyzer.drt_trips_df['departure_hour_group'] == hour])
            print(f"      - Hour {hour}: {hour_count} trips")
    else:
        print("    No synthetic trips to save")
    
    # Prepare return dictionary
    results = {
        "hexagons": analyzer.hexagons_df,
        "walking_times": analyzer.walking_times_df,
        "synthetic_drt_trips": analyzer.drt_trips_df,
        "analysis_date": analysis_date,
        "departure_hours": base_departure_hours,
        "time_windows": time_windows
    }
    
    if cpt_stops_df is not None:
        results["cpt_stops"] = cpt_stops_df
    if cpt_schedule_df is not None:
        results["cpt_schedule"] = cpt_schedule_df
    
    # Add breakdown by departure hour to results
    if len(analyzer.drt_trips_df) > 0:
        results["trips_by_hour"] = {}
        for hour in base_departure_hours:
            hour_trips = analyzer.drt_trips_df[analyzer.drt_trips_df['departure_hour_group'] == hour]
            results["trips_by_hour"][hour] = hour_trips
    
    print("\n" + "=" * 60)
    print("ENHANCED MULTI-WINDOW ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Analysis Date: {analysis_date}")
    print(f"Generated {len(results['hexagons'])} hexagons")
    print(f"Computed {len(results['walking_times'])} walking time connections")
    print(f"Generated {len(results['synthetic_drt_trips'])} total synthetic DRT trips")
    
    for i, (hour, window) in enumerate(zip(base_departure_hours, time_windows)):
        if len(analyzer.drt_trips_df) > 0:
            hour_count = len(analyzer.drt_trips_df[analyzer.drt_trips_df['departure_hour_group'] == hour])
            print(f"  - Hour {hour} ({window[0]}-{window[1]}): {hour_count} trips")
        else:
            print(f"  - Hour {hour} ({window[0]}-{window[1]}): 0 trips")
    
    if add_time_features and len(results['synthetic_drt_trips']) > 0:
        time_feature_cols = [col for col in results['synthetic_drt_trips'].columns if 'departure_time' in col]
        print(f"Added {len(time_feature_cols)} time features")
    
    print(f"Batch size used: {batch_size}")
    
    return results