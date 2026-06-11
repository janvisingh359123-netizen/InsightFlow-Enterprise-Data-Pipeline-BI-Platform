# Synthetic Real Estate Data Generator
# Generates 100,000+ properties with locations, amenities, and historical sales.
# Ensures realistic statistical distributions and strong correlations for ML training.

import os
import sys
import json
import random
import sqlite3
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path to import database adapter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.database import get_db_connection, init_db, DATABASE_TYPE

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Predefined cities with coordinates, pricing multipliers, and base neighborhood profiles
CITIES_DB = [
    {
        "city": "Seattle", "state": "WA", 
        "zips": ["98101", "98102", "98103", "98105", "98109", "98112"],
        "lat": 47.6062, "lng": -122.3321, "price_mult": 2.3,
        "school_base": 7.8, "crime_base": 2.5, "metro_dist_base": 2.5
    },
    {
        "city": "Austin", "state": "TX", 
        "zips": ["78701", "78702", "78704", "78705", "78741", "78746"],
        "lat": 30.2672, "lng": -97.7431, "price_mult": 1.6,
        "school_base": 6.8, "crime_base": 3.1, "metro_dist_base": 5.0
    },
    {
        "city": "Chicago", "state": "IL", 
        "zips": ["60611", "60614", "60618", "60622", "60637", "60657"],
        "lat": 41.8781, "lng": -87.6298, "price_mult": 1.35,
        "school_base": 6.2, "crime_base": 4.5, "metro_dist_base": 1.2
    },
    {
        "city": "Orlando", "state": "FL", 
        "zips": ["32801", "32803", "32804", "32806", "32819", "32828"],
        "lat": 28.5383, "lng": -81.3792, "price_mult": 1.05,
        "school_base": 5.8, "crime_base": 3.8, "metro_dist_base": 8.0
    },
    {
        "city": "Phoenix", "state": "AZ", 
        "zips": ["85003", "85004", "85006", "85012", "85016", "85018"],
        "lat": 33.4484, "lng": -112.0740, "price_mult": 0.95,
        "school_base": 5.5, "crime_base": 3.5, "metro_dist_base": 6.5
    }
]

PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Apartment"]
TYPE_MULTIPLIERS = {
    "Single Family": 1.25,
    "Townhouse": 1.05,
    "Condo": 0.95,
    "Apartment": 0.80
}

def generate_location_records():
    """Generates location master table records for all zip codes"""
    records = []
    for city_info in CITIES_DB:
        for idx, zip_code in enumerate(city_info["zips"]):
            # Small jitter to latitude/longitude for zip codes
            lat_jitter = np.random.uniform(-0.04, 0.04)
            lng_jitter = np.random.uniform(-0.04, 0.04)
            records.append((
                city_info["city"],
                city_info["state"],
                zip_code,
                round(city_info["lat"] + lat_jitter, 5),
                round(city_info["lng"] + lng_jitter, 5)
            ))
    return records

def generate_housing_data(num_properties=100000):
    """Generates properties, amenities, and historical sales records"""
    print(f"Generating synthetic housing data: {num_properties} properties...")
    
    locations = generate_location_records()
    
    properties = []
    amenities = []
    historical_sales = []
    
    current_year = 2026
    
    for i in range(num_properties):
        prop_id = f"PROP{i:06d}"
        
        # Select location index (corresponds to auto-incrementing ID starting at 1)
        loc_idx = random.randint(0, len(locations) - 1)
        loc_id = loc_idx + 1
        city_data = next(c for c in CITIES_DB if c["city"] == locations[loc_idx][0])
        
        # Structural details based on property type
        prop_type = np.random.choice(PROPERTY_TYPES, p=[0.45, 0.25, 0.15, 0.15])
        
        if prop_type == "Single Family":
            area = int(np.random.normal(2500, 700))
            area = max(1000, min(6000, area))
            bedrooms = max(2, min(6, int(np.ceil(area / 800) + np.random.choice([-1, 0, 1]))))
            bathrooms = max(1, min(5, int(bedrooms - np.random.choice([0, 1]) + 0.5)))
            floors = np.random.choice([1, 2], p=[0.6, 0.4])
            garage = np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2])
            year_built = int(np.random.normal(1985, 25))
        elif prop_type == "Townhouse":
            area = int(np.random.normal(1800, 400))
            area = max(900, min(3000, area))
            bedrooms = max(2, min(4, int(np.ceil(area / 700))))
            bathrooms = max(1, min(3, int(bedrooms - 0.5)))
            floors = np.random.choice([2, 3], p=[0.7, 0.3])
            garage = np.random.choice([1, 2], p=[0.6, 0.4])
            year_built = int(np.random.normal(2005, 12))
        elif prop_type == "Condo":
            area = int(np.random.normal(1200, 300))
            area = max(600, min(2500, area))
            bedrooms = max(1, min(3, int(np.ceil(area / 800))))
            bathrooms = max(1, min(2, bedrooms))
            floors = 1
            garage = np.random.choice([0, 1, 2], p=[0.3, 0.6, 0.1])
            year_built = int(np.random.normal(2000, 15))
        else: # Apartment (rental or listing)
            area = int(np.random.normal(900, 250))
            area = max(450, min(1800, area))
            bedrooms = max(1, min(3, int(np.ceil(area / 800))))
            bathrooms = max(1, min(2, bedrooms))
            floors = 1
            garage = np.random.choice([0, 1], p=[0.7, 0.3])
            year_built = int(np.random.normal(1995, 20))
            
        year_built = max(1950, min(current_year, year_built))
        
        # Amenities
        has_pool = random.random() < (0.35 if prop_type == "Single Family" else 0.15)
        has_garden = random.random() < (0.60 if prop_type == "Single Family" else 0.10)
        has_balcony = random.random() < (0.80 if prop_type in ["Condo", "Apartment"] else 0.30)
        has_parking = random.random() < (0.90 if prop_type != "Apartment" else 0.50)
        has_security = random.random() < (0.65 if prop_type in ["Condo", "Apartment"] else 0.25)
        has_gym = random.random() < (0.50 if prop_type in ["Condo", "Apartment"] else 0.05)
        
        # Neighborhood features (jittered from bases)
        school_rating = max(1.0, min(10.0, round(city_data["school_base"] + np.random.normal(0, 1.2), 1)))
        crime_rate = max(0.5, min(10.0, round(city_data["crime_base"] + np.random.normal(0, 1.0), 1)))
        
        dist_center = max(0.2, round(np.random.exponential(8.0) + 0.5, 1))
        dist_metro = max(0.1, round(np.random.exponential(city_data["metro_dist_base"]) + 0.1, 1))
        dist_airport = max(3.0, round(np.random.normal(25.0, 8.0), 1))
        
        nearby_hospitals = max(0, min(5, int(np.random.poisson(1.5))))
        nearby_shopping = max(0, min(10, int(np.random.poisson(3.0))))
        
        # Math-based price calculator representing actual valuation model
        # Base value per sqft
        price_per_sqft_base = 210.0
        
        # Location modifier
        loc_multiplier = city_data["price_mult"]
        
        # Adjust per zip code
        zip_offset = (int(locations[loc_idx][2]) % 5) * 0.03 - 0.06
        loc_multiplier += zip_offset
        
        # Calculate raw base price
        val_price = area * price_per_sqft_base
        val_price *= TYPE_MULTIPLIERS[prop_type]
        val_price *= loc_multiplier
        
        # Add value for bed, bath, floors, garage
        val_price += bedrooms * 24000.0
        val_price += bathrooms * 18000.0
        val_price += floors * 8000.0
        val_price += garage * 12000.0
        
        # Neighborhood effects
        school_effect = 1.0 + (school_rating - 5.0) * 0.035 # +3.5% per point above 5
        crime_effect = 1.0 - (crime_rate - 3.0) * 0.025 # -2.5% per point above 3
        
        # Distances effects
        metro_effect = 1.0 + (0.05 if dist_metro < 1.5 else (0.0 if dist_metro > 8.0 else (8.0 - dist_metro) * 0.007))
        center_effect = 1.0 - min(0.25, (dist_center * 0.012)) # -1.2% per km from center
        
        val_price *= (school_effect * crime_effect * metro_effect * center_effect)
        
        # Age depreciation (floor at 65% structure value)
        age = current_year - year_built
        age_effect = max(0.65, 1.0 - (age * 0.004))
        val_price *= age_effect
        
        # Amenities contributions
        amenity_bonus = 1.0
        if has_pool: amenity_bonus += 0.06
        if has_gym: amenity_bonus += 0.04
        if has_garden: amenity_bonus += 0.03
        if has_balcony: amenity_bonus += 0.02
        if has_parking: amenity_bonus += 0.04
        if has_security: amenity_bonus += 0.03
        
        val_price *= amenity_bonus
        
        # Target current sale price
        noise = np.random.normal(0, 0.04) # 4% noise for natural distribution
        current_market_price = round(val_price * (1.0 + noise), 2)
        
        # 1. Properties schema: (property_id, property_type, area_sqft, bedrooms, bathrooms, floors, garage_spaces, year_built, location_id)
        # Note: We must also store neighborhood scores, but in normalized database tables, 
        # let's keep structural features in properties, location details in locations, 
        # and neighborhood values inline or in properties. Let's look at schema.sql.
        # Wait! The schema.sql defined properties as:
        # property_id PRIMARY KEY, property_type, area_sqft, bedrooms, bathrooms, floors, garage_spaces, year_built, location_id
        # Wait, where do crime_rate, school_rating, dist_metro, etc. go?
        # Ah! Let's check schema.sql. In schema.sql, there are no neighborhood columns in properties!
        # Wait! Let's check schema.sql. It contains properties, locations, amenities, historical_sales, predictions.
        # Let's inspect where the neighborhood metrics (school_rating, crime_rate, metro_distance, etc.) are.
        # Oh! If schema.sql doesn't have them, we must modify schema.sql to include neighborhood metrics, or store them in properties table!
        # Yes! Properties table is the natural place for these metrics, or we can add a table `neighborhoods` or add columns to `properties`.
        # Let's look at schema.sql. Yes, properties table has:
        # `property_id`, `property_type`, `area_sqft`, `bedrooms`, `bathrooms`, `floors`, `garage_spaces`, `year_built`, `location_id`.
        # Wait, let's look at the user prompt's properties features section:
        # "Neighborhood Metrics: Crime Rate, School Rating, Distance to City Center, Distance to Metro, Distance to Airport, Nearby Hospitals, Nearby Shopping Centers"
        # Since these are neighborhood-level or property-level metrics, let's append them directly to the `properties` table! That is highly query-efficient.
        # Let's modify `schema.sql` to add these columns.
        # Let's add these columns to `properties`:
        # `crime_rate`, `school_rating`, `distance_to_center`, `distance_to_metro`, `distance_to_airport`, `nearby_hospitals`, `nearby_shopping`
        # Let's check if we should do this. Yes, definitely! It aligns perfectly with the requirements.
        
        # Get location center coords and jitter to make unique per property
        loc_lat = locations[loc_idx][3]
        loc_lng = locations[loc_idx][4]
        prop_lat = round(loc_lat + np.random.normal(0, 0.015), 5)
        prop_lng = round(loc_lng + np.random.normal(0, 0.015), 5)

        # Let's collect property details
        properties.append((
            prop_id, prop_type, float(area), int(bedrooms), int(bathrooms), int(floors), int(garage), int(year_built), int(loc_id),
            float(prop_lat), float(prop_lng),
            float(crime_rate), float(school_rating), float(dist_center), float(dist_metro), float(dist_airport), int(nearby_hospitals), int(nearby_shopping)
        ))
        
        # Amenities
        amenities.append((
            prop_id, int(has_pool), int(has_garden), int(has_balcony), int(has_parking), int(has_security), int(has_gym)
        ))
        
        # Historical sales: Let's generate 1 to 3 sales transactions per property over the past 4 years (2022 to 2026)
        num_sales = np.random.choice([1, 2, 3], p=[0.5, 0.4, 0.1])
        
        # Back-calculate prices using 5% average annual appreciation
        appreciation_rate = 0.05
        
        # Base sale dates:
        # Final sale (most recent, representing current market listing / sale price)
        # Sales dates between June 2022 and June 2026
        dates = []
        base_date = datetime(2026, 6, 1)
        
        # Generate random sale dates
        for s in range(num_sales):
            days_ago = s * random.randint(300, 500)
            sale_dt = base_date - timedelta(days=days_ago)
            dates.append(sale_dt)
            
        dates.sort() # Oldest to newest
        
        for idx, sale_date in enumerate(dates):
            # Price adjustment based on historical time
            years_diff = (base_date - sale_date).days / 365.25
            historical_price = current_market_price / ((1.0 + appreciation_rate) ** years_diff)
            # Add transaction noise
            tx_noise = np.random.normal(0, 0.02)
            historical_price = round(historical_price * (1.0 + tx_noise), 2)
            
            historical_sales.append((
                prop_id,
                float(historical_price),
                sale_date.strftime("%Y-%m-%d")
            ))
            
    return locations, properties, amenities, historical_sales

def save_to_database(locations, properties, amenities, historical_sales):
    """Inserts records in bulk using sqlite3 or psycopg2 transactions"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        print("Emptying existing records...")
        if DATABASE_TYPE.lower() == "postgres":
            cur.execute("TRUNCATE TABLE locations, properties, amenities, historical_sales, predictions CASCADE;")
        else:
            cur.execute("DELETE FROM historical_sales;")
            cur.execute("DELETE FROM amenities;")
            cur.execute("DELETE FROM properties;")
            cur.execute("DELETE FROM locations;")
            cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('locations', 'amenities', 'historical_sales', 'predictions');")
        conn.commit()
        
        print("Inserting locations...")
        loc_query = """
            INSERT INTO locations (city, state, zip_code, latitude, longitude)
            VALUES (?, ?, ?, ?, ?)
        """ if DATABASE_TYPE.lower() != "postgres" else """
            INSERT INTO locations (city, state, zip_code, latitude, longitude)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.executemany(loc_query, locations)
        conn.commit()
        
        print("Inserting properties...")
        prop_query = """
            INSERT INTO properties (
                property_id, property_type, area_sqft, bedrooms, bathrooms, floors, garage_spaces, year_built, location_id,
                latitude, longitude,
                crime_rate, school_rating, distance_to_center, distance_to_metro, distance_to_airport, nearby_hospitals, nearby_shopping
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """ if DATABASE_TYPE.lower() != "postgres" else """
            INSERT INTO properties (
                property_id, property_type, area_sqft, bedrooms, bathrooms, floors, garage_spaces, year_built, location_id,
                latitude, longitude,
                crime_rate, school_rating, distance_to_center, distance_to_metro, distance_to_airport, nearby_hospitals, nearby_shopping
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(prop_query, properties)
        conn.commit()
        
        print("Inserting amenities...")
        amen_query = """
            INSERT INTO amenities (property_id, has_pool, has_garden, has_balcony, has_parking, has_security, has_gym)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """ if DATABASE_TYPE.lower() != "postgres" else """
            INSERT INTO amenities (property_id, has_pool, has_garden, has_balcony, has_parking, has_security, has_gym)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(amen_query, amenities)
        conn.commit()
        
        print("Inserting historical sales...")
        sale_query = """
            INSERT INTO historical_sales (property_id, sale_price, sale_date)
            VALUES (?, ?, ?)
        """ if DATABASE_TYPE.lower() != "postgres" else """
            INSERT INTO historical_sales (property_id, sale_price, sale_date)
            VALUES (%s, %s, %s)
        """
        cur.executemany(sale_query, historical_sales)
        conn.commit()
        
        print("Data loaded successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Error executing data insertions: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base_dir, "data", "schema.sql")
    
    # Initialize the database schema
    init_db(schema_path)
    
    # Generate the synthetic data
    locations = generate_location_records()
    loc_records, prop_records, amen_records, sales_records = generate_housing_data(100000)
    
    # Save to the database
    save_to_database(locations, prop_records, amen_records, sales_records)
    print("Data generation pipeline completed successfully.")
