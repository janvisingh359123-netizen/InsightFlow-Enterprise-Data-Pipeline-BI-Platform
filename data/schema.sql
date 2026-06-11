-- PostgreSQL DDL Schema for SmartHouse AI Platform

-- 1. Locations Table
CREATE TABLE IF NOT EXISTS locations (
    location_id SERIAL PRIMARY KEY,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    zip_code VARCHAR(20) NOT NULL,
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL
);

-- 2. Properties Table
CREATE TABLE IF NOT EXISTS properties (
    property_id VARCHAR(50) PRIMARY KEY,
    property_type VARCHAR(50) NOT NULL, -- 'Single Family', 'Condo', 'Townhouse', 'Apartment'
    area_sqft NUMERIC(10, 2) NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms INTEGER NOT NULL,
    floors INTEGER NOT NULL,
    garage_spaces INTEGER NOT NULL,
    year_built INTEGER NOT NULL,
    location_id INTEGER NOT NULL REFERENCES locations(location_id) ON DELETE CASCADE,
    latitude NUMERIC(9, 6) NOT NULL,
    longitude NUMERIC(9, 6) NOT NULL,
    crime_rate NUMERIC(3, 1) NOT NULL,
    school_rating NUMERIC(3, 1) NOT NULL,
    distance_to_center NUMERIC(5, 1) NOT NULL,
    distance_to_metro NUMERIC(5, 1) NOT NULL,
    distance_to_airport NUMERIC(5, 1) NOT NULL,
    nearby_hospitals INTEGER NOT NULL,
    nearby_shopping INTEGER NOT NULL
);

-- 3. Amenities Table
CREATE TABLE IF NOT EXISTS amenities (
    amenity_id SERIAL PRIMARY KEY,
    property_id VARCHAR(50) NOT NULL REFERENCES properties(property_id) ON DELETE CASCADE,
    has_pool BOOLEAN NOT NULL DEFAULT FALSE,
    has_garden BOOLEAN NOT NULL DEFAULT FALSE,
    has_balcony BOOLEAN NOT NULL DEFAULT FALSE,
    has_parking BOOLEAN NOT NULL DEFAULT FALSE,
    has_security BOOLEAN NOT NULL DEFAULT FALSE,
    has_gym BOOLEAN NOT NULL DEFAULT FALSE
);

-- 4. Historical Sales Table
CREATE TABLE IF NOT EXISTS historical_sales (
    sale_id SERIAL PRIMARY KEY,
    property_id VARCHAR(50) NOT NULL REFERENCES properties(property_id) ON DELETE CASCADE,
    sale_price NUMERIC(15, 2) NOT NULL,
    sale_date DATE NOT NULL
);

-- 5. Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    property_id VARCHAR(50) REFERENCES properties(property_id) ON DELETE SET NULL,
    input_params TEXT NOT NULL, -- Store JSON representation of prediction parameters
    predicted_price NUMERIC(15, 2) NOT NULL,
    confidence_score NUMERIC(5, 2) NOT NULL,
    prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes for High Performance Queries
CREATE INDEX IF NOT EXISTS idx_properties_location ON properties(location_id);
CREATE INDEX IF NOT EXISTS idx_properties_type ON properties(property_type);
CREATE INDEX IF NOT EXISTS idx_properties_sqft ON properties(area_sqft);
CREATE INDEX IF NOT EXISTS idx_properties_year ON properties(year_built);
CREATE INDEX IF NOT EXISTS idx_amenities_property ON amenities(property_id);
CREATE INDEX IF NOT EXISTS idx_sales_property ON historical_sales(property_id);
CREATE INDEX IF NOT EXISTS idx_sales_date ON historical_sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date);
