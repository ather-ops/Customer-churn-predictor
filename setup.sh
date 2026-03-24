#!/bin/bash

# Create necessary directories
mkdir -p models
mkdir -p data

# Install dependencies
pip install -r requirements.txt

# Download sample data if not exists
if [ ! -f "data/sample_customer_churn.csv" ]; then
    echo "Creating sample data file..."
    # Create sample data if needed
fi

echo "Setup complete!"
