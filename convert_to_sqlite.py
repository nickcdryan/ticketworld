#!/usr/bin/env python3
"""
Script to convert customer_database.json to SQLite database format.
Creates a simplified 3-table schema: customers, products, and orders.

=== DATABASE SCHEMA ===

Table: customers
- customer_id (TEXT, PRIMARY KEY): Unique customer identifier (format: CUST-XXXX)
- name (TEXT, NOT NULL): Customer full name
- primary_email (TEXT, NOT NULL): Primary email address
- alternate_email (TEXT): Secondary email address (nullable)
- phone (TEXT): Phone number
- shipping_street (TEXT): Shipping address street
- shipping_city (TEXT): Shipping address city
- shipping_state (TEXT): Shipping address state
- shipping_zip (TEXT): Shipping address ZIP code
- billing_street (TEXT): Billing address street
- billing_city (TEXT): Billing address city
- billing_state (TEXT): Billing address state
- billing_zip (TEXT): Billing address ZIP code
- created_date (DATE): Customer account creation date

Table: products
- product_id (TEXT, PRIMARY KEY): Unique product identifier (format: PROD-XXXX)
- name (TEXT, NOT NULL): Product name
- category (TEXT): Product category
- brand (TEXT): Product brand
- base_price (DECIMAL): Product base price
- warranty_period (INTEGER): Warranty period in days
- weight (DECIMAL): Product weight
- requires_signature (BOOLEAN): Whether delivery requires signature
- in_stock (BOOLEAN): Whether product is currently in stock
- description (TEXT): Product description

Table: orders
- order_id (TEXT, PRIMARY KEY): Unique order identifier (format: ORD-YYYYMMDD-XXXX)
- customer_id (TEXT, NOT NULL): Foreign key to customers.customer_id
- order_date (DATE): Order placement date
- items (TEXT): JSON array of order items (see JSON structure below)
- shipping_method (TEXT): Shipping method used
- tracking_number (TEXT): Package tracking number
- total_amount (DECIMAL): Total order amount
- payment_method (TEXT): Payment method used
- order_status (TEXT): Current order status

=== JSON STRUCTURE FOR ORDER ITEMS ===

The orders.items column contains a JSON array with the following structure:
[
  {
    "product_id": "PROD-XXXX",
    "quantity": 1,
    "price_paid": 99.99,
    "item_status": "delivered"
  },
  ...
]

=== COMMON QUERY PATTERNS ===

1. Customer lookup by email:
   SELECT * FROM customers 
   WHERE primary_email = 'john@example.com' 
      OR alternate_email = 'john@example.com';

2. Customers in a specific state:
   SELECT * FROM customers WHERE shipping_state = 'CA';

3. Orders with customer details:
   SELECT o.*, c.name, c.primary_email 
   FROM orders o 
   JOIN customers c ON o.customer_id = c.customer_id;

4. Product sales summary:
   SELECT 
     JSON_EXTRACT(value, '$.product_id') as product_id,
     p.name,
     COUNT(*) as times_ordered,
     SUM(CAST(JSON_EXTRACT(value, '$.quantity') AS INTEGER)) as total_quantity,
     SUM(CAST(JSON_EXTRACT(value, '$.price_paid') AS REAL) * CAST(JSON_EXTRACT(value, '$.quantity') AS INTEGER)) as total_revenue
   FROM orders o, JSON_EACH(o.items) 
   JOIN products p ON JSON_EXTRACT(value, '$.product_id') = p.product_id
   GROUP BY JSON_EXTRACT(value, '$.product_id'), p.name
   ORDER BY total_revenue DESC;

=== NOTES ===
- All JSON queries use SQLite's built-in JSON functions
- For better performance on JSON queries, consider creating indexes on computed columns
- The schema is compatible with both old (array) and new (single) JSON data formats
- Foreign key constraints are enabled between orders.customer_id and customers.customer_id

"""

import json
import sqlite3
import os
from datetime import datetime

def create_database_schema(cursor):
    """Create the simplified database schema with 3 tables."""
    
    # Customers table with single address fields
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            primary_email TEXT NOT NULL,
            alternate_email TEXT,
            phone TEXT,
            shipping_street TEXT,
            shipping_city TEXT,
            shipping_state TEXT,
            shipping_zip TEXT,
            billing_street TEXT,
            billing_city TEXT,
            billing_state TEXT,
            billing_zip TEXT,
            created_date DATE
        )
    ''')
    
    # Products table (unchanged)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            product_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            brand TEXT,
            base_price DECIMAL(10,2),
            warranty_period INTEGER,
            weight DECIMAL(5,2),
            requires_signature BOOLEAN,
            in_stock BOOLEAN,
            description TEXT
        )
    ''')
    
    # Orders table with items as JSON field
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            order_date DATE,
            items TEXT,  -- JSON array of order items
            shipping_method TEXT,
            tracking_number TEXT,
            total_amount DECIMAL(10,2),
            payment_method TEXT,
            order_status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
    ''')

def insert_customers(cursor, customers_data):
    """Insert customer data into the simplified customers table."""
    for customer in customers_data:
        # Handle different data formats (old array vs new single)
        alternate_email = None
        if 'alternate_email' in customer and customer['alternate_email']:
            alternate_email = customer['alternate_email']
        elif 'alternate_emails' in customer and customer['alternate_emails']:
            # Handle old array format - take first email
            alternate_email = customer['alternate_emails'][0] if customer['alternate_emails'] else None
        
        # Handle shipping address
        shipping_street = shipping_city = shipping_state = shipping_zip = None
        if 'shipping_address' in customer and customer['shipping_address']:
            addr = customer['shipping_address']
            shipping_street = addr.get('street')
            shipping_city = addr.get('city') 
            shipping_state = addr.get('state')
            shipping_zip = addr.get('zip')
        elif 'shipping_addresses' in customer and customer['shipping_addresses']:
            # Handle old array format - take first address
            addr = customer['shipping_addresses'][0]
            shipping_street = addr.get('street')
            shipping_city = addr.get('city')
            shipping_state = addr.get('state')
            shipping_zip = addr.get('zip')
        
        # Handle billing address
        billing_street = billing_city = billing_state = billing_zip = None
        if 'billing_address' in customer and customer['billing_address']:
            addr = customer['billing_address']
            billing_street = addr.get('street')
            billing_city = addr.get('city')
            billing_state = addr.get('state') 
            billing_zip = addr.get('zip')
        elif 'billing_addresses' in customer and customer['billing_addresses']:
            # Handle old array format - take first address
            addr = customer['billing_addresses'][0]
            billing_street = addr.get('street')
            billing_city = addr.get('city')
            billing_state = addr.get('state')
            billing_zip = addr.get('zip')
        
        cursor.execute('''
            INSERT OR REPLACE INTO customers 
            (customer_id, name, primary_email, alternate_email, phone,
             shipping_street, shipping_city, shipping_state, shipping_zip,
             billing_street, billing_city, billing_state, billing_zip, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            customer['customer_id'],
            customer['name'],
            customer['primary_email'],
            alternate_email,
            customer['phone'],
            shipping_street, shipping_city, shipping_state, shipping_zip,
            billing_street, billing_city, billing_state, billing_zip,
            customer['created_date']
        ))

def insert_products(cursor, products_data):
    """Insert product data into the database."""
    for product in products_data:
        cursor.execute('''
            INSERT OR REPLACE INTO products 
            (product_id, name, category, brand, base_price, warranty_period, 
             weight, requires_signature, in_stock, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            product['product_id'],
            product['name'],
            product['category'],
            product['brand'],
            product['base_price'],
            product['warranty_period'],
            product['weight'],
            product['requires_signature'],
            product['in_stock'],
            product['description']
        ))

def insert_orders(cursor, orders_data):
    """Insert order data into the simplified orders table."""
    for order in orders_data:
        # Convert items array to JSON string
        items_json = json.dumps(order.get('items', []))
        
        cursor.execute('''
            INSERT OR REPLACE INTO orders 
            (order_id, customer_id, order_date, items, shipping_method, tracking_number,
             total_amount, payment_method, order_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            order['order_id'],
            order['customer_id'],
            order['order_date'],
            items_json,
            order['shipping_method'],
            order['tracking_number'],
            order['total_amount'],
            order['payment_method'],
            order['order_status']
        ))

def main():
    """Main function to convert JSON to SQLite database."""
    # Input and output file paths
    json_file = 'assets/customer_database.json'
    db_file = 'assets/customer_database.db'
    
    # Check if input file exists
    if not os.path.exists(json_file):
        print(f"Error: Input file '{json_file}' not found.")
        return
    
    # Remove existing database file if it exists
    if os.path.exists(db_file):
        os.remove(db_file)
        print(f"Removed existing database file: {db_file}")
    
    try:
        # Load JSON data
        print(f"Loading data from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Connect to SQLite database
        print(f"Creating SQLite database: {db_file}")
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Create database schema
        print("Creating simplified 3-table database schema...")
        create_database_schema(cursor)
        
        # Insert data
        print("Inserting customer data...")
        insert_customers(cursor, data['customers'])
        
        print("Inserting product data...")
        insert_products(cursor, data['products'])
        
        print("Inserting order data...")
        insert_orders(cursor, data['orders'])
        
        # Commit changes and close connection
        conn.commit()
        
        # Print summary statistics
        cursor.execute("SELECT COUNT(*) FROM customers")
        customer_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM products")
        product_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM orders")
        order_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"\nConversion completed successfully!")
        print(f"Database created: {db_file}")
        print(f"Simplified 3-table schema:")
        print(f"  - customers: {customer_count} records")
        print(f"  - products: {product_count} records") 
        print(f"  - orders: {order_count} records")
        print(f"\nNotes:")
        print(f"  - Customer addresses stored as individual columns")
        print(f"  - Order items stored as JSON in 'items' column")
        print(f"  - Compatible with both old (array) and new (single) data formats")
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_file}: {e}")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main() 