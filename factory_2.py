"""
Synthetic Customer Support Dataset Generator

This script generates a complete synthetic dataset for training ML models on customer support tasks.
It creates tightly coupled data including customer emails, resolution plans, company policies, 
and a customer/order database.
"""

import json
import random
import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import uuid
import argparse
import os
import sys
import re

# Configuration
@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""
    # Generation counts
    num_tickets: int = 5
    num_products: int = 6
    num_customers: int = 5
    num_orders: int = 8
    
    # File paths
    output_dir: str = "./assets"
    tickets_file: str = "support_tickets.json"
    policy_file: str = "company_policy.txt"
    database_file: str = "customer_database.json"
    
    # Generation parameters
    mode: str = "create"  # "create" or "append"
    company_name: str = "TechNest"
    
    # Product parameters
    min_product_price: float = 9.99
    max_product_price: float = 2499.99
    high_value_threshold: float = 500.00  # For signature required
    
    # Customer parameters
    customer_history_days: int = 1095  # 3 years
    vip_lifetime_value_threshold: float = 2000.00
    
    # Order parameters
    order_history_days: int = 180  # 6 months
    return_rate: float = 0.10  # 10% of orders have returns
    
    # Ticket parameters
    include_debug_info: bool = True  # Include hidden scenario dimensions
    
    def get_filepath(self, filename: str) -> str:
        """Get full filepath for a given filename"""
        return os.path.join(self.output_dir, filename)

# Scenario templates for different query types with policy mappings
SCENARIO_TEMPLATES = {
    "return_request": [
        {
            "scenario_id": "RETURN-001",
            "name": "return_outside_window",
            "description": "Customer wants to return item outside 30-day window",
            "policy_clauses": ["POL-RETURN-001"],
            "customer_situation": {
                "days_since_purchase": "31-60",
                "customer_expectation": "full_refund",
                "complication": "none"
            },
            "email_patterns": {
                "should_mention": ["purchase_date", "product_name", "return_reason"],
                "might_omit": ["order_number"],
                "tone_modifier": "disappointed"
            }
        },
        {
            "scenario_id": "RETURN-002", 
            "name": "return_opened_item",
            "description": "Customer wants to return opened electronic item",
            "policy_clauses": ["POL-RETURN-001", "POL-RETURN-003"],
            "customer_situation": {
                "days_since_purchase": "1-30",
                "customer_expectation": "full_refund",
                "complication": "item_opened"
            },
            "email_patterns": {
                "should_mention": ["product_name", "opened_status", "reason"],
                "might_omit": ["restocking_fee_awareness"],
                "tone_modifier": "hopeful"
            }
        },
        {
            "scenario_id": "RETURN-003",
            "name": "holiday_return_edge_case",
            "description": "Purchase during holiday period, return after standard window",
            "policy_clauses": ["POL-RETURN-001", "POL-RETURN-002"],
            "customer_situation": {
                "days_since_purchase": "31-60",
                "customer_expectation": "holiday_extension",
                "complication": "unclear_if_eligible"
            },
            "email_patterns": {
                "should_mention": ["purchase_date", "holiday_shopping"],
                "might_omit": ["exact_policy_dates"],
                "tone_modifier": "confused"
            }
        }
    ],
    "shipping_issue": [
        {
            "scenario_id": "SHIP-001",
            "name": "package_not_received",
            "description": "Customer never received package marked as delivered",
            "policy_clauses": ["POL-SHIP-002", "POL-SHIP-003"],
            "customer_situation": {
                "days_since_delivery": "1-7",
                "customer_expectation": "replacement_or_refund",
                "complication": "shows_delivered"
            },
            "email_patterns": {
                "should_mention": ["tracking_number", "delivery_date", "checked_neighbors"],
                "might_omit": ["exact_address"],
                "tone_modifier": "worried"
            }
        },
        {
            "scenario_id": "SHIP-002",
            "name": "wrong_item_received",
            "description": "Customer received different product than ordered",
            "policy_clauses": ["POL-SHIP-004"],
            "customer_situation": {
                "days_since_delivery": "1-3",
                "customer_expectation": "free_exchange",
                "complication": "none"
            },
            "email_patterns": {
                "should_mention": ["ordered_product", "received_product"],
                "might_omit": ["order_number"],
                "tone_modifier": "frustrated"
            }
        },
        {
            "scenario_id": "SHIP-003",
            "name": "damaged_in_shipping",
            "description": "Item arrived damaged",
            "policy_clauses": ["POL-SHIP-003", "POL-RETURN-004"],
            "customer_situation": {
                "days_since_delivery": "1-2",
                "customer_expectation": "replacement",
                "complication": "time_sensitive"
            },
            "email_patterns": {
                "should_mention": ["damage_description", "product_name"],
                "might_omit": ["photos_attached"],
                "tone_modifier": "upset"
            }
        }
    ],
    "product_defect": [
        {
            "scenario_id": "DEFECT-001",
            "name": "warranty_claim",
            "description": "Product failed within warranty period",
            "policy_clauses": ["POL-WARRANTY-001", "POL-WARRANTY-002"],
            "customer_situation": {
                "months_since_purchase": "2-11",
                "customer_expectation": "free_repair_or_replacement",
                "complication": "none"
            },
            "email_patterns": {
                "should_mention": ["product_name", "issue_description", "purchase_timeframe"],
                "might_omit": ["warranty_registration"],
                "tone_modifier": "matter_of_fact"
            }
        },
        {
            "scenario_id": "DEFECT-002",
            "name": "immediate_defect",
            "description": "Product defective out of box",
            "policy_clauses": ["POL-RETURN-004", "POL-WARRANTY-001"],
            "customer_situation": {
                "days_since_purchase": "1-7",
                "customer_expectation": "immediate_replacement",
                "complication": "needed_urgently"
            },
            "email_patterns": {
                "should_mention": ["product_name", "defect_description", "brand_new"],
                "might_omit": ["troubleshooting_attempted"],
                "tone_modifier": "angry"
            }
        }
    ],
    "pricing_dispute": [
        {
            "scenario_id": "PRICE-001",
            "name": "price_match_request",
            "description": "Customer found lower price elsewhere",
            "policy_clauses": ["POL-PRICE-001", "POL-PRICE-002"],
            "customer_situation": {
                "days_since_purchase": "1-14",
                "customer_expectation": "price_difference_refunded",
                "complication": "competitor_type"
            },
            "email_patterns": {
                "should_mention": ["competitor_name", "competitor_price", "product_name"],
                "might_omit": ["exact_url"],
                "tone_modifier": "expecting_cooperation"
            }
        },
        {
            "scenario_id": "PRICE-002",
            "name": "sale_price_dispute",
            "description": "Item went on sale shortly after purchase",
            "policy_clauses": ["POL-PRICE-003"],
            "customer_situation": {
                "days_since_purchase": "2-7",
                "customer_expectation": "price_adjustment",
                "complication": "vip_customer"
            },
            "email_patterns": {
                "should_mention": ["purchase_date", "current_price", "loyalty_status"],
                "might_omit": ["order_total"],
                "tone_modifier": "disappointed_loyal_customer"
            }
        }
    ],
    "order_status": [
        {
            "scenario_id": "STATUS-001",
            "name": "delayed_shipment",
            "description": "Order not shipped within expected timeframe",
            "policy_clauses": ["POL-SHIP-001", "POL-COMM-002"],
            "customer_situation": {
                "days_since_order": "5-10",
                "customer_expectation": "immediate_shipping",
                "complication": "express_shipping_paid"
            },
            "email_patterns": {
                "should_mention": ["order_date", "shipping_method", "still_processing"],
                "might_omit": ["order_number"],
                "tone_modifier": "impatient"
            }
        }
    ],
    "warranty_claim": [
        {
            "scenario_id": "WARRANTY-001",
            "name": "extended_warranty_confusion",
            "description": "Customer unsure if issue covered by extended warranty",
            "policy_clauses": ["POL-WARRANTY-003", "POL-WARRANTY-004"],
            "customer_situation": {
                "months_since_purchase": "13-24",
                "customer_expectation": "free_repair",
                "complication": "warranty_terms_unclear"
            },
            "email_patterns": {
                "should_mention": ["product_name", "issue_description", "warranty_purchase"],
                "might_omit": ["warranty_document_number"],
                "tone_modifier": "confused"
            }
        }
    ],
    "general_inquiry": [
        {
            "scenario_id": "GENERAL-001",
            "name": "product_availability",
            "description": "Customer asking about product availability or restock",
            "policy_clauses": ["POL-COMM-001", "POL-INVENTORY-001"],
            "customer_situation": {
                "days_since_order": "0",
                "customer_expectation": "information",
                "complication": "none"
            },
            "email_patterns": {
                "should_mention": ["product_interest", "availability_question"],
                "might_omit": ["specific_model"],
                "tone_modifier": "curious"
            }
        },
        {
            "scenario_id": "GENERAL-002",
            "name": "order_modification",
            "description": "Customer wants to modify an existing order",
            "policy_clauses": ["POL-ORDER-001", "POL-SHIP-001"],
            "customer_situation": {
                "days_since_order": "0-2",
                "customer_expectation": "change_accepted",
                "complication": "already_processing"
            },
            "email_patterns": {
                "should_mention": ["order_id", "requested_change"],
                "might_omit": ["original_items"],
                "tone_modifier": "hopeful"
            }
        },
        {
            "scenario_id": "GENERAL-003",
            "name": "account_question",
            "description": "Customer has question about their account or loyalty status",
            "policy_clauses": ["POL-LOYALTY-001", "POL-COMM-001"],
            "customer_situation": {
                "days_since_order": "0",
                "customer_expectation": "clarification",
                "complication": "none"
            },
            "email_patterns": {
                "should_mention": ["account_question", "loyalty_tier"],
                "might_omit": ["specific_benefits"],
                "tone_modifier": "inquisitive"
            }
        }
    ]
}

# Scenario dimension probabilities for coverage
SCENARIO_DIMENSIONS = {
    "query_type": {
        "return_request": 0.25,
        "shipping_issue": 0.20,
        "product_defect": 0.15,
        "order_status": 0.15,
        "pricing_dispute": 0.10,
        "general_inquiry": 0.10,
        "warranty_claim": 0.05
    },
    
    "information_completeness": {
        "complete": 0.30,
        "missing_order_number": 0.25,
        "wrong_email": 0.20,
        "minimal_details": 0.15,
        "incorrect_info": 0.10
    },
    
    "complexity": {
        "straightforward": 0.40,
        "requires_lookup": 0.35,
        "edge_case": 0.20,
        "requires_escalation": 0.05
    },
    
    "customer_sentiment": {
        "neutral": 0.40,
        "frustrated": 0.30,
        "angry": 0.15,
        "confused": 0.15
    },
    
    "customer_history": {
        "new_customer": 0.25,
        "regular_customer": 0.50,
        "vip_customer": 0.15,
        "problem_customer": 0.10
    }
}

# Standard resolution action types
RESOLUTION_ACTIONS = [
    "process_return",
    "issue_refund",
    "send_replacement",
    "provide_tracking",
    "honor_warranty",
    "escalate_to_manager",
    "request_more_info",
    "apply_discount",
    "update_shipping_address",
    "cancel_order",
    "price_adjustment"
]

def extract_json_from_text(text: str) -> str:
    """Extract JSON from LLM response that might contain extra text."""
    # Try to find JSON array or object in the text
    import re
    
    # First, try to find content between ```json and ``` markers
    code_block_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    
    # Look for JSON array pattern
    array_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if array_match:
        return array_match.group(0)
    
    # Look for JSON object pattern
    object_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if object_match:
        return object_match.group(0)
    
    # If no JSON found, return original text
    return text


def safe_json_parse(text: str, expected_type: str = "array") -> Any:
    """Safely parse JSON with error handling and debugging."""
    if not text:
        print("Warning: Empty text received from LLM")
        return [] if expected_type == "array" else {}
    
    try:
        # First try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from text
        extracted = extract_json_from_text(text)
        try:
            result = json.loads(extracted)
            return result
        except json.JSONDecodeError as e:
            print(f"\nError parsing JSON: {e}")
            print(f"Raw text (first 500 chars): {text[:500]}...")
            if extracted != text:
                print(f"Extracted (first 500 chars): {extracted[:500]}...")
            
            # Return empty structure based on expected type
            if expected_type == "array":
                return []
            else:
                return {}


def call_llm(prompt, system_instruction=None):
    """Call the Gemini LLM with a prompt and return the response."""
    try:
        import os
        from google import genai
        from google.genai import types

        # Configure the client with API key
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                seed=42,
            ),
        )
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return f"Error: {str(e)}"


def weighted_choice(choices: Dict[str, float]) -> str:
    """Select a random choice based on weights."""
    items = list(choices.keys())
    weights = list(choices.values())
    return random.choices(items, weights=weights)[0]


def generate_company_policy(config: DatasetConfig) -> str:
    """Generate a comprehensive company policy document."""
    
    system_prompt = f"""You are creating a detailed customer service policy document for {config.company_name}, 
    a mid-sized consumer electronics retailer. The document should be professional but accessible,
    with clear section numbering for easy reference."""
    
    prompt = f"""Generate a comprehensive customer service policy document for {config.company_name} with the following sections:

1. Return Policy
   - Standard 30-day return policy with receipt
   - Extended 60-day holiday returns (Nov 15 - Jan 31)
   - Opened items: 15% restocking fee except defective
   - Special rules for different product categories
   - Use tags like [POL-RETURN-001], [POL-RETURN-002], etc.

2. Shipping Policy  
   - Standard shipping (5-7 days): Free over $50
   - Express shipping (2-3 days): $12.99
   - Damage claims must be filed within 48 hours
   - Wrong item shipped: Free return shipping
   - Use tags like [POL-SHIP-001], [POL-SHIP-002], etc.

3. Warranty Policy
   - Manufacturer warranty pass-through
   - TechNest Extended Warranty options
   - Warranty claim process
   - Use tags like [POL-WARRANTY-001], etc.

4. Price Match Policy
   - Match authorized retailers within 14 days
   - Exclusions: marketplace sellers, clearance
   - Use tags like [POL-PRICE-001], etc.

5. Customer Communication Guidelines
   - Professional, empathetic tone
   - Resolution timeframes
   - Escalation procedures
   - Use tags like [POL-COMM-001], etc.

6. Loyalty Program Benefits
   - Bronze: 2% cashback
   - Silver: 3% cashback + free standard shipping
   - Gold: 5% cashback + free express shipping + extended returns
   - Use tags like [POL-LOYALTY-001], etc.

Also include these sections that won't be used in tickets:
7. Employee Code of Conduct [POL-EMPLOYEE-001]
8. Supplier Terms and Conditions [POL-SUPPLIER-001]  
9. Environmental and Recycling Policy [POL-ENV-001]
10. Data Privacy and Security [POL-PRIVACY-001]

Make it realistic with specific details, edge cases, and examples.

IMPORTANT: Return the policy document as plain text, NOT as JSON."""
    
    return call_llm(prompt, system_prompt)


def generate_product_catalog(config: DatasetConfig) -> List[Dict]:
    """Generate a product catalog."""
    
    system_prompt = "You are generating product data for an electronics retailer."
    
    prompt = f"""Generate {config.num_products} electronic products in JSON format. Include:
    - product_id (PROD-XXXX format)
    - name
    - category (phones, laptops, accessories, audio, gaming, smart_home, cameras, tablets)
    - brand
    - base_price (realistic pricing from ${config.min_product_price} to ${config.max_product_price})
    - warranty_period (days, typically 365 for most items, 90 for accessories)
    - weight (in pounds)
    - requires_signature (boolean, true for items over ${config.high_value_threshold})
    - in_stock (boolean, 90% should be true)
    - description (brief)

Mix of premium and budget items. Format as a JSON array.
Example: [{{"product_id": "PROD-1001", "name": "UltraBook Pro 15", ...}}]

IMPORTANT: Return ONLY the JSON array, no explanatory text before or after."""
    
    products_text = call_llm(prompt, system_prompt)
    products = safe_json_parse(products_text, "array")
    
    if not products:
        print("Warning: No products generated, using fallback...")
        # Fallback products if generation fails
        products = [
            {
                "product_id": f"PROD-{1000+i}",
                "name": f"Test Product {i}",
                "category": "accessories",
                "brand": "Generic",
                "base_price": 29.99,
                "warranty_period": 90,
                "weight": 0.5,
                "requires_signature": False,
                "in_stock": True,
                "description": "Test product for development"
            }
            for i in range(min(5, config.num_products))
        ]
    
    print(f"  ✓ Generated {len(products)} products")
    return products


def generate_customers(config: DatasetConfig) -> List[Dict]:
    """Generate customer database."""
    
    system_prompt = "You are generating realistic customer data for testing purposes."
    
    prompt = f"""Generate {config.num_customers} customers in JSON format. Include:
    - customer_id (CUST-XXXX format)
    - name (realistic mix)
    - primary_email 
    - alternate_email (20% have 1 alternate email as string, not array)
    - phone
    - shipping_address (single address object with street, city, state, zip)
    - billing_address (single address object, 70% same as shipping)
    - loyalty_tier (bronze: 60%, silver: 30%, gold: 10%)
    - created_date (distribute over past {config.customer_history_days} days)
    - lifetime_value (realistic based on tier: bronze $50-500, silver $500-{config.vip_lifetime_value_threshold}, gold ${config.vip_lifetime_value_threshold}+)

Include some quirks:
- 5% have slight typos in alternate emails
- Some customers have maiden/married name variations
- Mix of gmail, yahoo, outlook, and custom domains

Format as JSON array.

IMPORTANT: Return ONLY the JSON array, no explanatory text before or after."""
    
    customers_text = call_llm(prompt, system_prompt)
    customers = safe_json_parse(customers_text, "array")
    
    if not customers:
        print("Warning: No customers generated, using fallback...")
        # Fallback customer if generation fails
        customers = [{
            "customer_id": "CUST-1001",
            "name": "Test Customer",
            "primary_email": "test@example.com",
            "alternate_email": "test2@example.com",
            "phone": "555-0100",
            "shipping_address": {"street": "123 Test St", "city": "Test City", "state": "CA", "zip": "94555"},
            "billing_address": {"street": "123 Test St", "city": "Test City", "state": "CA", "zip": "94555"},
            "loyalty_tier": "bronze",
            "created_date": "2024-01-01",
            "lifetime_value": 250.00
        }]
    
    print(f"  ✓ Generated {len(customers)} customers")
    return customers


def generate_single_order(customer: Dict, products: List[Dict], order_date: str, order_number: int) -> Dict:
    """Generate a single order for a specific customer and products."""
    
    system_prompt = "You are generating a realistic order record. Use ONLY the provided customer and product information."
    
    # Calculate total from products
    total = sum(p['base_price'] for p in products)
    
    prompt = f"""Create ONE order using EXACTLY this customer and product information:

Customer:
{json.dumps(customer, indent=2)}

Products to order:
{json.dumps(products, indent=2)}

Order date: {order_date}
Order sequence number: {order_number}

Generate an order with this schema:
- order_id: ORD-YYYYMMDD-XXXX format using the provided date and sequence number
- customer_id: Use EXACTLY the customer_id from the customer object above
- order_date: Use the provided date
- items: Array with one entry for each product above:
  - product_id: Use EXACTLY the product_id from the product
  - quantity: 1-2 for most items, rarely 3+
  - price_paid: Use the base_price from the product (might be slightly less if on sale)
  - item_status: "delivered" (90%), "shipped" (5%), "processing" (5%)
- shipping_method: "standard" (70%), "express" (25%), "overnight" (5%)
- tracking_number: Realistic format like 1Z999AA10123456784
- total_amount: Sum of all (price_paid * quantity)
- payment_method: "credit_card" (60%), "paypal" (25%), "apple_pay" (10%), "affirm" (5%)
- order_status: Based on item statuses - "delivered" if all delivered, etc.

IMPORTANT: 
- Use ONLY the customer_id from the provided customer
- Use ONLY the product_ids and base_prices from the provided products
- price_paid should be the base_price or slightly less (5-15% discount max)
- total_amount must equal the sum of all (price_paid * quantity)

Return ONLY the JSON object, no explanatory text."""
    
    order_text = call_llm(prompt, system_prompt)
    order = safe_json_parse(order_text, "object")
    
    # Validate and fix if needed
    if order and isinstance(order, dict):
        # Ensure customer_id matches
        order['customer_id'] = customer['customer_id']
        
        # Ensure product_ids match
        if 'items' in order:
            for i, item in enumerate(order['items']):
                if i < len(products):
                    item['product_id'] = products[i]['product_id']
                    # Ensure price is reasonable
                    if 'price_paid' not in item or item['price_paid'] > products[i]['base_price']:
                        item['price_paid'] = products[i]['base_price']
    
    return order


def generate_orders(config: DatasetConfig, customers: List[Dict], products: List[Dict]) -> List[Dict]:
    """Generate order history with consistent customer and product references."""
    
    print(f"  Generating {config.num_orders} orders...")
    orders = []
    
    # Create a date range for orders
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=config.order_history_days)
    
    # Customer order distribution
    # VIP customers (gold) get more orders
    weighted_customers = []
    for customer in customers:
        weight = 3 if customer.get('loyalty_tier') == 'gold' else 2 if customer.get('loyalty_tier') == 'silver' else 1
        weighted_customers.extend([customer] * weight)
    
    for i in range(config.num_orders):
        # Progress indicator
        if i % 10 == 0:
            print(f"    Generated {i}/{config.num_orders} orders...")
        
        # Random date with some clustering around holidays
        days_ago = random.randint(0, config.order_history_days)
        order_date = (end_date - datetime.timedelta(days=days_ago))
        
        # Black Friday/Cyber Monday clustering
        if order_date.month == 11 and 24 <= order_date.day <= 30:
            # Higher chance of orders during Black Friday week
            if random.random() < 0.3:  # 30% chance to skip to Black Friday week
                order_date = order_date.replace(day=random.randint(24, 30))
        
        order_date_str = order_date.strftime('%Y-%m-%d')
        
        # Select customer (weighted by tier)
        customer = random.choice(weighted_customers)
        
        # Select products (1-3 items per order, occasionally more)
        num_items = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.3, 0.15, 0.04, 0.01])[0]
        selected_products = random.sample(products, min(num_items, len(products)))
        
        # Generate order
        order = generate_single_order(customer, selected_products, order_date_str, i + 1001)
        
        if order and isinstance(order, dict):
            orders.append(order)
        else:
            # Fallback order if generation fails
            fallback_order = {
                "order_id": f"ORD-{order_date.strftime('%Y%m%d')}-{i + 1001}",
                "customer_id": customer["customer_id"],
                "order_date": order_date_str,
                "items": [
                    {
                        "product_id": prod["product_id"],
                        "quantity": 1,
                        "price_paid": prod["base_price"],
                        "item_status": "delivered"
                    }
                    for prod in selected_products
                ],
                "shipping_method": "standard",
                "tracking_number": f"1Z999AA10{random.randint(10000000, 99999999)}",
                "total_amount": sum(p["base_price"] for p in selected_products),
                "payment_method": "credit_card",
                "order_status": "delivered"
            }
            orders.append(fallback_order)
    
    # Add some returns/refunds to random orders
    num_returns = int(len(orders) * config.return_rate)
    orders_with_returns = random.sample(range(len(orders)), num_returns)
    
    for idx in orders_with_returns:
        if orders[idx]["items"]:
            # Mark random item as returned
            item_idx = random.randint(0, len(orders[idx]["items"]) - 1)
            orders[idx]["items"][item_idx]["item_status"] = random.choice(["returned", "refunded"])
            orders[idx]["order_status"] = "partially_returned"
    
    print(f"  ✓ Generated {len(orders)} orders")
    return orders


def select_scenario_template(query_type: str, order: Dict, customer: Dict, products: List[Dict]) -> Dict:
    """Select and customize a scenario template based on order details."""
    
    # Get templates for this query type
    templates = SCENARIO_TEMPLATES.get(query_type, [])
    if not templates:
        # Fallback if query type not found
        templates = SCENARIO_TEMPLATES["return_request"]
    
    # Select a random template
    template = random.choice(templates)
    
    # Calculate days since order (if order exists)
    if order and "order_date" in order:
        order_date = datetime.datetime.strptime(order["order_date"], "%Y-%m-%d")
        days_since_order = (datetime.datetime.now() - order_date).days
    else:
        days_since_order = 0
    
    # Customize template based on actual order data
    scenario = {
        "scenario_id": template["scenario_id"],
        "name": template["name"],
        "description": template["description"],
        "policy_clauses": template["policy_clauses"],
        "query_type": query_type,
        "order": order,
        "customer": customer,
        "products": products,
        "days_since_order": days_since_order,
        "customer_situation": template["customer_situation"].copy(),
        "email_patterns": template["email_patterns"].copy()
    }
    
    # Adjust for VIP customers
    if customer.get("loyalty_tier") == "gold":
        scenario["customer_situation"]["complication"] = "vip_customer"
    
    return scenario


def generate_customer_email_v2(scenario: Dict, dimensions: Dict[str, str]) -> Dict:
    """Generate a customer email based on real order data and scenario."""
    
    system_prompt = """You are simulating realistic customer support emails. 
    Use ONLY the provided order, customer, and product information. Do not invent any details."""
    
    order = scenario.get("order")
    customer = scenario["customer"]
    products = scenario.get("products", [])
    
    # Build product details string
    product_details = []
    if order and "items" in order:
        for item in order["items"]:
            matching_product = next((p for p in products if p["product_id"] == item["product_id"]), None)
            if matching_product:
                product_details.append({
                    "name": matching_product["name"],
                    "price_paid": item["price_paid"],
                    "quantity": item["quantity"]
                })
    
    # Build order information section
    if order:
        order_info = f"""ORDER INFORMATION (use exactly):
- Order ID: {order['order_id']}
- Order Date: {order['order_date']} ({scenario['days_since_order']} days ago)
- Products: {json.dumps(product_details)}
- Total: ${order['total_amount']}
- Shipping Method: {order['shipping_method']}
- Tracking: {order['tracking_number']}
- Status: {order['order_status']}"""
    else:
        order_info = "ORDER INFORMATION: No specific order (general inquiry)"
    
    prompt = f"""Write a customer support email based on this EXACT information:

SCENARIO: {scenario['description']}
Query Type: {scenario['query_type']}

CUSTOMER INFORMATION (use exactly):
- Name: {customer['name']}
- Email: {customer['primary_email']}
- Loyalty Tier: {customer['loyalty_tier']}

{order_info}

EMAIL REQUIREMENTS:
- Sentiment: {dimensions['customer_sentiment']} with {scenario['email_patterns']['tone_modifier']} modifier
- Should mention: {scenario['email_patterns']['should_mention']}
- Might omit: {scenario['email_patterns']['might_omit']}
- Information completeness: {dimensions['information_completeness']}

Based on the scenario and information completeness:
- If "complete": Include order number and all relevant details
- If "missing_order_number": Don't mention the order ID
- If "wrong_email": Sign with a different email like {customer['name'].lower().replace(' ', '.')}@gmail.com
- If "minimal_details": Be vague about specifics
- If "incorrect_info": Slightly misremember one detail (like confusing product names)

Write a natural email that fits this scenario. The customer is contacting support because: {scenario['description']}

Format as JSON:
{{
    "subject": "Subject line",
    "body": "Complete email body with greeting and signature",
    "from_email": "email address used"
}}

IMPORTANT: Use ONLY the information provided above. Do not invent order numbers, product names, dates, or prices."""
    
    email_text = call_llm(prompt, system_prompt)
    email = safe_json_parse(email_text, "object")
    
    # Validate email
    if not email or not isinstance(email, dict):
        # Fallback email
        subject = f"Issue with order {order['order_id']}" if order and dimensions['information_completeness'] == 'complete' else "Need help"
        email = {
            "subject": subject,
            "body": f"Hello,\n\nI need help with my {'recent order' if order else 'account'}. Please assist.\n\nThank you,\n{customer['name']}",
            "from_email": customer["primary_email"]
        }
    
    return email


def generate_resolution_v2(email: Dict, scenario: Dict, policy: str, dimensions: Dict[str, str]) -> Dict:
    """Generate a resolution using exact order data and pre-selected policies."""
    
    system_prompt = """You are an expert customer service professional. Create resolutions using ONLY 
    the provided information and cite ONLY the specified policy sections."""
    
    order = scenario.get("order")
    customer = scenario["customer"]
    products = scenario.get("products", [])
    
    # Extract only the relevant policy sections
    relevant_policies = []
    for clause in scenario["policy_clauses"]:
        # Find lines containing this policy clause
        policy_lines = [line for line in policy.split('\n') if clause in line]
        if policy_lines:
            # Get the section around this clause
            start_idx = policy.find(policy_lines[0])
            end_idx = policy.find('\n\n', start_idx)
            if end_idx == -1:
                end_idx = len(policy)
            relevant_policies.append(policy[start_idx:end_idx])
    
    relevant_policy_text = "\n\n".join(relevant_policies)
    
    # Build order information section
    if order:
        order_info = f"""VERIFIED ORDER INFORMATION:
- Order ID: {order['order_id']}
- Order Date: {order['order_date']}
- Items: {json.dumps(order['items'], indent=2)}
- Total: ${order['total_amount']}
- Status: {order['order_status']}

PRODUCTS IN ORDER:
{json.dumps(products, indent=2)}"""
    else:
        order_info = "VERIFIED ORDER INFORMATION: No specific order (general inquiry)"
    
    prompt = f"""Create a professional resolution for this customer support case:

EMAIL FROM CUSTOMER:
{json.dumps(email, indent=2)}

VERIFIED CUSTOMER INFORMATION:
- Customer ID: {customer['customer_id']}
- Name: {customer['name']}
- Email: {customer['primary_email']}
- Loyalty Tier: {customer['loyalty_tier']}
- Lifetime Value: ${customer.get('lifetime_value', 0)}

{order_info}

SCENARIO CONTEXT:
- Issue Type: {scenario['description']}
- Customer Expectation: {scenario['customer_situation']['customer_expectation']}
- Complication: {scenario['customer_situation']['complication']}
- Days Since Order: {scenario.get('days_since_order', 'N/A')}

RELEVANT POLICIES (cite these EXACT sections):
{relevant_policy_text}

REQUIRED POLICY REFERENCES: {scenario['policy_clauses']}

Create a resolution with this structure:
{{
    "customer_lookup": {{
        "status": "found",
        "customer_id": "{customer['customer_id']}",
        "lookup_method": "email_match" or "order_lookup",
        "notes": "Any lookup observations"
    }},
    "policy_references": {scenario['policy_clauses']},
    "actions": [
        {{
            "type": "action from {RESOLUTION_ACTIONS}",
            "reason": "Detailed reason citing specific policies",
            "value": dollar amount if applicable,
            "details": "Specific implementation details"
        }}
    ],
    "escalation_required": boolean,
    "escalation_reason": "Why escalation needed" or null,
    "priority": "low/medium/high/urgent",
    "total_resolution_value": total monetary impact,
    "followup_required": {{
        "needed": boolean,
        "reason": "Why followup needed" or null,
        "questions": ["Specific questions to ask"] or []
    }}
}}

IMPORTANT: 
- Use ONLY the customer_id, order_id, and product_ids provided above
- Cite ONLY the policy clauses listed in REQUIRED POLICY REFERENCES
- Base actions on the actual scenario and customer tier
- If VIP customer, consider exceptions per loyalty policies

Return ONLY the JSON object."""
    
    resolution_text = call_llm(prompt, system_prompt)
    resolution = safe_json_parse(resolution_text, "object")
    
    # Validate and ensure consistency
    if resolution and isinstance(resolution, dict):
        # Ensure correct customer_id
        if "customer_lookup" in resolution:
            resolution["customer_lookup"]["customer_id"] = customer["customer_id"]
        
        # Ensure only specified policies are referenced
        resolution["policy_references"] = scenario["policy_clauses"]
    else:
        # Fallback resolution
        resolution = {
            "customer_lookup": {
                "status": "found",
                "customer_id": customer["customer_id"],
                "lookup_method": "email_match",
                "notes": None
            },
            "policy_references": scenario["policy_clauses"],
            "actions": [{
                "type": "process_return" if "return" in scenario["query_type"] else "request_more_info",
                "reason": f"Per policy {scenario['policy_clauses'][0]}",
                "value": 0,
                "details": "Processing according to policy"
            }],
            "escalation_required": False,
            "escalation_reason": None,
            "priority": "medium",
            "total_resolution_value": 0,
            "followup_required": {
                "needed": True,
                "reason": "Initial response sent",
                "questions": []
            }
        }
    
    return resolution


def create_complete_ticket_v2(config: DatasetConfig, scenario: Dict, email: Dict, 
                             resolution: Dict, dimensions: Dict[str, str]) -> Dict:
    """Combine all elements into a complete ticket with full traceability."""
    
    ticket = {
        "ticket_id": f"TK-{datetime.datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}",
        "customer_email": email["from_email"],
        "subject": email["subject"],
        "body": email["body"],
        "timestamp": datetime.datetime.now().isoformat(),
        "customer_id": scenario["customer"]["customer_id"],
        "resolution_plan": resolution
    }
    
    # Add order_id if there's an order
    if scenario.get("order"):
        ticket["order_id"] = scenario["order"]["order_id"]
    
    # Add debug info if configured
    if config.include_debug_info:
        ticket["_scenario_dimensions"] = dimensions
        ticket["_scenario_template"] = {
            "scenario_id": scenario["scenario_id"],
            "name": scenario["name"],
            "policy_clauses": scenario["policy_clauses"]
        }
    
    return ticket


def load_existing_data(config: DatasetConfig) -> Tuple[str, List[Dict], List[Dict], List[Dict]]:
    """Load existing data for append mode."""
    
    # Load policy
    policy_path = config.get_filepath(config.policy_file)
    with open(policy_path, "r") as f:
        policy = f.read()
    
    # Load database
    db_path = config.get_filepath(config.database_file)
    with open(db_path, "r") as f:
        database = json.load(f)
    
    return policy, database["customers"], database["orders"], database["products"]


def load_existing_tickets(config: DatasetConfig) -> List[Dict]:
    """Load existing tickets for append mode."""
    
    tickets_path = config.get_filepath(config.tickets_file)
    if os.path.exists(tickets_path):
        with open(tickets_path, "r") as f:
            return json.load(f)
    return []


def save_dataset(config: DatasetConfig, tickets: List[Dict], policy: str, 
                customers: List[Dict], orders: List[Dict], products: List[Dict]):
    """Save all generated data to files."""
    
    # Create output directory if needed
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save tickets
    tickets_path = config.get_filepath(config.tickets_file)
    with open(tickets_path, "w") as f:
        json.dump(tickets, f, indent=2)
    
    # Save policy (only in create mode)
    if config.mode == "create":
        policy_path = config.get_filepath(config.policy_file)
        with open(policy_path, "w") as f:
            f.write(policy)
    
    # Save database
    database = {
        "customers": customers,
        "orders": orders,
        "products": products
    }
    db_path = config.get_filepath(config.database_file)
    with open(db_path, "w") as f:
        json.dump(database, f, indent=2)
    
    print(f"\nDataset saved to '{config.output_dir}':")
    print(f"- {len(tickets)} tickets in {config.tickets_file}")
    if config.mode == "create":
        print(f"- Company policy in {config.policy_file}")
    print(f"- Database with {len(customers)} customers, {len(orders)} orders, {len(products)} products")


def main(config: DatasetConfig):
    """Main generation pipeline."""
    
    print("=== Synthetic Customer Support Dataset Generator ===")
    print(f"Mode: {config.mode}")
    print(f"Output directory: {config.output_dir}")
    
    if config.mode == "append":
        # Load existing data
        print("\nLoading existing data...")
        try:
            policy, customers, orders, products = load_existing_data(config)
            existing_tickets = load_existing_tickets(config)
            print(f"Loaded {len(existing_tickets)} existing tickets")
        except FileNotFoundError as e:
            print(f"Error: Could not find existing data files. Please run in 'create' mode first.")
            print(f"Missing file: {e}")
            return
    else:
        # Create mode - generate everything from scratch
        existing_tickets = []
        
        # Phase 1: Company Foundation
        print("\nPhase 1: Generating company foundation...")
        print("- Generating company policy document...")
        policy = generate_company_policy(config)
        
        print(f"- Generating {config.num_products} products...")
        products = generate_product_catalog(config)
        
        # Phase 2: Customer & Order Generation
        print("\nPhase 2: Generating customers and orders...")
        print(f"- Generating {config.num_customers} customers...")
        customers = generate_customers(config)
        
        print(f"- Generating {config.num_orders} orders...")
        orders = generate_orders(config, customers, products)
    
    # Phase 3: Ticket Generation
    print(f"\nPhase 3: Generating {config.num_tickets} support tickets...")
    new_tickets = []
    
    # Filter orders that can be used for tickets (delivered/shipped)
    eligible_orders = [o for o in orders if o["order_status"] in ["delivered", "shipped", "partially_returned"]]
    if not eligible_orders:
        print("Warning: No eligible orders for ticket generation, using all orders")
        eligible_orders = orders
    
    if not eligible_orders:
        print("Error: No orders available for ticket generation")
        return
    
    for i in range(config.num_tickets):
        print(f"\nGenerating ticket {i+1}/{config.num_tickets}")
        
        # Roll scenario dimensions first
        dimensions = {
            dim: weighted_choice(choices) 
            for dim, choices in SCENARIO_DIMENSIONS.items()
        }
        
        # For general inquiries, we might not need a specific order
        if dimensions['query_type'] == 'general_inquiry' and random.random() < 0.5:
            # 50% of general inquiries don't relate to a specific order
            order = None
            # Just pick a random customer
            customer = random.choice(customers)
            # But they might ask about products, so pick some random products
            order_products = random.sample(products, min(3, len(products)))
        else:
            # Select a random order
            order = random.choice(eligible_orders)
            
            # Find the customer for this order
            customer = next((c for c in customers if c["customer_id"] == order["customer_id"]), None)
            if not customer:
                print(f"Warning: Customer not found for order {order['order_id']}")
                continue
            
            # Get the products in this order
            order_products = []
            for item in order["items"]:
                product = next((p for p in products if p["product_id"] == item["product_id"]), None)
                if product:
                    order_products.append(product)
            
            if not order_products:
                print(f"Warning: No products found for order {order['order_id']}")
                continue
        print(f"  Dimensions: {dimensions['query_type']} / {dimensions['complexity']}")
        if order:
            print(f"  Order: {order['order_id']} / Customer: {customer['customer_id']}")
        else:
            print(f"  Customer: {customer['customer_id']} (no specific order)")
        
        # Select and customize scenario template
        scenario = select_scenario_template(dimensions['query_type'], order, customer, order_products)
        print(f"  Scenario: {scenario['name']} with policies {scenario['policy_clauses']}")
        
        # Generate email
        email = generate_customer_email_v2(scenario, dimensions)
        
        # Generate resolution
        resolution = generate_resolution_v2(email, scenario, policy, dimensions)
        
        # Create complete ticket
        ticket = create_complete_ticket_v2(config, scenario, email, resolution, dimensions)
        new_tickets.append(ticket)
        
        print(f"  ✓ Ticket {ticket['ticket_id']} generated")
    
    # Combine tickets for append mode
    all_tickets = existing_tickets + new_tickets if config.mode == "append" else new_tickets
    
    # Phase 4: Save Dataset
    print("\nPhase 4: Saving dataset...")
    save_dataset(config, all_tickets, policy, customers, orders, products)
    
    print("\n=== Generation Complete! ===")
    
    # Print summary statistics
    print(f"\nDataset Statistics (new tickets):")
    if config.include_debug_info and new_tickets:
        query_types = {}
        complexities = {}
        scenario_names = {}
        for ticket in new_tickets:
            if "_scenario_dimensions" in ticket:
                qt = ticket["_scenario_dimensions"]["query_type"]
                cx = ticket["_scenario_dimensions"]["complexity"]
                query_types[qt] = query_types.get(qt, 0) + 1
                complexities[cx] = complexities.get(cx, 0) + 1
            
            if "_scenario_template" in ticket:
                name = ticket["_scenario_template"]["name"]
                scenario_names[name] = scenario_names.get(name, 0) + 1
        
        if query_types:
            print("\nQuery Type Distribution:")
            for qt, count in sorted(query_types.items()):
                print(f"  {qt}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if complexities:
            print("\nComplexity Distribution:")
            for cx, count in sorted(complexities.items()):
                print(f"  {cx}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if scenario_names:
            print("\nScenario Template Distribution:")
            for name, count in sorted(scenario_names.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {count} ({count/len(new_tickets)*100:.1f}%)")
    else:
        print("Debug information not included in tickets.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic customer support dataset")
    
    # Main parameters
    parser.add_argument("--tickets", type=int, help="Number of tickets to generate")
    parser.add_argument("--customers", type=int, help="Number of customers to generate")
    parser.add_argument("--products", type=int, help="Number of products to generate")
    parser.add_argument("--orders", type=int, help="Number of orders to generate")
    
    # Mode and output
    parser.add_argument("--mode", choices=["create", "append"], help="Generation mode")
    parser.add_argument("--output-dir", type=str, help="Output directory for files")
    
    # Optional parameters
    parser.add_argument("--company-name", type=str, help="Company name for policy")
    parser.add_argument("--no-debug", action="store_true", help="Exclude debug info from tickets")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create config with defaults
    config = DatasetConfig()
    
    # Override with command line arguments if provided
    if args.tickets is not None:
        config.num_tickets = args.tickets
    if args.customers is not None:
        config.num_customers = args.customers
    if args.products is not None:
        config.num_products = args.products
    if args.orders is not None:
        config.num_orders = args.orders
    if args.mode is not None:
        config.mode = args.mode
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.company_name is not None:
        config.company_name = args.company_name
    if args.no_debug:
        config.include_debug_info = False
    
    # For testing, use smaller numbers by default
    if len(sys.argv) == 1:  # No arguments provided
        print("No arguments provided. Using test configuration with small dataset.")
        config.num_tickets = 5
        config.num_products = 6
        config.num_customers = 5
        config.num_orders = 11
    
    # Run generation
    main(config)