# 1. Fix RESOLUTION_ACTIONS - remove apply_discount
RESOLUTION_ACTIONS = [
    "process_return",
    "issue_refund", 
    "send_replacement",
    "provide_tracking",
    "honor_warranty",
    "escalate_to_manager",
    "request_more_info",
    # "apply_discount",  # REMOVED
    "update_shipping_address",
    "cancel_order",
    # "price_adjustment"  # REMOVED - too similar to discount
]

# 2. Simplified policy generation function
def generate_company_policy(config: DatasetConfig) -> str:
    """Generate a simple, rule-based company policy document."""
    
    system_prompt = f"""You are creating a simple, rule-based customer service policy document for {config.company_name}, 
    a consumer electronics retailer. Every rule must be absolute with no ambiguity - only yes/no, allowed/not allowed."""
    
    prompt = f"""Generate a simple customer service policy document for {config.company_name} with clear, discrete rules:

1. Return Policy [POL-RETURN-001 to POL-RETURN-004]
   - ALL items: 30-day return window from delivery date with receipt
   - Unopened items: Full refund
   - Opened items: 15% restocking fee  
   - Defective items: Full refund, no restocking fee
   - No returns accepted after 30 days for any reason

2. Shipping Policy [POL-SHIP-001 to POL-SHIP-004]
   - Standard shipping (5-7 days): Free on orders over $50, otherwise $9.99
   - Express shipping (2-3 days): $19.99
   - Package not received: Replacement sent after carrier investigation (3 business days)
   - Wrong item shipped: Free return label provided, correct item sent
   - Damaged items over $500: Photo required before replacement approved
   - Damaged items under $500: Immediate replacement

3. Warranty Policy [POL-WARRANTY-001 to POL-WARRANTY-003]
   - Manufacturer warranty: 1 year from purchase date
   - Defects covered: Manufacturing defects only
   - Not covered: Physical damage, water damage, normal wear
   - Process: Return to {config.company_name} within warranty period

4. Price Match Policy [POL-PRICE-001 to POL-PRICE-002]
   - Authorized retailers only: Price matched within 14 days of purchase
   - Exclusions: Marketplace sellers, clearance items, bundles
   - Refund method: Original payment method

5. Customer Service Standards [POL-COMM-001 to POL-COMM-002]
   - Response time: Within 24 hours
   - Escalation: Available for orders over $1000 or VIP Gold customers

6. Loyalty Program [POL-LOYALTY-001]
   - Bronze: 2% cashback
   - Silver: 3% cashback + free standard shipping  
   - Gold: 5% cashback + free express shipping + 45-day returns

IMPORTANT: 
- Use exact policy tags like [POL-RETURN-001]
- No conditional language (no "may", "might", "could", "should consider")
- Every rule is absolute - yes or no, allowed or not allowed
- Include specific dollar amounts and time periods
- Return as plain text, not JSON"""
    
    return call_llm(prompt, system_prompt)

# 3. Fix scenario templates to ensure date validation is correct
# Update SCENARIO_TEMPLATES for return_request to be more explicit
SCENARIO_TEMPLATES["return_request"] = [
    {
        "scenario_id": "RETURN-001",
        "name": "return_outside_window",
        "description": "Customer wants to return item outside 30-day window",
        "policy_clauses": ["POL-RETURN-001"],
        "customer_situation": {
            "days_since_purchase": "31-90",  # Clearly outside 30-day window
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
        "description": "Customer wants to return opened electronic item within window",
        "policy_clauses": ["POL-RETURN-001", "POL-RETURN-003"],
        "customer_situation": {
            "days_since_purchase": "10-25",  # Clearly within 30-day window
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
        "name": "return_defective_item",
        "description": "Customer received defective item",
        "policy_clauses": ["POL-RETURN-001", "POL-RETURN-004"],
        "customer_situation": {
            "days_since_purchase": "1-20",  # Within window
            "customer_expectation": "full_refund_no_fee",
            "complication": "item_defective"
        },
        "email_patterns": {
            "should_mention": ["product_name", "defect_description"],
            "might_omit": ["troubleshooting_attempted"],
            "tone_modifier": "frustrated"
        }
    }
]

# 4. Updated resolution generation to respect date boundaries
def generate_resolution_v2(email: Dict, scenario: Dict, policy: str, dimensions: Dict[str, str]) -> Dict:
    """Generate a resolution using exact order data and pre-selected policies."""
    
    system_prompt = """You are an expert customer service professional. Create resolutions using ONLY 
    the provided information and cite ONLY the specified policy sections. Follow policies exactly - no exceptions
    unless customer is VIP Gold tier."""
    
    order = scenario.get("order")
    customer = scenario["customer"]
    products = scenario.get("products", [])
    
    # Extract only the relevant policy sections
    relevant_policies = []
    for clause in scenario["policy_clauses"]:
        policy_lines = [line for line in policy.split('\n') if clause in line]
        if policy_lines:
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
- Days Since Order: {scenario.get('days_since_order', 'N/A')}
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

RELEVANT POLICIES (follow these EXACTLY):
{relevant_policy_text}

REQUIRED POLICY REFERENCES: {scenario['policy_clauses']}

CRITICAL RULES:
1. If days_since_order > 30, return is OUTSIDE window (policy says 30 days)
2. Never offer discounts or price adjustments as resolution
3. For damaged items over $500, MUST request photo before approving replacement
4. Only exception to policies: Gold tier customers get 45-day return window

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
            "reason": "Detailed reason citing specific policies and days since order",
            "value": dollar amount if applicable (refund amount, not discounts),
            "details": "Specific implementation details"
        }}
    ],
    "escalation_required": boolean,
    "escalation_reason": "Why escalation needed" or null,
    "priority": "low/medium/high/urgent",
    "total_resolution_value": total monetary impact,
    "followup_required": {{
        "needed": boolean,
        "reason": "Why followup needed (e.g., need photo for damaged item over $500)" or null,
        "questions": ["Specific questions to ask"] or []
    }}
}}

IMPORTANT: 
- Check days_since_order against 30-day policy (or 45 for Gold)
- NO discounts or price adjustments - only refunds/replacements per policy
- For damaged items, check if value > $500 to require photo
- Be explicit about whether request is within or outside return window

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
        
        # Additional validation for date logic
        if "actions" in resolution and scenario.get("days_since_order") is not None:
            days = scenario["days_since_order"]
            return_window = 45 if customer.get("loyalty_tier") == "gold" else 30
            
            for action in resolution["actions"]:
                if action["type"] == "process_return" and days > return_window:
                    # Ensure we're not processing returns outside window
                    action["type"] = "escalate_to_manager"
                    action["reason"] = f"Return request is outside {return_window}-day window (order is {days} days old)"
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