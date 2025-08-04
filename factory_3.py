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
    
    # Order parameters
    order_history_days: int = 180  # 6 months
    return_rate: float = 0.10  # 10% of orders have returns
    
    # Ticket parameters
    include_debug_info: bool = True  # Include hidden scenario dimensions
    
    def get_filepath(self, filename: str) -> str:
        """Get full filepath for a given filename"""
        return os.path.join(self.output_dir, filename)

# Enhanced Policy Structure with Interactions
@dataclass
class PolicyClause:
    """Represents a single policy clause with interaction metadata"""
    clause_id: str
    title: str
    rule: str
    conditions: List[str] = field(default_factory=list)
    interacts_with: List[str] = field(default_factory=list)
    modifies: List[str] = field(default_factory=list)
    modified_by: List[str] = field(default_factory=list)
    overrides: List[str] = field(default_factory=list)
    overridden_by: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    precedence: int = 5  # Lower numbers have higher precedence
    category: str = ""

class PolicyGraph:
    """Manages policy clauses and their interactions"""
    
    def __init__(self):
        self.clauses: Dict[str, PolicyClause] = {}
        self.interaction_graph: Dict[str, List[str]] = {}
    
    def add_clause(self, clause: PolicyClause):
        """Add a policy clause to the graph"""
        self.clauses[clause.clause_id] = clause
        
        # Build interaction graph
        all_interactions = (clause.interacts_with + clause.modifies + clause.modified_by + 
                          clause.overrides + clause.overridden_by + clause.requires)
        self.interaction_graph[clause.clause_id] = all_interactions
    
    def get_related_policies(self, clause_id: str, max_hops: int = 3) -> List[str]:
        """Get all policies related to a given clause within max_hops"""
        if clause_id not in self.clauses:
            return []
        
        visited = set()
        to_visit = [(clause_id, 0)]
        related = []
        
        while to_visit:
            current_id, hops = to_visit.pop(0)
            
            if current_id in visited or hops > max_hops:
                continue
                
            visited.add(current_id)
            if current_id != clause_id:  # Don't include the starting clause
                related.append(current_id)
            
            # Add connected clauses
            if current_id in self.interaction_graph:
                for connected_id in self.interaction_graph[current_id]:
                    if connected_id not in visited:
                        to_visit.append((connected_id, hops + 1))
        
        return related
    
    def resolve_conflicts(self, clause_ids: List[str], context: Dict[str, Any]) -> List[str]:
        """Resolve conflicts between clauses based on precedence and context"""
        if not clause_ids:
            return []
        
        # Sort by precedence (lower numbers first)
        sorted_clauses = sorted(clause_ids, key=lambda cid: self.clauses[cid].precedence)
        
        active_clauses = []
        for clause_id in sorted_clauses:
            clause = self.clauses[clause_id]
            
            # Check if this clause overrides any active clauses
            for active_id in active_clauses[:]:
                if active_id in clause.overrides:
                    active_clauses.remove(active_id)
            
            # Check if any active clause overrides this one
            is_overridden = any(clause_id in self.clauses[active_id].overrides 
                             for active_id in active_clauses)
            
            if not is_overridden:
                # Check if conditions are met
                if self._check_conditions(clause, context):
                    active_clauses.append(clause_id)
        
        return active_clauses
    
    def _check_conditions(self, clause: PolicyClause, context: Dict[str, Any]) -> bool:
        """Check if clause conditions are met given context"""
        for condition in clause.conditions:
            if condition == "receipt_required" and not context.get("has_receipt", True):
                return False
            elif condition == "within_return_window" and context.get("days_since_purchase", 0) > 30:
                return False
            elif condition == "within_price_match_window" and context.get("days_since_purchase", 0) > 14:
                return False
            elif condition == "order_not_shipped" and context.get("order_status") in ["shipped", "delivered"]:
                return False
            elif condition == "item_over_500" and context.get("item_value", 0) <= 500:
                return False
            elif condition == "warranty_period" and context.get("months_since_purchase", 0) > 12:
                return False
        return True
    
    def generate_policy_text(self) -> str:
        """Generate human-readable policy document (without metadata)"""
        categories = {}
        for clause in self.clauses.values():
            if clause.category not in categories:
                categories[clause.category] = []
            categories[clause.category].append(clause)
        
        policy_text = []
        
        for category, clauses in categories.items():
            policy_text.append(f"\n{category}")
            policy_text.append("=" * len(category))
            
            for clause in sorted(clauses, key=lambda c: c.clause_id):
                policy_text.append(f"\n[{clause.clause_id}] {clause.title}")
                policy_text.append(f"Rule: {clause.rule}")
                if clause.conditions:
                    policy_text.append(f"Conditions: {', '.join(clause.conditions)}")
        
        return "\n".join(policy_text)

def create_policy_graph(config: DatasetConfig) -> PolicyGraph:
    """Create the complete policy graph with all interactions"""
    graph = PolicyGraph()
    
    # Return Policy Clauses
    graph.add_clause(PolicyClause(
        clause_id="POL-RETURN-001",
        title="Return Window",
        rule="ALL items: 30-day return window from delivery date. No returns accepted after 30 days for any reason.",
        conditions=["within_return_window", "receipt_required"],
        modified_by=["POL-RETURN-004", "POL-HOLIDAY-001"],
        interacts_with=["POL-RETURN-002", "POL-RETURN-003"],  # Symmetric relationships
        precedence=3,
        category="Return Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-RETURN-002",
        title="Restocking Fee",
        rule="Opened items: 15% restocking fee applies unless item is defective",
        conditions=["within_return_window"],
        overridden_by=["POL-RETURN-004"],
        modified_by=["POL-RETURN-003"],
        interacts_with=["POL-RETURN-001", "POL-RETURN-003"],  # Symmetric with 001 & 003
        precedence=4,
        category="Return Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-RETURN-003",
        title="Unopened Items",
        rule="Unopened items: Full refund, no restocking fee",
        conditions=["within_return_window"],
        modifies=["POL-RETURN-002"],
        interacts_with=["POL-RETURN-001", "POL-RETURN-002"],  # Symmetric with 001 & 002
        precedence=3,
        category="Return Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-RETURN-004",
        title="Defective Items",
        rule="Defective items: Full refund, no restocking fee regardless of opened status",
        conditions=["within_return_window"],
        overrides=["POL-RETURN-002"],
        modifies=["POL-RETURN-001"],
        precedence=1,  # Highest precedence
        category="Return Policy"
    ))
    
    # Exchange Policy Clauses
    graph.add_clause(PolicyClause(
        clause_id="POL-EXCHANGE-001",
        title="Exchange Window",
        rule="Exchanges allowed within 30 days for same or higher value item. Customer pays price difference if applicable.",
        conditions=["within_return_window", "receipt_required"],
        interacts_with=["POL-RETURN-001"],
        precedence=3,
        category="Exchange Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-EXCHANGE-002",
        title="Defective Item Exchange",
        rule="Defective items exchanged for same item at no cost. If same item unavailable, full refund or credit offered.",
        overrides=["POL-EXCHANGE-001"],
        precedence=1,
        category="Exchange Policy"
    ))
    
    # Shipping Policy Clauses
    graph.add_clause(PolicyClause(
        clause_id="POL-SHIP-001",
        title="Shipping Costs",
        rule="Standard shipping (5-7 days): Free on orders over $50, otherwise $9.99",
        precedence=5,
        category="Shipping Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-SHIP-002",
        title="Package Not Received",
        rule="Package not received: Replacement sent after carrier investigation (3 business days)",
        requires=["POL-SHIP-003"],
        precedence=4,
        category="Shipping Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-SHIP-003",
        title="Investigation Required",
        rule="Investigation period: 3 business days required before replacement for lost packages",
        precedence=3,
        category="Shipping Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-SHIP-004",
        title="Wrong Item Shipped",
        rule="Wrong item shipped: Free return label provided, correct item sent immediately. No time limit - this is merchant error.",
        precedence=2,
        category="Shipping Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-SHIP-005",
        title="Damage Claims High Value",
        rule="Damaged items over $500: Photo required before replacement approved",
        conditions=["item_over_500"],
        modifies=["POL-SHIP-006"],
        precedence=2,
        category="Shipping Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-SHIP-006",
        title="Damage Claims Standard",
        rule="Damaged items under $500: Immediate replacement authorized",
        modified_by=["POL-SHIP-005"],
        precedence=4,
        category="Shipping Policy"
    ))
    
    # Warranty Policy Clauses
    graph.add_clause(PolicyClause(
        clause_id="POL-WARRANTY-001",
        title="Warranty Period",
        rule="Manufacturer warranty: 1 year from purchase date. No warranty service after 1 year",
        conditions=["warranty_period"],
        interacts_with=["POL-WARRANTY-002", "POL-WARRANTY-003"],
        precedence=3,
        category="Warranty Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-WARRANTY-002",
        title="Manufacturing Defects",
        rule="Defects covered: Manufacturing defects only",
        conditions=["warranty_period"],
        interacts_with=["POL-WARRANTY-001", "POL-WARRANTY-003"],  # Symmetric relationships
        overridden_by=["POL-WARRANTY-003"],  # Symmetric with overrides
        precedence=3,
        category="Warranty Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-WARRANTY-003",
        title="Warranty Exclusions",
        rule="Not covered: Physical damage, water damage, normal wear",
        conditions=["warranty_period"],
        interacts_with=["POL-WARRANTY-001", "POL-WARRANTY-002"],  # Symmetric relationships
        overrides=["POL-WARRANTY-002"],
        precedence=2,
        category="Warranty Policy"
    ))
    
    # Price Match Policy Clauses
    graph.add_clause(PolicyClause(
        clause_id="POL-PRICE-001",
        title="Price Match Window",
        rule="Authorized retailers only: Price matched within 14 days of purchase. No price match after 14 days",
        conditions=["within_price_match_window"],
        interacts_with=["POL-PRICE-002"],
        modified_by=["POL-PRICE-002"],  # Symmetric with modifies
        precedence=3,
        category="Price Match Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-PRICE-002",
        title="Price Match Exclusions",
        rule="Exclusions: Marketplace sellers, clearance items, bundles",
        interacts_with=["POL-PRICE-001"],  # Symmetric relationship
        modifies=["POL-PRICE-001"],
        precedence=2,
        category="Price Match Policy"
    ))
    
    # Order Policy Clauses
    graph.add_clause(PolicyClause(
        clause_id="POL-ORDER-001",
        title="Order Modification Window",
        rule="Changes allowed: Within 2 hours of order placement. After 2 hours: Order cannot be modified",
        conditions=["order_not_shipped"],
        precedence=3,
        category="Order Policy"
    ))
    
    # Holiday Policy (affects other policies)
    graph.add_clause(PolicyClause(
        clause_id="POL-HOLIDAY-001",
        title="Holiday Extension",
        rule="Holiday purchases: Return window extended to 45 days for purchases made November 1 - December 31",
        modifies=["POL-RETURN-001"],
        precedence=2,
        category="Special Conditions"
    ))
    
    # Communication Policy
    graph.add_clause(PolicyClause(
        clause_id="POL-COMM-001",
        title="Response Time",
        rule="Response time: Within 24 hours",
        precedence=5,
        category="Customer Service Standards"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-COMM-002",
        title="Escalation",
        rule="Escalation: Available for orders over $1000",
        conditions=["item_over_500"],  # Using similar condition
        precedence=4,
        category="Customer Service Standards"
    ))
    
    # Product Information Policy
    graph.add_clause(PolicyClause(
        clause_id="POL-INFO-001",
        title="Product Information Requests",
        rule="Product availability, specifications, and pricing information provided upon request. Stock status subject to real-time availability.",
        precedence=5,
        category="Product Information"
    ))
    
    return graph


@dataclass
class ScenarioTemplate:
    """Enhanced scenario template with policy graph integration"""
    scenario_id: str
    name: str
    description: str
    primary_policy: str  # Starting policy clause
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = "unknown"
    complexity_level: int = 1  # 1=simple, 2=moderate, 3=complex
    customer_situation: Dict[str, Any] = field(default_factory=dict)
    email_patterns: Dict[str, Any] = field(default_factory=dict)
    all_relevant_policies: List[str] = field(default_factory=list)  # Pre-validated policy interactions
    typical_days_after_order: Tuple[int, int] = (1, 30)  # Unused - timestamp now generated by analyzing email content


def generate_realistic_email_timestamp(order_date: str, email_content: Dict[str, str], 
                                     scenario: Dict, context: Dict[str, Any]) -> str:
    """Generate a realistic timestamp for when a customer would send an email
    
    Args:
        order_date: The order date in YYYY-MM-DD format
        email_content: The generated email with subject and body
        scenario: The scenario template with requirements
        context: Context dictionary with scenario details
        
    Returns:
        Timestamp string in ISO format
    """
    system_prompt = """You are an expert at analyzing customer emails and determining when they were likely sent.
    Based on the email content, order date, and scenario requirements, determine the most realistic date and time 
    for when this email would have been sent."""
    
    prompt = f"""Given the following information, determine when this customer email was most likely sent:

ORDER DATE: {order_date}

CUSTOMER EMAIL:
Subject: {email_content['subject']}
Body: {email_content['body']}

SCENARIO CONTEXT:
- Scenario Type: {scenario.get('name', 'unknown')}
- Description: {scenario.get('description', '')}
- Expected timing context: {json.dumps(context)}

IMPORTANT CONSIDERATIONS:
1. Look for time references in the email (e.g., "yesterday", "last week", "a few months ago")
2. The email timestamp MUST align with what the customer is saying:
   - If they say "I just ordered" → email within 1-3 days of order
   - If they say "last week" → email 5-10 days after order
   - If they say "a few months ago" → email 60-120 days after order
   - If they say "back in [month]" → email should be several months after that month
3. Consider the scenario type as secondary validation:
   - Cancellations: Usually within hours/days of order
   - Returns: Typically 1-4 weeks after receiving
   - Defects: Can be discovered anytime during use
   - Order status inquiries: Usually 3-10 days if not received
   - Warranty claims: Months after purchase
4. The email timestamp must be AFTER the order date
5. If there's a conflict between what the customer says and the scenario type, prioritize what the customer says

Generate a realistic date and time for when this email was sent. Consider:
- Business hours (9 AM - 6 PM) are more common but not exclusive
- Urgent issues might be sent outside business hours
- Match the urgency in the email tone

Return ONLY a JSON object like this example:
{{
    "email_sent_date": "2025-08-15",
    "email_sent_time": "14:23:45",
    "reasoning": "Customer says 'a few months ago' about March order, so email sent in August"
}}

The format must be:
- email_sent_date: YYYY-MM-DD format (e.g., 2025-08-15)
- email_sent_time: HH:MM:SS format in 24-hour time (e.g., 14:23:45)
- reasoning: One sentence explaining your choice"""

    response_text = call_llm(prompt, system_prompt)
    response = safe_json_parse(response_text, "object")
    
    if response and "email_sent_date" in response and "email_sent_time" in response:
        # Combine date and time into ISO format
        timestamp = f"{response['email_sent_date']}T{response['email_sent_time']}"
        return timestamp
    else:
        # Fallback to simple calculation if LLM fails
        print("Warning: LLM timestamp generation failed, using fallback")
        order_dt = datetime.datetime.strptime(order_date, "%Y-%m-%d")
        # Default to 7 days after order with random business hours
        email_dt = order_dt + datetime.timedelta(days=7, hours=random.randint(9, 17), 
                                               minutes=random.randint(0, 59), 
                                               seconds=random.randint(0, 59))
        return email_dt.isoformat()


def create_scenario_templates() -> Dict[str, List[ScenarioTemplate]]:
    """Create scenario templates organized by query type"""
    
    templates = {
    "return_request": [
            # Clear ACCEPT cases
            ScenarioTemplate(
                scenario_id="RETURN-001A",
                name="return_within_window_unopened",
                description="Customer wants to return unopened item within 30-day window",
                primary_policy="POL-RETURN-003",
                all_relevant_policies=["POL-RETURN-001", "POL-RETURN-003"],
                context_requirements={
                    "days_since_purchase": (7, 25),
                    "has_receipt": True,
                    "item_condition": "unopened"
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                "customer_expectation": "full_refund",
                "complication": "none"
            },
                email_patterns={
                    "should_mention": ["product_name", "return_reason", "unopened"],
                    "might_omit": ["exact_purchase_date"],
                    "tone_modifier": "polite"
                },
                typical_days_after_order=(7, 25)  # Email sent 7-25 days after order
            ),
            ScenarioTemplate(
                scenario_id="RETURN-001B",
                name="return_within_window_opened",
                description="Customer wants to return opened item within 30-day window",
                primary_policy="POL-RETURN-002",
                all_relevant_policies=["POL-RETURN-001", "POL-RETURN-002"],
                context_requirements={
                    "days_since_purchase": (10, 28),
                    "has_receipt": True,
                    "item_condition": "opened"
                },
                expected_outcome="approve_with_fee",
                complexity_level=2,
                customer_situation={
                "customer_expectation": "full_refund",
                    "complication": "unaware_of_restocking_fee"
            },
                email_patterns={
                    "should_mention": ["product_name", "used_briefly", "reason"],
                "might_omit": ["restocking_fee_awareness"],
                "tone_modifier": "hopeful"
            }
            ),
            # Clear DENY cases
            ScenarioTemplate(
                scenario_id="RETURN-002A",
                name="return_just_outside_window",
                description="Customer wants to return item just outside 30-day window",
                primary_policy="POL-RETURN-001",
                all_relevant_policies=["POL-RETURN-001"],
                context_requirements={
                    "days_since_purchase": (32, 40),
                    "has_receipt": True,
                    "item_condition": "unopened"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "full_refund",
                    "complication": "outside_window"
                },
                email_patterns={
                    "should_mention": ["busy_schedule", "product_name", "unopened"],
                    "might_omit": ["exact_days"],
                    "tone_modifier": "apologetic"
                }
            ),
            ScenarioTemplate(
                scenario_id="RETURN-002B",
                name="return_well_outside_window",
                description="Customer wants to return old purchase",
                primary_policy="POL-RETURN-001",
                all_relevant_policies=["POL-RETURN-001"],
                context_requirements={
                    "days_since_purchase": (60, 90),
                    "has_receipt": True,
                    "item_condition": "unopened"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "store_credit",
                    "complication": "outside_window"
                },
                email_patterns={
                    "should_mention": ["found_in_closet", "never_used", "product_name"],
                    "might_omit": ["purchase_date"],
                    "tone_modifier": "hopeful"
                }
            ),
            # Special cases
            ScenarioTemplate(
                scenario_id="RETURN-003",
                name="return_defective_within_window",
                description="Customer received defective item",
                primary_policy="POL-RETURN-004",
                all_relevant_policies=["POL-RETURN-004", "POL-WARRANTY-002"],
                context_requirements={
                    "days_since_purchase": (1, 20),
                    "has_receipt": True,
                    "item_condition": "defective"
                },
                expected_outcome="approve",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "full_refund_or_replacement",
                "complication": "item_defective"
            },
                email_patterns={
                    "should_mention": ["product_name", "defect_description", "stopped_working"],
                "might_omit": ["troubleshooting_attempted"],
                "tone_modifier": "frustrated"
            }
            ),
            ScenarioTemplate(
                scenario_id="RETURN-004",
                name="return_no_receipt",
                description="Customer wants to return item without receipt",
                primary_policy="POL-RETURN-001",
                all_relevant_policies=["POL-RETURN-001"],
                context_requirements={
                    "days_since_purchase": (10, 20),
                    "has_receipt": False,
                    "item_condition": "unopened"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                "customer_expectation": "store_credit",
                "complication": "no_receipt"
            },
                email_patterns={
                    "should_mention": ["lost_receipt", "product_name", "gift"],
                "might_omit": ["order_number", "purchase_date"],
                "tone_modifier": "apologetic"
            }
            ),
            ScenarioTemplate(
                scenario_id="RETURN-005",
                name="return_holiday_extended",
                description="Customer returning holiday purchase within extended window",
                primary_policy="POL-HOLIDAY-001",
                all_relevant_policies=["POL-HOLIDAY-001", "POL-RETURN-001", "POL-RETURN-003"],
                context_requirements={
                    "days_since_purchase": (35, 44),
                    "purchase_month": (11, 12),
                    "has_receipt": True,
                    "item_condition": "unopened"
                },
                expected_outcome="approve",
                complexity_level=3,
                customer_situation={
                    "customer_expectation": "full_refund",
                    "complication": "unsure_about_holiday_policy"
                },
                email_patterns={
                    "should_mention": ["holiday_gift", "product_name", "unopened"],
                    "might_omit": ["exact_policy_knowledge"],
                    "tone_modifier": "hopeful"
                }
            )
        ],
        "exchange_request": [
            ScenarioTemplate(
                scenario_id="EXCHANGE-001",
                name="exchange_within_window_size",
                description="Customer wants to exchange for different size",
                primary_policy="POL-EXCHANGE-001",
                all_relevant_policies=["POL-EXCHANGE-001", "POL-RETURN-001"],
                context_requirements={
                    "days_since_purchase": (5, 25),
                    "has_receipt": True,
                    "exchange_reason": "wrong_size"
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "direct_exchange",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["product_name", "size_needed", "exchange"],
                    "might_omit": ["return_label_request"],
                    "tone_modifier": "matter_of_fact"
                }
            ),
            ScenarioTemplate(
                scenario_id="EXCHANGE-002",
                name="exchange_defective_product",
                description="Customer wants to exchange defective item",
                primary_policy="POL-EXCHANGE-002",
                all_relevant_policies=["POL-EXCHANGE-002", "POL-RETURN-004"],
                context_requirements={
                    "days_since_purchase": (3, 15),
                    "has_receipt": True,
                    "item_condition": "defective",
                    "exchange_reason": "defective"
                },
                expected_outcome="approve",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "immediate_replacement",
                    "complication": "item_defective"
                },
                email_patterns={
                    "should_mention": ["product_name", "defect_description", "want_same_item"],
                    "might_omit": ["refund_preference"],
                    "tone_modifier": "frustrated"
                }
            ),
            ScenarioTemplate(
                scenario_id="EXCHANGE-003",
                name="exchange_for_upgrade",
                description="Customer wants to exchange for more expensive model",
                primary_policy="POL-EXCHANGE-001",
                all_relevant_policies=["POL-EXCHANGE-001", "POL-RETURN-001"],
                context_requirements={
                    "days_since_purchase": (7, 20),
                    "has_receipt": True,
                    "exchange_reason": "upgrade"
                },
                expected_outcome="approve_with_payment",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "pay_difference",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["current_product", "desired_product", "pay_difference"],
                    "might_omit": ["exact_price_difference"],
                    "tone_modifier": "inquiring"
                }
            )
    ],
    "shipping_issue": [
            ScenarioTemplate(
                scenario_id="SHIP-001",
                name="package_not_received_recent",
                description="Customer never received package marked as delivered",
                primary_policy="POL-SHIP-002",
                all_relevant_policies=["POL-SHIP-002", "POL-SHIP-003"],
                context_requirements={
                    "days_since_delivery": (1, 5),
                    "tracking_status": "delivered",
                    "item_value": (50, 800)
                },
                expected_outcome="investigate",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "immediate_replacement",
                "complication": "shows_delivered"
            },
                email_patterns={
                    "should_mention": ["tracking_shows_delivered", "checked_everywhere", "not_received"],
                    "might_omit": ["exact_delivery_time"],
                "tone_modifier": "worried"
            }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-002",
                name="wrong_item_received",
                description="Customer received different product than ordered",
                primary_policy="POL-SHIP-004",
                all_relevant_policies=["POL-SHIP-004"],
                context_requirements={
                    "days_since_delivery": (1, 200),  # No time limit for merchant error
                    "item_value": (20, 800)
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "correct_item_sent",
                    "complication": "merchant_error"
                },
                email_patterns={
                    "should_mention": ["ordered_product", "received_product", "order_number"],
                    "might_omit": ["want_to_keep_wrong_item"],
                    "tone_modifier": "confused"
                }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-003",
                name="damaged_high_value",
                description="High-value item arrived damaged",
                primary_policy="POL-SHIP-005",
                all_relevant_policies=["POL-SHIP-005", "POL-RETURN-004", "POL-COMM-002"],
                context_requirements={
                    "days_since_delivery": (1, 3),
                    "item_value": (501, 2000)
                },
                expected_outcome="conditional",
                complexity_level=3,
                customer_situation={
                    "customer_expectation": "immediate_replacement",
                    "complication": "high_value_damage"
                },
                email_patterns={
                    "should_mention": ["damage_description", "product_name", "expensive_item"],
                "might_omit": ["photos_attached"],
                "tone_modifier": "upset"
            }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-004",
                name="damaged_standard_value",
                description="Standard-value item arrived damaged",
                primary_policy="POL-SHIP-006",
                all_relevant_policies=["POL-SHIP-006", "POL-RETURN-004"],
                context_requirements={
                    "days_since_delivery": (1, 3),
                    "item_value": (10, 500)
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "replacement",
                    "complication": "shipping_damage"
                },
                email_patterns={
                    "should_mention": ["box_damaged", "product_damaged", "product_name"],
                    "might_omit": ["photos"],
                    "tone_modifier": "disappointed"
                }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-005",
                name="delayed_shipment",
                description="Order significantly delayed beyond promised date",
                primary_policy="POL-SHIP-001",
                all_relevant_policies=["POL-SHIP-001", "POL-COMM-001"],
                context_requirements={
                    "days_since_order": (10, 15),
                    "order_status": "shipped",
                    "promised_delivery_days": 7
                },
                expected_outcome="information",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "expedited_shipping_or_discount",
                    "complication": "delayed_delivery"
                },
                email_patterns={
                    "should_mention": ["order_date", "still_waiting", "promised_timeframe"],
                    "might_omit": ["tracking_number"],
                    "tone_modifier": "impatient"
                }
            )
        ],
        "warranty_claim": [
            ScenarioTemplate(
                scenario_id="WARRANTY-001",
                name="warranty_claim_valid",
                description="Product failed within warranty period",
                primary_policy="POL-WARRANTY-002",
                all_relevant_policies=["POL-WARRANTY-001", "POL-WARRANTY-002"],
                context_requirements={
                    "months_since_purchase": (3, 10),
                    "damage_type": "manufacturing"
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                "customer_expectation": "free_repair_or_replacement",
                "complication": "none"
            },
                email_patterns={
                    "should_mention": ["product_name", "stopped_working", "purchase_timeframe"],
                "might_omit": ["warranty_registration"],
                "tone_modifier": "matter_of_fact"
            }
            ),
            ScenarioTemplate(
                scenario_id="WARRANTY-002",
                name="water_damage_claim",
                description="Product damaged by liquid spill",
                primary_policy="POL-WARRANTY-003",
                all_relevant_policies=["POL-WARRANTY-003", "POL-WARRANTY-001"],
                context_requirements={
                    "months_since_purchase": (2, 8),
                    "damage_type": "water"
                },
                expected_outcome="deny",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "warranty_coverage",
                "complication": "water_damage"
            },
                email_patterns={
                    "should_mention": ["product_name", "accident", "still_under_warranty"],
                    "might_omit": ["liquid_spill_admission"],
                "tone_modifier": "hopeful"
            }
            ),
            ScenarioTemplate(
                scenario_id="WARRANTY-003",
                name="warranty_just_expired",
                description="Product failed just after warranty expired",
                primary_policy="POL-WARRANTY-001",
                all_relevant_policies=["POL-WARRANTY-001"],
                context_requirements={
                    "months_since_purchase": (13, 14),
                    "damage_type": "manufacturing"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "goodwill_repair",
                    "complication": "warranty_expired"
                },
                email_patterns={
                    "should_mention": ["just_over_year", "loyal_customer", "product_name"],
                    "might_omit": ["exact_warranty_period"],
                    "tone_modifier": "pleading"
                }
            )
    ],
    "pricing_dispute": [
            ScenarioTemplate(
                scenario_id="PRICE-001",
                name="price_match_authorized_retailer",
                description="Customer found lower price at Best Buy",
                primary_policy="POL-PRICE-001",
                all_relevant_policies=["POL-PRICE-001"],
                context_requirements={
                    "days_since_purchase": (3, 12),
                    "competitor_type": "authorized",
                    "competitor_name": "Best Buy"
                },
                expected_outcome="approve",
                complexity_level=2,
                customer_situation={
                "customer_expectation": "price_difference_refunded",
                    "complication": "none"
            },
                email_patterns={
                    "should_mention": ["Best Buy", "lower_price", "product_name", "price_difference"],
                "might_omit": ["exact_url"],
                "tone_modifier": "expecting_cooperation"
            }
            ),
            ScenarioTemplate(
                scenario_id="PRICE-002",
                name="price_match_marketplace_amazon",
                description="Customer found lower price on Amazon marketplace",
                primary_policy="POL-PRICE-002",
                all_relevant_policies=["POL-PRICE-002", "POL-PRICE-001"],
                context_requirements={
                    "days_since_purchase": (2, 10),
                    "competitor_type": "marketplace",
                    "competitor_name": "Amazon third-party seller"
                },
                expected_outcome="deny",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "price_match",
                    "complication": "marketplace_seller"
                },
                email_patterns={
                    "should_mention": ["Amazon", "much_cheaper", "product_name"],
                    "might_omit": ["third_party_seller", "marketplace"],
                    "tone_modifier": "disappointed"
                }
            ),
            ScenarioTemplate(
                scenario_id="PRICE-003",
                name="price_match_marketplace_ebay",
                description="Customer found lower price on eBay",
                primary_policy="POL-PRICE-002",
                all_relevant_policies=["POL-PRICE-002", "POL-PRICE-001"],
                context_requirements={
                    "days_since_purchase": (5, 13),
                    "competitor_type": "marketplace",
                    "competitor_name": "eBay seller"
                },
                expected_outcome="deny",
                complexity_level=2,
                customer_situation={
                    "customer_expectation": "price_match",
                    "complication": "marketplace_seller"
                },
                email_patterns={
                    "should_mention": ["eBay", "same_product", "lower_price"],
                    "might_omit": ["seller_type"],
                    "tone_modifier": "frustrated"
                }
            ),
            ScenarioTemplate(
                scenario_id="PRICE-004",
                name="price_match_too_late",
                description="Price match request after 14-day window",
                primary_policy="POL-PRICE-001",
                all_relevant_policies=["POL-PRICE-001"],
                context_requirements={
                    "days_since_purchase": (16, 25),
                    "competitor_type": "authorized",
                    "competitor_name": "Target"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                "customer_expectation": "price_adjustment",
                "complication": "outside_window"
            },
                email_patterns={
                    "should_mention": ["Target", "just_saw", "price_drop"],
                    "might_omit": ["exact_purchase_date"],
                "tone_modifier": "disappointed"
            }
            )
    ],
    "order_status": [
            ScenarioTemplate(
                scenario_id="ORDER-001",
                name="cancel_shipped_order",
                description="Customer wants to cancel already shipped order",
                primary_policy="POL-ORDER-001",
                all_relevant_policies=["POL-ORDER-001", "POL-RETURN-001"],
                context_requirements={
                    "days_since_order": (2, 4),
                    "order_status": "shipped"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                "customer_expectation": "order_cancelled",
                "complication": "already_shipped"
            },
                email_patterns={
                    "should_mention": ["cancel_order", "changed_mind", "order_number"],
                    "might_omit": ["shipping_notification"],
                "tone_modifier": "urgent"
            }
            ),
            ScenarioTemplate(
                scenario_id="ORDER-002",
                name="modify_order_too_late",
                description="Customer wants to add item to processing order",
                primary_policy="POL-ORDER-001",
                all_relevant_policies=["POL-ORDER-001"],
                context_requirements={
                    "hours_since_order": (3, 8),
                    "order_status": "processing"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "add_item",
                    "complication": "past_modification_window"
                },
                email_patterns={
                    "should_mention": ["forgot_to_add", "same_order", "additional_item"],
                    "might_omit": ["time_placed"],
                "tone_modifier": "hopeful"
            }
            ),
            ScenarioTemplate(
                scenario_id="ORDER-003",
                name="order_status_inquiry",
                description="Customer checking on order status",
                primary_policy="POL-COMM-001",
                all_relevant_policies=["POL-COMM-001", "POL-SHIP-001"],
                context_requirements={
                    "days_since_order": (3, 5),
                    "order_status": "processing"
                },
                expected_outcome="information",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "status_update",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["order_number", "when_ship", "order_status"],
                    "might_omit": ["tracking_request"],
                    "tone_modifier": "curious"
                }
            )
    ],
    "general_inquiry": [
            ScenarioTemplate(
                scenario_id="GENERAL-001",
                name="product_availability",
                description="Customer asking about availability or stock status of 1-2 specific products by name (not general inquiries about 'all products')",
                primary_policy="POL-INFO-001",
                all_relevant_policies=["POL-INFO-001", "POL-COMM-001"],
                context_requirements={},
                expected_outcome="information",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "stock_information",
                "complication": "none"
            },
                email_patterns={
                    "should_mention": ["specific_product_name", "in_stock", "when_available"],
                "might_omit": ["exact_model_number"],
                "tone_modifier": "curious"
            }
            ),
            ScenarioTemplate(
                scenario_id="GENERAL-002",
                name="product_comparison",
                description="Customer asking for comparison between 2 specific products in the company's catalog. Customer asks about warranty/specs for the products and mentions specific product names",
                primary_policy="POL-INFO-001",
                all_relevant_policies=["POL-INFO-001", "POL-COMM-001"],
                context_requirements={},
                expected_outcome="information",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "product_details",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["specific_product_names", "differences", "which_is_better"],
                    "might_omit": ["budget"],
                    "tone_modifier": "inquiring"
                }
            ),
            ScenarioTemplate(
                scenario_id="GENERAL-003",
                name="shipping_cost_inquiry",
                description="Customer asking about shipping costs and options",
                primary_policy="POL-SHIP-001",
                all_relevant_policies=["POL-SHIP-001", "POL-COMM-001"],
                context_requirements={},
                expected_outcome="information",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "shipping_information",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["shipping_cost", "delivery_time", "shipping_options"],
                    "might_omit": ["specific_items"],
                    "tone_modifier": "curious"
                }
            )
        ]
    }
    
    return templates 

def validate_scenario_templates(scenario_templates: Dict[str, List[ScenarioTemplate]], 
                               policy_graph: PolicyGraph) -> Dict[str, List[ScenarioTemplate]]:
    """Validate and enhance scenario templates by checking against policy groups"""
    
    print("\n=== Validating Scenario Templates Against Policy Groups ===")
    
    # Define policy groups for validation
    policy_groups = {
        "Return Policies": ["POL-RETURN-001", "POL-RETURN-002", "POL-RETURN-003", "POL-RETURN-004"],
        "Shipping Policies": ["POL-SHIP-001", "POL-SHIP-002", "POL-SHIP-003", "POL-SHIP-004", "POL-SHIP-005", "POL-SHIP-006"],
        "Warranty Policies": ["POL-WARRANTY-001", "POL-WARRANTY-002", "POL-WARRANTY-003"],
        "Price Match Policies": ["POL-PRICE-001", "POL-PRICE-002"],
        "Order Policies": ["POL-ORDER-001"],
        "Special Conditions": ["POL-HOLIDAY-001"],
        "Service Standards": ["POL-COMM-001", "POL-COMM-002"],
        "Information Policies": ["POL-INFO-001"]
    }
    
    # Track validation results
    validation_results = {}
    
    # Process each scenario template
    for query_type, templates in scenario_templates.items():
        for template in templates:
            print(f"\nValidating: {template.scenario_id} - {template.name}")
            
            # Collect all relevant policies for this scenario
            all_relevant_policies = [template.primary_policy]
            interaction_notes = {}
            
            # Check each policy group
            for group_name, group_policies in policy_groups.items():
                # Skip if primary policy is already in this group
                if template.primary_policy in group_policies:
                    print(f"  ✓ {group_name}: Primary policy in group")
                    continue
                
                # Check if any policies in this group apply
                group_result = check_policy_group_relevance(
                    template, group_name, group_policies, policy_graph
                )
                
                if group_result.get("applies", False):
                    relevant_policies = group_result.get("relevant_policies", [])
                    reason = group_result.get("reason", "")
                    
                    print(f"  ✓ {group_name}: {', '.join(relevant_policies)}")
                    print(f"    Reason: {reason}")
                    
                    all_relevant_policies.extend(relevant_policies)
                    for policy in relevant_policies:
                        interaction_notes[policy] = reason
                else:
                    print(f"  - {group_name}: Not applicable")
            
            # Update template with discovered interactions
            validation_results[template.scenario_id] = {
                "original_primary": template.primary_policy,
                "all_relevant_policies": list(set(all_relevant_policies)),  # Remove duplicates
                "interaction_notes": interaction_notes,
                "validation_status": "validated"
            }
            
            # Update the template's internal tracking (for generation use)
            if hasattr(template, '_all_relevant_policies'):
                template._all_relevant_policies = list(set(all_relevant_policies))
            
            print(f"  Total relevant policies: {len(set(all_relevant_policies))}")
    
    # Save validation results
    save_validation_results(validation_results)
    
    return scenario_templates

def check_policy_group_relevance(template: ScenarioTemplate, group_name: str, 
                                group_policies: List[str], policy_graph: PolicyGraph) -> Dict:
    """Check if any policies in a group are relevant to a scenario"""
    
    # Build context description
    context_parts = []
    if template.context_requirements:
        for key, value in template.context_requirements.items():
            if isinstance(value, tuple):
                context_parts.append(f"{key}: {value[0]}-{value[1]}")
            else:
                context_parts.append(f"{key}: {value}")
    
    context_description = ", ".join(context_parts) if context_parts else "No specific context requirements"
    
    # Get policy details for the group
    policy_details = []
    for policy_id in group_policies:
        if policy_id in policy_graph.clauses:
            clause = policy_graph.clauses[policy_id]
            policy_details.append(f"- [{policy_id}] {clause.title}: {clause.rule}")
    
    policy_list = "\n".join(policy_details)
    
    # Create focused prompt
    system_prompt = """You are a customer service policy expert. Your job is to identify ONLY obvious, 
    direct policy interactions - not theoretical or edge cases."""
    
    prompt = f"""Given this customer service scenario:

SCENARIO: {template.description}
CONTEXT: {context_description}
EXPECTED OUTCOME: {template.expected_outcome}
PRIMARY POLICY: {template.primary_policy}

Looking ONLY at these {group_name}:
{policy_list}

Do any of these policies OBVIOUSLY apply to this scenario in a way that would affect the resolution?

Consider only:
1. Clear, direct interactions that a customer service rep would immediately recognize
2. Policies that would change or add to the resolution actions
3. Conditions that are explicitly met by the scenario context

Do NOT consider:
- Theoretical edge cases
- Indirect connections through other policies
- General policies that apply to everything (unless they add specific actions)

Respond in JSON format:
{{
    "applies": true/false,
    "relevant_policies": ["POL-XXX-###", ...],  // Only policies that OBVIOUSLY apply
    "reason": "Brief explanation of why these policies clearly apply to this specific scenario"
}}"""
    
    try:
        response = call_llm(prompt, system_prompt)
        result = safe_json_parse(response, "object")
        
        # Validate the response
        if result and isinstance(result, dict):
            # Ensure we only include policies that actually exist
            if "relevant_policies" in result:
                result["relevant_policies"] = [
                    p for p in result["relevant_policies"] 
                    if p in group_policies
                ]
            return result
        else:
            return {"applies": False, "relevant_policies": [], "reason": "Failed to parse response"}
            
    except Exception as e:
        print(f"    Error checking {group_name}: {e}")
        return {"applies": False, "relevant_policies": [], "reason": f"Error: {str(e)}"}

def save_validation_results(results: Dict):
    """Save validation results for analysis"""
    import os
    
    output_path = os.path.join("./assets", "scenario_validation_results.json")
    
    # Add metadata
    output = {
        "metadata": {
            "validated_at": datetime.datetime.now().isoformat(),
            "total_scenarios": len(results),
            "description": "Validation results showing which policies interact with each scenario"
        },
        "validation_results": results,
        "summary": {
            "scenarios_with_single_policy": sum(1 for r in results.values() if len(r["all_relevant_policies"]) == 1),
            "scenarios_with_multiple_policies": sum(1 for r in results.values() if len(r["all_relevant_policies"]) > 1),
            "average_policies_per_scenario": sum(len(r["all_relevant_policies"]) for r in results.values()) / len(results) if results else 0
        }
    }
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nValidation results saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save validation results: {e}")

# Scenario dimension probabilities for coverage
SCENARIO_DIMENSIONS = {
    "query_type": {
        "return_request": 0.25,
        "exchange_request": 0.15,
        "shipping_issue": 0.20,
        "order_status": 0.10,
        "pricing_dispute": 0.10,
        "general_inquiry": 0.10,
        "warranty_claim": 0.10
    },
    
    "information_completeness": {
        "complete": 0.30,
        "missing_order_number": 0.30,
        "wrong_email": 0.10,
        "minimal_details": 0.20,
        "incorrect_info": 0.10
    },
    
    "complexity": {
        "straightforward": 0.40,
        "requires_lookup": 0.35,
        "edge_case": 0.20,
        "requires_escalation": 0.05
    },
    
    "customer_sentiment": {
        "neutral": 0.60,
        "frustrated": 0.20,
        "angry": 0.10,
        "confused": 0.10
    }
}

# Standard resolution action types - including denials
RESOLUTION_ACTIONS = [
    "process_return",
    "issue_refund", 
    "send_replacement",
    "provide_tracking",
    "honor_warranty",
    "escalate_to_manager",
    "request_more_info",
    "request_photo",
    "update_shipping_address",
    "cancel_order",
    "deny_return",
    "deny_refund",
    "deny_warranty_claim",
    "deny_price_match",
    "deny_order_modification",
    "deny_cancellation",
    "provide_information",
    "initiate_investigation",
    "process_exchange",
    "deny_exchange",
    "send_return_label"
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
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                seed=42,
                thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
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


def generate_company_policy_from_graph(config: DatasetConfig, policy_graph: PolicyGraph) -> str:
    """Generate policy document from policy graph (without metadata for ML training)."""
    return policy_graph.generate_policy_text()


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
        print("ERROR: Failed to generate products")
        return []
    
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
    - created_date (distribute over past {config.customer_history_days} days)

Include some quirks:
- 5% have slight typos in alternate emails
- Some customers have maiden/married name variations
- Mix of gmail, yahoo, outlook, and custom domains

Format as JSON array.

IMPORTANT: Return ONLY the JSON array, no explanatory text before or after."""
    
    customers_text = call_llm(prompt, system_prompt)
    customers = safe_json_parse(customers_text, "array")
    
    if not customers:
        print("ERROR: Failed to generate customers")
        return []
    
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
    else:
        print(f"ERROR: Failed to generate order for customer {customer['customer_id']}")
        return None


def generate_orders(config: DatasetConfig, customers: List[Dict], products: List[Dict]) -> List[Dict]:
    """Generate order history with consistent customer and product references."""
    
    print(f"  Generating {config.num_orders} orders...")
    orders = []
    
    # Create a date range for orders
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=config.order_history_days)
    
    # Simple customer distribution - each customer gets roughly equal orders
    for i in range(config.num_orders):
        # Progress indicator
        if i % 10 == 0:
            print(f"    Generated {i}/{config.num_orders} orders...")
        
        # Random date
        days_ago = random.randint(0, config.order_history_days)
        order_date = (end_date - datetime.timedelta(days=days_ago))
        order_date_str = order_date.strftime('%Y-%m-%d')
        
        # Select customer (round-robin with some randomness)
        customer = customers[i % len(customers)]
        
        # Select products (1-3 items per order, occasionally more)
        num_items = random.choices([1, 2, 3, 4, 5], weights=[0.5, 0.3, 0.15, 0.04, 0.01])[0]
        selected_products = random.sample(products, min(num_items, len(products)))
        
        # Generate order
        order = generate_single_order(customer, selected_products, order_date_str, i + 1001)
        
        if order:
            orders.append(order)
    
    # Add some returns/refunds to random orders
    num_returns = int(len(orders) * config.return_rate)
    if num_returns > 0 and orders:
        orders_with_returns = random.sample(range(len(orders)), min(num_returns, len(orders)))
        
        for idx in orders_with_returns:
            if orders[idx]["items"]:
                # Mark random item as returned
                item_idx = random.randint(0, len(orders[idx]["items"]) - 1)
                orders[idx]["items"][item_idx]["item_status"] = random.choice(["returned", "refunded"])
                orders[idx]["order_status"] = "partially_returned"
    
    print(f"  ✓ Generated {len(orders)} orders")
    return orders


def select_and_customize_scenario(policy_graph: PolicyGraph, scenario_templates: Dict[str, List[ScenarioTemplate]], 
                                 query_type: str, order: Dict, customer: Dict, products: List[Dict]) -> Dict:
    """Select and customize a scenario template with pre-validated policy interactions."""
    
    # Get templates for this query type
    templates = scenario_templates.get(query_type, [])
    if not templates:
        # Fallback if query type not found
        templates = scenario_templates["return_request"]
    
    # Select a random template
    template = random.choice(templates)
    
    # Calculate order context
    context = build_order_context(order, customer, products)
    
    # Override context with template requirements
    if template.context_requirements:
        for key, value in template.context_requirements.items():
            if isinstance(value, tuple):
                # For ranges, pick a random value within the range
                context[key] = random.randint(value[0], value[1])
            else:
                context[key] = value
        
        # Recalculate months_since_purchase if days_since_purchase was set
        if "days_since_purchase" in context:
            context["months_since_purchase"] = context["days_since_purchase"] / 30.44
        
        # Set purchase_month if not already set and we have order date
        if order and "purchase_month" not in context:
            order_date = datetime.datetime.strptime(order["order_date"], "%Y-%m-%d")
            context["purchase_month"] = order_date.month
    
    # Use pre-validated policies from template if available
    if template.all_relevant_policies:
        all_relevant_policies = template.all_relevant_policies
    else:
        # Fallback to just primary policy
        all_relevant_policies = [template.primary_policy]
    
    # Filter policies based on context
    applicable_policies = policy_graph.resolve_conflicts(all_relevant_policies, context)
    
    # Customize scenario based on actual order data and policy interactions
    scenario = {
        "scenario_id": template.scenario_id,
        "name": template.name,
        "description": template.description,
        "primary_policy": template.primary_policy,
        "all_relevant_policies": all_relevant_policies,  # For generation use
        "applicable_policies": applicable_policies,      # After conflict resolution
        "expected_outcome": template.expected_outcome,
        "complexity_level": template.complexity_level,
        "query_type": query_type,
        "order": order,
        "customer": customer,
        "products": products,
        "context": context,
        "customer_situation": template.customer_situation.copy(),
        "email_patterns": template.email_patterns.copy(),
        "_template_context_requirements": template.context_requirements  # Hidden for ML
    }
    
    return scenario

def build_order_context(order: Dict, customer: Dict, products: List[Dict]) -> Dict[str, Any]:
    """Build context dictionary for policy evaluation."""
    context = {
        "has_receipt": True,  # Default assumption for database orders
        "customer_tier": "standard"
    }
    
    if order:
        # Note: days_since_purchase will be overridden by template requirements
        # We just set defaults here for completeness
        context.update({
            "days_since_purchase": 10,  # Default, will be overridden
            "months_since_purchase": 0.33,  # Default, will be overridden
            "order_status": order.get("order_status", "delivered"),
            "total_order_value": order.get("total_amount", 0)
        })
        
        # Calculate order value context
        if order.get("total_amount", 0) > 500:
            context["item_over_500"] = True
        
        # Check if it's a holiday purchase (Nov-Dec)
        order_date = datetime.datetime.strptime(order["order_date"], "%Y-%m-%d")
        purchase_month = order_date.month
        if purchase_month in [11, 12]:
            context["purchase_month"] = purchase_month
        
        # Product-specific context
        if order.get("items") and products:
            max_item_value = 0
            for item in order["items"]:
                item_value = item.get("price_paid", 0) * item.get("quantity", 1)
                max_item_value = max(max_item_value, item_value)
            context["item_value"] = max_item_value
    
    return context


def generate_customer_email_v3(scenario: Dict, dimensions: Dict[str, str]) -> Dict:
    """Generate an email FROM a customer TO customer support.
    
    This simulates the initial incoming ticket - a customer writing to support 
    about their issue. The customer does NOT have access to internal policies
    or systems, they're just describing their problem from their perspective.
    """
    
    system_prompt = """You are simulating a customer writing an email TO customer support. 
    You are the customer who needs help, NOT the customer service representative.
    Write from the customer's perspective as someone contacting support for assistance.
    Use ONLY the provided order, customer, and product information. Do not invent any details."""
    
    order = scenario.get("order")
    customer = scenario["customer"]
    products = scenario.get("products", [])
    context = scenario.get("context", {})
    
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
        days_since_order = context.get("days_since_purchase", 0)
        order_info = f"""ORDER INFORMATION (use exactly):
- Order ID: {order['order_id']}
- Order Date: {order['order_date']} ({days_since_order} days ago)
- Products: {json.dumps(product_details)}
- Total: ${order['total_amount']}
- Shipping Method: {order['shipping_method']}
- Tracking: {order['tracking_number']}
- Status: {order['order_status']}"""
    else:
        # For general inquiries, provide a random sample of products to ask about
        available_products = []
        sample_products = random.sample(products, min(10, len(products)))  # Max 10 products
        for product in sample_products:
            available_products.append({
                "product_id": product["product_id"],
                "name": product["name"],
                "price": product["base_price"],
                "warranty_period": product["warranty_period"],
                "in_stock": product["in_stock"],
                "category": product.get("category", "Unknown")
            })
        
        order_info = f"""ORDER INFORMATION: No specific order (general inquiry)

AVAILABLE PRODUCTS (pick 1-2 specific products to ask about):
{json.dumps(available_products, indent=2)}

IMPORTANT: For general inquiries, ask about SPECIFIC products by name, not general comparisons of "all products"."""
    
    # Add scenario complexity and context hints
    complexity_note = ""
    if scenario.get("complexity_level", 1) > 1:
        complexity_note = f"\nComplexity Level: {scenario['complexity_level']} (customer may not understand all policy nuances)"
    
    prompt = f"""Write an email FROM a customer TO customer support based on this EXACT information:

SCENARIO: {scenario['description']}
Query Type: {scenario['query_type']}
Primary Policy Area: {scenario.get('primary_policy', 'Unknown')}
{complexity_note}

CUSTOMER INFORMATION (use exactly):
- Name: {customer['name']}
- Email: {customer['primary_email']}

{order_info}

CONTEXT CLUES (customer probably doesn't know these policy details):
- Days since purchase: {context.get('days_since_purchase', 'N/A')} (This affects your time references in the email)
- Item value: ${context.get('item_value', 'N/A')}
- Order status: {context.get('order_status', 'N/A')}
- Has receipt: {context.get('has_receipt', True)}

SCENARIO REQUIREMENTS:
- Scenario name: {scenario.get('name', 'unknown')}
- Description: {scenario['description']}
- Expected outcome: {scenario.get('expected_outcome', 'unknown')}

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

TIMING-APPROPRIATE LANGUAGE:
Based on "Days since purchase: {context.get('days_since_purchase', 'N/A')}", use appropriate language:
- If 0-2 days: Use phrases like "I just ordered", "yesterday", "earlier today"
- If 3-7 days: Use phrases like "last week", "a few days ago", "recently ordered"
- If 8-30 days: Use phrases like "I ordered [product] on [date]", "a couple weeks ago"
- If 31-60 days: Use phrases like "last month", "about a month ago"
- If 60+ days: Use specific dates or phrases like "back in [month]", "a few months ago"
- Match your urgency to the timing (immediate issues = more urgent tone)
- IMPORTANT: Your time references MUST match the days_since_purchase value!

Write a natural email from the CUSTOMER'S perspective. Remember:
- You are the customer who has the problem described in: {scenario['description']}
- You are writing TO customer support asking for help
- You do NOT have access to internal policies, systems, or detailed company procedures
- You only know what a typical customer would know about their own order
- For general inquiries: Ask about SPECIFIC products by name, not vague "all products" questions

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
        print(f"ERROR: Failed to generate email for scenario {scenario['name']}")
        return None
    
    return email


def generate_resolution_v3(email: Dict, scenario: Dict, policy_graph: PolicyGraph, dimensions: Dict[str, str]) -> Dict:
    """Generate a resolution plan FROM a customer service representative.
    
    This simulates what happens AFTER receiving the customer's email:
    - The CSR looks up customer information in internal systems
    - Reviews company policies and their interactions
    - Creates an action plan to resolve the customer's issue
    - Has full access to internal data, policies, and procedures
    """
    
    system_prompt = """You are an expert customer service professional. Create resolutions using 
    the provided information and policies. Follow policies exactly as written. When in doubt, 
    check the complete policy document to ensure nothing is missed. 
    In some cases a request will be straightforward and directly addressed in the company policy document by a single policy.
    In other cases a request may involve cross-referencing multiple policies and their interactions. DENY requests that violate policy.
    Your role is to apply company policy fairly and consistently while being helpful to customers."""
    
    order = scenario.get("order")
    customer = scenario["customer"]
    products = scenario.get("products", [])
    context = scenario.get("context", {})
    
    # Get applicable policies from the scenario (already resolved by policy graph)
    applicable_policies = scenario.get("applicable_policies", [scenario.get("primary_policy")])
    
    # Build policy text for the primary policies we think apply
    policy_text_sections = []
    for policy_id in applicable_policies:
        if policy_id in policy_graph.clauses:
            clause = policy_graph.clauses[policy_id]
            policy_text_sections.append(f"[{policy_id}] {clause.title}\nRule: {clause.rule}")
            if clause.conditions:
                policy_text_sections.append(f"Conditions: {', '.join(clause.conditions)}")
    
    primary_policy_text = "\n\n".join(policy_text_sections)
    
    # Get the complete policy document
    complete_policy_document = policy_graph.generate_policy_text()
    
    # Build order information section with product values
    product_values = {}
    if order:
        for item in order['items']:
            product_values[item['product_id']] = {
                'price_paid': item['price_paid'],
                'quantity': item['quantity'],
                'total_value': item['price_paid'] * item['quantity']
            }
        
        order_info = f"""VERIFIED ORDER INFORMATION:
- Order ID: {order['order_id']}
- Order Date: {order['order_date']}
- Days Since Purchase: {context.get('days_since_purchase', 'N/A')}
- Months Since Purchase: {context.get('months_since_purchase', 'N/A'):.1f}
- Items with values: {json.dumps(product_values, indent=2)}
- Total Order Value: ${order['total_amount']}
- Order Status: {order['order_status']}

PRODUCTS IN ORDER WITH PRICES:
{json.dumps(products, indent=2)}"""
    else:
        order_info = "VERIFIED ORDER INFORMATION: No specific order (general inquiry)"
    
    # Build detailed context section
    context_info = f"""POLICY EVALUATION CONTEXT:
- Days since purchase: {context.get('days_since_purchase', 'N/A')}
- Months since purchase: {context.get('months_since_purchase', 'N/A')}
- Item value: ${context.get('item_value', 'N/A')}
- Order status: {context.get('order_status', 'N/A')}
- Has receipt: {context.get('has_receipt', True)}
- Item over $500: {context.get('item_over_500', False)}
- Purchase month: {context.get('purchase_month', 'N/A')}"""
    
    prompt = f"""Create a professional resolution for this customer support case:

EMAIL FROM CUSTOMER:
{json.dumps(email, indent=2)}

VERIFIED CUSTOMER INFORMATION:
- Customer ID: {customer['customer_id']}
- Name: {customer['name']}
- Email: {customer['primary_email']}

{order_info}

{context_info}

SCENARIO DETAILS:
- Issue Type: {scenario['description']}
- Primary Policy: {scenario.get('primary_policy', 'Unknown')}
- Complexity Level: {scenario.get('complexity_level', 1)}
- Customer Expectation: {scenario['customer_situation']['customer_expectation']}
- Complication: {scenario['customer_situation']['complication']}
- Expected Outcome: {scenario.get('expected_outcome', 'unknown')}

PRIMARY POLICIES (We believe these are most relevant to this case):
{primary_policy_text}

COMPLETE COMPANY POLICY DOCUMENT:
{complete_policy_document}

RESOLUTION GUIDANCE:
1. Start with the PRIMARY POLICIES listed above - these should handle most cases
2. ALWAYS cross-reference the COMPLETE POLICY DOCUMENT to ensure nothing was missed
3. Look for edge cases, exceptions, or additional policies that might apply
4. Base ALL decisions on actual policy text, never make assumptions

COMMON PATTERNS TO WATCH FOR:
- Wrong item shipped = Merchant error (check POL-SHIP-004 - may have NO time limit)
- Exchanges vs Returns = Different policies (POL-EXCHANGE-XXX vs POL-RETURN-XXX)
- Defective items = Often override normal restrictions and fees
- Holiday purchases = May have extended return windows (check POL-HOLIDAY-001)
- High-value items = May require additional verification (photos, escalation)
- Time limits = Read carefully - some policies explicitly state "no time limit"
- Receipt requirements = Some situations may waive this requirement
- Precedence = When policies conflict, check which takes priority
- Customer asking for exchange = Don't force into return category, check exchange policies

RESOLUTION REQUIREMENTS:
1. Include order_id and order_date when applicable
2. Cite specific policy clauses (by POL-XXX-### ID) for every decision
3. If denying a request, explain exactly which policy prevents approval
4. Ensure monetary values match actual product prices from the order
5. Consider ALL relevant policies, not just the obvious ones

Create a resolution with this structure:
{{
    "order_id": "{order.get('order_id', 'N/A') if order else 'N/A'}",
    "order_date": "{order.get('order_date', 'N/A') if order else 'N/A'}",
    "customer_lookup": {{
        "status": "found",
        "customer_id": "{customer['customer_id']}",
        "lookup_method": "email_match",
        "notes": "Customer found in database"
    }},
    "policy_references": ["list of all relevant policy IDs including any you discover"],
    "policy_reasoning": "Explain which policies apply and how they interact",
    "actions": [
        {{
            "type": "action from {RESOLUTION_ACTIONS}",
            "reason": "Detailed reason citing specific policies by tag",
            "value": exact dollar amount from product prices if refund/replacement (0 for denials),
            "details": "Specific implementation details"
        }}
    ],
    "escalation_required": boolean,
    "escalation_reason": "Why escalation needed" or null,
    "priority": "low/medium/high/urgent",
    "total_resolution_value": sum of all monetary values in actions
}}

VERIFICATION CHECKLIST:
- Did you check the COMPLETE policy document for any policies we might have missed?
- Are you citing the actual policy text, not paraphrasing?
- For denials, is the specific policy violation clearly stated?
- Have you considered if this is a special case (merchant error, defective, holiday)?
- Are all monetary values taken from the actual order data?

Return ONLY the JSON object."""
    
    resolution_text = call_llm(prompt, system_prompt)
    resolution = safe_json_parse(resolution_text, "object")
    
    # Enhanced validation using policy graph
    if resolution and isinstance(resolution, dict):
        # Ensure correct order and customer info
        if order:
            resolution["order_id"] = order["order_id"]
            resolution["order_date"] = order["order_date"]
        else:
            resolution["order_id"] = "N/A"
            resolution["order_date"] = "N/A"
            
        if "customer_lookup" in resolution:
            resolution["customer_lookup"]["customer_id"] = customer["customer_id"]
        
        # Keep the LLM's policy references - it may have found additional relevant policies
        # Only override if it's missing or empty
        if not resolution.get("policy_references"):
            resolution["policy_references"] = applicable_policies
        
        # Trust the LLM's policy-based decisions rather than hard-coding rules
        # The LLM has access to the full policy document and should make correct decisions
    else:
        print(f"ERROR: Failed to generate resolution")
        return None
    
    return resolution


def create_complete_ticket_v3(config: DatasetConfig, scenario: Dict, email: Dict, 
                             resolution: Dict, dimensions: Dict[str, str]) -> Dict:
    """Combine all elements into a complete ticket with enhanced policy traceability."""
    
    # Use pre-calculated email timestamp from scenario
    email_timestamp = scenario.get("_email_timestamp", datetime.datetime.now().isoformat())
    
    # Generate ticket ID based on the email timestamp
    email_dt = datetime.datetime.fromisoformat(email_timestamp)
    ticket_id_date = email_dt.strftime('%Y%m%d')
    
    ticket = {
        "ticket_id": f"TK-{ticket_id_date}-{random.randint(1000, 9999)}",
        "customer_email": email["from_email"],
        "subject": email["subject"],
        "body": email["body"],
        "timestamp": email_timestamp,
        "customer_id": scenario["customer"]["customer_id"],
        "order_id": scenario["order"]["order_id"] if scenario.get("order") else "N/A",
        "resolution_plan": resolution
    }
    
    # Add debug info if configured (this will be stripped for ML training)
    if config.include_debug_info:
        ticket["_scenario_dimensions"] = dimensions
        ticket["_scenario_template"] = {
            "scenario_id": scenario["scenario_id"],
            "name": scenario["name"],
            "primary_policy": scenario.get("primary_policy"),
            "complexity_level": scenario.get("complexity_level"),
            "expected_outcome": scenario.get("expected_outcome", "unknown")
        }
        # Use regular context for debug info
        
        ticket["_policy_analysis"] = {
            "all_relevant_policies": scenario.get("all_relevant_policies", []),
            "applicable_policies": scenario.get("applicable_policies", []),
            "context_used": scenario.get("context", {}),
            "policy_interactions": "Multi-hop reasoning required" if len(scenario.get("all_relevant_policies", [])) > 1 else "Single policy"
        }
    
    return ticket

def strip_debug_metadata(ticket: Dict) -> Dict:
    """Remove debug metadata to create clean training data."""
    clean_ticket = ticket.copy()
    keys_to_remove = [
        "_scenario_dimensions", 
        "_scenario_template", 
        "_policy_analysis",
        "_template_context_requirements"
    ]
    for key in keys_to_remove:
        clean_ticket.pop(key, None)
    return clean_ticket


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

def save_policy_graph(policy_graph: PolicyGraph, config: DatasetConfig):
    """Save policy graph structure for analysis."""
    
    # Convert policy graph to serializable format
    graph_data = {
        "metadata": {
            "total_clauses": len(policy_graph.clauses),
            "generated_at": datetime.datetime.now().isoformat(),
            "description": "Policy graph showing clause interactions for multi-hop reasoning"
        },
        "clauses": {},
        "interaction_summary": {},
        "complexity_analysis": {}
    }
    
    # Serialize all policy clauses
    for clause_id, clause in policy_graph.clauses.items():
        graph_data["clauses"][clause_id] = {
            "title": clause.title,
            "rule": clause.rule,
            "category": clause.category,
            "precedence": clause.precedence,
            "conditions": clause.conditions,
            "interactions": {
                "interacts_with": clause.interacts_with,
                "modifies": clause.modifies,
                "modified_by": clause.modified_by,
                "overrides": clause.overrides,
                "overridden_by": clause.overridden_by,
                "requires": clause.requires
            },
            "total_connections": len(clause.interacts_with + clause.modifies + clause.modified_by + 
                                   clause.overrides + clause.overridden_by + clause.requires)
        }
    
    # Generate interaction summary
    for clause_id in policy_graph.clauses.keys():
        related = policy_graph.get_related_policies(clause_id, max_hops=3)
        graph_data["interaction_summary"][clause_id] = {
            "direct_connections": len(policy_graph.interaction_graph.get(clause_id, [])),
            "reachable_within_3_hops": len(related),
            "related_policies": related[:5]  # Top 5 for readability
        }
    
    # Complexity analysis
    precedence_groups = {}
    category_counts = {}
    for clause in policy_graph.clauses.values():
        # Group by precedence
        prec = clause.precedence
        if prec not in precedence_groups:
            precedence_groups[prec] = []
        precedence_groups[prec].append(clause.clause_id)
        
        # Count by category
        cat = clause.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    graph_data["complexity_analysis"] = {
        "precedence_groups": precedence_groups,
        "category_distribution": category_counts,
        "total_unique_interactions": len(set().union(*[
            clause.interacts_with + clause.modifies + clause.modified_by + 
            clause.overrides + clause.overridden_by + clause.requires
            for clause in policy_graph.clauses.values()
        ])),
        "max_precedence": max(c.precedence for c in policy_graph.clauses.values()),
        "min_precedence": min(c.precedence for c in policy_graph.clauses.values())
    }
    
    # Save to file
    graph_path = config.get_filepath("policy_graph.json")
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"- Policy graph structure in policy_graph.json")
    return graph_data


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
            
            # Recreate policy graph and scenario templates for consistency
            policy_graph = create_policy_graph(config)
            scenario_templates = create_scenario_templates()
            print(f"Recreated policy graph and scenario templates")
        except FileNotFoundError as e:
            print(f"Error: Could not find existing data files. Please run in 'create' mode first.")
            print(f"Missing file: {e}")
            return
    else:
        # Create mode - generate everything from scratch
        existing_tickets = []
        
        # Phase 1: Company Foundation
        print("\nPhase 1: Generating company foundation...")
        print("- Creating policy graph...")
        policy_graph = create_policy_graph(config)
        print(f"  ✓ Created policy graph with {len(policy_graph.clauses)} policy clauses")
        
        print("- Generating policy document from graph...")
        policy = generate_company_policy_from_graph(config, policy_graph)
        
        print("- Creating scenario templates...")
        scenario_templates = create_scenario_templates()
        template_count = sum(len(templates) for templates in scenario_templates.values())
        print(f"  ✓ Created {template_count} scenario templates across {len(scenario_templates)} categories")
        
        print(f"- Generating {config.num_products} products...")
        products = generate_product_catalog(config)
        if not products:
            print("ERROR: Cannot proceed without products")
            return
        
        # Phase 2: Customer & Order Generation
        print("\nPhase 2: Generating customers and orders...")
        print(f"- Generating {config.num_customers} customers...")
        customers = generate_customers(config)
        if not customers:
            print("ERROR: Cannot proceed without customers")
            return
        
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
                print(f"ERROR: Customer not found for order {order['order_id']}, skipping...")
                continue
            
            # Get the products in this order
            order_products = []
            for item in order["items"]:
                product = next((p for p in products if p["product_id"] == item["product_id"]), None)
                if product:
                    order_products.append(product)
            
            if not order_products:
                print(f"ERROR: No products found for order {order['order_id']}, skipping...")
                continue
        
        print(f"  Dimensions: {dimensions['query_type']} / {dimensions['complexity']}")
        if order:
            print(f"  Order: {order['order_id']} / Customer: {customer['customer_id']}")
        else:
            print(f"  Customer: {customer['customer_id']} (no specific order)")
        
        # Select and customize scenario template with pre-validated policies
        scenario = select_and_customize_scenario(policy_graph, scenario_templates, 
                                                dimensions['query_type'], order, customer, order_products)
        print(f"  Scenario: {scenario['name']} (complexity {scenario['complexity_level']})")
        print(f"  Primary policy: {scenario['primary_policy']}")
        print(f"  Applicable policies: {scenario['applicable_policies']}")
        print(f"  Expected outcome: {scenario.get('expected_outcome', 'unknown')}")
        
        # Generate email
        email = generate_customer_email_v3(scenario, dimensions)
        if not email:
            print(f"  ERROR: Failed to generate email, skipping ticket...")
            continue
        
        # Generate realistic timestamp and update scenario context
        if scenario.get("order"):
            order_date = scenario["order"]["order_date"]
            context = scenario.get("context", {})
            
            # Generate timestamp by analyzing the email content
            email_timestamp = generate_realistic_email_timestamp(
                order_date=order_date,
                email_content=email,
                scenario=scenario,
                context=context
            )
            
            # Calculate ACTUAL days_since_purchase from email timestamp and order date
            order_dt = datetime.datetime.strptime(order_date, "%Y-%m-%d")
            email_dt = datetime.datetime.fromisoformat(email_timestamp)
            actual_days_since_purchase = (email_dt - order_dt).days
            actual_months_since_purchase = actual_days_since_purchase / 30.44
            
            # Update scenario context with REAL timing
            scenario["context"]["days_since_purchase"] = actual_days_since_purchase
            scenario["context"]["months_since_purchase"] = actual_months_since_purchase
            scenario["_email_timestamp"] = email_timestamp  # Store for ticket creation
            
            print(f"  Email timestamp: {email_timestamp} ({actual_days_since_purchase} days after order)")
        else:
            scenario["_email_timestamp"] = datetime.datetime.now().isoformat()
        
        # Generate resolution using policy graph (now with corrected context)
        resolution = generate_resolution_v3(email, scenario, policy_graph, dimensions)
        if not resolution:
            print(f"  ERROR: Failed to generate resolution, skipping ticket...")
            continue
        
        # Create complete ticket
        ticket = create_complete_ticket_v3(config, scenario, email, resolution, dimensions)
        new_tickets.append(ticket)
        
        print(f"  ✓ Ticket {ticket['ticket_id']} generated")
    
    # Combine tickets for append mode
    all_tickets = existing_tickets + new_tickets if config.mode == "append" else new_tickets
    
    # Phase 4: Save Dataset
    print("\nPhase 4: Saving dataset...")
    save_dataset(config, all_tickets, policy, customers, orders, products)
    
    # Save policy graph for analysis
    if config.mode == "create":
        save_policy_graph(policy_graph, config)
    
    print("\n=== Generation Complete! ===")
    
    # Print summary statistics
    print(f"\nDataset Statistics (new tickets):")
    if config.include_debug_info and new_tickets:
        query_types = {}
        complexities = {}
        scenario_names = {}
        expected_outcomes = {}
        complexity_levels = {}
        policy_interactions = {}
        
        for ticket in new_tickets:
            if "_scenario_dimensions" in ticket:
                qt = ticket["_scenario_dimensions"]["query_type"]
                cx = ticket["_scenario_dimensions"]["complexity"]
                query_types[qt] = query_types.get(qt, 0) + 1
                complexities[cx] = complexities.get(cx, 0) + 1
            
            if "_scenario_template" in ticket:
                name = ticket["_scenario_template"]["name"]
                outcome = ticket["_scenario_template"].get("expected_outcome", "unknown")
                complexity_level = ticket["_scenario_template"].get("complexity_level", 1)
                scenario_names[name] = scenario_names.get(name, 0) + 1
                expected_outcomes[outcome] = expected_outcomes.get(outcome, 0) + 1
                complexity_levels[complexity_level] = complexity_levels.get(complexity_level, 0) + 1
            
            if "_policy_analysis" in ticket:
                interaction_type = ticket["_policy_analysis"]["policy_interactions"]
                policy_interactions[interaction_type] = policy_interactions.get(interaction_type, 0) + 1
        
        if query_types:
            print("\nQuery Type Distribution:")
            for qt, count in sorted(query_types.items()):
                print(f"  {qt}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if complexities:
            print("\nCustomer Complexity Distribution:")
            for cx, count in sorted(complexities.items()):
                print(f"  {cx}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if complexity_levels:
            print("\nScenario Complexity Levels:")
            for level, count in sorted(complexity_levels.items()):
                print(f"  Level {level}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if policy_interactions:
            print("\nPolicy Interaction Types:")
            for interaction, count in sorted(policy_interactions.items()):
                print(f"  {interaction}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if expected_outcomes:
            print("\nExpected Outcome Distribution:")
            for outcome, count in sorted(expected_outcomes.items()):
                print(f"  {outcome}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if scenario_names:
            print("\nTop Scenario Templates:")
            for name, count in sorted(scenario_names.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        # Policy utilization analysis
        all_policies_used = set()
        for ticket in new_tickets:
            if "_policy_analysis" in ticket:
                all_policies_used.update(ticket["_policy_analysis"].get("applicable_policies", []))
        
        if all_policies_used:
            print(f"\nPolicy Coverage: {len(all_policies_used)} unique policies used")
            print(f"Policies: {sorted(all_policies_used)}")
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
        config.num_tickets = 100
        config.num_products = 35
        config.num_customers = 50
        config.num_orders = 70
    
    # Run generation
    main(config)