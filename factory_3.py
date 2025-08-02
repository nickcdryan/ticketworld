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
        interacts_with=["POL-RETURN-002", "POL-RETURN-003"],
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
        precedence=4,
        category="Return Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-RETURN-003",
        title="Unopened Items",
        rule="Unopened items: Full refund, no restocking fee",
        conditions=["within_return_window"],
        modifies=["POL-RETURN-002"],
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
        rule="Wrong item shipped: Free return label provided, correct item sent immediately",
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
        precedence=3,
        category="Warranty Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-WARRANTY-003",
        title="Warranty Exclusions",
        rule="Not covered: Physical damage, water damage, normal wear",
        conditions=["warranty_period"],
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
        precedence=3,
        category="Price Match Policy"
    ))
    
    graph.add_clause(PolicyClause(
        clause_id="POL-PRICE-002",
        title="Price Match Exclusions",
        rule="Exclusions: Marketplace sellers, clearance items, bundles",
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
    
    return graph

# Enhanced Scenario Templates with Policy Graph Integration
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

def create_scenario_templates() -> Dict[str, List[ScenarioTemplate]]:
    """Create scenario templates organized by query type"""
    
    templates = {
        "return_request": [
            ScenarioTemplate(
                scenario_id="RETURN-001",
                name="return_outside_window",
                description="Customer wants to return item outside 30-day window",
                primary_policy="POL-RETURN-001",
                context_requirements={
                    "days_since_purchase": (31, 90),
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
                    "should_mention": ["purchase_date", "product_name", "return_reason"],
                    "might_omit": ["order_number"],
                    "tone_modifier": "disappointed"
                }
                         ),
            ScenarioTemplate(
                scenario_id="RETURN-002",
                name="return_opened_item",
                description="Customer wants to return opened electronic item within window",
                primary_policy="POL-RETURN-002",
                context_requirements={
                    "days_since_purchase": (10, 25),
                    "has_receipt": True,
                    "item_condition": "opened"
                },
                expected_outcome="approve_with_fee",
                complexity_level=2,  # Involves restocking fee calculation
                customer_situation={
                    "customer_expectation": "full_refund",
                    "complication": "item_opened"
                },
                email_patterns={
                    "should_mention": ["product_name", "opened_status", "reason"],
                    "might_omit": ["restocking_fee_awareness"],
                    "tone_modifier": "hopeful"
                }
            ),
            ScenarioTemplate(
                scenario_id="RETURN-003",
                name="return_defective_item",
                description="Customer received defective item",
                primary_policy="POL-RETURN-004",
                context_requirements={
                    "days_since_purchase": (1, 20),
                    "has_receipt": True,
                    "item_condition": "defective"
                },
                expected_outcome="approve",
                complexity_level=2,  # Defective override restocking fee
                customer_situation={
                    "customer_expectation": "full_refund_no_fee",
                    "complication": "item_defective"
                },
                email_patterns={
                    "should_mention": ["product_name", "defect_description"],
                    "might_omit": ["troubleshooting_attempted"],
                    "tone_modifier": "frustrated"
                }
            ),
            ScenarioTemplate(
                scenario_id="RETURN-004",
                name="return_no_receipt",
                description="Customer wants to return item without receipt",
                primary_policy="POL-RETURN-001",
                context_requirements={
                    "days_since_purchase": (5, 15),
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
                    "should_mention": ["lost_receipt", "product_name"],
                    "might_omit": ["order_number", "purchase_date"],
                    "tone_modifier": "apologetic"
                }
            ),
            ScenarioTemplate(
                scenario_id="RETURN-005",
                name="return_holiday_extended",
                description="Customer returning holiday purchase within extended window",
                primary_policy="POL-HOLIDAY-001",
                context_requirements={
                    "days_since_purchase": (35, 44),
                    "purchase_month": (11, 12),  # Nov-Dec
                    "has_receipt": True,
                    "item_condition": "unopened"
                },
                expected_outcome="approve",
                complexity_level=3,  # Holiday policy modifies standard return policy
                customer_situation={
                    "customer_expectation": "full_refund",
                    "complication": "holiday_extension"
                },
                email_patterns={
                    "should_mention": ["holiday_purchase", "product_name"],
                    "might_omit": ["exact_policy_knowledge"],
                    "tone_modifier": "hopeful"
                }
            )
        ],
        "shipping_issue": [
            ScenarioTemplate(
                scenario_id="SHIP-001",
                name="package_not_received",
                description="Customer never received package marked as delivered",
                primary_policy="POL-SHIP-002",
                context_requirements={
                    "days_since_delivery": (1, 7),
                    "tracking_status": "delivered",
                    "item_value": (50, 1000)
                },
                expected_outcome="investigate",
                complexity_level=2,  # Requires investigation period
                customer_situation={
                    "customer_expectation": "replacement_or_refund",
                    "complication": "shows_delivered"
                },
                email_patterns={
                    "should_mention": ["tracking_number", "delivery_date", "checked_neighbors"],
                    "might_omit": ["exact_address"],
                    "tone_modifier": "worried"
                }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-002",
                name="wrong_item_received",
                description="Customer received different product than ordered",
                primary_policy="POL-SHIP-004",
                context_requirements={
                    "days_since_delivery": (1, 3),
                    "item_value": (20, 800)
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "free_exchange",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["ordered_product", "received_product"],
                    "might_omit": ["order_number"],
                    "tone_modifier": "frustrated"
                }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-003",
                name="damaged_high_value",
                description="High-value item arrived damaged",
                primary_policy="POL-SHIP-005",
                context_requirements={
                    "days_since_delivery": (1, 2),
                    "item_value": (501, 2000)
                },
                expected_outcome="conditional",
                complexity_level=3,  # Photo requirement for high value
                customer_situation={
                    "customer_expectation": "replacement",
                    "complication": "high_value_damage"
                },
                email_patterns={
                    "should_mention": ["damage_description", "product_name"],
                    "might_omit": ["photos_attached"],
                    "tone_modifier": "upset"
                }
            ),
            ScenarioTemplate(
                scenario_id="SHIP-004",
                name="damaged_standard_value",
                description="Standard-value item arrived damaged",
                primary_policy="POL-SHIP-006",
                context_requirements={
                    "days_since_delivery": (1, 3),
                    "item_value": (10, 500)
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "replacement",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["damage_description", "product_name"],
                    "might_omit": ["extent_of_damage"],
                    "tone_modifier": "disappointed"
                }
            )
        ],
    "product_defect": [
        {
            "scenario_id": "DEFECT-001",
            "name": "warranty_claim",
            "description": "Product failed within warranty period",
            "policy_clauses": ["POL-WARRANTY-001", "POL-WARRANTY-002"],
            "expected_outcome": "approve",  # Should be approved
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
            "name": "water_damage_claim",
            "description": "Product damaged by water",
            "policy_clauses": ["POL-WARRANTY-001", "POL-WARRANTY-003"],
            "expected_outcome": "deny",  # Should be denied - water damage not covered
            "customer_situation": {
                "days_since_purchase": "30-60",
                "customer_expectation": "warranty_replacement",
                "complication": "water_damage"
            },
            "email_patterns": {
                "should_mention": ["product_name", "spilled_liquid", "stopped_working"],
                "might_omit": ["water_damage_admission"],
                "tone_modifier": "hopeful"
            }
        }
    ],
    "pricing_dispute": [
        {
            "scenario_id": "PRICE-001",
            "name": "price_match_request",
            "description": "Customer found lower price elsewhere",
            "policy_clauses": ["POL-PRICE-001", "POL-PRICE-002"],
            "expected_outcome": "conditional",  # Depends on retailer
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
            "name": "price_match_too_late",
            "description": "Price match request after 14-day window",
            "policy_clauses": ["POL-PRICE-001"],
            "expected_outcome": "deny",  # Should be denied - outside window
            "customer_situation": {
                "days_since_purchase": "20-30",
                "customer_expectation": "price_adjustment",
                "complication": "outside_window"
            },
            "email_patterns": {
                "should_mention": ["purchase_date", "current_price", "just_noticed"],
                "might_omit": ["exact_days"],
                "tone_modifier": "disappointed"
            }
        }
    ],
    "order_status": [
        {
            "scenario_id": "STATUS-001",
            "name": "cancel_shipped_order",
            "description": "Customer wants to cancel already shipped order",
            "policy_clauses": ["POL-ORDER-001", "POL-SHIP-001"],
            "expected_outcome": "deny",  # Should be denied - already shipped
            "customer_situation": {
                "days_since_order": "3-5",
                "customer_expectation": "order_cancelled",
                "complication": "already_shipped"
            },
            "email_patterns": {
                "should_mention": ["cancel_order", "changed_mind"],
                "might_omit": ["order_status"],
                "tone_modifier": "urgent"
            }
        }
    ],
    "warranty_claim": [
        {
            "scenario_id": "WARRANTY-001",
            "name": "expired_warranty",
            "description": "Product issue after warranty expired",
            "policy_clauses": ["POL-WARRANTY-001"],
            "expected_outcome": "deny",  # Should be denied - warranty expired
            "customer_situation": {
                "months_since_purchase": "13-18",
                "customer_expectation": "free_repair",
                "complication": "warranty_expired"
            },
            "email_patterns": {
                "should_mention": ["product_name", "issue_description", "purchase_timeframe"],
                "might_omit": ["exact_warranty_period"],
                "tone_modifier": "hopeful"
            }
        }
    ],
    "general_inquiry": [
        {
            "scenario_id": "GENERAL-001",
            "name": "product_availability",
            "description": "Customer asking about product availability or restock",
            "policy_clauses": ["POL-COMM-001"],
            "expected_outcome": "information",  # Just provide info
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
            "name": "modify_processing_order",
            "description": "Customer wants to modify an order already in processing",
            "policy_clauses": ["POL-ORDER-001"],
            "expected_outcome": "deny",  # Should be denied - past 2 hour window
            "customer_situation": {
                "days_since_order": "1-2",
                "customer_expectation": "change_accepted",
                "complication": "already_processing"
            },
            "email_patterns": {
                "should_mention": ["order_id", "requested_change"],
                "might_omit": ["original_items"],
                "tone_modifier": "hopeful"
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
    "initiate_investigation"
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
    """Generate a simple, rule-based company policy document."""
    
    system_prompt = f"""You are creating a simple, rule-based customer service policy document for {config.company_name}, 
    a consumer electronics retailer. Every rule must be absolute with no ambiguity - only yes/no, allowed/not allowed."""
    
    prompt = f"""Generate a simple customer service policy document for {config.company_name} with clear, discrete rules:

1. Return Policy [POL-RETURN-001 to POL-RETURN-004]
   - ALL returns require receipt
   - ALL items: 30-day return window from delivery date
   - Unopened items: Full refund
   - Opened items: 15% restocking fee  
   - Defective items: Full refund, no restocking fee
   - No returns accepted after 30 days for any reason
   - No returns without receipt

2. Shipping Policy [POL-SHIP-001 to POL-SHIP-005]
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
   - No warranty service after 1 year

4. Price Match Policy [POL-PRICE-001 to POL-PRICE-002]
   - Authorized retailers only: Price matched within 14 days of purchase
   - Exclusions: Marketplace sellers, clearance items, bundles
   - Refund method: Original payment method
   - No price match after 14 days

5. Order Modification Policy [POL-ORDER-001]
   - Changes allowed: Within 2 hours of order placement
   - After 2 hours: Order cannot be modified
   - Shipped orders: Cannot be cancelled

6. Customer Service Standards [POL-COMM-001 to POL-COMM-002]
   - Response time: Within 24 hours
   - Escalation: Available for orders over $1000

IMPORTANT: 
- Use exact policy tags like [POL-RETURN-001]
- No conditional language (no "may", "might", "could", "should consider")
- Every rule is absolute - yes or no, allowed or not allowed
- Include specific dollar amounts and time periods
- Return as plain text, not JSON"""
    
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
    - alternate_emails (20% have 1-2 alternates as array)
    - phone
    - shipping_addresses (array with 1-2 addresses, include street, city, state, zip)
    - billing_addresses (array, 70% same as shipping)
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
        "expected_outcome": template.get("expected_outcome", "unknown"),
        "query_type": query_type,
        "order": order,
        "customer": customer,
        "products": products,
        "days_since_order": days_since_order,
        "customer_situation": template["customer_situation"].copy(),
        "email_patterns": template["email_patterns"].copy()
    }
    
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
        print(f"ERROR: Failed to generate email for scenario {scenario['name']}")
        return None
    
    return email


def generate_resolution_v2(email: Dict, scenario: Dict, policy: str, dimensions: Dict[str, str]) -> Dict:
    """Generate a resolution using exact order data and pre-selected policies."""
    
    system_prompt = """You are an expert customer service professional. Create resolutions using ONLY 
    the provided information and cite ONLY the specified policy sections. Follow policies exactly with no exceptions.
    DENY requests that violate policy."""
    
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
- Days Since Order: {scenario.get('days_since_order', 'N/A')}
- Items with values: {json.dumps(product_values, indent=2)}
- Total Order Value: ${order['total_amount']}
- Status: {order['order_status']}

PRODUCTS IN ORDER WITH PRICES:
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

{order_info}

SCENARIO CONTEXT:
- Issue Type: {scenario['description']}
- Customer Expectation: {scenario['customer_situation']['customer_expectation']}
- Complication: {scenario['customer_situation']['complication']}
- Days Since Order: {scenario.get('days_since_order', 'N/A')}
- Expected Outcome Hint: {scenario.get('expected_outcome', 'unknown')}

RELEVANT POLICIES (follow these EXACTLY):
{relevant_policy_text}

REQUIRED POLICY REFERENCES: {scenario['policy_clauses']}

CRITICAL RULES:
1. Include order_id and order_date in the resolution
2. ALL actions must be based on cited policies - no exceptions
3. DENY requests that violate policy:
   - Returns after 30 days: DENY
   - Returns without receipt: DENY  
   - Price match after 14 days: DENY
   - Order modification after 2 hours: DENY
   - Cancellation of shipped orders: DENY
   - Warranty claims after 1 year: DENY
   - Water damage warranty claims: DENY
4. For damaged items over $500, MUST request photo before approving replacement
5. Value in actions must match actual product prices from order

Create a resolution with this structure:
{{
    "order_id": "{order.get('order_id', 'N/A') if order else 'N/A'}",
    "order_date": "{order.get('order_date', 'N/A') if order else 'N/A'}",
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
            "reason": "Detailed reason citing specific policies by tag (e.g., per POL-RETURN-001)",
            "value": exact dollar amount from product prices if refund/replacement (0 for denials),
            "details": "Specific implementation details"
        }}
    ],
    "escalation_required": boolean,
    "escalation_reason": "Why escalation needed" or null,
    "priority": "low/medium/high/urgent",
    "total_resolution_value": sum of all monetary values in actions
}}

IMPORTANT: 
- Must include order_id and order_date fields
- Check all policy requirements and DENY if violated
- Every action must cite a specific policy clause
- Use exact product prices from order for value calculations
- Be clear about denials - use deny_* action types

Return ONLY the JSON object."""
    
    resolution_text = call_llm(prompt, system_prompt)
    resolution = safe_json_parse(resolution_text, "object")
    
    # Validate and ensure consistency
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
        
        # Ensure only specified policies are referenced
        resolution["policy_references"] = scenario["policy_clauses"]
        
        # Additional validation for policy compliance
        if "actions" in resolution and scenario.get("days_since_order") is not None:
            days = scenario["days_since_order"]
            
            for action in resolution["actions"]:
                # Enforce return window
                if action["type"] == "process_return" and days > 30:
                    action["type"] = "deny_return"
                    action["reason"] = f"Return request is outside 30-day window (order is {days} days old) per POL-RETURN-001"
                    action["value"] = 0
    else:
        print(f"ERROR: Failed to generate resolution")
        return None
    
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
        "order_id": scenario["order"]["order_id"] if scenario.get("order") else "N/A",
        "resolution_plan": resolution
    }
    
    # Add debug info if configured
    if config.include_debug_info:
        ticket["_scenario_dimensions"] = dimensions
        ticket["_scenario_template"] = {
            "scenario_id": scenario["scenario_id"],
            "name": scenario["name"],
            "policy_clauses": scenario["policy_clauses"],
            "expected_outcome": scenario.get("expected_outcome", "unknown")
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
        
        # Select and customize scenario template
        scenario = select_scenario_template(dimensions['query_type'], order, customer, order_products)
        print(f"  Scenario: {scenario['name']} with policies {scenario['policy_clauses']}")
        print(f"  Expected outcome: {scenario.get('expected_outcome', 'unknown')}")
        
        # Generate email
        email = generate_customer_email_v2(scenario, dimensions)
        if not email:
            print(f"  ERROR: Failed to generate email, skipping ticket...")
            continue
        
        # Generate resolution
        resolution = generate_resolution_v2(email, scenario, policy, dimensions)
        if not resolution:
            print(f"  ERROR: Failed to generate resolution, skipping ticket...")
            continue
        
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
        expected_outcomes = {}
        for ticket in new_tickets:
            if "_scenario_dimensions" in ticket:
                qt = ticket["_scenario_dimensions"]["query_type"]
                cx = ticket["_scenario_dimensions"]["complexity"]
                query_types[qt] = query_types.get(qt, 0) + 1
                complexities[cx] = complexities.get(cx, 0) + 1
            
            if "_scenario_template" in ticket:
                name = ticket["_scenario_template"]["name"]
                outcome = ticket["_scenario_template"].get("expected_outcome", "unknown")
                scenario_names[name] = scenario_names.get(name, 0) + 1
                expected_outcomes[outcome] = expected_outcomes.get(outcome, 0) + 1
        
        if query_types:
            print("\nQuery Type Distribution:")
            for qt, count in sorted(query_types.items()):
                print(f"  {qt}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if complexities:
            print("\nComplexity Distribution:")
            for cx, count in sorted(complexities.items()):
                print(f"  {cx}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
        if expected_outcomes:
            print("\nExpected Outcome Distribution:")
            for outcome, count in sorted(expected_outcomes.items()):
                print(f"  {outcome}: {count} ({count/len(new_tickets)*100:.1f}%)")
        
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