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
import networkx as nx  # For graph operations and consistency checking

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

# Enhanced Policy Structure with LLM-Driven Interactions
@dataclass
class PolicyClause:
    """Represents a single policy clause parsed from LLM-generated policy document"""
    clause_id: str
    title: str
    rule: str
    category: str = ""
    # These will be populated by LLM analysis
    interacts_with: List[str] = field(default_factory=list)
    interaction_reasons: Dict[str, str] = field(default_factory=dict)
    
@dataclass
class PolicyInteraction:
    """Represents an interaction between two policies discovered by LLM analysis"""
    policy_a: str
    policy_b: str
    interaction_type: str  # "modifies", "overrides", "requires", "conflicts", "complements"
    reasoning: str
    scenarios: List[str] = field(default_factory=list)
    confidence: float = 1.0

class PolicyGraph:
    """Manages policy clauses and their LLM-discovered interactions"""
    
    def __init__(self):
        self.clauses: Dict[str, PolicyClause] = {}
        self.interactions: List[PolicyInteraction] = []
        self.nx_graph: nx.Graph = nx.Graph()  # NetworkX graph for analysis
        
    def add_clause(self, clause: PolicyClause):
        """Add a policy clause to the graph"""
        self.clauses[clause.clause_id] = clause
        self.nx_graph.add_node(clause.clause_id, 
                              title=clause.title, 
                              category=clause.category)
    
    def add_interaction(self, interaction: PolicyInteraction):
        """Add a policy interaction discovered by LLM analysis"""
        self.interactions.append(interaction)
        
        # Add to clause's interaction list
        if interaction.policy_a in self.clauses:
            self.clauses[interaction.policy_a].interacts_with.append(interaction.policy_b)
            self.clauses[interaction.policy_a].interaction_reasons[interaction.policy_b] = interaction.reasoning
            
        if interaction.policy_b in self.clauses:
            self.clauses[interaction.policy_b].interacts_with.append(interaction.policy_a)
            self.clauses[interaction.policy_b].interaction_reasons[interaction.policy_a] = interaction.reasoning
        
        # Add edge to NetworkX graph
        self.nx_graph.add_edge(interaction.policy_a, interaction.policy_b,
                              interaction_type=interaction.interaction_type,
                              reasoning=interaction.reasoning,
                              scenarios=interaction.scenarios)
    
    def get_related_policies(self, clause_id: str, max_hops: int = 3) -> List[str]:
        """Get all policies related to a given clause within max_hops using NetworkX"""
        if clause_id not in self.nx_graph:
            return []
        
        # Use NetworkX to find nodes within max_hops
        related = []
        for node in self.nx_graph.nodes():
            if node != clause_id:
                try:
                    path_length = nx.shortest_path_length(self.nx_graph, clause_id, node)
                    if path_length <= max_hops:
                        related.append(node)
                except nx.NetworkXNoPath:
                    # No path exists
                    continue
        
        return related
    
    def get_interaction_reasoning(self, policy_a: str, policy_b: str) -> str:
        """Get the reasoning for why two policies interact"""
        if policy_a in self.clauses and policy_b in self.clauses[policy_a].interaction_reasons:
            return self.clauses[policy_a].interaction_reasons[policy_b]
        return "No direct interaction found"
    
    def validate_graph_consistency(self) -> Dict[str, Any]:
        """Validate graph consistency and return analysis"""
        analysis = {
            "total_nodes": len(self.nx_graph.nodes()),
            "total_edges": len(self.nx_graph.edges()),
            "connected_components": list(nx.connected_components(self.nx_graph)),
            "average_degree": sum(dict(self.nx_graph.degree()).values()) / len(self.nx_graph.nodes()) if self.nx_graph.nodes() else 0,
            "isolated_nodes": list(nx.isolates(self.nx_graph)),
            "bridge_edges": list(nx.bridges(self.nx_graph))
        }
        return analysis
    
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
                policy_text.append(f"{clause.rule}")
        
        return "\n".join(policy_text)

def generate_policy_document_with_llm(config: DatasetConfig) -> str:
    """Generate a natural policy document using LLM for a consumer electronics retailer"""
    
    system_prompt = f"""You are creating a comprehensive customer service policy document for {config.company_name}, 
    a consumer electronics retailer. Create realistic, detailed policies that sound natural and professional.
    Each policy should have a clear policy code in [POL-XXXX-###] format."""
    
    prompt = f"""Generate a comprehensive customer service policy document for {config.company_name}, 
    a consumer electronics retailer. The document should include the following policy areas with specific policy codes:

REQUIRED POLICY AREAS:
1. Return Policy - multiple clauses covering different scenarios
2. Shipping Policy - covering various shipping issues  
3. Warranty Policy - covering warranty claims and exclusions
4. Price Match Policy - covering price matching rules
5. Order Policy - covering order modifications and cancellations
6. Holiday/Seasonal Policy - covering special conditions
7. Customer Service Standards - covering response times and escalation

FORMAT REQUIREMENTS:
- Each policy clause should have a unique code like [POL-RETURN-001], [POL-SHIP-001], etc.
- Include specific time periods (30 days, 14 days, etc.)
- Include specific dollar amounts and thresholds where relevant
- Include clear conditions and exceptions
- Make policies realistic for a consumer electronics business
- Write in professional but clear language

POLICY COMPLEXITY:
- Create 8-10 total policy clauses across all areas
- Some policies should naturally interact with others (e.g., return window vs holiday extension)
- Include edge cases and exceptions that would require multiple policies to resolve
- Make sure some policies complement each other and some might conflict in edge cases

EXAMPLE FORMAT:
[POL-RETURN-001] Return Window
All items may be returned within 30 days of delivery date. Returns after 30 days will not be accepted.

[POL-RETURN-002] Restocking Fee
Opened electronic items are subject to a 15% restocking fee, except when the item is defective.

Generate a complete policy document with this structure. Focus on creating realistic business policies that would actually be used by a consumer electronics retailer."""

    return call_llm(prompt, system_prompt)

def parse_policy_document(policy_text: str) -> List[PolicyClause]:
    """Parse the LLM-generated policy document into PolicyClause objects"""
    clauses = []
    
    # Find all policy clauses using regex
    pattern = r'\[([A-Z]+-[A-Z]+-\d+)\]\s*([^\n]+)\n([^[]*?)(?=\[[A-Z]+-[A-Z]+-\d+\]|$)'
    matches = re.findall(pattern, policy_text, re.DOTALL)
    
    for match in matches:
        clause_id, title, rule = match
        
        # Clean up the rule text
        rule = rule.strip()
        
        # Determine category from clause_id
        category = "Unknown"
        if "RETURN" in clause_id:
            category = "Return Policy"
        elif "SHIP" in clause_id:
            category = "Shipping Policy"
        elif "WARRANTY" in clause_id:
            category = "Warranty Policy"
        elif "PRICE" in clause_id:
            category = "Price Match Policy"
        elif "ORDER" in clause_id:
            category = "Order Policy"
        elif "HOLIDAY" in clause_id:
            category = "Holiday Policy"
        elif "COMM" in clause_id:
            category = "Customer Service"
        
        clauses.append(PolicyClause(
            clause_id=clause_id,
            title=title.strip(),
            rule=rule,
            category=category
        ))
    
    return clauses

def analyze_policy_interactions(clauses: List[PolicyClause]) -> List[PolicyInteraction]:
    """Use LLM to analyze interactions between all policy pairs"""
    
    interactions = []
    total_pairs = len(clauses) * (len(clauses) - 1) // 2
    processed = 0
    
    print(f"  Analyzing {total_pairs} policy pairs for interactions...")
    
    for i, policy_a in enumerate(clauses):
        for j, policy_b in enumerate(clauses[i+1:], i+1):
            processed += 1
            if processed % 10 == 0:
                print(f"    Analyzed {processed}/{total_pairs} pairs...")
            
            # Call LLM to analyze this specific pair
            interaction = analyze_single_policy_pair(policy_a, policy_b)
            if interaction:
                interactions.append(interaction)
    
    print(f"  ✓ Found {len(interactions)} policy interactions")
    return interactions

def analyze_single_policy_pair(policy_a: PolicyClause, policy_b: PolicyClause) -> Optional[PolicyInteraction]:
    """Analyze a single pair of policies to determine if they interact"""
    
    system_prompt = """You are a policy analysis expert. Your job is to determine if two customer service policies interact with each other.
    
    Policies interact if:
    - One modifies how the other is applied
    - One overrides the other in certain situations
    - One requires the other to be checked first
    - They conflict in certain scenarios
    - They complement each other in handling complex cases
    
    Consider realistic customer service scenarios where both policies might come into play."""
    
    prompt = f"""Analyze these two customer service policies to determine if and how they interact:

POLICY A:
[{policy_a.clause_id}] {policy_a.title}
{policy_a.rule}

POLICY B:
[{policy_b.clause_id}] {policy_b.title}
{policy_b.rule}

ANALYSIS TASK:
1. Do these policies interact in any customer service scenarios?
2. If yes, how do they interact? (modifies, overrides, requires, conflicts, complements)
3. What specific scenarios would involve both policies?
4. What is your reasoning?

Think step by step:
- Consider different customer situations where both policies might apply
- Look for time periods, dollar amounts, conditions that might overlap
- Consider edge cases where policies might conflict or complement
- Think about the order in which policies would need to be applied

Respond in JSON format:
{{
    "interacts": true/false,
    "interaction_type": "modifies/overrides/requires/conflicts/complements/none",
    "reasoning": "Detailed explanation of why and how they interact",
    "scenarios": ["scenario 1", "scenario 2", ...],
    "confidence": 0.0-1.0
}}

If the policies are completely unrelated (e.g., shipping costs vs warranty exclusions), return "interacts": false."""

    try:
        response = call_llm(prompt, system_prompt)
        analysis = safe_json_parse(response, "object")
        
        if analysis and analysis.get("interacts", False):
            return PolicyInteraction(
                policy_a=policy_a.clause_id,
                policy_b=policy_b.clause_id,
                interaction_type=analysis.get("interaction_type", "unknown"),
                reasoning=analysis.get("reasoning", ""),
                scenarios=analysis.get("scenarios", []),
                confidence=analysis.get("confidence", 0.5)
            )
    except Exception as e:
        print(f"    Error analyzing {policy_a.clause_id} vs {policy_b.clause_id}: {e}")
    
    return None

def build_policy_graph_with_llm(config: DatasetConfig) -> Tuple[PolicyGraph, str]:
    """Complete pipeline: Generate policy document, parse it, analyze interactions, build graph"""
    
    print("Step 1: Generating policy document with LLM...")
    policy_text = generate_policy_document_with_llm(config)
    
    print("Step 2: Parsing policy document...")
    clauses = parse_policy_document(policy_text)
    print(f"  ✓ Parsed {len(clauses)} policy clauses")
    
    if not clauses:
        raise Exception("Failed to parse any policy clauses from generated document")
    
    print("Step 3: Analyzing policy interactions...")
    interactions = analyze_policy_interactions(clauses)
    
    print("Step 4: Building policy graph...")
    graph = PolicyGraph()
    
    # Add all clauses
    for clause in clauses:
        graph.add_clause(clause)
    
    # Add all interactions
    for interaction in interactions:
        graph.add_interaction(interaction)
    
    print("Step 5: Validating graph consistency...")
    consistency = graph.validate_graph_consistency()
    print(f"  ✓ Graph has {consistency['total_nodes']} nodes, {consistency['total_edges']} edges")
    print(f"  ✓ Average degree: {consistency['average_degree']:.1f}")
    if consistency['isolated_nodes']:
        print(f"  ⚠️  {len(consistency['isolated_nodes'])} isolated policies: {consistency['isolated_nodes']}")
    
    return graph, policy_text

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
        "warranty_claim": [
            ScenarioTemplate(
                scenario_id="WARRANTY-001",
                name="warranty_claim_valid",
                description="Product failed within warranty period",
                primary_policy="POL-WARRANTY-002",
                context_requirements={
                    "months_since_purchase": (2, 11),
                    "damage_type": "manufacturing"
                },
                expected_outcome="approve",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "free_repair_or_replacement",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["product_name", "issue_description", "purchase_timeframe"],
                    "might_omit": ["warranty_registration"],
                    "tone_modifier": "matter_of_fact"
                }
            ),
            ScenarioTemplate(
                scenario_id="WARRANTY-002",
                name="water_damage_claim",
                description="Product damaged by water",
                primary_policy="POL-WARRANTY-003",
                context_requirements={
                    "months_since_purchase": (1, 8),
                    "damage_type": "water"
                },
                expected_outcome="deny",
                complexity_level=2,  # Warranty exclusion
                customer_situation={
                    "customer_expectation": "warranty_replacement",
                    "complication": "water_damage"
                },
                email_patterns={
                    "should_mention": ["product_name", "spilled_liquid", "stopped_working"],
                    "might_omit": ["water_damage_admission"],
                    "tone_modifier": "hopeful"
                }
            ),
            ScenarioTemplate(
                scenario_id="WARRANTY-003",
                name="expired_warranty",
                description="Product issue after warranty expired",
                primary_policy="POL-WARRANTY-001",
                context_requirements={
                    "months_since_purchase": (13, 18),
                    "damage_type": "manufacturing"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "free_repair",
                    "complication": "warranty_expired"
                },
                email_patterns={
                    "should_mention": ["product_name", "issue_description", "purchase_timeframe"],
                    "might_omit": ["exact_warranty_period"],
                    "tone_modifier": "hopeful"
                }
            )
        ],
        "pricing_dispute": [
            ScenarioTemplate(
                scenario_id="PRICE-001",
                name="price_match_valid",
                description="Customer found lower price at authorized retailer",
                primary_policy="POL-PRICE-001",
                context_requirements={
                    "days_since_purchase": (1, 14),
                    "competitor_type": "authorized"
                },
                expected_outcome="approve",
                complexity_level=2,  # Need to verify retailer
                customer_situation={
                    "customer_expectation": "price_difference_refunded",
                    "complication": "competitor_verification"
                },
                email_patterns={
                    "should_mention": ["competitor_name", "competitor_price", "product_name"],
                    "might_omit": ["exact_url"],
                    "tone_modifier": "expecting_cooperation"
                }
            ),
            ScenarioTemplate(
                scenario_id="PRICE-002",
                name="price_match_too_late",
                description="Price match request after 14-day window",
                primary_policy="POL-PRICE-001",
                context_requirements={
                    "days_since_purchase": (20, 30),
                    "competitor_type": "authorized"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "price_adjustment",
                    "complication": "outside_window"
                },
                email_patterns={
                    "should_mention": ["purchase_date", "current_price", "just_noticed"],
                    "might_omit": ["exact_days"],
                    "tone_modifier": "disappointed"
                }
            ),
            ScenarioTemplate(
                scenario_id="PRICE-003",
                name="price_match_marketplace",
                description="Customer found lower price at marketplace seller",
                primary_policy="POL-PRICE-002",
                context_requirements={
                    "days_since_purchase": (1, 10),
                    "competitor_type": "marketplace"
                },
                expected_outcome="deny",
                complexity_level=2,  # Policy exclusion
                customer_situation={
                    "customer_expectation": "price_difference_refunded",
                    "complication": "marketplace_exclusion"
                },
                email_patterns={
                    "should_mention": ["marketplace_seller", "much_lower_price", "product_name"],
                    "might_omit": ["seller_authorization"],
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
                context_requirements={
                    "days_since_order": (3, 5),
                    "order_status": "shipped"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "order_cancelled",
                    "complication": "already_shipped"
                },
                email_patterns={
                    "should_mention": ["cancel_order", "changed_mind"],
                    "might_omit": ["order_status"],
                    "tone_modifier": "urgent"
                }
            ),
            ScenarioTemplate(
                scenario_id="ORDER-002",
                name="modify_processing_order",
                description="Customer wants to modify an order in processing",
                primary_policy="POL-ORDER-001",
                context_requirements={
                    "hours_since_order": (3, 24),
                    "order_status": "processing"
                },
                expected_outcome="deny",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "change_accepted",
                    "complication": "past_modification_window"
                },
                email_patterns={
                    "should_mention": ["order_id", "requested_change"],
                    "might_omit": ["time_restriction"],
                    "tone_modifier": "hopeful"
                }
            )
        ],
        "general_inquiry": [
            ScenarioTemplate(
                scenario_id="GENERAL-001",
                name="product_availability",
                description="Customer asking about product availability",
                primary_policy="POL-COMM-001",
                context_requirements={},
                expected_outcome="information",
                complexity_level=1,
                customer_situation={
                    "customer_expectation": "information",
                    "complication": "none"
                },
                email_patterns={
                    "should_mention": ["product_interest", "availability_question"],
                    "might_omit": ["specific_model"],
                    "tone_modifier": "curious"
                }
            )
        ]
    }
    
    return templates

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


def check_dependencies():
    """Check that required dependencies are installed"""
    try:
        import networkx as nx
        return True
    except ImportError:
        print("ERROR: NetworkX is required but not installed.")
        print("Please install it with: pip install networkx")
        return False


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


def select_and_customize_scenario_v2(policy_graph: PolicyGraph, scenario_templates: Dict[str, List[ScenarioTemplate]], 
                                   query_type: str, order: Dict, customer: Dict, products: List[Dict]) -> Dict:
    """Select and customize a scenario template with LLM-based policy graph integration."""
    
    # Get templates for this query type
    templates = scenario_templates.get(query_type, [])
    if not templates:
        # Fallback if query type not found
        templates = scenario_templates["return_request"]
    
    # Select a random template
    template = random.choice(templates)
    
    # Calculate order context
    context = build_order_context(order, customer, products)
    
    # Get all related policies using the new policy graph
    primary_policy = template.primary_policy
    
    # Find if the primary policy exists in our generated policies
    available_policies = list(policy_graph.clauses.keys())
    if primary_policy not in available_policies:
        # If exact primary policy doesn't exist, find the closest match by category
        template_category = ""
        if "RETURN" in primary_policy:
            template_category = "Return Policy"
        elif "SHIP" in primary_policy:
            template_category = "Shipping Policy"
        elif "WARRANTY" in primary_policy:
            template_category = "Warranty Policy"
        elif "PRICE" in primary_policy:
            template_category = "Price Match Policy"
        elif "ORDER" in primary_policy:
            template_category = "Order Policy"
        
        # Find first policy in that category
        for policy_id, clause in policy_graph.clauses.items():
            if clause.category == template_category:
                primary_policy = policy_id
                break
        else:
            # If no match found, use first available policy
            primary_policy = available_policies[0] if available_policies else "POL-UNKNOWN-001"
    
    # Get related policies using NetworkX graph traversal
    related_policies = policy_graph.get_related_policies(primary_policy, max_hops=2)
    all_relevant_policies = [primary_policy] + related_policies
    
    # For now, use all relevant policies as applicable (we'll let LLM figure out conflicts)
    applicable_policies = all_relevant_policies
    
    # Customize scenario based on actual order data and policy interactions
    scenario = {
        "scenario_id": template.scenario_id,
        "name": template.name,
        "description": template.description,
        "primary_policy": primary_policy,
        "all_relevant_policies": all_relevant_policies,  # For generation use
        "applicable_policies": applicable_policies,      # All related policies
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
        # Calculate time-based context
        order_date = datetime.datetime.strptime(order["order_date"], "%Y-%m-%d")
        days_since_order = (datetime.datetime.now() - order_date).days
        months_since_order = days_since_order / 30.44  # Average month length
        
        context.update({
            "days_since_purchase": days_since_order,
            "months_since_purchase": months_since_order,
            "order_status": order.get("order_status", "delivered"),
            "total_order_value": order.get("total_amount", 0)
        })
        
        # Calculate order value context
        if order.get("total_amount", 0) > 500:
            context["item_over_500"] = True
        
        # Check if it's a holiday purchase (Nov-Dec)
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
        order_info = "ORDER INFORMATION: No specific order (general inquiry)"
    
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
- Days since purchase: {context.get('days_since_purchase', 'N/A')}
- Item value: ${context.get('item_value', 'N/A')}
- Order status: {context.get('order_status', 'N/A')}
- Has receipt: {context.get('has_receipt', True)}

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

Write a natural email from the CUSTOMER'S perspective. Remember:
- You are the customer who has the problem described in: {scenario['description']}
- You are writing TO customer support asking for help
- You do NOT have access to internal policies, systems, or detailed company procedures
- You only know what a typical customer would know about their own order

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


def generate_resolution_v4(email: Dict, scenario: Dict, policy_graph: PolicyGraph, dimensions: Dict[str, str]) -> Dict:
    """Generate a resolution plan FROM a customer service representative.
    
    This simulates what happens AFTER receiving the customer's email:
    - The CSR looks up customer information in internal systems
    - Reviews company policies and their interactions
    - Creates an action plan to resolve the customer's issue
    - Has full access to internal data, policies, and procedures
    """
    
    system_prompt = """You are an expert customer service professional. Create resolutions using ONLY 
    the provided information and policies. Follow policies exactly with no exceptions.
    Consider ALL applicable policies and their interactions. DENY requests that violate policy."""
    
    order = scenario.get("order")
    customer = scenario["customer"]
    products = scenario.get("products", [])
    context = scenario.get("context", {})
    
    # Get applicable policies from the scenario (discovered by LLM analysis)
    applicable_policies = scenario.get("applicable_policies", [scenario.get("primary_policy")])
    
    # Build policy text for these specific policies
    policy_text_sections = []
    for policy_id in applicable_policies:
        if policy_id in policy_graph.clauses:
            clause = policy_graph.clauses[policy_id]
            policy_text_sections.append(f"[{policy_id}] {clause.title}\nRule: {clause.rule}")
    
    # Add policy interaction insights discovered by LLM
    interaction_insights = []
    primary_policy = scenario.get("primary_policy")
    if primary_policy:
        for other_policy in applicable_policies:
            if other_policy != primary_policy:
                reasoning = policy_graph.get_interaction_reasoning(primary_policy, other_policy)
                if reasoning != "No direct interaction found":
                    interaction_insights.append(f"Interaction between {primary_policy} and {other_policy}: {reasoning}")
    
    relevant_policy_text = "\n\n".join(policy_text_sections)
    if interaction_insights:
        relevant_policy_text += "\n\nPOLICY INTERACTIONS:\n" + "\n".join(interaction_insights)
    
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

APPLICABLE POLICIES (These are the ONLY policies that apply after analyzing interactions):
{relevant_policy_text}

POLICY INTERACTION NOTES:
- All policy conflicts have been resolved
- Higher precedence policies override lower precedence ones
- Conditions have been checked against the context above

CRITICAL ENFORCEMENT RULES:
1. Include order_id and order_date in the resolution
2. ALL actions must be based on the applicable policies above
3. STRICTLY enforce policy conditions:
   - Returns after 30 days: DENY (unless holiday extension applies)
   - Returns without receipt: DENY  
   - Price match after 14 days: DENY
   - Order modification after 2 hours: DENY
   - Cancellation of shipped orders: DENY
   - Warranty claims after 1 year: DENY
   - Water damage warranty claims: DENY
4. For damaged items over $500, MUST request photo per POL-SHIP-005
5. Value in actions must match actual product prices from order
6. Consider policy interactions (e.g., defective items override restocking fees)

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
    "policy_references": {applicable_policies},
    "policy_reasoning": "Brief explanation of how policies interact for this case",
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

IMPORTANT: 
- Use ONLY the applicable policies listed above
- Every action must cite specific policy clauses
- Be precise about denials - use deny_* action types when policies are violated
- Consider all policy interactions when making decisions

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
        
        # Ensure only applicable policies are referenced
        resolution["policy_references"] = applicable_policies
        
        # Policy-aware validation
        if "actions" in resolution:
            for action in resolution["actions"]:
                # Use policy graph to validate action against context
                days_since_purchase = context.get("days_since_purchase", 0)
                
                # Validate return window
                if action["type"] == "process_return" and days_since_purchase > 30:
                    # Check for holiday extension
                    if "POL-HOLIDAY-001" in applicable_policies and context.get("purchase_month") in [11, 12] and days_since_purchase <= 45:
                        pass  # Holiday extension applies
                    else:
                        action["type"] = "deny_return"
                        action["reason"] = f"Return request outside 30-day window ({days_since_purchase} days) per POL-RETURN-001"
                        action["value"] = 0
                
                # Validate warranty claims
                months_since_purchase = context.get("months_since_purchase", 0)
                if action["type"] == "honor_warranty" and months_since_purchase > 12:
                    action["type"] = "deny_warranty_claim"
                    action["reason"] = f"Warranty expired ({months_since_purchase:.1f} months) per POL-WARRANTY-001"
                    action["value"] = 0
    else:
        print(f"ERROR: Failed to generate resolution")
        return None
    
    return resolution


def create_complete_ticket_v3(config: DatasetConfig, scenario: Dict, email: Dict, 
                             resolution: Dict, dimensions: Dict[str, str]) -> Dict:
    """Combine all elements into a complete ticket with enhanced policy traceability."""
    
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
            "interactions": {
                "interacts_with": clause.interacts_with,
                "interaction_reasons": clause.interaction_reasons
            },
            "total_connections": len(clause.interacts_with)
        }
    
    # Serialize discovered interactions with reasoning
    graph_data["discovered_interactions"] = []
    for interaction in policy_graph.interactions:
        graph_data["discovered_interactions"].append({
            "policy_a": interaction.policy_a,
            "policy_b": interaction.policy_b,
            "interaction_type": interaction.interaction_type,
            "reasoning": interaction.reasoning,
            "scenarios": interaction.scenarios,
            "confidence": interaction.confidence
        })
    
    # Generate interaction summary using NetworkX
    for clause_id in policy_graph.clauses.keys():
        related = policy_graph.get_related_policies(clause_id, max_hops=3)
        direct_connections = len(policy_graph.clauses[clause_id].interacts_with)
        graph_data["interaction_summary"][clause_id] = {
            "direct_connections": direct_connections,
            "reachable_within_3_hops": len(related),
            "related_policies": related[:5]  # Top 5 for readability
        }
    
    # Complexity analysis
    category_counts = {}
    interaction_types = {}
    
    for clause in policy_graph.clauses.values():
        # Count by category
        cat = clause.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    # Count interaction types
    for interaction in policy_graph.interactions:
        itype = interaction.interaction_type
        interaction_types[itype] = interaction_types.get(itype, 0) + 1
    
    graph_data["complexity_analysis"] = {
        "category_distribution": category_counts,
        "interaction_types": interaction_types,
        "total_interactions": len(policy_graph.interactions),
        "average_interactions_per_policy": len(policy_graph.interactions) / len(policy_graph.clauses) if policy_graph.clauses else 0,
        "networkx_analysis": {
            "total_nodes": policy_graph.nx_graph.number_of_nodes(),
            "total_edges": policy_graph.nx_graph.number_of_edges(),
            "average_degree": sum(dict(policy_graph.nx_graph.degree()).values()) / policy_graph.nx_graph.number_of_nodes() if policy_graph.nx_graph.number_of_nodes() else 0,
            "connected_components": len(list(nx.connected_components(policy_graph.nx_graph))),
            "diameter": nx.diameter(policy_graph.nx_graph) if nx.is_connected(policy_graph.nx_graph) else "Disconnected"
        }
    }
    
    # Save to file
    graph_path = config.get_filepath("policy_graph.json")
    with open(graph_path, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"- Policy graph structure in policy_graph.json")
    return graph_data


def main(config: DatasetConfig):
    """Main generation pipeline."""
    
    print("=== Synthetic Customer Support Dataset Generator (LLM-Driven Policy Analysis) ===")
    print(f"Mode: {config.mode}")
    print(f"Output directory: {config.output_dir}")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    if config.mode == "append":
        # Load existing data
        print("\nLoading existing data...")
        try:
            policy, customers, orders, products = load_existing_data(config)
            existing_tickets = load_existing_tickets(config)
            print(f"Loaded {len(existing_tickets)} existing tickets")
            
            # For append mode, we need to recreate the policy graph from the existing policy
            print("Recreating policy graph from existing policy document...")
            
            try:
                # Parse existing policy and rebuild graph
                clauses = parse_policy_document(policy)
                if not clauses:
                    raise Exception("Failed to parse existing policy document")
                
                # Rebuild interactions (this might be expensive but ensures consistency)
                print("Re-analyzing policy interactions...")
                interactions = analyze_policy_interactions(clauses)
                
                # Rebuild graph
                policy_graph = PolicyGraph()
                for clause in clauses:
                    policy_graph.add_clause(clause)
                for interaction in interactions:
                    policy_graph.add_interaction(interaction)
                
                scenario_templates = create_scenario_templates()
                print(f"✓ Recreated policy graph with {len(policy_graph.clauses)} clauses and {len(interactions)} interactions")
            except Exception as e:
                print(f"ERROR: Failed to recreate policy graph: {e}")
                return
        except FileNotFoundError as e:
            print(f"Error: Could not find existing data files. Please run in 'create' mode first.")
            print(f"Missing file: {e}")
            return
    else:
        # Create mode - generate everything from scratch
        existing_tickets = []
        
        # Phase 1: Company Foundation (NEW LLM-DRIVEN APPROACH)
        print("\nPhase 1: Generating company foundation with LLM analysis...")
        print("=" * 60)
        
        try:
            policy_graph, policy = build_policy_graph_with_llm(config)
            print(f"  ✓ Successfully built policy graph with {len(policy_graph.clauses)} policy clauses")
        except Exception as e:
            print(f"ERROR: Failed to build policy graph: {e}")
            return
        
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
        
        # Select and customize scenario template with LLM-based policy graph
        scenario = select_and_customize_scenario_v2(policy_graph, scenario_templates, 
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
        
        # Generate resolution using LLM-based policy graph
        resolution = generate_resolution_v4(email, scenario, policy_graph, dimensions)
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
        config.num_tickets = 5
        config.num_products = 6
        config.num_customers = 5
        config.num_orders = 7
    
    # Run generation
    main(config)