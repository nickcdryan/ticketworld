"""
Improved scenario templates based on user feedback.
Copy the create_scenario_templates function from this file to factory_3.py
"""

from typing import Dict, List
from dataclasses import dataclass, field
from typing import Any

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
                }
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
                description="Customer asking about product availability",
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
                    "should_mention": ["product_interest", "in_stock", "when_available"],
                    "might_omit": ["specific_model"],
                    "tone_modifier": "curious"
                }
            ),
            ScenarioTemplate(
                scenario_id="GENERAL-002",
                name="product_comparison",
                description="Customer asking to compare products",
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
                    "should_mention": ["product_options", "differences", "recommendation"],
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