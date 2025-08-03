"""
One-time validation script to analyze scenario templates and update them with 
discovered policy interactions. Outputs Python code ready to paste into factory_3.py.
"""

import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from factory_3 import (
    PolicyGraph, create_policy_graph, create_scenario_templates, 
    DatasetConfig, call_llm, safe_json_parse, ScenarioTemplate
)
from typing import Dict, List

def validate_and_update_templates():
    """Validate all templates and generate updated Python code"""
    
    # Create policy graph and templates
    config = DatasetConfig()
    policy_graph = create_policy_graph(config)
    scenario_templates = create_scenario_templates()
    
    print("=== Validating Scenario Templates ===\n")
    
    # Define policy groups
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
    
    # Store updated templates
    updated_templates = {}
    
    # Process each template
    for query_type, templates in scenario_templates.items():
        updated_templates[query_type] = []
        
        for template in templates:
            print(f"Validating: {template.scenario_id} - {template.name}")
            
            # Collect all relevant policies
            all_relevant_policies = [template.primary_policy]
            
            # Check each policy group
            for group_name, group_policies in policy_groups.items():
                # Skip if primary policy is in this group
                if template.primary_policy in group_policies:
                    continue
                
                # Check relevance
                result = check_policy_group_relevance(template, group_name, group_policies, policy_graph)
                
                if result.get("applies", False):
                    relevant = result.get("relevant_policies", [])
                    print(f"  ✓ {group_name}: {', '.join(relevant)}")
                    all_relevant_policies.extend(relevant)
            
            # Remove duplicates
            all_relevant_policies = list(set(all_relevant_policies))
            
            # Create updated template data
            updated_template = {
                "scenario_id": template.scenario_id,
                "name": template.name,
                "description": template.description,
                "primary_policy": template.primary_policy,
                "all_relevant_policies": all_relevant_policies,
                "context_requirements": template.context_requirements,
                "expected_outcome": template.expected_outcome,
                "complexity_level": template.complexity_level,
                "customer_situation": template.customer_situation,
                "email_patterns": template.email_patterns
            }
            
            updated_templates[query_type].append(updated_template)
            print(f"  Total policies: {len(all_relevant_policies)}\n")
    
    # Generate Python code
    generate_python_code(updated_templates)

def check_policy_group_relevance(template, group_name: str, group_policies: List[str], policy_graph: PolicyGraph) -> Dict:
    """Check if policies in a group are relevant to a scenario"""
    
    # Build context description
    context_parts = []
    if template.context_requirements:
        for key, value in template.context_requirements.items():
            if isinstance(value, tuple):
                context_parts.append(f"{key}: {value[0]}-{value[1]}")
            else:
                context_parts.append(f"{key}: {value}")
    
    context_description = ", ".join(context_parts) if context_parts else "No specific context"
    
    # Get policy details
    policy_details = []
    for policy_id in group_policies:
        if policy_id in policy_graph.clauses:
            clause = policy_graph.clauses[policy_id]
            policy_details.append(f"- [{policy_id}] {clause.title}: {clause.rule}")
    
    policy_list = "\n".join(policy_details)
    
    # Create prompt
    system_prompt = """You are a customer service policy expert. Identify ONLY obvious, 
    direct policy interactions - not theoretical edge cases."""
    
    prompt = f"""Given this scenario:

SCENARIO: {template.description}
CONTEXT: {context_description}
EXPECTED OUTCOME: {template.expected_outcome}
PRIMARY POLICY: {template.primary_policy}

Looking at these {group_name}:
{policy_list}

Do any of these policies OBVIOUSLY apply in a way that affects the resolution?

Consider only:
1. Clear, direct interactions a CSR would immediately recognize
2. Policies that change or add to the resolution
3. Conditions explicitly met by the scenario

Do NOT consider:
- Theoretical edge cases
- Indirect connections
- General policies unless they add specific actions

Respond in JSON:
{{
    "applies": true/false,
    "relevant_policies": ["POL-XXX-###", ...],
    "reason": "Brief explanation"
}}"""
    
    try:
        response = call_llm(prompt, system_prompt)
        result = safe_json_parse(response, "object")
        
        if result and isinstance(result, dict):
            # Validate policies exist
            if "relevant_policies" in result:
                result["relevant_policies"] = [
                    p for p in result["relevant_policies"] 
                    if p in group_policies
                ]
            return result
        else:
            return {"applies": False, "relevant_policies": []}
            
    except Exception as e:
        print(f"    Error: {e}")
        return {"applies": False, "relevant_policies": []}

def generate_python_code(updated_templates: Dict):
    """Generate Python code for updated templates"""
    
    print("\n=== UPDATED TEMPLATE CODE ===")
    print("Copy and paste this into factory_3.py:\n")
    print("-" * 80)
    
    print("""def create_scenario_templates() -> Dict[str, List[ScenarioTemplate]]:
    \"\"\"Create scenario templates organized by query type\"\"\"
    
    templates = {""")
    
    for i, (query_type, templates) in enumerate(updated_templates.items()):
        if i > 0:
            print(",")
        print(f'        "{query_type}": [')
        
        for j, template in enumerate(templates):
            if j > 0:
                print(",")
            print("            ScenarioTemplate(")
            print(f'                scenario_id="{template["scenario_id"]}",')
            print(f'                name="{template["name"]}",')
            print(f'                description="{template["description"]}",')
            print(f'                primary_policy="{template["primary_policy"]}",')
            
            # Add all_relevant_policies if more than just primary
            if len(template["all_relevant_policies"]) > 1:
                policies_str = json.dumps(template["all_relevant_policies"])
                print(f'                all_relevant_policies={policies_str},')
            
            # Context requirements
            if template["context_requirements"]:
                print("                context_requirements={")
                for k, (key, value) in enumerate(template["context_requirements"].items()):
                    comma = "," if k < len(template["context_requirements"]) - 1 else ""
                    if isinstance(value, str):
                        print(f'                    "{key}": "{value}"{comma}')
                    else:
                        print(f'                    "{key}": {value}{comma}')
                print("                },")
            else:
                print("                context_requirements={},")
            
            print(f'                expected_outcome="{template["expected_outcome"]}",')
            print(f'                complexity_level={template["complexity_level"]},')
            
            # Customer situation
            print("                customer_situation={")
            for k, (key, value) in enumerate(template["customer_situation"].items()):
                comma = "," if k < len(template["customer_situation"]) - 1 else ""
                print(f'                    "{key}": "{value}"{comma}')
            print("                },")
            
            # Email patterns
            print("                email_patterns={")
            for k, (key, value) in enumerate(template["email_patterns"].items()):
                comma = "," if k < len(template["email_patterns"]) - 1 else ""
                if isinstance(value, list):
                    print(f'                    "{key}": {json.dumps(value)}{comma}')
                else:
                    print(f'                    "{key}": "{value}"{comma}')
            print("                }")
            
            print("            )", end="")
        
        print("\n        ]", end="")
    
    print("""
    }
    
    return templates""")
    
    print("-" * 80)
    print("\nValidation complete! Copy the code above to replace create_scenario_templates() in factory_3.py")

if __name__ == "__main__":
    validate_and_update_templates() 