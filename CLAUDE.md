# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Dependencies and Environment
```bash
# Install dependencies
uv sync

# Run main dataset generator (test configuration: 100 tickets, 50 customers, 35 products, 70 orders)
uv run python factory.py

# Generate custom dataset sizes
uv run python factory.py --tickets 500 --customers 200 --products 100 --orders 300

# Append to existing dataset
uv run python factory.py --mode append --tickets 100

# Generate clean dataset without debug metadata
uv run python factory.py --no-debug --tickets 1000
```

### Utilities Workflow
```bash
# Run utilities (from utils/ directory)
cd utils && python policy_dilution_script.py    # Add realistic policy dilution
cd utils && python convert_to_sqlite.py         # Convert JSON database to SQLite
cd utils && python audit_tickets.py             # Quality assurance check
cd utils && python validate_templates.py        # Template validation (dev tool)
```

### Complete Dataset Generation Workflow
```bash
# 1. Generate core dataset
uv run python factory.py --tickets 500 --customers 200

# 2. Add policy dilution (makes policy document more realistic)
uv run python utils/policy_dilution_script.py

# 3. Convert to SQLite for easier querying
uv run python utils/convert_to_sqlite.py
```

## Architecture Overview

### Core Components

**TicketWorld** is a synthetic customer service dataset generator that creates realistic, interconnected datasets for training and evaluating LLM systems on customer support tasks.

#### Data Generation Pipeline
1. **Policy Graph Creation** (`factory.py`): Company policies modeled as interconnected clauses with relationships (overrides, modifies, requires)
2. **Asset Generation**: Customers → Products → Orders → Tickets (respecting dependencies)
3. **Scenario Templates**: Pre-built templates define customer situations requiring multi-policy reasoning
4. **Synthetic Data Generation**: LLM creates realistic customer emails and policy-compliant resolutions

#### Key Classes and Structures
- `PolicyClause`: Represents policy rules with interaction metadata
- `PolicyGraph`: Manages policy relationships and multi-hop reasoning
- `DatasetConfig`: Configuration management for generation parameters
- `ScenarioTemplate`: Templates for different customer service scenarios

### File Structure

#### Core Files
- `factory.py`: Main dataset generator with all generation logic
- `pyproject.toml`: Project dependencies (google-genai, networkx)

#### Generated Assets (`assets/` directory)
- `support_tickets.json`: Complete ticket dataset with customer emails and resolutions
- `customer_database.json`: Customer profiles, orders, and product catalog  
- `company_policy.txt`: Clean company policy document
- `policy_graph.json`: Policy interaction structure and metadata

#### Enhanced Assets (after running utils)
- `company_policy_full.txt`: Policy document with realistic dilution content
- `customer_database.db`: SQLite version of customer database
- `ticket_audit_results.json`: Quality analysis of generated tickets
- `ticket_audit_report.txt`: Human-readable audit summary

#### Utilities (`utils/` directory)
- `policy_dilution_script.py`: Adds irrelevant content to policy document
- `convert_to_sqlite.py`: Converts JSON database to SQLite with 3-table schema
- `audit_tickets.py`: Reviews tickets for policy compliance and errors
- `validate_templates.py`: Analyzes scenario templates and discovers policy interactions

### Key Design Patterns

#### Multi-hop Policy Reasoning
The system creates tickets requiring understanding interactions between multiple company policies. During generation, targeted information access is provided, but during evaluation, LLMs must retrieve information from large databases and reason over numerous pieces of data.

#### Synthetic Data Pipeline
- **Generation phase**: Provides targeted information access and deterministic metadata for consistency
- **Evaluation phase**: Removes scaffolds - LLM must accurately retrieve and reason over data
- **Data relationships**: Customers, orders, and products have realistic transaction histories

#### Asset Dependencies
Generation respects dependencies: Customers → Products → Orders → Tickets. Each component builds on previous components to maintain data integrity.

## Environment Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Google Gemini API access (requires `GEMINI_API_KEY` environment variable)

## Dataset Configuration

### Typical Distributions
- **Ticket Types**: Returns (25%), Shipping Issues (20%), Billing Disputes (20%), Warranty Claims (15%), etc.
- **Complexity Levels**: Simple (40%), Requires Lookup (35%), Edge Cases (20%), Escalation Required (5%)
- **Customer Tiers**: Standard (70%), Premium (20%), VIP (10%)
- **Information Completeness**: Complete (30%), Missing Details (40%), Wrong Info (30%)

### Quality Features
- **Policy Compliance**: All resolutions reference specific policy clauses
- **Realistic Timing**: Email timestamps align with customer descriptions
- **Data Consistency**: Customer/order relationships maintained across all tickets
- **Edge Cases**: Wrong emails, missing information, partial customer matches