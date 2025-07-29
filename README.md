# Synthetic Customer Service Data Generator

A comprehensive Python tool that generates realistic customer service datasets for testing and evaluating LLM systems. Creates authentic customer support scenarios with interconnected databases, policy documents, and resolution plans.

## What It Does

This generator creates a complete customer service ecosystem including:

- **Realistic Support Tickets**: Varied complexity levels from simple requests to complex multi-step issues
- **Customer Database**: Interconnected customer profiles with transaction histories and tier status
- **Order Database**: Realistic e-commerce transactions with edge cases (shipping errors, defects, etc.)
- **Company Policy Document**: 2000+ word policy with both relevant and irrelevant sections
- **Resolution Plans**: Policy-compliant solutions that reference specific policy sections

## What It Generates

The system outputs two main files:

### `full_customer_service_dataset.json`
Complete dataset containing:
- Company policy document (~2000 words)
- Customer database with tiered accounts (standard/premium/VIP)
- Order database with realistic transaction history
- Support tickets with resolution plans

### `customer_service_evaluation.json` 
Format optimized for evaluation systems with:
- Structured question-answer pairs
- Complete context for each ticket
- Expected resolution plans for validation

## Key Features

- **Realistic Edge Cases**: Wrong emails, missing information, partial customer matches
- **Policy Compliance**: All resolutions reference specific policy sections and authorization levels
- **Varied Complexity**: Simple requests to complex multi-step resolutions requiring escalation
- **Database Relationships**: Customers properly linked to orders with realistic transaction patterns
- **Comprehensive Policy**: Detailed policy document with relevant sections mixed with noise
- **Multi-Step Challenge**: Requires information extraction, database lookup, policy search, business logic application, and structured response generation

## Installation & Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync

# Set up your API key
export GEMINI_API_KEY="your-api-key-here"
```

## Usage Instructions

1. **Configure the LLM function**: The `call_llm()` function is already configured for Google Gemini API. Update it with your preferred LLM provider if needed.

2. **Run the generator**:
```bash
uv run python factory.py
```

3. **Adjust parameters as needed**:
```python
dataset = generator.generate_complete_dataset(
    num_customers=100,    # 50-200 recommended
    num_tickets=200,      # 100-500 recommended
)
```

4. **Set random seed** for reproducible datasets:
```python
generator = CustomerServiceDataGenerator(seed=42)
```

## Configuration Options

| Parameter | Description | Recommended Range |
|-----------|-------------|-------------------|
| `num_customers` | Total customers to generate | 50-200 |
| `num_tickets` | Total support tickets to generate | 100-500 |
| `seed` | Random seed for reproducible results | Any integer |

## Dataset Structure

### Ticket Categories
- **Shipping Errors** (25%): Wrong items, damaged packages, delivery issues
- **Billing Disputes** (20%): Unexpected charges, incorrect amounts
- **Product Defects** (20%): Manufacturing issues, quality problems
- **Refund Requests** (15%): Returns, cancellations
- **Account Issues** (10%): Login problems, security concerns
- **Technical Support** (10%): How-to questions, compatibility issues

### Information Completeness Levels
- **Complete** (40%): Full customer info (email + order + name)
- **Partial** (35%): Missing some details (email + order OR email + name)
- **Minimal** (20%): Basic info only (just email)
- **Insufficient** (5%): Wrong/missing info requiring follow-up

### Customer Tiers
- **Standard** (70%): Basic support, standard policies
- **Premium** (20%): Priority support, extended returns
- **VIP** (10%): Dedicated support, manager escalation privileges

## Use Cases

Perfect for testing LLM systems that need to:
- Handle multi-step customer service workflows
- Perform database lookups with partial information
- Apply business rules and policies consistently
- Manage escalation procedures
- Generate structured responses
- Deal with edge cases and missing information

## Example Output

The generated dataset creates challenging scenarios like:
- Customer emails from different addresses than their account
- Partial order numbers with missing digits
- Complex issues requiring manager-level authorization
- Policy boundary testing (return windows, refund limits)
- Cross-referenced customer and order data validation

---

*Generated datasets provide comprehensive testing environments for customer service AI systems, ensuring robust handling of real-world complexity and edge cases.*
