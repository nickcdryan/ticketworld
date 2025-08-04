#!/usr/bin/env python3
"""
Script to dilute company policy document with irrelevant sections.
Takes company_policy.txt and creates company_policy_full.txt with added content.
"""

import re
import random
from pathlib import Path

def read_policy_file(file_path):
    """Read the original policy file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_irrelevant_sections():
    """Return a list of irrelevant policy sections to add."""
    sections = [
        {
            "title": "Corporate Information",
            "content": """
[POL-CORP-001] Company Formation
Rule: TechNest Inc. was incorporated in Delaware on March 15, 2019. Registration number: 7834562
Conditions: legal_entity_status

[POL-CORP-002] Board of Directors
Rule: Board meetings conducted quarterly. Minutes maintained by corporate secretary for 7 years minimum
Conditions: governance_requirements

[POL-CORP-003] Subsidiary Companies
Rule: TechNest operates through subsidiaries: TechNest EU Ltd (Ireland), TechNest Asia Pte Ltd (Singapore)
"""
        },
        {
            "title": "Environmental Compliance",
            "content": """
[POL-ENV-001] Waste Disposal
Rule: All packaging materials must comply with local recycling regulations. Electronic waste processed through certified vendors only
Conditions: facility_operations

[POL-ENV-002] Carbon Footprint
Rule: Annual carbon footprint assessment required. Target: 15% reduction year-over-year in Scope 1 and 2 emissions
Conditions: sustainability_reporting

[POL-ENV-003] Green Packaging Initiative
Rule: Packaging materials: 85% recycled content minimum for all shipments over 2 lbs
Conditions: package_weight_threshold
"""
        },
        {
            "title": "Information Security",
            "content": """
[POL-SEC-001] Data Classification
Rule: All corporate data classified as Public, Internal, Confidential, or Restricted. Access controls enforced per classification level
Conditions: employee_access_level

[POL-SEC-002] Password Requirements
Rule: Employee passwords: Minimum 12 characters, changed every 90 days, multi-factor authentication required
Conditions: system_access

[POL-SEC-003] Vendor Security Assessment
Rule: Third-party vendors: Security questionnaire and penetration test results required before contract approval
Conditions: vendor_onboarding
"""
        },
        {
            "title": "Legal Disclaimers",
            "content": """
This document contains proprietary and confidential information of TechNest Inc. and its affiliates. 
Unauthorized reproduction, distribution, or disclosure is strictly prohibited and may result in legal action.

TechNest Inc. reserves the right to modify these policies at any time without prior notice. 
Continued use of services constitutes acceptance of policy modifications.

All trademarks, service marks, and trade names referenced herein are the property of their respective owners. 
TechNest Inc. makes no claim to ownership of third-party intellectual property.
"""
        },
        {
            "title": "Human Resources",
            "content": """
[POL-HR-001] Employee Code of Conduct
Rule: All employees must complete annual ethics training. Certification required by December 31st each year
Conditions: active_employment

[POL-HR-002] Workplace Safety
Rule: Safety training mandatory within 30 days of hire. Hard hat and safety vest required in warehouse areas
Conditions: facility_access

[POL-HR-003] Time and Attendance
Rule: Employee schedules: Submitted 2 weeks in advance. Overtime pre-approval required for hours exceeding 40 per week
Conditions: hourly_employee
"""
        },
        {
            "title": "Facility Management",
            "content": """
[POL-FAC-001] Building Access
Rule: Key card access: Granted based on role requirements. Visitor escort required for non-badged personnel
Conditions: building_security

[POL-FAC-002] Equipment Maintenance
Rule: HVAC systems: Quarterly maintenance schedule. Temperature maintained between 68-72°F in office areas
Conditions: facility_operations

[POL-FAC-003] Emergency Procedures
Rule: Fire drills: Conducted quarterly. Assembly point located in east parking lot. Headcount verification required
Conditions: emergency_response
"""
        },
        {
            "title": "Financial Controls",
            "content": """
[POL-FIN-001] Expense Approval
Rule: Expenses over $500: Director approval required. Expenses over $5000: VP approval required
Conditions: authorization_level

[POL-FIN-002] Audit Requirements
Rule: External audit: Conducted annually by certified public accounting firm. Working papers retained for 7 years
Conditions: financial_reporting

[POL-FIN-003] Petty Cash Management
Rule: Petty cash fund: Maximum $200 per location. Receipts required for all disbursements over $10
Conditions: cash_handling
"""
        },
        {
            "title": "Vendor Relations",
            "content": """
[POL-VENDOR-001] Supplier Qualification
Rule: New suppliers: Credit check and references required. Minimum 3 years operating history for preferred vendor status
Conditions: vendor_approval_process

[POL-VENDOR-002] Payment Terms
Rule: Standard payment terms: Net 30 days. Early payment discount: 2% if paid within 10 days
Conditions: accounts_payable

[POL-VENDOR-003] Contract Management
Rule: Vendor contracts: Legal review required for agreements over $25,000 annually
Conditions: contract_value_threshold
"""
        },
        {
            "title": "Intellectual Property",
            "content": """
TechNest™ and the TechNest logo are registered trademarks of TechNest Inc. in the United States and other countries.

All software, documentation, and related materials are protected by copyright laws and international treaty provisions. 
Unauthorized copying, modification, or distribution is prohibited.

Patent applications are pending for various aspects of our proprietary ticketing technology platform. 
Third-party use requires written authorization from TechNest Legal Department.
"""
        },
        {
            "title": "Regulatory Compliance",
            "content": """
[POL-REG-001] Data Protection
Rule: GDPR compliance: Data processing records maintained. Privacy impact assessments required for new systems
Conditions: eu_operations

[POL-REG-002] Export Controls
Rule: Technology exports: Export control classification required. ECCN determination documented before international transfer
Conditions: international_operations

[POL-REG-003] Anti-Money Laundering
Rule: Suspicious transactions: Reported to Financial Crimes Enforcement Network within 30 days of detection
Conditions: transaction_monitoring
"""
        }
    ]
    return sections

def insert_sections_strategically(original_content, irrelevant_sections):
    """Insert irrelevant sections between existing sections and at the end."""
    
    lines = original_content.split('\n')
    section_boundaries = []
    
    # Find complete section boundaries by looking for section titles
    # (non-empty lines followed by === underlines)
    i = 0
    while i < len(lines) - 1:
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        
        # If current line has content and next line is underline, this is a section start
        if current_line and next_line and '=' * 5 in next_line:
            section_boundaries.append(i)  # Insert before the section title
            
            # Skip ahead past this section header
            i += 2
            
            # Find the end of this section's content (next section or end of file)
            while i < len(lines):
                if i < len(lines) - 1:
                    next_title = lines[i].strip()
                    next_underline = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    
                    # If we hit another section, break
                    if next_title and next_underline and '=' * 5 in next_underline:
                        break
                i += 1
        else:
            i += 1
    
    # Add position for end of document
    section_boundaries.append(len(lines))
    
    # Shuffle irrelevant sections for random distribution
    shuffled_sections = irrelevant_sections.copy()
    random.shuffle(shuffled_sections)
    
    # Insert sections at boundaries
    result_lines = lines.copy()
    offset = 0
    
    # Insert between existing sections
    sections_to_insert = min(len(shuffled_sections), len(section_boundaries) - 1)
    
    for i in range(sections_to_insert):
        pos = section_boundaries[i]
        section = shuffled_sections[i]
        
        # Create the section content
        section_content = [
            "",  # Empty line before
            section["title"],
            "=" * len(section["title"]),
            section["content"].strip(),
            ""   # Empty line after
        ]
        
        # Insert at position + offset
        insert_pos = pos + offset
        result_lines[insert_pos:insert_pos] = section_content
        offset += len(section_content)
    
    # Add remaining sections at the end
    for i in range(sections_to_insert, len(shuffled_sections)):
        section = shuffled_sections[i]
        
        end_content = [
            "",  # Empty line before
            section["title"],
            "=" * len(section["title"]),
            section["content"].strip()
        ]
        
        result_lines.extend(end_content)
    
    return '\n'.join(result_lines)

def main():
    """Main function to process the policy file."""
    input_file = Path("../assets/company_policy.txt")
    output_file = Path("../assets/company_policy_full.txt")
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        return
    
    print(f"Reading original policy from {input_file}...")
    original_content = read_policy_file(input_file)
    
    print("Adding irrelevant sections to dilute content...")
    irrelevant_sections = get_irrelevant_sections()
    
    # Set random seed for reproducible results
    random.seed(42)
    
    diluted_content = insert_sections_strategically(original_content, irrelevant_sections)
    
    print(f"Writing diluted policy to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(diluted_content)
    
    # Print statistics
    original_lines = len(original_content.split('\n'))
    diluted_lines = len(diluted_content.split('\n'))
    
    print(f"\nDocument statistics:")
    print(f"Original: {original_lines} lines")
    print(f"Diluted:  {diluted_lines} lines")
    print(f"Ratio:    {diluted_lines / original_lines:.1f}x")
    print(f"\nDiluted policy document created: {output_file}")

if __name__ == "__main__":
    main() 