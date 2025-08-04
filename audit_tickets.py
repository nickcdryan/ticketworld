"""
Ticket Auditor - Reviews generated tickets for policy compliance and errors.
Checks each resolution against the complete company policy document.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from factory_3 import call_llm, safe_json_parse, DatasetConfig

def load_tickets(filepath: str) -> List[Dict]:
    """Load tickets from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_policy(filepath: str) -> str:
    """Load company policy document"""
    with open(filepath, 'r') as f:
        return f.read()

def audit_single_ticket(ticket: Dict, policy_document: str) -> Dict[str, Any]:
    """Audit a single ticket for policy compliance"""
    
    system_prompt = """You are a customer service quality auditor. Your job is to review support tickets 
    and ensure the resolution strictly follows company policy. Be thorough and critical - identify ANY 
    policy violations, missed policies, or incorrect actions."""
    
    # Extract key information
    ticket_id = ticket.get("ticket_id", "Unknown")
    ticket_timestamp = ticket.get("timestamp", "Unknown")
    customer_email = ticket.get("body", "No email body")
    subject = ticket.get("subject", "No subject")
    resolution = ticket.get("resolution_plan", {})
    
    # Build comprehensive prompt
    prompt = f"""Review this customer support ticket for policy compliance:

TICKET ID: {ticket_id}

CUSTOMER EMAIL:
Ticket Timestamp: {ticket_timestamp}
Subject: {subject}
Body: {customer_email}

RESOLUTION PLAN:
{json.dumps(resolution, indent=2)}

COMPLETE COMPANY POLICY DOCUMENT:
{policy_document}

AUDIT REQUIREMENTS:
1. Check if ALL relevant applicable policies were considered
2. Verify policy references in resolution are accurate
3. Ensure actions taken follow the policies exactly
4. Identify any policies that should have been applied but weren't
5. Check for any contradictions or errors in the resolution
6. Verify monetary values and timelines match policy requirements
7. Ensure denials are justified by specific policy violations

EXAMPLE CRITICAL CHECKS:
- Return window enforcement 
- Restocking fee application 
- Warranty period enforcement 
- Price match window 
- Requirements for items over or under a certain value
- Investigation period for lost packages 
- Order modification window 

Respond in JSON format:
{{
    "has_error": true/false,
    "severity": "critical/major/minor/none",
    "errors": [
        {{
            "type": "missed_policy/incorrect_action/wrong_timeline/incorrect_value/etc",
            "description": "Detailed description of the error",
            "policy_reference": "POL-XXX-###",
            "impact": "How this affects the customer or company"
        }}
    ],
    "warnings": [
        {{
            "description": "Less severe issues or potential improvements",
            "suggestion": "What could be done better"
        }}
    ],
    "audit_summary": "Overall assessment of the resolution",
    "policies_checked": ["List of all policies that were relevant to this case"],
    "policies_correctly_applied": ["Policies that were handled correctly"],
    "confidence": "high/medium/low confidence in this assessment"
}}"""

    try:
        response = call_llm(prompt, system_prompt)
        result = safe_json_parse(response, "object")
        
        if result and isinstance(result, dict):
            return result
        else:
            return {
                "has_error": True,
                "severity": "critical",
                "errors": [{"type": "audit_failure", "description": "Failed to parse audit response"}],
                "warnings": [],
                "audit_summary": "Audit failed",
                "confidence": "low"
            }
    except Exception as e:
        return {
            "has_error": True,
            "severity": "critical", 
            "errors": [{"type": "audit_error", "description": f"Audit exception: {str(e)}"}],
            "warnings": [],
            "audit_summary": "Audit crashed",
            "confidence": "low"
        }

def generate_audit_report(audit_results: List[Dict], output_path: str):
    """Generate human-readable audit report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"TICKET AUDIT REPORT\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 80 + "\n\n")
        
        # Summary statistics
        total_tickets = len(audit_results)
        error_tickets = sum(1 for r in audit_results if r['result'].get('has_error', False))
        critical_errors = sum(1 for r in audit_results if r['result'].get('severity') == 'critical')
        major_errors = sum(1 for r in audit_results if r['result'].get('severity') == 'major')
        minor_errors = sum(1 for r in audit_results if r['result'].get('severity') == 'minor')
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Tickets Audited: {total_tickets}\n")
        f.write(f"Tickets with Errors: {error_tickets} ({error_tickets/total_tickets*100:.1f}%)\n")
        f.write(f"  - Critical: {critical_errors}\n")
        f.write(f"  - Major: {major_errors}\n")
        f.write(f"  - Minor: {minor_errors}\n")
        f.write(f"Clean Tickets: {total_tickets - error_tickets}\n\n")
        
        # Error type breakdown
        error_types = {}
        for r in audit_results:
            for error in r['result'].get('errors', []):
                error_type = error.get('type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            f.write("ERROR TYPES\n")
            f.write("-" * 40 + "\n")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{error_type}: {count}\n")
            f.write("\n")
        
        # Detailed results for each ticket
        f.write("DETAILED RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # First show all tickets with errors
        error_results = [r for r in audit_results if r['result'].get('has_error', False)]
        if error_results:
            f.write("TICKETS WITH ERRORS\n")
            f.write("-" * 80 + "\n\n")
            
            for r in sorted(error_results, key=lambda x: x['result'].get('severity', 'none')):
                ticket_id = r['ticket_id']
                result = r['result']
                
                f.write(f"Ticket: {ticket_id}\n")
                f.write(f"Severity: {result.get('severity', 'unknown')}\n")
                f.write(f"Summary: {result.get('audit_summary', 'No summary')}\n")
                f.write(f"Confidence: {result.get('confidence', 'unknown')}\n\n")
                
                if result.get('errors'):
                    f.write("ERRORS:\n")
                    for i, error in enumerate(result['errors'], 1):
                        f.write(f"  {i}. Type: {error.get('type', 'unknown')}\n")
                        f.write(f"     Description: {error.get('description', 'No description')}\n")
                        if error.get('policy_reference'):
                            f.write(f"     Policy: {error['policy_reference']}\n")
                        if error.get('impact'):
                            f.write(f"     Impact: {error['impact']}\n")
                        f.write("\n")
                
                if result.get('warnings'):
                    f.write("WARNINGS:\n")
                    for warning in result['warnings']:
                        f.write(f"  - {warning.get('description', 'No description')}\n")
                        if warning.get('suggestion'):
                            f.write(f"    Suggestion: {warning['suggestion']}\n")
                    f.write("\n")
                
                f.write("-" * 80 + "\n\n")
        
        # Then show clean tickets summary
        clean_results = [r for r in audit_results if not r['result'].get('has_error', False)]
        if clean_results:
            f.write("CLEAN TICKETS (Summary)\n")
            f.write("-" * 80 + "\n\n")
            
            for r in clean_results:
                ticket_id = r['ticket_id']
                result = r['result']
                f.write(f"{ticket_id}: {result.get('audit_summary', 'Clean')}\n")
            
            f.write("\n")
        
        # Policy usage statistics
        f.write("POLICY USAGE STATISTICS\n")
        f.write("-" * 80 + "\n\n")
        
        all_policies_checked = {}
        correctly_applied = {}
        
        for r in audit_results:
            for policy in r['result'].get('policies_checked', []):
                all_policies_checked[policy] = all_policies_checked.get(policy, 0) + 1
            for policy in r['result'].get('policies_correctly_applied', []):
                correctly_applied[policy] = correctly_applied.get(policy, 0) + 1
        
        f.write("Most Frequently Checked Policies:\n")
        for policy, count in sorted(all_policies_checked.items(), key=lambda x: x[1], reverse=True)[:10]:
            correct_count = correctly_applied.get(policy, 0)
            accuracy = (correct_count / count * 100) if count > 0 else 0
            f.write(f"  {policy}: {count} times (correctly applied {accuracy:.1f}% of the time)\n")

def main():
    """Main audit function"""
    config = DatasetConfig()
    
    # Load tickets
    tickets_path = config.get_filepath(config.tickets_file)
    print(f"Loading tickets from {tickets_path}...")
    tickets = load_tickets(tickets_path)
    print(f"Loaded {len(tickets)} tickets")
    
    # Load policy
    policy_path = config.get_filepath(config.policy_file)
    print(f"Loading policy from {policy_path}...")
    policy_document = load_policy(policy_path)
    print(f"Policy document loaded ({len(policy_document)} characters)")
    
    # Audit each ticket
    print("\nAuditing tickets...")
    audit_results = []
    
    for i, ticket in enumerate(tickets):
        ticket_id = ticket.get('ticket_id', f'Unknown-{i}')
        print(f"Auditing {i+1}/{len(tickets)}: {ticket_id}", end='')
        
        result = audit_single_ticket(ticket, policy_document)
        audit_results.append({
            'ticket_id': ticket_id,
            'result': result
        })
        
        # Quick status
        if result.get('has_error'):
            print(f" - ERROR ({result.get('severity', 'unknown')})")
        else:
            print(" - CLEAN")
        
        # Optional: Add delay to avoid rate limiting
        # time.sleep(1)
    
    # Generate report
    report_path = config.get_filepath("ticket_audit_report.txt")
    print(f"\nGenerating audit report...")
    generate_audit_report(audit_results, report_path)
    print(f"Audit report saved to: {report_path}")
    
    # Also save raw JSON results for programmatic analysis
    json_path = config.get_filepath("ticket_audit_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_tickets': len(tickets),
                'tickets_with_errors': sum(1 for r in audit_results if r['result'].get('has_error', False))
            },
            'results': audit_results
        }, f, indent=2)
    print(f"Raw audit results saved to: {json_path}")
    
    # Print summary
    error_count = sum(1 for r in audit_results if r['result'].get('has_error', False))
    print(f"\nAudit complete: {error_count} tickets with errors out of {len(tickets)}")

if __name__ == "__main__":
    main() 