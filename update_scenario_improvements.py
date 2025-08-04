"""
Script to update factory_3.py with improved scenario templates based on user feedback.

Key improvements:
1. Added exchange scenarios and policies
2. Split return scenarios into clear accept/deny cases  
3. Made date ranges more realistic (centered around policy windows)
4. Updated marketplace scenarios to use specific retailer names
5. Fixed wrong item shipped to have no time limit
6. Added more balanced scenario distribution
"""

print("""
MANUAL STEPS TO UPDATE factory_3.py:

1. The exchange policies have already been added to create_policy_graph()

2. The SCENARIO_DIMENSIONS have been updated to include exchange_request and rebalanced

3. The RESOLUTION_ACTIONS have been updated to include:
   - process_exchange
   - deny_exchange  
   - send_return_label

4. NOW REPLACE the entire create_scenario_templates() function with the improved version from improved_templates.py

5. Key improvements in the new templates:
   - Split return scenarios into clear ACCEPT (within window) and DENY (outside window) cases
   - Added exchange scenarios (EXCHANGE-001 through EXCHANGE-003)
   - More realistic date ranges centered around policy windows
   - Marketplace price match scenarios now use specific retailers (Amazon, eBay)
   - Wrong item shipped (SHIP-002) now allows up to 200 days since it's merchant error
   - Removed duplicate/confusing scenarios
   - All scenarios have pre-validated policy lists

6. The improved templates ensure:
   - Fewer ambiguous date ranges that could go either way
   - Clear separation of accept/deny scenarios
   - More realistic customer behavior (e.g., not saying "marketplace seller")
   - Better distribution of outcomes (not just denials)

To apply: Copy the entire create_scenario_templates() function from improved_templates.py 
and replace the existing one in factory_3.py
""") 