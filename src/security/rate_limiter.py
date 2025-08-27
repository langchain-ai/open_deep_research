from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
        
    def check_limit(self, user_id, limits):
        now = datetime.now()
        user_requests = self.requests[user_id]
        
        # Clean old requests
        self.requests[user_id] = [
            req for req in user_requests 
            if now - req < timedelta(hours=1)
        ]
        
        # Check limits
        if len(self.requests[user_id]) >= limits['per_hour']:
            return False
            
        self.requests[user_id].append(now)
        return True

# Configuration
RATE_LIMITS = {
    'requests_per_hour': 100,
    'max_tokens_per_request': 4000,
    'max_cost_per_hour': 10.0  # USD
}