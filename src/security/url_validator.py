import ipaddress
from urllib.parse import urlparse

ALLOWED_DOMAINS = [
    "*.wikipedia.org",
    "*.arxiv.org", 
    "*.github.com",
    "news.ycombinator.com"
]

BLOCKED_IP_RANGES = [
    "127.0.0.0/8",      # Localhost
    "10.0.0.0/8",       # Private
    "172.16.0.0/12",    # Private
    "192.168.0.0/16",   # Private
    "169.254.0.0/16",   # Link-local
]

def is_url_safe(url):
    parsed = urlparse(url)
    
    # Check scheme
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Check domain whitelist
    if not any(domain_match(parsed.netloc, d) for d in ALLOWED_DOMAINS):
        return False
    
    # Check IP blacklist
    try:
        ip = socket.gethostbyname(parsed.netloc)
        ip_obj = ipaddress.ip_address(ip)
        
        for blocked_range in BLOCKED_IP_RANGES:
            if ip_obj in ipaddress.ip_network(blocked_range):
                return False
    except:
        pass
    
    return True