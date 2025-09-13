import random
import string

def generate_random_email(domain_list=None):
    if domain_list is None:
        domain_list = ["gmail.com", "yahoo.com", "outlook.com", "example.org"]
    username = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    domain = random.choice(domain_list)
    return f"{username}@{domain}"

def generate_random_mobile(country_code="+91"):
    # Generates a 10-digit Indian mobile number starting with 7-9
    first_digit = random.choice(["7", "8", "9"])
    number = first_digit + ''.join(random.choices(string.digits, k=9))
    return f"{country_code} {number}"

def generate_random_address():
    choices = ['Hyderabad', 'Pune', 'Noida', 'Bangalore', 'Gurgaon', 'Chennai', 'GIFT', 'Mumbai']
    return random.choice(choices)