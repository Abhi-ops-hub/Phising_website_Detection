import urllib.parse
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time

def extract_features(url, default_sample):
    """
    Given a URL, attempts to extract the 50 features used by the model.
    Falls back to `default_sample` for features that are too complex to compute in real time.
    """
    # Ensure scheme
    if not url.startswith('http'):
        url = 'http://' + url
        
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    
    # 1. Lexical features
    features = default_sample.copy()
    
    features['URLLength'] = len(url)
    features['DomainLength'] = len(domain)
    
    # Check if domain relies on IP
    ip_pattern = re.compile(r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])')
    features['IsDomainIP'] = 1 if ip_pattern.search(domain) else 0
    
    features['IsHTTPS'] = 1 if parsed.scheme == 'https' else 0
    features['NoOfSubDomain'] = max(0, len(domain.split('.')) - 2)
    
    features['NoOfLettersInURL'] = sum(c.isalpha() for c in url)
    features['NoOfDegitsInURL'] = sum(c.isdigit() for c in url)
    
    if len(url) > 0:
        features['LetterRatioInURL'] = features['NoOfLettersInURL'] / len(url)
        features['DegitRatioInURL'] = features['NoOfDegitsInURL'] / len(url)
    else:
        features['LetterRatioInURL'] = 0
        features['DegitRatioInURL'] = 0
        
    features['NoOfEqualsInURL'] = url.count('=')
    features['NoOfQMarkInURL'] = url.count('?')
    features['NoOfAmpersandInURL'] = url.count('&')
    
    special_chars = "!@#$%^*()_+-=[]{};:'\",.<>/|`~"
    features['NoOfOtherSpecialCharsInURL'] = sum(1 for c in url if c in special_chars and c not in '=?&')
    if len(url) > 0:
        features['SpacialCharRatioInURL'] = (features['NoOfEqualsInURL'] + features['NoOfQMarkInURL'] + features['NoOfAmpersandInURL'] + features['NoOfOtherSpecialCharsInURL']) / len(url)
        
    # Attempt to fetch content
    try:
        response = requests.get(url, timeout=5, allow_redirects=True)
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # Content Features
        lines = html.split('\n')
        features['LineOfCode'] = len(lines)
        if len(lines) > 0:
            features['LargestLineLength'] = max(len(line) for line in lines)
            
        features['HasTitle'] = 1 if soup.title else 0
        features['HasFavicon'] = 1 if soup.find('link', rel=lambda x: x and 'icon' in x.lower()) else 0
        features['Robots'] = 1 if soup.find('meta', attrs={'name': 'robots'}) else 0
        
        # Forms, inputs, iframes
        features['NoOfiFrame'] = len(soup.find_all('iframe'))
        forms = soup.find_all('form')
        
        has_ext_form = 0
        has_submit = 0
        for form in forms:
            action = form.get('action', '').lower()
            if action and not action.startswith('/') and not domain in action:
                has_ext_form = 1
            if form.find(['input', 'button'], {'type': 'submit'}):
                has_submit = 1
                
        features['HasExternalFormSubmit'] = has_ext_form
        features['HasSubmitButton'] = has_submit
        features['HasHiddenFields'] = 1 if soup.find('input', {'type': 'hidden'}) else 0
        features['HasPasswordField'] = 1 if soup.find('input', {'type': 'password'}) else 0
        
        features['NoOfImage'] = len(soup.find_all('img'))
        features['NoOfCSS'] = len(soup.find_all('link', {'rel': 'stylesheet'}))
        features['NoOfJS'] = len(soup.find_all('script'))
        
    except Exception as e:
        # If fetch fails, the site is dead or unreachable. 
        # Zero out all HTML features so the model sees it as lacking any normal web content.
        features['LineOfCode'] = 0
        features['LargestLineLength'] = 0
        features['HasTitle'] = 0
        features['HasFavicon'] = 0
        features['Robots'] = 0
        features['NoOfiFrame'] = 0
        features['HasExternalFormSubmit'] = 0
        features['HasSubmitButton'] = 0
        features['HasHiddenFields'] = 0
        features['HasPasswordField'] = 0
        features['NoOfImage'] = 0
        features['NoOfCSS'] = 0
        features['NoOfJS'] = 0
        print(f"Error fetching URL: {e}")
        
    # Return as a pandas DataFrame matching model structure
    return pd.DataFrame([features])
