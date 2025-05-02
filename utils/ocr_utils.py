import re
from difflib import get_close_matches

valid_rto_codes = {
    'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA', 'GJ', 'HP',
    'HR', 'JH', 'JK', 'KA', 'KL', 'LD', 'MH', 'ML', 'MN', 'MP', 'MZ', 'NL',
    'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}

def correct_common_ocr_errors(text):
    text = text.upper().replace('D', '0').replace('O', '0')
    text = text.replace('I', '1').replace('Z', '2')
    return re.sub(r'[^A-Z0-9]', '', text)

def fuzzy_correct_plate(text):
    text = correct_common_ocr_errors(text)
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$', text)
    if not match:
        return None
    state, district, series, num = match.groups()
    if state not in valid_rto_codes:
        corrected = get_close_matches(state, valid_rto_codes, n=1, cutoff=0.6)
        if corrected:
            state = corrected[0]
        else:
            return None
    return f"{state}{district}{series}{num}"

def is_valid_indian_plate(text):
    pattern = r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$'
    match = re.match(pattern, text)
    return match and match.group(1) in valid_rto_codes

def clean_plate_text(text):
    corrected = fuzzy_correct_plate(text)
    return corrected if corrected and is_valid_indian_plate(corrected) else None
