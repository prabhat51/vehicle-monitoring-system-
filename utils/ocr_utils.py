import re
from difflib import get_close_matches
from paddleocr import PaddleOCR

ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

valid_rto_codes = {'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN', 'GA',
                   'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LD', 'MH', 'ML',
                   'MN', 'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN',
                   'TR', 'TS', 'UK', 'UP', 'WB'}

def correct_common_ocr_errors(text):
    text = text.upper()
    text = text.replace('D', '0').replace('O', '0')
    text = text.replace('I', '1').replace('Z', '2')
    return re.sub(r'[^A-Z0-9]', '', text)

def fuzzy_correct_plate(text):
    text = correct_common_ocr_errors(text)
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$', text)
    if not match:
        return None
    state, district, series, num = match.groups()
    if state not in valid_rto_codes:
        matches = get_close_matches(state, valid_rto_codes, n=1, cutoff=0.6)
        state = matches[0] if matches else None
    return f"{state}{district}{series}{num}" if state else None

def is_valid_indian_plate(text):
    match = re.match(r'^([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{3,4})$', text)
    return bool(match and match.group(1) in valid_rto_codes)

def clean_plate_text(text):
    plate = fuzzy_correct_plate(text)
    return plate if plate and is_valid_indian_plate(plate) else None
