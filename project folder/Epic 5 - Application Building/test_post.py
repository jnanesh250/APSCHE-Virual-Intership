from app import app
c = app.test_client()
# representative inputs (reasonable values)
data = {
'gender':'Female',
'own_car':'Yes',
'own_real_estate':'Yes',
'income_type':'Working',
'education':'Higher education',
'family_status':'Married',
'housing_type':'House / apartment',
'annual_income':'75000',
'days_birth':'-12000',
'days_employed':'-3000',
'family_members':'3',
'emi_paid_off':'500',
'emi_past_dues':'50',
'number_of_loans':'2'
}
resp = c.post('/predict', data=data)
print('status', resp.status_code)
s = resp.get_data(as_text=True)
import re
m = re.search(r'Model probabilities:(.*?)</p>', s, re.S)
f = re.search(r'Computed feature vector.*?</pre>', s, re.S)
print('prob_block:', m.group(0) if m else 'none')
print('feat_block:', f.group(0) if f else 'none')
