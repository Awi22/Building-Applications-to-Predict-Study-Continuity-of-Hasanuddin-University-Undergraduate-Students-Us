import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'SKS':48, 'Semester':4, 'IPK':3})

print(r.json())