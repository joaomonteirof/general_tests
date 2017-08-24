import requests

#to run: python -m cProfile -o profiling_test.profile profiling_test.py
#to profile: snakeviz profiling_test.profile
#conda install libgcc

r = requests.get('http://api.open-notify.org/iss-now.json')
server_response = r.json()
iss_location = server_response['iss_position']

print('The International Space Station is currently at {}, {}'.format(iss_location['latitude'], iss_location['longitude']))
