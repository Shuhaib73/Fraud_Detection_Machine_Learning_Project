import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ['Administrator', 'Developer']
usernames = ['Admin', 'Shuhaib']
passwords = ['admin@@', 'shuhaib@@']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"

with open(file_path, 'wb') as file:
    pickle.dump(hashed_passwords, file)