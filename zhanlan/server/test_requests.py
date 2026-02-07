import requests

r = requests.post(
    "http://192.168.24.49:8000/search",
    files={"file": open(r"D:/zhanlan/qurrey_data/334.jpg", "rb")}
)
print(r.json())
