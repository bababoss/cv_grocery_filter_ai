
import requests
def sent_signal_to_pi(class_type):
    if class_type == "tometo":
        cls_id=1
    elif class_type == "lemon":
        cls_id=2
    x = requests.get(f"http://23.22.332.222:5000/home/{cls_id}")

    print(x.json())

