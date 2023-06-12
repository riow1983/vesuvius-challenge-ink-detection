##########################################
################ utils.py ################
##########################################
import numpy as np
import requests
import json

def rle(output, thres):
    output = output.flatten()
    flat_img = np.where(output > thres, 1, 0).astype(np.uint8)
    print("flat_img.shape: ", flat_img.shape)
    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    return " ".join(map(str, sum(zip(starts_ix, lengths), ())))


def send_line_notification(message, line_json_path):
    f = open(line_json_path, "r")
    json_data = json.load(f)
    line_token = json_data["kagglePush"]
    endpoint = 'https://notify-api.line.me/api/notify'
    message = "\n{}".format(message)
    payload = {'message': message}
    headers = {'Authorization': 'Bearer {}'.format(line_token)}
    requests.post(endpoint, data=payload, headers=headers)