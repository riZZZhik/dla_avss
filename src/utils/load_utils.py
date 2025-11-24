# Reference: https://gist.github.com/Yegorov/dc61c42aa4e89e139cd8248f59af6b3e

import json
import os
import sys
import urllib.parse as ul

sys.argv.append(".") if len(sys.argv) == 2 else None

base_url = (
    "https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key="
)
url = ul.quote_plus(sys.argv[1])
folder = sys.argv[2]
res = os.popen("wget -qO - {}{}".format(base_url, url)).read()
json_res = json.loads(res)
filename = ul.parse_qs(ul.urlparse(json_res["href"]).query)["filename"][0]
os.system("wget '{}' -P '{}' -O '{}'".format(json_res["href"], folder, filename))
