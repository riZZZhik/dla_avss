import json
import os
import sys
import urllib.parse as ul

from huggingface_hub import hf_hub_download


def download_custom_dataset():
    """
    Downloads custom dataset from url sys.argv[2] into current directory
    Reference: https://gist.github.com/Yegorov/dc61c42aa4e89e139cd8248f59af6b3e
    """
    base_url = (
        "https://cloud-api.yandex.net:443/v1/disk/public/resources/download?public_key="
    )
    url = ul.quote_plus(sys.argv[2])
    res = os.popen("wget -qO - {}{}".format(base_url, url)).read()
    json_res = json.loads(res)
    filename = ul.parse_qs(ul.urlparse(json_res["href"]).query)["filename"][0]
    os.system("wget '{}' -O '{}'".format(json_res["href"], filename))


def download_checkpoint():
    """
    Downloads best model checkpoint into folder = sys.argv[2]
    from repo = = sys.argv[3]
    and model_file = sys.argv[4]
    """
    hf_hub_download(
        repo_id=sys.argv[3],
        filename=sys.argv[4],
        local_dir=sys.argv[2],
    )


def calc_metrics():
    """
    ...
    """
    ...


if __name__ == "__main__":
    func_name = sys.argv[1]
    globals()[func_name]()
