import os
import signal
import subprocess
import unittest
from pathlib import Path
from pprint import pprint

import requests

IMAGE_DIR = Path("tests") / "images"


class TestImageOperations(unittest.TestCase):
    def test_add_stamp(self):
        url = 'http://127.0.0.1:5000/images/add_stamp'
        files = {
            'stamp_file1': (IMAGE_DIR / 'stamps1.png').open("rb"),
            "stamp_file2": (IMAGE_DIR / 'stamps2.jpg').open("rb")
        }
        data = {
            "stamp_file1_stamp_group_name": "test-stamp-group-1",
            "stamp_file2_stamp_group_name": "test-stamp-group-2"
        }
        response = requests.post(url, files=files, data=data)
        print("#####")
        pprint(response.json())
        print("#####")
        self.assertEqual(response.status_code, 200)

    def test_upload_image(self):
        url = "http://127.0.0.1:5000/images/upload"
        files = {"file": (IMAGE_DIR / "stamps1.png").open("rb")}
        response = requests.post(url, files=files)
        print("#####")
        pprint(response.json())
        print("#####")
        self.assertEqual(response.status_code, 200)

    def test_upload_multiple_images(self):
        url = "http://127.0.0.1:5000/images/upload"
        files = {
            "file_with_stamps1": (IMAGE_DIR / "stamps1.png").open("rb"),
            "file_with_stamps2": (IMAGE_DIR / "stamps2.jpg").open("rb"),
            "file_with_stamps3": (IMAGE_DIR / "stamps3.jpg").open("rb"),
            "file_without_stamps": (IMAGE_DIR / "no-stamps.png").open("rb"),
        }
        response = requests.post(url, files=files)
        print("#####")
        pprint(response.json())
        print("#####")
        self.assertEqual(response.status_code, 200)

    # Tests for future functions
    #
    # def test_login_with_google_account(self):
    #     url = 'http://127.0.0.1:5000/login/google'
    #     response = requests.get(url)
    #     self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    # start server process
    p = subprocess.Popen("./run.sh 127.0.0.1:5000", stdout=subprocess.PIPE,
                         shell=True, preexec_fn=os.setsid)

    # wait for startup

    while True:
        try:
            r = requests.get("http://127.0.0.1:5000/")
            assert r.status_code == 200
        except Exception:
            pass
        else:
            break

    # run tests
    try:
        unittest.main()
    except Exception as e:
        raise e
    finally:
        # kill server process
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
