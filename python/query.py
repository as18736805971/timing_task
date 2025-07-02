import requests
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoginResult:
    token: Optional[str] = None
    memberId: Optional[str] = None
    uid: Optional[str] = None
    phone: Optional[str] = None


def login(phone: str, password: str) -> LoginResult:
    url = "https://czyl.foton.com.cn/ehomes-new/homeManager/getLoginMember"
    body = {
        "version_name": "",
        "checkCode": "",
        "redisCheckCodeKey": "",
        "deviceSystem": "17.2.1",
        "version_auth": "VLQJq+FS9pSFabxQgcaDhA==",
        "device_type": "0",
        "password": password,
        "ip": "127.0.0.1",
        "device_id": "7AD11E8D-EC8F-4194-939F-BE62704DCFB7",
        "version_code": "0",
        "name": phone,
        "device_model": "iPhone13,4"
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Feature_Alimighty/7.4.9 (iPhone; iOS 17.2.1; Scale/3.00)"
    }

    response = requests.post(url, json=body, headers=headers, timeout=5)
    resp_json = response.json()

    if resp_json.get("code") != 200:
        return LoginResult()

    data = resp_json.get("data", {})
    return LoginResult(
        token=data.get("token"),
        memberId=data.get("memberID"),
        uid=data.get("uid"),
        phone=data.get("tel")
    )


def query_points(login: LoginResult) -> dict:
    url = "https://czyl.foton.com.cn/ehomes-new/homeManager/api/Member/findMemberPointsInfo"
    body = {
        "memberId": login.memberId,
        "userId": login.uid,
        "userType": "61",
        "uid": login.uid,
        "mobile": login.phone,
        "tel": login.phone,
        "phone": login.phone,
        "brandName": "",
        "seriesName": "",
        "token": login.token,
        "safeEnc": int(time.time() * 1000) - 2022020200,
        "businessId": 1
    }

    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "Connection": "Keep-Alive",
        "app-key": "7918d2d1a92a02cbc577adb8d570601e72d3b640",
        "app-token": "58891364f56afa1b6b7dae3e4bbbdfbfde9ef489",
        "User-Agent": "web",
        "Accept-Encoding": "gzip"
    }

    response = requests.post(url, json=body, headers=headers, timeout=5)
    return response.json()


# 示例调用：
if __name__ == "__main__":
    phone = "18736805971"
    password = "18736805971"
    login_result = login(phone, password)

    if login_result.token:
        points_info = query_points(login_result)
        print(points_info)
    else:
        print("登录失败")