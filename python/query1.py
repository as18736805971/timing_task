import requests
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


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


def query_multiple_accounts(accounts: List[Tuple[str, str]]) -> Dict[str, int]:
    """查询多个账号的积分
    
    Args:
        accounts: 包含(手机号, 密码)元组的列表
        
    Returns:
        包含手机号和积分值的字典
    """
    results = {}
    
    for phone, password in accounts:
        try:
            print(f"正在查询账号: {phone}")
            login_result = login(phone, password)
            
            if not login_result.token:
                print(f"账号 {phone} 登录失败")
                results[phone] = -1  # 使用-1表示登录失败
                continue
                
            points_info = query_points(login_result)
            
            if points_info.get("code") == 200 and "data" in points_info:
                point_value = points_info["data"].get("pointValue", 0)
                results[phone] = point_value
                print(f"账号 {phone} 积分: {point_value}")
            else:
                print(f"账号 {phone} 查询积分失败: {points_info.get('msg', '未知错误')}")
                results[phone] = -2  # 使用-2表示查询积分失败
        except Exception as e:
            print(f"账号 {phone} 查询出错: {str(e)}")
            results[phone] = -3  # 使用-3表示发生异常
        
        # 添加延时，避免请求过于频繁
        time.sleep(1)
    
    return results


# 示例调用：
if __name__ == "__main__":
    # 定义多个账号
    accounts = [
        ("19821919082", "stars123456"),
        ("17821963941", "stars123456"),
        ("13524386537", "stars123456"),
        ("18321857093", "stars123456"),
        ("13524507357", "stars123456"),
    ]
    
    # 查询所有账号的积分
    results = query_multiple_accounts(accounts)
    
    # 打印汇总结果
    print("\n积分查询结果汇总:")
    print("-" * 30)
    print("手机号         | 积分")
    print("-" * 30)
    
    for phone, points in results.items():
        if points >= 0:
            print(f"{phone} | {points}")
        elif points == -1:
            print(f"{phone} | 登录失败")
        elif points == -2:
            print(f"{phone} | 查询积分失败")
        else:
            print(f"{phone} | 查询出错")