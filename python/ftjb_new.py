import requests
import random
import time
import string
import json
import base64
import logging
import os
import uuid
import re
import numpy as np
from PIL import Image
import io
import time
import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("foton_verification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FotonVerifier")

class FotonVerifier:
    """福田验证码验证和短信获取类"""

    BASE_URL = "https://czyl.foton.com.cn/ehomes-new/homeManager"
    USER_AGENT = "Feature_Alimighty/7.4.9 (iPhone; iOS 17.2.1; Scale/3.00)"



    def __init__(self, phone_number=None):
        """初始化验证器"""
        self.phone_number = phone_number
        self.successful_distances = []
        # 生成一个固定的Cookie，整个流程中使用相同的Cookie
        hwwafsesid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=20))
        hwwafsestime = str(int(time.time() * 1000))
        self.cookie = f"HWWAFSESID={hwwafsesid}; HWWAFSESTIME={hwwafsestime}"
        self.error_count = 0
        self.last_error_type = None
        # 创建调试目录
        self.debug_dirs = ["debug_background", "debug_slide", "debug_result"]
        for dir_name in self.debug_dirs:
            os.makedirs(dir_name, exist_ok=True)
        # 确保debug_result目录存在
        os.makedirs("debug_result", exist_ok=True)
        logger.info(f"初始化生成固定Cookie: {self.cookie}")


    def get_cookie(self):
        """获取当前Cookie"""
        return self.cookie

    def get_headers(self):
        """获取请求头"""
        # 生成请求数据的JSON字符串用于计算Content-Length
        data = {
            "smsCode": "",  # 实际值会在请求时填充
            "version_code": "562",
            "version_name": "7.4.9",
            "regSourceType": "0698",
            "deviceSystem": "17.2.1",
            "safeEnc": int(time.time() * 1000),
            "recommender": "",
            "device_type": "0",
            "idCard": "",
            "ip": "127.0.0.1",
            "tel": self.phone_number if self.phone_number else "",
            "dealerCode": "",
            "idType": "",
            "device_id": str(uuid.uuid4()).upper(),
            "device_model": "iPhone13,4"
        }
        # 计算实际请求体的UTF-8编码字节长度
        content_length = len(json.dumps(data, ensure_ascii=False).encode('utf-8'))

        return {
            "POST": "/ehomes-new/homeManager/regMember2nd HTTP/1.1",
            "Host": "czyl.foton.com.cn",
            "Content-Type": "application/json",
            "Content-Length": str(content_length),
            "Cookie": self.cookie,
            "Connection": "keep-alive",
            "Accept": "*/*",
            "User-Agent": self.USER_AGENT,
            "Accept-Language": "zh-Hans-CN;q=1",
            "Accept-Encoding": "gzip, deflate, br"
        }



    def log(self,message):
        """带时间戳的日志函数"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")

    def get_captcha_images(self):
        """调用第一个API获取验证码图片"""
        url = "https://czyl.foton.com.cn/ehomes-new/homeManager/getImg"

        try:
            self.log("开始获取验证码图片...")
            start_time = time.time()
            response = requests.post(url)
            response.raise_for_status()  # 确保请求成功
            data = response.json()
            elapsed_time = time.time() - start_time

            self.log(f"成功获取验证码图片 (用时: {elapsed_time:.2f}秒)")
            return data
        except Exception as e:
            self.log(f"获取验证码图片失败: {e}")
            return None

    def base64_to_image(self,base64_string):
        """将base64字符串转换为PIL图像对象"""
        try:
            self.log("开始将base64转换为图像...")
            start_time = time.time()

            # 如果base64字符串包含头部信息，需要去除
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # 解码base64字符串
            image_data = base64.b64decode(base64_string)

            # 创建PIL图像对象
            image = Image.open(io.BytesIO(image_data))

            elapsed_time = time.time() - start_time
            self.log(f"成功转换base64为图像 (用时: {elapsed_time:.2f}秒)")
            return image
        except Exception as e:
            self.log(f"base64转图像失败: {e}")
            return None

    def get_leftmost_transparent_x(self,image):
        """获取图像中最左边的像素的X坐标"""
        try:
            self.log("开始识别最左边像素...")
            start_time = time.time()

            # 确保图像有Alpha通道
            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            # 转换为numpy数组
            img_array = np.array(image)

            # 获取Alpha通道
            alpha = img_array[:, :, 3]

            # 图像尺寸
            height, width = alpha.shape
            self.log(f"图像尺寸: {width}x{height}")

            # 从左到右逐列扫描
            for x in range(width):
                for y in range(height):
                    if alpha[y, x] < 255:  # 找到或半像素
                        elapsed_time = time.time() - start_time
                        self.log(f"找到最左边像素，X坐标: {x} (用时: {elapsed_time:.2f}秒)")
                        return x

            elapsed_time = time.time() - start_time
            self.log(f"图像中没有像素 (用时: {elapsed_time:.2f}秒)")
            return None
        except Exception as e:
            self.log(f"识别像素失败: {e}")
            return None

    def verify_captcha(self,img_key, x_distance):
        """调用第二个API验证验证码"""
        url = "https://czyl.foton.com.cn/ehomes-new/homeManager/verifyImgCode"

        payload = {
            "yzmimgKey": img_key,
            "xDistance": x_distance
        }
        time.sleep(random.uniform(1, 2))
        try:
            self.log(f"开始验证验证码，参数: yzmimgKey={img_key}, xDistance={x_distance}...")
            start_time = time.time()
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            elapsed_time = time.time() - start_time

            self.log(f"验证结果: {result} (用时: {elapsed_time:.2f}秒)")
            return result
        except Exception as e:
            self.log(f"验证失败: {e}")
            return None

    def get_img_key(self):
        self.log("开始执行验证码识别流程")
        total_start_time = time.time()

        # 步骤1: 获取验证码图片
        captcha_data = self.get_captcha_images()
        if not captcha_data:
            self.log("无法继续，获取验证码图片失败")
            return

        # 提取需要的数据
        ori_copy_image = captcha_data.get('oriCopyImage')
        img_key = captcha_data.get('imgXKey')

        if not ori_copy_image or not img_key:
            self.log("API响应中缺少必要的数据")
            return

        # 步骤2: 将base64字符串转换为图像
        image = self.base64_to_image(ori_copy_image)
        if image is None:
            self.log("无法继续，base64转图像失败")
            return

        # 步骤3: 获取最左边像素的X坐标
        x_coordinate = self.get_leftmost_transparent_x(image)
        if x_coordinate is None:
            self.log("无法继续，未能识别像素")
            return

        # 步骤4: 调用验证API
        verify_result = self.verify_captcha(img_key, x_coordinate)

        # 计算总用时
        total_elapsed_time = time.time() - total_start_time

        # 输出完整结果
        self.log(f"完整流程结果:")
        self.log(f"imgXKey: {img_key}")
        self.log(f"识别到的X坐标: {x_coordinate}")
        self.log(f"验证结果: {verify_result}")
        self.log(f"总用时: {total_elapsed_time:.2f}秒")
        return img_key,x_coordinate


    def get_sms_code(self, img_token, x_distance):
        """获取短信验证码"""
        if not self.phone_number:
            logger.error("未提供手机号，无法获取短信验证码")
            return {"error": "未提供手机号"}

        # 更新请求URL
        url = f"{self.BASE_URL}/getSmsCodeNew"

        # 更新请求数据结构
        data = {
            "yzmimgKey": img_token,
            "type": "2",  # 根据抓包信息添加type字段
            "msgyzm": str(x_distance),
            "tel": self.phone_number
        }

        try:
            logger.info(f"尝试获取短信验证码，手机号: {self.phone_number}")
            logger.debug(f"短信请求数据: {data}")

            response = requests.post(url, headers=self.get_headers(), json=data, timeout=10)
            logger.info(f"短信请求响应状态: {response.status_code}")
            logger.info(f"短信请求响应内容: {response.text}")

            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("code") == 200:
                    logger.info("短信验证码发送成功")
                return response_data

            return {"error": f"请求失败，状态码: {response.status_code}"}

        except requests.RequestException as e:
            logger.error(f"短信请求异常: {str(e)}")
            return {"error": f"请求异常: {str(e)}"}

    def release_number(self, phone_number):
        """释放手机号码资源"""
        logger.info(f"释放手机号: {phone_number}")
        # 这里可以添加实际的释放逻辑，比如调用API或清理资源
        return {"code": 200, "message": "号码释放成功"}

    def complete_verification_and_get_sms(self):
        """完成验证并获取短信验证码 - 增强版"""
        img_token,successful_distance = self.get_img_key()
        sms_result = self.get_sms_code(img_token, successful_distance)

        if sms_result.get("code") == 200:
            logger.info("获取短信验证码成功！")
            # 获取到验证码后立即返回结果
            return {
                "status": "success",
                "code": sms_result.get("code"),
                "message": sms_result.get("message"),
                "verification_code": sms_result.get("code")  # 假设验证码在code字段中
            }
        else:
            logger.error(f"获取短信验证码失败: {sms_result}")

            # 根据错误信息调整策略
            error_msg = sms_result.get("msg", "")
            if "频繁" in error_msg:
                time.sleep(random.uniform(5, 8))
            else:
                time.sleep(random.uniform(2, 3))

            # 如果错误代码是500，释放号码并重新开始流程
            if sms_result.get("code") == 500:
                logger.info("释放号码并重新开始流程")
                # 释放号码逻辑
                release_result = self.release_number(self.phone_number)
                if release_result.get("code") == 200:
                    logger.info(f"成功释放手机号: {self.phone_number}")
                else:
                    logger.error(f"释放手机号失败: {release_result}")

                # 等待一段时间后重新开始流程
                time.sleep(random.uniform(3, 5))

        return {"error": f"失败"}


class YeziPlatform:
    """椰子平台API封装"""

    def __init__(self, username, password, project_id):
        self.username = username
        self.password = password
        self.project_id = project_id
        self.token = None
        self.base_url = "http://api.sqhyw.net:90/api"

        # 登录获取token
        self.login()

    def login(self):
        """登录获取token"""
        url = f"{self.base_url}/logins"
        params = {
            'username': self.username,
            'password': self.password
        }

        try:
            logger.info(f"尝试登录椰子平台... 用户名: {self.username}")
            response = requests.get(url, params=params, timeout=10)
            logger.info(f"登录响应状态码: {response.status_code}")
            logger.info(f"登录响应内容: {response.text}")

            if response.status_code == 200:
                login_data = response.json()
                self.token = login_data.get("token")
                if self.token:
                    logger.info(f"登录成功，获取到token: {self.token}")
                else:
                    logger.error(f"登录响应成功但获取token失败: {login_data}")
            else:
                logger.error(f"登录失败，状态码: {response.status_code}")
        except Exception as e:
            logger.error(f"登录请求异常: {str(e)}")

    def get_phone_number(self, special=None, phone_num=None, scope=None):
        """获取手机号"""
        if not self.token:
            logger.error("未登录或token无效，尝试重新登录")
            self.login()
            if not self.token:
                return None

        url = f"{self.base_url}/get_mobile"
        params = {
            'token': self.token,
            'project_id': self.project_id,
            # 'scope_black':"192"
            # 'scope': "130,190,136,131"
        }

        # 添加可选参数
        if special:
            params['special'] = special
        if phone_num:
            params['phone_num'] = phone_num
        if scope:
            params['scope'] = scope

        try:
            logger.info(f"尝试获取手机号... 项目ID: {self.project_id}")
            logger.debug(f"请求参数: {params}")

            response = requests.get(url, params=params, timeout=10)
            logger.info(f"获取手机号响应状态码: {response.status_code}")
            logger.info(f"获取手机号响应内容: {response.text}")

            if response.status_code == 200:
                data = response.json()
                # 检查 message 而不是 code
                if data.get("message") == "ok" and data.get("mobile"):
                    mobile = data.get("mobile")
                    logger.info(f"成功获取手机号: {mobile}")
                    return mobile
                else:
                    logger.error(f"获取手机号失败: {data.get('message', '未知错误')}")
            return None
        except Exception as e:
            logger.error(f"获取手机号请求异常: {str(e)}")
            return None

    def free_mobile(self, phone_number, special=None):
        """释放手机号"""
        if not self.token:
            logger.error("未登录或token无效，尝试重新登录")
            self.login()
            if not self.token:
                return False

        url = f"{self.base_url}/free_mobile"
        params = {
            'token': self.token,
            'project_id': self.project_id,
            'phone_num': phone_number
        }

        # 添加可选参数
        if special:
            params['special'] = special

        try:
            logger.info(f"尝试释放手机号: {phone_number}")
            logger.debug(f"释放请求参数: {params}")

            response = requests.get(url, params=params, timeout=10)
            logger.info(f"释放手机号响应状态码: {response.status_code}")
            logger.info(f"释放手机号响应内容: {response.text}")

            if response.status_code == 200:
                data = response.json()
                if data.get("message") == "ok":
                    logger.info(f"成功释放手机号: {phone_number}")
                    return True
                else:
                    logger.error(f"释放手机号失败: {data.get('message', '未知错误')}")
            return False
        except Exception as e:
            logger.error(f"释放手机号请求异常: {str(e)}")
            return False

    def get_sms_content(self, phone_number, special=None, max_attempts=12, interval=5):
        """获取短信内容"""
        if not self.token:
            logger.error("未登录或token无效，尝试重新登录")
            self.login()
            if not self.token:
                return {"error": "获取token失败，无法获取短信"}

        url = f"{self.base_url}/get_message"
        params = {
            'token': self.token,
            'project_id': self.project_id,
            'phone_num': phone_number
        }

        # 添加可选参数
        if special:
            params['special'] = special

        logger.info(f"开始获取手机号 {phone_number} 的短信内容")
        logger.debug(f"请求参数: {params}")

        for attempt in range(max_attempts):
            try:
                logger.info(f"第 {attempt+1} 次尝试获取短信内容")
                response = requests.get(url, params=params, timeout=10)
                logger.info(f"获取短信响应状态码: {response.status_code}")
                logger.info(f"获取短信响应内容: {response.text}")

                if response.status_code == 200:
                    data = response.json()
                    # 检查是否有短信内容
                    if data.get("message") == "ok" and (data.get("data") and len(data["data"]) > 0 or data.get("code")):
                        sms_content = ""
                        verification_code = data.get("code", "")

                        if data.get("data") and len(data["data"]) > 0:
                            sms_content = data["data"][0].get("modle", "")

                        # 如果没有直接提供验证码，尝试从短信内容中提取
                        if not verification_code and sms_content:
                            code_patterns = [
                                r'验证码[是为:]?\s*([0-9]{4,6})',
                                r'码[是为:]?\s*([0-9]{4,6})',
                                r'([0-9]{4,6})[^0-9]'
                            ]
                            for pattern in code_patterns:
                                match = re.search(pattern, sms_content)
                                if match:
                                    verification_code = match.group(1)
                                    break

                        result = {
                            "success": True,
                            "sms_content": sms_content,
                            "verification_code": verification_code
                        }

                        logger.info(f"成功获取短信内容: {sms_content}")
                        if verification_code:
                            logger.info(f"验证码: {verification_code}")
                        else:
                            logger.warning(f"未能从短信中提取到验证码")

                        return result
                    elif "等待" in data.get("message", "") or (data.get("message") == "ok" and (not data.get("data") or len(data["data"]) == 0)):
                        logger.info("短信尚未收到，等待下一次查询")
                    else:
                        logger.warning(f"获取短信失败: {data.get('message', '未知错误')}")
                else:
                    logger.error(f"获取短信请求失败，状态码: {response.status_code}")

                # 如果达到最大尝试次数，返回失败
                if attempt == max_attempts - 1:
                    return {"error": "获取短信超时，达到最大尝试次数"}

                # 等待一段时间后再次尝试
                time.sleep(interval)

            except Exception as e:
                logger.error(f"获取短信请求异常: {str(e)}")
                if attempt == max_attempts - 1:
                    return {"error": f"获取短信异常: {str(e)}"}
                time.sleep(interval)

        return {"error": "获取短信超时"}


def main():
    """主函数"""
    # 配置参数
    username = "18736805971"#椰子账号
    password = "zxcvbnm0129."#椰子密码
    project_id = "865765"
    target_success_count = 1  # 需要成功修改密码的手机号数量
    current_success_count = 0
    successful_phones = []  # 用于存储成功完成的手机号

    # 初始化椰子平台
    yezi = YeziPlatform(username, password, project_id)

    while current_success_count < target_success_count:
        try:
            # 获取手机号
            phone_number = yezi.get_phone_number()
            if not phone_number:
                logger.error("获取手机号失败")
                time.sleep(3)
                continue

            logger.info(f"获取到手机号: {phone_number}")

            # 初始化验证器
            verifier = FotonVerifier(phone_number)
            result = verifier.complete_verification_and_get_sms()

            logger.info("短信发送结果:")
            logger.info(result)

            if result.get("code") == 200:
                # 获取短信验证码内容
                logger.info("开始获取短信验证码内容...")
                sms_result = yezi.get_sms_content(phone_number)

                if sms_result.get("success"):
                    logger.info(f"成功获取短信内容: {sms_result.get('sms_content')}")
                    logger.info(f"提取的验证码: {sms_result.get('verification_code')}")
                    logger.info(f"验证成功的手机号: {phone_number}")

                    # 构建修改密码请求
                    update_data = {
                        "memberId": "",
                        "password": "zhao12345678",#要修改的手机号
                        "smsCode": sms_result.get("verification_code"),
                        "ip": "127.0.0.1",
                        "tel": phone_number
                    }

                    # 发送修改密码请求
                    update_response = requests.post(
                        f"{verifier.BASE_URL}/userUpdatePassword",
                        headers=verifier.get_headers(),
                        json=update_data,
                        timeout=10
                    )

                    update_result = update_response.json()
                    logger.info(f"密码修改响应: {update_result}")
                    successful_phones.append(phone_number)  # 记录成功的手机号
                    current_success_count += 1
                    logger.info(f"当前成功数量: {current_success_count}/{target_success_count}")
                else:
                    logger.error(f"获取短信内容失败: {sms_result.get('error')}")
                    yezi.free_mobile(phone_number)
                    time.sleep(3)
            else:
                logger.warning(f"验证失败，释放手机号并重试: {result}")
                yezi.free_mobile(phone_number)
                time.sleep(3)

        except Exception as e:
            logger.error(f"验证过程出错: {str(e)}")
            if phone_number:
                yezi.free_mobile(phone_number)
            time.sleep(3)

    logger.info(f"已完成 {target_success_count} 个手机号的密码修改")
    if successful_phones:
        logger.info(f"成功完成的手机号列表: {','.join(successful_phones)}")






if __name__ == "__main__":
    main()
