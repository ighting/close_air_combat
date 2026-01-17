from socket import *
from struct import unpack, pack
from threading import Thread, Lock
import pandas as pd
import time
import json

# 第一步：host发送握手数据HandShakeData1、HandShakeData2、HandShakeData3，然后client响应握手数据
# 主机host（模拟器）和客户端client（Tacview）建立通信：握手数据
# 1）socket通信建立后，host先发送底层协议及版本XtraLib.Stream.0
# 2）再发送 Tacview高层协议及版本Tacview.RealTimeTelemetry.0
# 3）再发送 主机名称Host name，仅仅为了展示使用
# 4）每一行以"\n"结束，数据包以"\0"结束。
# 5）通信建立之后，客户端client会发送：握手数据
# 格式：
# --XtraLib.Stream.0␊
# --Tacview.RealTimeTelemetry.0␊
# --Client username␊
# --0␀
HandShakeData1 = 'XtraLib.Stream.0\n'
HandShakeData2 = 'Tacview.RealTimeTelemetry.0\n'
HandShakeData3 = 'alpha_dog_fight\n'

# 第二步：host发送头文件和数据格式
# 主机host（模拟器）和客户端client（Tacview）确认连接后，发送数据给client
# 数据格式：acmi format 2.x
# 头文件 TelFileHeader告诉Tacview所需的格式。
# --FileType=text/acmi/tacview␊
# --FileVersion=2.2␊
# TelReferenceTimeFormat，id 0指定的全局对象的属性ReferenceTime。换句话说：这一行定义了用于整个飞行记录的基准/参考时间。
# TelDataFormat 数据格式
# 格式一：T = Longitude | Latitude | Altitude
# 格式二：T = Longitude | Latitude | Altitude | U | V
# 格式三：T = Longitude | Latitude | Altitude | Roll | Pitch | Yaw
# 格式四：T = Longitude | Latitude | Altitude | Roll | Pitch | Yaw | U | V | Heading
# 备注：U&V代表原生（native）x和y
# Type：
# Name：飞机型号名称
# color：Red, Orange, Yellow (Tacview 1.8.8), Green, Cyan (Tacview 1.8.8), Blue, Violet
# Coalition:Allies
# 其他可选参数：HDG 航向角、AGL 飞机离地高度、IAS 指示空速、TAS 真空速等等
TelFileHeader = "FileType=text/acmi/tacview\nFileVersion=2.1\n"
TelReferenceTimeFormat = '0,ReferenceTime=%Y-%m-%dT%H:%M:%SZ\n'
TelDataFormat = '#%.2f\n3000102,T=%.7f|%.7f|%.7f|%.1f|%.1f|%.1f,AGL=%.3f,TAS=%.3f,CAS=%.3f,Type=Air+FixedWing,Name=F16,Color=Red,Coalition=Allies\n'
TelDataFormat_target = '#%.2f\n3000102,T=%.7f|%.7f|%.7f,Type=Air+FixedWing,Name=F16,Color=Blue,Coalition=Allies\n'
TelDataFormat_less = '#%.2f\n%s,T=%.7f|%.7f|%.7f|%.1f|%.1f|%.1f,Type=Air+FixedWing,Name=F16,Color=%s,Coalition=Allies\n'
# 定义TCP IP通信端口
LOCALPORT = 58008
LOCALIP = '127.0.0.1'
# so = socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP)


class TacviewClient:
    """Tacview实时通信客户端"""

    def __init__(self, serverip='127.0.0.1', serverport=15502, time_str='2023 10 29 12 00 00'):
        """
        初始化Tacview客户端

        Args:
            serverip: 服务器IP地址
            serverport: 服务器端口
            time_str: 参考时间字符串，格式: 'YYYY MM DD HH MM SS'
        """
        self.serverip = serverip
        self.serverport = serverport
        self.time_str = time_str
        self.so = None
        self.connected = False
        self._aircraft_objects = {}  # 存储已注册的飞机对象 {uid: id}
        self._missile_objects = {}   # 存储已注册的导弹对象 {uid: id}
        self._next_id = 100000       # 下一个可用ID
        self._start_time = time.time()
        self._lock = Lock()

    def connect(self):
        """建立与Tacview的连接"""
        try:
            with socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) as so:
                so.bind((self.serverip, self.serverport))
                so.listen()
                print(f"Tacview监听中: {self.serverip}:{self.serverport}")
                print("等待Tacview客户端连接...")
                ss = so.accept()
                print("Tacview客户端已连接")
                self.so, addr = ss
                print('发送握手数据: ')

                # 发送三条握手数据，并以"\0"结束
                self.so.send(HandShakeData1.encode('utf-8'))
                self.so.send(HandShakeData2.encode('utf-8'))
                self.so.send(HandShakeData3.encode('utf-8'))
                self.so.send(b'\x00')

                # 接收client的响应握手数据
                data = self.so.recv(1024)
                print('等待握手响应: ')
                print(data)

                # 发送文件头
                self.so.send(TelFileHeader.encode('utf-8'))

                # 发送参考时间
                ti = time.strptime(self.time_str, "%Y %m %d %H %M %S")
                t = time.strftime(TelReferenceTimeFormat, ti).encode('utf-8')
                print(t)
                self.so.send(t)

                self.connected = True
                print('Tacview连接已建立，准备传输数据')
                return True

        except Exception as e:
            print(f"Tacview连接失败: {e}")
            return False

    def _get_next_id(self):
        """获取下一个可用ID"""
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _get_elapsed_time(self):
        """获取从连接开始经过的时间（秒）"""
        return time.time() - self._start_time

    def update_aircraft(self, aircraft_uid, color, longitude, latitude, altitude,
                        roll, pitch, heading, tas=None, agl=None):
        """
        更新飞机状态

        Args:
            aircraft_uid: 飞机唯一标识符
            color: 颜色 (Red, Blue, etc.)
            longitude: 经度
            latitude: 纬度
            altitude: 高度 (m)
            roll: 滚转角 (度)
            pitch: 俯仰角 (度)
            heading: 航向角 (度)
            tas: 真空速 (m/s)，可选
            agl: 离地高度 (m)，可选
        """
        if not self.connected or not self.so:
            return

        try:
            # 检查是否已注册该飞机
            if aircraft_uid not in self._aircraft_objects:
                # 注册新飞机
                obj_id = str(self._get_next_id())
                self._aircraft_objects[aircraft_uid] = obj_id
                # 发送对象定义
                definition = f'{obj_id},Type=Air+FixedWing,Name=F-16,Color={color},Coalition=Allies\n'
                self.so.send(definition.encode('utf-8'))

            obj_id = self._aircraft_objects[aircraft_uid]
            elapsed = self._get_elapsed_time()

            # 构建更新数据
            if tas is not None and agl is not None:
                # 包含速度和离地高度
                data = TelDataFormat_less % (
                    elapsed,
                    obj_id,
                    longitude, latitude, altitude,
                    roll, pitch, heading,
                    color
                )
                # 添加速度和高度信息
                data = data.rstrip('\n') + f',AGL={agl:.3f},TAS={tas:.3f}\n'
            else:
                # 基本数据
                data = TelDataFormat_less % (
                    elapsed,
                    obj_id,
                    longitude, latitude, altitude,
                    roll, pitch, heading,
                    color
                )

            self.so.send(data.encode('utf-8'))

        except Exception as e:
            print(f"更新飞机状态失败: {e}")

    def update_missile(self, missile_uid, color, longitude, latitude, altitude,
                       roll, pitch, heading):
        """
        更新导弹状态

        Args:
            missile_uid: 导弹唯一标识符
            color: 颜色
            longitude: 经度
            latitude: 纬度
            altitude: 高度 (m)
            roll: 滚转角 (度)
            pitch: 俯仰角 (度)
            heading: 航向角 (度)
        """
        if not self.connected or not self.so:
            return

        try:
            # 检查是否已注册该导弹
            if missile_uid not in self._missile_objects:
                # 注册新导弹
                obj_id = str(self._get_next_id())
                self._missile_objects[missile_uid] = obj_id
                # 发送对象定义
                definition = f'{obj_id},Type=Weapon+Missile,Name=AIM-9,Color={color},Coalition=Allies\n'
                self.so.send(definition.encode('utf-8'))

            obj_id = self._missile_objects[missile_uid]
            elapsed = self._get_elapsed_time()

            # 构建更新数据
            data = TelDataFormat_less % (
                elapsed,
                obj_id,
                longitude, latitude, altitude,
                roll, pitch, heading,
                color
            )

            self.so.send(data.encode('utf-8'))

        except Exception as e:
            print(f"更新导弹状态失败: {e}")

    def remove_aircraft(self, aircraft_uid):
        """
        从Tacview中移除飞机

        Args:
            aircraft_uid: 飞机唯一标识符
        """
        if not self.connected or not self.so:
            return

        try:
            if aircraft_uid in self._aircraft_objects:
                obj_id = self._aircraft_objects[aircraft_uid]
                elapsed = self._get_elapsed_time()
                # 发送删除命令
                data = f'#{elapsed:.2f}\n-{obj_id}\n'
                self.so.send(data.encode('utf-8'))
                del self._aircraft_objects[aircraft_uid]

        except Exception as e:
            print(f"移除飞机失败: {e}")

    def remove_missile(self, missile_uid):
        """
        从Tacview中移除导弹

        Args:
            missile_uid: 导弹唯一标识符
        """
        if not self.connected or not self.so:
            return

        try:
            if missile_uid in self._missile_objects:
                obj_id = self._missile_objects[missile_uid]
                elapsed = self._get_elapsed_time()
                # 发送删除命令
                data = f'#{elapsed:.2f}\n-{obj_id}\n'
                self.so.send(data.encode('utf-8'))
                del self._missile_objects[missile_uid]

        except Exception as e:
            print(f"移除导弹失败: {e}")

    def add_explosion(self, explosion_uid, color, longitude, latitude, altitude, radius=300.0):
        """
        添加爆炸效果

        Args:
            explosion_uid: 爆炸唯一标识符
            color: 颜色
            longitude: 经度
            latitude: 纬度
            altitude: 高度 (m)
            radius: 爆炸半径 (m)
        """
        if not self.connected or not self.so:
            return

        try:
            obj_id = str(self._get_next_id())
            elapsed = self._get_elapsed_time()

            # 发送爆炸对象定义
            definition = f'{obj_id},Type=Explosion,Color={color}\n'
            self.so.send(definition.encode('utf-8'))

            # 发送爆炸位置数据
            data = f'#{elapsed:.2f}\n{obj_id},T=%.7f|%.7f|%.7f\n' % (
                longitude, latitude, altitude)
            self.so.send(data.encode('utf-8'))

        except Exception as e:
            print(f"添加爆炸效果失败: {e}")

    def clear_all(self):
        """清除所有对象"""
        if not self.connected or not self.so:
            return

        try:
            elapsed = self._get_elapsed_time()

            # 移除所有飞机
            for aircraft_uid in list(self._aircraft_objects.keys()):
                self.remove_aircraft(aircraft_uid)

            # 移除所有导弹
            for missile_uid in list(self._missile_objects.keys()):
                self.remove_missile(missile_uid)

        except Exception as e:
            print(f"清除对象失败: {e}")

    def close(self):
        """关闭连接"""
        if self.so:
            try:
                self.so.close()
            except:
                pass
        self.connected = False
        self.so = None


def tacivew_render():
    """
    旧版本的函数，保留用于兼容性
    建议使用TacviewClient类
    """
    print("启动接收线程")
    conf = {
        "serverip": "127.0.0.1",
        "serverport": 15502,
        "time": "2023 10 29 12 00 00"
    }

    with socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) as so:
        so.bind((conf["serverip"], conf["serverport"]))
        so.listen()
        print("listen")
        print("wait for connect")
        ss = so.accept()
        print("a client connected")
        so, addr = ss
        print('give a handshake: ')
        so.send(HandShakeData1.encode('utf-8'))
        so.send(HandShakeData2.encode('utf-8'))
        so.send(HandShakeData3.encode('utf-8'))
        so.send(b'\x00')
        data = so.recv(1024)
        print('wait for handshake: ')
        print(data)
        so.send(TelFileHeader.encode('utf-8'))
        ti = time.strptime(conf["time"], "%Y %m %d %H %M %S")
        t = time.strftime(TelReferenceTimeFormat, ti).encode('utf-8')
        print(t)
        so.send(t)
        print('已准备好传输数据')
        return so
