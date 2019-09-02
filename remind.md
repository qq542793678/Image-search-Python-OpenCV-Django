# 一、第一步；
    首先你需要建立一个虚拟环境（最好建立虚拟环境），来运行当前项目，创建好虚拟环境之后，你需要把本项目clone到本地；
# 二、第二步：
    进入你clone的项目中，找到找到 cv2文件夹 进去复制里面的cv2模块到 你的python环境里，如果用到虚拟环境就放到虚拟环境，没有放到物理环境。 ..\Lib\site-packages\  这个目录下。（windows环境。linux我没有成功过，似乎需要自己编译模块）
# 三、第三步：
    安装requirements.txt文件的模块， pip install -r requirements.txt


最好自己迁移数据库，注册超级管理员。


这样做完应该就可以跑起来了。直接python main.py runserver 就可以启动了、
简述一下文件的 
OpenCV2.4 模块
项目文件
searcher： 
    数据库迁移文件
    model文件
    图像搜索引擎核心代码
    adminx。 xadmin后台管理文件的配置
    apps 
    models 数据库models
    views。 视图函数
静态文件
工具包
模板文件
xadmin 文件
项目控制中心，启动文件
其他配置文件和
opencv-pipei opencv-ex 测试cv2 是否安装好
    
