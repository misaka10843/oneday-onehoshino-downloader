# oneday-onehoshino-downloader

一个不正经的qq群相册下载器

外加上一日一星野的下载器？！(将会使用ocr进行分类)

## 如何使用

普通群相册下载请确保将仓库下载下来解压之后运行`pip install requirements.txt`

然后再将`.env.sample`重命名为`.env`

最后配置成功后运行`python main.py`

## 配置文件

```dotenv
COOKIES= # 在要下载群相册中的自己的QQ的cookie
QUNID= # 要下载的群相册的群号
ALBUMID= # 要下载的群相册ID
QQNUM= # 在要下载群相册中的自己的qq号
TK= # TK
```

## 相关配置获取

注意！因为qq的cookie保持时间可能就只有几个小时，所以`cookie`与`TK`建议在每次运行前重新获取一次

### cookie

请打开`https://h5.qzone.qq.com/groupphoto/index?inqq=1&groupId={需要下载的qq群的群号}`

并且登录已经入群的账号，并F12打开控制台

刷新页面之后在**网络**中随便点开一个请求在**表头**找到`cookie`复制里面的值即可

![img.png](assets/img.png)

### ALBUMID

在完成上面的操作之后选择你需要下载的相册（请注意不要关闭控制台）

将筛选器选定为`Fetch/XHR`

选择前缀为`inqq`的链接并选择**响应**找到`albumid`

![img.png](assets/img1.png)

复制值即可

### TK

在上面操作中的链接中有`g_tk={数字}&....`

其中数字即是TK