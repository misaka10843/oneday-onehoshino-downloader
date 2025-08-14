# oneday-onehoshino-downloader

一个不正经的qq群相册下载器

外加上一日一星野的下载器？！(将会使用ocr进行分类)

## 相关配置获取

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