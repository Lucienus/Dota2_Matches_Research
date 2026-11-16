# 上传说明

将此ZIP解压后的内容上传至你选定的GitHub版本目录或分支，保留包内目录结构。

上传前后均可运行：

```text
python release.py verify
```

`README.md`提供完整命令。入口使用者先取得所需原始快照/训练输入，并在本地生成路径索引；不要把本地索引、API KEY、`outputs/`、虚拟环境、原始JSON、NPZ和检查点加入提交。

发布后记录实际commit或tag，再将该版本链接回填正文、SI和投稿信。此次没有替你上传，也没有更改远端仓库。
