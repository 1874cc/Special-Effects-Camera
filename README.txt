运行项目需要在vs中重新配置qt平台与设置。
1.项目->camera属性->常规->windows SDK版本改为当前版本，平台工具集改为当前vs版本（例如原来是vs2019，要在vs2015上跑就要改成vs2015）
2.qt project settings中qt installation改为当前qt版本（例如msvc2019_64）
3.修改opencv包含目录与库目录，链接文件为当前版本下
4.运行时建议选择release类型，要比debug模式下运气起来流畅
