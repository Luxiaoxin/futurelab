# futurelab
futurelab图像算法比赛的代码

程序简介，系统要求，所以依赖的第三方程序和库及其版本信息，部署和运行方法说明

1 程序简介 基于keras在resnet的基础上进行fine-tune
split.py              将训练数据集分割成20类，放入不同文件夹，以便
keras_resnet50.py     训练模型文件
test.py               生成测试csv

2 系统要求 python = 3.6
          keras = 2.1.3 (如果是2.1.5会报错)
          tensorflow = 1.7.0
          h5py = 2.7.1
          
3 测试方法 （需要下载resnet的预训练模型）  
在命令行输入
python test.py --dataset_dir=<testset_dir> --target_file=<target_file>

4 训练方法 先做数据集分隔 
python split.py 
再做模型训练 
python keras_resnet50.py


5 赛后的感想

最后top3准确率还是停留在了0.9783，没能进入复赛，有些遗憾吧，但是还是想把做项目的一些经历写出来。
有一些
