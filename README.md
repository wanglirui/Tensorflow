# Tensorflow
tensorflow object  detection
目标检测fine-tuning报告（男女检测）
一、	数据集
在数据库中下载了2000张标签为肖像画的画作，选择1000张作为训练集和验证集，1000张作为测试集。

二、	数据标注
采用labelImg软件来标记自己的数据集，选择存放图片的文件夹，选择图像，然后手动框出图像上的物体，并加类别标签（对于类别标签我们应该根据画作首先确定我们需要检测出来的类别数量，本次实验标记了男人和女人两个类别），然后保存就能够生成XML格式的标注文件，点击NEXT IMAGE 进行下一张图像的标注。如下图所示：在这幅图中手动标注出来了女人，并生成了对应的XML标注文件。
![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/1.jpg)
 
 ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/2.jpg)
三、	格式转换
1.	将XML的标注文件转换为CSV文件，并且将标记好的图片分为训练集和验证集。
这个脚本为wanglirui/TRF_data_get/data_split.py.
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/3.jpg)

2.	CSV文件转换为TFRecord文件
该项目需要把将图片和标注转换为TFRecord格式，这需要编写一个python的脚本文件wanglirui/ TRF_data_get/ csv_to_TFrecord.py。同时也需要一个类别ID与类别名称的关系文件，这个关系通常用pbtxt格式文件保存，例如新建一个person_map.pbtxt的文本文件，内容如下：
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/4.jpg)
最后执行格式转换的脚本来获分别得TFRecord格式训练集和验证集的数据。

  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/5.jpg)
四、	选择预训练模型
   选择我们需要的预训练模型，比如之前用的rfcn_resnet101_coco, faster_rcnn_nas_coco, ssd_mobilenet_v1_coco等。然后使用我们自己标注的数据在这个模型上进行fine-tuning。将模型上传到服务器。
五、	修改训练配置文件
例如使用ssd_mobilenet_v1_coco_2018_01_28模型，需要复制object_detection/samples/configs下对应的ssd_mobilenet_v1_coco.config，在复制的配置文件中修改，改成我们对应的文件路径即可：
1.	fine_tune_checkpoint: "/ 上面第四步下载的预训练模型路径/model.ckpt" 
2.	input_path: "/预处理数据生成的tfrecords格式数据的文件路径”，分为训练集和验证集两个。
3.	label_map_path: “/格式转换过程中使用过的类别与ID对应的pbtxt文件” 
4.	num_classes: 我们自己的数据集的类别数
同时在这个配置文件中还可以更改训练时的batch_size,学习率，epoch数量，数据增强的方式，优化算法的选择，评价指标等。
六、	重新训练模型
启动配置好的虚拟环境
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/6.jpg)
配置一下环境变量
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/7.jpg)
开始训练，执行object_detection/legacy/train.py这个文件，需要告诉训练集的路径和对应的配置文件的路径。
python./legacy/train.py --logtostderr --pipeline_config_path=./mymodel_ssd/ssd_mobilenet_v1_coco.config  --train_dir=./training_result/ssd_mobiel_model
在训练过程中使用验证集来评估模型的性能，执行object_detection/eval.py这个文件。执行训练完成后，保存模型。导出训练好的模型，执行object_detection/export_inference_graph.py：
python export_inference_graph.py --pipeline_config_path=./mymodel_ssd/ssd_mobilenet_v1_coco.config --trained_checkpoint_prefix=./training_result/ssd_mobiel_model/model.ckpt-2000 --output_directory=./training_result//ssd_mobiel_model
然后就可以用训练得到的模型来对新的图片进行目标检测了。
七、	目标检测
 将到出的模型中frozen_inference_graph.pb文件单独保存到mymodel_ssd中，打开jupyter notebook，执行object_detection/
object_detection_tutorial.ipynb 。最后的检测结果会保存到预习设定的文件夹result里面。
 ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/8.jpg)
 
八、	实验结果
 
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/9.jpg)
 
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/10.jpg)
  
 ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/11.jpg)
  ![Image text](https://github.com/wanglirui/Tensorflow/blob/master/image/12.jpg)
