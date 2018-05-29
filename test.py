import numpy as np
import csv,argparse
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

IM_WIDTH, IM_HEIGHT = 224, 224
FC_SIZE = 1024  # 全连接层的节点个数
nb_classes = 20


# 添加新层
def add_new_last_layer(base_model, nb_classes):
    """
    添加最后的层
    输入
    base_model和分类数量
    输出
    新的keras的model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


def predict(model, img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 插入这一个轴是关键，因为keras中的model的tensor的shape是（bath_size, h, w, c),如果是tf后台
    x = preprocess_input(x)
    return model.predict(x)


# 预测返回的类和原本类对应的数值对不上
# 原本类-返回类 0-0 1-1 2-12 3-13 4-14 5-15 6-16 7-17 8-18 9-19 10-2 11-3 12-4 13-5 14-6 15-7 16-8 17-9 18-10 19-11
def decode_class(preds):
    num1 = np.argmax(preds)
    preds[0,num1] = 0
    num2 = np.argmax(preds)
    preds[0,num2] = 0
    num3 = np.argmax(preds)

    dict = {0: '0,巴士,bus',
            1: '1,集市,bazaar',
            2: '10,足球/足球场,soccer',
            3: '11,雪山,snow mountain',
            4: '12,中式庙宇/建筑,Chinese temple',
            5: '13,滑雪/滑雪场,ski',
            6: '14,森林/树林,forest',
            7: '15,火车站/轨道交通,railway station',
            8: '16,水下-残骸/沉船,underwater-wreck',
            9: '17,荒漠/沙漠,desert',
            10: '18,高楼/大厦,skyscraper',
            11: '19,海滩,beach',
            12: '2,教堂,church',
            13: '3,咖啡馆,cafe',
            14: '4,篮球/篮球场,basketball',
            15: '5,湖泊,lake',
            16: '6,瀑布/溪流,waterfall',
            17: '7,水下-珊瑚礁,underwater-coral reef',
            18: '8,星空/夜空,starry sky',
            19: '9,跳伞,skydive'
            }
    dict1 = {
        0: 0, 1: 10, 2: 11, 3: 12, 4: 13, 5: 14, 6: 15, 7: 16, 8: 17, 9: 18, 10: 19, 11: 1, 12: 2, 13: 3,
        14: 4, 15: 5, 16: 6, 17: 7, 18: 8, 19: 9
            }
    list = [dict.get(num1),dict.get(num2),dict.get(num3)]
    list1 = [dict1.get(num1),dict1.get(num2),dict1.get(num3)]
    return list1


base_model = ResNet50(weights='imagenet', include_top=False)  # 预先要下载no_top模型
model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
model.load_weights('my_model_resnet50.h5')


# 预测图片
def pred(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    # 对图像进行分类
    preds1 = predict(model, img, (224, 224))
    # 输出预测概率
    # print('Predicted:', decode_class(preds))
    preds2 = decode_class(preds1)
    return preds2


def write_csvlist(listfile, data, header = []):
    with open(listfile, 'w') as fh:  # python3 这里的wb改成w
        csv_writer = csv.writer(fh)
        if (len(header)):
            csv_writer.writerow(header)
        for row in data:

            csv_writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='D:/chrome_download/image_scene_test_b_0515/image_scene_test_b_0515/', help="Path to directory of test set")
    parser.add_argument('--target_file', type=str, default='./test_results.csv', help='Path to test result file')

    args = parser.parse_args()
    # 读取测试集
    data_path = args.dataset_dir
    listfile = data_path + 'list.csv'
    csv_reader = csv.reader(open(listfile))
    results = []
    for row in csv_reader:  # 此时输出的是一行行的列表
        results.append(row)
    del results[0]
    jieguo = []
    for i in results:
        img_path = data_path + 'data/' + i[0] + '.jpg'
        y = pred(img_path)
        pred_list = [i[0],y[0],y[1],y[2]]
        jieguo.append(pred_list)
    traget_file = args.target_file
    write_csvlist(traget_file,jieguo,header=['FILE_ID', 'CATEGORY_ID0', 'CATEGORY_ID1', 'CATEGORY_ID2'])


