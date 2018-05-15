import os
import glob
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD


path = r'D:\py_futurelab_ai\contest2018-image_scene_classification-master\baseline\image_train'

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


# 数据准备
IM_WIDTH, IM_HEIGHT = 224, 224  # 图片尺寸
FC_SIZE = 1024  # 全连接层的节点个数
NB_LAYERS_TO_FREEZE = int(175 * .6)  # resnet层数*0.6

train_dir = os.path.join(path, 'classes')  # 训练集数据
val_dir = os.path.join(path, 'test_classes')  # 验证集数据
nb_classes = 20
nb_epoch = 60
batch_size = 64

nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数

# 图片生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')

# 输出类别信息
print(train_generator.class_indices)
print(validation_generator.class_indices)
'''
{'0_bus': 0, '10_soccer': 1, '11_snow_mountain': 2, '12_Chinese_temple': 3, '13_ski': 4, '14_forest': 5, 
'15_railway_station': 6, '16_underwater_wreck': 7, '17_desert': 8, '18_skyscraper': 9, '19_beach': 10, 
'1_bazaar': 11, '2_church': 12, '3_cafe': 13, '4_basketball': 14, '5_lake': 15, '6_waterfall': 16, 
'7_underwater-coral_reef': 17, '8_starry_sky': 18, '9_skydive': 19}
'''


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


# 冻上NB_IV3_LAYERS之前的层
def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

    Args:
      model: keras model
    """
    for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


# 定义网络框架
# 使用ResNet的结构，不包括最后一层，且加载ImageNet的预训练参数
base_model = ResNet50(weights='imagenet', include_top=False)
model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
setup_to_finetune(model)
# 模型保存
checkpoint = ModelCheckpoint('my_model_resnet50.h5', verbose=1, save_best_only=True)
early = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
callbacks_list = [checkpoint, early]
# 模式二训练
history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    validation_steps=nb_val_samples // batch_size,
    callbacks=callbacks_list,
    class_weight='auto')

model.save('last_model_resnet50.h5')


# 画图
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'r-')
    plt.title('Training and validation accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


# 训练的acc_loss图
plot_training(history_ft)
