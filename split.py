import os, shutil,argparse
import numpy as np
import pandas as pd


def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        shutil.rmtree(testing_data_dir, ignore_errors=False)
        os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print(
            "Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        shutil.rmtree(training_data_dir, ignore_errors=False)
        os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print(
            "Refusing to delete testing data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)

        for file in files:
            input_file = os.path.join(subdir, file)
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help="Define the running mode as 'training' or 'test'.")
    parser.add_argument('--all_data_dir',default='F:/image_scene_training/data', type=str, help="Path to directory of training set or test set, depends on the running mode.")
    args = parser.parse_args()
    # 分类
    all_data_dir = 'F:/image_scene_training/data'
    dst_data_dir = all_data_dir + '/dataset'
    os.mkdir(dst_data_dir)
    os.mkdir('F:/image_scene_training/train')
    os.mkdir('F:/image_scene_training/val')
    os.mkdir(dst_data_dir + '/' + '0_bus')
    os.mkdir(dst_data_dir + '/' + '1_bazaar')
    os.mkdir(dst_data_dir + '/' + '2_church')
    os.mkdir(dst_data_dir + '/' + '3_cafe')
    os.mkdir(dst_data_dir + '/' + '4_basketball')
    os.mkdir(dst_data_dir + '/' + '5_lake')
    os.mkdir(dst_data_dir + '/' + '6_waterfall')
    os.mkdir(dst_data_dir + '/' + '7_underwater-coral_reef')
    os.mkdir(dst_data_dir + '/' + '8_starry_sky')
    os.mkdir(dst_data_dir + '/' + '9_skydive')
    os.mkdir(dst_data_dir + '/' + '10_soccer')
    os.mkdir(dst_data_dir + '/' + '11_snow_mountain')
    os.mkdir(dst_data_dir + '/' + '12_Chinese_temple')
    os.mkdir(dst_data_dir + '/' + '13_ski')
    os.mkdir(dst_data_dir + '/' + '14_forest')
    os.mkdir(dst_data_dir + '/' + '15_railway_station')
    os.mkdir(dst_data_dir + '/' + '16_underwater_wreck')
    os.mkdir(dst_data_dir + '/' + '17_desert')
    os.mkdir(dst_data_dir + '/' + '18_skyscraper')
    os.mkdir(dst_data_dir + '/' + '19_beach')

    dataset = pd.read_csv('list.csv')
    file = list(dataset["FILE_ID"].values)
    CATEGORY = list(dataset['CATEGORY_ID'].values)
    for (name, ID) in zip(file, CATEGORY):
        if ID == 0:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '0_bus')
        if ID == 1:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '1_bazaar')
        if ID == 2:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '2_church')
        if ID == 3:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '3_cafe')
        if ID == 4:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '4_basketball')
        if ID == 5:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '5_lake')
        if ID == 6:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '6_waterfall')
        if ID == 7:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '7_underwater-coral_reef')
        if ID == 8:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '8_starry_sky')
        if ID == 9:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '9_skydive')
        if ID == 10:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '10_soccer')
        if ID == 11:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '11_snow_mountain')
        if ID == 12:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '12_Chinese_temple')
        if ID == 13:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '13_ski')
        if ID == 14:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '14_forest')
        if ID == 15:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '15_railway_station')
        if ID == 16:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '16_underwater_wreck')
        if ID == 17:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '17_desert')
        if ID == 18:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '18_skyscraper')
        if ID == 19:
            shutil.move(all_data_dir + '/' + name + '.jpg', dst_data_dir + '/' + '19_beach')

    # 分割数据集，给绝对路径
    split_dataset_into_test_and_train_sets(all_data_dir=dst_data_dir,
                                           training_data_dir='F:/image_scene_training/train',
                                           testing_data_dir='F:/image_scene_training/val',
                                           testing_data_pct=.2)
