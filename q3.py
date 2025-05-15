import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential

IMG_SIZE = 256

def main():
    dog = tf.keras.utils.load_img(r"C:\Users\user\Desktop\Medical-practice-1\Avoid_overfitting\q3\dog.jpg")
    cat = tf.keras.utils.load_img(r"C:\Users\user\Desktop\Medical-practice-1\Avoid_overfitting\q3\cat.jpg")
    
    dog_array = tf.keras.utils.img_to_array(dog)
    cat_array = tf.keras.utils.img_to_array(cat)
    
    # [필수] 배치 차원 추가
    dog_array = tf.expand_dims(dog_array, axis=0)
    cat_array = tf.expand_dims(cat_array, axis=0)

    # ✅ 개 전처리
    dog_augmentation = Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1. / 255),
        layers.RandomCrop(150, 200)
    ])
    
    # ✅ 고양이 전처리 (오타 수정)
    cat_augmentation = Sequential([
        layers.Resizing(IMG_SIZE, IMG_SIZE),
        layers.Rescaling(1. / 255),
        layers.RandomFlip(),
        layers.RandomRotation(0.5)
    ])
    
    # 전처리 및 저장
    dog_augmented_tensor = dog_augmentation(dog_array)[0]  # 배치 차원 제거
    dog_augmented = tf.keras.utils.array_to_img(dog_augmented_tensor)
    dog_augmented.save("./dog_augmented.jpg")
    print("=" * 25, "전처리된 개", "=" * 25)

    print()

    cat_augmented_tensor = cat_augmentation(cat_array)[0]
    cat_augmented = tf.keras.utils.array_to_img(cat_augmented_tensor)
    cat_augmented.save("./cat_augmented.jpg")
    print("=" * 25, "전처리된 고양이", "=" * 25)

    return dog_augmentation, cat_augmentation

if __name__ == "__main__":
    main()
