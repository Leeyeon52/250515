import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import SGD

SEED = 2022

def get_image_generator(img_size, batch_size):
    base_path = r"C:\Users\user\Desktop\Medical-practice-1\DL_review\q4\extracted_archive\dataset"
    
    train_generator = ImageDataGenerator(rescale=1./255)
    train_data_gen = train_generator.flow_from_directory(
        os.path.join(base_path, "train"),
        batch_size=batch_size,
        shuffle=True,
        target_size=img_size,
        class_mode="binary"
    )

    valid_generator = ImageDataGenerator(rescale=1./255)
    valid_data_gen = valid_generator.flow_from_directory(
        os.path.join(base_path, "validation"),
        batch_size=batch_size,
        shuffle=False,
        target_size=img_size,
        class_mode="binary"
    )

    test_generator = ImageDataGenerator(rescale=1./255)
    test_data_gen = test_generator.flow_from_directory(
        os.path.join(base_path, "test"),
        batch_size=batch_size,
        shuffle=False,
        target_size=img_size,
        class_mode="binary"
    )

    return train_data_gen, valid_data_gen, test_data_gen

def build_cnn_model(img_size, num_classes=2):
    model = Sequential()
    img_size += (3,)
    
    model.add(layers.Conv2D(16, (3,3), padding="same", activation="relu", input_shape=img_size))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(32, (3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, (3,3), padding="same", activation="relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

def build_reg_cnn_model(img_size, num_classes=2):
    model = Sequential()
    img_size += (3,)

    model.add(layers.Conv2D(16, (3,3), padding="same", input_shape=img_size))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(32, (3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(64, (3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPool2D())

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model

def run_model(model, train_data_gen, valid_data_gen, test_data_gen, batch_size, epochs=30):
    train_len, valid_len, test_len = len(train_data_gen), len(valid_data_gen), len(test_data_gen)
    train_len, valid_len, test_len = train_len * batch_size, valid_len * batch_size, test_len * batch_size

    optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    hist = model.fit(
        train_data_gen,
        epochs=epochs,
        steps_per_epoch=(train_len // batch_size),
        validation_data=valid_data_gen,
        validation_steps=(valid_len // batch_size),
        verbose=2
    )
    
    test_loss, test_acc = model.evaluate(test_data_gen)
    
    return optimizer, hist, test_loss, test_acc

def main():
    tf.random.set_seed(SEED)

    batch_size = 64
    img_size = (32, 32)
    train_data_gen, valid_data_gen, test_data_gen = get_image_generator(img_size, batch_size)

    cnn_model = build_cnn_model(img_size)
    reg_cnn_model = build_reg_cnn_model(img_size)

    print("기본 CNN 모델 학습 시작")
    _, _, cnn_test_loss, cnn_test_acc = run_model(cnn_model, train_data_gen, valid_data_gen, test_data_gen, batch_size)
    
    print("정규화된 CNN 모델 학습 시작")
    _, _, reg_cnn_test_loss, reg_cnn_test_acc = run_model(reg_cnn_model, train_data_gen, valid_data_gen, test_data_gen, batch_size)
    
    print("[기본 CNN] 테스트 Loss: {:.5f}, 테스트 정확도: {:.3f}%".format(cnn_test_loss, cnn_test_acc * 100))
    print("[정규화된 CNN] 테스트 Loss: {:.5f}, 테스트 정확도: {:.3f}%".format(reg_cnn_test_loss, reg_cnn_test_acc * 100))

if __name__ == "__main__":
    main()
