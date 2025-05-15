import warnings, logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from preprocess import *  # CIFAR10 데이터 전처리 함수

def load_transfer_model():
    # ImageNet으로 훈련된 ResNet-50 모델을 불러옵니다. 분류기 부분은 제외합니다.
    base_model = ResNet50(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
    
    # ✅ [지시사항 1번] base_model 학습되지 않도록 설정
    base_model.trainable = False
    
    # ✅ [지시사항 2번] transfer_model 완성
    transfer_model = Sequential([
        layers.UpSampling2D(size=(3, 3), interpolation='bilinear'),  # (32,32) -> (96,96)
        base_model,  # Feature extractor
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')  # CIFAR-10 (10개 클래스)
    ])

    return transfer_model

def main(transfer_model=None, epochs=3):
    np.random.seed(81)
    
    num_classes = 10
    x_train, y_train, x_test, y_test = cifar10_data(num_classes)
    x_train, y_train, x_test, y_test = x_train[:5000], y_train[:5000], x_test[:100], y_test[:100]

    if transfer_model is None:
        transfer_model = load_transfer_model()
    
    # ✅ [지시사항 3번] Optimizer, loss, metrics 설정
    optimizer = Adam(learning_rate=0.001)
    transfer_model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    # 학습 수행
    hist = transfer_model.fit(x_train, y_train, epochs=epochs, batch_size=500)
    
    # 모델 구조 출력
    print()
    transfer_model.summary()
    
    # 테스트 정확도 평가
    loss, accuracy = transfer_model.evaluate(x_test, y_test)
    print('\n훈련된 모델의 테스트 정확도는 {:.3f}% 입니다.\n'.format(accuracy * 100))
    
    return optimizer, hist

if __name__ == "__main__":
    main()
