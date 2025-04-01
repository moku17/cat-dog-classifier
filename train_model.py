print("모델 학습 시작!")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


train_gen = train_datagen.flow_from_directory(
    'images/',
    target_size=(100, 100),
    batch_size=16,
    class_mode='binary',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    'images/',
    target_size=(100, 100),
    batch_size=4,
    class_mode='binary',
    subset='validation'
)

# 모델 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[early_stopping])

# 학습 완료 후 검증 정확도 평가
loss, acc = model.evaluate(val_gen)
print(f"검증 정확도: {acc:.4f}")

# 모델 저장
model.save('model.h5')