import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import itertools
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(123)

# ----------------------------- Load dataset -----------------------------
csv_path = r"C:\Users\srinivas\Downloads\Skin_Cancer_Detection_MobileNetV2\HAM10000_metadata.csv"
skin_df = pd.read_csv(csv_path)

# Read all images recursively
project_dir = r"C:\Users\srinivas\Downloads\Skin_Cancer_Detection_MobileNetV2"

all_image_paths = glob(os.path.join(project_dir, "**", "*.jpg"), recursive=True)

print("Total image files found:", len(all_image_paths))

# Create image_id -> image_path dictionary
imageid_path_dict = {
    os.path.splitext(os.path.basename(path))[0]: path
    for path in all_image_paths
}

# Check CSV columns
print("CSV Columns:", skin_df.columns.tolist())

if "image_id" not in skin_df.columns:
    raise Exception("ERROR: 'image_id' column not found in HAM10000_metadata.csv")

# Map image paths
skin_df["path"] = skin_df["image_id"].map(imageid_path_dict)

# Remove missing images
skin_df = skin_df.dropna(subset=["path"])

print("Images after removing missing:", len(skin_df))
print(skin_df[["image_id", "path"]].head())
# ----------------------------- Load images -----------------------------
def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        return preprocess_input(np.array(img, dtype=np.float32))
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None
skin_df['image'] = skin_df['path'].map(load_image)
skin_df = skin_df[skin_df['image'].notna()]
print(f"Images successfully loaded: {len(skin_df)}")
# ----------------------------- Prepare features and labels -----------------------------
features = np.stack(skin_df['image'].values)
target = skin_df['dx'].values   
# Encode target as integers and then one-hot
lesion_type_dict = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma', 'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma', 'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'}
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
labels = to_categorical(skin_df['cell_type_idx'], num_classes=7)
# ----------------------------- Train/Validation/Test Split -----------------------------
x_train, x_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.25, random_state=1234, stratify=labels) # 75% train, 25% temp
x_val, x_test, y_val, y_test = train_test_split( x_temp, y_temp, test_size=0.4444, random_state=1234, stratify=y_temp) # ~20% test, ~5% val
print(f"x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")
# ----------------------------- MobileNetV2 Model -----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# Freeze base layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

for layer in base_model.layers[-30:]:
    layer.trainable = True
    # ----------------------------- Data Augmentation -----------------------------
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagen.fit(x_train)
# ----------------------------- Callbacks -----------------------------

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,
    factor=0.5,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ----------------------------- Train Model -----------------------------
epochs = 20
batch_size = 16
steps_per_epoch = int(np.ceil(x_train.shape[0] / batch_size))
model.compile(
    optimizer=Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_val, y_val),
    steps_per_epoch=steps_per_epoch,
    callbacks=[
        learning_rate_reduction,
        early_stop
    ]
)
# ----------------------------- Evaluate Model -----------------------------
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
val_loss, val_accuracy = model.evaluate(x_val, y_val, verbose=1)
print(f"Validation: accuracy = {val_accuracy:.4f} ; loss = {val_loss:.4f}")
 # ----------------------------- Save Model -----------------------------
save_dir = "saved_model"
os.makedirs(save_dir, exist_ok=True)
model_file = os.path.join(save_dir, "mobilenetv2_skin_cancer_model.keras")
model.save(model_file)
print("Model saved successfully!")
print("Location:", model_file)
# ----------------------------- Confusion Matrix -----------------------------
Y_pred = model.predict(x_val)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_val, axis=1)
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
def plot_confusion_matrix(cm, classes):

    plt.figure(figsize=(10,8))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title("Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.ylabel("True label")

    plt.xlabel("Predicted label")

    plt.tight_layout()

    plt.show()
# ----------------------------- Grad-CAM -----------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
            grads = tape.gradient(class_channel, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
# ----------Example Grad-CAM -----------------------------------------
img_array = np.expand_dims(x_val[0], axis=0)
heatmap = make_gradcam_heatmap(img_array, model, 'Conv_1')
heatmap = np.uint8(255 * heatmap)
original_img = (x_val[0]).astype("uint8")
heatmap = cv2.resize(
    heatmap,
    (original_img.shape[1], original_img.shape[0])
)
heatmap_color = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

superimposed_img = cv2.addWeighted(
    original_img,
    0.6,
    heatmap_color,
    0.4,
    0
)  
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
cv2.imwrite("gradcam_example.jpg", superimposed_img)
print("Grad-CAM image saved as: gradcam_example.jpg")
plt.imshow(superimposed_img)
plt.title("Grad-CAM Example")
plt.axis('off')
plt.show()