# # from django.shortcuts import render
# # from django.core.files.storage import FileSystemStorage
# # from django.conf import settings
# # from django.views import View
# # from keras.preprocessing import image
# # from keras.applications.mobilenet import preprocess_input
# # from keras.models import load_model
# # import os

# # class HomeView(View):
# #     def get(self, request):
# #         return render(request, 'home.html')
    
# #     def post(self, request):
# #         if 'image' in request.FILES:
# #             uploaded_image = request.FILES['image']
# #             fs = FileSystemStorage()
# #             image_path = fs.save(uploaded_image.name, uploaded_image)
# #             image_path = os.path.join(settings.MEDIA_ROOT, image_path)

# #             model = load_model("C:/Users/FABIAN CAMPOVERDE/IA/MobileNet/model_Mobilenet.h5")
# #             img = image.load_img(image_path, target_size=(224, 224))
# #             img_array = image.img_to_array(img)
# #             img_array = preprocess_input(img_array.reshape(1, 224, 224, 3))
# #             prediction = model.predict(img_array)
# #             predicted_class = "Class " + str(prediction.argmax())

# #             return render(request, 'home.html', {'predicted_class': predicted_class})
# #         else:
# #             return render(request, 'home.html')

# from django.shortcuts import render
# from django.conf import settings
# from django.core.files.storage import FileSystemStorage
# from keras.preprocessing import image
# from keras.applications.imagenet_utils import preprocess_input
# from keras.models import load_model
# import numpy as np
# import os

# def predict_image(request):
#     if request.method == 'POST' and request.FILES['image']:
#         uploaded_image = request.FILES['image']
        
#         # Guarda la imagen subida en el directorio de medios
#         fs = FileSystemStorage(location=settings.MEDIA_ROOT)
#         image_path = fs.save(uploaded_image.name, uploaded_image)
#         image_path = os.path.join(settings.MEDIA_ROOT, image_path)
        
#         # Carga el modelo
#         model_path = "model_Mobilenet-75_epochs.h5"   
#         model = load_model(model_path)
        
#         # Carga los nombres de las clases
#         names =  ['fresh_camote', 'fresh_cebolla', 'fresh_limon', 'fresh_mango', 'fresh_papa', 'fresh_tomate', 'fresh_zanahoria', 'overripe_platano', 
#                   'ripe_platano', 'rotten_camote', 'rotten_cebolla', 'rotten_limon', 'rotten_mango', 'rotten_papa', 'rotten_platano', 
#                   'rotten_tomate', 'rotten_zanahoria', 'unripe_platano']
        
#         # Carga y preprocesa la imagen
#         img = image.load_img(image_path, target_size=(224, 224))
#         img_array = image.img_to_array(img)
#         img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        
#         # Realiza la predicción
#         preds = model.predict(img_array)
#         predicted_class = names[np.argmax(preds)]
        
#         # Elimina la imagen subida del directorio de medios
#         if os.path.exists(image_path):
#             os.remove(image_path)
        
#         return render(request, 'result.html', {'predicted_class': predicted_class})
#     return render(request, 'index.html')


from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
# from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
import scipy

# Cargar el modelo previamente entrenado
custom_Model = load_model("model_Mobilenet-75_epochs.h5")

# Definir dimensiones de la imagen
width_shape = 224
height_shape = 224

# Nombres de las clases
names = ['fresh_camote', 'fresh_cebolla', 'fresh_limon', 'fresh_mango', 'fresh_papa', 'fresh_tomate', 
         'fresh_zanahoria', 'overripe_platano', 'ripe_platano', 'rotten_camote', 'rotten_cebolla',
         'rotten_limon', 'rotten_mango', 'rotten_papa', 'rotten_platano', 'rotten_tomate', 'rotten_zanahoria', 'unripe_platano']

# Mapeo de clases en español
mapeo_clases = {
    'fresh_camote': 'Camote fresco',
    'fresh_cebolla': 'Cebolla fresca',
    'fresh_limon': 'Limón fresco',
    'fresh_mango': 'Mango fresco',
    'fresh_papa': 'Papa fresca',
    'fresh_tomate': 'Tomate fresco',
    'fresh_zanahoria': 'Zanahoria fresca',
    'overripe_platano': 'Plátano sobremaduro',
    'ripe_platano': 'Plátano maduro',
    'rotten_camote': 'Camote podrido',
    'rotten_cebolla': 'Cebolla podrida',
    'rotten_limon': 'Limón podrido',
    'rotten_mango': 'Mango podrido',
    'rotten_papa': 'Papa podrida',
    'rotten_platano': 'Plátano podrido',
    'rotten_tomate': 'Tomate podrido',
    'rotten_zanahoria': 'Zanahoria podrida',
    'unripe_platano': 'Plátano verde'
}

# Función para preprocesar la imagen
def preprocess_image(img):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0,  
    zoom_range=0,  
    width_shift_range=0,  
    height_shift_range=0,  
    horizontal_flip=False,  
    vertical_flip=False,  
    preprocessing_function=preprocess_input
)
    
    img = cv2.resize(img, (width_shape, height_shape))
    img = np.expand_dims(img, axis=0)
    img = datagen.flow(img).next()[0]
    return img
def predict_image(request):
    if request.method == 'POST':
        try:
            imagen = request.FILES.get('image')
            if imagen:
                 img = Image.open(imagen)
                 img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                 img = cv2.resize(img, (400, 300))

                # Realizar la predicción solo si la imagen se cargó correctamente
                 if img is not None:
                    img = preprocess_image(img)
                    img_array = np.expand_dims(img, axis=0)
                    predictions = custom_Model.predict(img_array)
                    clase_pred = names[np.argmax(predictions)]
                    clase_pred_en_espanol = mapeo_clases.get(clase_pred)
                    return JsonResponse({'clase_pred_en_espanol': clase_pred_en_espanol})
                 else:
                    return JsonResponse({'error': 'La imagen no es válida'}, status=400)
            else:
                return JsonResponse({'error': 'No se proporcionó ninguna imagen'}, status=400)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return render(request, 'index.html')
# Vista para la página de clasificación de vegetales
# def predict_image(request):
#     if request.method == 'POST' and request.FILES['image']:
#         imagen = request.FILES['image']
#         img = Image.open(imagen)
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         img = preprocess_image(img)
#         img_array = np.expand_dims(img, axis=0)
#         predictions = custom_Model.predict(img_array)
#         clase_pred = names[np.argmax(predictions)]
#         clase_pred_en_espanol = mapeo_clases.get(clase_pred)
#         return render(request, 'result.html', {'clase_pred_en_espanol': clase_pred_en_espanol})
#     return render(request, 'index.html')

