from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivy.uix.screenmanager import ScreenManager
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.button import MDFloatingActionButton, MDFlatButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.toolbar import MDTopAppBar
from kivymd.uix.button import MDIconButton
from kivymd.uix.chip import MDChip
from kivymd.uix.card import  MDCard
from kivy.uix.camera import Camera
from kivymd.uix.label import MDLabel
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.scrollview import ScrollView
import pandas as pd
from PIL import Image as PILImage
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array



class HomePage(MDScreen):  
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toolbar = MDTopAppBar(
            title="DTreaty",
            pos_hint={"top": 1}
        )
        self.layout = MDBoxLayout(orientation='vertical')
        self.chip = MDChip(
            text = "Warning!!!",
            md_bg_color=(1,1,0,1),
            pos_hint={"center_x": .25, "center_y": .85},
                )
        self.card = MDCard(
                    size_hint=(0.9, 0.18),
                    pos_hint={"center_x": 0.5, "top": 0.8},
                    elevation=5,
                    padding = (dp(12), dp(12)),
                )
        self.text_label = MDLabel(
                    text="The App can generate false information..",
                    pos_hint= {'center_x': 0.5,'center_y': 0.5},
                    size_hint=(1,1)
)
         # Create an MDCamera widget
        self.camera = Camera(play=True, resolution=(640, 600), pos_hint= {'center_x': 0.5,'center_y': 0.4})
        
        # Create a floating action button for camera
        self.btn = MDFloatingActionButton(
            icon="camera",
            pos_hint={'center_x': 0.5, 'center_y': 0.1},
            on_release=self.capture,
        )

        self.add_widget(self.camera)
        self.add_widget(self.btn)
        self.add_widget(self.toolbar)
        self.add_widget(self.card)
        self.add_widget(self.chip)
        self.card.add_widget(self.text_label)
        self.add_widget(self.layout)
        
        # Initialize the label attribute
        self.label = ""

    # function for the screen navigation
    def go_to_treatment(self, instance):
        app = MDApp.get_running_app()
        treat_screen = Ilaj(name="TreatmentScreen", label=self.label)
        app.sm.add_widget(treat_screen)
        app.sm.current = "TreatmentScreen"
        self.dialog.dismiss()
    
# Function for caturing the image through camera and preprocessing it to pass to the model
    def capture(self, obj):
        img_texture = self.camera.texture

        if img_texture:
            pil_image = PILImage.frombytes('RGBA', img_texture.size, img_texture.pixels)
            pil_image = pil_image.convert('RGB')
            pil_image = pil_image.resize((224, 224))  # Resize the image to the desired input size
            image_array = img_to_array(pil_image)
            image_array = image_array / 255.0
            self.model = tf.keras.models.load_model('Resnet_Model.h5')
            prediction = self.model.predict(np.expand_dims(image_array, axis=0))
            predicted_class = np.argmax(prediction, axis=1)
            class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

            predicted_class_name = class_names[predicted_class[0]]

            self.label = f"{predicted_class_name}"

            print("the predicted class is : "+ self.label)  

            # Code for the alert where the predicted disease will be displayed. 
            
            if self.label != "":
                user_error = "The Specie is effeted by " + self.label
            else:
                user_error = "Error!! Please Try Again..."
            self.dialog = MDDialog(title='Disease Detection',
                               text=user_error, size_hint=(0.8, 1),
                               buttons=[MDFlatButton(text='Close', on_release=self.close_dialog),
                                        MDFlatButton(text='More', on_release = self.go_to_treatment)]
                               )
            self.dialog.open()

            return self.label

    def close_dialog(self, obj):
        self.dialog.dismiss()

class Ilaj(MDScreen):
    def __init__(self, label="", **kwargs):
        super().__init__(**kwargs)


        self.label = label  # Store the label value
        df = pd.read_csv('treatment-book.csv', encoding='ISO-8859-1')
        condition_column = 'disease_name' 
        condition_value = "Apple___Apple_scab"
        value_column = 'treatment'
        self.no_data = "No matching data found"


        self.treatment_series = df.loc[df[str(condition_column)] == str(condition_value), str(value_column)]
        self.treatment_series = self.treatment_series.astype(str)
        treatment = "\n".join(self.treatment_series) if not self.treatment_series.empty else self.no_data


        # Create a FloatLayout to position content
        self.box_layout1 = MDGridLayout(cols=1, spacing=10, padding=20, size_hint_y= None)
        self.box_layout1.bind(minimum_height = self.box_layout1.setter('height'))
        # Create a top app bar
        self.toolbar = MDTopAppBar(
            title="Treatment",
            pos_hint={"top": 1}
        )
        self.back_button = MDIconButton(
            icon="arrow-left",
            on_release=self.go_back
        )
        # Create a scrollable area
        self.scroll_view = ScrollView(size_hint=(1,None), size=(Window.width, Window.height))
        
        self.text_label = MDLabel(
            text= treatment,
            halign="center",
            valign="middle",
            size_hint_y=None,
            height = self.scroll_view.height - self.toolbar.height,
        )
        self.toolbar.left_action_items = [["arrow-left", self.go_back]]
        
        # new code.................
        self.text_label.text_size = (self.text_label.width, None)  # Allow text to wrap and be scrollable
        self.text_label.bind(texture_size=self.on_texture_size)

        # Add the components to the screen
        self.add_widget(self.toolbar)
        self.add_widget(self.scroll_view)
        self.scroll_view.add_widget(self.box_layout1)
        self.box_layout1.add_widget(self.text_label)

    def on_texture_size(self, instance, size):
        self.text_label.height = size[1]

    def go_back(self, *args):
        app = MDApp.get_running_app()
        app.root.current = "HomeScreen"


class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"  # Set dark theme
        self.theme_cls.primary_palette = "Blue"

        self.sm = ScreenManager()

        self.home_screen = HomePage(name="HomeScreen")

        self.sm.add_widget(self.home_screen)
        return self.sm

if __name__ == "__main__":
    MainApp().run()
