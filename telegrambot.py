import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO
from PIL import Image
from Detection import detection
 
# Class names and colors (BGR format for OpenCV)
class_names = ['bg', 'Platelets', 'RBC', 'WBC']
color_sample = [
    (0, 0, 0),      # Background (bg) - Black    
    (100, 0, 0),    # WBC - Darker Red (More Brownish Tone)
    (160, 20, 90),  # RBC - Dark Pink (More Purple Tone)
    (0, 0, 139),    # Platelets - Dark Blue
]

# Telegram Bot Token (replace with your own)
TOKEN = "your token id"

# Object Detection Function
def object_detection_image(image):
    image_with_boxes, counter = detection(image, class_names, color_sample)
    # Convert to NumPy array
    image = np.array(image_with_boxes)
    _, img_encoded = cv2.imencode('.jpg', image)
    return BytesIO(img_encoded.tobytes()), counter


async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "🔬 **Welcome to the Blood Cell Detection Bot!** 🩸\n\n"
        "📸 **Send me an image of a blood sample**, and I will:\n"
        "✅ Detect and classify blood cells\n"
        "✅ Count the number of each type\n"
        "✅ Generate an image with bounding boxes 📊\n\n"
        "**Detected Blood Cell Types:**\n"
        "🟢 **Platelets**\n"
        "🔴 **Red Blood Cells (RBC)**\n"
        "⚪ **White Blood Cells (WBC)**\n\n"
        "🚀 *Send an image to get started!*"
    )


async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "🔬 **Blood Cell Detection Bot Help** 🩸\n\n"
        "This bot detects and classifies blood cells in images. 📸\n"
        "Simply send an image of a blood sample, and I'll analyze it to detect:\n"
        "🟢 Platelets\n"
        "🔴 Red Blood Cells (RBC)\n"
        "⚪ White Blood Cells (WBC)\n\n"
        "🔢 The bot will also **generate an image with bounding boxes** and display the **count of each detected class**!\n\n"
        "**Available Commands:**\n"
        "📌 /start - Welcome message\n"
        "📌 /help - Information on how to use the bot\n"
        "📌 /about - Details about the detection model and blood cell types\n"
    )
    await update.message.reply_text(help_text)


# About Command
async def about_command(update: Update, context: CallbackContext):
    about_text = (
        "🩸 **Blood Cell Object Detection Bot** 🔬\n\n"
        "This bot detects and classifies blood cells from images using an advanced deep learning model. 💉\n\n"
        "**🔍 Features:**\n"
        "✅ Detects and classifies **Platelets, RBCs (Red Blood Cells), and WBCs (White Blood Cells)** 🟢🔴⚪\n"
        "✅ **Bounding Boxes** around detected cells for precise localization 📏\n"
        "✅ **Counts the number of each cell type** in the image 🔢\n"
        "✅ **Deep Learning Model**: Uses **SSD (Single Shot MultiBox Detector) with VGG-16** for high-accuracy detection 🧠\n"
        "✅ **Optimized Training Strategy**: Trained with SGD optimizer, StepLR learning rate scheduler, and data augmentation 🚀\n"
        "✅ **Hardware Acceleration**: Runs efficiently on GPU for faster inference ⚡\n\n"
        
        "**🧬 Fun Facts About Blood Cells:**\n"
        "🔴 **Red Blood Cells (RBCs)** carry oxygen throughout the body and have no nucleus! 🏃‍♂️💨\n"
        "⚪ **White Blood Cells (WBCs)** are the body's defense system, fighting infections like tiny superheroes! 🦸‍♂️🦸‍♀️\n"
        "🟢 **Platelets** help your blood clot when you get a cut, preventing excessive bleeding! 🩹\n"
        
        "🖼️ *Send an image to analyze your blood sample now!* 📷"
    )
    await update.message.reply_text(about_text)


# Handle Photo Messages
async def handle_photo(update: Update, context: CallbackContext):
    for photo in update.message.photo[-1:]:  # Process the highest resolution photo
        file = await photo.get_file()
        image_bytes = BytesIO(await file.download_as_bytearray())
        image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Convert NumPy array to PIL image
        image = Image.fromarray(image)
        
        image_with_boxes, counter = object_detection_image(image)
        platelets_count = counter[1]
        rbc_count = counter[2]
        wbc_count = counter[3]

        # Send Detected image with boxes result
        image_with_boxes.seek(0)  # Reset file pointer before sending    
        await update.message.reply_photo(photo=image_with_boxes, caption=f"Object Detection Count:\nPlatelets ==> {platelets_count}\nRBC ==> {rbc_count}\nWBC ==> {wbc_count}")


# Main Function
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about_command))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
