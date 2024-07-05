import qrcode
import streamlit as st
from PIL import Image
import io

# Function to generate QR code
def generate_qr_code(data):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="grey", back_color="white")
    return img

# Streamlit app
st.title("QR Code Generator")

# Input field for data
data = st.text_input("Enter the data/URL for the QR Code:")

if data:
    # Generate QR Code
    img = generate_qr_code(data)
    
    # Convert PIL image to a format that Streamlit can display
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    # Display QR Code
    st.image(img_bytes, caption="Generated QR Code", use_column_width=True)

    # Provide download link
    st.download_button(
        label="Download QR Code",
        data=img_bytes,
        file_name="qr_code.png",
        mime="image/png"
    )

