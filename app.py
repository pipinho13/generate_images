import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import requests
import random
import base64
import glob
from openai import OpenAI

# ========== SETUP ==========

st.set_page_config(page_title="AI Album Cover Generator", layout="wide")
st.title("🎨 AI Album Cover Generator")

# Load OpenAI API Key from secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Create images folder
os.makedirs("images", exist_ok=True)

# Variations
mood_variations = ["moody", "uplifting", "ethereal", "melancholic", "dramatic"]
composition_instructions = ["central focus", "rule of thirds", "symmetrical balance", "dynamic movement"]

# Clean file/folder names
def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name))


# ========== EXCEL UPLOAD & IMAGE GENERATION ==========

st.header("📄 Upload Excel File to Generate Album Covers")

uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        all_sheets = pd.read_excel(uploaded_file, sheet_name=["Album Names", "Genres & Keywords"])
        excel_album = all_sheets["Album Names"]
        excel_genres = all_sheets["Genres & Keywords"].dropna(axis=1, how='all')

        for i in range(excel_album.shape[0]):
            album_name = excel_album.loc[i]['Album Names']
            genre = excel_album.loc[i]['Genre']

            try:
                genre_row = excel_genres[excel_genres['Genre'] == genre].iloc[0]
            except IndexError:
                st.warning(f"⚠️ Genre not found in second sheet: {genre}")
                continue

            keywords = genre_row['Keywords']
            import_keywords = genre_row.get('Important Keywords', "")
            style = genre_row['Style']

            mood = random.choice(mood_variations)
            composition = random.choice(composition_instructions)

            # Prompt for image generation
            prompt = f"""
            Create a highly detailed album cover artwork.

            Visual Theme:
            - Use the album title: {album_name}
            - Keywords: {keywords}
            - Extra focus: {import_keywords if pd.notna(import_keywords) else "atmospheric elements"}
            - Genre: {genre}

            Artistic Style: {style}
            Composition: {composition}
            Mood: {mood}

            Format:
            - Square layout (1:1 aspect ratio)
            - High detail, suitable for an album cover
            - Include the album name visually
            """

            # Generate image
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="hd",
                response_format="url"
            )

            image_url = response.data[0].url

            # Save image to disk
            genre_folder = os.path.join("images", sanitize_filename(genre))
            os.makedirs(genre_folder, exist_ok=True)

            image_filename = f"{sanitize_filename(album_name)}.jpg"
            image_path = os.path.join(genre_folder, image_filename)

            image_data = requests.get(image_url).content
            with open(image_path, 'wb') as f:
                f.write(image_data)

            #st.image(image_url, caption=f"Generated: {album_name}", width=500)

            # Add album title to image
            title_prompt = f"""
            The image is a cover of a music album that is missing its title.
            Add the exact title: "{album_name}" once on the image.
            Place it in a visually appealing, appropriate location and style.
            **IMPORTANT: Make sure that the title has been added to the image!**
            """

            with open(image_path, "rb") as img_file:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=[img_file],
                    size="1024x1024",
                    prompt=title_prompt
                )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            st.success(f"✅ Saved: {image_path}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")



# ========== IMAGE BROWSER & BULK REGENERATOR ==========

st.header("🖼️ Browse & Bulk Re-Generate Album Covers")

image_paths = glob.glob("images/**/*.jpg", recursive=True)

if image_paths:
    selected_images = st.multiselect("Select images to re-generate:", image_paths)

    if selected_images:
        cols = st.columns(min(3, len(selected_images)))  # Display up to 3 images per row
        for i, img_path in enumerate(selected_images):
            with cols[i % len(cols)]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

        if st.button("🔄 Re-generate Selected Images"):
            for img_path in selected_images:
                try:
                    with open(img_path, "rb") as img_file:
                        result = client.images.edit(
                            model="gpt-image-1",
                            image=[img_file],
                            size="1024x1024",
                            prompt="""
                                - Generate a different image using the detected keywords of the existing image.
                                - Do not change the title
                            """
                        )

                    image_base64 = result.data[0].b64_json
                    image_bytes = base64.b64decode(image_base64)

                    # Save new image with a suffix to avoid overwrite
                    new_path = img_path.replace(".jpg", "_edited.jpg")
                    with open(new_path, "wb") as f:
                        f.write(image_bytes)

                    st.image(new_path, caption=f"🔁 Re-generated: {os.path.basename(new_path)}", width=500)
                    st.success(f"Re-generated and saved: {new_path}")

                except Exception as e:
                    st.error(f"Error during regeneration of {os.path.basename(img_path)}: {str(e)}")
    else:
        st.info("Select one or more images to regenerate.")
else:
    st.info("No images found. Upload an Excel file to generate album covers first.")
