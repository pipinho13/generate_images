# app.py

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
from PIL import Image
import io
import shutil

# ========== SETUP ==========

st.set_page_config(page_title="AI Album Cover Generator", layout="wide")
st.title("🎨 AI Album Cover Generator")

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

os.makedirs("images", exist_ok=True)
os.makedirs("images_good", exist_ok=True)
metadata_file = "image_metadata.csv"

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "", str(name))


# ========== EXCEL UPLOAD & IMAGE GENERATION ==========

st.header("📄 Upload Excel File to Generate Album Covers")

uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Clear existing metadata to avoid duplicates
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
            st.info("🗑️ Cleared existing metadata to avoid duplicates")
        
        all_sheets = pd.read_excel(uploaded_file, sheet_name=["Album Names", "Genres & Keywords"])
        excel_album = all_sheets["Album Names"]
        excel_genres = all_sheets["Genres & Keywords"].dropna(axis=1, how='all')

        for i in range(excel_album.shape[0]):
            album_name = excel_album.loc[i]['Album Names']
            genre = excel_album.loc[i]['Genre']
            
            # Extract keywords from album title (ignore numbers)
            title_keywords = []
            if pd.notna(album_name) and album_name.strip():
                # Split title into words and filter out numbers
                title_words = [word.strip() for word in str(album_name).replace('-', ' ').replace('_', ' ').split()]
                title_keywords = [word for word in title_words if not word.isdigit() and len(word) > 1]

            try:
                genre_row = excel_genres[excel_genres['Genre'] == genre].iloc[0]
            except IndexError:
                st.warning(f"⚠️ Genre not found in second sheet: {genre}")
                continue

            keywords = genre_row['Keywords']
            import_keywords = genre_row.get('Important Keywords', "")
            style = genre_row['Style']

            # Merge and randomly select 3 keywords
            all_keywords = []
            
            # Add title keywords first (most important)
            all_keywords.extend(title_keywords)
            
            # Add genre keywords
            if pd.notna(keywords) and keywords.strip():
                all_keywords.extend([kw.strip() for kw in str(keywords).split(',')])
            if pd.notna(import_keywords) and import_keywords.strip():
                all_keywords.extend([kw.strip() for kw in str(import_keywords).split(',')])
            
            # Remove empty keywords and duplicates
            all_keywords = list(set([kw for kw in all_keywords if kw]))
            
            # Randomly select 3 keywords (or all if less than 3)
            selected_keywords = random.sample(all_keywords, min(3, len(all_keywords))) if all_keywords else []
            selected_keywords_str = ", ".join(selected_keywords)

            # Create the main generation prompt with system and user instructions
            system_prompt = "You are an expert graphic designer responsible for creating professional music album covers."
            user_prompt = f"Design a square 1024x1024 artistic music album cover for the album titled '{album_name}', featuring: {selected_keywords_str}. Style: {style}, Genre: {genre}. Strictly no text, letters, numbers, borders, frames, outlines, vignettes, mockups, CDs, books, tapes, packaging, watermarks, or signatures. Not a product photo. Only include a clean, square, full-bleed illustration suitable for production. Max 3 anatomically correct creatures (not in background). Focus on mood, atmosphere, and visual storytelling aligned with the album's title and genre."

            # Combine system and user prompts
            prompt = f"{system_prompt} {user_prompt}"

            # Use the API with the stored prompts (which are now single-line)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="hd", 
                response_format="url"
            )

            image_url = response.data[0].url
            genre_folder = os.path.join("images", sanitize_filename(genre))
            os.makedirs(genre_folder, exist_ok=True)

            image_filename = f"{sanitize_filename(album_name)}.jpg"
            image_path = os.path.join(genre_folder, image_filename)

            image_data = requests.get(image_url).content
            with open(image_path, 'wb') as f:
                f.write(image_data)

            # Create the title addition prompt
            title_prompt = f"The image is a cover of a music album that is missing its title. Add the exact title: '{album_name}' once on the image. Place it in a visually appealing, appropriate location and style. IMPORTANT: Make sure that the title has been added correctly to the image!"

            with open(image_path, "rb") as img_file:
                result = client.images.edit(
                    model="gpt-image-1",
                    image=[img_file],
                    size="1024x1024",
                    prompt=title_prompt
                )

            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)

            image_pil = Image.open(io.BytesIO(image_bytes))
            resized_image = image_pil.resize((1440, 1440), Image.Resampling.LANCZOS)
            resized_image.save(image_path, "JPEG", quality=95)

            # Fixed: Create complete image record with all fields
            image_record = {
                "title": album_name,
                "genre": genre,
                "image_path": image_path,
                "prompt": prompt,  # This was missing proper assignment
                "title_prompt": title_prompt,  # This was missing proper assignment
                "review": ""
            }

            # Fixed: Proper DataFrame handling
            if os.path.exists(metadata_file):
                df = pd.read_csv(metadata_file)
                # Create new row as DataFrame and concatenate
                new_row_df = pd.DataFrame([image_record])
                df = pd.concat([df, new_row_df], ignore_index=True)
            else:
                df = pd.DataFrame([image_record])

            df.to_csv(metadata_file, index=False)
            st.success(f"✅ Image saved and metadata recorded: {album_name}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")


# ========== IMAGE REVIEW TOOL ==========

st.header("✅ Image Review")

if os.path.exists("images"):
    folders = next(os.walk("images"))[1]
    
    if folders:  # Check if there are any folders
        selected_folder = st.selectbox("Select a genre folder to review:", folders)

        if selected_folder:
            folder_path = os.path.join("images", selected_folder)
            image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))

            if image_files:
                # Create a selectbox to choose specific image
                image_names = [os.path.basename(img) for img in image_files]
                selected_image_name = st.selectbox("Select an image to review:", image_names)
                
                # Find the index of the selected image
                selected_index = image_names.index(selected_image_name) if selected_image_name in image_names else 0
                current_image = image_files[selected_index]
                
                # Display current image info
                st.write(f"Image {selected_index + 1} of {len(image_files)}")
                st.image(current_image, caption=os.path.basename(current_image), width=500)

                # Fixed: Better button layout and functionality with bigger buttons
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        height: 3em;
                        width: 100%;
                        font-size: 18px;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    if st.button("👍 Good", key=f"good_{selected_index}_{selected_folder}"):
                        # Create good images folder structure
                        new_path = current_image.replace("images", "images_good")
                        os.makedirs(os.path.dirname(new_path), exist_ok=True)
                        
                        # Move the image
                        shutil.move(current_image, new_path)

                        # Update metadata - more robust approach
                        if os.path.exists(metadata_file):
                            df = pd.read_csv(metadata_file)
                            # Find the row by matching the image path OR by title
                            mask = (df['image_path'] == current_image) | (df['title'] == os.path.splitext(selected_image_name)[0])
                            if mask.any():
                                df.loc[mask, 'review'] = 'Good'
                                df.loc[mask, 'image_path'] = new_path
                                df.to_csv(metadata_file, index=False)
                                st.success("✅ Marked as Good and moved!")
                            else:
                                st.warning("⚠️ Could not find image in metadata to update")
                        else:
                            st.warning("⚠️ No metadata file found")
                        
                        st.rerun()

                with col2:
                    if st.button("👎 Bad", key=f"bad_{selected_index}_{selected_folder}"):
                        # Update metadata - more robust approach
                        if os.path.exists(metadata_file):
                            df = pd.read_csv(metadata_file)
                            # Find the row by matching the image path OR by title
                            mask = (df['image_path'] == current_image) | (df['title'] == os.path.splitext(selected_image_name)[0])
                            if mask.any():
                                df.loc[mask, 'review'] = 'Bad'
                                df.to_csv(metadata_file, index=False)
                                st.warning("❌ Marked as Bad!")
                                
                                # Delete image file after updating metadata
                                if os.path.exists(current_image):
                                    os.remove(current_image)
                                    st.warning("🗑️ Image deleted!")
                            else:
                                st.warning("⚠️ Could not find image in metadata to update")
                        else:
                            st.warning("⚠️ No metadata file found")
                        
                        st.rerun()
                
                with col3:
                    # Show current review status if available
                    if os.path.exists(metadata_file):
                        df = pd.read_csv(metadata_file)
                        mask = (df['image_path'] == current_image) | (df['title'] == os.path.splitext(selected_image_name)[0])
                        if mask.any():
                            current_review = df.loc[mask, 'review'].iloc[0]
                            if current_review:
                                st.write(f"**Status:** {current_review}")
                            else:
                                st.write("**Status:** Not reviewed")
                        else:
                            st.write("**Status:** Not found in metadata")
                        
                # Navigation info
                st.write(f"Current image: **{selected_image_name}**")
                
                # Debug info (can be removed later)
                with st.expander("Debug Info"):
                    st.write(f"Image path: {current_image}")
                    st.write(f"Genre folder: {selected_folder}")
                    if os.path.exists(metadata_file):
                        df = pd.read_csv(metadata_file)
                        matching_rows = df[(df['image_path'] == current_image) | (df['title'] == os.path.splitext(selected_image_name)[0])]
                        st.write("Matching metadata rows:")
                        st.dataframe(matching_rows)
                
            else:
                st.info("No images found in the selected folder.")
    else:
        st.info("No genre folders found. Generate some images first!")


# ========== RE-GENERATE BAD IMAGES ==========

st.header("♻️ Re-Generate Bad Images")

if os.path.exists(metadata_file):
    df = pd.read_csv(metadata_file)
    bad_images = df[df['review'] == "Bad"]
    
    if not bad_images.empty:
        st.write(f"Found {len(bad_images)} bad images to regenerate:")
        st.dataframe(bad_images[['title', 'genre', 'review']])
        
        if st.button("Regenerate all bad images"):
            progress_bar = st.progress(0)
            
            for idx, (_, row) in enumerate(bad_images.iterrows()):
                album_name = row['title']
                genre = row['genre']
                prompt = row['prompt']
                title_prompt = row['title_prompt']
                
                st.write(f"Regenerating: {album_name}")

                try:
                    response = client.images.generate(
                        model="dall-e-3",
                        prompt=prompt,
                        n=1,
                        size="1024x1024",
                        quality="hd",
                        response_format="url"
                    )

                    image_url = response.data[0].url
                    genre_folder = os.path.join("images", sanitize_filename(genre))
                    os.makedirs(genre_folder, exist_ok=True)

                    image_filename = f"{sanitize_filename(album_name)}.jpg"
                    image_path = os.path.join(genre_folder, image_filename)

                    image_data = requests.get(image_url).content
                    with open(image_path, 'wb') as f:
                        f.write(image_data)

                    with open(image_path, "rb") as img_file:
                        result = client.images.edit(
                            model="gpt-image-1",
                            image=[img_file],
                            size="1024x1024",
                            prompt=title_prompt
                        )

                    image_base64 = result.data[0].b64_json
                    image_bytes = base64.b64decode(image_base64)

                    image_pil = Image.open(io.BytesIO(image_bytes))
                    resized_image = image_pil.resize((1440, 1440), Image.Resampling.LANCZOS)
                    resized_image.save(image_path, "JPEG", quality=95)

                    # Update metadata - reset review status
                    df.loc[df['title'] == album_name, 'review'] = ""
                    df.loc[df['title'] == album_name, 'image_path'] = image_path
                    
                    st.success(f"♻️ Regenerated: {album_name}")
                    
                except Exception as e:
                    st.error(f"Failed to regenerate {album_name}: {e}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(bad_images))

            # Save updated metadata
            df.to_csv(metadata_file, index=False)
            st.success("✅ All bad images have been regenerated!")
    else:
        st.info("No bad images found to regenerate.")
else:
    st.warning("No metadata found to regenerate from.")