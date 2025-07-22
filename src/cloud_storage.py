import cloudinary
import cloudinary.uploader
import os

def upload_to_cloudinary(file_path, config, resource_type="auto"):
    cloudinary.config(
        cloud_name=config["cloud_name"],
        api_key=config["api_key"],
        api_secret=config["api_secret"]
    )

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return None

    file_ext = os.path.splitext(file_path)[1].lower()
    if resource_type == "auto":
        if file_ext in (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"):
            resource_type = "video"
            format_param = "mp4"
        elif file_ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"):
            resource_type = "image"
            format_param = "jpg"
        else:
            print(f"Unsupported file extension: {file_ext}")
            return None

    try:
        print(f"Uploading {file_path} as {resource_type} with preset 'motion'...")
        response = cloudinary.uploader.upload(
            file_path,
            resource_type=resource_type,
            format=format_param,
            upload_preset="motion"
        )
        secure_url = response["secure_url"]
        print(f"Upload successful: {secure_url}")
        return secure_url
    except Exception as e:
        print(f"Error uploading to Cloudinary: {str(e)}")
        if "Upload preset must be whitelisted for unsigned uploads" in str(e):
            print("Please ensure the 'motion' preset is set to 'Unsigned' in Cloudinary Settings > Upload > Upload Presets.")
        elif "Upload preset not found" in str(e):
            print("Please create an unsigned upload preset named 'motion' in Cloudinary Settings > Upload > Upload Presets.")
        return None