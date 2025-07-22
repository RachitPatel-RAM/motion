import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import os
from datetime import datetime
import time
import mimetypes
import traceback

def send_email_alert(sender_email, app_password, receiver_email, subject, body, attachment_paths, video_path=None):
    try:
        msg = MIMEMultipart('mixed')
        msg["From"] = sender_email
        msg["To"] = receiver_email

        current_time = datetime.now()
        day_name = current_time.strftime("%A")
        date_str = current_time.strftime("%B %d, %Y")
        time_str = current_time.strftime("%I:%M:%S %p")
        formatted_date = f"{day_name}, {date_str} at {time_str}"
        msg["Subject"] = f"🚨 Motion Alert: {formatted_date}"

        msg_alternative = MIMEMultipart('alternative')
        msg.attach(msg_alternative)

        text_body = f"""Motion detected at {formatted_date}.

The security system detected significant movement.

Please check the attached images and video.
Note: If you don't receive this email, verify that the sender email ({sender_email}) has a valid app password and Gmail's 2-Step Verification is enabled. Check https://myaccount.google.com/security for details."""
        msg_alternative.attach(MIMEText(text_body, 'plain'))

        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .header {{ background-color: #f8f8f8; padding: 15px; border-bottom: 2px solid #e74c3c; }}
                h2 {{ color: #e74c3c; margin-top: 0; }}
                .content {{ padding: 20px 0; }}
                .timestamp {{ background-color: #f1f1f1; padding: 10px; border-radius: 4px; margin-bottom: 20px; }}
                .timestamp span {{ font-weight: bold; }}
                .image-container {{ margin: 20px 0; }}
                .image-title {{ font-weight: bold; margin-bottom: 10px; color: #3498db; }}
                .video-container {{ margin: 20px 0; background-color: #f9f9f9; padding: 15px; border-left: 4px solid #3498db; }}
                .footer {{ margin-top: 30px; font-size: 12px; color: #777; border-top: 1px solid #eee; padding-top: 15px; }}
                .code-block {{ background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 3px; padding: 10px; font-family: monospace; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>🚨 Motion Alert</h2>
                </div>
                <div class="content">
                    <div class="timestamp">
                        <p><span>Day:</span> {day_name}</p>
                        <p><span>Date:</span> {date_str}</p>
                        <p><span>Time:</span> {time_str}</p>
                    </div>
                    <p>The security system detected significant movement that appears to be unnatural (person, animal, or object).</p>
                    <p><strong>Note:</strong> If you don't receive this email, verify that the sender email ({sender_email}) has a valid app password and Gmail's 2-Step Verification is enabled. Check <a href="https://myaccount.google.com/security">Google Account Security</a> for details.</p>
                    <div class="code-block">
                    <pre># Upload snapshot\nresponse = cloudinary.uploader.upload("images/motion_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg", format="jpg", upload_preset="motion")\n\n# Upload video\nresponse = cloudinary.uploader.upload("videos/motion_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4", resource_type="video", format="mp4", upload_preset="motion")</pre>
                    </div>
                    <h3 class="image-title">📸 Captured Images:</h3>
                    <div class="image-container">
        """

        if isinstance(attachment_paths, list) and attachment_paths:
            for i, attachment_path in enumerate(attachment_paths):
                try:
                    if os.path.exists(attachment_path) and os.path.getsize(attachment_path) > 0:
                        img_id = f'image{i}'
                        with open(attachment_path, "rb") as attachment:
                            img_data = attachment.read()
                        img = MIMEImage(img_data)
                        img.add_header('Content-ID', f'<{img_id}>')
                        img.add_header('Content-Disposition', 'inline', filename=os.path.basename(attachment_path))
                        msg.attach(img)
                        capture_time = os.path.basename(attachment_path).replace('motion_', '').replace('.jpg', '')
                        html_body += f'<p><strong>Capture {i+1}:</strong> {capture_time}</p>\n'
                        html_body += f'<p><img src="cid:{img_id}" style="max-width:100%; border:1px solid #ddd; border-radius:4px; padding:5px;" /></p>\n'
                    else:
                        print(f"Warning: Image file not found or empty: {attachment_path}")
                except Exception as e:
                    print(f"Error attaching image {attachment_path}: {str(e)}")
        elif isinstance(attachment_paths, str) and os.path.exists(attachment_paths) and os.path.getsize(attachment_paths) > 0:
            try:
                img_id = 'image0'
                with open(attachment_paths, "rb") as attachment:
                    img_data = attachment.read()
                img = MIMEImage(img_data)
                img.add_header('Content-ID', f'<{img_id}>')
                img.add_header('Content-Disposition', 'inline', filename=os.path.basename(attachment_paths))
                msg.attach(img)
                capture_time = os.path.basename(attachment_paths).replace('motion_', '').replace('.jpg', '')
                html_body += f'<p><strong>Capture:</strong> {capture_time}</p>\n'
                html_body += f'<p><img src="cid:{img_id}" style="max-width:100%; border:1px solid #ddd; border-radius:4px; padding:5px;" /></p>\n'
            except Exception as e:
                print(f"Error attaching single image: {str(e)}")

        if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            video_filename = os.path.basename(video_path)
            video_time = video_filename.replace('motion_', '').replace('.avi', '')
            html_body += f"""
                <div class="video-container">
                    <h3 class="image-title">🎥 Video Recording</h3>
                    <p><strong>Filename:</strong> {video_filename}</p>
                    <p><strong>Recorded at:</strong> {video_time}</p>
                    <p>A video recording of the motion event is attached to this email.</p>
                </div>
            """
            try:
                with open(video_path, "rb") as attachment:
                    content_type, encoding = mimetypes.guess_type(video_path)
                    if content_type is None or encoding is not None:
                        content_type = 'application/octet-stream'
                    main_type, sub_type = content_type.split('/', 1)
                    part = MIMEBase(main_type, sub_type)
                    part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', 'attachment', filename=video_filename)
                    msg.attach(part)
                    print(f"Video attached: {video_path} ({os.path.getsize(video_path)} bytes)")
            except Exception as e:
                print(f"Error attaching video: {str(e)}")
                traceback.print_exc()

        html_body += """
                </div>
                <div class="footer">
                    <p>This is an automated alert from your security system.</p>
                    <p>© 2025 Motion Detection System</p>
                </div>
            </div>
        </body>
        </html>"""
        msg_alternative.attach(MIMEText(html_body, "html"))

        max_retries = 5
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                    server.ehlo()
                    server.starttls()
                    server.ehlo()
                    server.login(sender_email, app_password)
                    server.send_message(msg)
                    print(f"Email sent successfully on attempt {attempt+1}")
                    return True
            except smtplib.SMTPAuthenticationError:
                print("Authentication failed. Please verify the sender email and app password at https://myaccount.google.com/security.")
                return False
            except smtplib.SMTPException as e:
                print(f"SMTP error on attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Failed to send email after multiple attempts. Check Gmail settings and network connection.")
                    traceback.print_exc()
                    return False
            except Exception as e:
                print(f"Unexpected error on attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print("Failed to send email after multiple attempts")
                    traceback.print_exc()
                    return False
    except Exception as e:
        print(f"Unexpected error in send_email_alert: {str(e)}")
        traceback.print_exc()
        return False