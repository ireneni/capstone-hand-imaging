import firebase_admin
from firebase_admin import credentials, storage
import os
from datetime import timedelta

# Initialize Firebase Admin SDK (Ensure you replace with your Firebase credentials JSON)
cred = credentials.Certificate("smartsplint-6b8ec-firebase-adminsdk-fbsvc-2e1819cf47.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'smartsplint-6b8ec.firebasestorage.app'
})

bucket = storage.bucket()

# Step 1: Find the Most Recent Session for 'guest'
def get_latest_guest_session():
    print("🔍 Fetching latest guest session from Firebase Storage...")

    # List all session folders under "scans/guest/"
    blobs = list(bucket.list_blobs(prefix="scans/guest/"))

    session_timestamps = {}

    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) > 2:  # Ensure it's a valid session folder
            session_id = parts[2]  # Extract session ID
            last_updated = blob.updated.timestamp()  # Get last modified time
            
            # Track the most recent modification time for each session
            if session_id not in session_timestamps or last_updated > session_timestamps[session_id]:
                session_timestamps[session_id] = last_updated

    if not session_timestamps:
        print("❌ No sessions found for guest.")
        return None

    # Get session with the most recent timestamp
    latest_session = max(session_timestamps, key=session_timestamps.get)
    print(f"✅ Most recent session: {latest_session}")
    return latest_session

# Step 2: Download Images to `hand_pics/`
def download_session_images(session_id):
    if not session_id:
        print("❌ No recent session found.")
        return None, None

    print(f"📂 Downloading images for session: {session_id}")

    blobs = list(bucket.list_blobs(prefix=f"scans/guest/{session_id}/"))

    local_front_path = None
    local_side_path = None

    os.makedirs("hand_pics", exist_ok=True)  # Ensure the local directory exists

    for blob in blobs:
        filename = os.path.basename(blob.name).replace(" ", "_")  # Remove spaces

        if "front" in filename.lower():
            local_front_path = f"hand_pics/{filename}"
            blob.download_to_filename(local_front_path)
            print(f"✅ Downloaded front scan: {local_front_path}")

        elif "side" in filename.lower():
            local_side_path = f"hand_pics/{filename}"
            blob.download_to_filename(local_side_path)
            print(f"✅ Downloaded side scan: {local_side_path}")

    return local_front_path, local_side_path

# Step 3: Run the Measurement Extraction Script with Local Paths
def run_measurement_extraction(front_path, side_path):
    if not front_path or not side_path:
        print("❌ Missing images. Skipping extraction.")
        return

    print("⏳ Measurement extraction in progress...")
    os.system(f"python3 MeasureLength.py \"{front_path}\" \"{side_path}\"")
    print("✅ Measurement extraction completed.")

# 🔹 Main Function: Trigger Processing
if __name__ == "__main__":
    latest_session = get_latest_guest_session()
    front_path, side_path = download_session_images(latest_session)

    run_measurement_extraction(front_path, side_path)
