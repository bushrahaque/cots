import os
import shutil


# RELEVANT PATHS
VIDEO_DIR = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/all_images'
IMAGES_DIR = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/images'


def rename_and_merge(video_id, video_dir, output_image_dir):
    """
    Renames image files from a given video directory by prefixing them with the video ID,
    then copies them into a single merged output directory.

    Inputs:
    - video_id (int): Identifier for the video, used to prefix image filenames.
    - video_dir (str): Path to the directory containing images for one video.
    - output_image_dir (str): Path to the directory where renamed images will be copied.

    Outputs:
    - None
    """
    for filename in os.listdir(video_dir):
        if filename.endswith('.jpg'):
            # Construct new filename with video ID prefix (e.g., "0-10.jpg")
            new_filename = f"{video_id}-{filename}"

            # Copy the image to the output directory with the new name
            shutil.copy(
                os.path.join(video_dir, filename),
                os.path.join(output_image_dir, new_filename)
            )

    print("Renaming and merging completed!")


if __name__ == "__main__":

    # Verify the output directory exists
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Loop over videos (0, 1, 2), merging each folder's images into one directory
    for v in range(3):
        CURR_VIDEO_DIR = os.path.join(VIDEO_DIR, f"video_{v}")
        rename_and_merge(v, CURR_VIDEO_DIR, IMAGES_DIR)

