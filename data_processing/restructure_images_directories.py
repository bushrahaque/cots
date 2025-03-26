import os
import shutil

def rename_and_merge(video_dirs, output_image_dir):
    os.makedirs(output_image_dir, exist_ok=True)
    
    for video_id, video_dir in enumerate(video_dirs):
        
        for filename in os.listdir(video_dir):
            if filename.endswith(('.jpg')):
                # Rename and copy image
                new_filename = f"{video_id}-{filename}"
                shutil.copy(os.path.join(video_dir, filename), os.path.join(output_image_dir, new_filename))

    print("Renaming and merging completed!")

# Example usage
video_dirs = [
    '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_images/video_0',
    '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_images/video_1',
    '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/train_images/video_2'
]
out_image_dir = '/Users/bushra/Documents/STA2453/tensorflow-great-barrier-reef/images'

rename_and_merge(video_dirs, out_image_dir)