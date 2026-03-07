import reapy
import os
import time

def populate_test_project():
    project = reapy.Project()
    sample_dir = os.path.normpath(os.path.join(os.getcwd(), "samples"))
    
    print(f"Checking in: {sample_dir}")

    samples = [f for f in os.listdir(sample_dir) if f.lower().endswith((".wav", ".mp3"))]

    for i, sample_name in enumerate(samples):
        file_path = os.path.join(sample_dir, sample_name)
        
        # 1. Create a brand new track
        new_track = project.add_track()
        new_track.name = f"Sample_{i+1}_{sample_name[:10]}"
        
        # 2. Deselect all other tracks, then select ONLY our new track
        for t in project.tracks:
            t.unselect()
        new_track.select()
        
        # 3. Move the playhead (cursor) to the very beginning
        project.cursor_position = 0.0
        
        # 4. Insert the media! It will now land exactly on the selected track
        reapy.reascript_api.InsertMedia(file_path, 0)
        
        time.sleep(0.1) # Quick breath for REAPER
        
    print(f"✅ Successfully added {len(samples)} SEPARATE tracks!")

if __name__ == "__main__":
    populate_test_project()