import reapy

try:
    # Connect to the active project
    project = reapy.Project()
    
    # Add a track
    new_track = project.add_track()
    new_track.name = "AI_STATIONS_READY"
    
    print("✅ SUCCESS: REAPER and Python are talking!")
except Exception as e:
    print(f"❌ FAILED: Could not connect to REAPER. Error: {e}")