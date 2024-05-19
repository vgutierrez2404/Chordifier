import os
from pydub import AudioSegment
import soundfile as sf

# Define the folder containing the .m4a files
input_folder = r'./our_chords_m4a/'
output_folder = r'./converted_chords_wav/'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each .m4a file in the input folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.m4a'):
        # Construct the full file path
        input_path = os.path.join(input_folder, file_name)
        
        # Load the audio file
        audio = AudioSegment.from_mp3(input_path)
        
        # Extract the first 2 seconds
        audio_2s = audio[:2000]
        
        # Resample to 44.1 kHz
        audio_2s = audio_2s.set_frame_rate(44100)
        
        # Export the audio as a .wav file (24-bit PCM)
        output_file_name = os.path.splitext(file_name)[0] + '.wav'
        output_path = os.path.join(output_folder, output_file_name)
        
        # Save as 24-bit PCM .wav
        audio_2s.export(output_path, format='wav', bitrate="24")

print("Conversion complete!")