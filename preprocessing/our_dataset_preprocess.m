% Define the folder containing the .m4a files
inputFolder = 'our_chords_m4a';
outputFolder = 'our_chords_wav';

% Create the output folder if it doesn't exist
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get a list of all .m4a files in the input folder
m4aFiles = dir(fullfile(inputFolder, '*.m4a'));

% Process each .m4a file in the input folder
for i = 1:length(m4aFiles)
    % Get the full file path
    inputFilePath = fullfile(inputFolder, m4aFiles(i).name);
    
    % Read the audio file
    [audioData, originalSampleRate] = audioread(inputFilePath);
    
    % Extract the first 2 seconds
    durationInSeconds = 2;
    numSamples = min(durationInSeconds * originalSampleRate, size(audioData, 1));
    audioData_2s = audioData(1:numSamples, :);
    
    % Resample to 44.1 kHz if necessary
    targetSampleRate = 44100;
    if originalSampleRate ~= targetSampleRate
        audioData_2s = resample(audioData_2s, targetSampleRate, originalSampleRate);
    end
    
    % Construct the output file path
    [~, baseFileName, ~] = fileparts(m4aFiles(i).name);
    outputFileName = [baseFileName, '.wav'];
    outputFilePath = fullfile(outputFolder, outputFileName);
    
    % Save the audio as a 24-bit PCM .wav file
    audiowrite(outputFilePath, audioData_2s, targetSampleRate, 'BitsPerSample', 24);
end

disp('Conversion complete!');